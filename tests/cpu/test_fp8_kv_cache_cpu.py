# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU tests for the FP8 KV cache math (hf_adapters.fp8_kv) — no Spyre needed.

These cover what is ours to get right: the GQA fold, the per-tensor K/V scales
and their freezing, the clamp, and the scale bookkeeping around the FP8 Q·Kᵀ.
Device-side questions — scaled_bmm (torch-spyre#5158), the cache scatter onto a
QFP8WT/QFP8CH-tagged tensor — cannot be tested here.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from hf_adapters.fp8_kv import fp8_attention, store_fp8_kv

B, N_Q, N_KV, DH = 1, 8, 2, 128
SCALING = DH**-0.5


def _attn():
    """The attributes fp8_kv reads from an attention module."""
    return SimpleNamespace(
        scaling=SCALING,
        _k_scale=torch.ones(B, 1, 1, 1, dtype=torch.float16),
        _v_scale=torch.ones(B, 1, 1, 1, dtype=torch.float16),
        _kv_scales_set=False,
    )


def _cache(m):
    # Byte 0x00 is +0.0 in E4M3.
    return torch.zeros(B, N_KV, m, DH, dtype=torch.uint8).view(torch.float8_e4m3fn)


def _qkv(length, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, N_Q, length, DH, generator=g) * scale
    k = torch.randn(B, N_KV, length, DH, generator=g) * scale
    v = torch.randn(B, N_KV, length, DH, generator=g) * scale
    return q.half(), k.half(), v.half()


def _causal(length, m):
    mask = torch.full((length, m), float("-inf")).triu(1)
    return mask.half().view(1, 1, length, m)


def _sdpa(q, k, v, mask):
    return F.scaled_dot_product_attention(
        q.float(),
        k.float(),
        v.float(),
        attn_mask=None if mask is None else mask.float(),
        scale=SCALING,
        enable_gqa=True,
    )


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def test_gqa_fold_is_exact():
    """[B, n_q, L, Dh] -> [B, n_kv, g*L, Dh] reproduces SDPA(enable_gqa) exactly."""
    q, k, v = (t.float() for t in _qkv(16, seed=0))
    mask = _causal(16, 16).float()

    q_f = q.reshape(B, N_KV, -1, DH)
    scores = (q_f @ k.transpose(-2, -1) * SCALING).reshape(B, N_Q, 16, 16) + mask
    probs = torch.softmax(scores, dim=-1)
    out = (probs.reshape(B, N_KV, -1, 16) @ v).reshape(B, N_Q, 16, DH)

    torch.testing.assert_close(out, _sdpa(q, k, v, mask), rtol=1e-5, atol=1e-5)


def test_fp8_attention_tracks_fp16_reference():
    """Prefill: FP8 Q·Kᵀ with per-token Q and per-tensor K/V stays close to fp16."""
    q, k, v = _qkv(16, seed=1)
    attn, kc, vc = _attn(), _cache(16), _cache(16)
    mask = _causal(16, 16)

    store_fp8_kv(attn, k, v, kc, vc, torch.arange(16))
    got = fp8_attention(attn, q, kc, vc, mask)

    assert torch.isfinite(got).all()
    # E4M3 keeps 3 mantissa bits (~3.6% RMS per value); a wrong scale costs far
    # more than this threshold allows.
    assert _cos(got, _sdpa(q, k, v, mask)) > 0.995


def test_first_store_sets_scales_then_they_freeze():
    """Scales come from the first store; later stores reuse them unchanged."""
    q, k, v = _qkv(8, seed=2)
    attn, kc, vc = _attn(), _cache(16), _cache(16)

    store_fp8_kv(attn, k, v, kc, vc, torch.arange(8))
    expect_k = k.float().abs().amax() / 200.0
    expect_v = v.float().abs().amax() / 100.0
    got_k, got_v = attn._k_scale.float().flatten(), attn._v_scale.float().flatten()
    torch.testing.assert_close(got_k, expect_k.view(1), rtol=1e-3, atol=0)
    torch.testing.assert_close(got_v, expect_v.view(1), rtol=1e-3, atol=0)

    attn._kv_scales_set = True  # StandardGQABlock flips this after the first store
    frozen_k, frozen_v = attn._k_scale.clone(), attn._v_scale.clone()
    _, k2, v2 = _qkv(1, seed=3, scale=5.0)  # a much larger decode token
    store_fp8_kv(attn, k2, v2, kc, vc, torch.tensor([8]))
    assert torch.equal(attn._k_scale, frozen_k)
    assert torch.equal(attn._v_scale, frozen_v)


def test_outlier_token_saturates_instead_of_nan():
    """A token beyond the headroom is clamped to +-448, never stored as NaN."""
    _, k, v = _qkv(8, seed=4)
    attn, kc, vc = _attn(), _cache(16), _cache(16)
    store_fp8_kv(attn, k, v, kc, vc, torch.arange(8))
    attn._kv_scales_set = True

    _, k_big, v_big = _qkv(1, seed=5, scale=50.0)  # far past 2.24x headroom
    store_fp8_kv(attn, k_big, v_big, kc, vc, torch.tensor([8]))

    stored = kc[:, :, 8].float()
    assert torch.isfinite(stored).all()
    assert stored.abs().max().item() == 448.0


def test_unwritten_slots_are_masked_out():
    """A long cache with only a prefix written matches the prefix-only result."""
    q, k, v = _qkv(8, seed=6)
    full, short = _attn(), _attn()
    kc, vc = _cache(32), _cache(32)
    kc8, vc8 = _cache(8), _cache(8)

    store_fp8_kv(full, k, v, kc, vc, torch.arange(8))
    store_fp8_kv(short, k, v, kc8, vc8, torch.arange(8))

    mask = torch.full((8, 32), float("-inf")).triu(1)
    mask[:, 8:] = float("-inf")
    got = fp8_attention(full, q, kc, vc, mask.half().view(1, 1, 8, 32))
    ref = fp8_attention(short, q, kc8, vc8, _causal(8, 8))

    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)
