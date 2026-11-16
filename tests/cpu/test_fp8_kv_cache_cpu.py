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

These cover what is ours to get right: the GQA fold, the K block / V row cache
layouts and their indices, the decode staging block, the per-tensor K/V scales
and their freezing, the clamp, and the scale bookkeeping around the FP8 Q·Kᵀ.
Device-side questions — scaled_bmm (torch-spyre#5158), whole-block QFP8WT
scatter, cross-graph layouts — cannot be tested here.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from hf_adapters.fp8_kv import (
    KV_BLOCK,
    fp8_attention,
    fp8_cache_index,
    store_fp8_kv,
)

N_Q, N_KV, DH = 8, 2, 128
SCALING = DH**-0.5


def _state(b, m):
    """An attention module's FP8 attributes, plus empty K and V caches."""
    attn = SimpleNamespace(
        scaling=SCALING,
        _k_scale=torch.ones(1, 1, 1, 1, dtype=torch.float16),
        _v_scale=torch.ones(1, 1, 1, dtype=torch.float16),
        _kv_scales_set=False,
        _k_staging=torch.zeros(KV_BLOCK, b, N_KV, DH, dtype=torch.float16),
    )
    # Byte 0x00 is +0.0 in E4M3.
    kc = torch.zeros(b * m // KV_BLOCK, N_KV, DH, KV_BLOCK, dtype=torch.uint8)
    vc = torch.zeros(b * m, N_KV, DH, dtype=torch.uint8)
    return attn, kc.view(torch.float8_e4m3fn), vc.view(torch.float8_e4m3fn)


def _qkv(b, length, seed, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, N_Q, length, DH, generator=g) * scale
    k = torch.randn(b, N_KV, length, DH, generator=g) * scale
    v = torch.randn(b, N_KV, length, DH, generator=g) * scale
    return q.half(), k.half(), v.half()


def _mask(pos0, length, m):
    """Causal additive mask [1, 1, length, m]: row i sees columns <= pos0 + i."""
    i = torch.arange(length).view(length, 1) + pos0
    j = torch.arange(m).view(1, m)
    return torch.where(j <= i, 0.0, float("-inf")).half().view(1, 1, length, m)


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


def _ref(q, k, v, pos0, m):
    """fp32 attention of q (at pos0..) over the first pos0 + L positions of k/v."""
    n = pos0 + q.shape[2]
    pad = (0, 0, 0, m - n)
    kp, vp = F.pad(k[:, :, :n].float(), pad), F.pad(v[:, :, :n].float(), pad)
    return _sdpa(q, kp, vp, _mask(pos0, q.shape[2], m))


def test_gqa_fold_is_exact():
    """[B, n_q, L, Dh] -> [B, n_kv, g*L, Dh] reproduces SDPA(enable_gqa) exactly."""
    q, k, v = (t.float() for t in _qkv(1, 16, seed=0))
    mask = _mask(0, 16, 16).float()

    q_f = q.reshape(1, N_KV, -1, DH)
    scores = (q_f @ k.transpose(-2, -1) * SCALING).reshape(1, N_Q, 16, 16) + mask
    probs = torch.softmax(scores, dim=-1)
    out = (probs.reshape(1, N_KV, -1, 16) @ v).reshape(1, N_Q, 16, DH)

    torch.testing.assert_close(out, _sdpa(q, k, v, mask), rtol=1e-5, atol=1e-5)


def test_cache_index_prefill_and_decode():
    """Prefill names whole K blocks; decode names the block and staging slot."""
    m = 3 * KV_BLOCK
    k_blocks, _, v_rows = fp8_cache_index(2, m, 0, 2 * KV_BLOCK)
    assert k_blocks.tolist() == [0, 1, 3, 4]  # sequence b owns blocks 3b .. 3b+2
    assert v_rows.tolist() == list(range(128)) + list(range(m, m + 128))

    k_blocks, k_slot, v_rows = fp8_cache_index(2, m, 130, 1)
    assert k_blocks.tolist() == [2, 5]
    assert k_slot.tolist() == [130 - 2 * KV_BLOCK]
    assert v_rows.tolist() == [130, m + 130]


def test_prefill_tracks_fp16_reference():
    """Prefill: FP8 Q·Kᵀ with per-token Q and per-tensor K/V stays close to fp32."""
    b, length, m = 2, 128, 256
    q, k, v = _qkv(b, length, seed=1)
    attn, kc, vc = _state(b, m)

    store_fp8_kv(attn, k, v, kc, vc, fp8_cache_index(b, m, 0, length))
    got = fp8_attention(attn, q, kc, vc, _mask(0, length, m))

    assert torch.isfinite(got).all()
    # E4M3 keeps 3 mantissa bits (~3.6% RMS per value); a wrong scale, block or
    # row costs far more than this threshold allows.
    assert _cos(got, _ref(q, k, v, 0, m)) > 0.995


@pytest.mark.parametrize("b", [1, 2])  # B == 1 decodes in the block domain
@pytest.mark.parametrize("mask_heads", [1, N_Q])  # hf_common may give one per head
def test_decode_across_a_block_boundary(b, mask_heads):
    """Every decode step, including the first of a new block, tracks fp32."""
    pre, steps, m = 64, 66, 192  # decode writes 64..129: blocks 1 and 2
    q, k, v = _qkv(b, pre + steps, seed=2)
    attn, kc, vc = _state(b, m)

    store_fp8_kv(
        attn, k[:, :, :pre], v[:, :, :pre], kc, vc, fp8_cache_index(b, m, 0, pre)
    )
    attn._kv_scales_set = True  # freeze_kv_scales does this after prefill
    for pos in range(pre, pre + steps):
        sl = slice(pos, pos + 1)
        idx = fp8_cache_index(b, m, pos, 1)
        store_fp8_kv(attn, k[:, :, sl], v[:, :, sl], kc, vc, idx)
        mask = _mask(pos, 1, m).expand(b, mask_heads, 1, m).contiguous()
        got = fp8_attention(attn, q[:, :, sl], kc, vc, mask)
        assert torch.isfinite(got).all(), pos
        assert _cos(got, _ref(q[:, :, sl], k, v, pos, m)) > 0.995, pos


def test_first_store_sets_scales_then_they_freeze():
    """Scales come from the first store; later stores reuse them unchanged."""
    _, k, v = _qkv(1, 64, seed=3)
    attn, kc, vc = _state(1, 128)

    store_fp8_kv(attn, k, v, kc, vc, fp8_cache_index(1, 128, 0, 64))
    expect_k = k.float().abs().amax() / 200.0
    expect_v = v.float().abs().amax() / 100.0
    got_k, got_v = attn._k_scale.float().flatten(), attn._v_scale.float().flatten()
    torch.testing.assert_close(got_k, expect_k.view(1), rtol=1e-3, atol=0)
    torch.testing.assert_close(got_v, expect_v.view(1), rtol=1e-3, atol=0)

    attn._kv_scales_set = True
    frozen_k, frozen_v = attn._k_scale.clone(), attn._v_scale.clone()
    _, k2, v2 = _qkv(1, 1, seed=4, scale=5.0)  # a much larger decode token
    store_fp8_kv(attn, k2, v2, kc, vc, fp8_cache_index(1, 128, 64, 1))
    assert torch.equal(attn._k_scale, frozen_k)
    assert torch.equal(attn._v_scale, frozen_v)


def test_decode_rewrite_keeps_earlier_tokens():
    """Rewriting a block from staging leaves its earlier positions' bytes as is."""
    _, k, v = _qkv(1, 66, seed=5)
    attn, kc, vc = _state(1, 128)
    store_fp8_kv(
        attn, k[:, :, :64], v[:, :, :64], kc, vc, fp8_cache_index(1, 128, 0, 64)
    )
    attn._kv_scales_set = True

    store_fp8_kv(
        attn, k[:, :, 64:65], v[:, :, 64:65], kc, vc, fp8_cache_index(1, 128, 64, 1)
    )
    first = kc[1, :, :, 0].view(torch.uint8).clone()
    store_fp8_kv(
        attn, k[:, :, 65:66], v[:, :, 65:66], kc, vc, fp8_cache_index(1, 128, 65, 1)
    )
    assert torch.equal(kc[1, :, :, 0].view(torch.uint8), first)
    assert kc[1, :, :, 1].float().abs().sum() > 0


def test_outlier_token_saturates_instead_of_nan():
    """A token beyond the headroom is clamped to +-448, never stored as NaN."""
    _, k, v = _qkv(1, 64, seed=6)
    attn, kc, vc = _state(1, 128)
    store_fp8_kv(attn, k, v, kc, vc, fp8_cache_index(1, 128, 0, 64))
    attn._kv_scales_set = True

    _, k_big, v_big = _qkv(1, 1, seed=7, scale=50.0)  # far past 2.24x headroom
    store_fp8_kv(attn, k_big, v_big, kc, vc, fp8_cache_index(1, 128, 64, 1))

    for stored in (kc[1, :, :, 0].float(), vc[64].float()):
        assert torch.isfinite(stored).all()
        assert stored.abs().max().item() == 448.0
