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
layouts and their indices, the decode staging block, chunked prefill, the
static Q/K/V scales (checkpoint or 1.0), the clamp, and the scale bookkeeping
around the FP8 Q·Kᵀ.
Device-side questions — scaled_bmm (torch-spyre#5158), whole-block QFP8WT
scatter, cross-graph layouts — cannot be tested here.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from hf_adapters.fp8_kv import (
    KV_BLOCK,
    _sliced_reads,
    _static_scale,
    block_decode_mask,
    decode_group_blocks,
    fp8_attention,
    fp8_cache_index,
    fp8_qk_attention,
    prefill_chunk_size,
    store_fp8_kv,
)

N_Q, N_KV, DH = 8, 2, 128
SCALING = DH**-0.5


def _state(b, m, scales=(1.0, 1.0, 1.0)):
    """An attention module's FP8 attributes, plus empty K and V caches."""
    _q_s, k_s, v_s = scales
    attn = SimpleNamespace(
        scaling=SCALING,
        _fp8_scales=scales,
        _k_scale=torch.full((1, 1, 1, 1), k_s, dtype=torch.float16),
        _v_scale=torch.full((1, 1, 1), v_s, dtype=torch.float16),
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
    """Prefill: FP8 Q·Kᵀ with static unit Q/K/V scales stays close to fp32."""
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
    for pos in range(pre, pre + steps):
        sl = slice(pos, pos + 1)
        idx = fp8_cache_index(b, m, pos, 1)
        store_fp8_kv(attn, k[:, :, sl], v[:, :, sl], kc, vc, idx)
        mask = _mask(pos, 1, m).expand(b, mask_heads, 1, m).contiguous()
        if b == 1:  # as hf_common's decode loop hands it over
            mask = block_decode_mask(mask)
        got = fp8_attention(attn, q[:, :, sl], kc, vc, mask)
        assert torch.isfinite(got).all(), pos
        assert _cos(got, _ref(q[:, :, sl], k, v, pos, m)) > 0.995, pos


@pytest.mark.parametrize("b", [1, 2])
def test_chunked_prefill_matches_one_shot(b):
    """Prefill in 128-position chunks stores the same bytes as one shot, and each
    chunk's attention over the whole cache (later positions masked) tracks fp32."""
    length, chunk, m = 384, 128, 512
    q, k, v = _qkv(b, length, seed=12)

    attn1, kc1, vc1 = _state(b, m)
    store_fp8_kv(attn1, k, v, kc1, vc1, fp8_cache_index(b, m, 0, length))

    attn, kc, vc = _state(b, m)
    for start in range(0, length, chunk):
        sl = slice(start, start + chunk)
        idx = fp8_cache_index(b, m, start, chunk)
        store_fp8_kv(attn, k[:, :, sl], v[:, :, sl], kc, vc, idx)
        got = fp8_attention(attn, q[:, :, sl], kc, vc, _mask(start, chunk, m))
        assert torch.isfinite(got).all(), start
        assert _cos(got, _ref(q[:, :, sl], k, v, start, m)) > 0.995, start

    assert torch.equal(kc.view(torch.uint8), kc1.view(torch.uint8))
    assert torch.equal(vc.view(torch.uint8), vc1.view(torch.uint8))


def test_stores_never_change_the_static_scales():
    """Scales are static: no store, prefill or decode, writes them."""
    _, k, v = _qkv(1, 65, seed=3, scale=5.0)
    attn, kc, vc = _state(1, 128)
    before = (attn._k_scale.clone(), attn._v_scale.clone(), attn._fp8_scales)
    store_fp8_kv(
        attn, k[:, :, :64], v[:, :, :64], kc, vc, fp8_cache_index(1, 128, 0, 64)
    )
    store_fp8_kv(
        attn, k[:, :, 64:], v[:, :, 64:], kc, vc, fp8_cache_index(1, 128, 64, 1)
    )
    assert torch.equal(attn._k_scale, before[0])
    assert torch.equal(attn._v_scale, before[1])
    assert attn._fp8_scales == before[2]


def test_checkpoint_scales_are_used():
    """Static scales from the checkpoint: 1.0 when absent, the value when per
    layer, rejected when per KV head (not supported yet)."""
    plain = SimpleNamespace(_checkpoint_scales={})
    assert _static_scale(plain, "k_scale") == 1.0
    ckpt = SimpleNamespace(_checkpoint_scales={"k_scale": torch.tensor([0.25])})
    assert _static_scale(ckpt, "k_scale") == 0.25
    per_head = SimpleNamespace(_checkpoint_scales={"v_scale": torch.ones(N_KV)})
    with pytest.raises(NotImplementedError):
        _static_scale(per_head, "v_scale")


def test_small_activations_need_their_static_scale():
    """Q/K/V around 1e-3: E4M3 at scale 1.0 is subnormal there, while a matching
    static scale keeps full precision. Shows the scales are applied end to end."""
    b, length, m, s = 1, 64, 128, 1e-3
    q, k, v = _qkv(b, length, seed=13, scale=s)
    ref = _ref(q, k, v, 0, m)
    cos = {}
    for scales in ((1.0, 1.0, 1.0), (s, s, s)):
        attn, kc, vc = _state(b, m, scales)
        store_fp8_kv(attn, k, v, kc, vc, fp8_cache_index(b, m, 0, length))
        cos[scales[0]] = _cos(fp8_attention(attn, q, kc, vc, _mask(0, length, m)), ref)
    assert cos[s] > 0.995
    assert cos[s] > cos[1.0]


def test_decode_rewrite_keeps_earlier_tokens():
    """Rewriting a block from staging leaves its earlier positions' bytes as is."""
    _, k, v = _qkv(1, 66, seed=5)
    attn, kc, vc = _state(1, 128)
    store_fp8_kv(
        attn, k[:, :, :64], v[:, :, :64], kc, vc, fp8_cache_index(1, 128, 0, 64)
    )

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
    """A value beyond +-448 (at its static scale) is clamped, never stored as NaN."""
    _, k, v = _qkv(1, 64, seed=6)
    attn, kc, vc = _state(1, 128)
    store_fp8_kv(attn, k, v, kc, vc, fp8_cache_index(1, 128, 0, 64))

    _, k_big, v_big = _qkv(1, 1, seed=7, scale=1000.0)  # randn * 1000 >> 448
    store_fp8_kv(attn, k_big, v_big, kc, vc, fp8_cache_index(1, 128, 64, 1))

    for stored in (kc[1, :, :, 0].float(), vc[64].float()):
        assert torch.isfinite(stored).all()
        assert stored.abs().max().item() == 448.0


@pytest.mark.parametrize("b", [1, 2])
def test_fp8_qk_attention_on_fp16_cache(b):
    """FP8_QK_ATTN: prefill and decode on an ordinary fp16 cache track fp32."""
    pre, m = 64, 128
    q, k, v = _qkv(b, pre + 3, seed=11)
    attn = SimpleNamespace(scaling=SCALING)
    kc = torch.zeros(b, N_KV, m, DH, dtype=torch.float16)
    vc = torch.zeros(b, N_KV, m, DH, dtype=torch.float16)
    kc[:, :, :pre], vc[:, :, :pre] = k[:, :, :pre], v[:, :, :pre]
    got = fp8_qk_attention(attn, q[:, :, :pre], kc, vc, _mask(0, pre, m))
    assert torch.isfinite(got).all()
    assert _cos(got, _ref(q[:, :, :pre], k, v, 0, m)) > 0.995
    for pos in range(pre, pre + 3):  # decode
        sl = slice(pos, pos + 1)
        kc[:, :, sl], vc[:, :, sl] = k[:, :, sl], v[:, :, sl]
        got = fp8_qk_attention(attn, q[:, :, sl], kc, vc, _mask(pos, 1, m))
        assert _cos(got, _ref(q[:, :, sl], k, v, pos, m)) > 0.995, pos


def test_prefill_chunk_size_from_env(monkeypatch):
    """FP8_KV_PREFILL_CHUNK overrides the chunk; it must be a multiple of 64."""
    monkeypatch.delenv("FP8_KV_PREFILL_CHUNK", raising=False)
    assert prefill_chunk_size(512) == 512
    monkeypatch.setenv("FP8_KV_PREFILL_CHUNK", "")
    assert prefill_chunk_size(512) == 512
    monkeypatch.setenv("FP8_KV_PREFILL_CHUNK", "64")
    assert prefill_chunk_size(512) == 64
    for bad in ("0", "96", "-64", "abc"):
        monkeypatch.setenv("FP8_KV_PREFILL_CHUNK", bad)
        with pytest.raises(ValueError):
            prefill_chunk_size(512)


def test_decode_group_blocks_default_and_env(monkeypatch):
    """Default group keeps one group's scores at 1 MiB; the env var overrides."""
    monkeypatch.delenv("FP8_KV_DECODE_GROUP", raising=False)
    assert decode_group_blocks(8, 4) == 256  # Granite 3.3 8B: 256 x 4 KiB
    assert decode_group_blocks(N_KV, N_Q // N_KV) == 1024
    monkeypatch.setenv("FP8_KV_DECODE_GROUP", "3")
    assert decode_group_blocks(8, 4) == 3
    for bad in ("0", "-2", "x"):
        monkeypatch.setenv("FP8_KV_DECODE_GROUP", bad)
        with pytest.raises(ValueError):
            decode_group_blocks(8, 4)


def test_sliced_reads_from_env(monkeypatch):
    """Split decode slices K, V and the mask unless FP8_KV_DECODE_SLICE narrows it;
    unsplit decode never slices."""
    monkeypatch.delenv("FP8_KV_DECODE_SLICE", raising=False)
    assert _sliced_reads(True) == "kvm"
    assert _sliced_reads(False) == ""
    for env, want in (("0", ""), ("1", "kvm"), ("K", "k"), ("vm", "vm")):
        monkeypatch.setenv("FP8_KV_DECODE_SLICE", env)
        assert _sliced_reads(True) == want
        assert _sliced_reads(False) == ""
    monkeypatch.setenv("FP8_KV_DECODE_SLICE", "kx")
    with pytest.raises(ValueError):
        _sliced_reads(True)


# split: caches whole / sliced per group; slice: FP8_KV_DECODE_SLICE when split.
@pytest.mark.parametrize(
    "split, slice_env", [(False, "1"), (True, "1"), (True, "k"), (True, "vm")]
)
@pytest.mark.parametrize("group", ["1", "2", "3"])
def test_grouped_decode_matches_one_group(monkeypatch, group, split, slice_env):
    """Decode over groups of K blocks (online softmax, uneven last group) gives
    the same output as one group over the whole cache, and tracks fp32."""
    pre, m = 192, 320  # 5 K blocks; decode at positions 192..195 (block 3)
    q, k, v = _qkv(1, pre + 4, seed=21)
    monkeypatch.setenv("FP8_KV_DECODE_SLICE", slice_env)
    outs = {}
    for g in ("1000", group):
        monkeypatch.setenv("FP8_KV_DECODE_GROUP", g)
        attn, kc, vc = _state(1, m)
        # The reference (one group) always reads the caches whole.
        attn._fp8_split_decode = split and g != "1000"
        store_fp8_kv(
            attn, k[:, :, :pre], v[:, :, :pre], kc, vc, fp8_cache_index(1, m, 0, pre)
        )
        got = []
        for pos in range(pre, pre + 4):
            sl = slice(pos, pos + 1)
            store_fp8_kv(
                attn, k[:, :, sl], v[:, :, sl], kc, vc, fp8_cache_index(1, m, pos, 1)
            )
            mask = block_decode_mask(_mask(pos, 1, m))
            got.append(fp8_attention(attn, q[:, :, sl], kc, vc, mask))
        outs[g] = got
    for pos, (one, grouped) in enumerate(zip(outs["1000"], outs[group]), start=pre):
        torch.testing.assert_close(grouped.float(), one.float(), rtol=2e-3, atol=2e-3)
        assert _cos(grouped, _ref(q[:, :, pos : pos + 1], k, v, pos, m)) > 0.995
