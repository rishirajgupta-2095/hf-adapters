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

"""CPU tests for the FP8 key cache math — no Spyre device needed.

These check the part that is ours to get right: that a per-token scale on Q and
on each cached K row is compensated correctly after the matmul, so FP8 attention
tracks an fp16 reference. The device-side questions (the 4D FP8 matmul, and
whether the scatter lowers onto a QFP8WT cache) cannot be tested here.
"""

import torch
import torch.nn.functional as F

from hf_adapters.fp8_kv import (
    MASK_FLOOR,
    fp8_gqa_attention_cpu,
    quantize_k_for_store,
)

B, N_Q, N_KV, L, DH = 1, 8, 2, 16, 128
SCALING = DH**-0.5


def _reference(q, k, v, mask):
    """fp16 attention over the un-quantized tensors."""
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, scale=SCALING, enable_gqa=True
    )


def _fp8_path(q, k, v, mask):
    """Quantize K as the cache would, then run the FP8 CPU attention path."""
    k_q, s_k = quantize_k_for_store(k)
    return fp8_gqa_attention_cpu(q, k_q, s_k, v, mask, SCALING)


def _inputs(seed=0, scale_spread=1.0):
    """Q/K/V with per-row magnitudes spread over ``scale_spread`` decades."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, N_Q, L, DH, generator=g, dtype=torch.float16)
    k = torch.randn(B, N_KV, L, DH, generator=g, dtype=torch.float16)
    v = torch.randn(B, N_KV, L, DH, generator=g, dtype=torch.float16)
    if scale_spread != 1.0:
        # Give each cached K row a different magnitude: this is what a single
        # static scale cannot represent and a per-token scale can.
        decades = torch.linspace(-scale_spread, scale_spread, L).view(1, 1, L, 1)
        k = (k.float() * torch.pow(10.0, decades)).to(torch.float16)
    return q, k, v


def test_fp8_kv_matches_fp16_reference():
    """FP8 K with per-token scales stays close to the fp16 result."""
    q, k, v = _inputs()
    ref = _reference(q, k, v, None).float()
    got = _fp8_path(q, k, v, None).float()

    cos = F.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
    rel = ((got - ref).norm() / ref.norm()).item()
    assert cos > 0.999, f"cosine {cos}"
    assert rel < 0.02, f"relative error {rel}"


def test_per_token_scale_handles_mixed_magnitude_rows():
    """Per-row K magnitudes spanning decades still track the reference.

    A single static scale for the whole cache would clip the large rows or
    flush the small ones; this is the case that justifies the scale cache.
    """
    q, k, v = _inputs(seed=1, scale_spread=2.0)
    ref = _reference(q, k, v, None).float()
    got = _fp8_path(q, k, v, None).float()

    cos = F.cosine_similarity(ref.flatten(), got.flatten(), dim=0).item()
    assert cos > 0.999, f"cosine {cos}"


def test_causal_mask_and_unwritten_slots():
    """Masked positions contribute nothing, and zero-scale slots stay finite.

    Cache slots past the current length hold zeros with a zero scale. Their
    scores come out 0 and the mask drives them to ``MASK_FLOOR``, so the result
    must match attention over the written prefix alone.
    """
    q, k, v = _inputs(seed=2)
    written = L // 2

    v_cache = torch.zeros_like(v)
    k_q, s_k = quantize_k_for_store(k[:, :, :written])
    k_cache_q = torch.zeros(k.shape, dtype=k_q.dtype)
    scale_cache = torch.zeros(B, N_KV, L, 1, dtype=s_k.dtype)
    k_cache_q[:, :, :written] = k_q
    scale_cache[:, :, :written] = s_k
    v_cache[:, :, :written] = v[:, :, :written]

    mask = torch.zeros(B, 1, L, L, dtype=torch.float16)
    mask[:, :, :, written:] = MASK_FLOOR

    got = fp8_gqa_attention_cpu(q, k_cache_q, scale_cache, v_cache, mask, SCALING)
    assert torch.isfinite(got).all()

    # Same thing with only the written prefix present must agree.
    short_mask = torch.zeros(B, 1, L, written, dtype=torch.float16)
    ref = fp8_gqa_attention_cpu(
        q,
        k_cache_q[:, :, :written],
        scale_cache[:, :, :written],
        v_cache[:, :, :written],
        short_mask,
        SCALING,
    )
    torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3)


def test_quantize_k_round_trip_is_lossy_but_unbiased():
    """Dequantizing a stored K row returns it to within E4M3 resolution."""
    _, k, _ = _inputs(seed=3)
    k_q, s_k = quantize_k_for_store(k)
    back = k_q.float() * s_k.float()

    rel = ((back - k.float()).norm() / k.float().norm()).item()
    assert rel < 0.05, f"round-trip relative error {rel}"
    # E4M3 has ~2 decimal digits; the mean error should be far below the spread.
    assert (back - k.float()).mean().abs().item() < 1e-2
