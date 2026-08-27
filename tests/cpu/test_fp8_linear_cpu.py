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

"""CPU verification for ``FP8Linear`` v2: does the swap + on-the-fly quantize work?

``weight`` is now plain fp16, ``[in_features, out_features]`` — quantized fresh
on every forward call (both CPU and Spyre; see fp8_linear.py's module docstring
for why weight can't just be pre-quantized and moved once). This test builds a
random weight, computes the same reference an unquantized adapter would expect
(quantize per-channel, dequantize, plain matmul), replaces an nn.Linear with
FP8Linear, and checks the outputs track within FP8 quantization noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hf_adapters.fp8_linear import FP8_DTYPE, FP8_MAX, FP8Linear, replace_linear_with_fp8

DTYPE = torch.float16
IN_FEATURES = 128
OUT_FEATURES = 64


def _rel_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    diff = (actual.float() - expected.float()).norm()
    return (diff / expected.float().norm()).item()


def test_replace_linear_with_fp8_matches_dequantized_reference():
    torch.manual_seed(0)
    raw_weight = torch.randn(OUT_FEATURES, IN_FEATURES, dtype=DTYPE)  # [out, in]
    scale = (raw_weight.abs().amax(dim=1) / FP8_MAX).clamp(min=1e-12)  # [out]

    # Reference: what FP8Linear represents internally — quantize raw_weight
    # per-channel, dequantize, plain matmul. Not what FP8Linear literally does
    # (it quantizes fresh every call), but numerically what it should produce.
    qweight = (raw_weight / scale.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    dequantized = qweight.to(DTYPE) * scale.to(DTYPE).unsqueeze(1)
    ref = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=DTYPE)
    ref.weight = nn.Parameter(dequantized, requires_grad=False)

    parent = nn.Module()
    parent.proj = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=DTYPE)
    fp8 = replace_linear_with_fp8(parent, "proj", dtype=DTYPE)
    fp8.weight.copy_(raw_weight.t())  # [in, out] — orientation flip, plain fp16
    fp8.weight_scale.copy_(scale.to(DTYPE))

    assert parent.proj is fp8
    assert isinstance(fp8, FP8Linear)
    assert fp8.weight.dtype == DTYPE  # fp16 at rest — quantized inside forward()
    assert fp8.weight.shape == (IN_FEATURES, OUT_FEATURES)

    x = torch.randn(2, 16, IN_FEATURES, dtype=DTYPE)
    out = parent.proj(x)
    expected = ref(x)

    assert out.shape == expected.shape
    assert out.dtype == DTYPE
    assert torch.isfinite(out).all()
    assert _rel_error(out, expected) < 0.05


def test_weight_is_a_buffer_not_a_parameter():
    fp8 = FP8Linear(IN_FEATURES, OUT_FEATURES, dtype=DTYPE)
    names = {name for name, _ in fp8.named_parameters()}
    assert "weight" not in names and "weight_scale" not in names


def test_all_zero_row_does_not_produce_nan():
    fp8 = FP8Linear(IN_FEATURES, OUT_FEATURES, dtype=DTYPE)
    fp8.weight.copy_(torch.randn(IN_FEATURES, OUT_FEATURES, dtype=DTYPE))
    fp8.weight_scale.fill_(1.0)

    x = torch.zeros(3, IN_FEATURES, dtype=DTYPE)
    x[1] = torch.randn(IN_FEATURES, dtype=DTYPE)

    out = fp8(x)
    assert torch.isfinite(out).all()
    assert torch.equal(out[0], torch.zeros(OUT_FEATURES, dtype=DTYPE))
