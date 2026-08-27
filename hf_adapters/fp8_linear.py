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

"""FP8 (E4M3) linear layer — v1, CPU-only.

Deliberately narrow first cut, hardcoded to what
``ibm-granite/granite-3.3-8b-instruct-FP8`` actually needs: per-channel
static E4M3 weights, per-token dynamic E4M3 activations, no bias. The goal of
this version is just to validate the basic swap-``nn.Linear``-for-
``FP8Linear`` mechanic; strategy options (per-tensor weights, static
activation scale, bias) and the Spyre compute path are deliberately left out
and will be layered in once this is verified.
"""

from __future__ import annotations

import torch
import torch.nn as nn

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max

# Floor for the dynamically-computed activation scale, guarding the all-zero
# row (scale 0 -> division by zero). Quantization multiplies by
# reciprocal(scale), so the floor must keep that reciprocal finite in fp16.
SCALE_EPS = 1e-4


class FP8Linear(nn.Module):
    """``nn.Linear`` stand-in: E4M3 weight, dynamic per-token FP8 matmul.

    ``weight``/``weight_scale`` are buffers, not parameters — ``E4M3`` reports
    ``is_floating_point() == True``, so as a parameter it could be mistaken
    for the model's compute dtype by code that infers dtype from the first
    floating-point parameter.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = dtype

        self.register_buffer(
            "weight", torch.empty(out_features, in_features, dtype=FP8_DTYPE)
        )
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scale = (x.abs().amax(dim=-1, keepdim=True) * (1.0 / FP8_MAX)).clamp(
            min=SCALE_EPS
        )
        xq = (x * torch.reciprocal(x_scale)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)

        # Raw (unscaled) matmul result can reach ~448*448*K, well past fp16's
        # 65504 — accumulate in fp32 and only cast back after the rescale.
        acc = xq.to(torch.float32) @ self.weight.t().to(torch.float32)
        y = acc * x_scale * self.weight_scale
        return y.to(x.dtype)

    def _apply(self, fn, recurse=True):
        """Keep the weight in E4M3 across ``.to()``/``.half()``/``.float()``.

        ``nn.Module._apply`` casts every buffer reporting
        ``is_floating_point()`` — and E4M3 does — so a plain cast would
        silently dequantize the weight. Restoring is lossless: every E4M3
        value round-trips exactly through fp16/bf16/fp32.
        """
        module = super()._apply(fn, recurse)
        if module.weight.dtype != FP8_DTYPE:
            module.weight = module.weight.to(FP8_DTYPE)
        return module

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


def replace_linear_with_fp8(
    parent: nn.Module, attr: str, *, dtype: torch.dtype = torch.float16
) -> FP8Linear:
    """Swap ``parent.<attr>`` (an ``nn.Linear``) for an empty ``FP8Linear``.

    Shape is taken from the ``nn.Linear`` being replaced; the buffers are left
    uninitialized for the caller to fill.
    """
    linear = getattr(parent, attr)
    fp8 = FP8Linear(linear.in_features, linear.out_features, dtype=dtype)
    setattr(parent, attr, fp8)
    return fp8
