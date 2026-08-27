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

"""FP8 (E4M3) linear layer — v2, CPU + Spyre.

Deliberately narrow, hardcoded to what ``ibm-granite/granite-3.3-8b-instruct-FP8``
actually needs: per-channel weight scale, per-token dynamic activation scale, no
bias. Strategy options (per-tensor weights, static activation scale, bias) are
still deliberately left out.

Weight storage — two things that are NOT how a "normal" quantized linear would
be designed, both forced by the current state of the Spyre stack, not by choice:

1. **The weight is stored as plain fp16, not E4M3, and quantized on every
   forward call.** The natural design would hold the already-quantized E4M3
   weight at rest (as v1 did) and just move it to the device once. That's not
   possible yet: there is no supported way to transfer an already-quantized
   E4M3 tensor directly onto Spyre in the physical layout ``scaled_mm`` needs
   (a special 2-stick ``[2, 64]`` arrangement — confirmed by tracing
   ``propagate_layouts.py``'s ``_qfp8wt_stl``, which every 2-D weight normally
   moved via the generic ``[1, 0]`` ``SpyreTensorLayout`` does *not* land in).
   The only path that reliably produces that layout today is running the
   weight through ``quantize_weight_fp8_with_scale`` *inside a compiled
   region* — so that's what happens, every call, until direct-layout transfer
   for FP8 tensors is supported. Wasteful, but it's what works right now.

2. **The weight is stored ``[in_features, out_features]``, not
   ``[out_features, in_features]`` (the ``nn.Linear``/checkpoint convention).**
   Calling ``.t()`` on an *already-quantized* E4M3 tensor inside a compiled
   region triggers exactly this failure on real hardware:
   ``Unsupported: Spyre backend does not support: ReStickifyOpHBM on
   DataFormats.SEN143_FP8`` — a transpose changes the logical shape, and the
   codegen that would physically rearrange an FP8 tensor to match isn't
   implemented. Storing the weight pre-transposed avoids ever needing a
   transpose after quantization. A caller filling this module from a
   checkpoint (``[out, in]``) needs to transpose once, in plain fp16, on CPU,
   before it ever reaches this module — cheap, and nowhere near a device.

Both of these constraints are believed to be temporary — properties of what
the Spyre compiler supports *today*, not of the FP8 math — and are expected to
simplify once direct FP8 layout transfer and in-place transpose support land.
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


def _is_spyre(t: torch.Tensor) -> bool:
    return t.device.type == "spyre"


class FP8Linear(nn.Module):
    """``nn.Linear`` stand-in: FP8 matmul, weight quantized on every call.

    ``weight`` is ``[in_features, out_features]`` fp16 (see module docstring
    for why — both the dtype and the orientation are load-bearing, not
    stylistic). ``weight_scale`` is ``[out_features]`` — a per-output-channel
    scale, used both to quantize the weight and to rescale the matmul output.
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
            "weight", torch.empty(in_features, out_features, dtype=dtype)
        )
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_scale = (x.abs().amax(dim=-1, keepdim=True) * (1.0 / FP8_MAX)).clamp(
            min=SCALE_EPS
        )

        if _is_spyre(x):
            # aten._scaled_mm's decomposition (scaled_mm_decomp) applies
            # scale_a/scale_b itself, inside the compiled graph — so the
            # result here is already fully rescaled. Do NOT multiply by
            # x_scale/weight_scale again below; that would double-scale.
            wq = torch.ops.spyre.quantize_weight_fp8_with_scale(
                self.weight, self.weight_scale
            )
            xq = torch.ops.spyre.quantize_fp8_with_scale(x, x_scale)
            y = torch.ops.aten._scaled_mm(
                xq,
                wq,
                scale_a=x_scale,
                scale_b=self.weight_scale,
                out_dtype=self.compute_dtype,
            )
            return y.to(x.dtype)

        # CPU reference path — mirrors the Spyre decompositions' arithmetic
        # exactly (reciprocal-multiply, not divide; raw matmul, then an
        # explicit rescale afterward), not just "numerically equivalent" math.
        wq = (self.weight * torch.reciprocal(self.weight_scale)).clamp(
            -FP8_MAX, FP8_MAX
        ).to(FP8_DTYPE)
        xq = (x * torch.reciprocal(x_scale)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)

        # Raw (unscaled) matmul result can reach ~448*448*K, well past fp16's
        # 65504 — accumulate in fp32 and only cast back after the rescale.
        acc = xq.to(torch.float32) @ wq.to(torch.float32)
        y = acc * x_scale * self.weight_scale
        return y.to(x.dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


def replace_linear_with_fp8(
    parent: nn.Module, attr: str, *, dtype: torch.dtype = torch.float16
) -> FP8Linear:
    """Swap ``parent.<attr>`` (an ``nn.Linear``) for an empty ``FP8Linear``.

    Shape is taken from the ``nn.Linear`` being replaced; the buffers are left
    uninitialized for the caller to fill. Note the orientation flip: filling
    ``weight`` from ``linear.weight`` (``[out, in]``) needs a ``.t()`` — do it
    once here, in plain fp16, on CPU; never transpose the module's weight
    buffer again after that (see module docstring for why).
    """
    linear = getattr(parent, attr)
    fp8 = FP8Linear(linear.in_features, linear.out_features, dtype=dtype)
    setattr(parent, attr, fp8)
    return fp8
