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

"""FP8 (E4M3) linear layer — v3, CPU + Spyre.

Deliberately narrow, hardcoded to what ``ibm-granite/granite-3.3-8b-instruct-FP8``
actually needs: per-channel weight scale, per-token dynamic activation scale, no
bias. Strategy options (per-tensor weights, static activation scale, bias) are
still deliberately left out.

Design constraints — three things that are NOT how a "normal" quantized linear
would be designed, all forced by the current state of the Spyre stack, not by
choice (1 and 2 are about weight storage; 3 is about op ordering):

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

3. **``x_scale`` is cloned before the rescale; ``xq`` is flattened only
   immediately before ``scaled_mm``.** This mirrors the fms-model-optimizer
   pattern (``fp8_linear.py:259``: ``inpt_data.reshape(-1, K)`` after
   quantization) and is forced by a layout-propagation bug in torch-spyre.

   Spyre's ``propagate_layouts.py`` assigns a layout to every computed buffer
   by tracing backward through the op chain.  A rank-reducing reshape on a
   computed buffer (e.g. rank 3 → rank 2) produces a result whose layout
   still carries the ancestry of the higher-rank source.  When that buffer
   participates in a *multi-argument pointwise* (``len(args) > 1``), the pass
   calls ``_multi_arg_pointwise_layouts``, which computes::

       rank_diff = len(output.size) - len(arg.layout.size)   # e.g. 2 − 3 = −1

   The filter+shift projection assumes rank_diff ≥ 0 (broadcast direction).
   At −1 it produces a ``dim_order`` one entry short of the host dims, and
   ``SpyreTensorLayout`` rejects it: "Incompatible host_size and dim_order."

   ``x_scale`` inherits this tainted ancestry when ``x`` itself came from a
   rank-reducing reshape — e.g. ``o_proj`` receives SDPA's rank-4 output
   after ``transpose(1,2).reshape(bsz, seq_len, -1)`` (rank 4 → 3).
   ``abs().amax(dim=-1)`` is single-arg (safe), but the rescale
   ``y.reshape(out_shape) * x_scale * weight_scale`` is multi-arg and hits
   the bug.  Similarly, ``quantize_fp8_with_scale(x, x_scale)`` is
   multi-arg on a tainted ``x``.

   The fix: ``.clone()`` on ``x_scale`` after computing it.
   ``aten.clone.default`` is special-cased in ``propagate_layouts.py`` to
   always materialise a fresh, clean row-major buffer with no backward
   ancestry.  Once ``x_scale`` is clean, all multi-arg ops that consume it
   — ``quantize_fp8_with_scale(x, x_scale)`` and the final rescale — have
   no rank mismatch.  ``xq.reshape(-1, K)`` is then safe because ``xq``
   (FP8, fresh from ``quantize_fp8_with_scale``) is itself a clean buffer
   whose only consumer is the matmul, not a multi-arg pointwise.
   (``check_reshape_rank_bug_spyre.py`` isolates the pattern;
   ``check_fp8_attn_eager_probe_spyre.py`` documents clone as the remedy.)

All three constraints are believed to be temporary — properties of what
the Spyre compiler supports *today*, not of the FP8 math — and are expected to
simplify once direct FP8 layout transfer, in-place transpose support, and
rank-reducing layout propagation land.
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
        # False: ``weight`` is fp16 and forward re-quantizes it on every call.
        # True:  ``weight`` is already E4M3 in QFP8WT arrangement and forward
        #        feeds it to scaled_mm directly -- see prequantize_fp8_weights.
        # A plain bool, constant after prepare, so Dynamo specializes on it
        # without a graph break.
        self.prequantized = False
        self.register_buffer(
            "weight", torch.empty(in_features, out_features, dtype=dtype)
        )
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_shape = (*x.shape[:-1], self.out_features)

        if _is_spyre(x):
            # Compute x_scale at full input rank (per-token, shape (*batch, 1)).
            # .clone() on x_scale breaks any layout ancestry that x inherits
            # from its producer (e.g. SDPA rank-4 output → transpose+reshape to
            # rank-3 for o_proj). Without the clone, x_scale carries that
            # rank-reducing ancestry; when it then participates in the rescale
            # `y.reshape(out_shape) * x_scale` as a multi-arg pointwise,
            # propagate_layouts.py traces back through the ancestry and hits
            # "Incompatible host_size and dim_order" (see constraint 3 in the
            # module docstring and check_reshape_rank_bug_spyre.py).
            # aten.clone.default is special-cased to always produce a fresh
            # row-major buffer with no ancestry (propagate_layouts.py:
            # aten.clone.default handler).  x_scale after clone is a clean
            # rank-(*batch,1) tensor; the rescale is then a safe rank-matched
            # multi-arg pointwise.
            x_scale = (
                x.abs().amax(dim=-1, keepdim=True) * (1.0 / FP8_MAX)
            ).clamp(min=SCALE_EPS).clone()             # (*batch, 1), clean layout
            # When prequantized, `weight` IS the QFP8WT E4M3 tensor and enters
            # the graph as an input rather than being produced inside it. That
            # is the whole point -- it removes one op per projection per forward
            # and lets the fp16 copy be freed -- but it also means the layout
            # pass sees an opaque input arrangement rather than a chain it built
            # itself, which is exactly where a restickify can appear.
            wq = (
                self.weight
                if self.prequantized
                else torch.ops.spyre.quantize_weight_fp8_with_scale(
                    self.weight, self.weight_scale
                )
            )
            # Quantize at full rank so xq retains the input's natural shape,
            # then flatten only xq for scaled_mm — exactly the fms-model-
            # optimizer pattern (fp8_linear.py:259: inpt_data.reshape(-1, K)).
            # The rescale multiplies run at the output's full rank (out_shape),
            # matching x_scale's rank; no rank mismatch for propagate_layouts.
            xq = torch.ops.spyre.quantize_fp8_with_scale(x, x_scale)
            y = torch.ops.spyre.scaled_mm(
                xq.reshape(-1, xq.shape[-1]), wq, out_dtype=self.compute_dtype
            )
            y = y.reshape(out_shape) * x_scale * self.weight_scale
            return y.to(x.dtype)

        # CPU reference path: accumulate in fp32 to avoid fp16 overflow on
        # large K; no clone needed (CPU Inductor has no Spyre layout pass).
        x_scale = (
            x.abs().amax(dim=-1, keepdim=True) * (1.0 / FP8_MAX)
        ).clamp(min=SCALE_EPS)

        wq = (self.weight * torch.reciprocal(self.weight_scale)).clamp(
            -FP8_MAX, FP8_MAX
        ).to(FP8_DTYPE)
        xq = (x * torch.reciprocal(x_scale)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)

        # Raw (unscaled) matmul result can reach ~448*448*K, well past fp16's
        # 65504 — accumulate in fp32 and only cast back after the rescale.
        acc = xq.reshape(-1, xq.shape[-1]).to(torch.float32) @ wq.to(torch.float32)
        y = (acc.reshape(out_shape) * x_scale * self.weight_scale).to(x.dtype)
        return y

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


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

# Projections excluded from the FP8 swap and rebuilt as plain fp16 nn.Linear.
#
# o_proj: its FP8 computation fails DeepTools codegen with
#   "sbf-ddc: DtException: Type incompatible with tensor role in operation"
#   (ddl_conversion.cpp:2217, program sdsc_13), because its input arrives via a
#   dimension-MERGING reshape -- attn_out.transpose(1,2).reshape(bsz, seq, -1),
#   folding heads x head_dim into hidden -- rather than the flat pass-through
#   q/k/v read. Isolated 2026-09-06: o_proj ALONE as FP8, with everything else
#   plain fp16, still crashes, so this is not about fusion count and not about
#   the other projections. Remove this exclusion once that codegen bug is
#   fixed; nothing in this file needs to change when it is.
# DEFAULT_FP8_EXCLUDE = ("o_proj",)
DEFAULT_FP8_EXCLUDE = ("o_proj", "down_proj")

def prequantize_fp8_weights(model: nn.Module) -> int:
    """Quantize every ``FP8Linear`` weight to E4M3/QFP8WT once, in place.

    Call AFTER the device move and after any ``_spyre_cpu_submodules`` restore,
    but BEFORE the first forward. ``torch.compile`` is lazy, so the blocks are
    wrapped but not yet traced at that point -- they get traced against the
    already-quantized weight, with ``prequantized`` True.

    Without this, ``forward`` re-runs ``quantize_weight_fp8_with_scale`` on every
    call for a weight that never changes: 200 redundant ops per token at the
    current 5/7 projection set, and the fp16 copy stays resident (~15.9 GB rather
    than ~10.7 GB for Granite 3.3 8B). Rebinding the buffer here drops the last
    reference to the fp16 tensor, which is where the memory actually comes back.

    REQUIRES a torch-spyre whose ``quantize_weight_fp8_with_scale`` has a real
    eager implementation. On stock builds it is a tracer stub with a literal
    ``pass`` body that returns **None** outside a compiled region -- assigning
    that to ``weight`` would produce a model that fails much later, somewhere
    unrecognizable, so the result is checked rather than trusted.

    Skips modules whose weight is not on Spyre (a QFP8WT arrangement is a device
    layout and cannot exist on CPU), which also makes this safe to call from
    CPU-only tests. Idempotent. Returns the number of modules quantized.
    """
    n = 0
    for name, m in model.named_modules():
        if not isinstance(m, FP8Linear) or m.prequantized:
            continue
        if m.weight.device.type != "spyre":
            continue

        wq = torch.ops.spyre.quantize_weight_fp8_with_scale(m.weight, m.weight_scale)

        if wq is None or not isinstance(wq, torch.Tensor):
            raise RuntimeError(
                f"quantize_weight_fp8_with_scale returned {type(wq).__name__} for "
                f"{name!r}. This torch-spyre has only the tracer stub (a literal "
                f"`pass` body), which is real solely inside a compiled region. "
                f"Prequantization needs a build with the eager implementation; "
                f"without it, leave FP8Linear.prequantized False and let forward "
                f"quantize inside the graph."
            )
        if wq.dtype != FP8_DTYPE:
            raise RuntimeError(
                f"expected {FP8_DTYPE} from quantize_weight_fp8_with_scale for "
                f"{name!r}, got {wq.dtype}"
            )

        m.weight = wq  # rebind: drops the last reference to the fp16 buffer
        m.prequantized = True
        n += 1

    if n:
        print(f"FP8: {n} weight(s) prequantized to E4M3/QFP8WT at load time")
    return n



def fp8_status(model: nn.Module) -> dict:
    """Report whether FP8 is actually in effect. Safe to call anywhere.

    Reads module types and weight metadata only -- never runs a forward and
    never touches ``FP8Linear.forward``. Instrumenting that method causes Dynamo
    graph breaks that change which kernels Inductor generates, which once
    produced a false "six projections work" result that stood for weeks. Call
    this AFTER ``move_model_to_spyre`` and outside any compiled region.

    Note what this does and does not prove. It confirms the swap happened and
    survived the device move. It does NOT confirm that ``scaled_mm`` is in the
    compiled graph -- if the blocks were compiled before the swap, these counts
    look right while the graph still holds a plain fp16 matmul. For that proof
    run with ``TORCH_COMPILE_DEBUG=1`` and grep ``fx_graph_readable.py`` for
    ``scaled_mm``.

    Returns ``n_fp8``, ``n_linear``, ``quantized_checkpoint`` (whether any
    un-swapped E4M3 Linear remains -- nonzero means the swap missed some), the
    sorted projection names in layer 0, and ``orientation_ok``.
    """
    n_fp8 = 0
    n_linear = 0
    n_unswapped_e4m3 = 0
    n_prequantized = 0
    n_weight_fp8 = 0
    orientation_ok = True

    for _, m in model.named_modules():
        if isinstance(m, FP8Linear):
            n_fp8 += 1
            if m.prequantized:
                n_prequantized += 1
            # Must track prequantized exactly: a prequantized module holds E4M3,
            # a non-prequantized one holds fp16. A count mismatch means the
            # buffer rebind failed, or a weight was quantized without the flag
            # being set -- either way forward takes the wrong branch.
            if m.weight.dtype == FP8_DTYPE:
                n_weight_fp8 += 1
            # FP8Linear stores [in, out]; nn.Linear stores [out, in]. A mismatch
            # means something re-wrote the buffer after the swap. Checked for
            # prequantized modules too: QFP8WT changes the physical arrangement
            # but should leave the logical shape alone.
            if tuple(m.weight.shape) != (m.in_features, m.out_features):
                orientation_ok = False
        elif isinstance(m, nn.Linear):
            n_linear += 1
            if m.weight.dtype == FP8_DTYPE:
                n_unswapped_e4m3 += 1

    names = sorted(
        name.split(".")[-1]
        for name, m in model.named_modules()
        if isinstance(m, FP8Linear) and ".0." in f".{name}."
    )

    return {
        "n_fp8": n_fp8,
        "n_linear": n_linear,
        "n_unswapped_e4m3": n_unswapped_e4m3,
        "n_prequantized": n_prequantized,
        "n_weight_fp8": n_weight_fp8,
        "layer0_fp8_projections": names,
        "orientation_ok": orientation_ok,
    }

def _dequantize_checkpoint_weight(linear: nn.Module) -> torch.Tensor:
    """Reconstruct the fp16 weight from a compressed-tensors E4M3 Linear.

    A compressed-tensors checkpoint stores ``weight`` as E4M3 ``[out, in]`` and
    ``weight_scale`` as ``[out, 1]`` (per-output-channel, symmetric). The real
    weight is the product; ``weight_scale`` broadcasts along ``in``.

    Returns ``[out, in]`` fp16 -- the ``nn.Linear`` orientation. Callers wanting
    ``FP8Linear``'s ``[in, out]`` storage transpose afterwards.
    """
    scale = linear.weight_scale.to(torch.float32)
    return (linear.weight.to(torch.float32) * scale).to(torch.float16)


def swap_linears_to_fp8(
    model: nn.Module,
    *,
    exclude: tuple[str, ...] | None = None,
    dtype: torch.dtype = torch.float16,
) -> tuple[int, int]:
    """Replace a compressed-tensors checkpoint's quantized Linears with FP8Linear.

    Operates on a model freshly loaded by ``AutoModelForCausalLM.from_pretrained``
    and NOT yet moved to Spyre or forward-ed. compressed-tensors keeps weights in
    E4M3 until the first forward, so the genuine quantized values are still
    present here -- this reads them directly, with no re-quantization.

    Every quantized Linear is handled, one of two ways:

    * **swapped** -> ``FP8Linear``, weight dequantized to fp16 and transposed to
      ``[in, out]``, ``weight_scale`` flattened to ``[out]``. Because the
      checkpoint's own per-channel scale is reused verbatim, ``FP8Linear.forward``'s
      re-quantization (``clamp(weight * reciprocal(weight_scale)).to(E4M3)``)
      round-trips back to the checkpoint's original E4M3 values.
    * **excluded** (see ``DEFAULT_FP8_EXCLUDE``) -> plain fp16 ``nn.Linear``,
      weight dequantized, orientation unchanged. Rebuilt rather than left alone:
      an untouched module keeps its E4M3 weight, and a later ``.to(fp16)`` would
      cast those bytes WITHOUT applying ``weight_scale`` -- silently ~448x wrong.

    Matching is on the attribute name and on every component of the dotted path,
    since compressed-tensors can wrap a named projection inside a container whose
    own attribute name is generic.

    ``exclude`` defaults to ``DEFAULT_FP8_EXCLUDE``, resolved HERE rather than as
    a default argument so the module-level constant can be overridden at runtime
    (a def-time default would capture the tuple at import and ignore later
    changes). Test harnesses sweep exclusion sets that way.

    Returns ``(n_swapped, n_excluded)``.
    """
    if exclude is None:
        exclude = DEFAULT_FP8_EXCLUDE

    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.weight.dtype == FP8_DTYPE
    ]

    n_swapped = 0
    n_excluded = 0
    for name in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, attr)

        is_excluded = attr in exclude or any(
            part in exclude for part in name.split(".")
        )
        weight_fp16 = _dequantize_checkpoint_weight(old)  # [out, in]

        if is_excluded:
            plain = nn.Linear(
                old.in_features, old.out_features, bias=False, dtype=dtype
            )
            plain.weight.data.copy_(weight_fp16)
            setattr(parent, attr, plain)
            n_excluded += 1
            continue

        fp8 = replace_linear_with_fp8(parent, attr, dtype=dtype)
        # [out, in] -> [in, out]; see the module docstring on why FP8Linear
        # stores pre-transposed (a .t() on an already-quantized E4M3 tensor
        # inside a compiled region hits ReStickifyOpHBM).
        fp8.weight.copy_(weight_fp16.t().contiguous())
        fp8.weight_scale.copy_(old.weight_scale.reshape(-1).to(dtype))
        n_swapped += 1

    return n_swapped, n_excluded
