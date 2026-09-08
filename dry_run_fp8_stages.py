"""Dry run: trace checkpoint weights through every FP8 stage. CPU only.

No Spyre, no torch.compile, no device transfer. Prints dtype/shape/device/bytes
at each stage of the pipeline so the transformation is visible rather than
inferred:

    Stage 0/3  from_pretrained          E4M3 [out, in] + weight_scale [out, 1]
    Stage 4    _dequantize_checkpoint_weight   fp16 [out, in]
    Stage 5    swap_linears_to_fp8      fp16 [in, out] + weight_scale [out]
    Stage 7    FP8Linear.forward        re-quantize -> E4M3 -> matmul -> rescale

Stage 6 (``_move_to_spyre_with_layout``) is the only one not shown: by then the
weights are ordinary fp16 tensors and the transfer is not FP8-aware at all --
which is itself the point.

Two checks run alongside the trace:

* **Round trip.** Stage 4 multiplies by weight_scale; Stage 7 divides by the
  same buffer. E4M3's 3-bit mantissa is a coarse subset of fp16's 10-bit, so
  this must recover the checkpoint's ORIGINAL E4M3 values bit-exactly. Reported
  per module rather than asserted, so a near-miss stays visible.

* **Numerics.** FP8Linear's output vs an exact fp32 reference matmul on the
  same input. Because the weight round trip is exact, every bit of the observed
  error comes from the ACTIVATION scale (per-token, dynamic) -- the weight
  contributes none. Excluded projections (plain fp16 nn.Linear) are shown
  alongside as a control: their error is fp16 rounding only.

Run:
    python3 dry_run_fp8_stages.py                      # all 7 projections, layer 0
    python3 dry_run_fp8_stages.py --proj q_proj,k_proj,v_proj
    python3 dry_run_fp8_stages.py --layer 39 --seq 4
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # checkpoint is already cached

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from hf_adapters.fp8_linear import (
    DEFAULT_FP8_EXCLUDE,
    FP8_DTYPE,
    FP8_MAX,
    FP8Linear,
    _dequantize_checkpoint_weight,
    swap_linears_to_fp8,
)

PATH = "ibm-granite/granite-3.3-8b-instruct-FP8"
ALL_PROJ = (
    "q_proj", "k_proj", "v_proj", "o_proj",  # self_attn
    "gate_proj", "up_proj", "down_proj",     # mlp
)


def _opt(name, default=None):
    for i, a in enumerate(sys.argv):
        if a == name and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


LAYER = int(_opt("--layer", "0"))
SEQ = int(_opt("--seq", "4"))
PROJ = tuple(p.strip() for p in _opt("--proj", ",".join(ALL_PROJ)).split(",") if p.strip())


def mb(t):
    return t.numel() * t.element_size() / 1024**2


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main():
    print(f"Loading {PATH} on CPU (no Spyre) ...")
    model = AutoModelForCausalLM.from_pretrained(
        PATH, dtype=torch.float16, device_map="cpu"
    ).eval()

    # Resolve dotted paths: q/k/v/o live under self_attn, gate/up/down under mlp.
    names = {}
    for p in PROJ:
        sub = "mlp" if p in ("gate_proj", "up_proj", "down_proj") else "self_attn"
        names[p] = f"model.layers.{LAYER}.{sub}.{p}"

    # ---------------------------------------------------------------- Stage 3
    rule(f"STAGE 0/3 -- after from_pretrained (layer {LAYER})")
    print("compressed-tensors keeps weights in E4M3 until the first forward, and")
    print("none has run. The true value is weight * weight_scale, UNEVALUATED.\n")
    print(f"{'proj':<11} {'type':<10} {'weight':<26} {'dtype':<18} {'scale':<12} {'MB':>7}")
    print("-" * 78)
    for p, name in names.items():
        m = model.get_submodule(name)
        print(
            f"{p:<11} {type(m).__name__:<10} {str(tuple(m.weight.shape)):<26} "
            f"{str(m.weight.dtype).replace('torch.',''):<18} "
            f"{str(tuple(m.weight_scale.shape)):<12} {mb(m.weight):>7.1f}"
        )

    # Snapshot the genuine checkpoint values BEFORE the swap destroys them.
    originals = {
        p: (
            model.get_submodule(n).weight.detach().clone(),
            model.get_submodule(n).weight_scale.detach().clone(),
        )
        for p, n in names.items()
    }

    # ---------------------------------------------------------------- Stage 4
    rule("STAGE 4 -- _dequantize_checkpoint_weight (CPU, fp32 intermediate)")
    print("    scale  = weight_scale.to(fp32)                 # [out, 1]")
    print("    result = (weight.to(fp32) * scale).to(fp16)    # [out, in]")
    print("scale broadcasts along `in`: one scale per OUTPUT channel.\n")
    print(f"{'proj':<11} {'E4M3 [out,in]':<20} {'->':<3} {'fp16 [out,in]':<20} {'MB':>7} {'was':>7}")
    print("-" * 78)
    dequantized = {}
    for p, name in names.items():
        m = model.get_submodule(name)
        w16 = _dequantize_checkpoint_weight(m)
        dequantized[p] = w16
        print(
            f"{p:<11} {str(tuple(m.weight.shape)):<20} {'->':<3} "
            f"{str(tuple(w16.shape)):<20} {mb(w16):>7.1f} {mb(m.weight):>7.1f}"
        )

    # ---------------------------------------------------------------- Stage 5
    torch.manual_seed(0)
    n_fp8, n_excl = swap_linears_to_fp8(model)
    rule("STAGE 5 -- swap_linears_to_fp8 (whole model)")
    print(f"excluded projections : {DEFAULT_FP8_EXCLUDE}")
    print(f"280 quantized Linears -> {n_fp8} FP8Linear + {n_excl} fp16 nn.Linear\n")
    print("FP8Linear stores TRANSPOSED [in, out] -- so `out` is the LAST axis and a")
    print("flat weight_scale [out] broadcasts correctly in both directions later.\n")
    print(f"{'proj':<11} {'now':<11} {'weight':<20} {'dtype':<10} {'scale':<12} {'orient':<10}")
    print("-" * 78)
    for p, name in names.items():
        m = model.get_submodule(name)
        orient = "[in, out]" if isinstance(m, FP8Linear) else "[out, in]"
        print(
            f"{p:<11} {type(m).__name__:<11} {str(tuple(m.weight.shape)):<20} "
            f"{str(m.weight.dtype).replace('torch.',''):<10} "
            f"{str(tuple(m.weight_scale.shape)) if isinstance(m, FP8Linear) else '--':<12} "
            f"{orient:<10}"
        )

    # ------------------------------------------------------- round-trip check
    rule("ROUND TRIP -- Stage 4 (x scale) then Stage 7 (/ scale) must be EXACT")
    print("E4M3 grid spacing is ~2^-3 relative; fp16 rounding error is ~2^-11, so")
    print("re-quantizing lands back on the same grid point with huge margin.\n")
    for p, name in names.items():
        m = model.get_submodule(name)
        orig_w, orig_scale = originals[p]
        if isinstance(m, FP8Linear):
            s = m.weight_scale.to(torch.float32)
            requant = (
                (m.weight.to(torch.float32) * torch.reciprocal(s))
                .clamp(-FP8_MAX, FP8_MAX)
                .to(FP8_DTYPE)
            )
            got, want = requant.t().contiguous(), orig_w
        else:
            want = (orig_w.to(torch.float32) * orig_scale.to(torch.float32)).to(torch.float16)
            got = m.weight.detach()
        a, b = got.to(torch.float32), want.to(torch.float32)
        n_diff = int((a != b).sum())
        verdict = "exact" if n_diff == 0 else f"{n_diff}/{b.numel()} differ"
        print(f"  {p:<11} {type(m).__name__:<11} {verdict}")

    # ---------------------------------------------------------------- Stage 7
    rule(f"STAGE 7 -- forward (CPU reference path), x = [1, {SEQ}, in_features]")
    print("    x_scale = (|x|.amax(-1, keepdim) / 448).clamp(min=1e-4)   # per-TOKEN")
    print("    wq      = clamp(weight * reciprocal(weight_scale)) -> E4M3")
    print("    xq      = clamp(x * reciprocal(x_scale))           -> E4M3")
    print("    y       = (xq @ wq) * x_scale * weight_scale")
    print("\nWeights round-trip exactly, so ALL of the error below is activation-side.")
    print("Excluded (fp16 nn.Linear) rows are the control: fp16 rounding only.\n")
    print(f"{'proj':<11} {'module':<11} {'out shape':<16} {'cosine':<11} {'rel err':<10} {'max |d|':>9}")
    print("-" * 78)
    for p, name in names.items():
        m = model.get_submodule(name)
        in_f = m.in_features if isinstance(m, FP8Linear) else m.in_features
        torch.manual_seed(1234)  # identical x per projection shape, run to run
        x = (torch.randn(1, SEQ, in_f, dtype=torch.float16) * 0.5)

        with torch.no_grad():
            y = m(x)

        # Exact reference: the unquantized fp16 weight, matmul in fp32.
        orig_w, orig_scale = originals[p]
        w_ref = orig_w.to(torch.float32) * orig_scale.to(torch.float32)   # [out, in]
        y_ref = (x.to(torch.float32).reshape(-1, in_f) @ w_ref.t()).reshape(y.shape)

        yf = y.to(torch.float32)
        cos = torch.nn.functional.cosine_similarity(
            yf.flatten(), y_ref.flatten(), dim=0
        ).item()
        rel = ((yf - y_ref).norm() / y_ref.norm()).item()
        print(
            f"{p:<11} {type(m).__name__:<11} {str(tuple(y.shape)):<16} "
            f"{cos:<11.6f} {rel:<10.5f} {(yf - y_ref).abs().max().item():>9.4f}"
        )

    rule("SUMMARY")
    print("The E4M3 -> fp16 conversion happens ONCE, on CPU, at load (Stage 4).")
    print("The fp16 -> E4M3 direction happens on EVERY forward, inside the compiled")
    print("graph (Stage 7), using the same stored weight_scale -- hence bit-exact.")
    print()
    print("Cost of the detour: the weight sits on device as fp16, 2x the bytes it")
    print("would need as E4M3, plus one quantize op per projection per forward for")
    print("a weight that never changes. Removing it needs direct FP8 weight DMA")
    print("into QFP8WT layout (blocked on dciInfo) -- the parked optimization.")


if __name__ == "__main__":
    main()

