"""Verify swap_linears_to_fp8 loads the REAL checkpoint weights correctly.

CPU only -- no Spyre, no torch.compile, no forward pass. Runs in seconds. This
checks the one piece of the FP8 path that had never existed until now: getting
the checkpoint's genuine E4M3 weights into FP8Linear (and into the excluded
fp16 Linears) without corrupting them.

Two properties are checked, per module:

1. SWAPPED (-> FP8Linear). FP8Linear stores a DEQUANTIZED fp16 weight and
   re-quantizes it on every forward with the checkpoint's own per-channel
   scale. Since the same scale is used in both directions, that round trip must
   return the checkpoint's original E4M3 values:

       stored[i,o]      = e4m3[o,i] * scale[o]                (this script's load)
       forward's wq[i,o] = clamp(stored[i,o] / scale[o]).to(e4m3)
                        == e4m3[o,i]                          (must hold)

   Re-quantization here mirrors FP8Linear.forward's Spyre-path arithmetic
   exactly (reciprocal-multiply then clamp, not divide), so a mismatch means
   the load is wrong, not that the check is approximating.

2. EXCLUDED (-> plain fp16 nn.Linear, currently o_proj). Must equal
   e4m3 * scale in the ORIGINAL [out, in] orientation. The failure mode this
   catches is the one the old check_granite_fp8_swap.py had: copying E4M3 bytes
   without applying weight_scale, which is silently ~448x wrong.

Exact equality is the bar for (1): E4M3's 3-bit mantissa is a coarse subset of
fp16's 10-bit, so multiplying and dividing by the same scale should land back
on the same E4M3 grid point. Any mismatch is reported with counts rather than
asserted away, so a near-miss is visible instead of hidden.

Run: python3 check_fp8_checkpoint_load_cpu.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # checkpoint is already cached

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from hf_adapters.fp8_linear import (
    DEFAULT_FP8_EXCLUDE,
    FP8_DTYPE,
    FP8_MAX,
    FP8Linear,
    swap_linears_to_fp8,
)

PATH = "ibm-granite/granite-3.3-8b-instruct-FP8"
N_SAMPLE = 8  # modules to verify in full; the rest are only counted


def main():
    print(f"Loading {PATH} on CPU ...")
    model = AutoModelForCausalLM.from_pretrained(
        PATH, dtype=torch.float16, device_map="cpu"
    ).eval()

    # Snapshot the genuine checkpoint values BEFORE the swap. compressed-tensors
    # keeps them in E4M3 until the first forward, and none has run.
    originals = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear) and m.weight.dtype == FP8_DTYPE:
            originals[name] = (
                m.weight.detach().clone(),        # E4M3 [out, in]
                m.weight_scale.detach().clone(),  # [out, 1]
            )
    print(f"Found {len(originals)} quantized Linear modules.")

    n_fp8, n_excluded = swap_linears_to_fp8(model)
    print(f"swap_linears_to_fp8 -> {n_fp8} FP8Linear, {n_excluded} fp16 nn.Linear")
    print(f"excluded projections: {DEFAULT_FP8_EXCLUDE}\n")

    sampled = sorted(originals)[:N_SAMPLE] + sorted(originals)[-N_SAMPLE:]
    checked = 0
    failures = []

    for name in sampled:
        orig_w, orig_scale = originals[name]
        mod = model.get_submodule(name)

        if isinstance(mod, FP8Linear) and mod.prequantized:
            # DIRECT_FP8_LOAD: the stored bytes ARE the checkpoint's, only
            # transposed. No round trip to verify -- compare them directly.
            got = mod.weight.t().contiguous()                    # [out, in]
            want = orig_w
            kind = "FP8Linear/direct"
        elif isinstance(mod, FP8Linear):
            # Mirror FP8Linear.forward's Spyre-path quantization exactly.
            scale = mod.weight_scale.to(torch.float32)          # [out]
            requant = (
                (mod.weight.to(torch.float32) * torch.reciprocal(scale))
                .clamp(-FP8_MAX, FP8_MAX)
                .to(FP8_DTYPE)
            )                                                    # [in, out]
            got = requant.t().contiguous()                       # [out, in]
            want = orig_w
            kind = "FP8Linear"
        else:
            scale = orig_scale.to(torch.float32)                 # [out, 1]
            want = (orig_w.to(torch.float32) * scale).to(torch.float16)
            got = mod.weight.detach()
            kind = type(mod).__name__

        same = torch.equal(got.to(torch.float32), want.to(torch.float32))
        n_diff = int((got.to(torch.float32) != want.to(torch.float32)).sum())
        total = want.numel()
        checked += 1
        status = "OK " if same else "MISMATCH"
        print(
            f"{status} {kind:<17} {name:<45} "
            f"{'exact' if same else f'{n_diff}/{total} differ'}"
        )
        if not same:
            failures.append((name, n_diff, total))

    print(f"\nchecked {checked} module(s) in full ({len(originals)} swapped total)")
    if failures:
        print("FAIL — weight load does not round-trip:")
        for name, n_diff, total in failures:
            print(f"  {name}: {n_diff}/{total} elements differ")
    else:
        print(
            "PASS — every sampled module round-trips to the checkpoint's "
            "original E4M3 values; excluded modules carry a correctly "
            "dequantized fp16 weight."
        )


if __name__ == "__main__":
    main()
