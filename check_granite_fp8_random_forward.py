"""First real forward pass on the swapped Granite model — random weights.

Goal: does the wiring work at all? Does FP8Linear interoperate correctly with
real Granite's RMSNorm/residual/attention structure, sitting in place of the
real Linears? Deliberately NOT testing numerical correctness here (that's the
next milestone, with real checkpoint weights + the dequantize verification) —
random data isolates "does this run and produce a sane shape/dtype/finite
output" from "is the specific data right".

Loads the real 8B checkpoint (for its architecture — GQA head counts,
embedding/logits/residual multipliers, etc.), swaps every quantized Linear for
FP8Linear, fills each with random data, and calls model(input_ids=...) once —
the model's own stock forward(), not the full adapter pipeline
(prepare_for_spyre/_run_forward/compiled blocks) — since the question here is
purely "does FP8Linear plug in correctly", not "does the Spyre-specific
generate() path work".

Run: python3 check_granite_fp8_random_forward.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # checkpoint is already cached

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from hf_adapters.fp8_linear import FP8_DTYPE, replace_linear_with_fp8

PATH = "ibm-granite/granite-3.3-8b-instruct-FP8"


def main():
    torch.manual_seed(0)

    print(f"Loading {PATH} (stock AutoModelForCausalLM.from_pretrained) ...")
    model = AutoModelForCausalLM.from_pretrained(
        PATH, dtype=torch.float16, device_map="cpu"
    ).eval()

    # Collect targets first — mutating the module tree while iterating
    # named_modules() over it is unsafe.
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.weight.dtype == FP8_DTYPE
    ]
    print(f"Found {len(targets)} quantized nn.Linear modules to swap.")

    for name in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        fp8 = replace_linear_with_fp8(parent, attr, dtype=torch.float16)

        # Random fill, not dequantized checkpoint data — see module docstring
        # for why: this step is purely about whether the wiring works.
        fp8.weight.copy_(
            torch.randn(fp8.in_features, fp8.out_features, dtype=torch.float16) * 0.02
        )
        fp8.weight_scale.copy_(
            torch.full((fp8.out_features,), 0.1, dtype=torch.float16)
        )

    # Safety check: no module was missed by the swap. A module
    # compressed_tensors still considers quantized (still carries
    # `quantization_scheme`) would silently decompress to bf16 on this exact
    # forward call, via its leftover forward-pre-hook, instead of erroring.
    leftover = [
        name for name, m in model.named_modules() if hasattr(m, "quantization_scheme")
    ]
    print(f"\nSwap-completeness check: {len(leftover)} leftover quantized module(s) "
          f"{'(FAIL — these will silently decompress on forward)' if leftover else '(PASS)'}")
    if leftover:
        print(f"  {leftover[:5]}{' ...' if len(leftover) > 5 else ''}")

    print(f"\nq_proj[0] after swap: {type(model.model.layers[0].self_attn.q_proj).__name__}, "
          f"weight dtype={model.model.layers[0].self_attn.q_proj.weight.dtype}")

    print("\n" + "=" * 70)
    print("Running one forward pass — model(input_ids=...), stock forward()")
    print("=" * 70)
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits

    print(f"logits: dtype={logits.dtype}, shape={tuple(logits.shape)}")
    finite = torch.isfinite(logits).all().item()
    print(f"finite: {finite}")

    expected_shape = (1, 8, model.config.vocab_size)
    shape_ok = tuple(logits.shape) == expected_shape

    if finite and shape_ok and not leftover:
        print(f"\nPASS — forward pass ran end to end with FP8Linear in place, "
              f"correct shape {expected_shape}, no NaN/Inf")
    else:
        print(f"\nFAIL — finite={finite}, shape_ok={shape_ok} (expected "
              f"{expected_shape}, got {tuple(logits.shape)}), leftover={len(leftover)}")


if __name__ == "__main__":
    main()
