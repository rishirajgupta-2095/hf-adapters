"""Load the real Granite FP8 checkpoint and swap its quantized Linears for
FP8Linear, printing layer 0's architecture before and after.

Loads via stock AutoModelForCausalLM.from_pretrained — the checkpoint's own
compressed-tensors integration keeps weights in E4M3 until the first forward
pass, so as long as we never call the model, we can copy those genuine E4M3
weights straight into FP8Linear with no re-quantization.

Run: python3 check_granite_fp8_swap.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # checkpoint is already cached

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from hf_adapters.fp8_linear import FP8_DTYPE, replace_linear_with_fp8

PATH = "ibm-granite/granite-3.3-8b-instruct-FP8"


def main():
    print(f"Loading {PATH} (stock AutoModelForCausalLM.from_pretrained) ...")
    model = AutoModelForCausalLM.from_pretrained(
        PATH, dtype=torch.float16, device_map="cpu"
    ).eval()

    print("\n" + "=" * 70)
    print("BEFORE swap — model.model.layers[0]")
    print("=" * 70)
    print(model.model.layers[0])

    # Collect targets first — mutating the module tree while iterating
    # named_modules() over it is unsafe.
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.weight.dtype == FP8_DTYPE
    ]
    print(f"\nFound {len(targets)} quantized nn.Linear modules (still genuine "
          f"E4M3 — no forward pass has run yet, so nothing decompressed).")

    for name in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, attr)
        fp8 = replace_linear_with_fp8(parent, attr, dtype=torch.float16)
        fp8.weight.copy_(old.weight)          # already E4M3 — no re-quantization
        fp8.weight_scale.copy_(old.weight_scale.reshape(-1).to(torch.float16))

    print("\n" + "=" * 70)
    print("AFTER swap — model.model.layers[0]")
    print("=" * 70)
    print(model.model.layers[0])

    print(f"\nlm_head untouched (never matched the filter): "
          f"{type(model.lm_head).__name__}, dtype={model.lm_head.weight.dtype}")


if __name__ == "__main__":
    main()
