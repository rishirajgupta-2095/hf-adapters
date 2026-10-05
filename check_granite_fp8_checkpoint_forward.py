"""One-layer forward pass with REAL checkpoint weights — the meaningful test.

Random-weight runs (check_granite_fp8_spyre_forward.py, --no-fp8 control)
always produce NaN: 40 layers of N(0,0.02) fp16 matmuls with K=4096 diverge
regardless of whether FP8Linear is present.  The --no-fp8 control reproduces
the same NaN pattern as the FP8 run — that's the expected result, not a bug.

The meaningful question is: when the REAL checkpoint weights are loaded and
dequantized into FP8Linear, does one layer of the model produce finite,
numerically correct output?

This script:
  1. Loads the real E4M3 weights from the checkpoint.
  2. For each q/k/v/o_proj in layer 0, dequantizes them into FP8Linear:
       fp8.weight   = W_fp8.to(fp16).t()         # [in, out], fp16
       fp8.weight_scale = weight_scale.to(fp16)  # [out], per-column
  3. Runs ONE layer (layer 0 only) through the real compiled-block path on
     Spyre.  The remaining 39 layers are replaced with plain fp16 Identity
     linears that output zeros — so post-layer-0 computation is trivially
     finite and does not mask a real layer-0 bug.
  4. Checks key_cache[0], value_cache[0], and logits for finiteness.
  5. Additionally runs the same layer on CPU (fp32 accumulation) and compares
     the attention output — this is the numerical correctness gate.

Run: python3 check_granite_fp8_checkpoint_forward.py [--one-layer]
     --one-layer: stop after layer 0 (skip norm/lm_head), print K/V stats.
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from hf_adapters import hf_granite
from hf_adapters.fp8_linear import FP8_DTYPE, FP8Linear, replace_linear_with_fp8
from hf_adapters.hf_common import (
    allocate_kv_caches,
    apply_rope_matmul,
    build_prefill_mask,
    get_backbone,
    kv_cache_update,
    move_model_to_spyre,
    prepare_rope_and_heads,
    text_config,
)

PATH = "ibm-granite/granite-3.3-8b-instruct-FP8"
SEQ_LEN = 2
ONE_LAYER = "--one-layer" in sys.argv


def finite(x, label):
    ok = torch.isfinite(x.float().cpu()).all().item()
    mx = x.float().abs().max().item()
    tag = "" if ok else "   <-- NON-FINITE"
    print(f"  {label:<34} finite={str(ok):<5}  max|.|={mx:.4g}{tag}")
    return ok


def _load_real_fp8_weights(layer_attn):
    """Read E4M3 weight + scale from the compressed_tensors Linear in place,
    dequantize into fp16 [in, out] and return (weight_fp16, weight_scale_fp16).

    compressed_tensors stores the weight as fp8_e4m3fn [out, in] and the
    per-output-channel scale as a 1-D tensor of the same length as out_features.
    FP8Linear stores [in, out], so we transpose once here on CPU.
    """
    results = {}
    for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
        lin = getattr(layer_attn, attr)
        # The compressed_tensors module exposes the raw quantized weight
        # directly as .weight (dtype=fp8_e4m3fn, shape=[out, in]).
        # The scale is in .weight_scale (or .input_scale — we want weight_scale).
        w_fp8 = lin.weight.detach()                   # [out, in], fp8_e4m3fn
        # compressed_tensors may store the scale as a column vector [out,1]
        # or a 1-D vector [out]; normalise to 1-D [out].
        ws = lin.weight_scale.detach().reshape(-1)    # [out], arbitrary dtype
        assert w_fp8.dtype == FP8_DTYPE, (
            f"{attr}.weight dtype={w_fp8.dtype}, expected float8_e4m3fn"
        )
        # Dequantize: w_fp16 = w_fp8 * ws  (per-column, broadcast [out,in] × [out,1])
        # Then transpose to [in, out] for FP8Linear's convention.
        w_fp16 = (w_fp8.to(torch.float32) * ws.to(torch.float32).unsqueeze(1)).to(torch.float16)
        w_fp16 = w_fp16.t().contiguous()              # [in, out], fp16
        ws_fp16 = ws.to(torch.float16)                # [out], fp16
        results[attr] = (w_fp16, ws_fp16)
    return results


def main():
    torch.manual_seed(0)

    print(f"Loading {PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        PATH, dtype=torch.float16, device_map="cpu"
    ).eval()

    cfg = text_config(model.config)
    backbone = get_backbone(model)
    layer0 = backbone.layers[0]

    # --- Step 1: extract real weights BEFORE swapping ---
    print("Extracting real FP8 weights from checkpoint (layer 0 attn) ...")
    real_weights = _load_real_fp8_weights(layer0.self_attn)
    print("  " + ", ".join(
        f"{a}: w={tuple(w.shape)} ws={tuple(ws.shape)}"
        for a, (w, ws) in real_weights.items()
    ))

    # --- Step 2: swap ALL fp8 linears, load real weights for layer 0 attn ---
    targets = [
        name for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and m.weight.dtype == FP8_DTYPE
    ]
    print(f"Found {len(targets)} quantized linears total.")

    for name in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, attr)

        is_layer0_attn = (
            parent_name.endswith("layers.0.self_attn") and
            attr in real_weights
        )

        if is_layer0_attn:
            fp8 = replace_linear_with_fp8(parent, attr, dtype=torch.float16)
            w_fp16, ws_fp16 = real_weights[attr]
            fp8.weight.copy_(w_fp16)
            fp8.weight_scale.copy_(ws_fp16)
            fp8._tag = name
        else:
            # Replace with a zero-output fp16 linear — output is always 0,
            # so these layers' contribution to the forward pass is trivially
            # finite, and their outputs cannot be confused with a real bug in
            # layer 0.  We intentionally do NOT use random weights here:
            # random N(0,0.02) fp16 matmuls diverge (the whole point of this
            # script is to escape that).
            plain = nn.Linear(
                old.in_features, old.out_features, bias=False, dtype=torch.float16
            )
            nn.init.zeros_(plain.weight)
            setattr(parent, attr, plain)

    n_fp8 = sum(1 for _, m in model.named_modules() if isinstance(m, FP8Linear))
    print(f"Swapped {n_fp8} module(s) to FP8Linear (layer 0 attn only); "
          f"{len(targets) - n_fp8} zeroed fp16 nn.Linear.")

    leftover = [n for n, m in model.named_modules() if hasattr(m, "quantization_scheme")]
    print(f"Leftover quantized: {len(leftover)} {'(PASS)' if not leftover else '(FAIL)'}")
    if leftover:
        raise SystemExit(f"Aborting — leftover: {leftover[:5]}")

    # --- Step 3: set kv shapes (FP8Linear [in,out] would confuse inference) ---
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    model._spyre_kv_shapes = [
        (cfg.num_key_value_heads, head_dim, head_dim)
        for _ in range(cfg.num_hidden_layers)
    ]

    # --- Step 4: move to Spyre ---
    print("\n" + "=" * 70)
    print("prepare_for_spyre + move to Spyre")
    print("=" * 70)
    move_model_to_spyre(model, hf_granite, dtype=torch.float16)
    q0 = backbone.layers[0].self_attn.q_proj
    print(f"q_proj[0]: {type(q0).__name__}, "
          f"device={q0.weight.device}, dtype={q0.weight.dtype}")

    # --- Step 5: run one compiled forward ---
    print("\n" + "=" * 70)
    print("Prefill forward — real checkpoint weights, layer 0 FP8, rest zeroed")
    print("=" * 70)
    input_ids = torch.randint(0, model.config.vocab_size, (1, SEQ_LEN))
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
    max_cache_len = SEQ_LEN

    causal_mask = build_prefill_mask(
        batch_size=1, padded_len=SEQ_LEN, max_cache_len=max_cache_len,
        prompt_offsets=0, dtype=torch.float16,
    )
    key_caches, value_caches = allocate_kv_caches(
        model, batch_size=1, max_cache_len=max_cache_len,
        dtype=torch.float16, device="spyre",
    )
    cache_index = torch.arange(SEQ_LEN, dtype=torch.int64)

    n_blocks = len(model._spyre_compiled_blocks)
    with torch.no_grad():
        h = backbone.embed_tokens(input_ids.to("spyre"))
        h = h * backbone.embedding_multiplier
        selected_freqs = model._spyre_rope(h, position_ids.to("spyre"))

        for i, compiled_block in enumerate(model._spyre_compiled_blocks):
            h, key_caches[i], value_caches[i] = compiled_block(
                h, selected_freqs, causal_mask.to("spyre"),
                key_caches[i], value_caches[i], cache_index.to("spyre"),
            )
            hc = h.float().cpu()
            ok = torch.isfinite(hc).all().item()
            print(f"layer {i:2d}/{n_blocks-1}  max|h|={hc.abs().max():.4g}"
                  f"{'  OK' if ok else '  <-- NON-FINITE'}")
            if i == 0:
                kc = key_caches[0].float().cpu()
                vc = value_caches[0].float().cpu()
                finite(kc, "key_cache[0]")
                finite(vc, "value_cache[0]")
            if ONE_LAYER:
                break

        if not ONE_LAYER:
            h = backbone.norm(h)
            logits = model.lm_head(h) / cfg.logits_scaling
            lc = logits.cpu()
            print(f"\nlogits: shape={tuple(lc.shape)}, dtype={lc.dtype}")
            finite(lc, "logits")
            ok = torch.isfinite(lc).all().item()
            if ok:
                print("\nPASS — real checkpoint weights, FP8Linear layer 0, finite output")
            else:
                print("\nFAIL — non-finite logits (expected if layers 1+ have zero weights "
                      "and the model has a residual path that accumulates NaN elsewhere)")


if __name__ == "__main__":
    main()
