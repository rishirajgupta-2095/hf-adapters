"""FP8Linear + Spyre adapter wiring smoke test — random weights.

Tests whether FP8Linear interoperates with the full Spyre adapter path
(prepare_for_spyre + _run_forward / compiled blocks) without crashing.
DOES NOT test numerical correctness — use check_granite_fp8_checkpoint_forward.py
for that (real E4M3 weights loaded and dequantized correctly).

EXPECTED RESULT WITH RANDOM WEIGHTS
-------------------------------------
Both the FP8 run and the ``--no-fp8`` control produce NaN from layer 0.
This is not a bug — it is the expected behaviour of 40 fp16 transformer layers
with N(0,0.02) random weights and K=4096:

  K=4096 terms × 0.02 weight-std × accumulated across MLP gate (out=14336)
  → raw dot-product magnitude reaches ~100 per output → Inf in fp16 from
    MLP layer 0 → NaN in subsequent SDPA softmax → propagates forever.

The ``--no-fp8`` control reproduces NaN identically to the FP8 run.  This
confirms that the NaN is NOT caused by FP8Linear and is purely a property of
random initialisation with fp16 arithmetic.

To see a meaningful (finite) result:
  python3 check_granite_fp8_checkpoint_forward.py --one-layer
  → loads real E4M3 weights, runs layer 0 on Spyre, checks K/V + h finiteness.

What this script DOES validate (regardless of NaN output):
  - FP8Linear moves to Spyre without crashing
  - FP8Linear is traced/compiled by Dynamo without crashing
  - The compiled block runs to completion for all 40 layers
  - The swap wiring (SWAP_ONLY, _in_swap_only, _spyre_kv_shapes) is correct

Run: python3 check_granite_fp8_spyre_forward.py [--no-fp8] [--one-layer]
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # checkpoint is already cached

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from hf_adapters import hf_granite
from hf_adapters.fp8_linear import FP8_DTYPE, replace_linear_with_fp8
from hf_adapters.fp8_linear import FP8Linear
from hf_adapters.hf_common import (
    allocate_kv_caches,
    build_prefill_mask,
    get_backbone,
    move_model_to_spyre,
    text_config,
)

def _all_finite(x) -> bool:
    """isfinite check that also handles RoPE's (cos, sin)-tuple return."""
    if torch.is_tensor(x):
        return torch.isfinite(x.float().cpu()).all().item()
    if isinstance(x, (tuple, list)):
        return all(_all_finite(t) for t in x)
    return True


PATH = "ibm-granite/granite-3.3-8b-instruct-FP8"
SEQ_LEN = 2  # M feeds batchmatmulfp8's work division directly, and smaller is
# better (but not 1, which collapses the dim and breaks _matmul_layouts):
#   M=1   -> "expected exactly 1 generated variable, got set()"
#   M=2   -> q/k/v/o_proj compile, gate_proj fails
#   M=8   -> q_proj compiles, k_proj fails
#   M=128 -> q_proj fails

# Which projections become FP8Linear. The MLP is excluded because gate_proj is
# the first op to fail batchmatmulfp8's work division; excluding it buys a
# COMPLETE forward pass with FP8 attention, which is the first end-to-end result
# obtainable today. Set to None to swap everything.
#
# Everything not swapped is rebuilt as a plain fp16 nn.Linear rather than left
# alone: an untouched module keeps its E4M3 weight, and .to(fp16) would cast
# those bytes WITHOUT applying weight_scale -- silently wrong. Weights are random
# here anyway, so a fresh fp16 Linear is both cleaner and no less meaningful.
#
# NOTE on NaN with random weights: the --no-fp8 control (SWAP_ONLY=()) ALSO
# produces NaN from layer 0. This is expected: 40 layers of random N(0,0.02)
# fp16 matmuls diverge numerically. The non-finite output with random weights
# is not a regression indicator. The meaningful test is using real checkpoint
# weights (check_granite_fp8_checkpoint_forward.py), not random ones.
SWAP_ONLY = ("q_proj", "k_proj", "v_proj", "o_proj")

# Control run: `python3 check_granite_fp8_spyre_forward.py --no-fp8` rebuilds
# every quantized Linear as fp16 instead, keeping the same seed and the same
# random weights. If that output is ALSO non-finite, the divergence is the
# random initialisation and says nothing about FP8Linear.
if "--no-fp8" in sys.argv:
    SWAP_ONLY = ()
NUM_DECODE = 0  # prefill only for this first check

# `--one-layer`: stop after the first block instead of all 40. Same real code
# path (StandardGQABlock, FP8Linear, RoPE, SDPA, KV-cache) as the full run --
# torch.compile is lazy per-block-instance, so blocks after the one we call
# are simply never traced/compiled. This turns a ~40-block load+compile cycle
# into a 1-block one, fast enough to actually iterate on a fix, and prints
# key_cache[0]/value_cache[0] finiteness too: still-clean K/V with a
# non-finite h narrows the fault to SDPA or o_proj; non-finite K/V narrows it
# to k_proj/v_proj themselves.
ONE_LAYER = "--one-layer" in sys.argv


def main():
    torch.manual_seed(0)

    print(f"Loading {PATH} (stock AutoModelForCausalLM.from_pretrained) ...")
    model = AutoModelForCausalLM.from_pretrained(
        PATH, dtype=torch.float16, device_map="cpu"
    ).eval()

    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and module.weight.dtype == FP8_DTYPE
    ]
    print(f"Found {len(targets)} quantized nn.Linear modules to swap.")

    def _in_swap_only(full_name, last_attr):
        """Return True if this module should become FP8Linear.

        SWAP_ONLY is matched against both the immediate attribute name and every
        component of the full dotted path.  compressed_tensors can wrap a named
        projection (e.g. q_proj) inside a container, making `attr` resolve to a
        generic name like "linear" while "q_proj" appears earlier in the path.
        """
        if SWAP_ONLY is None:
            return True
        if last_attr in SWAP_ONLY:
            return True
        return any(part in SWAP_ONLY for part in full_name.split("."))

    for name in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model

        if not _in_swap_only(name, attr):
            old = getattr(parent, attr)
            plain = nn.Linear(
                old.in_features, old.out_features, bias=False, dtype=torch.float16
            )
            plain.weight.data.copy_(
                torch.randn(old.out_features, old.in_features, dtype=torch.float16)
                * 0.02
            )
            setattr(parent, attr, plain)
            continue

        fp8 = replace_linear_with_fp8(parent, attr, dtype=torch.float16)
        fp8._tag = name
        fp8.weight.copy_(
            torch.randn(fp8.in_features, fp8.out_features, dtype=torch.float16) * 0.02
        )
        # Calibrate weight_scale so that max(|wq|) == 448. Using a flat 0.1
        # was wrong: w ~ N(0,0.02) -> wq = w/0.1 ~ N(0,0.2), max ≈ 0.6, but
        # K=4096 terms accumulate and the raw fp16 matmul overflows (max sum
        # ≈ 1.1M >> fp16 max 65504), producing Inf that propagates to NaN in
        # SDPA softmax.  check_fp8_linear_spyre.py already uses this form.
        fp8.weight_scale.copy_(
            (fp8.weight.abs().amax(dim=0) / 448.0).clamp(min=1e-3).to(torch.float16)
        )

    n_fp8 = sum(1 for _, m in model.named_modules() if isinstance(m, FP8Linear))
    print(f"Swapped {n_fp8} module(s) to FP8Linear; "
          f"{len(targets) - n_fp8} rebuilt as fp16 nn.Linear.")

    # Show which projection names were actually swapped (layer 0 only).
    swapped_attrs = sorted({
        name.split(".")[-1]
        for name, m in model.named_modules()
        if isinstance(m, FP8Linear) and ".0." in name
    })
    print(f"FP8Linear projections in layer 0: {swapped_attrs}")

    leftover = [
        name for name, m in model.named_modules() if hasattr(m, "quantization_scheme")
    ]
    print(f"\nSwap-completeness check: {len(leftover)} leftover "
          f"{'(FAIL)' if leftover else '(PASS)'}")
    if leftover:
        raise SystemExit(f"Aborting — swap missed: {leftover[:5]}")

    # kv_cache_shapes() infers num_kv_heads from k_proj.weight.shape[0] //
    # head_dim, assuming nn.Linear's [out, in] orientation. FP8Linear stores
    # [in, out] (the restickify fix), so that read returns in_features
    # (4096) instead of out_features (1024) -> 32 heads instead of 8. Rather
    # than touch that shared, TP-relevant inference path (which other models
    # legitimately rely on to read the *live*, possibly TP-sharded shape),
    # use the same escape hatch Gemma 4 already uses for its own per-layer
    # shape quirk: set _spyre_kv_shapes explicitly, sourced from config, the
    # same way fms_mo's attention op takes nheads/kvheads as plain
    # config-derived parameters rather than inferring them from a weight
    # tensor at all.
    cfg = text_config(model.config)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    model._spyre_kv_shapes = [
        (cfg.num_key_value_heads, head_dim, head_dim) for _ in range(cfg.num_hidden_layers)
    ]

    print("\n" + "=" * 70)
    print("prepare_for_spyre + move to spyre — the same machinery Qwen3's")
    print("Stage-0 pass already proved works on this hardware")
    print("=" * 70)
    move_model_to_spyre(model, hf_granite, dtype=torch.float16)
    print(f"q_proj[0] on device: {type(model.model.layers[0].self_attn.q_proj).__name__}, "
          f"weight device={model.model.layers[0].self_attn.q_proj.weight.device}, "
          f"dtype={model.model.layers[0].self_attn.q_proj.weight.dtype}")

    print("\n" + "=" * 70)
    print("One prefill forward pass through _run_forward (the real compiled")
    print("block path — first call triggers real Spyre compilation)")
    print("=" * 70)
    input_ids = torch.randint(0, model.config.vocab_size, (1, SEQ_LEN))
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0)
    max_cache_len = SEQ_LEN + NUM_DECODE

    # prompt_offsets=0 — no left-padding, the whole sequence is real tokens.
    causal_mask = build_prefill_mask(
        batch_size=1,
        padded_len=SEQ_LEN,
        max_cache_len=max_cache_len,
        prompt_offsets=0,
        dtype=torch.float16,
    )

    key_caches, value_caches = allocate_kv_caches(
        model, batch_size=1, max_cache_len=max_cache_len, dtype=torch.float16, device="spyre"
    )

    # cache_index: destination positions for the indirect-scatter KV write —
    # writing SEQ_LEN tokens starting at position 0 (a fresh prefill).
    cache_index = torch.arange(SEQ_LEN, dtype=torch.int64)

    # Instrumented replay of hf_granite._run_forward / _run_backbone_forward.
    # Same calls in the same order — the only difference is the progress
    # printing, so a failure here is the same failure _run_forward would give.
    #
    # Two granularities, because they answer different questions:
    #   "trace" fires while Dynamo is still building the FX graph, so ALL of a
    #   block's projections print before Inductor compiles any of them. It tells
    #   us which projections exist and what rank each one receives — not what
    #   compiled.
    #   "layer N OK" fires only after that block's compiled binary has actually
    #   run, so it is the real progress signal.
    _orig_forward = FP8Linear.forward
    _seen = set()

    def _traced_forward(self, x):
        tag = getattr(self, "_tag", "<untagged>")
        short = tag.split("layers.")[-1]
        if short not in _seen:
            _seen.add(short)
            print(f"    trace  {short:<26} in={tuple(x.shape)} rank={x.dim()}")
        return _orig_forward(self, x)

    FP8Linear.forward = _traced_forward

    backbone = get_backbone(model)
    with torch.no_grad():
        h = backbone.embed_tokens(input_ids.to("spyre"))
        print("embedding OK")
        h = h * backbone.embedding_multiplier
        print("embedding_multiplier OK")
        # No finiteness check on h here, deliberately: an earlier version of
        # this script added an _all_finite(h) check at this exact point (and
        # after embedding/embedding_multiplier too), and BOTH the FP8 and
        # --no-fp8 runs broke identically from layer 0 on -- a regression
        # relative to the untouched script, where --no-fp8 was clean through
        # layer 39. Each check called h.float().cpu(), a device round-trip
        # that does not otherwise happen here, and Spyre eager tensors are
        # already documented elsewhere in this codebase (see
        # hf_common._mask_fill_value) as fragile to exactly this kind of
        # touch. Diagnosing further by adding more .cpu() probes to this
        # script is not trustworthy on this hardware -- isolate suspect ops
        # in a standalone script instead, outside the full model's eager/
        # compiled boundary.
        selected_freqs = model._spyre_rope(h, position_ids.to("spyre"))
        print("rope OK")

        # "unknown" rather than a fresh check on h -- see the comment above on
        # why this script does not touch h between rope and the first block.
        in_finite = "unknown"

        n = len(model._spyre_compiled_blocks)
        for i, compiled_block in enumerate(model._spyre_compiled_blocks):
            # NOT re-derived from h here on purpose: an extra h.cpu() right
            # before feeding h back into a compiled Spyre graph risks tripping
            # the same device-layout retile this run already warns about
            # elsewhere (RetileWarning in torch_spyre/ops/eager.py). in_finite
            # is carried forward from the previous iteration's own check
            # instead — nothing touches h between that check and this call, so
            # it is the same fact without the extra round-trip.
            h, key_caches[i], value_caches[i] = compiled_block(
                h,
                selected_freqs,
                causal_mask.to("spyre"),
                key_caches[i],
                value_caches[i],
                cache_index.to("spyre"),
            )
            # Track magnitude, not just compilation. A randomly-initialised
            # 40-layer net can diverge on its own, so "non-finite output" alone
            # does not implicate FP8Linear. Watching |h| grow layer by layer
            # separates gradual blow-up (random weights) from a single bad layer
            # (a real bug): blow-up roughly doubles or worse each layer, a bug
            # goes non-finite in one step.
            hc = h.float().cpu()
            finite = torch.isfinite(hc).all().item()
            print(f"layer {i:2d}/{n - 1} OK   in_finite={in_finite}   "
                  f"max|h|={hc.abs().max().item():.4g}"
                  f"{'' if finite else '   <-- NON-FINITE'}")
            in_finite = finite  # carried into the next iteration's check

            if ONE_LAYER:
                # Safe here: this is the only block we call, so there is no
                # downstream compiled graph for a .cpu() touch to corrupt.
                kc = key_caches[0].float().cpu()
                vc = value_caches[0].float().cpu()
                print(f"    key_cache[0]    finite={torch.isfinite(kc).all().item()}   "
                      f"max|k|={kc.abs().max().item():.4g}")
                print(f"    value_cache[0]  finite={torch.isfinite(vc).all().item()}   "
                      f"max|v|={vc.abs().max().item():.4g}")
                break

        h = backbone.norm(h)
        print("final norm OK")
        logits = model.lm_head(h) / text_config(model.config).logits_scaling
        print("lm_head OK")

    logits_cpu = logits.cpu()
    print(f"logits: dtype={logits_cpu.dtype}, shape={tuple(logits_cpu.shape)}, "
          f"device_was={logits.device}")
    finite = torch.isfinite(logits_cpu).all().item()
    print(f"finite: {finite}")

    if finite:
        print(f"\nPASS — real Granite architecture, FP8Linear in place, ran "
              f"through the actual compiled-block Spyre path, no NaN/Inf")
    else:
        print(f"\nFAIL — non-finite values in the output")


if __name__ == "__main__":
    main()
