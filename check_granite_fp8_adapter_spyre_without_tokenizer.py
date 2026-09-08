"""First forward pass with REAL checkpoint weights, through the adapter path.

Everything before this ran on hardware with RANDOM weights and a test script
doing its own module swap. This runs the actual production path instead:

    move_model_to_spyre(model, hf_granite, ...)
      -> hf_granite.prepare_for_spyre(model)
           -> _prepare_fp8_if_quantized(model)      # detects E4M3, swaps, pins KV shapes
           -> prepare_standard_gqa_blocks(...)      # compiles blocks over FP8Linear
      -> _move_to_spyre_with_layout(model, dtype)

So there is no swap logic here, and no ``_spyre_kv_shapes`` override -- if this
script needs either, the adapter is incomplete. That shrinkage is the point.

Weights are the checkpoint's genuine E4M3 values, dequantized on CPU by
``swap_linears_to_fp8`` and re-quantized inside the compiled graph. The load
half is verified separately and exactly by ``check_fp8_checkpoint_load_cpu.py``.

REQUIRES torch-spyre branch ``fp8-fix-4179`` (or any tree containing #4246,
``ed4aa21a``). On ``main`` the SDPA tile-advance bug silently corrupts
head-tiled KV reads and this returns NaN with no error.

Validated envelope: prefill only, SEQ_LEN 2-4, o_proj excluded from FP8.
Decode (M=1) and realistic sequence lengths are untested.

Run: python3 check_granite_fp8_adapter_spyre.py
"""

import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # checkpoint is already cached

import torch
from transformers import AutoModelForCausalLM

from hf_adapters import hf_granite
from hf_adapters.fp8_linear import FP8Linear
from hf_adapters.hf_common import (
    allocate_kv_caches,
    build_prefill_mask,
    get_backbone,
    move_model_to_spyre,
    text_config,
)

PATH = "ibm-granite/granite-3.3-8b-instruct-FP8"
SEQ_LEN = 4  # within the validated envelope; M feeds batchmatmulfp8 work division
NUM_DECODE = 0  # prefill only -- decode (M=1) is a known-untested risk

# `--random-weights`: run the ADAPTER path but fill weights with randn instead
# of the checkpoint's dequantized values.
#
# Why this exists: check_granite_fp8_spyre_forward.py passed (finite, 40 layers,
# same six projections, same SEQ_LEN=4) using random weights and its OWN swap,
# done BEFORE move_model_to_spyre. This script uses real weights and the
# adapter's swap, done INSIDE prepare_for_spyre -- and hits
# "ReStickifyOpHBM on DataFormats.SEN143_FP8". Two variables changed at once.
# This flag holds the weights random and varies only the swap path, so a
# failure here indicts the path and exonerates the weights (and vice versa).
#
# Layout/codegen decisions are shape- and graph-driven, not value-driven, so on
# paper weight VALUES should not matter -- which is exactly why it is worth
# confirming rather than assuming. weight_scale is left as the checkpoint's real
# per-channel scale: it is not a compile-time input either, and holding it fixed
# keeps the number of changed variables at one.
RANDOM_WEIGHTS = "--random-weights" in sys.argv

# `--exclude a,b,c`: override which projections are kept out of the FP8 swap,
# without editing fp8_linear.py. Use to find the FP8 set that actually compiles
# as a single fused block.
#
# Needed because the six-projection result recorded earlier was an
# INSTRUMENTATION ARTIFACT: check_granite_fp8_spyre_forward.py monkeypatches
# FP8Linear.forward with a wrapper containing a print, Dynamo graph-breaks on it,
# and the resulting smaller subgraphs avoid a restickify that a single-graph
# compile produces. Comment out that one line and it fails with
# "ReStickifyOpHBM on DataFormats.SEN143_FP8" on down_proj's FP8 input
# ([1,4,12800]) -- identical to what this script hits. So the real working set
# is unknown, and this flag is how to establish it.
#
#   --exclude o_proj,down_proj    # next cut to try: drop the projection the
#                                 # restickify actually lands on
#   --exclude o_proj              # current default, known to FAIL
def _parse_exclude(argv):
    for i, a in enumerate(argv):
        if a == "--exclude" and i + 1 < len(argv):
            return tuple(p.strip() for p in argv[i + 1].split(",") if p.strip())
        if a.startswith("--exclude="):
            return tuple(p.strip() for p in a.split("=", 1)[1].split(",") if p.strip())
    return None


EXCLUDE_OVERRIDE = _parse_exclude(sys.argv)


def _parse_opt(argv, name):
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


# `--prompt TEXT`: tokenize real text instead of using random token ids.
#
# Needed because comparing top-5 predictions on RANDOM token ids is close to
# meaningless: on nonsense input the distribution is nearly flat (an observed
# fp16 run had top-1 4.688 vs top-2 4.664 -- a 0.024 gap), so any perturbation
# reshuffles the ranking and FP8-vs-fp16 ordering differences say nothing.
#
# NOTE this leaves the validated envelope: a real prompt is longer than
# SEQ_LEN=4, and larger S is explicitly one of the deferred risks (bug (A), the
# SDPA tile-advance fix, concerned head-tiled KV reads that only engage at
# larger S -- #4246's own repro cites S=128). That is deliberate: this doubles
# as the realistic-S probe. If it fails here but passes at SEQ_LEN=4, that IS
# the S risk showing up, not an accuracy problem.
PROMPT = _parse_opt(sys.argv, "--prompt")

# `--save PATH`: write the logits tensor for offline comparison.
# `--compare A.pt B.pt`: diff two saved runs numerically and exit.
SAVE_PATH = _parse_opt(sys.argv, "--save")


def _patch_exclude(exclude):
    """Override DEFAULT_FP8_EXCLUDE, which swap_linears_to_fp8 resolves at call time."""
    import hf_adapters.fp8_linear as _fp8

    _fp8.DEFAULT_FP8_EXCLUDE = exclude


def _patch_dequant_to_random():
    """Make the adapter's swap produce random weights, path otherwise unchanged."""
    import hf_adapters.fp8_linear as _fp8

    def _random_dq(linear):
        out_f, in_f = linear.weight.shape
        return torch.randn(out_f, in_f, dtype=torch.float16) * 0.02

    _fp8._dequantize_checkpoint_weight = _random_dq


def compare(path_a, path_b):
    """Numerically diff two saved logit tensors.

    Top-k ID agreement alone is a weak signal when the distribution is flat, so
    report distribution-level metrics too: cosine similarity and KL divergence
    say whether the two models behave the same, independent of whether a
    near-tie happened to reorder.
    """
    a = torch.load(path_a).float()
    b = torch.load(path_b).float()
    if a.shape != b.shape:
        print(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
        return

    last_a, last_b = a[0, -1], b[0, -1]
    diff = (last_a - last_b).abs()
    cos = torch.nn.functional.cosine_similarity(last_a, last_b, dim=0).item()
    pa = torch.softmax(last_a, dim=-1)
    pb = torch.softmax(last_b, dim=-1)
    kl = torch.sum(pa * (torch.log(pa + 1e-12) - torch.log(pb + 1e-12))).item()
    ta, tb = last_a.topk(5).indices.tolist(), last_b.topk(5).indices.tolist()

    print(f"A: {path_a}")
    print(f"B: {path_b}\n")
    print(f"max |A-B|        : {diff.max().item():.4f}")
    print(f"mean |A-B|       : {diff.mean().item():.4f}")
    print(f"cosine similarity: {cos:.6f}")
    print(f"KL(A||B)         : {kl:.6f}")
    print(f"top-1 agree      : {ta[0] == tb[0]}   (A={ta[0]}, B={tb[0]})")
    print(f"top-5 overlap    : {len(set(ta) & set(tb))}/5")
    print(f"  A top-5: {ta}\n  B top-5: {tb}")
    print(
        "\nRule of thumb: cosine > 0.99 and small KL means FP8 is tracking "
        "fp16 and any top-k reordering is just near-ties being reshuffled. "
        "Low cosine or large KL means real accuracy loss."
    )


def main():
    if "--compare" in sys.argv:
        i = sys.argv.index("--compare")
        compare(sys.argv[i + 1], sys.argv[i + 2])
        return

    torch.manual_seed(0)

    if RANDOM_WEIGHTS:
        _patch_dequant_to_random()
        print("--random-weights: adapter path, randn(0, 0.02) weights")
    if EXCLUDE_OVERRIDE is not None:
        _patch_exclude(EXCLUDE_OVERRIDE)
        print(f"--exclude: {EXCLUDE_OVERRIDE}")
    print()

    print(f"Loading {PATH} ...")
    model = AutoModelForCausalLM.from_pretrained(
        PATH, dtype=torch.float16, device_map="cpu"
    ).eval()

    print("\nmove_model_to_spyre -> prepare_for_spyre -> FP8 swap (adapter path)")
    move_model_to_spyre(model, hf_granite, dtype=torch.float16)

    # Report what the adapter actually did, rather than doing it here.
    layer0 = get_backbone(model).layers[0]
    fp8_names = sorted(
        name.split(".")[-1]
        for name, m in layer0.named_modules()
        if isinstance(m, FP8Linear)
    )
    n_fp8 = sum(1 for _, m in model.named_modules() if isinstance(m, FP8Linear))
    print(f"FP8Linear modules: {n_fp8}")
    print(f"FP8Linear projections in layer 0: {fp8_names}")
    print(f"q_proj[0]: {type(layer0.self_attn.q_proj).__name__}, "
          f"o_proj[0]: {type(layer0.self_attn.o_proj).__name__}")

    if PROMPT is not None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(PATH)
        input_ids = tok(PROMPT, return_tensors="pt").input_ids
        seq_len = input_ids.shape[1]
        print(f"prompt: {PROMPT!r} -> {seq_len} tokens {input_ids[0].tolist()}")
        if seq_len != SEQ_LEN:
            print(
                f"NOTE: S={seq_len} leaves the validated envelope (SEQ_LEN={SEQ_LEN}). "
                "Larger S is a known-untested risk -- a failure here may be that, "
                "not an FP8 accuracy problem."
            )
    else:
        seq_len = SEQ_LEN
        # Re-seed HERE, not just at the top of main(): swap_linears_to_fp8
        # constructs an nn.Linear for every EXCLUDED module, and
        # nn.Linear.__init__ consumes RNG for its default init before .copy_()
        # overwrites it. A run excluding 40 modules and one excluding 280
        # therefore reach this line with the generator in different states, and
        # would get DIFFERENT random token ids -- making an FP8-vs-fp16 logit
        # comparison meaningless (different inputs, not different precision).
        torch.manual_seed(1234)
        input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len))
        print(f"input ids (seeded, identical across runs): {input_ids[0].tolist()}")

    position_ids = torch.arange(seq_len).unsqueeze(0)
    max_cache_len = seq_len + NUM_DECODE

    causal_mask = build_prefill_mask(
        batch_size=1,
        padded_len=seq_len,
        max_cache_len=max_cache_len,
        prompt_offsets=0,
        dtype=torch.float16,
    )
    key_caches, value_caches = allocate_kv_caches(
        model,
        batch_size=1,
        max_cache_len=max_cache_len,
        dtype=torch.float16,
        device="spyre",
    )
    cache_index = torch.arange(seq_len, dtype=torch.int64)

    print("\nOne prefill forward through hf_granite._run_forward")
    with torch.no_grad():
        logits = hf_granite._run_forward(
            model,
            input_ids.to("spyre"),
            position_ids.to("spyre"),
            causal_mask.to("spyre"),
            key_caches,
            value_caches,
            cache_index.to("spyre"),
        )

    lg = logits.float().cpu()
    finite = torch.isfinite(lg).all().item()
    print(f"logits: shape={tuple(lg.shape)} dtype={logits.dtype}")
    print(f"finite: {finite}   max|logits|={lg.abs().max().item():.4g}")

    if SAVE_PATH:
        torch.save(lg, SAVE_PATH)
        print(f"saved logits -> {SAVE_PATH}")

    if not finite:
        print(
            "\nFAIL — non-finite. Check that torch-spyre has #4246 "
            "(ed4aa21a); on main the SDPA bug produces exactly this."
        )
        return

    if RANDOM_WEIGHTS:
        print(
            "\nPASS (compile/run only) — adapter path, RANDOM weights, "
            "all 40 layers, finite logits."
        )
        print(
            "This says nothing about numerics: the point of --random-weights is "
            "to test whether the adapter's swap PATH compiles, holding weight "
            "values constant against the older script that passed."
        )
        return

    # Only meaningful with real weights; random ones make argmax noise.
    top = lg[0, -1].topk(5)
    print(f"\ntop-5 next-token ids at last position: {top.indices.tolist()}")
    print(f"top-5 logits: {[round(v, 3) for v in top.values.tolist()]}")
    print(
        "\nPASS — real checkpoint weights, FP8 via the adapter path, "
        "all 40 layers, finite logits."
    )
    print(
        "NOT yet verified: accuracy vs an fp16 reference. Finite != correct; "
        "a logit comparison against the unquantized model is the next check."
    )


if __name__ == "__main__":
    main()

