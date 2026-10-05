"""Compiled, multi-output probe of one layer's attention, staged.

An earlier version of this script tried true eager execution (no
torch.compile) to check finiteness after every stage safely. That failed
immediately: torch.ops.spyre.quantize_fp8_with_scale (and scaled_mm,
quantize_weight_fp8_with_scale, ...) have a literal `pass` body in their
Python definition (see torch_spyre/_inductor/customops.py) -- they are
schema/shape-registration stubs for Inductor's tracer, not real callables.
Their actual computation only exists via register_spyre_decompositions /
register_spyre_lowering INSIDE a compiled graph. Called eagerly, they return
None, and FP8Linear.forward's next line (`xq.reshape(...)`) blows up with
AttributeError: 'NoneType' object has no attribute 'reshape'. There is no
eager path for these ops at all -- that rules out "run it uncompiled" as a
diagnostic technique for anything touching FP8Linear.

The property we actually wanted from eager execution -- checking many
intermediate values without repeatedly crossing a compiled-graph boundary --
is still achievable: put the whole q_proj -> RoPE -> SDPA -> o_proj sequence
in ONE torch.compile'd function that returns every intermediate as part of
its output tuple, then check them all with a single .cpu() after that one
call returns. Same safety profile as everything that has already worked
(the --one-layer harness, check_fp8_pr_regression_spyre.py's cases): one
compile, one post-hoc inspection, no cross-call boundary to corrupt.

The one-layer harness (check_granite_fp8_spyre_forward.py --one-layer) already
showed key_cache[0]/value_cache[0] finite, h non-finite -- so k_proj/v_proj/
RoPE-on-k/the cache write are not the fault. This script returns q_proj's raw
output, q after RoPE, SDPA's output, and o_proj's output as well, to find
which of those three it is.

Run: python3 check_fp8_attn_eager_probe_spyre.py
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # checkpoint is already cached

import torch
from transformers import AutoModelForCausalLM

from hf_adapters.fp8_linear import replace_linear_with_fp8
from hf_adapters.hf_common import (
    allocate_kv_caches,
    apply_rope_matmul,
    build_prefill_mask,
    get_backbone,
    kv_cache_update,
    prepare_rope_and_heads,
    text_config,
)
import torch.nn.functional as F

PATH = "ibm-granite/granite-3.3-8b-instruct-FP8"
SEQ_LEN = 2  # matches the --one-layer / full-model runs


def finite(x, label):
    ok = torch.isfinite(x.float().cpu()).all().item()
    mx = x.float().abs().max().item()
    print(f"{label:<28} finite={ok!s:<5} max|.|={mx:.4g}")
    return ok


def main():
    torch.manual_seed(0)

    print(f"Loading {PATH} (layer 0 only needed) ...")
    model = AutoModelForCausalLM.from_pretrained(
        PATH, dtype=torch.float16, device_map="cpu"
    ).eval()

    layer0_attn = model.model.layers[0].self_attn

    # Swap layer 0's q/k/v/o_proj to FP8Linear, same random weights/scale as
    # every other check in this investigation -- so this run is directly
    # comparable to the compiled one-layer result.
    for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
        fp8 = replace_linear_with_fp8(layer0_attn, attr, dtype=torch.float16)
        fp8.weight.copy_(
            torch.randn(fp8.in_features, fp8.out_features, dtype=torch.float16) * 0.02
        )
        # Calibrated scale: max(|wq|) == 448 by construction. A flat 0.1 lets
        # wq ~ N(0, 0.2) and the K=4096 raw-matmul accumulation overflows fp16.
        fp8.weight_scale.copy_(
            (fp8.weight.abs().amax(dim=0) / 448.0).clamp(min=1e-3).to(torch.float16)
        )

    # RoPE + head-dim setup only -- not the full prepare_for_spyre (which
    # would wrap every layer in torch.compile via move_model_to_spyre). We
    # compile exactly one function ourselves below instead.
    prepare_rope_and_heads(model)

    q_proj, k_proj, v_proj, o_proj = (
        layer0_attn.q_proj, layer0_attn.k_proj, layer0_attn.v_proj, layer0_attn.o_proj
    )
    head_dim = layer0_attn.head_dim
    v_head_dim = getattr(layer0_attn, "v_head_dim", head_dim)
    scaling = layer0_attn.scaling

    for m in (q_proj, k_proj, v_proj, o_proj):
        m.to("spyre")  # nn.Module.to() is patched -- same layout path real
                        # weights would get, per the startup log.

    cfg = text_config(model.config)
    input_ids = torch.randint(0, model.config.vocab_size, (1, SEQ_LEN))
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0)

    backbone = get_backbone(model)
    with torch.no_grad():
        h = backbone.embed_tokens(input_ids)
        h = h * backbone.embedding_multiplier
        # RMSNorm is irrelevant to isolating q/SDPA/o_proj -- feed embed
        # output directly, just like case B in the isolated matmul test did
        # with its own synthetic "computed activation".
        selected_freqs = model._spyre_rope(h.to("spyre"), position_ids.to("spyre"))

    h = h.to("spyre")
    finite(h, "input h")

    causal_mask = build_prefill_mask(
        batch_size=1, padded_len=SEQ_LEN, max_cache_len=SEQ_LEN,
        prompt_offsets=0, dtype=torch.float16,
    ).to("spyre")

    # allocate_kv_caches (not raw torch.zeros): it pins the device layout
    # kv_cache_update's indirect scatter requires (cache-position dim at
    # device position 0). _spyre_kv_shapes is set for the same reason the
    # full-model scripts set it: kv_cache_shapes() would otherwise infer
    # num_kv_heads from FP8Linear's [in,out] weight shape and get it backwards.
    model._spyre_kv_shapes = [(cfg.num_key_value_heads, head_dim, head_dim)]
    key_caches, value_caches = allocate_kv_caches(
        model, batch_size=1, max_cache_len=SEQ_LEN, dtype=torch.float16,
        device="spyre",
    )
    key_cache, value_cache = key_caches[0], value_caches[0]
    cache_index = torch.arange(SEQ_LEN, dtype=torch.int64).to("spyre")

    # One compiled function, staged -- mirrors StandardGQAAttention.forward's
    # body exactly, but returns every intermediate instead of just the final
    # (out, key_cache, value_cache). All the FP8 custom ops only exist inside
    # this compiled region; every checkpoint below is read back AFTER this
    # single call returns, not between separate compiled calls.
    def staged_forward(h, selected_freqs, causal_mask, key_cache, value_cache, cache_index):
        bsz, seq_len, _ = h.shape

        q_raw = q_proj(h)
        k_raw = k_proj(h)
        v_raw = v_proj(h)

        q = q_raw.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
        k = k_raw.view(bsz, seq_len, -1, head_dim).transpose(1, 2)
        v = v_raw.view(bsz, seq_len, -1, v_head_dim).transpose(1, 2)

        q_rope = apply_rope_matmul(q, selected_freqs)
        k_rope = apply_rope_matmul(k, selected_freqs)

        key_cache, value_cache = kv_cache_update(
            k_rope, v, key_cache, value_cache, cache_index
        )

        attn_out = F.scaled_dot_product_attention(
            q_rope, key_cache, value_cache,
            attn_mask=causal_mask,
            dropout_p=0.0,
            scale=scaling,
            enable_gqa=True,
        )
        attn_out_flat = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        out = o_proj(attn_out_flat)

        # .clone() on everything that is ALSO consumed further downstream
        # (q_raw/k_raw/v_raw feed view+transpose+RoPE, attn_out feeds
        # transpose+reshape+o_proj). Returning the tensor itself here, still
        # feeding a further reshape elsewhere, forces Inductor to reconcile
        # two different layout requirements on the same buffer -- exactly the
        # original "Incompatible host_size and dim_order" crash from earlier
        # in this investigation, just re-triggered by this probe's own
        # structure rather than by anything in the real model. clone always
        # materializes a fresh, clean row-major buffer regardless of the
        # source's stick layout (see propagate_layouts.py's aten.clone.default
        # special-case) -- safe as a dedicated output, and it does not change
        # how the original (uncloned) tensor is scheduled for its real use.
        # key_cache/value_cache/out are already terminal, exactly like the
        # real block's own return values -- no clone needed for those.
        return (
            q_raw.clone(), k_raw.clone(), v_raw.clone(), q_rope.clone(),
            key_cache, value_cache, attn_out.clone(), out,
        )

    compiled = torch.compile(staged_forward, backend="inductor", dynamic=False)

    print("\n--- compiled, staged (one compile, all outputs checked after) ---")
    with torch.no_grad():
        q_raw, k_raw, v_raw, q_rope, key_cache, value_cache, attn_out, out = compiled(
            h, selected_freqs, causal_mask, key_cache, value_cache, cache_index
        )

    finite(q_raw, "q_proj raw")
    finite(k_raw, "k_proj raw")
    finite(v_raw, "v_proj raw")
    finite(q_rope, "q after RoPE")
    finite(key_cache, "key_cache after write")
    finite(value_cache, "value_cache after write")
    finite(attn_out, "SDPA output")
    finite(out, "o_proj output (final)")

    print("\nDone. First 'finite=False' line above is where this diverges "
          "from the earlier one-layer result (key_cache/value_cache clean, "
          "h non-finite).")


if __name__ == "__main__":
    main()
