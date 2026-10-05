"""Staged probe of one full StandardGQABlock — confirms FP8Linear v3 fix.

History:
  - one-layer harness showed key_cache/value_cache finite, h NaN → NaN
    enters somewhere after the KV write.
  - Case A (four parallel FP8 matmuls from layernorm output) passed.
  - Case B (full attention chain including o_proj on SDPA output) crashed:
      RuntimeError: Incompatible host_size and dim_order
    The SDPA output goes through transpose+reshape (rank 4→3) before o_proj.
    FP8Linear v2 computed x_scale from that rank-3 computed buffer, then used
    it in a multi-arg pointwise against the rank-2 scaled_mm output, tripping
    the layout-propagation rank-mismatch bug in propagate_layouts.py.
  - FP8Linear v3 fixes this by flattening x to 2-D BEFORE computing x_scale,
    keeping all FP8 ops and the rescale at rank 2 throughout.

This script runs four staged variants to confirm v3 is clean end-to-end.
Each variant returns its intermediate as PART OF THE SAME compiled graph
(no cross-call .float().cpu() between stages):

  A  just the four FP8 matmuls (parallel, same input) — baseline
  B  FP8 attention chain: q_proj → RoPE → K/V write → SDPA → o_proj
  C  B + residual1 (residual + attn_out * res_mul)
  D  C + post_attn_layernorm + MLP
  E  D + residual2 (= full block)

The first case that returns NaN isolates the component.  All five share
the same weights and input so they are directly comparable.

Run: python3 check_fp8_block_staged_spyre.py
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
import torch.nn as nn

from hf_adapters.fp8_linear import replace_linear_with_fp8, FP8_DTYPE
from hf_adapters.hf_common import (
    allocate_kv_caches,
    apply_rope_matmul,
    build_prefill_mask,
    get_backbone,
    kv_cache_update,
    prepare_rope_and_heads,
    text_config,
    patch_rmsnorm,
)

PATH = "ibm-granite/granite-3.3-8b-instruct-FP8"
SEQ_LEN = 2


def finite(x, label):
    ok = torch.isfinite(x.float().cpu()).all().item()
    mx = x.float().abs().max().item()
    print(f"  {label:<34} finite={str(ok):<5} max|.|={mx:.4g}")
    return ok


def _w_calibrated(in_f, out_f):
    """Random fp16 weight [in, out] + per-column calibrated scale."""
    w = torch.randn(in_f, out_f, dtype=torch.float16) * 0.02
    ws = (w.abs().amax(dim=0) / 448.0).clamp(min=1e-3).to(torch.float16)
    return w, ws


def main():
    torch.manual_seed(0)

    print(f"Loading {PATH} (one layer needed) ...")
    model = AutoModelForCausalLM.from_pretrained(
        PATH, dtype=torch.float16, device_map="cpu"
    ).eval()
    from transformers.models.granite.modeling_granite import GraniteRMSNorm
    patch_rmsnorm(GraniteRMSNorm)

    layer0 = get_backbone(model).layers[0]
    attn = layer0.self_attn

    # --- swap q/k/v/o_proj to FP8Linear with calibrated scales ---
    for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
        fp8 = replace_linear_with_fp8(attn, attr, dtype=torch.float16)
        w, ws = _w_calibrated(fp8.in_features, fp8.out_features)
        fp8.weight.copy_(w)
        fp8.weight_scale.copy_(ws)

    # --- rebuild MLP projections as fresh random fp16 ---
    targets_mlp = [
        name for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and m.weight.dtype == FP8_DTYPE
    ]
    for name in targets_mlp:
        parent_name, _, a = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old = getattr(parent, a)
        plain = nn.Linear(old.in_features, old.out_features, bias=False, dtype=torch.float16)
        plain.weight.data.copy_(
            torch.randn(old.out_features, old.in_features, dtype=torch.float16) * 0.02
        )
        setattr(parent, a, plain)

    prepare_rope_and_heads(model)

    cfg = text_config(model.config)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    # Set kv shapes to avoid the [in,out] FP8Linear shape being misread.
    model._spyre_kv_shapes = [
        (cfg.num_key_value_heads, head_dim, head_dim)
        for _ in range(cfg.num_hidden_layers)
    ]

    q_proj = attn.q_proj
    k_proj = attn.k_proj
    v_proj = attn.v_proj
    o_proj = attn.o_proj
    scaling = attn.scaling
    v_head_dim = getattr(attn, "v_head_dim", head_dim)
    input_ln = layer0.input_layernorm
    post_attn_ln = layer0.post_attention_layernorm
    mlp = layer0.mlp
    res_mul = layer0.residual_multiplier  # scalar or 0-dim tensor

    for m in (q_proj, k_proj, v_proj, o_proj):
        m.to("spyre")
    input_ln.to("spyre")
    post_attn_ln.to("spyre")
    mlp.to("spyre")

    backbone = get_backbone(model)
    input_ids = torch.randint(0, model.config.vocab_size, (1, SEQ_LEN))
    position_ids = torch.arange(SEQ_LEN).unsqueeze(0)

    with torch.no_grad():
        h_cpu = backbone.embed_tokens(input_ids)
        h_cpu = h_cpu * backbone.embedding_multiplier
        h = h_cpu.to("spyre")
        selected_freqs = model._spyre_rope(h, position_ids.to("spyre"))

    causal_mask = build_prefill_mask(
        batch_size=1, padded_len=SEQ_LEN, max_cache_len=SEQ_LEN,
        prompt_offsets=0, dtype=torch.float16,
    ).to("spyre")

    key_caches, value_caches = allocate_kv_caches(
        model, batch_size=1, max_cache_len=SEQ_LEN, dtype=torch.float16, device="spyre",
    )
    key_cache0, value_cache0 = key_caches[0], value_caches[0]
    cache_index = torch.arange(SEQ_LEN, dtype=torch.int64).to("spyre")

    res_mul_spyre = torch.tensor(float(res_mul), dtype=torch.float16).to("spyre")

    # -----------------------------------------------------------------------
    # Case A: four parallel FP8 matmuls only (case_c equivalent)
    # Input to all four is layernorm(h) — as in the real block.
    # -----------------------------------------------------------------------
    def case_a_fn(h, sf, mask, kc, vc, ci):
        bsz, seq_len, _ = h.shape
        h_ln = input_ln(h)
        q_raw = q_proj(h_ln)
        k_raw = k_proj(h_ln)
        v_raw = v_proj(h_ln)
        o_raw = o_proj(h_ln)  # feeding h_ln directly, not SDPA output
        return q_raw.clone(), k_raw.clone(), v_raw.clone(), o_raw.clone()

    # -----------------------------------------------------------------------
    # Case B: full FP8 attention chain q→RoPE→SDPA→o_proj (no residual/MLP)
    # -----------------------------------------------------------------------
    def case_b_fn(h, sf, mask, kc, vc, ci):
        bsz, seq_len, _ = h.shape
        h_ln = input_ln(h)
        q = q_proj(h_ln).view(bsz, seq_len, -1, head_dim).transpose(1, 2)
        k = k_proj(h_ln).view(bsz, seq_len, -1, head_dim).transpose(1, 2)
        v = v_proj(h_ln).view(bsz, seq_len, -1, v_head_dim).transpose(1, 2)
        q = apply_rope_matmul(q, sf)
        k = apply_rope_matmul(k, sf)
        kc, vc = kv_cache_update(k, v, kc, vc, ci)
        attn_out = F.scaled_dot_product_attention(
            q, kc, vc, attn_mask=mask, dropout_p=0.0, scale=scaling, enable_gqa=True,
        )
        attn_out_flat = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        out = o_proj(attn_out_flat)
        return out.clone(), kc, vc

    # -----------------------------------------------------------------------
    # Case C: B + residual1 (residual + attn_out * res_mul)
    # -----------------------------------------------------------------------
    def case_c_fn(h, sf, mask, kc, vc, ci, rm):
        bsz, seq_len, _ = h.shape
        residual = h
        h_ln = input_ln(h)
        q = q_proj(h_ln).view(bsz, seq_len, -1, head_dim).transpose(1, 2)
        k = k_proj(h_ln).view(bsz, seq_len, -1, head_dim).transpose(1, 2)
        v = v_proj(h_ln).view(bsz, seq_len, -1, v_head_dim).transpose(1, 2)
        q = apply_rope_matmul(q, sf)
        k = apply_rope_matmul(k, sf)
        kc, vc = kv_cache_update(k, v, kc, vc, ci)
        attn_out = F.scaled_dot_product_attention(
            q, kc, vc, attn_mask=mask, dropout_p=0.0, scale=scaling, enable_gqa=True,
        )
        attn_out_flat = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_proj = o_proj(attn_out_flat)
        h2 = residual + attn_proj * rm
        return h2.clone(), kc, vc

    # -----------------------------------------------------------------------
    # Case D: C + post_attn_layernorm + MLP (no final residual)
    # -----------------------------------------------------------------------
    def case_d_fn(h, sf, mask, kc, vc, ci, rm):
        bsz, seq_len, _ = h.shape
        residual = h
        h_ln = input_ln(h)
        q = q_proj(h_ln).view(bsz, seq_len, -1, head_dim).transpose(1, 2)
        k = k_proj(h_ln).view(bsz, seq_len, -1, head_dim).transpose(1, 2)
        v = v_proj(h_ln).view(bsz, seq_len, -1, v_head_dim).transpose(1, 2)
        q = apply_rope_matmul(q, sf)
        k = apply_rope_matmul(k, sf)
        kc, vc = kv_cache_update(k, v, kc, vc, ci)
        attn_out = F.scaled_dot_product_attention(
            q, kc, vc, attn_mask=mask, dropout_p=0.0, scale=scaling, enable_gqa=True,
        )
        attn_out_flat = attn_out.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_proj = o_proj(attn_out_flat)
        h2 = residual + attn_proj * rm
        residual2 = h2
        h3 = post_attn_ln(h2)
        h4 = mlp(h3)
        return h4.clone(), residual2.clone(), kc, vc

    comp_a = torch.compile(case_a_fn, backend="inductor", dynamic=False)
    comp_b = torch.compile(case_b_fn, backend="inductor", dynamic=False)
    comp_c = torch.compile(case_c_fn, backend="inductor", dynamic=False)
    comp_d = torch.compile(case_d_fn, backend="inductor", dynamic=False)

    # Fresh caches for each case — re-allocate between runs to avoid
    # cross-contamination of the KV cache state.
    def fresh_caches():
        kcs, vcs = allocate_kv_caches(
            model, batch_size=1, max_cache_len=SEQ_LEN,
            dtype=torch.float16, device="spyre",
        )
        return kcs[0], vcs[0]

    print("\n--- Case A: four parallel FP8 matmuls, input = layernorm(h) ---")
    kc, vc = fresh_caches()
    with torch.no_grad():
        q_raw, k_raw, v_raw, o_raw = comp_a(h, selected_freqs, causal_mask, kc, vc, cache_index)
    finite(q_raw, "q_proj raw (A)")
    finite(k_raw, "k_proj raw (A)")
    finite(v_raw, "v_proj raw (A)")
    finite(o_raw, "o_proj raw (h_ln input) (A)")

    print("\n--- Case B: full FP8 attention q→RoPE→SDPA→o_proj ---")
    kc, vc = fresh_caches()
    with torch.no_grad():
        attn_proj, kc_b, vc_b = comp_b(h, selected_freqs, causal_mask, kc, vc, cache_index)
    finite(attn_proj, "o_proj output (B)")
    finite(kc_b, "key_cache (B)")
    finite(vc_b, "value_cache (B)")

    print("\n--- Case C: B + residual1 (residual + attn_proj * res_mul) ---")
    kc, vc = fresh_caches()
    with torch.no_grad():
        h_after_res1, kc_c, vc_c = comp_c(
            h, selected_freqs, causal_mask, kc, vc, cache_index, res_mul_spyre
        )
    finite(h_after_res1, "h after residual1 (C)")

    print("\n--- Case D: C + post_attn_ln + MLP ---")
    kc, vc = fresh_caches()
    with torch.no_grad():
        mlp_out, res2, kc_d, vc_d = comp_d(
            h, selected_freqs, causal_mask, kc, vc, cache_index, res_mul_spyre
        )
    finite(mlp_out, "mlp output (D)")
    finite(res2, "residual2 before mlp (D)")

    print("\n--- Summary ---")
    print("First 'finite=False' above reveals which component introduces NaN.")
    print("  A-finite  but B-NaN  -> SDPA or o_proj (FP8 on SDPA output)")
    print("  B-finite  but C-NaN  -> residual1 add (unlikely)")
    print("  C-finite  but D-NaN  -> post_attn_ln or MLP")
    print("  All finite           -> NaN only appears in full compiled block (layout?)")


if __name__ == "__main__":
    main()
