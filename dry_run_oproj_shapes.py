"""Dry run: follow one Granite 3.3 8B layer's tensors, focused on o_proj.

CPU only -- no Spyre, no weights, no compile. Real dims from the model config,
random data. Explains why o_proj alone fails FP8 codegen with

    Unexpected stick expression Mod(d0, 64): expected Mod(var, 128)

while q/k/v/gate/up compile. All 40 layers have identical dims, so layer 0 is
representative of every layer.

Projections are modelled as "fresh contiguous buffer of the output shape":
the matmul itself is irrelevant here, only the views between them matter.
RoPE and the KV-cache update are skipped -- neither changes a shape or stride
on the path to o_proj.

Run: python3 dry_run_oproj_shapes.py [--seq 4]
"""

import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
import torch.nn.functional as F
from torch.fx.experimental.proxy_tensor import make_fx

MODEL = "ibm-granite/granite-3.3-8b-instruct-FP8"
FP16, FP8 = torch.float16, torch.float8_e4m3fn
FP8_MAX = 448.0
STICK_BYTES = 128  # one Spyre stick


def eps(dtype):
    """Elements per stick: fp32 32, fp16 64, fp8 128."""
    return STICK_BYTES // torch.empty(0, dtype=dtype).element_size()


def load_dims():
    try:
        from transformers import AutoConfig

        c = AutoConfig.from_pretrained(MODEL)
        return dict(
            layers=c.num_hidden_layers,
            hidden=c.hidden_size,
            inter=c.intermediate_size,
            n_q=c.num_attention_heads,
            n_kv=c.num_key_value_heads,
            hd=c.hidden_size // c.num_attention_heads,
            src="config.json",
        )
    except Exception as e:  # noqa: BLE001
        print(f"(config not loadable: {e}; using known Granite 3.3 8B dims)")
        return dict(
            layers=40, hidden=4096, inter=12800, n_q=32, n_kv=8, hd=128, src="hardcoded"
        )


def proj(x, out_features):
    """A Linear's output: a fresh, contiguous buffer."""
    return torch.randn(*x.shape[:-1], out_features, dtype=x.dtype)


HDR = f"{'step':<5}{'tensor':<34}{'shape':<20}{'stride':<24}{'contig':<8}{'kind':<6}"


def show(step, name, t, note=""):
    kind = "view" if t._base is not None else "own"
    c = "yes" if t.is_contiguous() else "NO"
    print(
        f"{step:<5}{name:<34}{str(list(t.shape)):<20}{str(t.stride()):<24}"
        f"{c:<8}{kind:<6}{note}"
    )


def section(title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=4)
    S = ap.parse_args().seq
    d = load_dims()
    B, H, HQ, HKV, HD, I = 1, d["hidden"], d["n_q"], d["n_kv"], d["hd"], d["inter"]

    section(f"Granite 3.3 8B dims ({d['src']})")
    print(
        f"layers={d['layers']}  hidden={H}  intermediate={I}  "
        f"q_heads={HQ}  kv_heads={HKV}  head_dim={HD}  seq={S}"
    )
    print(f"stick size: fp16 = {eps(FP16)} elems, fp8 = {eps(FP8)} elems")
    print(
        f"head_dim {HD} = {HD // eps(FP16)} fp16 sticks = {HD // eps(FP8)} fp8 stick"
    )

    # ------------------------------------------------------------------
    section("1. Forward through one layer (layer 0; every layer is identical)")
    print(HDR)
    h = torch.randn(B, S, H, dtype=FP16)
    show("in", "hidden_states", h)
    x = torch.randn_like(h)  # RMSNorm output: fresh buffer, same shape
    show("1", "input_layernorm(h)", x, "<- q/k/v input: flat, contiguous")

    q = proj(x, HQ * HD)
    show("2", "q_proj(x)", q)
    q = q.view(B, S, HQ, HD)
    show("3", "  .view(B,S,H,hd)", q, "split last dim: free")
    q = q.transpose(1, 2)
    show("4", "  .transpose(1,2)", q, "strides permuted, no copy")

    k = proj(x, HKV * HD).view(B, S, HKV, HD).transpose(1, 2)
    v = proj(x, HKV * HD).view(B, S, HKV, HD).transpose(1, 2)
    show("5", "k (same recipe, kv heads)", k)
    show("6", "v (same recipe, kv heads)", v)

    attn = F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(), enable_gqa=True
    ).to(FP16)
    show("7", "SDPA(q,k,v)", attn, "fresh buffer, heads-major")

    t = attn.transpose(1, 2)
    show("8", "  .transpose(1,2)", t, "<- NOT contiguous")
    merged = t.reshape(B, S, -1)
    show(
        "9",
        "  .reshape(B,S,-1)",
        merged,
        "eager copies here; Inductor does NOT (see 2)",
    )
    print(f"{'':<5}^^^ o_proj input: last dim {H} is built by MERGING heads x head_dim")

    o = proj(merged, H)
    show("10", "o_proj(merged)", o)
    x2 = torch.randn_like(o)
    show("11", "post_attention_layernorm", x2, "<- gate/up input: flat, contiguous")
    act = proj(x2, I) * proj(x2, I)
    show("12", "act(gate(x)) * up(x)", act, "<- down_proj input: flat, contiguous")
    show("13", "down_proj(...)", proj(act, H))

    # ------------------------------------------------------------------
    section("2. What Inductor sees for step 9 (aten graph of the merge)")
    gm = make_fx(
        lambda a: a.transpose(1, 2).reshape(a.shape[0], a.shape[2], -1),
        tracing_mode="fake",
    )(attn)
    for n in gm.graph.nodes:
        if n.op != "call_function":
            continue
        val = n.meta.get("val")
        shape = list(val.shape) if val is not None else "?"
        stride = val.stride() if val is not None else "?"
        print(f"  {str(n.target):<34}{str(shape):<20}{stride}")
    print(
        "\n  The reshape of a non-contiguous view lowers to clone + view. Inductor"
        "\n  inlines that clone into its consumer (the FP8 quantize) as an INDEX"
        "\n  EXPRESSION instead of materializing a buffer -- which is why an"
        "\n  explicit x.clone() before o_proj changed nothing."
    )

    # ------------------------------------------------------------------
    section("3. Stick arithmetic along the merged dim  k = head * head_dim + d")
    e16, e8 = eps(FP16), eps(FP8)
    print(
        f"  source  attn_out[b, head, s, d]  fp16  stick = {e16} elems along d"
        f"\n  target  quantize(merged)[b,s,k]  fp8   stick = {e8} elems along k\n"
    )
    print(
        f"  {'k':>6}{'head':>6}{'d':>6}   {'fp16 stick':>11}{'Mod(k,64)':>11}"
        f"   {'fp8 stick':>10}{'Mod(k,128)':>12}"
    )
    for kk in (0, 63, 64, 127, 128, 191, 192, 255, H - 1):
        print(
            f"  {kk:>6}{kk // HD:>6}{kk % HD:>6}   {kk // e16:>11}{kk % e16:>11}"
            f"   {kk // e8:>10}{kk % e8:>12}"
        )
    print(
        f"\n  The source is tiled in {e16}-elem fp16 sticks, so its natural stick"
        f"\n  coordinate along the merged k is Mod(k, {e16}). The FP8 quantize output"
        f"\n  must be tiled Mod(k, {e8}). Evaluating the fp16-derived Mod(d0, {e16}) under"
        f"\n  a {e8}-elem stick layout is exactly the expression the error rejects."
        f"\n  q/k/v/gate/up never hit this: their input's last dim is a real memory"
        f"\n  dim, not a merge, so no view-derived stick expression reaches them."
    )

    # ------------------------------------------------------------------
    section("4. Every FP8 projection's input")
    rows = [
        ("q/k/v_proj", "input_layernorm", "own", H, "flat"),
        ("o_proj", "transpose(1,2).reshape", "view", H, f"MERGE {HQ}x{HD}"),
        ("gate/up_proj", "post_attention_layernorm", "own", H, "flat"),
        ("down_proj", "act(gate) * up", "own", I, "flat (fails differently)"),
    ]
    print(
        f"  {'module':<14}{'produced by':<28}{'buffer':<8}{'last dim':<10}"
        f"{'fp16 sticks':<13}{'fp8 sticks':<12}{'origin'}"
    )
    for mod, src, buf, last, origin in rows:
        print(
            f"  {mod:<14}{src:<28}{buf:<8}{last:<10}{last / e16:<13g}"
            f"{last / e8:<12g}{origin}"
        )

    # ------------------------------------------------------------------
    section("5. Option B check: quantize BEFORE the merge, per-token scale kept")
    t32 = attn.transpose(1, 2)  # [B,S,H,hd], d is one fp8 stick
    m16 = t32.reshape(B, S, -1)  # merged fp16, what o_proj sees today
    scale = (m16.abs().amax(-1, keepdim=True) * (1.0 / FP8_MAX)).clamp(min=1e-4)

    ref = (m16 * torch.reciprocal(scale)).clamp(-FP8_MAX, FP8_MAX).to(FP8)
    opt_b = (
        (t32 * torch.reciprocal(scale).unsqueeze(-1))
        .clamp(-FP8_MAX, FP8_MAX)
        .to(FP8)
        .reshape(B, S, -1)
    )
    head_scale = (t32.abs().amax(-1, keepdim=True) * (1.0 / FP8_MAX)).clamp(min=1e-4)
    naive = (
        (t32 * torch.reciprocal(head_scale))
        .clamp(-FP8_MAX, FP8_MAX)
        .to(FP8)
        .reshape(B, S, -1)
    )

    def bytes_equal(a, b):
        return torch.equal(a.view(torch.uint8), b.view(torch.uint8))

    def n_diff(a, b):
        return int((a.view(torch.uint8) != b.view(torch.uint8)).sum())

    print(f"  quantize input shape  [B,S,H,hd] = {list(t32.shape)}  (last dim = 1 fp8 stick)")
    print(f"  per-token scale       {list(scale.shape)} -> broadcast {list(scale.unsqueeze(-1).shape)}")
    print(f"  option B vs today:    bytes equal = {bytes_equal(opt_b, ref)}")
    print(
        f"  naive per-head scale: bytes equal = {bytes_equal(naive, ref)}  "
        f"({n_diff(naive, ref)}/{ref.numel()} differ -- accuracy change, not a fix)"
    )


if __name__ == "__main__":
    main()
