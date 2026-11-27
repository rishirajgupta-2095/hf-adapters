"""Which FP8 KV cache slices read correctly on device? (decode debugging probe)

Fills an FP8 K/V cache with random data through the real store, computes one
layer of decode attention with Q·Kᵀ and the V dequantize over the whole caches
(the trusted path), then recomputes it per group of K blocks with the K cache,
the V cache and/or the decode mask read through slices ``x[b0:b1]``. Each
variant prints its cosine against the trusted output: 1.000000 means the slice
reads correctly.

    python scripts/probe_fp8_decode_slices.py              # on Spyre
    python scripts/probe_fp8_decode_slices.py --device cpu # logic check
"""

import argparse
from types import SimpleNamespace

import torch

from hf_adapters import fp8_kv as F

NKV, G, DH, BLK = 8, 4, 128, 64
SCALE = 0.0078125  # Granite 3.3 attention_multiplier


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", default="spyre")
    p.add_argument("--cache-len", type=int, default=2560)  # 40 K blocks
    p.add_argument("--group", type=int, default=16)  # groups at 0, 16, 32
    p.add_argument(
        "--scores",
        action="store_true",
        help="compare raw Q·Kᵀ per block (sliced K vs whole K) instead of outputs",
    )
    p.add_argument(
        "--no-decode-store",
        dest="decode_store",
        action="store_false",
        help="skip the one-token decode store before attention",
    )
    a = p.parse_args()
    dev, m_len, grp = a.device, a.cache_len, a.group
    nb_all = m_len // BLK
    torch.manual_seed(0)

    kc = F._allocate_k_cache(nb_all, NKV, DH, dev)
    vc = F._allocate_v_cache(m_len, NKV, DH, dev)
    attn = SimpleNamespace(
        scaling=SCALE,
        _fp8_scales=(1.0, 1.0, 1.0),
        _k_scale=torch.ones((1, 1, 1, 1), dtype=torch.float16).to(dev),
        _v_scale=torch.ones((1, 1, 1), dtype=torch.float16).to(dev),
        _k_staging=torch.zeros((BLK, 1, NKV, DH), dtype=torch.float16).to(dev),
    )
    k = (torch.randn(1, NKV, m_len, DH) * 0.5).half().contiguous().to(dev)
    v = (torch.randn(1, NKV, m_len, DH) * 0.5).half().contiguous().to(dev)
    i0, i1, i2 = F.fp8_cache_index(1, m_len, 0, m_len, device=dev)

    def store(k, v, kc, vc, i0, i1, i2):
        return F.store_fp8_kv(attn, k, v, kc, vc, (i0, i1, i2))

    compile_ = (
        (lambda f: f) if dev == "cpu" else (lambda f: torch.compile(f, dynamic=False))
    )
    compile_(store)(k, v, kc, vc, i0, i1, i2)

    # Decode-style store, as the model does each step: one token into the
    # staging slot, its whole K block requantized and rewritten, one V row.
    pos = m_len - 2 * BLK  # first slot of the second-to-last block
    k1 = (torch.randn(1, NKV, 1, DH) * 0.5).half().contiguous().to(dev)
    v1 = (torch.randn(1, NKV, 1, DH) * 0.5).half().contiguous().to(dev)
    d0, d1, d2 = F.fp8_cache_index(1, m_len, pos, 1, device=dev)
    if a.decode_store:
        compile_(store)(k1, v1, kc, vc, d0, d1, d2)

    q = torch.randn(1, NKV * G, 1, DH).half().contiguous().to(dev)
    mask_host = torch.zeros(1, 1, 1, m_len, dtype=torch.float16)
    mask_host[..., pos + 1 :] = float("-inf")  # decode at pos: later positions masked
    mask = F.block_decode_mask(mask_host).to(dev)

    def make(slice_k, slice_v, slice_m):
        def f(q, kc, vc, mask):
            q_fp8 = F._quantize_q(q.reshape(1, NKV, G, DH), 1.0)
            raw_all = F._scaled_bmm(
                q_fp8.expand(nb_all, NKV, G, DH), kc, out_dtype=torch.float16
            )
            mask_all = mask.clamp(min=F.MASK_FLOOR)
            v_all = F._dequantize(vc, attn._v_scale)
            parts = []
            for b0 in range(0, nb_all, grp):
                b1 = min(b0 + grp, nb_all)
                nb = b1 - b0
                if slice_k:
                    raw = F._scaled_bmm(
                        q_fp8.expand(nb, NKV, G, DH), kc[b0:b1], out_dtype=torch.float16
                    )
                else:
                    raw = raw_all[b0:b1]
                m_g = (
                    mask[b0:b1].clamp(min=F.MASK_FLOOR) if slice_m else mask_all[b0:b1]
                )
                s = (raw * SCALE).to(torch.float16) + m_g
                if slice_v:
                    v_rows = F._dequantize(vc[b0 * BLK : b1 * BLK], attn._v_scale)
                else:
                    v_rows = v_all[b0 * BLK : b1 * BLK]
                parts.append(F._group_part(s, v_rows, nb, NKV))
            m_all = parts[0][0]
            for part in parts[1:]:
                m_all = torch.maximum(m_all, part[0])
            l_all = o_all = None
            for m_g2, l_g, o_g in parts:
                w = torch.exp(m_g2 - m_all)
                l_all = l_g * w if l_all is None else l_all + l_g * w
                o_all = o_g * w if o_all is None else o_all + o_g * w
            return (o_all / l_all).reshape(1, NKV * G, 1, -1)

        return compile_(f)

    if a.scores:
        return compare_k_scores(q, kc, nb_all, grp, compile_)

    ref = make(False, False, False)(q, kc, vc, mask).cpu().float()
    print(
        f"cache {m_len} ({nb_all} K blocks), groups of {grp}, device {dev}, "
        f"decode store at {pos}: {a.decode_store}"
    )
    for name, flags in [
        ("K sliced", (True, False, False)),
        ("V sliced", (False, True, False)),
        ("mask sliced", (False, False, True)),
        ("all sliced", (True, True, True)),
    ]:
        out = make(*flags)(q, kc, vc, mask).cpu().float()
        cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0)
        diff = (out - ref).abs().max().item()
        print(
            f"{name:12s} cos vs whole-cache path: {cos.item():.6f}   max abs diff {diff:.4f}"
        )


def compare_k_scores(q, kc, nb_all, grp, compile_):
    """Raw Q·Kᵀ per K block, from ``kc[b0:b1]`` per group vs one product over the
    whole cache. Outputs hide score errors (random data gives a flat softmax),
    so compare the scores themselves; for a wrong block, name the whole-cache
    block it matches instead."""

    def f(q, kc):
        q_fp8 = F._quantize_q(q.reshape(1, NKV, G, DH), 1.0)
        whole = F._scaled_bmm(
            q_fp8.expand(nb_all, NKV, G, DH), kc, out_dtype=torch.float16
        )
        groups = [
            F._scaled_bmm(
                q_fp8.expand(min(b0 + grp, nb_all) - b0, NKV, G, DH),
                kc[b0 : min(b0 + grp, nb_all)],
                out_dtype=torch.float16,
            )
            for b0 in range(0, nb_all, grp)
        ]
        return whole, *groups

    whole, *groups = (t.cpu().float() for t in compile_(f)(q, kc))
    flat = whole.reshape(nb_all, -1)
    print(f"raw Q·Kᵀ per block, {nb_all} K blocks in groups of {grp}")
    for gi, raw in enumerate(groups):
        b0 = gi * grp
        bad = []
        for j in range(raw.shape[0]):
            x = raw[j].reshape(-1)
            if torch.allclose(x, flat[b0 + j], rtol=1e-2, atol=1e-2):
                continue
            cos = torch.nn.functional.cosine_similarity(flat, x[None], dim=1)
            best = int(cos.argmax())
            bad.append(f"{b0 + j}->{best}({cos[best].item():.3f})")
        n = raw.shape[0]
        print(
            f"group {gi} blocks {b0}..{b0 + n - 1}: {n - len(bad)}/{n} match"
            + (f"; sliced block -> best whole block: {' '.join(bad)}" if bad else "")
        )


if __name__ == "__main__":
    main()
