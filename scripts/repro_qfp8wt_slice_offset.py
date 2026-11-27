"""Standalone repro: a slice of a QFP8WT tensor along dim 0 starts at twice its offset.

Block ``b`` of an 8-block QFP8WT tensor ``[8, 8, 128, 64]`` holds ``(b + 1) / 4``
(exact in E4M3). Scored against a Q of ones, block ``b`` gives
``128 * (b + 1) / 4 = 32 * (b + 1)`` everywhere. The slice ``kc[2:4]`` should give
blocks 2 and 3 (96, 128); on Spyre it gives blocks 4 and 5 (160, 192), whether
``scaled_bmm`` reads the slice or a clone of it.

    python scripts/repro_qfp8wt_slice_offset.py
"""

import torch

NB, NKV, G, DH, N = 8, 8, 4, 128, 64
DEV = "spyre"


def quantize_weight(w, s):
    return torch.ops.spyre.quantize_weight_fp8_with_scale(w, s)


def scores(q, kc):
    q8 = torch.ops.spyre.qfp8ch(q.clamp(-448.0, 448.0))
    whole = torch.ops.spyre.scaled_bmm(
        q8.expand(NB, NKV, G, DH), kc, out_dtype=torch.float16
    )
    sliced = torch.ops.spyre.scaled_bmm(
        q8.expand(2, NKV, G, DH), kc[2:4], out_dtype=torch.float16
    )
    cloned = torch.ops.spyre.scaled_bmm(
        q8.expand(2, NKV, G, DH), kc[2:4].clone(), out_dtype=torch.float16
    )
    return whole, sliced, cloned


def main():
    vals = (torch.arange(NB, dtype=torch.float16) + 1) / 4
    w = vals.view(NB, 1, 1, 1).expand(NB, NKV, DH, N).contiguous()
    s = torch.ones((1, 1, 1, 1), dtype=torch.float16)
    kc = torch.compile(quantize_weight, dynamic=False)(w.to(DEV), s.to(DEV))
    q = torch.ones((1, NKV, G, DH), dtype=torch.float16).to(DEV)
    outs = torch.compile(scores, dynamic=False)(q, kc)
    whole, sliced, cloned = (o.cpu().float().mean(dim=(1, 2, 3)) for o in outs)
    print("whole kc, per block :", whole.tolist())
    print("expected kc[2:4]    :", whole[2:4].tolist())
    print("kc[2:4]             :", sliced.tolist())
    print("kc[2:4].clone()     :", cloned.tolist())
    ok = torch.allclose(sliced, whole[2:4]) and torch.allclose(cloned, whole[2:4])
    print("OK" if ok else "MISMATCH: the slice read other blocks")


if __name__ == "__main__":
    main()
