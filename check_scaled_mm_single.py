"""One shape, torch-spyre's own test_fp8_scaled_mm_cpu, standalone.

Same op chain and same scalar scales as
torch-spyre/tests/inductor/test_inductor_ops.py:7743 — so this is their
supported path, not ours. Change M/K/N below to probe other shapes.

Note this differs from FP8Linear in two ways that matter: `a` is 2D (no
reshape, so no rank change), and it routes through aten._scaled_mm rather than
calling spyre.scaled_mm raw.

Their matrix covers M in {1,2,4} at K=N=4096; M=8 is just outside it.

Run: python3 check_scaled_mm_single.py [M K N]
"""

import sys

import torch

# Shape from argv: `python3 check_scaled_mm_single.py 8 4096 4096`
M, K, N = (int(a) for a in (sys.argv[1:4] or (4, 4096, 4096)))
SCALE_A, SCALE_B, BIAS = 1.0, 1.0, 0.0


@torch.compile(backend="inductor", dynamic=False)
def spyre_fn(a, b, scale_a, scale_b, bias):
    q_a = torch.ops.spyre.quantize_fp8_with_scale(a, scale_a)
    q_b = torch.ops.spyre.quantize_weight_fp8_with_scale(b, scale_b)
    return torch.ops.aten._scaled_mm(
        q_a, q_b, scale_a=scale_a, scale_b=scale_b, bias=bias,
        out_dtype=torch.float16,
    )


def pytorch_fn(a, b, scale_a, scale_b, bias):
    q = lambda t, s: (t / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float16)
    return (q(a, scale_a) @ q(b, scale_b)) * (scale_a * scale_b) + bias


def main():
    torch.manual_seed(0)
    a = torch.rand((M, K), dtype=torch.float16)
    b = torch.rand((K, N), dtype=torch.float16)
    sa = torch.tensor(SCALE_A, dtype=torch.float16)
    sb = torch.tensor(SCALE_B, dtype=torch.float16)
    bias = torch.tensor(BIAS, dtype=torch.float16)

    print(f"shape {M}x{K}x{N}  scale_a={SCALE_A} scale_b={SCALE_B} bias={BIAS}")

    ref = pytorch_fn(a, b, sa, sb, bias)

    with torch.no_grad():
        out = spyre_fn(
            a.to("spyre"), b.to("spyre"),
            sa.to("spyre"), sb.to("spyre"), bias.to("spyre"),
        ).cpu()

    diff = (out.float() - ref.float()).abs()
    # atol/rtol 0.1 are the tolerances their own test uses.
    ok = torch.allclose(out.float(), ref.float(), atol=0.1, rtol=0.1)
    print(f"max abs diff: {diff.max().item():.4f}")
    print(f"{'PASS' if ok else 'FAIL'} (atol=rtol=0.1, matching their test)")


if __name__ == "__main__":
    main()
