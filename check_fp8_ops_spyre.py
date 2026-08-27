"""Stage 1 smoke test, v2 — mirrors torch-spyre's own test_fp8_scaled_mm_cpu
(tests/inductor/test_inductor_ops.py) exactly, to isolate whether the
ReStickifyOpHBM failure from v1 was caused by transposing an already-quantized
FP8 tensor inside the compiled region.

Two changes from v1, both matching the reference test:
  1. The weight is built directly in [K, N] orientation (matching the
     reference test's `b`) — no .t() call anywhere inside the compiled
     region. v1 built it [N, K] (nn.Linear/checkpoint convention) and called
     .t() on the already-FP8-quantized tensor right before the matmul; that
     transpose-after-quantize is the leading suspect for the restickify.
  2. Uses torch.ops.aten._scaled_mm (not torch.ops.spyre.scaled_mm directly),
     passing scale_a/scale_b straight through — this routes through
     scaled_mm_decomp, so the rescale happens INSIDE the compiled graph,
     exactly like the reference test. v1 did the rescale in eager code after
     the compiled call returned.

Run: python3 check_fp8_ops_spyre.py
"""

import torch

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0

M, K, N = 1, 128, 128  # from torch-spyre's own _SCALED_MM_SHAPES (known-good, avoids
# any unrelated padding/stick-alignment question — isolates this test to just
# the transpose/aten._scaled_mm question


def quantize_cpu_reference(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Mirrors the Spyre decomposition of quantize_fp8_with_scale exactly:
    reciprocal-multiply, not divide, then clamp, then cast."""
    return (x * torch.reciprocal(scale)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


@torch.compile(backend="inductor", dynamic=False)
def fp8_op_chain(a, b, scale_a, scale_b):
    """Matches test_fp8_scaled_mm_cpu's spyre_fn exactly: b is already [K, N],
    no transpose anywhere, aten._scaled_mm does the rescale in-graph."""
    q_a = torch.ops.spyre.quantize_fp8_with_scale(a, scale_a)
    q_b = torch.ops.spyre.quantize_weight_fp8_with_scale(b, scale_b)
    return torch.ops.aten._scaled_mm(
        q_a, q_b, scale_a=scale_a, scale_b=scale_b, out_dtype=torch.float16
    )


def main():
    torch.manual_seed(0)

    activation = torch.randn(M, K, dtype=torch.float16)  # [M, K]
    weight = torch.randn(K, N, dtype=torch.float16)       # [K, N] directly — no .t() needed

    x_scale = torch.tensor([1.5], dtype=torch.float16)   # static, chosen
    w_scale = torch.tensor([2.0], dtype=torch.float16)   # static, chosen

    print("=" * 70)
    print("Moving inputs to spyre, then calling the compiled op chain")
    print("(quantize_fp8_with_scale -> quantize_weight_fp8_with_scale -> aten._scaled_mm)")
    print("=" * 70)
    x_spyre = activation.to("spyre")
    w_spyre = weight.to("spyre")
    x_scale_spyre = x_scale.to("spyre")
    w_scale_spyre = w_scale.to("spyre")

    result_spyre = fp8_op_chain(x_spyre, w_spyre, x_scale_spyre, w_scale_spyre)

    print(f"result dtype={result_spyre.dtype}, shape={tuple(result_spyre.shape)}, "
          f"device={result_spyre.device}")
    assert result_spyre.shape == (M, N), result_spyre.shape
    assert torch.isfinite(result_spyre.cpu()).all(), "scaled_mm result has NaN/Inf"
    print("finite: True")

    print("\n" + "=" * 70)
    print("Compare against CPU reference")
    print("=" * 70)
    result_spyre = result_spyre.cpu()

    # CPU reference: quantize the SAME weight/activation the same way (mirroring
    # the Spyre decomposition exactly), raw matmul in fp32 (avoids the fp16
    # overflow we hit building FP8Linear), then the identical rescale — no
    # transpose here either, since weight is already [K, N].
    xq_cpu = quantize_cpu_reference(activation, x_scale)
    wq_cpu = quantize_cpu_reference(weight, w_scale)
    raw_cpu = xq_cpu.to(torch.float32) @ wq_cpu.to(torch.float32)
    result_cpu = (raw_cpu * x_scale.float() * w_scale.float()).to(torch.float16)

    diff = (result_spyre.float() - result_cpu.float())
    rel_error = diff.norm() / result_cpu.float().norm()
    print(f"relative error (Spyre vs CPU reference): {rel_error.item():.6f}")
    print(f"max abs diff: {diff.abs().max().item():.4f}")

    threshold = 0.05
    if rel_error < threshold:
        print(f"\nPASS — within {threshold:.0%} relative error")
    else:
        print(f"\nFAIL — exceeds {threshold:.0%} relative error threshold")


if __name__ == "__main__":
    main()
