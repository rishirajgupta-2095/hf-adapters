"""Shape sweep for spyre.scaled_mm — where does batchmatmulfp8 actually compile?

The 8B runs told us Granite's projections fail in the DeepTools backend
("[distributeElemArrToTemporalLoops] Not enough elements to distribute", always
on program sdsc_13 = spyre.scaled_mm), and that which projection fails moves
with M. But each of those runs costs a full model load, so we learned one data
point at a time.

This does the same op chain as FP8Linear.forward with no model and no
checkpoint, so a shape costs one compile instead of one 8B load. Two targeted
sweeps rather than a full grid, since each compile invokes dxp_standalone:

  1. vary M at k_proj's shape  (K=4096, N=1024)
  2. vary N at the best M      (K=4096)

Known from the 8B runs, as a sanity check on this harness:
  M=1  -> Unsupported: expected exactly 1 generated variable, got set()
  M=2  -> reaches gate_proj (N=intermediate) before failing
  M=8  -> N=4096 compiles, N=1024 fails
  M=128-> N=4096 fails

Run: python3 check_fp8_scaled_mm_shapes_spyre.py
"""

import torch

FP8_MAX = 448.0
SCALE_EPS = 1e-4
DTYPE = torch.float16


@torch.compile(backend="inductor", dynamic=False)
def fp8_matmul(x, w, w_scale):
    """Mirrors FP8Linear.forward's Spyre path exactly — same ops, same order,
    same reshape-late structure. Anything that fails here fails there."""
    x_scale = (x.abs().amax(dim=-1, keepdim=True) * (1.0 / FP8_MAX)).clamp(
        min=SCALE_EPS
    )
    wq = torch.ops.spyre.quantize_weight_fp8_with_scale(w, w_scale)
    xq = torch.ops.spyre.quantize_fp8_with_scale(x, x_scale)
    y = torch.ops.spyre.scaled_mm(
        xq.reshape(-1, xq.shape[-1]), wq, out_dtype=DTYPE
    )
    y = y.reshape(*x.shape[:-1], w.shape[-1]) * x_scale * w_scale
    return y.to(x.dtype)


def _classify(exc: Exception) -> str:
    """Collapse an exception to a short signature, so the table stays readable."""
    msg = str(exc)
    for needle, label in (
        ("Not enough elements to distribute", "work-division"),
        ("expected exactly 1 generated variable", "no-generated-var"),
        ("Incompatible host_size and dim_order", "layout-rank"),
        ("ReStickifyOpHBM", "restickify"),
    ):
        if needle in msg:
            return label
    return type(exc).__name__


def try_shape(m: int, k: int, n: int) -> str:
    torch.manual_seed(0)
    x = torch.randn(1, m, k, dtype=DTYPE).to("spyre")
    w = torch.randn(k, n, dtype=DTYPE).to("spyre")  # [K, N] — pre-transposed
    w_scale = torch.full((n,), 0.1, dtype=DTYPE).to("spyre")
    try:
        with torch.no_grad():
            out = fp8_matmul(x, w, w_scale)
        finite = torch.isfinite(out.cpu()).all().item()
        return "OK" if finite else "NaN/Inf"
    except Exception as exc:  # noqa: BLE001 — any failure is a data point
        return _classify(exc)


def sweep(title: str, shapes: list[tuple[int, int, int]]) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)
    print(f"{'M':>6} {'K':>7} {'N':>7}   result")
    print("-" * 62)
    for m, k, n in shapes:
        print(f"{m:>6} {k:>7} {n:>7}   ", end="", flush=True)
        print(try_shape(m, k, n), flush=True)


def main():
    # 1. M axis, at k_proj's shape — the projection that failed first at M=8.
    sweep(
        "M sweep  (K=4096, N=1024 — k_proj/v_proj shape)",
        [(m, 4096, 1024) for m in (1, 2, 4, 8, 16, 32)],
    )

    # 2. N axis at M=2 — the M that got furthest in the real model. 12800/16384
    #    bracket Granite's intermediate_size, i.e. the MLP projections.
    sweep(
        "N sweep  (M=2, K=4096)",
        [(2, 4096, n) for n in (128, 512, 1024, 4096, 12800, 16384)],
    )


if __name__ == "__main__":
    main()
