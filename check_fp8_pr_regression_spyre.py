"""Minimal reproducer for the fp8-fix-4179 regression, and a bisector for it.

Branch under test:
  Akash-Nayak/torch-spyre  fix/issue-4179-fp8-scaled-mm-fused-n-m  (eb4e3aa)

That branch fixes real shape limits in batchmatmulfp8 — an isolated
spyre.scaled_mm now compiles at every Granite shape, including N=12800 and
M up to 32, which all failed on main. But it regresses the fused-model path:
Granite 3.3 8B with FP8 q/k/v/o_proj compiles all 40 layers on main and fails
on the branch with

    unary_parallel.ddl:15:39: error: Problem with type
    sbf-ddc: DtException: Type incompatible with tensor role in operation
      (ddl_conversion.cpp:2217)
    failed on program 'sdsc_13'

Not M-dependent (same at SEQ_LEN 2 and 8), and not reproducible with the op in
isolation — which is what this script is for.

Working hypothesis. The PR couples two changes: BATCH_MATMUL_FP8_OP now routes
through _cost_model_matmul_planner (work_division.py), and the QFP8WT
outer-stick elemArr encoding became size-dependent (compute_ops.py:
elem_arr_{2,1,0} = [size // 64, 8, 8], alpha = 64). So the encoding's validity
now depends on the n_split the cost model chooses — and the cost model sees a
different graph when the matmul is fused into a block than when it stands
alone. Note also that the PR's own new tests use *fused* shapes (N=6144 QKV,
N=25600 gate_up); stock HF Granite has separate q/k/v (N=4096/1024/1024).

Three cases, each adding one property of the real block, so a failure localises
the trigger rather than just reproducing it:

  A  one matmul, graph-input activation        == the isolated case that PASSES
  B  one matmul, COMPUTED activation           adds a producer op before it
  C  four INDEPENDENT matmuls in one graph     adds q/k/v/o-style fusion
  D  one matmul FED BY another                 adds an FP8->FP8 chain

D is not in Granite's attention (o_proj is fed by SDPA, not by k_proj), but it
IS the MLP's shape: gate_proj -> activation -> down_proj. It was originally
folded into C, where it masked the fusion question with its own failure --
a restickify on a SEN143_FP8 buffer -- so it is now separate.

Run on both branches and compare:
    git -C <torch-spyre> checkout main && rm -rf /tmp/torchinductor_$USER
    python3 check_fp8_pr_regression_spyre.py
    git -C <torch-spyre> checkout fp8-fix-4179 && rm -rf /tmp/torchinductor_$USER
    python3 check_fp8_pr_regression_spyre.py

Clearing the Inductor cache between branches is required — bundles are keyed by
graph hash, not by compiler version, so a stale entry will silently replay.
"""

import torch

FP8_MAX = 448.0
SCALE_EPS = 1e-4
DTYPE = torch.float16

# Granite 3.3 8B attention, at the SEQ_LEN that works on main.
M, K = 2, 4096
N_Q, N_KV, N_O = 4096, 1024, 4096


def fp8_linear(x, w, w_scale):
    """Byte-for-byte the Spyre path of hf_adapters/fp8_linear.py FP8Linear.forward:
    scale and quantize at full rank, flatten only the quantized tensor so the
    matmul is its only consumer, restore rank before rescaling."""
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


@torch.compile(backend="inductor", dynamic=False)
def case_a(x, w, ws):
    return fp8_linear(x, w, ws)


@torch.compile(backend="inductor", dynamic=False)
def case_b(x, g, w, ws):
    h = x * g  # activation is now a computed buffer, as after RMSNorm
    return fp8_linear(h, w, ws)


@torch.compile(backend="inductor", dynamic=False)
def case_c(x, g, wq, wsq, wk, wsk, wv, wsv, wo, wso):
    """Four matmuls in one graph, all reading the same activation — exactly how
    q/k/v_proj consume the RMSNorm output in a real block. No chaining, so a
    failure here is about fusion alone."""
    h = x * g
    return (
        fp8_linear(h, wq, wsq),
        fp8_linear(h, wk, wsk),
        fp8_linear(h, wv, wsv),
        fp8_linear(h, wo, wso),
    )


@torch.compile(backend="inductor", dynamic=False)
def case_d(x, g, wk, wsk, wd, wsd):
    """One FP8 matmul consuming another's output — the MLP's pattern."""
    h = x * g
    return fp8_linear(fp8_linear(h, wk, wsk), wd, wsd)


def _classify(exc):
    msg = str(exc)
    for needle, label in (
        ("Type incompatible with tensor role", "TYPE-ROLE  <-- the regression"),
        ("Not enough elements to distribute", "work-division"),
        ("Incompatible host_size and dim_order", "layout-rank"),
        ("expected exactly 1 generated variable", "no-generated-var"),
        ("ReStickifyOpHBM", "restickify"),
    ):
        if needle in msg:
            return label
    return f"{type(exc).__name__}: {msg.splitlines()[0][:70]}"


def _w(k, n):
    return (
        torch.randn(k, n, dtype=DTYPE).to("spyre"),
        torch.full((n,), 0.1, dtype=DTYPE).to("spyre"),
    )


def run(label, fn, *args):
    print(f"{label:<58} ", end="", flush=True)
    try:
        with torch.no_grad():
            out = fn(*args)
        out = out[0] if isinstance(out, tuple) else out
        ok = torch.isfinite(out.cpu()).all().item()
        print("OK" if ok else "OK (non-finite values)")
    except Exception as exc:  # noqa: BLE001 — the failure is the result
        print(_classify(exc))


def main():
    torch.manual_seed(0)
    x = torch.randn(1, M, K, dtype=DTYPE).to("spyre")
    g = torch.randn(K, dtype=DTYPE).to("spyre")
    wq, wsq = _w(K, N_Q)
    wk, wsk = _w(K, N_KV)
    wv, wsv = _w(K, N_KV)
    wo, wso = _w(K, N_O)
    wd, wsd = _w(N_KV, N_O)

    print(f"M={M} K={K}  (Granite 3.3 8B attention shapes)\n")
    run("A  single matmul, graph-input activation", case_a, x, wq, wsq)
    run("B  single matmul, computed activation", case_b, x, g, wq, wsq)
    run("C  four independent matmuls in one graph", case_c,
        x, g, wq, wsq, wk, wsk, wv, wsv, wo, wso)
    run("D  matmul fed by another matmul (MLP pattern)", case_d,
        x, g, wk, wsk, wd, wsd)


if __name__ == "__main__":
    main()
