"""Minimal reproducer: multi-arg pointwise on a rank-reducing reshape.

    RuntimeError: Incompatible host_size and dim_order

No FP8 anywhere — plain fp16. The point is that this is a general layout
bug, not something specific to quantization; FP8Linear just happens to be the
first thing in hf-adapters that flattens a 3D activation inside a compiled
region.

Modelled directly on the failing op from the real Granite run (buf16):

    op16: buffer=buf16
      layout=FixedLayout(... size=[8, 4096] ...)      <- rank 2 output
          tmp0 = ops.load(buf10, i1 + 4096 * i0)      <- buf10 is [1,8,4096], rank 3
          tmp1 = ops.load(buf15, i0)                  <- buf15 is [8,1],      rank 2

Two things are load-bearing and neither can be dropped:

  1. The rank-3 operand must be a *computed buffer*, not a graph input — a bare
     placeholder can be reinterpreted as rank 2 for free, which would sidestep
     the bug. Hence `h = x * w` first, mirroring RMSNorm's buf10.
  2. The consuming op must have **more than one** argument. `compute_layouts`
     only routes to `_multi_arg_pointwise_layouts` when `len(args) > 1`; the
     single-arg path handles the rank drop fine (in the real graph, buf11's
     `abs()` reduced rank identically and survived).

Why it fails, in `propagate_layouts.py::_is_supported_layout`:

    rank_diff = len(output.size) - len(arg.layout.size)      # 2 - 3 = -1
    projected  = [d - rank_diff for d in dim_order if d >= rank_diff]

The filter+shift models broadcast only ("drop leading dims the input lacks"),
so it assumes rank_diff >= 0. At -1 nothing is filtered and every entry shifts
up: dim_order [0,1] -> [1,2], length 2, against a host_size of length 3. The
layout constructor (spyre_tensor_impl.cpp) requires dim_order to be a full
permutation of the host dims and rejects it. Note the values [1,2] are the
*correct* correspondence — what's wrong is that input dim 0, the size-1 axis
the reshape absorbed, is never named.

Run: python3 check_reshape_rank_bug_spyre.py
"""

import torch

B, S, H = 1, 8, 4096  # the real Granite prefill shape
DTYPE = torch.float16


@torch.compile(backend="inductor", dynamic=False)
def control_no_reshape(x, w, s3):
    """Same ops, rank 3 throughout — rank_diff is 0/positive, so this is the
    case the projection was written for. Expected to compile."""
    h = x * w
    return h * s3


@torch.compile(backend="inductor", dynamic=False)
def repro_reshape(x, w, s2):
    """Identical, except the reshape drops the leading batch dim before the
    second multiply — reproducing buf10 -> buf16."""
    h = x * w
    h2 = h.reshape(-1, h.shape[-1])
    return h2 * s2


def main():
    torch.manual_seed(0)

    x = torch.randn(B, S, H, dtype=DTYPE).to("spyre")
    w = torch.randn(H, dtype=DTYPE).to("spyre")
    s3 = torch.randn(B, S, 1, dtype=DTYPE).to("spyre")  # rank 3 scale
    s2 = torch.randn(B * S, 1, dtype=DTYPE).to("spyre")  # rank 2 scale

    print("=" * 70)
    print(f"CONTROL: rank 3 throughout — [{B},{S},{H}] * [{H}] * [{B},{S},1]")
    print("=" * 70)
    try:
        out = control_no_reshape(x, w, s3)
        print(f"PASS — compiled, out shape {tuple(out.shape)}, "
              f"finite={torch.isfinite(out.cpu()).all().item()}")
    except Exception as e:
        print(f"UNEXPECTED FAILURE — the control was supposed to compile:\n  "
              f"{type(e).__name__}: {e}")
        return

    print("\n" + "=" * 70)
    print(f"REPRO: reshape [{B},{S},{H}] -> [{B * S},{H}], then * [{B * S},1]")
    print("=" * 70)
    try:
        out = repro_reshape(x, w, s2)
        print(f"NO REPRO — it compiled, out shape {tuple(out.shape)}. The bug "
              f"is not (or no longer) triggered by input rank alone.")
    except Exception as e:
        msg = str(e)
        hit = "Incompatible host_size and dim_order" in msg
        print(f"{'REPRODUCED' if hit else 'FAILED (different error)'} — "
              f"{type(e).__name__}: {msg.splitlines()[0]}")


if __name__ == "__main__":
    main()
