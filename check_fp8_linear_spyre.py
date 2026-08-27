"""Stage 2: does FP8Linear itself work correctly on real Spyre hardware?

Builds one FP8Linear with a random weight, runs it on CPU (eager) to get a
reference, then moves the SAME module + input to spyre and runs it again —
so this is a true same-weights, same-input, different-device comparison, not
two independently-built references.

FP8Linear.forward() is not itself @torch.compile'd (by design — it's meant to
be called from inside an already-compiled block, exactly like nn.Linear.forward()
is today). Standalone here, it needs to be called from inside a torch.compile'd
wrapper for the same reason Stage 1's op chain did: the spyre:: ops it calls
have no real eager body.

Run: python3 check_fp8_linear_spyre.py
"""

import torch

from hf_adapters.fp8_linear import FP8Linear

IN_FEATURES = 128
OUT_FEATURES = 128  # matches _SCALED_MM_SHAPES' (1, 128, 128) — known-good, no
BATCH = 1           # unrelated padding/stick-alignment question in the mix
DTYPE = torch.float16


@torch.compile(backend="inductor", dynamic=False)
def run(module, x):
    return module(x)


def main():
    torch.manual_seed(0)

    fp8 = FP8Linear(IN_FEATURES, OUT_FEATURES, dtype=DTYPE)
    fp8.weight.copy_(torch.randn(IN_FEATURES, OUT_FEATURES, dtype=DTYPE))
    fp8.weight_scale.copy_(
        (fp8.weight.abs().amax(dim=0) / 448.0).clamp(min=1e-3).to(DTYPE)
    )
    fp8.eval()

    x = torch.randn(BATCH, IN_FEATURES, dtype=DTYPE)

    print("=" * 70)
    print("CPU reference — eager, same module, same input")
    print("=" * 70)
    with torch.no_grad():
        out_cpu = fp8(x)
    print(f"out_cpu: dtype={out_cpu.dtype}, shape={tuple(out_cpu.shape)}")
    assert torch.isfinite(out_cpu).all()

    print("\n" + "=" * 70)
    print("Spyre — same module and input, moved to device, called through")
    print("torch.compile (required — see module docstring)")
    print("=" * 70)
    fp8_spyre = fp8.to("spyre")
    x_spyre = x.to("spyre")

    with torch.no_grad():
        out_spyre = run(fp8_spyre, x_spyre)
    print(f"out_spyre: dtype={out_spyre.dtype}, shape={tuple(out_spyre.shape)}, "
          f"device={out_spyre.device}")
    assert torch.isfinite(out_spyre.cpu()).all(), "spyre output has NaN/Inf"

    print("\n" + "=" * 70)
    print("Compare — same weights, same input, CPU vs Spyre")
    print("=" * 70)
    out_spyre_cpu = out_spyre.cpu()
    diff = out_spyre_cpu.float() - out_cpu.float()
    rel_error = diff.norm() / out_cpu.float().norm()
    print(f"relative error (Spyre vs CPU, same module): {rel_error.item():.6f}")
    print(f"max abs diff: {diff.abs().max().item():.4f}")

    threshold = 0.05
    if rel_error < threshold:
        print(f"\nPASS — within {threshold:.0%} relative error")
    else:
        print(f"\nFAIL — exceeds {threshold:.0%} relative error threshold")


if __name__ == "__main__":
    main()
