# FP8 inference in torch-spyre

## Summary

torch-spyre runs LLM inference with FP8 (E4M3) weights, activations and KV cache on Spyre.
Linear layers run on the FP8 engine through `torch._scaled_mm` / `spyre.scaled_mm`; attention
computes Q·Kᵀ on the FP8 engine through the batched `spyre.scaled_bmm`, reading a KV cache
stored in FP8, and computes A·V on DL16. FP8 halves the bytes of every tensor it is applied
to. On Granite 3.3 8B FP8 (all seven projections in FP8, FP8 KV cache, FP8 Q·Kᵀ), steady
decode is **1.65× faster than fp16** with the same output.

This page complements the FP8 quantization design doc
([FP8_Quantization.md](https://github.com/joyalbin/torch-spyre/blob/dev_fp8_design_doc/docs/source/compiler/FP8_Quantization.md),
in progress), which covers the custom ops, the `_scaled_mm` lowering and the element
arrangements in more depth. Op names and signatures are from torch-spyre `main`
(`4787c10`, 2026-11-22).

**Contents:**

- [FP8 basics on Spyre](#1-fp8-basics-on-spyre)
- [End-to-end model flow](#2-end-to-end-model-flow)
- [Supported ops](#3-supported-ops)
- [FP8 linear pipeline](#4-fp8-linear-pipeline)
- [FP8 attention: the approaches](#5-fp8-attention-the-approaches)
- [Using the FP8 ops in prefill and decode](#6-using-the-fp8-ops-in-prefill-and-decode)
- [Layout propagation and work division](#7-layout-propagation-and-work-division)
- [Results](#8-results)
- [Status](#9-status)

---

## 1. FP8 basics on Spyre

**Format.** FP8 here is E4M3 (`torch.float8_e4m3fn`): 1 sign, 4 exponent and 3 mantissa
bits, range ±448. Being a floating-point format, it keeps about the same relative precision
(~6%) for any magnitude between about 2⁻⁶ and 448.

| PyTorch dtype | Spyre data format | Supported |
|---|---|---|
| `torch.float8_e4m3fn` | `SEN143_FP8` (`"fp8_143"`) | yes |
| `torch.float8_e5m2` | `SEN152_FP8` (`"fp8_152"`) | not yet |

**Quantize and dequantize.**

```
quantize:    s = max(|x|) / 448            (or a given scale)
             x_fp8 = clamp(x / s, -448, 448)  →  E4M3
dequantize:  x ≈ x_fp8 · s
```

**Stick.** A Spyre stick is 128 bytes, so it holds twice as many FP8 values as fp16 values:

```
one stick = 128 bytes
fp16      [ 64 values  ·····································································]
fp8 E4M3  [ 128 values ·····································································]
```

Every tensor in FP8 moves half the bytes of its fp16 form, which is what decode, bound by
memory traffic, gains from.

**Benefits:**

- 50% less memory for weights and KV cache than fp16
- half the memory traffic for the same data, so decode (memory-bound) runs faster
- twice the values per scratchpad buffer, so larger tiles
- FP8 matmuls on the FP8 engine
- accuracy kept by scaling (per-channel weights, per-token activations)

**Two element arrangements** (`torch_spyre._C.ElementArrangement`). An FP8 matmul takes its
two operands in different stick layouts. Both pack two fp16 input sticks (64 values each)
into one 128-byte FP8 stick, differently:

| Arrangement | Produced by | Packing | Covers | Used for |
|---|---|---|---|---|
| **QFP8CH** | `qfp8ch` | two input sticks from the **same** dim, alternating every 8 elements | 128 values along the last dim (1D stick) | activations, `mat1` |
| **QFP8WT** | `qfp8wt` | two input sticks from **different** dims, alternating every element | rows k, k+1 × 64 columns (2D stick `[2, 64]`) | weights / kernel, `mat2` |

```
QFP8CH (mat1 [M, K]):  one stick = 128 consecutive K values of one row
QFP8WT (mat2 [K, N]):  one stick = rows k, k+1  ×  columns n .. n+63   (treated as one atomic unit)
```

Going back, `fp8todl16` turns one FP8 stick into two fp16 sticks.

An FP8 tensor cannot be re-laid-out on device (there is no FP8 restickify), so a `mat2` is
produced directly in QFP8WT from a contiguous `[.., K, N]` fp16 tensor rather than
transposed afterwards.

**Scales.** A value is stored as `x_fp8 = clamp(x / s, ±448)` and recovered as `x_fp8 · s`.
A matmul of two FP8 operands is rescaled once at the end: `raw · s_a · s_b`. Common choices:

| Tensor | Scale |
|---|---|
| weights | per output channel, from the checkpoint, applied once at load |
| activations | per token, computed at run time (`quantscalepertokenfp8`) |
| Q, K, V | static per layer (or per KV head) from the checkpoint; 1.0 when the checkpoint has none (vLLM / spyre-inference convention) |

The FP8 matmul returns fp16 (DL16 on device). DL16's range (up to about 8.6·10⁹) holds the
raw product of two FP8 operands before the rescale, which IEEE fp16 (max 65504) would not.

---

## 2. End-to-end model flow

```
PHASE 1  model preparation (offline)
  fp16 / bf16 model ── quantize (e.g. compressed-tensors) ──► FP8 checkpoint
     weights      E4M3, per-output-channel scale (static)
     activations  per-token scale, computed at run time (dynamic)
     LM head      left in fp16 / bf16
     KV cache     optional static K/V scales (kv_cache_scheme); none → 1.0

PHASE 2  model loading (runtime)
  read the checkpoint's quantization_config ── present ──► FP8 model
     linear layers  → FP8 linears; weights quantized once on device into QFP8WT
     KV cache       → allocated in FP8: K in QFP8WT, V as QFP8CH rows
  absent ──► fp16 model

PHASE 3  inference (runtime)
  token ids ─► embedding (fp16) ─► decoder layers ×N ─► final norm ─► LM head (fp16) ─► logits
  in each decoder layer:
     RMSNorm ─► q/k/v (FP8) ─► RoPE ─► attention (Q·Kᵀ FP8, A·V DL16) ─► o_proj (FP8)
             ─► RMSNorm ─► gate/up/down (FP8)
```

Example checkpoint, `ibm-granite/granite-3.3-8b-instruct-FP8` (`quantization_config`):

| Field | Value |
|---|---|
| format / method | `float-quantized` / `compressed-tensors` |
| weights | 8-bit float, `strategy: channel`, static, symmetric |
| input activations | 8-bit float, `strategy: token`, dynamic, symmetric |
| ignored | `lm_head` |
| `kv_cache_scheme` | `null` (no K/V scales: 1.0) |

Model shapes: hidden 4096, 32 query heads, 8 KV heads, head dim 128, 40 layers,
intermediate 12800, vocab 49159. Seven FP8 linears per layer, 280 in total.

---

## 3. Supported ops

All are `torch.ops.spyre.*` custom ops (`torch_spyre/_inductor/customops.py`).

### Format conversion (no scaling; the input must already be scaled and clamped)

| Op | Signature | Output |
|---|---|---|
| `qfp8ch` | `(input) -> Tensor` | E4M3, QFP8CH |
| `qfp8wt` | `(input) -> Tensor` | E4M3, QFP8WT |

Use these, not `x.to(torch.float8_e4m3fn)`, which on Spyre falls back to the CPU and writes
the bytes in the wrong in-stick order.

### Scale, quantize, dequantize

| Op | Signature | Steps |
|---|---|---|
| `quantscalepertokenfp8` | `(input, scale_ub=448) -> Tensor [*, 1]` | fused `max(\|x\|, dim=-1) / scale_ub`, clamped from below and above |
| `quantize_fp8_with_scale` | `(input, scale) -> Tensor` | `reciprocal(scale)` → `x · inv` → `clamp(±448)` → `qfp8ch` |
| `quantize_weight_fp8_with_scale` | `(input, scale) -> Tensor` | same, ending in `qfp8wt` |
| `dequantize_fp8_with_scale` | `(input, scale) -> Tensor` | `fp8todl16` → `· scale` |

### Backend operations

The custom ops lower to these Spyre operations:

| Operation | Role | Constant |
|---|---|---|
| `qfp8ch` | fp16 → FP8, QFP8CH | in `SPYRE_FP8_OPS` |
| `qfp8wt` | fp16 → FP8, QFP8WT (2D stick) | in `SPYRE_FP8_OPS`, `FP8_2D_STICK_OPS` |
| `fp8todl16` | FP8 → fp16 | in `SPYRE_FP8_OPS` |
| `batchmatmulfp8` | FP8 × FP8 matmul | `BATCH_MATMUL_FP8_OP`; in `SPYRE_FP8_OPS`, `FP8_2D_STICK_OPS` |
| `quantscalepertokenfp8` | per-token scale (fp16 in, fp16 out) | `QUANTSCALEPERTOKENFP8_OP` |

(`torch_spyre/_inductor/constants.py`; `FP8_E4M3FN_MAX = 448.0`.)

### Matmuls

| Op | Signature | Shapes |
|---|---|---|
| `scaled_mm` | `(mat1, mat2, out_dtype=None) -> Tensor` | raw 2D FP8 matmul, `[M, K] @ [K, N]`, no scales |
| `bmm_fp8` | `(mat1, mat2, out_dtype=None) -> Tensor` | raw batched FP8 matmul (deeptools `batchmatmulfp8`): `[*b, M, K] @ [*b, K, N]` with one or two matching batch dims, or `[B, M, K] @ [K, N]` |
| `scaled_bmm` | `(mat1, mat2, scale_a=None, scale_b=None, out_dtype=None, use_fast_accum=False) -> Tensor` | batched FP8 matmul with scales: `bmm_fp8` then `· scale_a · scale_b` (#5158) |

`aten._scaled_mm(mat1, mat2, scale_a, scale_b, bias, scale_result, out_dtype, use_fast_accum)`
decomposes to `spyre.scaled_mm`, then `· scale_a`, `· scale_b` and `+ bias`, each a separate
pointwise op. `scale_result` and `use_fast_accum` are not supported yet (a warning is logged).
The scales are applied after the matmul, not inside the quantize.

In every FP8 matmul `mat1` is QFP8CH and `mat2` is QFP8WT. Batch dims must match; there is
no broadcast of one operand's batch dim inside the op.

---

## 4. FP8 linear pipeline

```
LOAD, once per weight
  W  (E4M3 checkpoint, per-channel scale s_w)
   │  quantize_weight_fp8_with_scale
   ▼
  W_fp8  [in, out]   QFP8WT   (stays on device)

EVERY CALL
  x  (fp16 activation)
   ├── quantscalepertokenfp8 ──────────►  s_x  [*, 1]           per-token scale
   └── quantize_fp8_with_scale(x, s_x) ─►  x_fp8   QFP8CH ──┐
                                                            ├─►  scaled_mm  (FP8 engine)
                                           W_fp8   QFP8WT ──┘          │
                                                                       ▼
                                                     raw (DL16) · s_x · s_w  ──►  y  (fp16)
```

- The weight is quantized once, on the device, into QFP8WT, and stored `[in, out]` so no
  transpose is ever needed.
- Only the activation is quantized at run time, with a per-token scale.
- The two scales are applied after the matmul.

Inside one Granite decoder layer:

```
RMSNorm ─► q/k/v proj (FP8 ×3) ─► RoPE ─► attention ─► o_proj (FP8) ─► RMSNorm ─► gate/up/down (FP8 ×3)
```

7 FP8 linears × 40 layers = 280 in Granite 3.3 8B; the LM head stays fp16.

---

## 5. FP8 attention: the approaches

Attention has two matmuls, each of which can run on the FP8 engine or the DL16 engine.
K and V are stored in FP8 in every option; the choice is which engine runs each matmul.

```
                      ┌──────────────────────────┐     ┌──────────────────┐     ┌──────────────────────────┐
  Q, K cache  ──────► │ STAGE 1  Q·Kᵀ  (scores)  │ ──► │ STAGE 2  mask +  │ ──► │ STAGE 3  A·V  (context)  │ ──► out
                      │ Option 1: FP8 engine     │     │ softmax, DL16    │     │ Approach 1: FP8 engine   │
                      │ Option 2: DL16 engine    │     │ A ∈ [0, 1]       │     │ Approach 2: DL16 engine  │
                      └──────────────────────────┘     └──────────────────┘     └──────────────────────────┘
                                   ▲                                                         ▲
                              K cache (FP8)                                             V cache (FP8)
```

### Stage 1, Option 1: Q·Kᵀ in FP8

```
  Q  (fp16, after RoPE)
   ├── per-token scale ───────────────────────►  s_q
   └── quantize_fp8_with_scale ──►  Q_fp8  QFP8CH ──┐
                                                    ├─►  scaled_bmm  (FP8 engine)
  K cache  E4M3  QFP8WT  (stored as Kᵀ) ────────────┘          │
                                                               ▼
                                         raw · s_q · s_k · 1/√D  ──►  scores (DL16)
```

- K is read as stored: no dequantize, FP8 throughput on the matmul that grows with context.
- Requires K in QFP8WT (it is `mat2`), and the batched FP8 matmul `scaled_bmm`.

### Stage 1, Option 2: Q·Kᵀ in DL16

```
  K cache  E4M3 ──► dequantize_fp8_with_scale(K, s_k) ──► K (DL16) ──┐
                                                                     ├─►  batchmatmul  (DL16 engine)
  Q  (fp16, after RoPE)  ────────────────────────────────────────────┘          │
                                                                                ▼
                                                              raw · 1/√D  ──►  scores (DL16)
```

- Nothing new to build; Q is untouched.
- K widens to DL16 on every read, and the matmul runs at DL16 throughput.

### Stage 2: mask and softmax

```
  scores ──► + causal / padding mask ──► softmax (dim = -1) ──► A  (DL16, values in [0, 1])
```

Always DL16, in every option.

### Stage 3, Approach 1: A·V in FP8

```
  A  (DL16) ──► quantize, constant scale s_a = 1/448 ──►  A_fp8  QFP8CH ──┐
                                                                          ├─►  scaled_bmm  (FP8 engine)
  V cache  E4M3  QFP8WT ──────────────────────────────────────────────────┘          │
                                                                                     ▼
                                                                  raw · s_a · s_v  ──►  out (fp16)
```

- A needs no amax pass: it is bounded in [0, 1], so its scale is a constant.
- V must be stored in QFP8WT, because it becomes `mat2`.

### Stage 3, Approach 2: hybrid, A·V in DL16

```
  V cache  E4M3  QFP8CH rows ──► dequantize_fp8_with_scale(V, s_v) ──► V (DL16) ──┐
                                                                                  ├─►  batchmatmul  (DL16 engine)
  A  (DL16) ──────────────────────────────────────────────────────────────────────┘          │
                                                                                             ▼
                                                                                     out (fp16), no rescale
```

- Nothing new to build: `dequantize_fp8_with_scale` and the DL16 batched matmul exist.
- V keeps the plain row layout (QFP8CH, 1D stick); `s_v` is applied in the dequantize.
- V widens to DL16 on load, and the second matmul runs at DL16 throughput.

### Chosen: Option 1 + Approach 2

**Q·Kᵀ on the FP8 engine, A·V on DL16, K and V stored in FP8.**

```
  Q (fp16) ──► quantize ──► Q_fp8 QFP8CH ──┐
                                           ├─► scaled_bmm ──► · s_q · s_k · 1/√D ──► + mask ──► softmax ──► A (DL16) ──┐
  K cache  E4M3 QFP8WT Kᵀ ─────────────────┘                                                                          │
                                                                                                                      ├─► batchmatmul ──► out (fp16)
  V cache  E4M3 QFP8CH rows ──► dequantize ──► V (DL16) ──────────────────────────────────────────────────────────────┘
```

- K, the operand that grows with context, never widens: it goes as FP8 straight into the
  FP8 engine.
- V keeps the plain row layout, and A·V reuses the existing DL16 path.
- `scaled_bmm` does not broadcast batch dims, so for grouped-query attention the query heads
  that share a KV head are folded into the row dim
  (`[B, H_q, L, D] → [B, H_kv, (H_q/H_kv)·L, D]`) rather than copying K per query head.

---

## 6. Using the FP8 ops in prefill and decode

What a model or serving engine has to respect when it keeps an FP8 KV cache and runs the
chosen attention design. These follow from the ops and arrangements above, not from any one
implementation.

### Cache layouts

| Cache | Arrangement | Shape (per layer) | Requirement |
|---|---|---|---|
| K | **QFP8WT**, stored as Kᵀ | `[blocks, H_kv, D, 64]` | the tensor must be produced by a QFP8WT op (`qfp8wt` / `quantize_weight_fp8_with_scale`, e.g. of zeros) so a later graph reads it as `mat2` |
| V | QFP8CH rows | `[positions, H_kv, D]` | rows written by `index_copy_` on dim 0 need that dim outermost on device |

General rule for indirect writes: the indexed dim must sit at device position 0, or the
scatter writes the wrong rows without an error.

### Write granularity

- A QFP8WT stick is 2 rows of D × 64 positions, so K is written in **whole blocks of 64
  positions** (`index_copy_(0, block_ids, …)`).
- Writing a single position of a QFP8WT cache falls inside a stick and needs a restickify
  that is not available for FP8. Engines that write one token at a time need sub-stick
  scatter (torch-spyre#5178) or must keep the partial block elsewhere and rewrite it whole.
- V rows are one or more whole QFP8CH sticks (D = 128 bytes in FP8), so a single position can
  be written directly.

### Op sequence per phase

```
PREFILL (L positions)
  K  [.., L, D] fp16 ─► group into 64-position blocks ─► Kᵀ, contiguous ─► quantize_weight_fp8_with_scale ─► index_copy_(0, blocks)
  V  [.., L, D] fp16 ─► rows ─► quantize_fp8_with_scale ─► index_copy_(0, rows)
  Q  [.., L, D] fp16 ─► quantize (QFP8CH) ─► scaled_bmm(Q_fp8, K blocks) ─► · s_q · s_k · 1/√D ─► + mask ─► softmax
                    ─► dequantize_fp8_with_scale(V) ─► batchmatmul(A, V) ─► out

DECODE (one position)
  K  ─► the position's 64-position block, complete, ─► quantize_weight_fp8_with_scale ─► index_copy_(0, block)
  V  ─► one row ─► quantize_fp8_with_scale ─► index_copy_(0, row)
  Q  ─► same attention ops as prefill, with L = 1, reading K and V written by earlier graphs
```

### Points that apply to every implementation

- **GQA:** `scaled_bmm` needs matching batch dims. Fold the query heads that share a KV
  head into the row dim (`[B, H_q, L, D] → [B, H_kv, (H_q/H_kv)·L, D]`), or copy K per query head.
- **Scales:** with a static scale of 1.0, quantizing is `clamp(±448)` then `qfp8ch` / `qfp8wt`,
  with no division; the rescale after `scaled_bmm` is `· 1/√D` alone.
- **Cache allocation** happens once per request (or once per pool, when paged); FP8 has no
  device fill, so zeros are written by a quantize op (K) or copied from the host (V).

### Example implementations

- hf-adapters `hf_adapters/fp8_kv.py`: contiguous per-request cache, fp16 staging block for
  the decode K block, GQA fold, Granite 3.3 8B (see that repo's `docs/fp8-on-spyre.md`).
- torch-spyre#5178 tests: paged K/V with block tables and sub-stick scatter.

---

## 7. Layout propagation and work division

**Element arrangements are propagated** through compilation like any other layout property:
the output of `qfp8ch` carries QFP8CH, the output of `qfp8wt` carries QFP8WT, and
`batchmatmulfp8` expects QFP8CH for `mat1` and QFP8WT for `mat2`.

- **No FP8 restickify.** A layout change of an FP8 tensor (`ReStickifyOpHBM` on `SEN143_FP8`)
  is not supported, so the producer must emit the arrangement the consumer needs. In
  practice: quantize a contiguous fp16 `[.., K, N]` tensor with `qfp8wt` to get a `mat2`,
  instead of transposing an FP8 tensor.
- **Arrangement across graphs.** A QFP8WT tensor read by a later graph keeps its arrangement
  only if the tensor itself was produced by a QFP8WT op; a host-allocated FP8 tensor has the
  standard arrangement.
- **QFP8CH and standard.** For a 1D stick along the last dim the two hold the same bytes, so
  QFP8CH-quantized rows can be stored in a standard-arranged buffer and dequantized correctly.

**Work division.** Ops in `FP8_2D_STICK_OPS` (`batchmatmulfp8`, `qfp8wt`) use the 2D stick
`[2, 64]`: two rows of K × 64 of N form one 128-byte unit that is never split. Work is divided
along stick-aligned dims, so a QFP8WT tensor is split in multiples of 2 along K and 64 along N.

---

## 8. Results

Granite 3.3 8B FP8, batch 1, steady decode (warm compile cache), ms per token:

| Config | ms / token | Speedup vs fp16 | Latency | Throughput |
|---|---|---|---|---|
| fp16 | 125.9 | 1.00× | — | — |
| all 7 linears FP8, fp16 attention | 78.5 | 1.60× | −37.6% | +60.4% |
| all 7 linears FP8 + FP8 KV cache + FP8 Q·Kᵀ | 76.1 | 1.65× | −39.6% | +65.4% |

Same output in all three. Over three alternating runs the means were 1.62× and 1.66×; the
FP8 KV cache with FP8 Q·Kᵀ was faster than fp16 attention in each run.

The bandwidth bound for this model is about 1.9× (weights 16.8 GB → 8.8 GB read per token;
the LM head stays fp16). The difference to the measured 1.6–1.65× is fixed per-token work
that FP8 does not shrink (norms, RoPE, softmax, quantize steps, kernel launches).

---

## 9. Status

| Item | State |
|---|---|
| FP8 activation / weight quantize and dequantize custom ops | done |
| `torch._scaled_mm`, 2D FP8 matmul | done |
| Batched FP8 matmul, 3D and 4D (`bmm_fp8`, `scaled_bmm`, #5158) | done |
| Per-token scale on Spyre (`quantscalepertokenfp8`, #3457) | done |
| Chained FP8 matmuls in an MLP (#4741) | done |
| FP8 Q·Kᵀ with K stored in QFP8WT, A·V in DL16 (Granite 3.3 8B, 1.65× decode vs fp16) | runs end to end |
| Paged FP8 KV cache, sub-stick scatter (#5178) | open |
| FP8 A·V (Approach 1) | not started |
| E5M2 | not supported |
