# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FP8 (E4M3) key cache and FP8 ``Q · Kᵀ`` for Spyre.

Stage 1 "Option 1": K is quantized once when it is written to the cache and
stays E4M3 all the way into the score matmul, so the largest matmul in the
layer runs on the FP8 engine. Everything else stays fp16 — softmax, ``A · V``,
the value cache and the mask.

Why quantize at store rather than on read: a key row is written once and read
on every later decode step (~L times at sequence length L). The same argument
applies to the element *arrangement*, which is why the cache is allocated
QFP8WT — the arrangement an FP8 matmul wants for its ``mat2`` operand — so a
read is a plain load with no restickify.

Scale granularity is forced, not chosen::

    score[i,j] = Σ_d q[i,d]·k[j,d] = S_q[i] · S_k[j] · Σ_d qq[i,d]·kq[j,d]

The scales factor out of the sum only because each is constant along the
contraction dim ``head_dim``. A scale varying along ``head_dim`` could not be
compensated after the matmul at all. So both Q and K use one scale per
(batch, head, token) row — exactly what ``quantscalepertokenfp8`` produces,
since it reduces over the last dim.

Roles decide which quantize op to use, mirroring ``FP8Linear``: Q is ``mat1``
(activation, QFP8CH) and K is ``mat2`` (kernel, QFP8WT).

OPEN / needs a torch-spyre answer (torch-spyre#4568): the name and operand
convention of the 4D FP8 BMM, and whether ``index_copy_`` lowers onto a
QFP8WT-arranged tensor. Both are isolated here — ``_fp8_bmm`` for the first,
``allocate_k_cache`` for the second — so neither is spread through the adapter.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .fp8_linear import FP8_DTYPE, FP8_MAX

# Finite stand-in for ``-inf`` in the additive mask. Explicit softmax cannot
# consume an all-``-inf`` row, which block-padded prefill produces for
# left-padding queries; ``F.scaled_dot_product_attention`` handled that
# internally. At this magnitude exp() underflows to exactly 0 in fp16, so real
# rows are unaffected, while a fully masked row yields uniform weights and a
# finite output that is never read.
MASK_FLOOR = -1.0e4

# Candidate names for the 4D FP8 BMM, newest first. Resolved once, lazily.
_FP8_BMM_CANDIDATES = ("scaled_bmm", "batched_scaled_mm", "batchmatmulfp8")


def fp8_kv_enabled(module) -> bool:
    """True when this attention module (or model) runs the FP8 key-cache path."""
    return getattr(module, "_spyre_fp8_kv_cache", False)


def _resolve_fp8_bmm():
    """Return the torch-spyre 4D FP8 matmul op, or None when the build lacks it."""
    for name in _FP8_BMM_CANDIDATES:
        try:
            op = getattr(torch.ops.spyre, name)
        except (AttributeError, RuntimeError):
            continue
        if op is not None:
            return op
    return None


def _fp8_bmm(q_fp8: torch.Tensor, k_fp8: torch.Tensor) -> torch.Tensor:
    """Raw ``Q · Kᵀ`` on E4M3 operands; no scales applied (the caller rescales).

    ``q_fp8`` is ``[b, g, Lq, Dh]`` (QFP8CH) and ``k_fp8`` is ``[b, 1, Lkv, Dh]``
    (QFP8WT). The contraction is over ``Dh``, so K enters transposed.
    """
    op = _resolve_fp8_bmm()
    if op is None:
        raise RuntimeError(
            "FP8 KV cache needs a 4D FP8 matmul op in torch-spyre "
            f"(looked for torch.ops.spyre.{{{', '.join(_FP8_BMM_CANDIDATES)}}}); "
            "see torch-spyre#4568"
        )
    return op(q_fp8, k_fp8.transpose(2, 3))


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


def k_cache_arrangement():
    """``ElementArrangement.QFP8WT``, or None off-device / without torch-spyre."""
    try:
        from torch_spyre._C import ElementArrangement  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - torch-spyre absent
        return None
    return ElementArrangement.QFP8WT


def allocate_k_scale_cache(batch_size, num_kv_heads, max_cache_len, dtype, device):
    """Per-token key scales ``[B, n_kv, max_cache_len, 1]``, zero-filled.

    Written by the same scatter as the key cache, so the scale for a cache slot
    always travels with the row it belongs to. Zero is the right fill: an
    unwritten slot contributes a 0 score, and the mask then drives it to
    ``MASK_FLOOR``.

    One element per (head, position) — 1/128th of the key cache at
    ``head_dim=128``.
    """
    from .hf_common import allocate_kv_cache_tensor

    return allocate_kv_cache_tensor(
        batch_size, num_kv_heads, max_cache_len, 1, dtype, device
    )


def attach_k_scale_caches(model, scale_caches) -> int:
    """Give each attention module its layer's scale cache and turn the path on.

    The key/value caches are passed through the block signature, but the scale
    cache rides on the module instead: an extra per-layer tensor argument would
    have to be threaded through every block and backbone signature in the
    adapter. Assigned per request, since ``max_cache_len`` is per request.

    Returns the number of layers wired up.
    """
    n = 0
    for module in model.modules():
        attn = getattr(module, "self_attn", None)
        if attn is None or not hasattr(attn, "pre_attn"):
            continue
        if n >= len(scale_caches):
            break
        attn._k_scale_cache = scale_caches[n]
        attn._spyre_fp8_kv_cache = True
        n += 1
    return n


# ---------------------------------------------------------------------------
# Store side
# ---------------------------------------------------------------------------


def quantize_k_for_store(k: torch.Tensor):
    """``k`` fp16 ``[B, n_kv, n, Dh]`` -> ``(k_e4m3 QFP8WT, scale [B, n_kv, n, 1])``.

    ``quantize_weight_fp8_with_scale`` is the ``mat2``/kernel-arrangement
    quantize, the same op ``FP8Linear`` uses for its weight; the scale
    broadcasts over the last dim.
    """
    if k.device.type != "spyre":
        return _quantize_k_cpu(k)
    s_k = torch.ops.spyre.quantscalepertokenfp8(k, FP8_MAX)
    k_q = torch.ops.spyre.quantize_weight_fp8_with_scale(k, s_k)
    return k_q, s_k


def _quantize_k_cpu(k: torch.Tensor):
    """CPU reference for ``quantize_k_for_store``, for tests without Spyre."""
    s_k = (k.abs().amax(dim=-1, keepdim=True) * (1.0 / FP8_MAX)).clamp(min=1e-4)
    k_q = (k * torch.reciprocal(s_k)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return k_q, s_k.to(k.dtype)


def store_fp8_kv(
    k, v, key_cache, value_cache, k_scale_cache, cache_index
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize K, then scatter K, its scales and (fp16) V into the caches.

    Writes in place and returns ``(key_cache, value_cache)``, matching
    ``kv_cache_update``. Correctness depends on the pinned device layout — see
    ``allocate_kv_cache_tensor`` and torch-spyre#3705.
    """
    k_q, s_k = quantize_k_for_store(k)
    key_cache.index_copy_(2, cache_index, k_q)
    k_scale_cache.index_copy_(2, cache_index, s_k.to(k_scale_cache.dtype))
    value_cache.index_copy_(2, cache_index, v.to(value_cache.dtype))
    return key_cache, value_cache


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def fp8_gqa_attention(
    q, key_cache, k_scale_cache, value_cache, attn_mask, scaling
) -> torch.Tensor:
    """GQA attention with ``Q · Kᵀ`` in FP8 and everything after it in fp16.

    ``q`` is ``[B, n_q, Lq, Dh]`` fp16; ``key_cache`` is E4M3/QFP8WT
    ``[B, n_kv, L, Dh]`` with ``k_scale_cache`` ``[B, n_kv, L, 1]``;
    ``value_cache`` is fp16. Returns ``[B, n_q, Lq, Dh]`` fp16.

    One BMM pair per KV head, as in ``hf_gemma2``: materializing ``repeat_kv``
    produces an unsupported Spyre layout, and a single grouped 5D BMM exceeds
    the backend's rank limit.
    """
    n_q = q.shape[1]
    n_kv = key_cache.shape[1]
    groups = n_q // n_kv

    s_q = torch.ops.spyre.quantscalepertokenfp8(q, FP8_MAX)  # [B, n_q, Lq, 1]
    q_fp8 = torch.ops.spyre.quantize_fp8_with_scale(q, s_q)  # QFP8CH

    mask = attn_mask.clamp(min=MASK_FLOOR) if attn_mask is not None else None

    outs = []
    for kv in range(n_kv):
        lo, hi = kv * groups, (kv + 1) * groups
        raw = _fp8_bmm(q_fp8[:, lo:hi], key_cache[:, kv : kv + 1])
        # S_q scales query rows, S_k scales key columns: [.., Lq, 1] * [.., 1, L].
        s_k_row = k_scale_cache[:, kv : kv + 1].transpose(2, 3)
        scores = raw.to(q.dtype) * s_q[:, lo:hi] * s_k_row * scaling
        if mask is not None:
            scores = scores + mask
        probs = F.softmax(scores, dim=-1)
        outs.append(torch.matmul(probs, value_cache[:, kv : kv + 1]))
    return torch.cat(outs, dim=1)


def fp8_gqa_attention_cpu(
    q, key_cache, k_scale_cache, value_cache, attn_mask, scaling
) -> torch.Tensor:
    """CPU reference for ``fp8_gqa_attention`` — same math, fp32 accumulation.

    Lets the quantization error and the scale bookkeeping be tested without a
    Spyre device.
    """
    n_q = q.shape[1]
    n_kv = key_cache.shape[1]
    groups = n_q // n_kv

    s_q = (q.abs().amax(dim=-1, keepdim=True) * (1.0 / FP8_MAX)).clamp(min=1e-4)
    q_fp8 = (q * torch.reciprocal(s_q)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    mask = attn_mask.clamp(min=MASK_FLOOR) if attn_mask is not None else None

    outs = []
    for kv in range(n_kv):
        lo, hi = kv * groups, (kv + 1) * groups
        qg = q_fp8[:, lo:hi].to(torch.float32)
        kg = key_cache[:, kv : kv + 1].to(torch.float32)
        raw = torch.matmul(qg, kg.transpose(2, 3))
        s_k_row = k_scale_cache[:, kv : kv + 1].transpose(2, 3).to(torch.float32)
        scores = raw * s_q[:, lo:hi].to(torch.float32) * s_k_row * scaling
        if mask is not None:
            scores = scores + mask.to(torch.float32)
        probs = F.softmax(scores, dim=-1)
        outs.append(torch.matmul(probs, value_cache[:, kv : kv + 1].to(torch.float32)))
    return torch.cat(outs, dim=1).to(q.dtype)
