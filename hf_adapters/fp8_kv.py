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

"""FP8 (E4M3) KV cache with ``Q · Kᵀ`` on the FP8 engine and ``A · V`` in DL16.

Scheme::

    Q   per-token scale (quantscalepertokenfp8), quantize_fp8_with_scale   -> QFP8CH
    K   per-tensor scale, quantize_weight_fp8_with_scale, stored E4M3      -> QFP8WT
    V   per-tensor scale, quantize_fp8_with_scale, stored E4M3             -> QFP8CH
    Q · Kᵀ   spyre.scaled_bmm, 4D                       (needs torch-spyre#5158)
    A · V    dequantize V, then an fp16 matmul

Q and K take the matmul roles ``FP8Linear`` gives its activation and weight:
Q is ``mat1`` (QFP8CH), K is ``mat2`` (QFP8WT).

K and V scales follow ``fms_mo/aiu_addons/fp8/fp8_attn.py``: one scalar per
sequence per layer, ``amax / 200`` for K and ``amax / 100`` for V, computed on
the first store and then frozen. Mapping the prefill amax below 448 leaves
headroom (2.24x for K, 4.48x for V) for decode tokens larger than anything seen
at prefill; beyond that the quantize clamps rather than overflowing to NaN.

GQA: ``scaled_bmm`` needs matching batch dims, and Granite has 32 query heads
against 8 KV heads. Instead of repeating K and V four times over (FMS's
``repeat_interleave``), the 4 query heads that share a KV head are folded into
the row dim: ``[B, 32, L, Dh] -> [B, 8, 4L, Dh]``. Exact, because those heads
all multiply the same K, and a single reshape since the heads are KV-major.

Nothing here is imported by ``hf_common`` at module load: ``hf_common`` must
not import from its own package. ``attach_fp8_kv`` hands these functions to the
attention modules instead.
"""

from __future__ import annotations

import torch

from .fp8_linear import FP8_DTYPE, FP8_MAX

K_RANGE = 200.0  # fms_mo fp8_attn.py; "these come from vLLM"
V_RANGE = 100.0
SCALE_DTYPE = torch.float16  # dequantize_fp8_with_scale requires an fp16 scale

# Finite stand-in for ``-inf`` in the additive mask. Explicit softmax cannot
# consume an all-``-inf`` row, which block-padded prefill produces for
# left-padding queries (``F.scaled_dot_product_attention`` handled that
# internally). exp() of this underflows to exactly 0 in fp16, so real rows are
# unaffected; a fully masked row gets uniform weights and a finite output that
# is never read.
MASK_FLOOR = -1.0e4


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


def _arrangements():
    """``(QFP8WT, QFP8CH)``, or ``(None, None)`` without torch-spyre."""
    try:
        from torch_spyre._C import ElementArrangement  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - torch-spyre absent
        return None, None
    return ElementArrangement.QFP8WT, ElementArrangement.QFP8CH


def allocate_fp8_kv_caches(model, shapes, batch_size, max_cache_len, device):
    """E4M3 key (QFP8WT) and value (QFP8CH) caches, plus per-layer scales.

    Called from ``hf_common.allocate_kv_caches`` once per request; also resets
    each layer's K/V scales so the request's first store computes fresh ones.
    """
    from .hf_common import allocate_kv_cache_tensor

    k_ea, v_ea = _arrangements()
    key_caches = [
        allocate_kv_cache_tensor(
            batch_size, n_kv, max_cache_len, hd, FP8_DTYPE, device, k_ea
        )
        for (n_kv, hd, _vhd) in shapes
    ]
    value_caches = [
        allocate_kv_cache_tensor(
            batch_size, n_kv, max_cache_len, vhd, FP8_DTYPE, device, v_ea
        )
        for (n_kv, _hd, vhd) in shapes
    ]

    n = attach_fp8_kv(model, batch_size, device)
    if n != len(shapes):
        raise RuntimeError(
            f"FP8 KV cache: wired {n} attention module(s) but allocated "
            f"{len(shapes)} cache(s) — the blocks must expose `self_attn` with "
            "`pre_attn`"
        )
    return key_caches, value_caches


def attach_fp8_kv(model, batch_size, device) -> int:
    """Turn the FP8 KV path on in every attention module, with fresh scales.

    Scales start at 1 and ``_kv_scales_set`` at False, so the first store of the
    request overwrites them. Returns the number of modules wired up.
    """
    n = 0
    for module in model.modules():
        attn = getattr(module, "self_attn", None)
        if attn is None or not hasattr(attn, "pre_attn"):
            continue
        scale_shape = (batch_size, 1, 1, 1)
        attn._k_scale = torch.ones(scale_shape, dtype=SCALE_DTYPE, device=device)
        attn._v_scale = torch.ones(scale_shape, dtype=SCALE_DTYPE, device=device)
        attn._kv_scales_set = False
        attn._fp8_store = store_fp8_kv
        attn._fp8_attention = fp8_attention
        n += 1
    return n


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _per_tensor_scale(x, rng):
    """One scale per sequence, ``amax(|x|) / rng`` -> ``[B, 1, 1, 1]``.

    The per-row amax comes from ``quantscalepertokenfp8`` (its ``scale_ub`` is
    the divisor); the remaining reduction runs over a single flattened dim
    rather than three at once.
    """
    B = x.shape[0]
    if x.device.type == "spyre":
        rows = torch.ops.spyre.quantscalepertokenfp8(x, rng)  # [B, n_kv, n, 1]
    else:
        rows = (x.abs().amax(dim=-1, keepdim=True) / rng).clamp(min=1e-5)
    return rows.reshape(B, -1).amax(dim=1).reshape(B, 1, 1, 1).to(SCALE_DTYPE)


def _quantize(x, scale, kernel: bool):
    """E4M3 quantize: QFP8WT when ``kernel`` (a matmul mat2), else QFP8CH.

    Both device ops run reciprocal -> mul -> clamp(+-448) -> convert, so the
    clamp is built in. The CPU path clamps explicitly: float8_e4m3fn has no
    infinity, and an out-of-range cast becomes NaN rather than saturating.
    """
    if x.device.type == "spyre":
        if kernel:
            return torch.ops.spyre.quantize_weight_fp8_with_scale(x, scale)
        return torch.ops.spyre.quantize_fp8_with_scale(x, scale)
    return (x.float() / scale.float()).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


def _index_copy(cache, cache_index, x):
    """``index_copy_`` along the position dim; CPU has no FP8 kernel, so use bytes."""
    if cache.device.type == "spyre":
        cache.index_copy_(2, cache_index, x)
    else:
        cache.view(torch.uint8).index_copy_(2, cache_index, x.view(torch.uint8))


def store_fp8_kv(attn, k, v, key_cache, value_cache, cache_index):
    """Quantize ``k`` and ``v`` with the layer's scales and scatter them in.

    On the request's first store the scales are computed from these ``k``/``v``
    and written into the layer's scale tensors; every later store reuses them.
    ``attn._kv_scales_set`` is a plain bool, so Dynamo builds one graph for the
    first store and one for the rest instead of branching at runtime. It is
    flipped after the first call by ``StandardGQABlock``.

    Writes in place and returns ``(key_cache, value_cache)``, like
    ``kv_cache_update``.
    """
    if not attn._kv_scales_set:
        attn._k_scale.copy_(_per_tensor_scale(k, K_RANGE))
        attn._v_scale.copy_(_per_tensor_scale(v, V_RANGE))

    _index_copy(key_cache, cache_index, _quantize(k, attn._k_scale, kernel=True))
    _index_copy(value_cache, cache_index, _quantize(v, attn._v_scale, kernel=False))
    return key_cache, value_cache


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def _scaled_bmm(q_fp8, k_fp8_t, s_q, s_k, out_dtype):
    """``(q_fp8 @ k_fp8_t) * s_q * s_k`` — torch-spyre#5158's op on device."""
    if q_fp8.device.type == "spyre":
        if not hasattr(torch.ops.spyre, "scaled_bmm"):
            raise RuntimeError(
                "FP8 KV cache needs torch.ops.spyre.scaled_bmm (torch-spyre#5158)"
            )
        return torch.ops.spyre.scaled_bmm(
            q_fp8, k_fp8_t, s_q, s_k, out_dtype=out_dtype
        )
    raw = q_fp8.float() @ k_fp8_t.float()  # FP32 accumulate, as on the engine
    return (raw * s_q.float() * s_k.float()).to(out_dtype)


def _dequantize(x_fp8, scale):
    if x_fp8.device.type == "spyre":
        return torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)
    return (x_fp8.float() * scale.float()).to(scale.dtype)


def fp8_attention(attn, q, key_cache, value_cache, attn_mask):
    """``softmax(Q · Kᵀ · scale + mask) · V`` with ``Q · Kᵀ`` in FP8.

    ``q`` fp16 ``[B, n_q, L, Dh]``; caches E4M3 ``[B, n_kv, M, Dh]``. Returns
    fp16 ``[B, n_q, L, Dh]``, the shape ``F.scaled_dot_product_attention``
    returned, so the o_proj tail is unchanged.
    """
    B, n_q, L, Dh = q.shape
    n_kv, M = key_cache.shape[1], key_cache.shape[2]

    # GQA fold, in fp16 before quantizing: [B, n_q, L, Dh] -> [B, n_kv, g*L, Dh].
    q_f = q.reshape(B, n_kv, -1, Dh)
    if q_f.device.type == "spyre":
        s_q = torch.ops.spyre.quantscalepertokenfp8(q_f, FP8_MAX)
    else:
        s_q = (q_f.abs().amax(dim=-1, keepdim=True) / FP8_MAX).clamp(min=1e-4)
    q_fp8 = _quantize(q_f, s_q, kernel=False)  # QFP8CH, mat1

    scores = _scaled_bmm(
        q_fp8, key_cache.transpose(-2, -1), s_q, attn._k_scale, out_dtype=q.dtype
    )  # [B, n_kv, g*L, M]
    scores = (scores * attn.scaling).reshape(B, n_q, L, M)
    if attn_mask is not None:
        scores = scores + attn_mask.clamp(min=MASK_FLOOR)
    probs = torch.softmax(scores, dim=-1)

    v = _dequantize(value_cache, attn._v_scale)  # fp16 [B, n_kv, M, Dh]
    out = torch.matmul(probs.reshape(B, n_kv, -1, M), v)  # DL16, [B, n_kv, g*L, Dh]
    return out.reshape(B, n_q, L, Dh)
