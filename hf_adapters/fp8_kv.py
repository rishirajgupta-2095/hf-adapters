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
    Q · Kᵀ   spyre.scaled_bmm, 4D, reads the K cache as is  (torch-spyre#5158)
    A · V    dequantize V, then an fp16 matmul

Q and K take the matmul roles ``FP8Linear`` gives its activation and weight:
Q is ``mat1`` (QFP8CH), K is ``mat2`` (QFP8WT).

Cache layouts (not paged: every sequence owns ``NB = max_cache_len / 64``
consecutive blocks)::

    K   [B*NB, n_kv, Dh, 64]   Kᵀ blocks, QFP8WT; block ``b*NB + j`` holds
                               positions ``64j .. 64j+63`` of sequence ``b``
    V   [B*M, n_kv, Dh]        one row per position, QFP8CH; row ``b*M + p``

A QFP8WT stick spans 2 rows of ``Dh`` x 64 positions, so a single position
cannot be written on its own: the scatter would restickify, which torch-spyre
cannot emit for FP8. K is therefore only ever written as whole blocks, with
``index_copy_`` on dim 0. Decode keeps the current block's K in an fp16
staging buffer ``[64, B, n_kv, Dh]`` per layer, adds the new token to it, and
re-quantizes and rewrites the whole block each step; with the scale frozen,
the bytes of earlier positions come out the same. Prefill always ends on a
block boundary (it is padded to 64), so decode starts each block at slot 0.
Slots past the current token keep stale K from the previous block; they are
finite and masked, so they never contribute.

The K cache is allocated by quantizing zeros on device, so the tensor itself
carries the QFP8WT layout and a later graph reads it correctly. A host
allocation copied to the device gets the STANDARD layout, which reads back
wrong in the next graph.

K and V scales follow ``fms_mo/aiu_addons/fp8/fp8_attn.py``: ``amax / 200`` for
K and ``amax / 100`` for V, computed on the first store and then frozen. Mapping
the prefill amax below 448 leaves headroom (2.24x for K, 4.48x for V) for decode
tokens larger than anything seen at prefill; beyond that the quantize clamps
rather than overflowing to NaN. One scale per layer covers the whole batch, so
every K block shares it.

GQA: ``scaled_bmm`` needs matching batch dims, and Granite has 32 query heads
against 8 KV heads. Instead of repeating K and V four times over (FMS's
``repeat_interleave``), the 4 query heads that share a KV head are folded into
the row dim: ``[B, 32, L, Dh] -> [B, 8, 4L, Dh]``. Exact, because those heads
all multiply the same K, and a single reshape since the heads are KV-major. The
folded Q is then repeated over the ``NB`` blocks so one ``scaled_bmm`` scores
every block, and softmax runs over all ``M`` positions at once.

Nothing here is imported by ``hf_common`` at module load: ``hf_common`` must
not import from its own package. ``attach_fp8_kv`` hands these functions to the
attention modules instead.
"""

from __future__ import annotations

import functools

import torch

from .fp8_linear import FP8_DTYPE, FP8_MAX

KV_BLOCK = 64  # positions per K block: the N extent of one QFP8WT stick
K_RANGE = 200.0  # fms_mo fp8_attn.py; "these come from vLLM"
V_RANGE = 100.0
SCALE_DTYPE = torch.float16  # dequantize_fp8_with_scale requires an fp16 scale
STAGING_DTYPE = torch.float16  # FP8Linear outputs fp16

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


def _qfp8wt_zeros(z, s):
    return torch.ops.spyre.quantize_weight_fp8_with_scale(z, s)


_qfp8wt_zeros_compiled = None


def _fp8_zeros(shape, device):
    # Byte 0x00 is +0.0 in E4M3; CPU has no FP8 fill.
    return torch.zeros(shape, dtype=torch.uint8).view(FP8_DTYPE).to(device)


def _allocate_k_cache(n_blocks, n_kv, head_dim, device):
    """Zeroed ``[n_blocks, n_kv, head_dim, KV_BLOCK]`` K cache carrying QFP8WT."""
    shape = (n_blocks, n_kv, head_dim, KV_BLOCK)
    if torch.device(device).type != "spyre":
        return _fp8_zeros(shape, device)
    global _qfp8wt_zeros_compiled
    if _qfp8wt_zeros_compiled is None:
        _qfp8wt_zeros_compiled = torch.compile(_qfp8wt_zeros, dynamic=False)
    z = torch.zeros(shape, dtype=torch.float16, device=device)
    s = torch.ones((1, 1, 1, 1), dtype=SCALE_DTYPE, device=device)
    return _qfp8wt_zeros_compiled(z, s)


def allocate_fp8_kv_caches(model, shapes, batch_size, max_cache_len, device):
    """E4M3 key (QFP8WT blocks) and value (QFP8CH rows) caches, plus scales.

    Called from ``hf_common.allocate_kv_caches`` once per request. Also resets
    each layer's K/V scales and K staging buffer, and installs the cache-index
    builder the generate loop uses for these layouts.
    """
    if max_cache_len % KV_BLOCK:
        raise ValueError(
            f"FP8 KV cache: max_cache_len {max_cache_len} is not a multiple of "
            f"{KV_BLOCK}"
        )
    n_blocks = max_cache_len // KV_BLOCK
    key_caches = [
        _allocate_k_cache(batch_size * n_blocks, n_kv, hd, device)
        for (n_kv, hd, _vhd) in shapes
    ]
    # V rows are written with a plain dim-0 scatter, which a host-allocated
    # (STANDARD) FP8 tensor takes; torch-spyre#5178's tests do the same.
    value_caches = [
        _fp8_zeros((batch_size * max_cache_len, n_kv, vhd), device)
        for (n_kv, _hd, vhd) in shapes
    ]

    n = attach_fp8_kv(model, shapes, batch_size, device)
    if n != len(shapes):
        raise RuntimeError(
            f"FP8 KV cache: wired {n} attention module(s) but allocated "
            f"{len(shapes)} cache(s) — the blocks must expose `self_attn` with "
            "`pre_attn`"
        )
    model._spyre_cache_index_fn = functools.partial(
        fp8_cache_index, batch_size, max_cache_len
    )
    model._spyre_after_prefill_fn = functools.partial(freeze_kv_scales, model)
    return key_caches, value_caches


def attach_fp8_kv(model, shapes, batch_size, device) -> int:
    """Turn the FP8 KV path on in every attention module, with fresh state.

    Scales start at 1 and ``_kv_scales_set`` at False, so the first store of the
    request overwrites them. Returns the number of modules wired up.
    """
    attns = [
        attn
        for module in model.modules()
        if (attn := getattr(module, "self_attn", None)) is not None
        and hasattr(attn, "pre_attn")
    ]
    for attn, (n_kv, hd, _vhd) in zip(attns, shapes):
        attn._k_scale = torch.ones((1, 1, 1, 1), dtype=SCALE_DTYPE, device=device)
        attn._v_scale = torch.ones((1, 1, 1), dtype=SCALE_DTYPE, device=device)
        attn._kv_scales_set = False
        attn._k_staging = torch.zeros(
            (KV_BLOCK, batch_size, n_kv, hd), dtype=STAGING_DTYPE, device=device
        )
        attn._fp8_store = store_fp8_kv
        attn._fp8_attention = fp8_attention
    return len(attns)


def freeze_kv_scales(model) -> None:
    """After prefill: mark every layer's K/V scales as set, so decode reuses them.

    ``_kv_scales_set`` is a plain bool read inside the compiled block, so Dynamo
    specializes on it: the prefill graph computes the scales and the decode
    graphs only read them. It is flipped here, from ``generate`` between prefill
    and the first decode step, because the block is one compiled graph with no
    eager step between the store and attention.
    """
    for module in model.modules():
        attn = getattr(module, "self_attn", None)
        if attn is not None and getattr(attn, "_fp8_store", None) is not None:
            attn._kv_scales_set = True


def fp8_cache_index(batch_size, max_cache_len, start, length, device=None):
    """Destination indices for positions ``[start, start + length)``.

    Replaces ``hf_common.make_cache_index`` while the FP8 KV cache is on.
    Returns ``(k_blocks, k_slot, v_rows)``:

      * ``k_blocks`` — K blocks to rewrite: every block in the range at
        prefill, the block holding ``start`` at decode
      * ``k_slot``   — decode only: the staging slot of ``start``
      * ``v_rows``   — one V row per sequence and position
    """
    n_blocks = max_cache_len // KV_BLOCK
    if length == 1:
        block, slot = divmod(start, KV_BLOCK)
        blocks = [block]
    else:
        if start % KV_BLOCK or length % KV_BLOCK:
            raise ValueError(
                f"FP8 KV cache: prefill range [{start}, {start + length}) is not "
                f"aligned to {KV_BLOCK}"
            )
        blocks = list(range(start // KV_BLOCK, (start + length) // KV_BLOCK))
        slot = 0
    k_blocks = torch.tensor(
        [b * n_blocks + j for b in range(batch_size) for j in blocks], dtype=torch.long
    )
    k_slot = torch.tensor([slot], dtype=torch.long)
    v_rows = torch.tensor(
        [
            b * max_cache_len + p
            for b in range(batch_size)
            for p in range(start, start + length)
        ],
        dtype=torch.long,
    )
    if device is not None:
        k_blocks, k_slot, v_rows = (t.to(device) for t in (k_blocks, k_slot, v_rows))
    return k_blocks, k_slot, v_rows


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def _per_tensor_scale(x, rng):
    """One scale for the whole tensor, ``amax(|x|) / rng`` -> ``[1, 1, 1, 1]``.

    The per-row amax comes from ``quantscalepertokenfp8`` (its ``scale_ub`` is
    the divisor); the rest is reduced one dim at a time. Flattening first
    (``reshape(1, -1).amax(1)``) merges the head and position dims into one
    reduction, which the device rejects: "More than one reduction dim is split
    across cores".
    """
    if x.device.type == "spyre":
        rows = torch.ops.spyre.quantscalepertokenfp8(x, rng)  # [B, n_kv, n, 1]
    else:
        rows = (x.abs().amax(dim=-1, keepdim=True) / rng).clamp(min=1e-5)
    s = rows.amax(dim=2, keepdim=True)  # positions -> [B, n_kv, 1, 1]
    s = s.amax(dim=1, keepdim=True)  # heads -> [B, 1, 1, 1]
    return s.amax(dim=0, keepdim=True).to(SCALE_DTYPE)  # batch -> [1, 1, 1, 1]


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


def _index_copy(cache, index, x):
    """``index_copy_`` along dim 0; CPU has no FP8 kernel, so use bytes."""
    if cache.device.type == "spyre":
        cache.index_copy_(0, index, x)
    else:
        cache.view(torch.uint8).index_copy_(0, index, x.view(torch.uint8))


def _k_blocks(blocks, scale):
    """fp16 ``[n, 64, n_kv, Dh]`` (position-major) -> QFP8WT ``[n, n_kv, Dh, 64]``."""
    return _quantize(blocks.permute(0, 2, 3, 1).contiguous(), scale, kernel=True)


def store_fp8_kv(attn, k, v, key_cache, value_cache, cache_index):
    """Quantize ``k`` and ``v`` with the layer's scales and scatter them in.

    ``k``/``v`` are fp16 ``[B, n_kv, L, Dh]``; ``cache_index`` comes from
    ``fp8_cache_index``. Prefill (``L > 1``) writes whole K blocks straight
    from ``k``; decode (``L == 1``) adds the token to the staging block and
    rewrites that block.

    On the request's first store the scales are computed from these ``k``/``v``
    and written into the layer's scale tensors; every later store reuses them.
    ``attn._kv_scales_set`` is a plain bool, so Dynamo builds one graph for the
    first store and one for the rest instead of branching at runtime. It is
    flipped after prefill by ``freeze_kv_scales``.

    Writes in place and returns ``(key_cache, value_cache)``, like
    ``kv_cache_update``.
    """
    B, n_kv, L, Dh = k.shape
    k_blocks, k_slot, v_rows = cache_index
    if not attn._kv_scales_set:
        attn._k_scale.copy_(_per_tensor_scale(k, K_RANGE))
        attn._v_scale.copy_(_per_tensor_scale(v, V_RANGE).reshape(1, 1, 1))

    if L == 1:
        staging = attn._k_staging  # [64, B, n_kv, Dh]
        staging.index_copy_(0, k_slot, k.permute(2, 0, 1, 3).contiguous())
        blocks = staging.permute(1, 0, 2, 3)  # [B, 64, n_kv, Dh]
    else:
        blocks = (
            k.reshape(B, n_kv, L // KV_BLOCK, KV_BLOCK, Dh)
            .permute(0, 2, 3, 1, 4)
            .reshape(-1, KV_BLOCK, n_kv, Dh)
        )
    _index_copy(key_cache, k_blocks, _k_blocks(blocks, attn._k_scale))

    v_new = v.permute(0, 2, 1, 3).reshape(B * L, n_kv, -1).contiguous()
    _index_copy(value_cache, v_rows, _quantize(v_new, attn._v_scale, kernel=False))
    return key_cache, value_cache


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def _scaled_bmm(q_fp8, k_fp8, out_dtype):
    """Raw ``q_fp8 @ k_fp8`` — torch-spyre#5158's op on device, scales applied by
    the caller. The raw product exceeds fp16's range; the device result is DL16,
    which holds it."""
    if q_fp8.device.type == "spyre":
        if not hasattr(torch.ops.spyre, "scaled_bmm"):
            raise RuntimeError(
                "FP8 KV cache needs torch.ops.spyre.scaled_bmm (torch-spyre#5158)"
            )
        return torch.ops.spyre.scaled_bmm(q_fp8, k_fp8, out_dtype=out_dtype)
    return q_fp8.float() @ k_fp8.float()  # FP32, standing in for DL16


def _dequantize(x_fp8, scale):
    if x_fp8.device.type == "spyre":
        return torch.ops.spyre.dequantize_fp8_with_scale(x_fp8, scale)
    return (x_fp8.float() * scale.float()).to(scale.dtype)


def fp8_attention(attn, q, key_cache, value_cache, attn_mask):
    """``softmax(Q · Kᵀ · scale + mask) · V`` with ``Q · Kᵀ`` in FP8.

    ``q`` fp16 ``[B, n_q, L, Dh]``; caches as in the module docstring;
    ``attn_mask`` ``[B, 1 or n_q, L, M]``. Returns fp16 ``[B, n_q, L, Dh]``, the
    shape ``F.scaled_dot_product_attention`` returned, so the o_proj tail is
    unchanged.
    """
    B, n_q, L, Dh = q.shape
    n_kv = key_cache.shape[1]
    n_blocks = key_cache.shape[0] // B
    M = n_blocks * KV_BLOCK
    gL = n_q // n_kv * L

    # GQA fold: [B, n_kv, g*L, Dh].
    q_f = q.reshape(B, n_kv, gL, Dh)
    # Quantize the folded Q once, then repeat the FP8 rows over the blocks.
    # Neither the per-token scale nor quantize may read a repeated (broadcast)
    # view on device: each failed in dbo-opt. The scale is applied to the
    # scores instead, once they are back in [B, n_q, L, M].
    if q_f.device.type == "spyre":
        s_f = torch.ops.spyre.quantscalepertokenfp8(q_f, FP8_MAX)
    else:
        s_f = (q_f.abs().amax(dim=-1, keepdim=True) / FP8_MAX).clamp(min=1e-4)
    q_fp8 = _quantize(q_f, s_f, kernel=False)  # QFP8CH, mat1

    # One copy per K block: [B*NB, n_kv, g*L, Dh].
    if B == 1:
        # A plain 4D broadcast of the size-1 batch dim, without the general
        # form's 5D unsqueeze -> expand -> reshape view.
        q_fp8 = q_fp8.expand(n_blocks, n_kv, gL, Dh)
    else:
        # TODO: B > 1 repeats through a 5D view; untested on device.
        q_fp8 = (
            q_fp8.unsqueeze(1)
            .expand(B, n_blocks, n_kv, gL, Dh)
            .reshape(B * n_blocks, n_kv, gL, Dh)
        )

    raw = _scaled_bmm(q_fp8, key_cache, out_dtype=q.dtype)  # [B*NB, n_kv, g*L, 64]
    # Scale in the block domain. With B == 1 the folded Q scale
    # [1, n_kv, g*L, 1] broadcasts over the blocks as is.
    if B == 1:
        s_q = s_f
    else:
        # TODO: B > 1 repeats the scale through a 5D view; untested on device.
        s_q = (
            s_f.unsqueeze(1)
            .expand(B, n_blocks, n_kv, gL, 1)
            .reshape(B * n_blocks, n_kv, gL, 1)
        )
    scores = (raw * s_q * attn._k_scale * attn.scaling).to(q.dtype)
    v = _dequantize(value_cache, attn._v_scale)  # fp16 [B*M, n_kv, Dh]

    if B == 1 and L == 1:
        return _decode_in_blocks(scores, v, attn_mask, n_q)

    # Blocks back into positions; heads are KV-major, so (kv, g) -> q head.
    scores = scores.reshape(B, n_blocks, n_kv, gL, KV_BLOCK).permute(0, 2, 3, 1, 4)
    scores = scores.reshape(B, n_q, L, M)
    if attn_mask is not None:
        scores = scores + attn_mask.clamp(min=MASK_FLOOR)
    probs = torch.softmax(scores, dim=-1)

    v = v.reshape(B, M, n_kv, -1).permute(0, 2, 1, 3)  # [B, n_kv, M, Dh]
    out = torch.matmul(probs.reshape(B, n_kv, gL, M), v)  # DL16
    return out.reshape(B, n_q, L, -1)


def _decode_in_blocks(scores, v, attn_mask, n_q):
    """Decode (B == 1, L == 1) softmax and A·V without leaving the block domain.

    ``scores`` ``[NB, n_kv, g, 64]`` (scaled); ``v`` fp16 ``[M, n_kv, Dh]``;
    ``attn_mask`` ``[1, 1 or n_q, 1, M]``. Returns ``[1, n_q, 1, Dh]``.

    Exactly ``softmax`` over all ``M = NB*64`` positions followed by ``A·V``,
    but the scores are never permuted back to ``[1, n_q, 1, M]``: at L == 1 any
    multi-input op that reads that permuted view has no device layout (the
    view decomposes into 6D). The max and the sum are taken over the 64
    positions and then over the blocks, one dim at a time, and ``A·V`` runs per
    block and is summed over the blocks.
    """
    n_blocks, n_kv = scores.shape[:2]
    if attn_mask is not None:
        # Every head shares the decode mask: head 0's row, reshaped into blocks.
        mask = attn_mask[:, :1].clamp(min=MASK_FLOOR)  # [1, 1, 1, M]
        scores = scores + mask.reshape(n_blocks, 1, 1, KV_BLOCK)
    m = scores.amax(dim=-1, keepdim=True).amax(dim=0, keepdim=True)  # [1, n_kv, g, 1]
    e = torch.exp(scores - m)
    denom = e.sum(dim=-1, keepdim=True).sum(dim=0, keepdim=True)  # [1, n_kv, g, 1]

    # [M, n_kv, Dh] -> [NB, n_kv, 64, Dh]
    v_blk = v.reshape(n_blocks, KV_BLOCK, n_kv, -1).permute(0, 2, 1, 3)
    out = torch.matmul(e, v_blk).sum(dim=0, keepdim=True) / denom  # [1, n_kv, g, Dh]
    return out.reshape(1, n_q, 1, -1)
