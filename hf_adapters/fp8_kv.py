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

Scheme (all scales static, see "Scales" below)::

    Q   x / q_scale, clamp(+-448), spyre.qfp8ch                         -> QFP8CH
    K   quantize_weight_fp8_with_scale(x, k_scale), stored E4M3          -> QFP8WT
    V   quantize_fp8_with_scale(x, v_scale), stored E4M3                 -> QFP8CH
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

Scales: static, one per layer for each of Q, K and V, as in vLLM and
spyre-inference. They come from the checkpoint (``q_scale`` / ``k_scale`` /
``v_scale`` on the attention module) when it ships them, else 1.0; the FP8
Granite checkpoints ship none (``kv_cache_scheme: null``), so all three are 1.0
there. The clamp to +-448 is explicit (built into the quantize ops for K and V,
written out for Q). A static scale is what makes chunked prefill and the decode
block rewrite exact: every chunk and every rewrite quantizes with the same
scale, so positions written earlier keep their bytes. Q is quantized with
one-input ops only (a multiply by a constant, clamp, ``qfp8ch``): a two-input op
on RoPE's output fails layout propagation at L == 1 (see ``_quantize_q``).

GQA: ``scaled_bmm`` needs matching batch dims, and Granite has 32 query heads
against 8 KV heads. Instead of repeating K and V four times over (FMS's
``repeat_interleave``), the 4 query heads that share a KV head are folded into
the row dim: ``[B, 32, L, Dh] -> [B, 8, 4L, Dh]``. Exact, because those heads
all multiply the same K, and a single reshape since the heads are KV-major. The
folded Q is then repeated over the ``NB`` blocks so one ``scaled_bmm`` scores
every block, and softmax runs over all ``M`` positions at once. Decode
(B == 1) does the same per group of blocks and combines the groups with an
online softmax, so no tensor spans the whole cache (``_decode_in_groups``).

Nothing here is imported by ``hf_common`` at module load: ``hf_common`` must
not import from its own package. ``attach_fp8_kv`` hands these functions to the
attention modules instead.
"""

from __future__ import annotations

import functools
import os

import torch

from .fp8_linear import FP8_DTYPE, FP8_MAX

KV_BLOCK = 64  # positions per K block: the N extent of one QFP8WT stick
DEFAULT_SCALE = 1.0  # static Q/K/V scale when the checkpoint ships none
PREFILL_CHUNK_ENV = "FP8_KV_PREFILL_CHUNK"
DECODE_GROUP_ENV = "FP8_KV_DECODE_GROUP"
# Decode works on groups of K blocks whose score tensor stays this size or
# smaller (see decode_group_blocks): 1 MiB, under the 1,625,344 B per-core LX
# budget torch-spyre plans against.
DECODE_GROUP_SCORE_BYTES = 1 << 20
SCALE_DTYPE = torch.float16  # dequantize_fp8_with_scale requires an fp16 scale
STAGING_DTYPE = torch.float16  # FP8Linear outputs fp16

# Finite stand-in for ``-inf`` in the additive mask. Explicit softmax cannot
# consume an all-``-inf`` row, which block-padded prefill produces for
# left-padding queries (``F.scaled_dot_product_attention`` handled that
# internally). exp() of this underflows to exactly 0 in fp16, so real rows are
# unaffected; a fully masked row gets uniform weights and a finite output that
# is never read.
MASK_FLOOR = -1.0e4


def prefill_chunk_size(default: int) -> int:
    """Prefill chunk size with the FP8 KV cache: ``$FP8_KV_PREFILL_CHUNK`` or ``default``.

    Each prefill chunk scores its queries against every K block of the cache in
    one ``scaled_bmm``, so the score tensor ``[NB, n_kv, g * chunk, 64]`` grows
    with cache length times chunk size. Spyre caps one tensor's per-core span at
    256 MB, which Granite 3.3 8B (n_kv 8, g 4, fp16 scores) exceeds at chunk 512
    from an 8K prompt on (136 blocks: 272 MB). Until prefill attention walks the
    blocks with an online softmax, a smaller chunk keeps it under the limit:

        prompt   K blocks   largest chunk that fits
        4K        72        512
        8K       136        256
        16K      264        128
        32K      520         64

    Decode is not affected: its scores are [NB, n_kv, g, 64]. The chunk must be
    a positive multiple of the 64-position K block.
    """
    raw = os.environ.get(PREFILL_CHUNK_ENV)
    if raw is None or raw.strip() == "":
        return default
    try:
        chunk = int(raw)
    except ValueError:
        raise ValueError(f"{PREFILL_CHUNK_ENV}={raw!r} is not an integer") from None
    if chunk <= 0 or chunk % KV_BLOCK:
        raise ValueError(
            f"{PREFILL_CHUNK_ENV}={chunk} must be a positive multiple of {KV_BLOCK}"
        )
    return chunk


def decode_group_blocks(n_kv: int, g: int) -> int:
    """K blocks per decode group: ``$FP8_KV_DECODE_GROUP``, else what keeps one
    group's scores ``[G, n_kv, g, 64]`` (fp16) within ``DECODE_GROUP_SCORE_BYTES``.

    Decode scores every K block of the cache. Held as one ``[NB, n_kv, g, 64]``
    tensor, they fit a core's LX scratchpad only up to a size: for Granite 3.3
    8B (n_kv 8, g 4) that is 396 blocks (1,622,016 B against a 1,625,344 B
    budget); at 397 the fused decode kernel takes another plan and runs 5.6x
    slower (measured: 1.80 -> 10.0 ms per layer). Working on groups of blocks
    and combining them with an online softmax keeps every group's scores small
    at any cache length. Granite: 256 blocks per group (1 MiB).
    """
    raw = os.environ.get(DECODE_GROUP_ENV)
    if raw is not None and raw.strip() != "":
        try:
            group = int(raw)
        except ValueError:
            raise ValueError(f"{DECODE_GROUP_ENV}={raw!r} is not an integer") from None
        if group <= 0:
            raise ValueError(f"{DECODE_GROUP_ENV}={group} must be positive")
        return group
    return max(1, DECODE_GROUP_SCORE_BYTES // (n_kv * g * KV_BLOCK * 2))


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


def _allocate_v_cache(n_rows, n_kv, head_dim, device):
    """Zeroed ``[n_rows, n_kv, head_dim]`` V cache with the row dim outermost on device.

    V rows are written by ``index_copy_`` along dim 0, and an indirect scatter
    needs the indexed dim at device position 0: otherwise torch-spyre writes the
    wrong rows, silently (torch-spyre#3705; see ``hf_common.kv_cache_update``).
    The default layout of a ``[128, 8, 128]`` FP8 tensor puts the heads first
    (device_size ``[8, 1, 128, 128]``, measured), so the layout is pinned here,
    as ``hf_common._cache_position_first_stl`` does for the fp16 caches.
    """
    shape = (n_rows, n_kv, head_dim)
    if torch.device(device).type != "spyre":
        return _fp8_zeros(shape, device)
    # Start the runtime first: allocating with device_layout= needs a live
    # RuntimeContext (as in hf_common._cache_position_first_stl).
    torch.empty(1, device=device)
    from torch_spyre._C import (  # type: ignore[import-not-found]
        SpyreTensorLayout,
        get_device_dtype,
    )

    eps = SpyreTensorLayout(list(shape), FP8_DTYPE).elems_per_stick()
    sticks = (head_dim + eps - 1) // eps
    stl = SpyreTensorLayout(
        device_size=[n_rows, n_kv, sticks, eps],
        stride_map=[n_kv * head_dim, head_dim, eps, 1],
        device_dtype=get_device_dtype(FP8_DTYPE),
    )
    cache: torch.Tensor = torch.empty(  # type: ignore[call-overload]
        shape, device=torch.device(device), device_layout=stl, dtype=FP8_DTYPE
    )
    # The device has no FP8 fill; a copy from the host keeps the pinned layout.
    cache.copy_(torch.zeros(shape, dtype=torch.uint8).view(FP8_DTYPE))
    return cache


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
    value_caches = [
        _allocate_v_cache(batch_size * max_cache_len, n_kv, vhd, device)
        for (n_kv, _hd, vhd) in shapes
    ]

    n = attach_fp8_kv(model, shapes, batch_size, device, n_blocks=n_blocks)
    if n != len(shapes):
        raise RuntimeError(
            f"FP8 KV cache: wired {n} attention module(s) but allocated "
            f"{len(shapes)} cache(s) — the blocks must expose `self_attn` with "
            "`pre_attn`"
        )
    model._spyre_cache_index_fn = functools.partial(
        fp8_cache_index, batch_size, max_cache_len
    )
    # The K cache's block layout has no position slice, so prefill (chunked or
    # not) is handed the whole caches and a mask over every cache position.
    model._spyre_prefill_full_cache = True
    if batch_size == 1:
        model._spyre_decode_mask_fn = block_decode_mask
    return key_caches, value_caches


def _static_scale(attn, name) -> float:
    """The layer's static ``name`` scale from the checkpoint, else 1.0.

    ``StandardGQAAttention`` keeps any ``q_scale`` / ``k_scale`` / ``v_scale``
    the checkpoint put on the attention module in ``_checkpoint_scales``.
    """
    scale = getattr(attn, "_checkpoint_scales", {}).get(name)
    if scale is None:
        return DEFAULT_SCALE
    if scale.numel() != 1:
        # TODO: per-KV-head static scales broadcast as [1, n_kv, 1, 1] on K and
        # the scores and [1, n_kv, 1] on V; Q would need a two-input quantize.
        raise NotImplementedError(
            f"FP8 KV cache: per-KV-head {name} {tuple(scale.shape)} is not "
            "supported yet; only one static scale per layer"
        )
    value = float(scale.detach().to("cpu").reshape(()))
    if not value > 0.0:
        raise ValueError(f"FP8 KV cache: {name} must be positive, got {value}")
    return value


def _decode_groups(n_blocks, group, device):
    """``[(block_index, row_index), ...]`` for each decode group of K blocks, or
    ``[]`` when one group covers the whole cache (read it whole, no gather).

    The indices are built on the host and moved once: decode gathers each
    group's K blocks, V rows and mask blocks with ``index_select`` on dim 0.
    """
    if n_blocks <= group:
        return []
    groups = []
    for b0 in range(0, n_blocks, group):
        b1 = min(b0 + group, n_blocks)
        blk = torch.arange(b0, b1, dtype=torch.long)
        row = torch.arange(b0 * KV_BLOCK, b1 * KV_BLOCK, dtype=torch.long)
        groups.append((blk.to(device), row.to(device)))
    return groups


def attach_fp8_kv(model, shapes, batch_size, device, n_blocks=None) -> int:
    """Turn the FP8 KV path on in every attention module, with fresh state.

    Sets the layer's static scales (``_fp8_scales`` as floats for Q and the
    score rescale; ``_k_scale`` / ``_v_scale`` as fp16 tensors for the K/V
    quantize and the V dequantize), a zeroed K staging block, and for B == 1
    the decode groups' gather indices (``_decode_groups``). Returns the number
    of modules wired up.
    """
    attns = [
        attn
        for module in model.modules()
        if (attn := getattr(module, "self_attn", None)) is not None
        and hasattr(attn, "pre_attn")
    ]
    for attn, (n_kv, hd, _vhd) in zip(attns, shapes):
        q_s, k_s, v_s = (
            _static_scale(attn, n) for n in ("q_scale", "k_scale", "v_scale")
        )
        attn._fp8_scales = (q_s, k_s, v_s)
        # Built on the host, then moved: a contiguous H2D copy.
        attn._k_scale = torch.full((1, 1, 1, 1), k_s, dtype=SCALE_DTYPE).to(device)
        attn._v_scale = torch.full((1, 1, 1), v_s, dtype=SCALE_DTYPE).to(device)
        attn._k_staging = torch.zeros(
            (KV_BLOCK, batch_size, n_kv, hd), dtype=STAGING_DTYPE, device=device
        )
        attn._decode_groups = None
        if batch_size == 1 and n_blocks is not None:
            g = attn.q_proj.out_features // attn.head_dim // n_kv
            attn._decode_groups = _decode_groups(
                n_blocks, decode_group_blocks(n_kv, g), device
            )
        attn._fp8_store = store_fp8_kv
        attn._fp8_attention = fp8_attention
    return len(attns)


def block_decode_mask(mask):
    """Host-side decode mask ``[1, 1 or n_q, 1, M]`` -> ``[NB, 1, 1, 64]``.

    The B == 1 decode path (``_decode_in_groups``) adds the mask, a group of
    blocks at a time, to scores shaped ``[G, n_kv, g, 64]``. Splitting ``M`` into ``NB x 64`` would reshape
    the stick dim, which a view on device may not do, so it is done here, on
    the CPU mask, before it is moved. Every head shares the decode mask, so
    head 0's row is enough.
    """
    return mask[:, :1].reshape(-1, 1, 1, KV_BLOCK).contiguous()


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
    """Quantize ``k`` and ``v`` with the layer's static scales and scatter them in.

    ``k``/``v`` are fp16 ``[B, n_kv, L, Dh]``; ``cache_index`` comes from
    ``fp8_cache_index``. Prefill (``L > 1``, the whole prompt or one chunk of
    it) writes whole K blocks straight from ``k``; decode (``L == 1``) adds the
    token to the staging block and rewrites that block.

    Writes in place and returns ``(key_cache, value_cache)``, like
    ``kv_cache_update``.
    """
    B, n_kv, L, Dh = k.shape
    k_blocks, k_slot, v_rows = cache_index

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


def _quantize_q(q_f, scale: float):
    """Q -> E4M3 QFP8CH with a static scale: ``q / scale``, clamp, convert.

    One-input ops only (a multiply by a constant, clamp, ``qfp8ch``). At
    L == 1, Q (RoPE's output seen through flatten(3)/transpose(1, 2), plus the
    GQA fold) composes into a 6D view, and any two-input pointwise op on it --
    such as quantize_fp8_with_scale's ``q * (1/scale)`` with a scale tensor --
    fails layout propagation (_project_pointwise_dim_order adds one leading
    axis too many when the output order ends in the sparse-stick marker;
    AssertionError in _is_supported_pointwise_layout).

    E4M3 keeps 3 mantissa bits for any |q / scale| in [2^-6, 448], so only
    values outside that range lose precision (clamped above, subnormal below).
    """
    if scale != 1.0:
        q_f = q_f * (1.0 / scale)
    if q_f.device.type == "spyre":
        return torch.ops.spyre.qfp8ch(q_f.clamp(-FP8_MAX, FP8_MAX))
    return q_f.float().clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


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

    q_s, k_s, _v_s = attn._fp8_scales

    # GQA fold: [B, n_kv, g*L, Dh].
    q_f = q.reshape(B, n_kv, gL, Dh)
    # Quantize the folded Q once, then repeat the FP8 rows over the blocks:
    # quantize may not read a repeated (broadcast) view on device (dbo-opt).
    q_fp8 = _quantize_q(q_f, q_s)  # QFP8CH, mat1

    if B == 1 and L == 1:
        return _decode_in_groups(attn, q_fp8, key_cache, value_cache, attn_mask, n_q)

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
    # Static scales: one constant rescale, in the block domain.
    scores = (raw * (q_s * k_s * attn.scaling)).to(q.dtype)
    v = _dequantize(value_cache, attn._v_scale)  # fp16 [B*M, n_kv, Dh]

    # Blocks back into positions; heads are KV-major, so (kv, g) -> q head.
    scores = scores.reshape(B, n_blocks, n_kv, gL, KV_BLOCK).permute(0, 2, 3, 1, 4)
    scores = scores.reshape(B, n_q, L, M)
    if attn_mask is not None:
        scores = scores + attn_mask.clamp(min=MASK_FLOOR)
    probs = torch.softmax(scores, dim=-1)

    v = v.reshape(B, M, n_kv, -1).permute(0, 2, 1, 3)  # [B, n_kv, M, Dh]
    out = torch.matmul(probs.reshape(B, n_kv, gL, M), v)  # DL16
    return out.reshape(B, n_q, L, -1)


def _gather(x, index):
    """``x.index_select(0, index)``; CPU has no FP8 kernel, so FP8 goes as bytes."""
    if x.device.type != "spyre" and x.dtype == FP8_DTYPE:
        return x.view(torch.uint8).index_select(0, index).view(FP8_DTYPE)
    return x.index_select(0, index)


def _decode_in_groups(attn, q_fp8, key_cache, value_cache, attn_mask, n_q):
    """Decode (B == 1, L == 1) attention over groups of K blocks, online softmax.

    ``q_fp8`` ``[1, n_kv, g, Dh]`` (QFP8CH, folded); caches as in the module
    docstring; ``attn_mask`` ``[NB, 1, 1, 64]``, already in blocks
    (``block_decode_mask``). Returns ``[1, n_q, 1, Dh]``.

    Exactly ``softmax`` over all ``M = NB*64`` positions followed by ``A·V``.
    The K blocks are taken ``decode_group_blocks`` at a time (gathered with
    ``index_select``, never sliced; one group reads the caches whole); each group runs
    Q·Kᵀ on the FP8 engine, its mask, its max and sum over 64 positions then
    over its blocks, and ``A·V`` on just its V rows (dequantized per group),
    giving a running ``(max, sum, output)``. The groups are then combined:
    ``m = max m_g``, ``l = sum l_g * exp(m_g - m)``,
    ``out = sum o_g * exp(m_g - m) / l``. No tensor ever spans all NB blocks,
    so the per-core working set stays bounded (see ``decode_group_blocks``).

    Everything stays in the block domain: the scores are never permuted back to
    ``[1, n_q, 1, M]``, which at L == 1 is a view (6D) with no device layout.
    """
    _, n_kv, g, Dh = q_fp8.shape
    n_blocks = key_cache.shape[0]
    q_s, k_s, _v_s = attn._fp8_scales
    scale = q_s * k_s * attn.scaling
    groups = getattr(attn, "_decode_groups", None)
    if groups is None:  # not attached (CPU tests): build them here
        groups = _decode_groups(
            n_blocks, decode_group_blocks(n_kv, g), key_cache.device
        )

    # Never slice the caches here: the store has just written them in place in
    # this same compiled graph, and on Spyre a view of them (even a full-range
    # slice) read the pre-store contents, so decode lost its newest tokens.
    # One group reads the tensors whole; several gather with index_select.
    if not groups:
        inputs = [(key_cache, value_cache, attn_mask, n_blocks)]
    else:
        inputs = [
            (
                _gather(key_cache, blk),
                _gather(value_cache, row),
                None if attn_mask is None else _gather(attn_mask, blk),
                blk.shape[0],
            )
            for blk, row in groups
        ]

    parts = []
    for k_g, v_g_fp8, mask_g, nb in inputs:
        raw = _scaled_bmm(  # [nb, n_kv, g, 64]
            q_fp8.expand(nb, n_kv, g, Dh), k_g, out_dtype=torch.float16
        )
        s = (raw * scale).to(torch.float16)
        if mask_g is not None:
            s = s + mask_g.clamp(min=MASK_FLOOR)
        m = s.amax(dim=-1, keepdim=True).amax(dim=0, keepdim=True)  # [1, n_kv, g, 1]
        e = torch.exp(s - m)
        l_sum = e.sum(dim=-1, keepdim=True).sum(dim=0, keepdim=True)  # [1, n_kv, g, 1]
        # This group's V rows: [nb*64, n_kv, Dh] -> [nb, n_kv, 64, Dh].
        v = _dequantize(v_g_fp8, attn._v_scale)
        v_blk = v.reshape(nb, KV_BLOCK, n_kv, -1).permute(0, 2, 1, 3)
        o = torch.matmul(e, v_blk).sum(dim=0, keepdim=True)  # [1, n_kv, g, Dh]
        parts.append((m, l_sum, o))

    m_all = parts[0][0]
    for m_g, _, _ in parts[1:]:
        m_all = torch.maximum(m_all, m_g)
    l_all = o_all = None
    for m_g, l_g, o_g in parts:
        w = torch.exp(m_g - m_all)  # 1 for a group holding the overall max
        l_all = l_g * w if l_all is None else l_all + l_g * w
        o_all = o_g * w if o_all is None else o_all + o_g * w
    return (o_all / l_all).reshape(1, n_q, 1, -1)


# ---------------------------------------------------------------------------
# Q·Kᵀ in FP8 on the fp16 KV cache (FP8_QK_ATTN=1)
# ---------------------------------------------------------------------------


def _quantize_k_unit(kt):
    """fp16 Kᵀ ``[B, n_kv, Dh, M]`` -> E4M3 QFP8WT with a unit scale.

    Clamp then convert: one-input ops only. K after RoPE stays well inside
    E4M3's +-448, and E4M3 keeps 3 mantissa bits anywhere in [2^-6, 448].
    """
    if kt.device.type == "spyre":
        return torch.ops.spyre.qfp8wt(kt.clamp(-FP8_MAX, FP8_MAX))
    return kt.float().clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)


def fp8_qk_attention(attn, q, key_cache, value_cache, attn_mask):
    """``softmax(Q · Kᵀ · scale + mask) · V`` with only ``Q · Kᵀ`` in FP8.

    For the ordinary fp16 KV cache (``[B, n_kv, M, Dh]``, written by
    ``kv_cache_update``): nothing is stored in FP8. Each call quantizes Q and
    the whole K cache, transposed, with unit scales, then runs one
    ``scaled_bmm`` over all positions; softmax and ``A · V`` stay fp16 on the
    V cache as is. Batch dims match, so there is no per-tile repeat.
    Returns fp16 ``[B, n_q, L, Dh]``, like SDPA.
    """
    B, n_q, L, Dh = q.shape
    n_kv, M = key_cache.shape[1], key_cache.shape[2]
    gL = n_q // n_kv * L

    q_f = q.reshape(B, n_kv, gL, Dh)  # GQA fold
    q_fp8 = _quantize_q(q_f, DEFAULT_SCALE)  # QFP8CH, mat1

    k_fp8 = _quantize_k_unit(key_cache.transpose(-2, -1).contiguous())  # mat2
    raw = _scaled_bmm(q_fp8, k_fp8, out_dtype=q.dtype)  # [B, n_kv, g*L, M]
    scores = raw * attn.scaling
    scores = scores.to(q.dtype).reshape(B, n_q, L, M)  # (kv, g) -> q head
    if attn_mask is not None:
        scores = scores + attn_mask.clamp(min=MASK_FLOOR)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs.reshape(B, n_kv, gL, M), value_cache)
    return out.reshape(B, n_q, L, Dh)
