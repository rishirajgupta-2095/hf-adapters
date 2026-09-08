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

"""
HuggingFace Transformers adapter for Granite 3.3 models on Spyre.

Usage::

    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer

    model = AutoSpyreModelForCausalLM.from_pretrained(
        "/path/to/granite-3.3-8b-instruct")
    tokenizer = AutoTokenizer.from_pretrained("/path/to/granite-3.3-8b-instruct")
    outputs = model.generate(tokenizer, ["Hello!"], max_new_tokens=32)
"""

from hf_adapters.fp8_linear import swap_linears_to_fp8
from hf_adapters.hf_common import (
    get_backbone,
    pad_lm_head,
    patch_rmsnorm,
    prepare_rope_and_heads,
    prepare_standard_gqa_blocks,
    text_config,
)


def _run_backbone_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    """Granite 3.3 backbone: embedding * multiplier, blocks, norm."""
    backbone = get_backbone(model)
    h = backbone.embed_tokens(input_ids)
    h = h * backbone.embedding_multiplier

    selected_freqs = model._spyre_rope(h, position_ids)

    for i, compiled_block in enumerate(model._spyre_compiled_blocks):
        h, key_caches[i], value_caches[i] = compiled_block(
            h,
            selected_freqs,
            attn_mask,
            key_caches[i],
            value_caches[i],
            cache_index,
        )

    h = backbone.norm(h)
    return h


def _run_forward(
    model,
    input_ids,
    position_ids,
    attn_mask,
    key_caches,
    value_caches,
    cache_index,
):
    """Granite 3.3 causal-LM forward: backbone + head / scaling."""
    h = _run_backbone_forward(
        model,
        input_ids,
        position_ids,
        attn_mask,
        key_caches,
        value_caches,
        cache_index,
    )
    logits = model.lm_head(h)
    return logits / text_config(model.config).logits_scaling


def _prepare_fp8_if_quantized(model):
    """Swap a compressed-tensors checkpoint's Linears for FP8Linear, if present.

    No-op for ordinary fp16/bf16 checkpoints -- keyed on whether any quantized
    (E4M3) ``nn.Linear`` is actually present, which is a more robust signal than
    introspecting ``config.quantization_config`` (whose representation differs
    between a parsed config object and a raw dict). compressed-tensors keeps the
    weights in E4M3 until the first forward, and nothing has forward-ed yet at
    this point, so the check is reliable here.

    Runs BEFORE ``prepare_standard_gqa_blocks`` so the compiled blocks close over
    ``FP8Linear``, and before ``move_model_to_spyre``'s device transfer (which
    calls this function first, then moves weights) so dequantization happens on
    CPU where it is cheap and exact.

    REQUIRES torch-spyre with #4246 (``ed4aa21a``, the SDPA tile-advance fix).
    Without it, SDPA silently corrupts head-tiled reads of a sliced KV block and
    the model returns NaN -- with no error. That fix is NOT on torch-spyre
    ``main`` as of 2026-09-06; branch ``fp8-fix-4179`` has it, plus the
    batchmatmulfp8 shape fix the MLP's N=12800 projections need.

    Validated envelope: prefill only, SEQ_LEN 2-4. Decode (M=1) and realistic
    sequence lengths are untested -- see the deferred-risks note before relying
    on this for generation.
    """
    n_fp8, n_excluded = swap_linears_to_fp8(model)
    if n_fp8 or n_excluded:
        print(
            f"FP8: {n_fp8} module(s) -> FP8Linear, "
            f"{n_excluded} dequantized to fp16 nn.Linear"
        )
        # kv_cache_shapes() reads k_proj.weight.shape[0] // head_dim to infer
        # num_kv_heads, which assumes nn.Linear's [out, in]. FP8Linear stores
        # [in, out], so that read returns in_features and yields the wrong head
        # count. Pin the shapes from config instead -- the same escape hatch
        # Gemma 4 uses for its own per-layer shape quirk.
        cfg = text_config(model.config)
        head_dim = (
            getattr(cfg, "head_dim", None)
            or cfg.hidden_size // cfg.num_attention_heads
        )
        model._spyre_kv_shapes = [
            (cfg.num_key_value_heads, head_dim, head_dim)
            for _ in range(cfg.num_hidden_layers)
        ]


def prepare_for_spyre(model):
    """Apply Spyre adaptations to Granite 3.3 model in-place."""
    from transformers.models.granite.modeling_granite import GraniteRMSNorm

    _prepare_fp8_if_quantized(model)
    prepare_rope_and_heads(model)
    patch_rmsnorm(GraniteRMSNorm)
    pad_lm_head(model)
    model._spyre_compiled_blocks = prepare_standard_gqa_blocks(
        get_backbone(model).layers, True
    )
