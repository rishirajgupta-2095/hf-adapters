# Copyright 2026 The Torch-Spyre Authors.
# SPDX-License-Identifier: Apache-2.0

"""generate() with the FP8 KV cache on CPU: chunked prefill and whole caches.

A tiny Granite with ``_spyre_fp8_kv_cache`` set runs hf_common.generate end to
end, so the parts the unit tests in test_fp8_kv_cache_cpu.py cannot reach are
covered: the prefill chunk loop handing the FP8 caches over whole, the prefill
mask spanning every cache position, and decode after a chunked prefill.
"""

import importlib

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, GraniteConfig

from hf_adapters import hf_common

PROMPT_LEN, NEW_TOKENS = 139, 4


@pytest.fixture
def granite(monkeypatch):
    torch.manual_seed(23)
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    config = GraniteConfig(
        vocab_size=129,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        logits_scaling=2.0,
        pad_token_id=0,
        eos_token_id=None,
    )
    adapter = importlib.import_module("hf_adapters.hf_granite")
    model = AutoModelForCausalLM.from_config(config).eval().half()
    adapter.prepare_for_spyre(model)
    return model, adapter


def _generate(model, adapter, ids, *, fp8_kv, chunk_size, record=None):
    model._spyre_fp8_kv_cache = fp8_kv
    model._spyre_prefill_chunk_size = chunk_size
    block = model._spyre_compiled_blocks[0]
    if record is not None:

        def recording_block(hidden, freqs, attn_mask, key_cache, value_cache, *args):
            record.append(
                (hidden.shape[1], attn_mask.shape[-1], tuple(key_cache.shape))
            )
            return block(hidden, freqs, attn_mask, key_cache, value_cache, *args)

        model._spyre_compiled_blocks[0] = recording_block
    try:
        return hf_common.generate(
            adapter._run_forward,
            model,
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=NEW_TOKENS,
            do_sample=False,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_logits=True,
        )
    finally:
        model._spyre_compiled_blocks[0] = block


def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


@pytest.mark.parametrize("b", [1, 2])
@torch.no_grad()
def test_chunked_prefill_with_fp8_kv(granite, b):
    """Chunked prefill gets whole FP8 caches and a full-width mask, and its
    logits match one-shot FP8 prefill and track the fp16 KV cache."""
    model, adapter = granite
    ids = torch.randint(1, 129, (b, PROMPT_LEN))

    # fp16 KV first: attach_fp8_kv leaves its hooks on the attention modules,
    # as the FP8_KV_CACHE choice is fixed for a model once it is prepared.
    fp16_kv = _generate(model, adapter, ids, fp8_kv=False, chunk_size=64)
    shapes = []
    chunked = _generate(model, adapter, ids, fp8_kv=True, chunk_size=64, record=shapes)
    one_shot = _generate(model, adapter, ids, fp8_kv=True, chunk_size=None)

    # 139 tokens pad to 192: three 64-row chunks, then one row per decode step.
    prefill = [s for s in shapes if s[0] > 1]
    assert [s[0] for s in prefill] == [64, 64, 64]
    # FP8 K blocks [B*NB, n_kv, Dh, 64], handed over whole, never position-sliced.
    n_k_blocks = shapes[0][2][0]
    assert all(s[2][0] == n_k_blocks and s[2][-1] == 64 for s in shapes)
    m = n_k_blocks // b * 64  # cache length
    assert all(s[1] == m for s in prefill), "prefill mask must span the whole cache"

    # Chunked and one-shot FP8 prefill store the same bytes (static scales), so
    # they decode the same tokens.
    assert torch.equal(chunked.sequences, one_shot.sequences)
    for got, want in zip(chunked.logits, one_shot.logits):
        assert _cos(got, want) > 0.9999
    # Against the fp16 KV cache only the prefill logits are compared: a free
    # running decode can take the other side of a near-tie and diverge (here a
    # top-2 margin of 0.0015 in a random tiny model).
    assert _cos(chunked.logits[0], fp16_kv.logits[0]) > 0.999
    assert torch.isfinite(torch.stack(chunked.logits)).all()


@pytest.mark.parametrize("fp8_kv", [True, False])
def test_granite_prefill_chunk_override_is_fp8_kv_only(monkeypatch, fp8_kv):
    """FP8_KV_PREFILL_CHUNK sets the chunk only with the FP8 KV cache; the fp16
    paths keep SDPA's 512."""
    monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
    monkeypatch.setenv("FP8_KV_PREFILL_CHUNK", "128")
    config = GraniteConfig(
        vocab_size=129,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        logits_scaling=2.0,
        pad_token_id=0,
    )
    model = AutoModelForCausalLM.from_config(config).eval().half()
    # Set by prepare_for_spyre for an FP8 checkpoint with FP8_KV_CACHE=1.
    model._spyre_fp8_kv_cache = fp8_kv
    importlib.import_module("hf_adapters.hf_granite").prepare_for_spyre(model)
    assert model._spyre_prefill_chunk_size == (128 if fp8_kv else 512)


@pytest.mark.parametrize("group", ["1", "2"])
@torch.no_grad()
def test_grouped_decode_in_generate(granite, monkeypatch, group):
    """generate() with decode split into groups of K blocks decodes the same
    tokens as one group over the whole cache."""
    model, adapter = granite
    ids = torch.randint(1, 129, (1, PROMPT_LEN))
    monkeypatch.setenv("FP8_KV_DECODE_GROUP", "100000")
    whole = _generate(model, adapter, ids, fp8_kv=True, chunk_size=64)
    monkeypatch.setenv("FP8_KV_DECODE_GROUP", group)
    grouped = _generate(model, adapter, ids, fp8_kv=True, chunk_size=64)
    assert torch.equal(grouped.sequences, whole.sequences)
    for got, want in zip(grouped.logits, whole.logits):
        assert _cos(got, want) > 0.9999
