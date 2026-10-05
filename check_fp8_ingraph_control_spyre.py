"""Control run: FP8 with the weight quantized INSIDE the compiled graph.

Neither direct E4M3 DMA nor eager prequantization -- the original working
config, and the only FP8 config this torch-spyre tree can run besides direct
DMA. Use it to decide whether wrong output under DIRECT_FP8_LOAD comes from the
DMA'd QFP8WT layout or from something wider on this tree.

Run: HF_ADAPTERS_FP8_DIRECT_LOAD=0 python3 check_fp8_ingraph_control_spyre.py
"""

import sys
import time

import hf_adapters.fp8_linear as fl

# hf_common imports this inside move_model_to_spyre, so the patch takes effect.
# Leaves every FP8Linear weight fp16; forward quantizes it in-graph.
fl.prequantize_fp8_weights = lambda model: 0

from transformers import AutoTokenizer  # noqa: E402

from hf_adapters import AutoSpyreModelForCausalLM  # noqa: E402
from hf_adapters.hf_common import encode_prompts  # noqa: E402

MODEL = "ibm-granite/granite-3.3-8b-instruct-FP8"
PROMPT = "The capital of France is"


def main() -> int:
    assert not fl.DIRECT_FP8_LOAD, "run with HF_ADAPTERS_FP8_DIRECT_LOAD=0"

    t0 = time.time()
    model = AutoSpyreModelForCausalLM.from_pretrained(MODEL)
    print(f"Load time: {time.time() - t0:.1f}s")

    status = fl.fp8_status(model)
    print(f"FP8 status: {status}")
    assert status["n_fp8"] == 240, status
    assert status["n_weight_fp8"] == 0, f"weights should still be fp16: {status}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    encoded = encode_prompts(tokenizer, [PROMPT])
    sequences = model.generate(
        **encoded, max_new_tokens=8, do_sample=False, timing=True
    )
    text = tokenizer.decode(
        sequences[0, encoded["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(f"Output: {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
