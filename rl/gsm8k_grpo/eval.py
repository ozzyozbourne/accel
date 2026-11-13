"""Greedy accuracy on the GSM8K test set, for the base model or base + LoRA adapter.

Usage: python eval.py [--adapter DIR] [--n 1319]
"""
import argparse
import json

from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from common import MODEL, extract, gold_answer, to_prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter")
    p.add_argument("--n", type=int, default=1319)  # full test set
    p.add_argument("--out", default=None)
    args = p.parse_args()

    test = load_dataset("openai/gsm8k", "main", split="test").select(range(args.n))
    llm = LLM(MODEL, enable_lora=bool(args.adapter), max_lora_rank=32, gpu_memory_utilization=0.8,
              max_model_len=2048, seed=0)
    lora = LoRARequest("grpo", 1, args.adapter) if args.adapter else None
    outs = llm.chat([to_prompt(q) for q in test["question"]],
                    SamplingParams(temperature=0.0, max_tokens=512), lora_request=lora)

    texts = [o.outputs[0].text for o in outs]
    golds = [gold_answer(a) for a in test["answer"]]
    preds = [extract(t) for t in texts]
    correct = sum(p == g for p, g in zip(preds, golds))
    no_box = sum(p is None for p in preds)
    tokens = sum(len(o.outputs[0].token_ids) for o in outs) / len(outs)
    name = args.adapter or "base"
    print(f"RESULT {name}: accuracy {correct}/{len(golds)} = {correct / len(golds):.1%}, "
          f"missing \\boxed: {no_box}, mean length {tokens:.0f} tokens")
    if args.out:
        with open(args.out, "w") as f:
            for q, t, p_, g in zip(test["question"], texts, preds, golds):
                f.write(json.dumps({"question": q, "completion": t, "pred": p_, "gold": g}) + "\n")


if __name__ == "__main__":
    main()
