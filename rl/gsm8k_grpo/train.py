"""GRPO on GSM8K with TRL + vLLM (colocated) and a LoRA adapter.

Reward: 1 if the last \\boxed{} matches the GSM8K answer, else 0.
Usage: python train.py --max_steps 30 --output_dir /content/smoke
"""
import argparse

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

from common import MODEL, gold_answer, is_correct, to_prompt


def correctness_reward(completions, answer, **kwargs):
    return [1.0 if is_correct(c[0]["content"], a) else 0.0 for c, a in zip(completions, answer)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_steps", type=int, default=30)
    p.add_argument("--output_dir", default="/content/grpo_gsm8k")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--save_steps", type=int, default=50)
    # Completions per backward pass. The full-vocab logits are the memory peak
    # (batch x ~700 tokens x 152k vocab), so keep this small on 24 GB GPUs.
    p.add_argument("--micro_batch", type=int, default=8)
    args = p.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0)
    ds = ds.map(lambda r: {"prompt": to_prompt(r["question"]), "answer": gold_answer(r["answer"])},
                remove_columns=["question"])

    config = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5,
        # 16 prompts x 8 samples = 128 completions per optimizer step.
        num_generations=8,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=128 // args.micro_batch,
        max_completion_length=512,
        temperature=1.0,
        beta=0.0,
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.35,
        logging_steps=1,
        log_completions=True,
        num_completions_to_print=1,
        save_steps=args.save_steps,
        report_to="none",
        seed=0,
    )
    lora = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules="all-linear",
                      task_type="CAUSAL_LM")
    trainer = GRPOTrainer(model=MODEL, reward_funcs=correctness_reward, args=config,
                          train_dataset=ds, peft_config=lora)
    trainer.train()
    trainer.save_model(f"{args.output_dir}/final")


if __name__ == "__main__":
    main()
