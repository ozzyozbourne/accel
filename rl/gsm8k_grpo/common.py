"""Prompt format and answer checking shared by training (the reward) and evaluation."""
import re

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SYSTEM = (
    "Solve the math problem step by step. "
    "Put the final numeric answer in \\boxed{}, for example \\boxed{42}."
)


def to_prompt(question: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]


def gold_answer(gsm8k_answer: str) -> str:
    """GSM8K solutions end with '#### <number>'."""
    return normalize(gsm8k_answer.split("####")[-1])


def normalize(s: str) -> str:
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        x = float(s)
        return str(int(x)) if x == int(x) else str(x)
    except ValueError:
        return s


_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def extract(text: str) -> str | None:
    """The last \\boxed{...} in the completion, or None. No fallback to 'last number',
    so the model must also learn to state its answer in the required format."""
    found = _BOXED.findall(text)
    return normalize(found[-1]) if found else None


def is_correct(text: str, gold: str) -> bool:
    return extract(text) == gold
