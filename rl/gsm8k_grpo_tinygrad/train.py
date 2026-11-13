"""GRPO on GSM8K in pure tinygrad: sampling, reward, advantages, policy-gradient loss, AdamW on LoRA.

Same recipe as ../gsm8k_grpo/train.py (TRL): 16 prompts x 8 samples per step, temperature 1.0,
max 512 new tokens, beta=0, one update per batch (ratio = 1, so no clipping), LoRA r=32 alpha=64 on
all linear layers, AdamW lr 3e-5 with 5 linear warmup steps, grad-norm clip 1.0, loss summed over
completion tokens and divided by the step's total completion tokens.
Saves a PEFT-format adapter, so ../gsm8k_grpo/eval.py can score it with vLLM.

Usage: python train.py --model DIR --steps 15 --out /content/tg_grpo
"""
import argparse, json, math, os, sys, time
import numpy as np
from tokenizers import Tokenizer
from datasets import load_dataset
from tinygrad import Tensor, TinyJit, Variable, dtypes, Context
from tinygrad.nn.optim import AdamW
from safetensors.numpy import save_file

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gsm8k_grpo"))
from common import MODEL, SYSTEM, gold_answer, is_correct  # noqa: E402
from qwen import Qwen2, LORA_TARGETS  # noqa: E402

PAD, EOS = 151643, {151643, 151645}  # <|endoftext|>, <|im_end|>


def chat(q: str) -> str:
  return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def bucket(n: int, b: int) -> int: return -(-n // b) * b


class Sampler:
  """Batched temperature sampling with a KV cache; the decode step is one TinyJit graph."""
  def __init__(self, m: Qwen2, bsz: int, temp: float):
    self.m, self.bsz, self.temp = m, bsz, temp
    m.reset_cache(bsz)
    self.step = TinyJit(self._step)

  def _pick(self, logits: Tensor) -> Tensor:  # Gumbel-max == sampling from softmax(logits / temp)
    g = -(-(Tensor.rand(*logits.shape) * (1 - 2e-7) + 1e-7).log()).log()
    return (logits / self.temp + g).argmax(-1).reshape(-1, 1)

  def _step(self, tok: Tensor, sp, valid: Tensor) -> Tensor:
    h = self.m(tok, sp, valid.reshape(self.bsz, 1, 1, 1, -1)[..., :sp + 1], "decode")
    return self._pick(self.m.logits(h[:, -1])).realize()

  def __call__(self, prompts: list[list[int]], max_new: int) -> list[list[int]]:
    B, C = self.bsz, self.m.max_context
    Tp = bucket(max(map(len, prompts)), 256)  # GSM8K prompts are at most 251 tokens: one prefill shape
    assert Tp + max_new <= C, (Tp, max_new, C)
    toks = np.full((B, Tp), PAD, np.int32)
    valid = np.ones((B, C), bool)
    for i, p in enumerate(prompts):
      toks[i, Tp - len(p):] = p
      valid[i, :Tp - len(p)] = False
    vt = Tensor(valid).realize()
    v = vt[:, :Tp].reshape(B, 1, 1, 1, Tp)
    mask = (Tensor.ones(Tp, Tp, dtype=dtypes.bool).tril().reshape(1, 1, 1, Tp, Tp) & v) | Tensor.eye(Tp, dtype=dtypes.bool)
    tok = self._pick(self.m.logits(self.m(Tensor(toks), 0, mask, "prefill")[:, -1])).realize()
    out, done = [tok], np.zeros(B, bool)
    for t in range(1, max_new):
      # TinyJit returns the same output buffer on every call, so keep a copy of each token
      tok = self.step(tok, Variable("start_pos", 1, C - 1).bind(Tp + t - 1), vt).clone().realize()
      out.append(tok)
      if t % 32 == 0:  # check for EOS without syncing every token
        done |= np.isin(Tensor.cat(*out[-32:], dim=1).numpy(), list(EOS)).any(1)
        if done.all(): break
    gen = Tensor.cat(*out, dim=1).numpy()
    res = []
    for row in gen.tolist():
      cut = next((j + 1 for j, x in enumerate(row) if x in EOS), len(row))
      res.append(row[:cut])
    return res


def make_batch(rows, adv, n_rows: int):
  """rows: list of (prompt ids, completion ids). Returns tokens and per-position weights (the advantage on
  positions that predict a completion token). Padded to n_rows x (384 or 768) so there are only 2 shapes:
  each shape gets its own TinyJit, and each JIT keeps its own activation memory."""
  L = bucket(max(len(p) + len(c) for p, c in rows), 384)
  toks = np.full((n_rows, L), PAD, np.int32)
  w = np.zeros((n_rows, L - 1), np.float32)  # padding rows keep weight 0
  for i, ((p, c), a) in enumerate(zip(rows, adv)):
    toks[i, :len(p) + len(c)] = p + c
    w[i, len(p) - 1:len(p) + len(c) - 1] = a  # position j predicts token j+1
  return toks, w


def policy_loss(m: Qwen2, x: Tensor, w: Tensor) -> Tensor:
  """-sum(w * logp(next token)); the caller divides by the step's completion-token count."""
  L = x.shape[1]
  causal = Tensor.ones(L, L, dtype=dtypes.bool).tril().reshape(1, 1, 1, L, L)
  logits = m.logits(m(x, 0, causal, "train")[:, :L - 1])
  logp = logits.gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1) - logits.logsumexp(-1)
  return -(w * logp).sum()


class GradAccumulator:
  """acc += d policy_loss / d params, as one TinyJit graph per sequence length. Without the JIT, tinygrad
  rebuilds and schedules the 28-layer backward graph in Python for every micro-batch (~30 s on the L4)."""
  def __init__(self, m: Qwen2, params: list[Tensor]):
    self.m, self.params = m, params
    self.acc = [Tensor.zeros(*p.shape, dtype=dtypes.float32).contiguous().realize() for p in params]
    self.jits: dict[int, TinyJit] = {}

  def _accum(self, x: Tensor, w: Tensor):
    grads = policy_loss(self.m, x, w).gradient(*self.params)
    Tensor.realize(*[a.assign(a + g) for a, g in zip(self.acc, grads)])

  def __call__(self, toks: np.ndarray, w: np.ndarray):
    jit = self.jits.setdefault(toks.shape[1], TinyJit(self._accum))
    jit(Tensor(toks).realize(), Tensor(w).realize())

  def zero(self):
    for a in self.acc: a.assign(a.zeros_like()).realize()


def save_adapter(m: Qwen2, out: str, r: int, alpha: int):
  os.makedirs(out, exist_ok=True)
  # tinygrad's safe_save writes through its DISK device, which needs clang (missing on Colab); go through numpy
  save_file({k: v.numpy() for k, v in m.lora_params().items()}, f"{out}/adapter_model.safetensors")
  cfg = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "base_model_name_or_path": MODEL, "r": r, "lora_alpha": alpha,
         "lora_dropout": 0.0, "bias": "none", "fan_in_fan_out": False, "inference_mode": True, "use_rslora": False,
         "use_dora": False, "target_modules": [hf.split(".")[-1] for hf in LORA_TARGETS.values()]}
  with open(f"{out}/adapter_config.json", "w") as f: json.dump(cfg, f, indent=2)


def main():
  import signal  # log catchable signals: on Colab the job has been killed with no traceback
  for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
    signal.signal(s, lambda n, f: (print(f"GOT SIGNAL {n}", flush=True), os._exit(128 + n)))
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", required=True)
  ap.add_argument("--out", default="/content/tg_grpo")
  ap.add_argument("--steps", type=int, default=15)
  ap.add_argument("--prompts", type=int, default=16)
  ap.add_argument("--gens", type=int, default=8)
  ap.add_argument("--max_new", type=int, default=512)
  ap.add_argument("--micro", type=int, default=2)
  ap.add_argument("--lr", type=float, default=3e-5)
  ap.add_argument("--warmup", type=int, default=5)
  ap.add_argument("--dtype", default="bfloat16")
  a = ap.parse_args()
  Tensor.manual_seed(0)
  r, alpha = 32, 64

  tok = Tokenizer.from_file(os.path.join(a.model, "tokenizer.json"))
  ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0)
  m = Qwen2(a.model, dtype=getattr(dtypes, a.dtype), lora_r=r, lora_alpha=alpha, max_context=1024)
  params = list(m.lora_params().values())
  opt = AdamW(params, lr=a.lr, weight_decay=0.0)
  ga = GradAccumulator(m, params)
  sampler = Sampler(m, a.prompts * a.gens, temp=1.0)
  print(f"loaded {a.model}: {sum(p.numel() for p in params) / 1e6:.1f}M LoRA params", flush=True)

  for step in range(a.steps):
    t0 = time.time()
    batch = ds[step * a.prompts:(step + 1) * a.prompts]
    prompts = [tok.encode(chat(q), add_special_tokens=False).ids for q in batch["question"]]
    golds = [gold_answer(x) for x in batch["answer"]]
    comps = sampler([p for p in prompts for _ in range(a.gens)], a.max_new)
    t_gen = time.time() - t0

    texts = [tok.decode(c, skip_special_tokens=True) for c in comps]
    rew = np.array([float(is_correct(t, golds[i // a.gens])) for i, t in enumerate(texts)]).reshape(a.prompts, a.gens)
    adv = ((rew - rew.mean(1, keepdims=True)) / (rew.std(1, ddof=1, keepdims=True) + 1e-4)).reshape(-1)
    total = sum(map(len, comps))
    # rows with zero advantage contribute nothing to the loss or its gradient, so skip them
    live = sorted((i for i in range(len(comps)) if adv[i] != 0), key=lambda i: len(prompts[i // a.gens]) + len(comps[i]))

    ga.zero()
    for j in range(0, len(live), a.micro):
      ids = live[j:j + a.micro]
      ga(*make_batch([(prompts[i // a.gens], comps[i]) for i in ids], adv[ids], a.micro))
    gnorm = math.sqrt(sum(x.square().sum() for x in ga.acc).item()) / total  # loss is divided by total tokens
    lr = a.lr * min(1.0, step / a.warmup) if a.warmup else a.lr  # HF constant_with_warmup: step 0 has lr 0
    opt.lr.assign(Tensor([lr], dtype=opt.lr.dtype)).realize()
    scale = min(1.0, 1.0 / (gnorm + 1e-6))
    for p, x in zip(params, ga.acc): p.grad = x * (scale / total)
    with Context(TRAINING=1): opt.step()
    opt.zero_grad()

    print(json.dumps({"step": step + 1, "reward": round(float(rew.mean()), 4),
                      "groups_with_signal": int((rew.std(1) > 0).sum()), "trained_rows": len(live),
                      "mean_len": round(total / len(comps), 1), "clipped": int(sum(len(c) == a.max_new for c in comps)),
                      "grad_norm": round(gnorm, 4), "lr": lr, "gen_s": round(t_gen, 1),
                      "train_s": round(time.time() - t0 - t_gen, 1), "step_s": round(time.time() - t0, 1)}), flush=True)
    save_adapter(m, f"{a.out}/final", r, alpha)  # every step, so a crash late in the run still leaves an adapter
    if step == 0: print("SAMPLE:", texts[0][:600].replace("\n", " | "), "| GOLD", golds[0], flush=True)

  print("saved", f"{a.out}/final", flush=True)


if __name__ == "__main__":
  main()
