"""CPU check: the GRPO loss and its LoRA gradients in tinygrad match PEFT + transformers in PyTorch.
B is set to random values so the A gradients are non-zero too. Rows have different prompt and completion
lengths (right padding, as in train.py) and signed advantages.
Usage: python check_grad.py MODEL_DIR   (a 2-layer copy of Qwen2.5-0.5B is enough)"""
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from tinygrad import Tensor, dtypes

from qwen import Qwen2
import train
from train import make_batch, policy_loss, GradAccumulator
train.bucket = lambda n, b: -(-n // 32) * 32  # small padding on CPU (the orb has 7 GB); training uses 384/768

path = sys.argv[1]
rows = [(list(range(100, 117)), list(range(300, 309))), (list(range(500, 512)), list(range(700, 720)))]
adv = np.array([1.3, -0.7], np.float32)
total = 40  # pretend the step had 40 completion tokens

m = Qwen2(path, dtype=dtypes.float32, max_context=384)
rng = np.random.default_rng(0)
for layer in m.layers:
  for lin in layer.values():
    if hasattr(lin, "B"): lin.B.assign(Tensor(rng.normal(0, 0.02, lin.B.shape).astype(np.float32))).realize()
lp = m.lora_params()
toks, w = make_batch(rows, adv, 3)  # 3 rows: also checks a zero-weight padding row
loss = policy_loss(m, Tensor(toks), Tensor(w)) / total
grads = dict(zip(lp, loss.gradient(*lp.values())))
Tensor.realize(loss, *grads.values())
tg_loss = loss.item()
grads = {k: g.numpy() for k, g in grads.items()}
del loss
# the TinyJit accumulator: runs 1-2 capture the graph, run 3 replays it; it must read each call's new inputs
ga = GradAccumulator(m, list(lp.values()))
for k in (1.0, 2.0, -1.0): ga(toks, w * k)
jit_grads = {k: a.numpy() / total / 2 for k, a in zip(lp, ga.acc)}  # 1 + 2 - 1 = 2x the gradient
lp = {k: v.numpy() for k, v in lp.items()}
del m, ga  # free tinygrad memory before loading the PyTorch model (the orb has 7 GB)

ref = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
ref = get_peft_model(ref, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, target_modules="all-linear"))
named = dict(ref.named_parameters())
with torch.no_grad():
  for k, v in lp.items(): named[k.replace(".weight", ".default.weight")].copy_(torch.tensor(v))
pt_loss = 0
for (p, c), a in zip(rows, adv):
  x = torch.tensor([p + c])
  logp = ref(x).logits[0, :-1].log_softmax(-1).gather(-1, x[0, 1:, None])[:, 0]
  pt_loss = pt_loss - a * logp[len(p) - 1:].sum() / total
pt_loss.backward()
print(f"loss tinygrad {tg_loss:.6f} torch {pt_loss.item():.6f}")
worst = 0.0
for k, g in grads.items():
  want = named[k.replace(".weight", ".default.weight")].grad.numpy()
  rel = np.abs(g - want).max() / (np.abs(want).max() + 1e-12)
  worst = max(worst, rel, np.abs(jit_grads[k] - want).max() / (np.abs(want).max() + 1e-12))
print(f"{len(grads)} LoRA grads (direct and TinyJit-accumulated), worst max-abs error relative to max |grad|: {worst:.2e}")
