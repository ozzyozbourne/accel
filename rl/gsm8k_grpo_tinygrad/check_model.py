"""CPU check: tinygrad Qwen2 logits match HF transformers, KV-cache decode matches full forward,
and LoRA gradients are non-zero only for the LoRA tensors.
Usage: python check_model.py MODEL_DIR"""
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from tinygrad import Tensor, dtypes, Variable

from qwen import Qwen2

path = sys.argv[1]
ids = [[151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 3838, 374, 220, 17, 10, 17, 30]]
T = len(ids[0])

ref = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
with torch.no_grad(): want = ref(torch.tensor(ids)).logits[0].numpy()
del ref

m = Qwen2(path, dtype=dtypes.float32, max_context=64)
causal = Tensor.ones(T, T, dtype=dtypes.bool).tril().reshape(1, 1, 1, T, T)
got = m.logits(m(Tensor(ids), 0, causal, "train"))[0].numpy()
print("max |logit diff| vs HF:", np.abs(got - want).max(), "argmax agree:", (got.argmax(-1) == want.argmax(-1)).mean())

# left-padded copy of the same prompt (3 pad tokens) must give the same last-token logits
pad = 3
toks = Tensor([[151643] * pad + ids[0]])
valid = Tensor([[False] * pad + [True] * T])
L = T + pad
mask = (Tensor.ones(L, L, dtype=dtypes.bool).tril() & valid.reshape(1, 1, L)) | Tensor.eye(L, dtype=dtypes.bool)
got_pad = m.logits(m(toks, 0, mask.reshape(1, 1, 1, L, L), "train"))[0, -1].numpy()
print("left-pad last-token diff:", np.abs(got_pad - want[-1]).max())

# prefill T-1 tokens, then decode token T-1 from the cache
m.reset_cache(1)
m(Tensor([ids[0][:-1]]), 0, causal[..., :T - 1, :T - 1], "prefill").realize()
dmask = Tensor.ones(1, 1, 1, 1, 64, dtype=dtypes.bool)
sp = Variable("sp", 1, 63).bind(T - 1)
got_dec = m.logits(m(Tensor([[ids[0][-1]]]), sp, dmask[..., :sp + 1], "decode"))[0, -1].numpy()
print("kv-cache decode diff:", np.abs(got_dec - want[-1]).max())

# gradients reach every LoRA tensor (B is 0 at init, so A's grad is 0 until B moves; check B)
params = list(m.lora_params().values())
loss = m.logits(m(Tensor(ids), 0, causal, "train"))[0, -1].log_softmax()[17]
grads = loss.gradient(*params)
Bnorm = [g.square().sum().item() for k, g in zip(m.lora_params(), grads) if "lora_B" in k]
print("lora_B grads non-zero:", sum(x > 0 for x in Bnorm), "/", len(Bnorm))
