"""Qwen2 (Qwen2.5) in tinygrad, with LoRA adapters on every linear layer.

The base weights are frozen; only the LoRA A/B matrices are trained. The model has three modes:
  "train":   full sequence, no KV cache (used for the policy-gradient logprobs).
  "prefill": full prompt, writes the KV cache.
  "decode":  one token per row at position start_pos (a bound Variable), reads the KV cache.

Rows are left-padded. RoPE only depends on position differences, so every row can use the
same absolute positions as long as the pad keys are masked out.
"""
import json, math, pathlib
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import safe_load

LORA_TARGETS = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
                "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}


class Linear:
  def __init__(self, w: Tensor, b: Tensor | None, r: int, alpha: float):
    self.w, self.b, self.scale = w, b, alpha / r
    # PEFT's default init: A ~ kaiming_uniform(a=sqrt(5)) = U(-1/sqrt(fan_in), 1/sqrt(fan_in)), B = 0
    bound = 1 / math.sqrt(w.shape[1])
    self.A = Tensor.uniform(r, w.shape[1], low=-bound, high=bound, dtype=dtypes.float32).contiguous().realize()
    self.B = Tensor.zeros(w.shape[0], r, dtype=dtypes.float32).contiguous().realize()

  def __call__(self, x: Tensor) -> Tensor:
    y = x.linear(self.w.T, self.b)
    return y + (x @ self.A.T.cast(x.dtype)) @ self.B.T.cast(x.dtype) * self.scale


def rms_norm(x: Tensor, w: Tensor, eps: float) -> Tensor:
  xf = x.float()
  return (xf * (xf.square().mean(-1, keepdim=True) + eps).rsqrt()).cast(x.dtype) * w


def rotate_half(x: Tensor) -> Tensor:
  d = x.shape[-1] // 2
  return (-x[..., d:]).cat(x[..., :d], dim=-1)


class Qwen2:
  def __init__(self, path: str, dtype=dtypes.bfloat16, lora_r=32, lora_alpha=64, max_context=1024):
    path = pathlib.Path(path)
    cfg = json.loads((path / "config.json").read_text())
    self.dim, self.n_layers = cfg["hidden_size"], cfg["num_hidden_layers"]
    self.n_heads, self.n_kv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    self.head_dim, self.eps, self.max_context, self.dtype = self.dim // self.n_heads, cfg["rms_norm_eps"], max_context, dtype

    sd = {}
    for f in sorted(path.glob("*.safetensors")): sd.update(safe_load(str(f)))
    def W(name): return sd[name].to(None).cast(dtype).contiguous().realize().is_param_(False)

    self.embed = W("model.embed_tokens.weight")
    self.lm_head = W("lm_head.weight") if "lm_head.weight" in sd and not cfg.get("tie_word_embeddings") else self.embed
    self.norm = W("model.norm.weight")
    self.layers = []
    for l in range(self.n_layers):
      p = f"model.layers.{l}."
      layer = {"in_norm": W(p + "input_layernorm.weight"), "post_norm": W(p + "post_attention_layernorm.weight")}
      for short, hf in LORA_TARGETS.items():
        bias = p + hf + ".bias"
        layer[short] = Linear(W(p + hf + ".weight"), W(bias) if bias in sd else None, lora_r, lora_alpha)
      self.layers.append(layer)
    del sd

    inv = 1.0 / (cfg.get("rope_theta", 10000.0) ** (Tensor.arange(0, self.head_dim, 2, dtype=dtypes.float32) / self.head_dim))
    freqs = Tensor.arange(max_context, dtype=dtypes.float32).unsqueeze(1) * inv.unsqueeze(0)
    emb = freqs.cat(freqs, dim=-1)
    self.cos = emb.cos().cast(dtype).contiguous().realize().is_param_(False)
    self.sin = emb.sin().cast(dtype).contiguous().realize().is_param_(False)
    self.cache = None

  def lora_params(self) -> dict[str, Tensor]:
    """PEFT adapter key -> tensor, so the trained adapter can be served with vLLM/PEFT."""
    out = {}
    for l, layer in enumerate(self.layers):
      for short, hf in LORA_TARGETS.items():
        k = f"base_model.model.model.layers.{l}.{hf}"
        out[f"{k}.lora_A.weight"], out[f"{k}.lora_B.weight"] = layer[short].A, layer[short].B
    return out

  def reset_cache(self, bsz: int):
    shape = (bsz, self.n_kv, self.max_context, self.head_dim)
    self.cache = [(Tensor.zeros(*shape, dtype=self.dtype).contiguous().realize(),
                   Tensor.zeros(*shape, dtype=self.dtype).contiguous().realize()) for _ in range(self.n_layers)]

  def _attn(self, i: int, layer, x: Tensor, start_pos, mask: Tensor, mode: str) -> Tensor:
    B, T, _ = x.shape
    H, KV, D = self.n_heads, self.n_kv, self.head_dim
    q = layer["q"](x).reshape(B, T, H, D).transpose(1, 2)
    k = layer["k"](x).reshape(B, T, KV, D).transpose(1, 2)
    v = layer["v"](x).reshape(B, T, KV, D).transpose(1, 2)
    cos, sin = self.cos[start_pos:start_pos + T], self.sin[start_pos:start_pos + T]
    q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
    if mode == "train":
      keys, vals = k, v
    else:
      ck, cv = self.cache[i]
      ck[:, :, start_pos:start_pos + T].assign(k).realize()
      cv[:, :, start_pos:start_pos + T].assign(v).realize()
      keys, vals = (k, v) if mode == "prefill" else (ck[:, :, 0:start_pos + T], cv[:, :, 0:start_pos + T])
    # grouped-query attention without materializing repeated K/V
    q = q.reshape(B, KV, H // KV, T, D)
    s = (q @ keys.unsqueeze(2).transpose(-1, -2)).float() * (1 / math.sqrt(D))
    p = mask.where(s, -math.inf).softmax(-1).cast(x.dtype)
    o = (p @ vals.unsqueeze(2)).reshape(B, H, T, D).transpose(1, 2).reshape(B, T, H * D)
    return layer["o"](o)

  def __call__(self, tokens: Tensor, start_pos, mask: Tensor, mode: str) -> Tensor:
    """tokens (B, T) int; mask (B, 1, 1, T, S) bool, True where a query may attend to a key.
    Returns the final-norm hidden states (B, T, dim)."""
    h = self.embed[tokens]
    for i, layer in enumerate(self.layers):
      h = h + self._attn(i, layer, rms_norm(h, layer["in_norm"], self.eps), start_pos, mask, mode)
      x = rms_norm(h, layer["post_norm"], self.eps)
      h = (h + layer["down"](layer["gate"](x).silu() * layer["up"](x))).contiguous()
    return rms_norm(h, self.norm, self.eps)

  def logits(self, h: Tensor) -> Tensor:
    return (h @ self.lm_head.T).float()
