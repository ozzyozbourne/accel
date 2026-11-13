"""The kernels the book lowers. Each example builds its inputs from a seeded numpy RNG, then gives
   tg(T):    the tinygrad expression (T maps input name -> realized Tensor), returns the output Tensor
   ref(N):   the numpy reference (N maps input name -> ndarray), used to check the GPU result
   torch(X): the same expression in PyTorch (X maps input name -> CUDA tensor), for the comparison
Variants are environment settings applied in a fresh process, so each one compiles from scratch."""
import numpy as np


def _f32(*shape): return lambda rng: rng.standard_normal(shape, dtype=np.float32)
def _f16(*shape): return lambda rng: rng.standard_normal(shape, dtype=np.float32).astype(np.float16)


def _softmax(x): e = np.exp(x - x.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
def _layernorm(x):
  m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
  return (x - m) / np.sqrt(v + 1e-5)


EXAMPLES = {
  # ch01: one elementwise kernel, end to end
  "add": dict(chapter=1, inputs={"a": _f32(1024, 1024), "b": _f32(1024, 1024)},
              tg=lambda T: T["a"] + T["b"], ref=lambda N: N["a"] + N["b"], torch=lambda X: X["a"] + X["b"]),
  # ch02: fusion
  "fused_chain": dict(chapter=2, inputs={"a": _f32(1024, 1024), "b": _f32(1024, 1024), "c": _f32(1024, 1024)},
                      tg=lambda T: ((T["a"] * T["b"] + T["c"]).relu() * 2.0),
                      ref=lambda N: np.maximum(N["a"] * N["b"] + N["c"], 0) * 2.0,
                      torch=lambda X: (X["a"] * X["b"] + X["c"]).relu() * 2.0),
  "gelu": dict(chapter=2, inputs={"a": _f32(1024, 1024)},
               tg=lambda T: T["a"].gelu(),
               ref=lambda N: 0.5 * N["a"] * (1 + np.tanh(0.7978845608 * (N["a"] + 0.044715 * N["a"] ** 3))),
               torch=lambda X: __import__("torch").nn.functional.gelu(X["a"], approximate="tanh"), tol=1e-4),
  # ch03: movement ops become index arithmetic
  "transpose": dict(chapter=3, inputs={"a": _f32(1024, 1024)},
                    tg=lambda T: T["a"].T.contiguous(), ref=lambda N: N["a"].T.copy(), torch=lambda X: X["a"].t().contiguous()),
  "broadcast_add": dict(chapter=3, inputs={"a": _f32(1024, 1024), "b": _f32(1024)},
                        tg=lambda T: T["a"] + T["b"].reshape(1, 1024), ref=lambda N: N["a"] + N["b"][None, :],
                        torch=lambda X: X["a"] + X["b"][None, :]),
  "pad_slice": dict(chapter=3, inputs={"a": _f32(1000, 1000)},
                    tg=lambda T: T["a"][::2, 1::3].pad(((1, 1), (2, 2))).contiguous(),
                    ref=lambda N: np.pad(N["a"][::2, 1::3], ((1, 1), (2, 2))),
                    torch=lambda X: __import__("torch").nn.functional.pad(X["a"][::2, 1::3], (2, 2, 1, 1)).contiguous()),
  "permute4d": dict(chapter=3, inputs={"a": _f32(32, 32, 32, 32)},
                    tg=lambda T: T["a"].permute(0, 2, 1, 3).contiguous(), ref=lambda N: N["a"].transpose(0, 2, 1, 3).copy(),
                    torch=lambda X: X["a"].permute(0, 2, 1, 3).contiguous()),
  # ch04: reductions
  "sum_rows": dict(chapter=4, inputs={"a": _f32(1024, 1024)},
                   tg=lambda T: T["a"].sum(1), ref=lambda N: N["a"].sum(1), torch=lambda X: X["a"].sum(1), tol=1e-3),
  "max_cols": dict(chapter=4, inputs={"a": _f32(1024, 1024)},
                   tg=lambda T: T["a"].max(0), ref=lambda N: N["a"].max(0), torch=lambda X: X["a"].amax(0)),
  "sum_all": dict(chapter=4, inputs={"a": _f32(4096, 4096)},
                  tg=lambda T: T["a"].sum(), ref=lambda N: N["a"].astype(np.float64).sum(), torch=lambda X: X["a"].sum(), tol=2e-2),
  # ch05: multi-kernel schedules
  "softmax": dict(chapter=5, inputs={"a": _f32(1024, 1024)},
                  tg=lambda T: T["a"].softmax(-1), ref=lambda N: _softmax(N["a"]), torch=lambda X: X["a"].softmax(-1), tol=1e-5),
  "layernorm": dict(chapter=5, inputs={"a": _f32(1024, 1024)},
                    tg=lambda T: T["a"].layernorm(), ref=lambda N: _layernorm(N["a"]),
                    torch=lambda X: __import__("torch").nn.functional.layer_norm(X["a"], (1024,)), tol=1e-4),
  # ch06: fp32 matmul and the optimizer
  "matmul": dict(chapter=6, inputs={"a": _f32(1024, 1024), "b": _f32(1024, 1024)},
                 tg=lambda T: T["a"] @ T["b"], ref=lambda N: N["a"] @ N["b"], torch=lambda X: X["a"] @ X["b"], tol=1e-2,
                 variants={"default": {}, "noopt": {"NOOPT": "1"}, "beam2": {"BEAM": "2"}}),
  # ch07: tensor cores (fp16 in, fp32 accumulate)
  "matmul_fp16": dict(chapter=7, inputs={"a": _f16(2048, 2048), "b": _f16(2048, 2048)},
                      tg=lambda T: T["a"].matmul(T["b"], dtype=__import__("tinygrad").dtypes.float),
                      ref=lambda N: N["a"].astype(np.float32) @ N["b"].astype(np.float32),
                      torch=lambda X: (X["a"] @ X["b"]).float(), tol=5e-2,
                      variants={"default": {}, "no_tc": {"TC": "0"}}),
  # ch08: transcendentals and decompositions
  "exp": dict(chapter=8, inputs={"a": _f32(1024, 1024)},
              tg=lambda T: T["a"].exp(), ref=lambda N: np.exp(N["a"]), torch=lambda X: X["a"].exp(), tol=1e-5,
              variants={"default": {}, "transcendental2": {"TRANSCENDENTAL": "2"}}),
  "sin": dict(chapter=8, inputs={"a": _f32(1024, 1024)},
              tg=lambda T: T["a"].sin(), ref=lambda N: np.sin(N["a"]), torch=lambda X: X["a"].sin(), tol=1e-5,
              variants={"default": {}, "transcendental2": {"TRANSCENDENTAL": "2"}}),
  "int_divmod": dict(chapter=8, inputs={"a": lambda rng: rng.integers(0, 1 << 20, (1024, 1024), dtype=np.int32)},
                     tg=lambda T: T["a"] // 7 + T["a"] % 10, ref=lambda N: N["a"] // 7 + N["a"] % 10,
                     torch=lambda X: X["a"] // 7 + X["a"] % 10),
  # ch09: tinygrad's own PTX renderer versus CUDA C through NVRTC
  "add_ptx": dict(chapter=9, same_as="add", variants={"cuda": {}, "ptx": {"DEV": "CUDA:PTX"}}),
  "sum_rows_ptx": dict(chapter=9, same_as="sum_rows", variants={"cuda": {}, "ptx": {"DEV": "CUDA:PTX"}}),
  "matmul_ptx": dict(chapter=9, same_as="matmul", variants={"cuda": {}, "ptx": {"DEV": "CUDA:PTX"}}),
  # ch10: autograd (the JIT is captured by jit_demo.py)
  "mlp_backward": dict(chapter=10, inputs={"x": _f32(256, 512), "w": _f32(512, 256)}, backward=True),
}
