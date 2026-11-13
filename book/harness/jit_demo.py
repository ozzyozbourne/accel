"""Chapter 10 data: what TinyJit does across calls, and a kernel compiled for a symbolic size.
Usage: python jit_demo.py OUT_DIR   (writes OUT_DIR/jit/default.json)"""
import contextlib, io, json, os, re, sys
import numpy as np
from tinygrad import Tensor, TinyJit, Variable, Context, Device
from tinygrad.helpers import GlobalCounters


def strip_ansi(s): return re.sub("\x1b\\[(K|.*?m)", "", s)


out_dir = sys.argv[1]
w = Tensor(np.random.default_rng(0).standard_normal((512, 512), dtype=np.float32)).realize()


@TinyJit
def step(x: Tensor) -> Tensor:
  return ((x @ w).relu() * 2 + 1).sum(1).realize()


calls = []
for i in range(4):
  x = Tensor(np.full((256, 512), i + 1, dtype=np.float32)).realize()
  buf = io.StringIO()
  GlobalCounters.reset()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    y = step(x)
    Device[Device.DEFAULT].synchronize()
  calls.append({"call": i, "jit_cnt_after": step.cnt, "kernel_count": GlobalCounters.kernel_count,
                "log": strip_ansi(buf.getvalue()), "y0": float(y.numpy()[0])})

# Symbolic shapes: one kernel for every n in [1, 1024]; n is a kernel argument, not a constant.
big = Tensor(np.arange(1024, dtype=np.float32)).realize()
sym = []
for n_val in (10, 700):
  n = Variable("n", 1, 1024).bind(n_val)
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=4):
    s = (big[:n] * 2).sum().item()
  sym.append({"n": n_val, "sum": s, "expected": float(np.arange(n_val).sum() * 2), "log": strip_ansi(buf.getvalue())[-6000:]})

os.makedirs(f"{out_dir}/jit", exist_ok=True)
json.dump({"example": "jit", "variant": "default", "chapter": 10, "device": Device.DEFAULT, "calls": calls,
           "symbolic": sym}, open(f"{out_dir}/jit/default.json", "w"), indent=1)
print("jit: cnt", [c["jit_cnt_after"] for c in calls], "kernels/call", [c["kernel_count"] for c in calls],
      "symbolic ok", all(abs(s["sum"] - s["expected"]) < 1e-3 for s in sym))
