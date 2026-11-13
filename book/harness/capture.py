"""Capture every lowering stage of one example in tinygrad on the current device.

Usage: python capture.py EXAMPLE VARIANT OUT_DIR
Environment variables of the variant (NOOPT, BEAM, TC, TRANSCENDENTAL, DEV=CUDA:PTX) must be set by
the caller before this process starts; run_all.py does that.

Writes OUT_DIR/EXAMPLE/VARIANT.json with:
  tensor_graph   pyrender of the lazy Tensor graph before realize
  passes         every named graph_rewrite pass on the way down: stage (tensor/schedule/codegen/program),
                 name, kernel index, node count, whether it changed the graph, wall time, IR after the pass
  kernels        per kernel: input AST, applied opts, launch dims, linearized uops, rendered source,
                 PTX, SASS (ptxas + cuobjdump), ptxas resource usage, measured time
  check          max abs error against the numpy reference
"""
import contextlib, importlib, io, json, os, re, subprocess, sys, tempfile, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from examples import EXAMPLES  # noqa: E402

MAX_IR_LINES = 400
STAGE_OF_MODULE = {"tinygrad.tensor": "tensor", "tinygrad.schedule": "schedule", "tinygrad.schedule.rangeify": "schedule",
                   "tinygrad.schedule.indexing": "schedule", "tinygrad.schedule.multi": "schedule",
                   "tinygrad.codegen": "codegen", "tinygrad.codegen.opt.postrange": "codegen", "tinygrad.engine.realize": "program"}


def strip_ansi(s: str) -> str: return re.sub("\x1b\\[(K|.*?m)", "", s)


def clip(text: str, n=MAX_IR_LINES) -> dict:
  lines = text.splitlines()
  return {"text": "\n".join(lines[:n]), "lines": len(lines), "clipped": len(lines) > n}


def main():
  name, variant, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
  ex = EXAMPLES[name]
  if "same_as" in ex: ex = {**EXAMPLES[ex["same_as"]], **{k: v for k, v in ex.items() if k != "same_as"}}

  import tinygrad.uop.ops as ops
  from tinygrad.uop.render import pyrender
  orig = ops.graph_rewrite
  passes, kernels = [], []
  state = {"active": False, "kernel": -1}

  def render(u):
    try: return pyrender(u)
    except Exception as e: return f"<pyrender failed: {e}>"

  def make_hook(module):
    stage = STAGE_OF_MODULE[module]
    def hooked(sink, pm, ctx=None, bottom_up=False, name=None, **kw):
      t0 = time.perf_counter()
      out = orig(sink, pm, ctx=ctx, bottom_up=bottom_up, name=name, **kw)
      dt = time.perf_counter() - t0
      if not state["active"]: return out
      if stage == "codegen" and name == "multi_pm":  # first pass of full_rewrite_to_sink: a new kernel starts
        state["kernel"] += 1
        kernels.append({"ast": clip(render(sink))})
      try: n = len(out.toposort())
      except Exception: n = -1
      rec = {"stage": stage, "name": name, "kernel": state["kernel"] if stage in ("codegen", "program") else None,
             "nodes": n, "changed": out is not sink, "ms": round(dt * 1e3, 3)}
      if name == "linearize/render" and out.op is ops.Ops.PROGRAM: grab_program(out)
      elif out.op is not ops.Ops.PROGRAM and (out is not sink or not passes or passes[-1].get("kernel") != rec["kernel"]):
        rec["ir"] = clip(render(out))  # unchanged passes keep no copy: the IR is the previous pass's
      passes.append(rec)
      return out
    return hooked

  def grab_program(prg):
    k = kernels[state["kernel"]] if 0 <= state["kernel"] < len(kernels) else kernels.append({}) or kernels[-1]
    info = prg.arg
    k.update({"name": strip_ansi(info.name), "function_name": info.function_name,
              "global_size": [str(x) for x in info.global_size],
              "local_size": None if info.local_size is None else [str(x) for x in info.local_size]})
    sink = prg.src[0]
    if isinstance(sink.arg, ops.KernelInfo):
      k["applied_opts"] = [str(o) for o in sink.arg.applied_opts]
      if sink.arg.estimates is not None: k["estimates"] = {f: str(getattr(sink.arg.estimates, f)) for f in ("ops", "lds", "mem")}
    for s in prg.src[1:]:
      if s.op is ops.Ops.LINEAR:
        idx = {u: i for i, u in enumerate(s.src)}
        rows = []
        for i, u in enumerate(s.src):
          srcs = [(idx[x] if x.op is not ops.Ops.CONST else f"{x.arg}") if x in idx else "--" for x in u.src]
          rows.append(f"{i:4d} {str(u.op):20s} {str(u.dtype):32s} {str(srcs):28s} {strip_ansi(str(u.arg))[:120]}")
        k["uops"] = clip("\n".join(rows), 600)
      elif s.op is ops.Ops.SOURCE: k["source"] = s.arg
      elif s.op is ops.Ops.BINARY: k["binary"] = s.arg

  for module in STAGE_OF_MODULE:
    m = importlib.import_module(module)
    if hasattr(m, "graph_rewrite"): m.graph_rewrite = make_hook(module)

  from tinygrad import Tensor, Device, dtypes, Context  # noqa: F401
  rng = np.random.default_rng(0)
  host = {k: f(rng) for k, f in ex["inputs"].items()}
  T = {k: Tensor(v).realize() for k, v in host.items()}
  Device[Device.DEFAULT].synchronize()

  def build(T):
    if ex.get("backward"):
      loss = (T["x"] @ T["w"]).relu().sum()
      return loss.gradient(T["w"])[0]
    return ex["tg"](T)

  out_t = build(T)
  tensor_graph = clip(render(out_t.uop.sink()))
  state["active"] = True
  t0 = time.perf_counter()
  out_t.realize()
  Device[Device.DEFAULT].synchronize()
  compile_and_run_s = time.perf_counter() - t0
  state["active"] = False
  got = out_t.numpy()

  if ex.get("backward"):
    x, w = host["x"], host["w"]
    want = x.T @ (x @ w > 0).astype(np.float32)
  else:
    want = ex["ref"](host)
  err = float(np.max(np.abs(got.astype(np.float64) - np.asarray(want, dtype=np.float64))))
  scale = float(np.max(np.abs(want))) or 1.0
  tol = ex.get("tol", 1e-6) * max(1.0, scale)
  check = {"max_abs_err": err, "max_abs_ref": scale, "tol": tol, "ok": bool(err <= tol), "shape": list(got.shape),
           "dtype": str(got.dtype)}

  # Time the cached kernels: rerun the same graph on the same inputs, DEBUG=2 prints one line per kernel.
  times: list = []  # per executed kernel, in launch order: [name, best us]
  for _ in range(5):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), Context(DEBUG=2):
      build(T).realize()
      Device[Device.DEFAULT].synchronize()
    rows = []
    for line in strip_ansi(buf.getvalue()).splitlines():
      m = re.match(r"\*\*\*\s+\S+\s+\d+\s+(\S+).*?tm\s+([\d.]+)(us|ms|s)", line)
      if m: rows.append([m.group(1), float(m.group(2)) * {"us": 1, "ms": 1e3, "s": 1e6}[m.group(3)]])
    times = rows if not times else [[n, min(t, r[1])] for (n, t), r in zip(times, rows)]

  dev = Device.DEFAULT
  if os.environ.get("BEAM"):  # BEAM compiles candidates through the same passes; keep the last kernel of each name
    keep = {k.get("name"): i for i, k in enumerate(kernels)}
    keep_ids = set(keep.values())
    remap = {old: new for new, old in enumerate(sorted(keep_ids))}
    kernels = [k for i, k in enumerate(kernels) if i in keep_ids]
    passes = [p for p in passes if p["kernel"] is None or p["kernel"] in keep_ids]
    for p in passes:
      if p["kernel"] is not None: p["kernel"] = remap[p["kernel"]]
  compiled = [k for k in kernels if "name" in k]
  for k, (tname, us) in zip(compiled, times):
    if tname == k["name"]: k["time_us"] = us
  for k in kernels:
    binary = k.pop("binary", None)
    if dev.startswith("CUDA") and binary is not None: k.update(sass_of(binary))
  arch = getattr(Device[dev], "arch", None)
  result = {"example": name, "variant": variant, "device": dev, "arch": arch, "chapter": ex["chapter"],
            "env": {k: os.environ[k] for k in ("DEV", "NOOPT", "BEAM", "TC", "TRANSCENDENTAL") if k in os.environ},
            "tinygrad_code": example_source(name), "tensor_graph": tensor_graph, "passes": passes, "kernels": kernels,
            "check": check, "compile_and_run_s": round(compile_and_run_s, 3), "kernel_times_us": times}
  os.makedirs(f"{out_dir}/{name}", exist_ok=True)
  with open(f"{out_dir}/{name}/{variant}.json", "w") as f: json.dump(result, f, indent=1)
  print(f"{name}/{variant}: {len(kernels)} kernels, {len(passes)} passes, check {'OK' if check['ok'] else 'FAIL'} "
        f"(err {err:.2e}), times {times}")


def example_source(name):
  import inspect
  src = inspect.getsource(sys.modules["examples"])
  m = re.search(rf'  "{name}": dict\(.*?(?=\n  "|\n  # |\n}}\n)', src, re.S)
  return m.group(0).strip() if m else ""


def sass_of(binary: bytes) -> dict:
  """PTX from NVRTC (or tinygrad's PTX renderer) -> cubin with ptxas -> SASS with cuobjdump."""
  ptx = binary.rstrip(b"\x00").decode(errors="replace")
  out = {"ptx": clip(ptx, 1500)}
  with tempfile.TemporaryDirectory() as d:
    open(f"{d}/k.ptx", "w").write(ptx)
    arch = re.search(r"\.target\s+(sm_\d+)", ptx)
    arch = arch.group(1) if arch else "sm_89"
    r = subprocess.run(["ptxas", f"-arch={arch}", "-v", "-o", f"{d}/k.cubin", f"{d}/k.ptx"], capture_output=True, text=True)
    out["ptxas_info"] = (r.stdout + r.stderr).strip()
    if r.returncode == 0:
      s = subprocess.run(["cuobjdump", "-sass", f"{d}/k.cubin"], capture_output=True, text=True)
      out["sass"] = clip(s.stdout, 1500)
  return out


if __name__ == "__main__":
  main()
