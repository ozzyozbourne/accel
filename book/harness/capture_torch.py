"""Capture how PyTorch runs the same example: eager CUDA kernels, then torch.compile (Inductor) stages.
Usage: python capture_torch.py EXAMPLE OUT_DIR   (writes OUT_DIR/EXAMPLE/torch.json)

Eager:  kernel names and times from torch.profiler.
Inductor (TORCH_COMPILE_DEBUG=1 writes the debug files): the FX graph Dynamo traced, Inductor's IR before
and after fusion, the generated output code with Triton kernels; then each Triton kernel's compile
stages from TRITON_CACHE_DIR (ttir, ttgir, llir, ptx, cubin) and SASS from cuobjdump."""
import glob, json, os, subprocess, sys, tempfile

work = tempfile.mkdtemp(prefix="torchcap_")
os.environ["TORCH_COMPILE_DEBUG"] = "1"
os.environ["TORCH_COMPILE_DEBUG_DIR"] = work
os.environ["TRITON_CACHE_DIR"] = f"{work}/triton"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = f"{work}/inductor"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
os.chdir(work)

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from examples import EXAMPLES  # noqa: E402

MAX = 1500


def clip(text, n=MAX):
  lines = text.splitlines()
  return {"text": "\n".join(lines[:n]), "lines": len(lines), "clipped": len(lines) > n}


def main():
  name, out_dir = sys.argv[1], sys.argv[2]
  ex = EXAMPLES[name]
  rng = np.random.default_rng(0)
  host = {k: f(rng) for k, f in ex["inputs"].items()}
  X = {k: torch.from_numpy(v).cuda() for k, v in host.items()}

  if ex.get("backward"):
    def fn(X):
      w = X["w"].detach().requires_grad_()
      (X["x"] @ w).relu().sum().backward()
      return w.grad
  else:
    fn = ex["torch"]

  ref = fn(X)
  torch.cuda.synchronize()

  def profile(f):
    for _ in range(3): f(X)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
      f(X)
      torch.cuda.synchronize()
    ks = []
    for e in prof.events():
      if e.device_type == torch.autograd.DeviceType.CUDA:
        ks.append({"name": e.name, "us": round(e.device_time if hasattr(e, "device_time") else e.cuda_time, 2)})
    return ks

  eager = profile(fn)
  compiled = torch.compile(fn)
  out = compiled(X)
  torch.cuda.synchronize()
  inductor_kernels = profile(compiled)
  err = float((out.double() - ref.double()).abs().max())

  dbg = {}
  for pat, key in [("**/fx_graph_readable.py", "fx_graph"), ("**/ir_pre_fusion.txt", "ir_pre_fusion"),
                   ("**/ir_post_fusion.txt", "ir_post_fusion"), ("**/output_code.py", "output_code")]:
    files = sorted(glob.glob(f"{work}/{pat}", recursive=True), key=os.path.getmtime)
    if files: dbg[key] = clip(open(files[-1]).read())
    dbg[key + "_files"] = len(files)

  # Inductor compiles Triton under its own cache dir; one directory per compiled variant (autotune configs)
  triton, seen = [], set()
  for ptx in sorted(glob.glob(f"{work}/**/*.ptx", recursive=True)):
    base = ptx[:-4]
    k = {"name": os.path.basename(base), "cache_dir": os.path.basename(os.path.dirname(base))}
    for ext in ("ttir", "ttgir", "llir", "ptx"):
      if os.path.exists(f"{base}.{ext}"): k[ext] = clip(open(f"{base}.{ext}").read())
    if k["ptx"]["text"] in seen: continue
    seen.add(k["ptx"]["text"])
    if os.path.exists(base + ".cubin"):
      s = subprocess.run(["cuobjdump", "-sass", base + ".cubin"], capture_output=True, text=True)
      k["sass"] = clip(s.stdout)
      r = subprocess.run(["cuobjdump", "-res-usage", base + ".cubin"], capture_output=True, text=True)
      k["res_usage"] = r.stdout.strip()[-800:]
    metas = [m for m in glob.glob(os.path.dirname(base) + "/*.json") if "__grp__" not in m]
    if metas:
      try: k["metadata"] = {kk: v for kk, v in json.load(open(metas[0])).items() if kk in ("num_warps", "num_stages", "shared", "name", "target")}
      except Exception: pass
    triton.append(k)

  res = {"example": name, "variant": "torch", "torch_version": torch.__version__,
         "triton_version": getattr(__import__("triton"), "__version__", None), "gpu": torch.cuda.get_device_name(0),
         "eager_kernels": eager, "inductor_kernels": inductor_kernels, "inductor_debug": dbg, "triton_kernels": triton,
         "check": {"max_abs_err_compiled_vs_eager": err}}
  os.makedirs(f"{out_dir}/{name}", exist_ok=True)
  json.dump(res, open(f"{out_dir}/{name}/torch.json", "w"), indent=1)
  print(f"{name}/torch: eager {[k['name'][:40] for k in eager]}, inductor {len(inductor_kernels)} launches, "
        f"{len(triton)} triton kernels, err {err:.2e}, debug files {[k for k in dbg if not k.endswith('_files')]}")


if __name__ == "__main__":
  main()
