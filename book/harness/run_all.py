"""Run every example and variant in its own process (so env settings and caches start fresh).
Usage: python run_all.py OUT_DIR [example ...]"""
import json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from examples import EXAMPLES

out = sys.argv[1]
names = sys.argv[2:] or list(EXAMPLES)
here = os.path.dirname(os.path.abspath(__file__))
summary = []
for name in names:
  for variant, env in EXAMPLES[name].get("variants", {"default": {}}).items():
    e = {**os.environ, "PARALLEL": "0", "DEV": "CUDA", **env}
    t0 = time.time()
    r = subprocess.run([sys.executable, f"{here}/capture.py", name, variant, out], env=e, capture_output=True, text=True, timeout=1800)
    line = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    print(f"[{time.time()-t0:6.1f}s] {line if r.returncode == 0 else 'FAILED ' + name + '/' + variant + chr(10) + r.stderr[-3000:]}", flush=True)
    summary.append({"example": name, "variant": variant, "ok": r.returncode == 0, "line": line})
  if not EXAMPLES[name].get("same_as"):
    r = subprocess.run([sys.executable, f"{here}/capture_torch.py", name, out], capture_output=True, text=True, timeout=1800,
                       env={**os.environ, "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1"})
    line = (r.stdout.strip().splitlines() or ["(no output)"])[-1]
    print(f"   torch: {line if r.returncode == 0 else 'FAILED ' + r.stderr[-3000:]}", flush=True)
    summary.append({"example": name, "variant": "torch", "ok": r.returncode == 0, "line": line})
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
