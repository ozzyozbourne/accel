# Runs a matmul + relu on each tinygrad GPU backend and checks it against NumPy.
import sys, time
import numpy as np
import tinygrad
from tinygrad import Tensor, Device

print("tinygrad", tinygrad.__version__ if hasattr(tinygrad, "__version__") else "?",
      "| default device:", Device.DEFAULT)

rng = np.random.default_rng(0)
a = rng.standard_normal((1024, 512), dtype=np.float32)
b = rng.standard_normal((512, 768), dtype=np.float32)
expected = np.maximum(a @ b, 0)

ok = False
for dev in ["CUDA", "NV"]:  # CUDA = runtime/nvrtc backend, NV = tinygrad's direct driver backend
    try:
        out = (Tensor(a, device=dev) @ Tensor(b, device=dev)).relu().numpy()
        t = time.perf_counter()
        for _ in range(10):
            (Tensor(a, device=dev) @ Tensor(b, device=dev)).relu().realize()
        Device[dev].synchronize()
        ms = (time.perf_counter() - t) / 10 * 1e3
        err = np.abs(out - expected).max()
        good = np.allclose(out, expected, rtol=1e-3, atol=1e-3)
        ok |= good
        print(f"{dev}: {'PASS' if good else 'FAIL'} max_abs_err={err:.2e} ~{ms:.2f} ms/iter, "
              f"device={Device[dev].__class__.__name__}")
    except Exception as e:
        import traceback
        print(f"{dev}: unavailable ({type(e).__name__}: {str(e)[:300]!r})")
        traceback.print_exc(limit=-4)
sys.exit(0 if ok else 1)
