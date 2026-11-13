# From Python to SASS

A book on how tinygrad 0.14.0 (commit `6f87158`) lowers `Tensor` code to CUDA C, PTX and SASS on an
NVIDIA L4 (sm_89), with PyTorch eager and `torch.compile` (Inductor + Triton) side by side.

Open `index.html` in a browser. It is static HTML with no server and no network access: every IR dump,
kernel, PTX and SASS listing comes from `data/*.js`, which were generated from real L4 runs.

| Chapter | Topic |
|---|---|
| `index.html` | Preface: the whole pipeline, both lanes, all 21 examples |
| `ch01-journey.html` | `a + b` from Python to SASS, every pass |
| `ch02-uops.html` | UOps, hash-consing, UPat/PatternMatcher, `graph_rewrite` |
| `ch03-scheduling.html` | Scheduling, rangeify and fusion (softmax, layernorm) |
| `ch04-movement.html` | Movement ops become index math; coalescing |
| `ch05-reductions.html` | Reductions, group reduce, `split_reduceop` |
| `ch06-optimizer.html` | OptOps, the heuristic, BEAM search (matmul) |
| `ch07-tensor-cores.html` | Tensor cores: WMMA, `mma.sync`, `HMMA` |
| `ch08-decompositions.html` | exp/sin/integer math decompositions |
| `ch09-rendering.html` | Linearizing, the CUDA C and PTX renderers, NVRTC, ptxas |
| `ch10-runtime.html` | Runtime, TinyJit, autograd |
| `ch11-toolbox.html` | Appendix: DEBUG levels, VIZ, env vars, disassembly tools |

## Regenerating the data

`harness/` holds the capture scripts:

- `examples.py`: the 21 examples and their variants (NOOPT, BEAM, TC=0, TRANSCENDENTAL=2, PTX, ...).
- `capture.py`: runs one tinygrad variant. It hooks `graph_rewrite` to record the IR after every named
  pass, then saves the kernel AST, uops, CUDA C, PTX, SASS (ptxas + cuobjdump), timings and a NumPy check.
- `capture_torch.py`: eager kernel names from the profiler, Dynamo/AOT graphs, Inductor IR and output
  code, and Triton TTIR/TTGIR/LLVM IR/PTX/SASS.
- `jit_demo.py`: the TinyJit capture.
- `run_all.py OUT_DIR`: runs every example and variant in its own process (`PARALLEL=0`).
- `build_data.py`: converts `captures/*.json` into `data/*.js`.

On a CUDA machine with tinygrad, torch and the CUDA toolkit (`ptxas`, `cuobjdump`) on `PATH`:

```sh
python harness/run_all.py captures
python harness/build_data.py
```

`captures/` is the raw L4 output (tinygrad 0.14.0, PyTorch 2.14.1+cu130, Triton 3.8.0).
`AUTHORING.md` is the contract the chapters were written against: the widgets and their attributes.

Timings: tinygrad times each launch with CUDA events, PyTorch with the profiler's device time, and
the 48 MB L2 is warm in both. Compare times within one tool; the chapters point this out where it matters.
