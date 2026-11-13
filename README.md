# accel

GPU experiments run on Google Colab GPUs from a cloud sandbox with the
[Colab CLI](https://github.com/googlecolab/google-colab-cli): CUDA C++, Mojo,
tinygrad and Rust (NVIDIA's cuda-oxide), plus the debugger and profilers.

| Directory | What it is |
|---|---|
| `colab/colab_sh` | Runs a shell command on a Colab session and prints its output (the Colab kernel only shows Python-level output). |
| `cuda/` | CUDA vector add checked against the CPU; `run_on_colab.py` compiles and runs it with `nvcc` on the VM. |
| `tinygrad/` | Matmul + ReLU on tinygrad's `CUDA` and `NV` backends, checked against NumPy. |
| `mojo/` | Mojo GPU vector add and the `cuda-gdb` command file used to debug it. |
| `async/` | Streams, `cp.async` and tensor cores in C++ (`*.cu`), Mojo (`async_test.mojo`) and Rust (`rust/`). |
| `async/asm/` | PTX and SASS of the async/tensor core kernels from all three languages. |
| `profiles/` | Nsight Systems (`.nsys-rep`) and Nsight Compute (`.ncu-rep`) reports, C++ (T4) and Mojo (A100). |
| `book/` | *From Python to SASS*: an interactive HTML book on how tinygrad lowers kernels to CUDA C, PTX and SASS on an L4, next to PyTorch. Open `book/index.html`. |
| `setup/` | Installs CUDA 13, Mojo, LLVM 21 and cuda-oxide on a GPU-less machine, and builds the Rust code. |

## Results

**Tools** (T4 and A100): `cuda-gdb`, `compute-sanitizer`, Nsight Systems and Nsight
Compute all work on CUDA C++ and Mojo kernels. tinygrad 0.14.0 works through its
`CUDA` backend; its `NV` backend needs Ampere or newer. Mojo 1.2 GPU code lives
in the `max` package (`from max.gpu import ...`). Neither Mojo nor tinygrad
supports TPUs today.

**Async experiments** (L4, sm_89):

| | C++ CUDA | Mojo | Rust |
|---|---|---|---|
| copy -> kernel -> copy, 1 stream | 43.7 ms | 43.7 ms | 43.7 ms |
| same, 4 streams | 26.9 ms | 26.9 ms | 26.8 ms |
| matmul, plain loads + `mma.sync` | 1.55 ms | 1.56 ms | 1.56 ms |
| matmul, `cp.async` + `mma.sync` | 2.54 ms | 2.55 ms | 2.54 ms |

All three compile to the same SASS: `cp.async` -> `LDGSTS.E.BYPASS.128`,
commit/wait -> `LDGDEPBAR` / `DEPBAR.LE SB0`, tensor cores -> `HMMA.16816.F32`.

## What is async and what is not

- **Streams** are host-side queues. Work in one stream runs in order; different
  streams overlap (the L4 has 2 copy engines), which is the 43.7 -> 26.9 ms above.
- **Warps are not async.** A warp issues instructions in order and stalls when it
  needs data that has not arrived ("long scoreboard" in Nsight Compute); the
  scheduler hides this by switching to another ready warp.
- **`cp.async`** (Ampere+) is async: the warp starts a global -> shared copy and
  keeps going until `cp.async.wait_group`. It cut memory-wait stalls from 25 to
  3.9 per issued instruction, but was not faster here: tiles are tiny, `.cg`
  bypasses L1 (`.ca` gives 1.62 ms), and the scheduler was already hiding latency.
- **Tensor cores on these GPUs are not async.** `mma.sync` (`HMMA` on T4, L4,
  A100, G4) is synchronous: the whole warp issues it and waits for the result
  in registers. Async tensor core instructions exist only on Hopper
  (`wgmma.mma_async`, H100) and datacenter Blackwell (`tcgen05.mma`, B200),
  neither of which this Colab plan offers.

## Gotchas found

- Mojo: `DeviceStream` can only launch kernels; each `DeviceContext` is its own
  CUDA context, so per-stream buffers and kernels must be created on that context.
  `load[width=8]()` without `alignment=16` compiles to scalar 2-byte loads (5x slower).
- Rust (cuda-oxide): forked streams must be `join`ed into the parent before they
  are dropped, or later calls fail with `invalid resource handle`. A loop of four
  `u32` copies compiles to 4-byte loads; copying one `#[repr(align(16))]` value
  gives `ld.global.v4.b32`.
- Colab bills for the time a GPU is assigned, idle or not, not in hourly blocks
  (A100 6.73 units/h, L4 1.94 units/h).
