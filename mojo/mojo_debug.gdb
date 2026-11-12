# cuda-gdb commands for debugging the add kernel in vector_add.mojo.
# Run: mojo debug --cuda-gdb --X -batch --X -x --X mojo_debug.gdb vector_add.mojo
# Mojo GPU tips (from Mojo's GPU debugging docs and what worked on Colab):
# - The kernel is compiled at run time, so stop at its launch first, then set
#   file:line breakpoints (symbol breakpoints are not supported).
# - Stepping is unreliable on GPU; use breakpoints and continue instead.
# - Mojo locals like `i` show as <unavailable>; use blockIdx/threadIdx/blockDim
#   and read tensor data through each TileTensor's `_storage` pointer.
set pagination off
set breakpoint pending on
set cuda break_on_launch application
run
info cuda kernels
set cuda break_on_launch none
break vector_add.mojo:24
continue
cuda kernel block thread
cuda block (2,0,0) thread (37,0,0)
print blockIdx.x * blockDim.x + threadIdx.x
print a._storage[blockIdx.x * blockDim.x + threadIdx.x]
print b._storage[blockIdx.x * blockDim.x + threadIdx.x]
info args
delete
continue
