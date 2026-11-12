# Adds two vectors on the GPU with Mojo and checks every element on the CPU.
from std.math import ceildiv
from std.sys import has_accelerator

# GPU modules live in the `max` package (pip install modular), not in std.
from max.gpu.host import DeviceContext
from max.gpu import block_dim, block_idx, thread_idx
from layout import TileTensor, row_major

comptime dtype = DType.float32
comptime n = (1 << 20) + 7  # not a multiple of block_size, so the bounds check matters
comptime layout = row_major[n]()
comptime block_size = 256
comptime num_blocks = ceildiv(n, block_size)


def add(
    a: TileTensor[dtype, type_of(layout), ImmutAnyOrigin],
    b: TileTensor[dtype, type_of(layout), ImmutAnyOrigin],
    c: TileTensor[dtype, type_of(layout), MutAnyOrigin],
):
    var i = block_idx.x * block_dim.x + thread_idx.x
    if i < n:
        c[i] = a[i] + b[i]


def main() raises:
    comptime if not has_accelerator():
        print("FAIL: no compatible GPU found")
    else:
        var ctx = DeviceContext()
        print("GPU:", ctx.name())

        var a_host = ctx.enqueue_create_host_buffer[dtype](n)
        var b_host = ctx.enqueue_create_host_buffer[dtype](n)
        ctx.synchronize()
        for i in range(n):
            a_host[i] = Float32(i % 1000)
            b_host[i] = Float32(2 * (i % 777) + 1)

        var a_dev = ctx.enqueue_create_buffer[dtype](n)
        var b_dev = ctx.enqueue_create_buffer[dtype](n)
        var c_dev = ctx.enqueue_create_buffer[dtype](n)
        ctx.enqueue_copy(dst_buf=a_dev, src_buf=a_host)
        ctx.enqueue_copy(dst_buf=b_dev, src_buf=b_host)

        ctx.enqueue_function[add](
            TileTensor(a_dev, layout),
            TileTensor(b_dev, layout),
            TileTensor(c_dev, layout),
            grid_dim=num_blocks,
            block_dim=block_size,
        )

        var c_host = ctx.enqueue_create_host_buffer[dtype](n)
        ctx.enqueue_copy(dst_buf=c_host, src_buf=c_dev)
        ctx.synchronize()

        var bad = 0
        for i in range(n):
            if c_host[i] != a_host[i] + b_host[i]:
                bad += 1
        print("n =", n, "blocks =", num_blocks, "c[n-1] =", c_host[n - 1])
        if bad == 0:
            print("PASS: all elements match")
        else:
            print("FAIL:", bad, "mismatches")
