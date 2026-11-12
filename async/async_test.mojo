# Mojo version of streams.cu and async_mma.cu.
# Part 1: copy -> kernel -> copy over 16 chunks, on 1 vs 4 streams. In Mojo each
#         DeviceContext is one stream (DeviceStream can only launch kernels).
# Part 2: one warp per 16x8 tile of C = A*B with tensor cores (mma -> mma.sync
#         m16n8k16), feeding them with async_copy (cp.async) vs plain loads.
# Needs sm_80+. Run: mojo run async_test.mojo  (PTX is dumped to /content).
from std.math import abs
from std.time import perf_counter_ns

from max.gpu import block_idx, thread_idx, syncwarp
from max.gpu.host import DeviceContext
from max.gpu.memory import (
    AddressSpace,
    async_copy,
    async_copy_commit_group,
    async_copy_wait_group,
)
from max.gpu.compute.mma import mma
from std.memory import stack_allocation

# ---------------------------------------------------------------- part 1
comptime CHUNKS = 16
comptime CHUNK = 4 << 20  # floats per chunk; 256 MB total


def work(x: Pointer[Float32, MutAnyOrigin], n: Int32):
    var i = Int(block_idx.x) * 256 + Int(thread_idx.x)
    if i >= Int(n):
        return
    var v = x[unsafe_offset=i]
    for _ in range(200):
        v = v * 0.999 + 0.001
    x[unsafe_offset=i] = v


def expected(v0: Float32) -> Float32:
    var v = v0
    for _ in range(200):
        v = v * 0.999 + 0.001
    return v


def streams_test() raises:
    var ctxs = List[DeviceContext]()
    for _ in range(4):
        ctxs.append(DeviceContext())
    print("GPU:", ctxs[0].name())
    var fs = List[type_of(ctxs[0].compile_function[work]())]()
    for i in range(4):  # a compiled kernel belongs to one context
        fs.append(ctxs[i].compile_function[work]())
    for ns in [1, 4, 1, 4]:  # each config twice; the first pair is warm-up
        # Each DeviceContext is its own CUDA context, so a chunk's buffers must be
        # allocated on the context (stream) that processes it.
        var hosts = List[type_of(ctxs[0].enqueue_create_host_buffer[DType.float32](1))]()
        var devs = List[type_of(ctxs[0].enqueue_create_buffer[DType.float32](1))]()
        for c in range(CHUNKS):
            hosts.append(ctxs[c % ns].enqueue_create_host_buffer[DType.float32](CHUNK))
            devs.append(ctxs[c % ns].enqueue_create_buffer[DType.float32](CHUNK))
        for i in range(ns):
            ctxs[i].synchronize()
        for c in range(CHUNKS):
            for i in range(CHUNK):
                hosts[c][i] = Float32((c * CHUNK + i) % 1000)
        var t0 = perf_counter_ns()
        for c in range(CHUNKS):
            ref ctx = ctxs[c % ns]
            ctx.enqueue_copy(devs[c], hosts[c])
            ctx.enqueue_function(
                fs[c % ns], devs[c].unsafe_ptr(), Int32(CHUNK), grid_dim=CHUNK // 256, block_dim=256
            )
            ctx.enqueue_copy(hosts[c], devs[c])
        for i in range(ns):
            ctxs[i].synchronize()
        var ms = Float64(perf_counter_ns() - t0) / 1e6
        var bad = 0
        for j in range(0, CHUNKS * CHUNK, 997):
            var got = hosts[j // CHUNK][j % CHUNK]
            var e = expected(Float32(j % 1000))
            if abs(got - e) > 1e-4 * (1 + abs(e)):  # GPU fuses into FMA
                bad += 1
        print(ns, "stream(s):", ms, "ms", "FAIL" if bad else "results OK")


# ---------------------------------------------------------------- part 2
comptime M = 2048
comptime N = 2048
comptime K = 2048


def gemm[ASYNC: Bool](
    A: Pointer[Float16, MutAnyOrigin],
    Bt: Pointer[Float16, MutAnyOrigin],
    C: Pointer[Float32, MutAnyOrigin],
):
    var As = stack_allocation[
        2 * 16 * 16, Float16, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        2 * 8 * 16, Float16, alignment=16, address_space=AddressSpace.SHARED
    ]()
    var lane = Int(thread_idx.x)
    var g = lane >> 2
    var t = lane & 3
    var row0 = Int(block_idx.y) * 16
    var col0 = Int(block_idx.x) * 8

    @parameter
    def load(buf: Int, k0: Int):
        var ga = A + ((row0 + lane // 2) * K + k0 + (lane % 2) * 8)
        var sa = As + (buf * 256 + (lane // 2) * 16 + (lane % 2) * 8)
        comptime if ASYNC:
            async_copy[16](ga.unsafe_address_space_cast[AddressSpace.GLOBAL](), sa)
        else:
            sa.store[alignment=16](ga.load[width=8, alignment=16]())  # 16-byte vector ops
        if lane < 16:
            var gb = Bt + ((col0 + lane // 2) * K + k0 + (lane % 2) * 8)
            var sb = Bs + (buf * 128 + (lane // 2) * 16 + (lane % 2) * 8)
            comptime if ASYNC:
                async_copy[16](gb.unsafe_address_space_cast[AddressSpace.GLOBAL](), sb)
            else:
                sb.store[alignment=16](gb.load[width=8, alignment=16]())
        comptime if ASYNC:
            async_copy_commit_group()

    var c = SIMD[DType.float32, 4](0)
    comptime tiles = K // 16
    load(0, 0)
    for i in range(tiles):
        var buf = i & 1
        if i + 1 < tiles:
            load(buf ^ 1, (i + 1) * 16)  # start the next slice...
            comptime if ASYNC:
                async_copy_wait_group(1)  # ...but only wait for the current one
        else:
            comptime if ASYNC:
                async_copy_wait_group(0)
        syncwarp()
        # Fragment layout for mma.m16n8k16.row.col (PTX ISA, "Matrix Fragments").
        var a_base = buf * 256
        var b_base = buf * 128
        # 32-bit shared loads (pairs of f16), like the C++ version.
        var pa = As + a_base
        var pb = Bs + b_base
        var a = (
            (pa + g * 16 + t * 2).load[width=2, alignment=4]()
            .join((pa + (g + 8) * 16 + t * 2).load[width=2, alignment=4]())
            .join(
                (pa + g * 16 + t * 2 + 8).load[width=2, alignment=4]()
                .join((pa + (g + 8) * 16 + t * 2 + 8).load[width=2, alignment=4]())
            )
        )
        var b = (pb + g * 16 + t * 2).load[width=2, alignment=4]().join(
            (pb + g * 16 + t * 2 + 8).load[width=2, alignment=4]()
        )
        mma(c, a, b, c)  # tensor cores: mma.sync.m16n8k16.row.col.f32.f16.f16.f32
        syncwarp()  # the buffer read above is overwritten next iteration
    C[unsafe_offset=(row0 + g) * N + col0 + t * 2] = c[0]
    C[unsafe_offset=(row0 + g) * N + col0 + t * 2 + 1] = c[1]
    C[unsafe_offset=(row0 + g + 8) * N + col0 + t * 2] = c[2]
    C[unsafe_offset=(row0 + g + 8) * N + col0 + t * 2 + 1] = c[3]


def va(i: Int) -> Float32:
    return Float32((i * 7 % 17) - 8) / 8.0


def vb(i: Int) -> Float32:
    return Float32((i * 5 % 13) - 6) / 4.0


def mma_test() raises:
    var ctx = DeviceContext()
    var hA = ctx.enqueue_create_host_buffer[DType.float16](M * K)
    var hB = ctx.enqueue_create_host_buffer[DType.float16](N * K)
    var hC = ctx.enqueue_create_host_buffer[DType.float32](M * N)
    ctx.synchronize()
    for i in range(M * K):
        hA[i] = va(i).cast[DType.float16]()
    for i in range(N * K):
        hB[i] = vb(i).cast[DType.float16]()
    var A = ctx.enqueue_create_buffer[DType.float16](M * K)
    var B = ctx.enqueue_create_buffer[DType.float16](N * K)
    var Cd = ctx.enqueue_create_buffer[DType.float32](M * N)
    ctx.enqueue_copy(A, hA)
    ctx.enqueue_copy(B, hB)

    var f_plain = ctx.compile_function[
        gemm[False], dump_asm=StaticString("/content/mojo_gemm_plain.ptx")
    ]()
    var f_async = ctx.compile_function[
        gemm[True], dump_asm=StaticString("/content/mojo_gemm_async.ptx")
    ]()
    for mode in range(2):
        var best = Float64(1e30)
        for _ in range(5):
            Cd.enqueue_fill(0)
            ctx.synchronize()
            var t0 = perf_counter_ns()
            if mode == 1:
                ctx.enqueue_function(
                    f_async, A.unsafe_ptr(), B.unsafe_ptr(), Cd.unsafe_ptr(),
                    grid_dim=(N // 8, M // 16), block_dim=32,
                )
            else:
                ctx.enqueue_function(
                    f_plain, A.unsafe_ptr(), B.unsafe_ptr(), Cd.unsafe_ptr(),
                    grid_dim=(N // 8, M // 16), block_dim=32,
                )
            ctx.synchronize()
            best = min(best, Float64(perf_counter_ns() - t0) / 1e6)
        ctx.enqueue_copy(hC, Cd)
        ctx.synchronize()
        var bad = 0
        for s in range(4096):  # spot-check against a CPU dot product
            var r = (s * 7919) % M
            var col = (s * 104729) % N
            var ref_val = Float64(0)
            for k in range(K):
                ref_val += Float64(va(r * K + k)) * Float64(vb(col * K + k))
            var got = Float64(hC[r * N + col])
            if abs(got - ref_val) > 1e-2 * (1 + abs(ref_val)):
                bad += 1
        var label = "async_copy + mma:" if mode == 1 else "plain loads + mma:"
        print(
            label, best, "ms", 2.0 * M * N * K / best / 1e9, "TFLOP/s",
            "FAIL" if bad else "results OK (4096 checked)",
        )


def main() raises:
    streams_test()
    mma_test()
