//! Rust (cuda-oxide) version of streams.cu and async_mma.cu.
//! Part 1: copy -> kernel -> copy over 16 chunks, on 1 vs 4 streams.
//! Part 2: one warp per 16x8 tile of C = A*B on tensor cores (mma.sync m16n8k16),
//!         fed by cp.async vs plain loads. Needs sm_80+.
//! Run: cargo oxide run   (cargo oxide inspect prints the PTX)

use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig, PinnedHostBuffer};
use cuda_device::async_copy::{cp_async_cg_16, cp_async_commit_group, cp_async_wait_group};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread, warp, wmma};
use cuda_host::cuda_module;

const CHUNKS: usize = 16;
const CHUNK: usize = 4 << 20; // floats per chunk; 256 MB total
const M: usize = 2048;
const N: usize = 2048;
const K: usize = 2048;
const KW: usize = K / 2; // row length in u32 words (two f16 per word)

/// 16 bytes with 16-byte alignment, so plain copies become one vector load/store.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct V4([u32; 4]);

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn work(mut data: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(x) = data.get_mut(idx) {
            let mut v = *x;
            for _ in 0..200 {
                v = v * 0.999 + 0.001;
            }
            *x = v;
        }
    }

    /// A: M x K f16 row-major, Bt: N x K f16 row-major, both packed as u32 pairs.
    #[kernel]
    pub unsafe fn gemm<const ASYNC: bool>(a: *const u32, bt: *const u32, c: *mut f32) {
        // Two buffers of a 16x16 A slice and an 8x16 B slice, in u32 words.
        static mut AS: SharedArray<u32, 256, 16> = SharedArray::UNINIT;
        static mut BS: SharedArray<u32, 128, 16> = SharedArray::UNINIT;
        let lane = thread::threadIdx_x() as usize;
        let (g, t) = (lane >> 2, lane & 3);
        let row0 = thread::blockIdx_y() as usize * 16;
        let col0 = thread::blockIdx_x() as usize * 8;
        let as_ptr = core::ptr::addr_of_mut!(AS) as *mut u32;
        let bs_ptr = core::ptr::addr_of_mut!(BS) as *mut u32;

        // Each lane moves 16 bytes of the A slice; lanes 0-15 also of the B slice.
        let load = |buf: usize, kw0: usize| unsafe {
            let ga = a.add((row0 + lane / 2) * KW + kw0 + (lane % 2) * 4);
            let sa = as_ptr.add(buf * 128 + (lane / 2) * 8 + (lane % 2) * 4);
            if ASYNC {
                cp_async_cg_16(sa, ga);
            } else {
                *(sa as *mut V4) = *(ga as *const V4);
            }
            if lane < 16 {
                let gb = bt.add((col0 + lane / 2) * KW + kw0 + (lane % 2) * 4);
                let sb = bs_ptr.add(buf * 64 + (lane / 2) * 8 + (lane % 2) * 4);
                if ASYNC {
                    cp_async_cg_16(sb, gb);
                } else {
                    *(sb as *mut V4) = *(gb as *const V4);
                }
            }
            if ASYNC {
                cp_async_commit_group();
            }
        };

        let mut acc = [0.0f32; 4];
        let tiles = K / 16;
        load(0, 0);
        for i in 0..tiles {
            let buf = i & 1;
            unsafe {
                if i + 1 < tiles {
                    load(buf ^ 1, (i + 1) * 8); // start the next slice...
                    if ASYNC {
                        cp_async_wait_group(1); // ...but only wait for the current one
                    }
                } else if ASYNC {
                    cp_async_wait_group(0);
                }
                warp::sync_mask(0xffff_ffff);
                // Fragment layout for mma.m16n8k16.row.col (PTX ISA, "Matrix Fragments").
                let sa = as_ptr.add(buf * 128);
                let sb = bs_ptr.add(buf * 64);
                let fa = [
                    *sa.add(g * 8 + t),
                    *sa.add((g + 8) * 8 + t),
                    *sa.add(g * 8 + t + 4),
                    *sa.add((g + 8) * 8 + t + 4),
                ];
                let fb = [*sb.add(g * 8 + t), *sb.add(g * 8 + t + 4)];
                acc = wmma::mma_m16n8k16_f32_f16(acc, fa, fb); // tensor cores
                warp::sync_mask(0xffff_ffff); // buffer is overwritten next iteration
            }
        }
        unsafe {
            let r0 = (row0 + g) * N + col0 + t * 2;
            let r1 = (row0 + g + 8) * N + col0 + t * 2;
            *c.add(r0) = acc[0];
            *c.add(r0 + 1) = acc[1];
            *c.add(r1) = acc[2];
            *c.add(r1 + 1) = acc[3];
        }
    }
}

fn expected(mut v: f32) -> f32 {
    for _ in 0..200 {
        v = v * 0.999 + 0.001;
    }
    v
}

fn streams_test(ctx: &Arc<CudaContext>, main: &Arc<CudaStream>, module: &kernels::LoadedModule) -> Result<(), Box<dyn Error>> {
    let streams: Vec<_> = (0..4).map(|_| main.fork()).collect::<Result<_, _>>()?;
    let mut hosts: Vec<_> = (0..CHUNKS).map(|_| PinnedHostBuffer::<f32>::zeroed(ctx, CHUNK)).collect::<Result<_, _>>()?;
    let mut devs: Vec<_> = (0..CHUNKS).map(|_| DeviceBuffer::<f32>::zeroed(main, CHUNK)).collect::<Result<_, _>>()?;
    main.synchronize()?;
    for ns in [1usize, 4, 1, 4] {
        // each config twice; the first pair is warm-up
        for (c, h) in hosts.iter_mut().enumerate() {
            for (i, x) in h.as_mut_slice().iter_mut().enumerate() {
                *x = ((c * CHUNK + i) % 1000) as f32;
            }
        }
        let t0 = Instant::now();
        for c in 0..CHUNKS {
            let s = &streams[c % ns];
            // SAFETY: buffers stay alive and untouched until the streams sync below.
            unsafe {
                devs[c].copy_from_pinned_host_async(s, &hosts[c])?;
                module.work(s, LaunchConfig::for_num_elems(CHUNK as u32), &mut devs[c])?;
                devs[c].copy_to_pinned_host_async(s, &mut hosts[c])?;
            }
        }
        for s in &streams[..ns] {
            s.synchronize()?;
        }
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let bad = (0..CHUNKS * CHUNK)
            .step_by(997)
            .filter(|&j| {
                let e = expected((j % 1000) as f32); // GPU fuses into FMA
                (hosts[j / CHUNK].as_slice()[j % CHUNK] - e).abs() > 1e-4 * (1.0 + e.abs())
            })
            .count();
        println!("{ns} stream(s): {ms:7.2} ms  {}", if bad > 0 { "FAIL" } else { "results OK" });
    }
    // Forked streams must be joined back into their parent before they are dropped.
    for s in &streams {
        main.join(s).map_err(|e| format!("join: {e:?}"))?;
    }
    Ok(())
}

fn va(i: usize) -> f32 {
    ((i * 7 % 17) as i32 - 8) as f32 / 8.0
}
fn vb(i: usize) -> f32 {
    ((i * 5 % 13) as i32 - 6) as f32 / 4.0
}
/// f32 -> f16 bits; exact for the small dyadic values used here.
fn f16_bits(x: f32) -> u16 {
    if x == 0.0 {
        return 0;
    }
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = (((b >> 23) & 0xff) as i32 - 127 + 15) as u16;
    sign | (exp << 10) | ((b >> 13) & 0x3ff) as u16
}
fn pack(len: usize, f: fn(usize) -> f32) -> Vec<u32> {
    (0..len / 2).map(|w| f16_bits(f(2 * w)) as u32 | (f16_bits(f(2 * w + 1)) as u32) << 16).collect()
}

fn mma_test(main: &Arc<CudaStream>, module: &kernels::LoadedModule) -> Result<(), Box<dyn Error>> {
    let a = DeviceBuffer::from_host(main, &pack(M * K, va)).map_err(|e| format!("upload A: {e:?}"))?;
    let bt = DeviceBuffer::from_host(main, &pack(N * K, vb)).map_err(|e| format!("upload B: {e:?}"))?;
    let c = DeviceBuffer::<f32>::zeroed(main, M * N).map_err(|e| format!("alloc C: {e:?}"))?;
    let cfg = LaunchConfig { grid_dim: ((N / 8) as u32, (M / 16) as u32, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
    for asynchronous in [false, true] {
        let mut best = f32::MAX;
        for _ in 0..5 {
            let timed = Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT); // None = no timing
            let start = main.record_event(timed).map_err(|e| format!("event: {e:?}"))?;
            // SAFETY: grid covers M x N exactly with one warp per 16x8 tile; buffers live until sync.
            unsafe {
                let (pa, pb, pc) = (a.cu_deviceptr() as *const u32, bt.cu_deviceptr() as *const u32, c.cu_deviceptr() as *mut f32);
                if asynchronous {
                    module.gemm::<true>(main, cfg, pa, pb, pc).map_err(|e| format!("launch gemm<true>: {e:?}"))?;
                } else {
                    module.gemm::<false>(main, cfg, pa, pb, pc).map_err(|e| format!("launch gemm<false>: {e:?}"))?;
                }
            }
            let end = main.record_event(timed)?;
            best = best.min(start.elapsed_ms(&end).map_err(|e| format!("elapsed: {e:?}"))?);
        }
        let h = c.to_host_vec(main).map_err(|e| format!("download C: {e:?}"))?;
        let bad = (0..4096usize)
            .filter(|s| {
                let (r, col) = (s * 7919 % M, s * 104729 % N);
                let r#ref: f64 = (0..K).map(|k| va(r * K + k) as f64 * vb(col * K + k) as f64).sum();
                (h[r * N + col] as f64 - r#ref).abs() > 1e-2 * (1.0 + r#ref.abs())
            })
            .count();
        println!(
            "{:<26} {best:7.3} ms  {:6.2} TFLOP/s  {}",
            if asynchronous { "cp.async + mma.sync:" } else { "plain loads + mma.sync:" },
            2.0 * (M * N * K) as f64 / best as f64 / 1e9,
            if bad > 0 { "FAIL" } else { "results OK (4096 checked)" }
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let main = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    streams_test(&ctx, &main, &module)?;
    mma_test(&main, &module)
}
