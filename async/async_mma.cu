// Experiment 2: async copies feeding tensor cores, inside one kernel.
// One warp computes a 16x8 tile of C = A * B with mma.sync (tensor cores),
// double-buffering 16-wide K slices through shared memory.
//   ASYNC=true : cp.async (Ampere+) starts the copy of slice k+1, then the warp
//                runs the MMA on slice k while that copy is in flight.
//   ASYNC=false: plain loads; the warp must wait for each load before storing it.
// Needs sm_80+ (A100, L4). Build: nvcc -O3 -arch=sm_80 async_mma.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem) {
  unsigned s = (unsigned)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// A: M x K row-major. Bt: N x K row-major (B stored column-major). C: M x N.
template <bool ASYNC>
__global__ void gemm(const half *A, const half *Bt, float *C, int N, int K) {
  __shared__ __align__(16) half As[2][16][16];
  __shared__ __align__(16) half Bs[2][8][16];
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  const int row0 = blockIdx.y * 16, col0 = blockIdx.x * 8;

  // Each lane moves 16 bytes (8 halves) of the A slice; lanes 0-15 also of the B slice.
  auto load = [&](int buf, int k0) {
    const half *ga = A + (size_t)(row0 + lane / 2) * K + k0 + (lane % 2) * 8;
    half *sa = &As[buf][lane / 2][(lane % 2) * 8];
    if constexpr (ASYNC) cp_async16(sa, ga); else *(int4 *)sa = *(const int4 *)ga;
    if (lane < 16) {
      const half *gb = Bt + (size_t)(col0 + lane / 2) * K + k0 + (lane % 2) * 8;
      half *sb = &Bs[buf][lane / 2][(lane % 2) * 8];
      if constexpr (ASYNC) cp_async16(sb, gb); else *(int4 *)sb = *(const int4 *)gb;
    }
    if constexpr (ASYNC) cp_async_commit();
  };

  float c[4] = {0.f, 0.f, 0.f, 0.f};
  const int tiles = K / 16;
  load(0, 0);
  for (int i = 0; i < tiles; i++) {
    const int buf = i & 1;
    if (i + 1 < tiles) {
      load(buf ^ 1, (i + 1) * 16);              // start next slice...
      if constexpr (ASYNC) cp_async_wait<1>();  // ...but only wait for the current one
    } else if constexpr (ASYNC) {
      cp_async_wait<0>();
    }
    __syncwarp();
    // Fragment layout for mma.m16n8k16.row.col (PTX ISA, "Matrix Fragments").
    unsigned a0 = *(const unsigned *)&As[buf][g][t * 2];
    unsigned a1 = *(const unsigned *)&As[buf][g + 8][t * 2];
    unsigned a2 = *(const unsigned *)&As[buf][g][t * 2 + 8];
    unsigned a3 = *(const unsigned *)&As[buf][g + 8][t * 2 + 8];
    unsigned b0 = *(const unsigned *)&Bs[buf][g][t * 2];
    unsigned b1 = *(const unsigned *)&Bs[buf][g][t * 2 + 8];
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    __syncwarp();  // the buffer read above is overwritten by the next iteration's load
  }
  C[(size_t)(row0 + g) * N + col0 + t * 2] = c[0];
  C[(size_t)(row0 + g) * N + col0 + t * 2 + 1] = c[1];
  C[(size_t)(row0 + g + 8) * N + col0 + t * 2] = c[2];
  C[(size_t)(row0 + g + 8) * N + col0 + t * 2 + 1] = c[3];
}

static float va(size_t i) { return (float)((int)(i * 7 % 17) - 8) / 8.f; }   // exact in fp16
static float vb(size_t i) { return (float)((int)(i * 5 % 13) - 6) / 4.f; }

int main() {
  const int M = 2048, N = 2048, K = 2048;
  cudaDeviceProp p;
  CHECK(cudaGetDeviceProperties(&p, 0));
  printf("GPU: %s (sm_%d%d)\n", p.name, p.major, p.minor);

  std::vector<half> hA((size_t)M * K), hB((size_t)N * K);
  for (size_t i = 0; i < hA.size(); i++) hA[i] = __float2half(va(i));
  for (size_t i = 0; i < hB.size(); i++) hB[i] = __float2half(vb(i));
  half *A, *Bt;
  float *C;
  CHECK(cudaMalloc(&A, hA.size() * 2));
  CHECK(cudaMalloc(&Bt, hB.size() * 2));
  CHECK(cudaMalloc(&C, (size_t)M * N * 4));
  CHECK(cudaMemcpy(A, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(Bt, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice));

  dim3 grid(N / 8, M / 16), block(32);
  cudaEvent_t t0, t1;
  CHECK(cudaEventCreate(&t0));
  CHECK(cudaEventCreate(&t1));
  std::vector<float> hC((size_t)M * N);
  for (int mode = 0; mode < 2; mode++) {
    float best = 1e30f;
    for (int rep = 0; rep < 5; rep++) {
      CHECK(cudaMemset(C, 0, (size_t)M * N * 4));
      CHECK(cudaEventRecord(t0));
      if (mode) gemm<true><<<grid, block>>>(A, Bt, C, N, K);
      else gemm<false><<<grid, block>>>(A, Bt, C, N, K);
      CHECK(cudaGetLastError());
      CHECK(cudaEventRecord(t1));
      CHECK(cudaEventSynchronize(t1));
      float ms;
      CHECK(cudaEventElapsedTime(&ms, t0, t1));
      best = std::min(best, ms);
    }
    CHECK(cudaMemcpy(hC.data(), C, hC.size() * 4, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int s = 0; s < 4096; s++) {  // spot-check against a CPU dot product
      size_t r = (size_t)(s * 7919) % M, col = (size_t)(s * 104729) % N;
      double ref = 0;
      for (int k = 0; k < K; k++) ref += (double)va(r * K + k) * vb(col * K + k);
      if (fabs(hC[r * N + col] - ref) > 1e-2 * (1 + fabs(ref))) bad++;
    }
    printf("%-26s %7.3f ms  %6.2f TFLOP/s  %s\n",
           mode ? "cp.async + mma.sync:" : "plain loads + mma.sync:", best,
           2.0 * M * N * K / best / 1e9, bad ? "FAIL" : "results OK (4096 checked)");
  }
  return 0;
}
