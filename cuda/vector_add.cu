// Adds two vectors on the GPU and checks every element against the CPU.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                        \
  do {                                                                     \
    cudaError_t err = (call);                                              \
    if (err != cudaSuccess) {                                              \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), \
              __FILE__, __LINE__);                                         \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

__global__ void add(const float *a, const float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s, compute %d.%d, %.1f GB, %d SMs\n", prop.name, prop.major,
         prop.minor, prop.totalGlobalMem / 1e9, prop.multiProcessorCount);

  // Not a multiple of the block size, so the last block exercises the bounds check.
  const int n = (1 << 24) + 7;
  size_t bytes = n * sizeof(float);
  float *a = (float *)malloc(bytes), *b = (float *)malloc(bytes),
        *c = (float *)malloc(bytes);
  for (int i = 0; i < n; i++) {
    a[i] = (float)(i % 1000);
    b[i] = 2.0f * (i % 777) + 1.0f;
  }

  float *da, *db, *dc;
  CHECK(cudaMalloc(&da, bytes));
  CHECK(cudaMalloc(&db, bytes));
  CHECK(cudaMalloc(&dc, bytes));
  CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemset(dc, 0, bytes));

  int threads = 256, blocks = (n + threads - 1) / threads;
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  CHECK(cudaEventRecord(start));
  add<<<blocks, threads>>>(da, db, dc, n);
  CHECK(cudaGetLastError());
  CHECK(cudaEventRecord(stop));
  CHECK(cudaEventSynchronize(stop));
  float ms;
  CHECK(cudaEventElapsedTime(&ms, start, stop));
  CHECK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int i = 0; i < n; i++)
    if (c[i] != a[i] + b[i]) {
      if (bad++ < 5) printf("mismatch at %d: %f != %f\n", i, c[i], a[i] + b[i]);
    }
  printf("n=%d, blocks=%d, kernel %.3f ms, %.1f GB/s\n", n, blocks, ms,
         3.0 * bytes / ms / 1e6);
  printf("c[0]=%g c[n-1]=%g\n", c[0], c[n - 1]);
  printf(bad ? "FAIL: %d mismatches\n" : "PASS: all elements match%.0d\n", bad);

  cudaFree(da); cudaFree(db); cudaFree(dc);
  free(a); free(b); free(c);
  return bad ? 1 : 0;
}
