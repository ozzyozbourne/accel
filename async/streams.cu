// Experiment 1: CUDA streams. The same work (copy chunk to GPU -> kernel ->
// copy back) runs on 1 stream, then spread over 4 streams. Streams are host-side
// queues: work in one stream runs in order, work in different streams may overlap.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// Compute-heavy enough that kernel time is similar to copy time.
__global__ void work(float *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = x[i];
  for (int k = 0; k < 200; k++) v = v * 0.999f + 0.001f;
  x[i] = v;
}

static float expected(float v) {
  for (int k = 0; k < 200; k++) v = v * 0.999f + 0.001f;
  return v;
}

int main() {
  cudaDeviceProp p;
  CHECK(cudaGetDeviceProperties(&p, 0));
  printf("GPU: %s, copy engines (asyncEngineCount) = %d, concurrentKernels = %d\n",
         p.name, p.asyncEngineCount, p.concurrentKernels);

  const int chunks = 16, chunk = 4 << 20;  // 16 chunks x 4M floats = 256 MB
  const size_t n = (size_t)chunks * chunk, bytes = n * sizeof(float);
  float *h, *d;
  CHECK(cudaMallocHost(&h, bytes));  // pinned host memory: required for async copies
  CHECK(cudaMalloc(&d, bytes));

  cudaStream_t s[4];
  for (auto &st : s) CHECK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  cudaEvent_t t0, t1;
  CHECK(cudaEventCreate(&t0));
  CHECK(cudaEventCreate(&t1));

  for (int nstreams : {1, 4, 1, 4}) {  // each config twice; the first pair is warm-up
    for (size_t i = 0; i < n; i++) h[i] = (float)(i % 1000);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaEventRecord(t0));
    for (int c = 0; c < chunks; c++) {
      cudaStream_t st = s[c % nstreams];
      float *hc = h + (size_t)c * chunk, *dc = d + (size_t)c * chunk;
      CHECK(cudaMemcpyAsync(dc, hc, chunk * sizeof(float), cudaMemcpyHostToDevice, st));
      work<<<chunk / 256, 256, 0, st>>>(dc, chunk);
      CHECK(cudaMemcpyAsync(hc, dc, chunk * sizeof(float), cudaMemcpyDeviceToHost, st));
    }
    // The host reaches this line almost immediately: everything above only queued work.
    for (auto &st : s) CHECK(cudaStreamSynchronize(st));
    CHECK(cudaEventRecord(t1));
    CHECK(cudaEventSynchronize(t1));
    float ms;
    CHECK(cudaEventElapsedTime(&ms, t0, t1));
    int bad = 0;
    for (size_t i = 0; i < n; i += 997)
      { float e = expected((float)(i % 1000)); if (fabsf(h[i] - e) > 1e-4f * (1 + fabsf(e))) bad++; }  // GPU fuses into FMA
    printf("%d stream(s): %7.2f ms  %s\n", nstreams, ms, bad ? "FAIL" : "results OK");
  }
  return 0;
}
