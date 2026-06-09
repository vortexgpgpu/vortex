#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

#include "common.h"

#define KERNEL_NAME "sgemm"
#define FLOAT_ULP   6

#define HIP_CHECK(_expr)                                                  \
  do {                                                                    \
    hipError_t _err = (_expr);                                            \
    if (_err != hipSuccess) {                                             \
      fprintf(stderr, "HIP Error: '%s' returned %d (%s)\n",               \
              #_expr, (int)_err, hipGetErrorString(_err));                \
      exit(-1);                                                           \
    }                                                                     \
  } while (0)

// Column-major C = A * B, with A: KxM (col-major), B: NxK (col-major).
// Same indexing convention as tests/opencl/sgemm/kernel.cl.
__global__ void sgemm(const TYPE* A, const TYPE* B, TYPE* C, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < N && col < N) {
    TYPE acc = 0;
    for (int k = 0; k < N; ++k) {
      acc += A[k * N + row] * B[col * N + k];
    }
    C[col * N + row] = acc;
  }
}

static void sgemm_cpu(TYPE* C, const TYPE* A, const TYPE* B, int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      TYPE acc = 0;
      for (int k = 0; k < K; ++k) {
        acc += A[k * M + m] * B[n * K + k];
      }
      C[n * M + m] = acc;
    }
  }
}

static bool fp_close(float a, float b) {
  union fi_t { float f; int32_t i; };
  fi_t fa, fb;
  fa.f = a;
  fb.f = b;
  return std::abs(fa.i - fb.i) <= FLOAT_ULP;
}

static uint32_t size = 64;

static void show_usage() {
  printf("Usage: [-n size] [-h: help]\n");
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:h")) != -1) {
    switch (c) {
    case 'n': size = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
  if (size < 2) {
    fprintf(stderr, "Error: invalid size!\n");
    exit(-1);
  }
  printf("Workload size=%ux%u\n", size, size);
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  uint32_t size_sq = size * size;
  size_t nbytes = size_sq * sizeof(TYPE);
  std::vector<TYPE> h_a(size_sq), h_b(size_sq), h_c(size_sq), h_ref(size_sq);

  srand(50);
  for (uint32_t i = 0; i < size_sq; ++i) {
    h_a[i] = static_cast<TYPE>(rand()) / RAND_MAX;
    h_b[i] = static_cast<TYPE>(rand()) / RAND_MAX;
  }

  printf("Allocate device buffers\n");
  TYPE *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  HIP_CHECK(hipMalloc((void**)&d_a, nbytes));
  HIP_CHECK(hipMalloc((void**)&d_b, nbytes));
  HIP_CHECK(hipMalloc((void**)&d_c, nbytes));

  printf("Upload source buffers\n");
  HIP_CHECK(hipMemcpy(d_a, h_a.data(), nbytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), nbytes, hipMemcpyHostToDevice));

  printf("Execute the kernel '%s'\n", KERNEL_NAME);
  // Clamp block_dim so block_dim*block_dim <= device maxThreadsPerBlock.
  int dev_id = 0;
  HIP_CHECK(hipGetDevice(&dev_id));
  hipDeviceProp_t dev_props{};
  HIP_CHECK(hipGetDeviceProperties(&dev_props, dev_id));
  uint32_t block_dim = 8;
  while (block_dim > 1 &&
         (int)(block_dim * block_dim) > dev_props.maxThreadsPerBlock)
    block_dim /= 2;
  const uint32_t grid_dim  = (size + block_dim - 1) / block_dim;
  dim3 block(block_dim, block_dim);
  dim3 grid(grid_dim, grid_dim);
  printf("block=%ux%u (device max=%d)\n",
         block_dim, block_dim, dev_props.maxThreadsPerBlock);

  auto t0 = std::chrono::high_resolution_clock::now();
  sgemm<<<grid, block, 0, 0>>>(d_a, d_b, d_c, (int)size);
  HIP_CHECK(hipDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  printf("Elapsed time: %lld ms\n",
         (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

  printf("Download destination buffer\n");
  HIP_CHECK(hipMemcpy(h_c.data(), d_c, nbytes, hipMemcpyDeviceToHost));

  printf("Verify result\n");
  sgemm_cpu(h_ref.data(), h_a.data(), h_b.data(), size, size, size);
  int errors = 0;
  for (uint32_t i = 0; i < size_sq; ++i) {
    if (!fp_close(h_c[i], h_ref[i])) {
      if (errors < 100) {
        printf("*** error: [%u] expected=%f, actual=%f\n",
               i, (float)h_ref[i], (float)h_c[i]);
      }
      ++errors;
    }
  }

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));

  if (errors == 0) {
    printf("PASSED!\n");
  } else {
    printf("FAILED! - %d errors\n", errors);
  }
  return errors;
}
