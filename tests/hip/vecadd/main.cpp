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

#define KERNEL_NAME "vecadd"
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

__global__ void vecadd(const TYPE* A, const TYPE* B, TYPE* C, int N) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N) {
    C[gid] = A[gid] + B[gid];
  }
}

static void vecadd_cpu(TYPE* C, const TYPE* A, const TYPE* B, int N) {
  for (int i = 0; i < N; ++i) {
    C[i] = A[i] + B[i];
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
  printf("Workload size=%u\n", size);
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  size_t nbytes = size * sizeof(TYPE);
  std::vector<TYPE> h_a(size), h_b(size), h_c(size), h_ref(size);

  srand(50);
  for (uint32_t i = 0; i < size; ++i) {
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
  // Query the device-supported max work-group size and clamp block_size
  // to it. The Vortex build_test{32,64} default config exposes
  // num_warps*num_threads = 32 (4 warps x 8 threads), so a hard-coded
  // 64 trips hipErrorLaunchFailure ("Requested local size exceeds HW
  // max"). Other vendors typically advertise >= 256, so this is a
  // no-op there.
  int dev_id = 0;
  HIP_CHECK(hipGetDevice(&dev_id));
  hipDeviceProp_t dev_props{};
  HIP_CHECK(hipGetDeviceProperties(&dev_props, dev_id));
  uint32_t block_size = 64;
  if ((int)block_size > dev_props.maxThreadsPerBlock)
    block_size = (uint32_t)dev_props.maxThreadsPerBlock;
  const uint32_t grid_size  = (size + block_size - 1) / block_size;
  printf("block_size=%u (device max=%d)\n",
         block_size, dev_props.maxThreadsPerBlock);

  auto t0 = std::chrono::high_resolution_clock::now();
  vecadd<<<dim3(grid_size), dim3(block_size), 0, 0>>>(d_a, d_b, d_c, (int)size);
  HIP_CHECK(hipDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  printf("Elapsed time: %lld ms\n",
         (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

  printf("Download destination buffer\n");
  HIP_CHECK(hipMemcpy(h_c.data(), d_c, nbytes, hipMemcpyDeviceToHost));

  printf("Verify result\n");
  vecadd_cpu(h_ref.data(), h_a.data(), h_b.data(), size);
  int errors = 0;
  for (uint32_t i = 0; i < size; ++i) {
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
