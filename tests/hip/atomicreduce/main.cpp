#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

#include "common.h"

#define KERNEL_NAME "atomicreduce"

#define HIP_CHECK(_expr)                                                  \
  do {                                                                    \
    hipError_t _err = (_expr);                                            \
    if (_err != hipSuccess) {                                             \
      fprintf(stderr, "HIP Error: '%s' returned %d (%s)\n",               \
              #_expr, (int)_err, hipGetErrorString(_err));                \
      exit(-1);                                                           \
    }                                                                     \
  } while (0)

// Every thread accumulates its element into a single global counter.
// Maximum-contention atomicAdd, lowering to a hardware RVA amoadd.w on
// Vortex (requires the A extension).
__global__ void atomicreduce(const int* data, int* result, int N) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N) {
    atomicAdd(&result[0], data[gid]);
  }
}

static uint32_t size = 1024;

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

  std::vector<int> h_data(size);
  int h_ref = 0;

  srand(50);
  for (uint32_t i = 0; i < size; ++i) {
    h_data[i] = rand() % 1000;
    h_ref += h_data[i];
  }
  int h_result = 0;

  printf("Allocate device buffers\n");
  int *d_data = nullptr, *d_result = nullptr;
  HIP_CHECK(hipMalloc((void**)&d_data, size * sizeof(int)));
  HIP_CHECK(hipMalloc((void**)&d_result, sizeof(int)));

  printf("Upload source buffers\n");
  HIP_CHECK(hipMemcpy(d_data, h_data.data(), size * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_result, &h_result, sizeof(int), hipMemcpyHostToDevice));

  printf("Execute the kernel '%s'\n", KERNEL_NAME);
  int dev_id = 0;
  HIP_CHECK(hipGetDevice(&dev_id));
  hipDeviceProp_t dev_props{};
  HIP_CHECK(hipGetDeviceProperties(&dev_props, dev_id));
  uint32_t block_size = 64;
  if ((int)block_size > dev_props.maxThreadsPerBlock)
    block_size = (uint32_t)dev_props.maxThreadsPerBlock;
  const uint32_t grid_size = (size + block_size - 1) / block_size;
  printf("block_size=%u (device max=%d)\n", block_size, dev_props.maxThreadsPerBlock);

  auto t0 = std::chrono::high_resolution_clock::now();
  atomicreduce<<<dim3(grid_size), dim3(block_size), 0, 0>>>(d_data, d_result, (int)size);
  HIP_CHECK(hipDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  printf("Elapsed time: %lld ms\n",
         (long long)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count());

  printf("Download destination buffer\n");
  HIP_CHECK(hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost));

  printf("Verify result\n");
  int errors = 0;
  if (h_result != h_ref) {
    printf("*** error: expected=%d, actual=%d\n", h_ref, h_result);
    ++errors;
  }

  HIP_CHECK(hipFree(d_data));
  HIP_CHECK(hipFree(d_result));

  if (errors == 0) {
    printf("PASSED!\n");
  } else {
    printf("FAILED! - %d errors\n", errors);
  }
  return errors;
}
