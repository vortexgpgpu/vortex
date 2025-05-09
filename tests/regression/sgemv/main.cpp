#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <vortex.h>
#include <cmath>
#include "common.h"

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

const char* kernel_file = "kernel.vxbin";
uint32_t M = 1024;  // Rows (output vector size)
uint32_t N = 1024;  // Columns (input vector size)

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;  // Matrix (M x N)
vx_buffer_h x_buffer = nullptr;  // Vector (N x 1)
vx_buffer_h y_buffer = nullptr;  // Output (M x 1)
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex SGEMV (Matrix-Vector Multiplication)." << std::endl;
   std::cout << "Usage: [-k: kernel] [-m rows] [-n cols] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "m:n:k:h")) != -1) {
    switch (c) {
    case 'm':
      M = atoi(optarg);
      break;
    case 'n':
      N = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'h':
      show_usage();
      exit(0);
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(A_buffer);
    vx_mem_free(x_buffer);
    vx_mem_free(y_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

// CPU reference implementation
void sgemv_cpu(float* y, const float* A, const float* x, uint32_t M, uint32_t N) {
  for (uint32_t i = 0; i < M; ++i) {
    float sum = 0.0f;
    for (uint32_t j = 0; j < N; ++j) {
      sum += A[i * N + j] * x[j];
    }
    y[i] = sum;
  }
}

int main(int argc, char *argv[]) {
  // Parse command arguments
  parse_args(argc, argv);

  std::cout << "Matrix dimensions: " << M << " x " << N << std::endl;

  // Open device connection
  std::cout << "Opening device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint32_t A_size = M * N * sizeof(float);
  uint32_t x_size = N * sizeof(float);
  uint32_t y_size = M * sizeof(float);

  // Initialize kernel arguments
  kernel_arg.grid_dim[0] = M;  // 1 thread per output row
  kernel_arg.grid_dim[1] = 1;
  kernel_arg.M = M;
  kernel_arg.N = N;

  // Allocate device memory
  std::cout << "Allocating device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, A_size, VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_mem_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_mem_alloc(device, x_size, VX_MEM_READ, &x_buffer));
  RT_CHECK(vx_mem_address(x_buffer, &kernel_arg.x_addr));
  RT_CHECK(vx_mem_alloc(device, y_size, VX_MEM_WRITE, &y_buffer));
  RT_CHECK(vx_mem_address(y_buffer, &kernel_arg.y_addr));

  // Generate synthetic data
  std::vector<float> h_A(M * N);
  std::vector<float> h_x(N);
  std::vector<float> h_y(M, 0.0f);
  for (uint32_t i = 0; i < M * N; ++i) {
    h_A[i] = static_cast<float>(rand()) / RAND_MAX;  // Random matrix (0-1)
  }
  for (uint32_t i = 0; i < N; ++i) {
    h_x[i] = static_cast<float>(rand()) / RAND_MAX;  // Random vector (0-1)
  }

  // Upload input buffers
  std::cout << "Uploading matrix A" << std::endl;
  RT_CHECK(vx_copy_to_dev(A_buffer, h_A.data(), 0, A_size));
  std::cout << "Uploading vector x" << std::endl;
  RT_CHECK(vx_copy_to_dev(x_buffer, h_x.data(), 0, x_size));

  // Upload program
  std::cout << "Uploading kernel" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // Upload kernel arguments
  std::cout << "Uploading kernel arguments" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  // Execute kernel
  auto time_start = std::chrono::high_resolution_clock::now();
  std::cout << "Launching kernel" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

  // Download results
  std::cout << "Downloading results" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_y.data(), y_buffer, 0, y_size));

  // Verify results
  std::cout << "Verifying results" << std::endl;
  std::vector<float> h_ref(M, 0.0f);
  sgemv_cpu(h_ref.data(), h_A.data(), h_x.data(), M, N);

  int errors = 0;
  for (uint32_t i = 0; i < M; ++i) {
    if (fabs(h_y[i] - h_ref[i]) > 1e-3f) {
      if (errors < 10) {
        printf("*** error: [%d] expected=%f, actual=%f\n", i, h_ref[i], h_y[i]);
      }
      ++errors;
    }
  }

  // Cleanup
  std::cout << "Cleaning up" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}