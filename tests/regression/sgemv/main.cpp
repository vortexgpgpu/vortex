#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <vortex2.h>
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
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
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
    if (A_buffer) vx_buffer_release(A_buffer);
    if (x_buffer) vx_buffer_release(x_buffer);
    if (y_buffer) vx_buffer_release(y_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_release(device);
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
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint32_t A_size = M * N * sizeof(float);
  uint32_t x_size = N * sizeof(float);
  uint32_t y_size = M * sizeof(float);

  // Initialize kernel arguments
  kernel_arg.M = M;
  kernel_arg.N = N;

  // Allocate device memory
  std::cout << "Allocating device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, A_size, VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_buffer_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_buffer_create(device, x_size, VX_MEM_READ, &x_buffer));
  RT_CHECK(vx_buffer_address(x_buffer, &kernel_arg.x_addr));
  RT_CHECK(vx_buffer_create(device, y_size, VX_MEM_WRITE, &y_buffer));
  RT_CHECK(vx_buffer_address(y_buffer, &kernel_arg.y_addr));

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
  RT_CHECK(vx_enqueue_write(queue, A_buffer, 0, h_A.data(), A_size, 0, nullptr, nullptr));
  std::cout << "Uploading vector x" << std::endl;
  RT_CHECK(vx_enqueue_write(queue, x_buffer, 0, h_x.data(), x_size, 0, nullptr, nullptr));

  // Load kernel module
  std::cout << "Loading kernel" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  // Execute kernel
  auto time_start = std::chrono::high_resolution_clock::now();
  std::cout << "Launching kernel" << std::endl;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    uint32_t grid_dim[1], block_dim[1];
    RT_CHECK(vx_device_max_occupancy_grid(device, 1, &M, grid_dim, block_dim));
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = grid_dim[0];
    li.block_dim[0] = block_dim[0];
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  // Download results
  std::cout << "Downloading results" << std::endl;
  RT_CHECK(vx_enqueue_read(queue, h_y.data(), y_buffer, 0, y_size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);
  auto time_end = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Elapsed time: %lg ms\n", elapsed);

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
