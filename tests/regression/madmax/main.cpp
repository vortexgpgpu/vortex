#include "common.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <vortex.h>

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                      \
  do {                                                       \
    int _ret = _expr;                                        \
    if (0 == _ret)                                           \
      break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup();                                               \
    exit(-1);                                                \
  } while (false)

const char *kernel_file = "kernel.vxbin";
uint32_t size = 32;

vx_device_h device = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

// Synthetic computation replica for verification
void compute_reference(float *ref, uint32_t size) {
  for (uint32_t row = 0; row < size; ++row) {
    for (uint32_t col = 0; col < size; ++col) {
      ref[row * size + col] = madmax_compute(row, col, size);
    }
  }
}

static void show_usage() {
  std::cout << "Vortex Madmax Test." << std::endl;
  std::cout << "Usage: [-k: kernel] [-n size] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
    case 'n':
      size = atoi(optarg);
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
    vx_mem_free(dst_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));
  uint32_t buf_size = size * size * sizeof(float);

  std::cout << "number of points: " << size << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  kernel_arg.grid_dim[0] = size;
  kernel_arg.grid_dim[1] = size;
  kernel_arg.size = size;

  // allocate buffers
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_mem_address(dst_buffer, &kernel_arg.dst_addr));
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::endl;

  // Upload kernel binary
  std::cout << "upload kernel binary" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // download destination buffer
  std::vector<float> h_C(size * size);
  RT_CHECK(vx_copy_from_dev(h_C.data(), dst_buffer, 0, buf_size));

  // Verify results
  std::cout << "verify result" << std::endl;
  int errors = 0;
  std::vector<float> h_ref(size * size);
  compute_reference(h_ref.data(), size);

  for (uint32_t i = 0; i < h_ref.size(); ++i) {
    union fi_t {
      float f;
      int32_t i;
    };
    fi_t actual, expected;
    actual.f = h_C[i];
    expected.f = h_ref[i];

    if (std::abs(actual.i - expected.i) > FLOAT_ULP) {
      if (errors < 3) {
        printf("*** error: [%d] expected=%f, actual=%f\n", i, expected.f, actual.f);
      }
      ++errors;
    }
  }

  cleanup();

  return (errors != 0);
}