#include "common.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <vortex2.h>

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
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
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
    if (dst_buffer) vx_buffer_release(dst_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_release(device);
  }
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint32_t buf_size = size * size * sizeof(float);

  std::cout << "number of points: " << size << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  kernel_arg.size = size;

  // allocate buffers
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::endl;

  // load kernel module
  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  // start device
  std::cout << "start device" << std::endl;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    uint64_t num_threads;
    RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
    uint32_t NT = (uint32_t)num_threads;
    uint32_t grid_dim[2]  = {(size + NT - 1) / NT, size};
    uint32_t block_dim[2] = {NT, 1};
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 2;
    li.grid_dim[0]  = grid_dim[0];
    li.grid_dim[1]  = grid_dim[1];
    li.block_dim[0] = block_dim[0];
    li.block_dim[1] = block_dim[1];
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  // download destination buffer
  std::vector<float> h_C(size * size);
  RT_CHECK(vx_enqueue_read(queue, h_C.data(), dst_buffer, 0, buf_size, 1, &launch_ev, &read_ev));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

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