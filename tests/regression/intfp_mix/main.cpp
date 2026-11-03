#include "common.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <vortex2.h>

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
vx_buffer_h dst_f_buffer = nullptr;
vx_buffer_h dst_i_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

// The kernel's computation on the host, for verification
void compute_reference(float *ref_f, uint32_t *ref_i, uint32_t size) {
  for (uint32_t row = 0; row < size; ++row) {
    for (uint32_t col = 0; col < size; ++col) {
      intfp_mix_compute(row, col, size, &ref_f[row * size + col], &ref_i[row * size + col]);
    }
  }
}

static void show_usage() {
  std::cout << "Vortex mixed INT/FP32 throughput test." << std::endl;
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
    if (dst_f_buffer) vx_buffer_release(dst_f_buffer);
    if (dst_i_buffer) vx_buffer_release(dst_i_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
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

  uint32_t buf_size = size * size * sizeof(float); // = sizeof(uint32_t)

  std::cout << "number of points: " << size << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  kernel_arg.size = size;

  // allocate buffers
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &dst_f_buffer));
  RT_CHECK(vx_buffer_address(dst_f_buffer, &kernel_arg.dst_f_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &dst_i_buffer));
  RT_CHECK(vx_buffer_address(dst_i_buffer, &kernel_arg.dst_i_addr));
  std::cout << "dev_dst_f=0x" << std::hex << kernel_arg.dst_f_addr << std::endl;
  std::cout << "dev_dst_i=0x" << std::hex << kernel_arg.dst_i_addr << std::dec << std::endl;

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

  // download destination buffers
  std::vector<float> h_f(size * size);
  std::vector<uint32_t> h_i(size * size);
  vx_event_h read_f_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_f.data(), dst_f_buffer, 0, buf_size, 1, &launch_ev, &read_f_ev));
  RT_CHECK(vx_enqueue_read(queue, h_i.data(), dst_i_buffer, 0, buf_size, 1, &read_f_ev, &read_ev));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(read_f_ev);
  vx_event_release(launch_ev);

  // Verify results: bit-exact (fused multiply-adds on both sides)
  std::cout << "verify result" << std::endl;
  int errors = 0;
  std::vector<float> ref_f(size * size);
  std::vector<uint32_t> ref_i(size * size);
  compute_reference(ref_f.data(), ref_i.data(), size);
  for (uint32_t i = 0; i < ref_f.size(); ++i) {
    uint32_t af, ef;
    memcpy(&af, &h_f[i], 4);
    memcpy(&ef, &ref_f[i], 4);
    if (af != ef || h_i[i] != ref_i[i]) {
      if (errors < 3) {
        printf("*** error: [%d] expected=%f/0x%08x, actual=%f/0x%08x\n", i, ref_f[i], ref_i[i], h_f[i], h_i[i]);
      }
      ++errors;
    }
  }

  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!\nFAILED!" << std::endl;
    return errors;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}