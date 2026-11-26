#include <iostream>
#include <unistd.h>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <vortex2.h>
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
uint32_t num_rows = 1024;

vx_device_h device = nullptr;
vx_buffer_h keys_buf = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-n rows] [-h]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "k:n:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'n': num_rows = atoi(optarg); break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (keys_buf) vx_buffer_release(keys_buf);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  std::srand(50);

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));
  std::cout << "rows=" << num_rows << " keys=" << KEYS << std::endl;

  uint32_t size = KEYS * num_rows * sizeof(uint32_t);
  kernel_arg.num_rows = num_rows;
  RT_CHECK(vx_buffer_create(device, size, VX_MEM_READ_WRITE, &keys_buf));
  RT_CHECK(vx_buffer_address(keys_buf, &kernel_arg.keys_addr));
  std::vector<uint32_t> h_keys(KEYS * num_rows);
  for (auto& k : h_keys) k = (uint32_t(rand()) << 16) ^ uint32_t(rand());
  RT_CHECK(vx_enqueue_write(queue, keys_buf, 0, h_keys.data(), size, 0, nullptr, nullptr));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    uint32_t grid_dim[1], block_dim[1];
    RT_CHECK(vx_device_max_occupancy_grid(device, 1, &num_rows, grid_dim, block_dim));
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
  std::vector<uint32_t> d_keys(KEYS * num_rows);
  RT_CHECK(vx_enqueue_read(queue, d_keys.data(), keys_buf, 0, size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  for (uint32_t r = 0; r < num_rows; ++r) {
    std::vector<uint32_t> ref(KEYS);
    for (int k = 0; k < KEYS; ++k)
      ref[k] = h_keys[k * num_rows + r];
    std::sort(ref.begin(), ref.end());
    for (int k = 0; k < KEYS; ++k) {
      uint32_t d = d_keys[k * num_rows + r];
      if (d != ref[k] && errors++ < 10)
        printf("*** error: row %u key %d expected=0x%08x, actual=0x%08x\n", r, k, ref[k], d);
    }
  }

  cleanup();
  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
