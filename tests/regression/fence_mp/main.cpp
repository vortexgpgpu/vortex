#include <iostream>
#include <unistd.h>
#include <string.h>
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
uint32_t count = 64;
uint32_t spin_limit = 4096;

vx_device_h device = nullptr;
vx_buffer_h data_buffer = nullptr;
vx_buffer_h flag_buffer = nullptr;
vx_buffer_h out_buffer = nullptr;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n words] [-s spin limit] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:s:k:h")) != -1) {
    switch (c) {
    case 'n': count = atoi(optarg); break;
    case 's': spin_limit = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0); break;
    default: show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (data_buffer) vx_buffer_release(data_buffer);
    if (flag_buffer) vx_buffer_release(flag_buffer);
    if (out_buffer)  vx_buffer_release(out_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_release(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  // one thread per hardware slot, so producer and consumer are resident together
  uint32_t total_threads = num_cores * num_warps * num_threads;

  kernel_arg.num_tasks  = total_threads;
  kernel_arg.count      = count;
  kernel_arg.spin_limit = spin_limit;

  std::vector<uint32_t> zeros(count, 0);
  std::vector<uint32_t> out(4, 0);
  uint32_t data_size = count * sizeof(uint32_t);
  uint32_t out_size  = out.size() * sizeof(uint32_t);
  RT_CHECK(vx_buffer_create(device, data_size, VX_MEM_READ_WRITE, &data_buffer));
  RT_CHECK(vx_buffer_address(data_buffer, &kernel_arg.data_addr));
  RT_CHECK(vx_buffer_create(device, sizeof(uint32_t), VX_MEM_READ_WRITE, &flag_buffer));
  RT_CHECK(vx_buffer_address(flag_buffer, &kernel_arg.flag_addr));
  RT_CHECK(vx_buffer_create(device, out_size, VX_MEM_READ_WRITE, &out_buffer));
  RT_CHECK(vx_buffer_address(out_buffer, &kernel_arg.out_addr));
  RT_CHECK(vx_enqueue_write(queue, data_buffer, 0, zeros.data(), data_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, flag_buffer, 0, zeros.data(), sizeof(uint32_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, out_buffer, 0, out.data(), out_size, 0, nullptr, nullptr));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    uint32_t n = total_threads;
    uint32_t grid_dim[1], block_dim[1];
    RT_CHECK(vx_device_max_occupancy_grid(device, 1, &n, grid_dim, block_dim));
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
  RT_CHECK(vx_enqueue_read(queue, out.data(), out_buffer, 0, out_size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  if (out[1] != 0) {
    std::cout << "consumer timed out waiting for the flag" << std::endl;
    ++errors;
  }
  if (out[0] != 0) {
    std::cout << "consumer saw " << std::dec << out[0] << " stale data words" << std::endl;
    errors += out[0];
  }

  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }
  std::cout << "PASSED!" << std::endl;
  return 0;
}
