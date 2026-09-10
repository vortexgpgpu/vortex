#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex2.h>
#include <vector>
#include <algorithm>
#include "common.h"

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret)                                              \
      break;                                                    \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);    \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

const char* kernel_file = "kernel.vxbin";
uint32_t count = 0;
uint32_t depth = 16; // deep enough to overflow the old always-push sbr model

vx_device_h device = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex sbr_nest test.\nUsage: [-k kernel] [-n words] [-d depth] [-h]\n";
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:d:h")) != -1) {
    switch (c) {
      case 'n': count = atoi(optarg); break;
      case 'k': kernel_file = optarg; break;
      case 'd': depth = std::max(1, std::min(16, atoi(optarg))); break;
      case 'h': show_usage(); exit(0);
      default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (dst_buffer) vx_buffer_release(dst_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

// Host mirror of the kernel's truly-nested chain. Because the guards are
// monotonic (tid>=k+1 implies every outer tid>=j+1, depth>k implies depth>j),
// "all outer conditions hold" reduces to the innermost pair, so the nested form
// equals this flat accumulation.
static int deep_nest_host(uint32_t tid, uint32_t d) {
  int v = (int)tid;
  for (uint32_t k = 0; k < 16; ++k)
    if (d > k && tid >= (k + 1u))
      v += (int)(k + 1);
  return v;
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);
  if (count == 0) count = 1;

  std::cout << "open device connection\n";
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  uint32_t total_threads = num_cores * num_warps * num_threads;
  uint32_t num_points = count * total_threads;
  uint32_t buf_size = num_points * sizeof(int32_t);

  std::cout << "number of points: " << num_points << ", nesting depth: " << depth << "\n";

  kernel_arg.num_points = num_points;
  kernel_arg.depth      = depth;

  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));

  std::vector<int32_t> h_dst(num_points);

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    uint32_t grid_dim[1], block_dim[1];
    RT_CHECK(vx_device_max_occupancy_grid(device, 1, &num_points, grid_dim, block_dim));
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

  RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, buf_size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    int ref = deep_nest_host(i, depth);
    if (h_dst[i] != ref) {
      if (errors < 16)
        std::cout << "error at #" << std::dec << i << ": actual " << h_dst[i]
                  << ", expected " << ref << "\n";
      ++errors;
    }
  }

  cleanup();
  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!\nFAILED!\n";
    return errors;
  }
  std::cout << "PASSED!\n";
  return 0;
}
