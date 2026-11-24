// Register windows: kernel results computed through registers above x31/f31.
#include <iostream>
#include <unistd.h>
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
uint32_t count = 2;

vx_device_h device = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
kernel_arg_t kernel_arg = {};

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
    case 'n': count = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    default:
      std::cout << "Usage: [-k: kernel] [-n: points per hardware thread] [-h: help]" << std::endl;
      exit(c == 'h' ? 0 : -1);
    }
  }
}

void cleanup() {
  if (device) {
    if (src_buffer) vx_buffer_release(src_buffer);
    if (dst_buffer) vx_buffer_release(dst_buffer);
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

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  uint32_t block_dim = num_threads;
  uint32_t num_points = count * num_cores * num_warps * num_threads;
  uint32_t num_blocks = (num_points + block_dim - 1) / block_dim;
  std::cout << "number of points: " << num_points << std::endl;

  std::vector<uint32_t> h_src(num_points);
  std::vector<uint32_t> h_dst(num_points * CHK_COUNT, 0);
  for (auto& v : h_src) {
    v = std::rand();
  }

  kernel_arg.num_points = num_points;
  RT_CHECK(vx_buffer_create(device, h_src.size() * 4, VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_buffer_address(src_buffer, &kernel_arg.src_addr));
  RT_CHECK(vx_buffer_create(device, h_dst.size() * 4, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));
  RT_CHECK(vx_enqueue_write(queue, src_buffer, 0, h_src.data(), h_src.size() * 4, 0, nullptr, nullptr));

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &kernel_arg;
  li.args_size    = sizeof(kernel_arg);
  li.ndim         = 1;
  li.grid_dim[0]  = num_blocks;
  li.block_dim[0] = block_dim;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, h_dst.size() * 4, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  static const char* names[CHK_COUNT] = {"fields", "sticky", "field0", "fp", "isolated"};
  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    uint32_t a = h_src[i], b = a * 3 + 1;
    uint32_t ref[CHK_COUNT] = {a + b, (a + 1) + b * b, (a + 2) + (a + 1), a, a ^ 0x55};
    for (uint32_t k = 0; k < CHK_COUNT; ++k) {
      uint32_t cur = h_dst[i * CHK_COUNT + k];
      if (cur != ref[k]) {
        if (errors < 20) {
          printf("*** error: [%u] %s expected=0x%x, actual=0x%x\n", i, names[k], ref[k], cur);
        }
        ++errors;
      }
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
