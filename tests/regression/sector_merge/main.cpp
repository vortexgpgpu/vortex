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

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.vxbin";
uint32_t num_lines  = 256;
uint32_t line_bytes = 128;

vx_device_h device     = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_queue_h  queue      = nullptr;
vx_module_h module_    = nullptr;
vx_kernel_h kernel     = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n lines] [-l line bytes] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:l:k:h")) != -1) {
    switch (c) {
    case 'n':
      num_lines = atoi(optarg);
      break;
    case 'l':
      line_bytes = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'h':
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
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

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  uint32_t line_words = line_bytes / sizeof(uint32_t);
  uint32_t num_tasks  = num_cores * num_warps * num_threads;
  uint32_t num_words  = num_lines * line_words;

  kernel_arg.num_lines  = num_lines;
  kernel_arg.line_words = line_words;
  kernel_arg.group      = num_threads;

  std::cout << "number of tasks: " << num_tasks << ", lines: " << num_lines
            << ", line bytes: " << line_bytes << std::endl;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, num_words * sizeof(uint32_t), VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_buffer_address(src_buffer, &kernel_arg.src_addr));
  RT_CHECK(vx_buffer_create(device, num_tasks * sizeof(uint32_t), VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));

  // upload source buffer
  std::cout << "upload source buffer" << std::endl;
  std::vector<uint32_t> h_src(num_words);
  for (uint32_t i = 0; i < num_words; ++i) {
    h_src[i] = i * 2654435761u + 12345u;
  }
  RT_CHECK(vx_enqueue_write(queue, src_buffer, 0, h_src.data(), num_words * sizeof(uint32_t), 0, nullptr, nullptr));

  // load kernel module
  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  // start device
  std::cout << "start device" << std::endl;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    uint32_t n = num_tasks;
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

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  std::vector<uint32_t> h_dst(num_tasks);
  RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, num_tasks * sizeof(uint32_t), 1, &launch_ev, &read_ev));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  // verify result
  std::cout << "verify result" << std::endl;
  uint32_t half_words = line_words / 2;
  int errors = 0;
  for (uint32_t gid = 0; gid < num_tasks; ++gid) {
    uint32_t half = (gid / num_threads) & 1;
    uint32_t word = half * half_words + (gid % num_threads) % half_words;
    uint32_t ref = 0;
    for (uint32_t k = 0; k < num_lines; ++k) {
      ref += h_src[k * line_words + word] * (k + 1);
    }
    if (h_dst[gid] != ref) {
      if (errors < 100) {
        std::cout << "error at result #" << gid << ": actual 0x" << std::hex << h_dst[gid]
                  << ", expected 0x" << ref << std::dec << std::endl;
      }
      ++errors;
    }
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}
