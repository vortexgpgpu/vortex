// Dispatch-shape coverage down to the degenerate grid: one CTA of one thread.
// A grid of a single CTA gives the launch and completion path the least slack —
// there is no following CTA to keep the device busy across the handoff — so it
// exercises cases a multi-CTA launch cannot reach.

#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex2.h>
#include <vector>
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
uint32_t grid_size = 1;
uint32_t block_size = 1;
// Completion wait: 0 waits on the launch event, 1 drains the queue. Both are
// public API and a correct device must retire the CTA under either, so the
// option keeps the two observation points distinguishable.
int use_queue_finish = 0;
// Exercise a thread-divergent branch that no thread takes (see kernel.cpp).
int use_diverge = 0;

vx_device_h device = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Single-CTA Test." << std::endl;
   std::cout << "Usage: [-g grid][-b block][-q: queue finish][-d: divergence][-k: kernel][-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "g:b:qdk:h")) != -1) {
    switch (c) {
    case 'g':
      grid_size = atoi(optarg);
      break;
    case 'b':
      block_size = atoi(optarg);
      break;
    case 'q':
      use_queue_finish = 1;
      break;
    case 'd':
      use_diverge = 1;
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
  if (0 == grid_size || 0 == block_size) {
    std::cout << "Error: grid and block must be non-zero." << std::endl;
    exit(-1);
  }
}

void cleanup() {
  if (device) {
    if (dst_buffer) vx_buffer_release(dst_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    // Retired-instruction count distinguishes a kernel that ran to completion
    // from one that stopped partway, which the output buffer alone cannot show.
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  // Past the data: the CTA dimension CSRs read back from the device, then a
  // run of markers (see kernel.cpp).
  static const uint32_t NUM_TAIL = NUM_DIMS + NUM_MARKERS;
  uint32_t num_points = grid_size * block_size;
  uint32_t buf_size = (num_points + NUM_TAIL) * sizeof(uint32_t);

  std::cout << "grid=" << grid_size << " block=" << block_size
            << " points=" << num_points << std::endl;

  RT_CHECK(vx_device_open(0, &device));
  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));
  kernel_arg.count = num_points;
  kernel_arg.diverge = (uint32_t)use_diverge;

  // Poison the destination so an unwritten slot cannot pass by coincidence:
  // the expected stamp starts at 1, so a leftover zero is always an error.
  std::vector<uint32_t> h_dst(num_points + NUM_TAIL, 0);
  RT_CHECK(vx_enqueue_write(queue, dst_buffer, 0, h_dst.data(), buf_size, 0, nullptr, nullptr));

  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::cout << "start execution (wait=" << (use_queue_finish ? "queue_finish" : "event")
            << ")" << std::endl;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = grid_size;
    li.block_dim[0] = block_size;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr,
                               use_queue_finish ? nullptr : &launch_ev));
  }
  if (use_queue_finish) {
    RT_CHECK(vx_queue_finish(queue, VX_TIMEOUT_INFINITE));
  } else {
    RT_CHECK(vx_event_wait_value(launch_ev, 1, VX_TIMEOUT_INFINITE));
  }

  std::cout << "read destination buffer from device memory" << std::endl;
  RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, buf_size,
                           launch_ev ? 1 : 0, launch_ev ? &launch_ev : nullptr, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  if (launch_ev) {
    vx_event_release(launch_ev);
  }

  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    uint32_t ref = i + 1;
    if (h_dst[i] != ref) {
      printf("*** error: [%d] expected=%d, actual=%d\n", i, ref, h_dst[i]);
      ++errors;
    }
  }

  const uint32_t *dims = &h_dst[num_points];
  printf("device dims: gridDim=[%u,%u] blockDim=[%u,%u]\n",
         dims[0], dims[2], dims[1], dims[3]);
  const uint32_t dim_ref[NUM_DIMS] = { grid_size, block_size, 1, 1 };
  const char *dim_name[NUM_DIMS] = { "gridDim.x", "blockDim.x",
                                     "gridDim.y", "blockDim.y" };
  for (uint32_t i = 0; i < NUM_DIMS; ++i) {
    if (dims[i] != dim_ref[i]) {
      printf("*** error: %s expected=%d, actual=%d\n",
             dim_name[i], dim_ref[i], dims[i]);
      ++errors;
    }
  }

  const uint32_t *marks = &h_dst[num_points + NUM_DIMS];
  printf("markers:");
  for (uint32_t i = 0; i < NUM_MARKERS; ++i) {
    printf(" %s", marks[i] == MARKER_BASE + i ? "." : "X");
  }
  printf("   (. = written, X = missing)\n");
  for (uint32_t i = 0; i < NUM_MARKERS; ++i) {
    if (marks[i] != MARKER_BASE + i) {
      ++errors;
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
