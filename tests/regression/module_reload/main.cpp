// Module residency: a released module's device address must become reusable,
// and a still-resident one must refuse to be overwritten.
//
// A device image is linked at a fixed VMA, so the address it occupies can host
// one image at a time. Callers that keep an image resident across launches --
// the graphics driver keeps one per pipeline stage -- rely on two guarantees
// from the loader: releasing a module returns its range, and loading over a
// range that is still held is rejected rather than silently aliased. A test
// that loads one image per session exercises neither, because a leaked
// reservation and a silent alias both look exactly like a pass there.
//
// Every round launches the kernel with its own nonce and checks the result, so
// a round proves the image at that address executes, not merely that the load
// call returned success.

#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex2.h>
#include <vector>
#include "common.h"

#define NONCE 0xdeadbeef

// Bounded waits so a wedged device fails the run instead of hanging it.
#define WAIT_TIMEOUT_NS (60ull * 1000 * 1000 * 1000)

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
uint32_t count = 0;

vx_device_h device = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;

static uint64_t num_cores = 0;
static uint64_t num_threads = 0;
static uint32_t num_points = 0;
static uint32_t buf_size = 0;
static kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex module-reload test." << std::endl;
  std::cout << "Usage: [-k: kernel][-n words][-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h")) != -1) {
    switch (c) {
    case 'n':
      count = atoi(optarg);
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
    if (src_buffer) {
      vx_buffer_release(src_buffer);
    }
    if (dst_buffer) {
      vx_buffer_release(dst_buffer);
    }
    if (kernel) {
      vx_kernel_release(kernel);
    }
    if (module_) {
      vx_module_release(module_);
    }
    if (queue) {
      vx_queue_release(queue);
    }
    vx_device_release(device);
  }
  device = nullptr;
  src_buffer = nullptr;
  dst_buffer = nullptr;
  kernel = nullptr;
  module_ = nullptr;
  queue = nullptr;
}

inline uint32_t shuffle(int i, uint32_t value) {
  return (value << i) | (value & ((1 << i) - 1));
}

// Load the image and run it once over a nonce-derived pattern. The nonce makes
// a launch that never executed detectable: dst still holds the previous
// round's pattern rather than this one's.
static int load_and_run(const char* round, uint32_t nonce) {
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::vector<uint32_t> h_src(num_points);
  std::vector<uint32_t> h_dst(num_points);
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src[i] = shuffle(i, nonce);
    h_dst[i] = 0;
  }

  RT_CHECK(vx_enqueue_write(queue, src_buffer, 0, h_src.data(), buf_size, 0, nullptr, nullptr));

  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = (uint32_t)num_cores;
    li.block_dim[0] = (uint32_t)num_threads;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }
  RT_CHECK(vx_event_wait_value(launch_ev, 1, WAIT_TIMEOUT_NS));

  RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, buf_size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, WAIT_TIMEOUT_NS));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    auto cur = h_dst[i];
    auto ref = shuffle(i, nonce);
    if (cur != ref) {
      if (errors < 8) {
        printf("*** %s: [%d] expected=0x%x, actual=0x%x\n", round, i, ref, cur);
      }
      ++errors;
    }
  }
  std::cout << round << ": loaded, launched, verified" << std::endl;
  return errors;
}

static void release_module() {
  if (kernel) {
    vx_kernel_release(kernel);
    kernel = nullptr;
  }
  if (module_) {
    vx_module_release(module_);
    module_ = nullptr;
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  if (count == 0) {
    count = 1;
  }

  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  num_points = count * num_cores;
  buf_size = num_points * sizeof(int32_t);

  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_buffer_address(src_buffer, &kernel_arg.src_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));
  kernel_arg.count = count;

  int errors = 0;

  // Round 1 establishes the baseline: the address is free and the image runs.
  errors += load_and_run("round 1 (first load)", NONCE);
  release_module();

  // Round 2 is the reservation-leak check. A loader that kept the range on
  // release fails the load here; one that returned it runs the fresh image.
  errors += load_and_run("round 2 (reload after release)", NONCE ^ 0x9e3779b9u);

  // Round 3 is the known-bad input, and it runs while round 2's module is
  // still resident. The load must be refused; a success means two images claim
  // one address, and the handle it produced is released before failing so the
  // remaining rounds still run against a known allocator state.
  {
    vx_module_h dup = nullptr;
    std::cout << "round 3 (overlapping load): expecting refusal" << std::endl;
    int r = vx_module_load_file(device, kernel_file, &dup);
    if (r == 0) {
      printf("*** round 3: overlapping load succeeded; two images share one address\n");
      vx_module_release(dup);
      ++errors;
    } else {
      std::cout << "round 3: refused, as required" << std::endl;
    }
  }

  // Round 4 confirms the refusal above consumed nothing: once round 2's module
  // is released the address is usable again. A loader that leaked on the
  // rejected path passes round 2 and fails only here.
  release_module();
  errors += load_and_run("round 4 (reload after refusal)", NONCE ^ 0x7f4a7c15u);

  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "Test PASSED" << std::endl;
  return 0;
}
