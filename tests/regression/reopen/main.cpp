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
uint32_t iterations = 2;

vx_device_h device = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;

static void show_usage() {
  std::cout << "Vortex device-reopen test." << std::endl;
  std::cout << "Usage: [-i iterations][-k: kernel][-n words][-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:i:k:h")) != -1) {
    switch (c) {
    case 'n':
      count = atoi(optarg);
      break;
    case 'i':
      iterations = atoi(optarg);
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

// One full device session: open, run the copy kernel, verify, close.
// The per-iteration nonce makes results from a previous session detectable:
// a launch that silently never executes leaves the old pattern in dst.
int run_session(uint32_t iter) {
  kernel_arg_t kernel_arg = {};
  uint64_t num_cores = 0, num_threads = 0;
  uint32_t nonce = NONCE ^ (iter * 0x9e3779b9u);

  std::cout << "session " << iter << ": open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  uint32_t num_points = count * num_cores;
  uint32_t buf_size = num_points * sizeof(int32_t);

  std::vector<uint32_t> h_src(num_points);
  std::vector<uint32_t> h_dst(num_points);
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src[i] = shuffle(i, nonce);
    h_dst[i] = 0;
  }

  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_buffer_address(src_buffer, &kernel_arg.src_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));
  kernel_arg.count = count;

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::cout << "session " << iter << ": upload + launch + readback" << std::endl;
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
        printf("*** error: [%d] expected=0x%x, actual=0x%x\n", i, ref, cur);
      }
      ++errors;
    }
  }

  std::cout << "session " << iter << ": close device connection" << std::endl;
  cleanup();

  return errors;
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  if (count == 0) {
    count = 1;
  }
  if (iterations < 2) {
    iterations = 2;
  }

  int errors = 0;
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    errors += run_session(iter);
  }

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "Test PASSED" << std::endl;
  return 0;
}
