#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <vector>
#include <vortex.h>
#include "common.h"

#define RT_CHECK(_expr)                                       \
  do {                                                        \
    int _ret = _expr;                                        \
    if (0 == _ret)                                           \
      break;                                                  \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret); \
    cleanup();                                                \
    exit(-1);                                                 \
  } while (false)

const char* kernel_file = "kernel.vxbin";
uint32_t payload_bytes = 1024;
uint32_t iterations = 4;
uint32_t mode = BARRIER_OVERHEAD_MODE_SOFT;

vx_device_h device = nullptr;
vx_buffer_h results_buffer = nullptr;
vx_queue_h queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex shared-memory atomic barrier test." << std::endl;
  std::cout << "Usage: [-b payload_bytes] [-i iterations] [-m hard|soft] [-k kernel] [-h help]" << std::endl;
}

static const char* mode_name(uint32_t value) {
  switch (value) {
  case BARRIER_OVERHEAD_MODE_HARD:
    return "hard_barrier";
  case BARRIER_OVERHEAD_MODE_SOFT:
    return "soft_smem_atomic";
  default:
    return "unknown";
  }
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "b:i:m:k:h")) != -1) {
    switch (c) {
    case 'b':
      payload_bytes = std::atoi(optarg);
      break;
    case 'i':
      iterations = std::atoi(optarg);
      break;
    case 'm':
      if (std::strcmp(optarg, "hard") == 0) {
        mode = BARRIER_OVERHEAD_MODE_HARD;
      } else if (std::strcmp(optarg, "soft") == 0) {
        mode = BARRIER_OVERHEAD_MODE_SOFT;
      } else {
        show_usage();
        exit(-1);
      }
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
    if (results_buffer) vx_buffer_release(results_buffer);
    if (kernel) vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue) vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  if (payload_bytes == 0 || (payload_bytes % sizeof(uint32_t)) != 0) {
    std::cout << "Error: payload_bytes must be a non-zero multiple of 4" << std::endl;
    return -1;
  }
  if (iterations == 0) {
    std::cout << "Error: iterations must be greater than zero" << std::endl;
    return -1;
  }

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_threads = 0;
  uint64_t num_warps = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));

  kernel_arg.payload_bytes = payload_bytes;
  kernel_arg.iterations = iterations;
  kernel_arg.num_warps = static_cast<uint32_t>(num_warps);
  kernel_arg.mode = mode;

  std::cout << "threads: " << num_threads << std::endl;
  std::cout << "warps: " << kernel_arg.num_warps << std::endl;
  std::cout << "mode: " << mode_name(kernel_arg.mode) << std::endl;
  std::cout << "payload bytes: " << kernel_arg.payload_bytes << std::endl;
  std::cout << "iterations: " << kernel_arg.iterations << std::endl;

  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_buffer_create(device, sizeof(barrier_result_t), VX_MEM_READ_WRITE, &results_buffer));
  RT_CHECK(vx_buffer_address(results_buffer, &kernel_arg.results_addr));

  barrier_result_t zero = {};
  RT_CHECK(vx_enqueue_write(queue, results_buffer, 0, &zero, sizeof(zero), 0, nullptr, nullptr));

  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  uint32_t lmem_size = 64 + payload_bytes;
  RT_CHECK(vx_check_occupancy(device, static_cast<uint32_t>(num_threads * num_warps), lmem_size));

  std::cout << "start device" << std::endl;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    uint32_t grid_one = 1;
    vx_launch_info_t li = {};
    li.struct_size = sizeof(li);
    li.kernel = kernel;
    li.args_host = &kernel_arg;
    li.args_size = sizeof(kernel_arg);
    li.ndim = 1;
    li.grid_dim[0] = grid_one;
    li.block_dim[0] = static_cast<uint32_t>(num_threads * num_warps);
    li.lmem_size = lmem_size;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  barrier_result_t result = {};
  std::cout << "download results" << std::endl;
  RT_CHECK(vx_enqueue_read(queue, &result, results_buffer, 0, sizeof(result), 1, &launch_ev, &read_ev));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  cleanup();

  std::cout << "result: failures=" << result.failures
            << " pending=" << result.observed_pending
            << " phase=" << result.observed_phase
            << " arrived=" << result.observed_arrived
            << " register_cycles=" << result.register_cycles
            << " event_cycles=" << result.event_cycles
            << " release_cycles=" << result.release_cycles
            << " wait_iters=" << result.wait_iters
            << " checksum=" << result.checksum
            << std::endl;
  std::cout << "SMEM_BARRIER_RESULT"
            << " mode=" << mode_name(kernel_arg.mode)
            << " payload_bytes=" << kernel_arg.payload_bytes
            << " iterations=" << kernel_arg.iterations
            << " failures=" << result.failures
            << " pending=" << result.observed_pending
            << " phase=" << result.observed_phase
            << " arrived=" << result.observed_arrived
            << " register_cycles=" << result.register_cycles
            << " event_cycles=" << result.event_cycles
            << " release_cycles=" << result.release_cycles
            << " wait_iters=" << result.wait_iters
            << " checksum=" << result.checksum
            << std::endl;

  if (result.failures != 0) {
    std::cout << "FAILED!" << std::endl;
    return result.failures;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
