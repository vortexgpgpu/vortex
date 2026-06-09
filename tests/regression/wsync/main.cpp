#include <cstdlib>
#include <iostream>
#include <unistd.h>
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

const char* kernel_file = "kernel.vxbin";
uint32_t iterations = 1024;

vx_device_h device = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h results_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex WSYNC Test." << std::endl;
  std::cout << "Usage: [-i iterations] [-k kernel] [-h help]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "i:k:h")) != -1) {
    switch (c) {
    case 'i':
      iterations = std::atoi(optarg);
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
    if (src_buffer)     vx_buffer_release(src_buffer);
    if (results_buffer) vx_buffer_release(results_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  if (0 == iterations) {
    std::cout << "Error: iterations must be greater than zero" << std::endl;
    return -1;
  }

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_threads = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  kernel_arg.num_threads = static_cast<uint32_t>(num_threads);
  kernel_arg.iterations = iterations;

  std::cout << "warp size: " << kernel_arg.num_threads << std::endl;
  std::cout << "iterations: " << kernel_arg.iterations << std::endl;

  std::cout << "allocate device memory" << std::endl;

  uint32_t src_size = WSYNC_BUF_LINES * WSYNC_LINE_WORDS * sizeof(uint32_t);
  RT_CHECK(vx_buffer_create(device, src_size, VX_MEM_READ_WRITE, &src_buffer));
  RT_CHECK(vx_buffer_address(src_buffer, &kernel_arg.src_addr));

  std::vector<uint32_t> src_data(WSYNC_BUF_LINES * WSYNC_LINE_WORDS);
  for (uint32_t i = 0; i < src_data.size(); ++i)
    src_data[i] = i + 1;
  RT_CHECK(vx_enqueue_write(queue, src_buffer, 0, src_data.data(), src_size, 0, nullptr, nullptr));

  uint32_t results_size = kernel_arg.num_threads * sizeof(lane_result_t);
  RT_CHECK(vx_buffer_create(device, results_size, VX_MEM_READ_WRITE, &results_buffer));
  RT_CHECK(vx_buffer_address(results_buffer, &kernel_arg.results_addr));

  std::vector<lane_result_t> results(kernel_arg.num_threads);

  RT_CHECK(vx_enqueue_write(queue, results_buffer, 0, results.data(), results_size, 0, nullptr, nullptr));

  std::cout << "load kernel module" << std::endl;
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::cout << "start device" << std::endl;
  vx_event_h launch_ev = nullptr, read_ev = nullptr;
  {
    uint32_t grid_one = 1;
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = grid_one;
    li.block_dim[0] = kernel_arg.num_threads;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  std::cout << "download results" << std::endl;
  RT_CHECK(vx_enqueue_read(queue, results.data(), results_buffer, 0, results_size, 1, &launch_ev, &read_ev));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  std::cout << "cleanup" << std::endl;
  cleanup();

  uint32_t errors = 0;
  for (uint32_t lane = 0; lane < kernel_arg.num_threads; ++lane) {
    auto& result = results[lane];
    if (0 == result.failures) {
      continue;
    }

    if (0 == errors) {
      std::cout << "first failure: lane=" << lane
                << ", iteration=" << result.first_iteration
                << ", baseline_gap=" << result.baseline_gap
                << ", raw_cycle=" << result.raw_cycle
                << ", sync_cycle=" << result.sync_cycle
                << ", gap=" << result.gap << std::endl;
    }
    errors += result.failures;
  }

  if (0 != errors) {
    std::cout << "WSYNC timing mismatches: " << errors << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
