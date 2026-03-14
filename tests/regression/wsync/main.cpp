#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <vortex.h>
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
vx_buffer_h results_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
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
    vx_mem_free(results_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  if (0 == iterations) {
    std::cout << "Error: iterations must be greater than zero" << std::endl;
    return -1;
  }

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint64_t num_threads = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));

  kernel_arg.num_threads = static_cast<uint32_t>(num_threads);
  kernel_arg.iterations = iterations;

  std::cout << "warp size: " << kernel_arg.num_threads << std::endl;
  std::cout << "iterations: " << kernel_arg.iterations << std::endl;

  std::cout << "allocate device memory" << std::endl;
  uint32_t results_size = kernel_arg.num_threads * sizeof(lane_result_t);
  RT_CHECK(vx_mem_alloc(device, results_size, VX_MEM_READ_WRITE, &results_buffer));
  RT_CHECK(vx_mem_address(results_buffer, &kernel_arg.results_addr));

  std::vector<lane_result_t> results(kernel_arg.num_threads);

  RT_CHECK(vx_copy_to_dev(results_buffer, results.data(), 0, results_size));

  std::cout << "upload kernel" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  std::cout << "upload args" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::cout << "download results" << std::endl;
  RT_CHECK(vx_copy_from_dev(results.data(), results_buffer, 0, results_size));

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
