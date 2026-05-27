#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
#include <vortex.h>
#include "common.h"

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
	 cleanup();			                                              \
     exit(-1);                                                  \
   } while (false)

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.vxbin";
uint32_t num_points = 64;
uint32_t stride = 1;
bool enable_dfv_test = false;

vx_device_h device = nullptr;
vx_buffer_h src0_buffer = nullptr;
vx_buffer_h src1_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "DFV Stress Test." << std::endl;
   std::cout << "Usage: [-k kernel] [-n points] [-s stride] [-d: DFV] [-h help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "dn:s:k:h")) != -1) {
    switch (c) {
    case 'n':
      num_points = atoi(optarg);
      break;
    case 's':
      stride = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'd':
      enable_dfv_test = true;
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
    vx_mem_free(src0_buffer);
    vx_mem_free(src1_buffer);
    vx_mem_free(dst_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int run_phase(int phase) {
  // Buffer size accounts for strided access pattern
  uint32_t max_idx = num_points * stride;
  uint32_t buf_size = max_idx * sizeof(TYPE);

  std::vector<TYPE> h_src0(max_idx, 0);
  std::vector<TYPE> h_src1(max_idx, 0);
  std::vector<TYPE> h_dst(max_idx, 0);

  // Generate source data at strided positions
  for (uint32_t i = 0; i < num_points; ++i) {
    uint32_t idx = i * stride;
    h_src0[idx] = (TYPE)(i + 1);
    h_src1[idx] = (TYPE)(i * 2);
  }

  // Upload source buffers
  RT_CHECK(vx_copy_to_dev(src0_buffer, h_src0.data(), 0, buf_size));
  RT_CHECK(vx_copy_to_dev(src1_buffer, h_src1.data(), 0, buf_size));

  // Configure kernel arguments
  kernel_arg.num_points = num_points;
  kernel_arg.stride = stride;
  kernel_arg.enable_dfv_test = enable_dfv_test ? 1 : 0;
  kernel_arg.dfv_phase = phase;

  // Upload kernel argument
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  // Start device
  std::cout << "  Phase " << phase << ": starting..." << std::flush;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // Download and verify
  RT_CHECK(vx_copy_from_dev(h_dst.data(), dst_buffer, 0, buf_size));

  int errors = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    uint32_t idx = i * stride;
    TYPE expected = h_src0[idx] + h_src1[idx];
    if (h_dst[idx] != expected) {
      if (errors < 10) {
        printf("\n  *** error: phase=%d [%d] expected=%d, actual=%d",
               phase, idx, (int)expected, (int)h_dst[idx]);
      }
      ++errors;
    }
  }

  if (errors == 0) {
    std::cout << " PASSED" << std::endl;
  } else {
    std::cout << " FAILED (" << errors << " errors)" << std::endl;
  }
  return errors;
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  std::srand(50);

  std::cout << "DFV Stress Test" << std::endl;
  std::cout << "  points=" << num_points << ", stride=" << stride
            << ", dfv=" << (enable_dfv_test ? "ON" : "OFF") << std::endl;

  // Open device
  RT_CHECK(vx_dev_open(&device));

  uint32_t max_idx = num_points * stride;
  uint32_t buf_size = max_idx * sizeof(TYPE);

  // Allocate device memory
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &src0_buffer));
  RT_CHECK(vx_mem_address(src0_buffer, &kernel_arg.src0_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &src1_buffer));
  RT_CHECK(vx_mem_address(src1_buffer, &kernel_arg.src1_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_mem_address(dst_buffer, &kernel_arg.dst_addr));

  // Upload kernel
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // Run all DFV phases
  int total_errors = 0;
  int num_phases = enable_dfv_test ? DFV_NUM_PHASES : 1;

  for (int phase = 0; phase < num_phases; ++phase) {
    total_errors += run_phase(phase);
  }

  // Cleanup
  cleanup();

  if (total_errors != 0) {
    std::cout << "FAILED! (" << total_errors << " total errors)" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
