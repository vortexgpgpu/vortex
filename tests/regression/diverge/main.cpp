#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include <vector>
#include <assert.h>
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
uint32_t count = 0;

vx_device_h device = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n words] [-h: help]" << std::endl;
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
    vx_mem_free(src_buffer);
    vx_mem_free(dst_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

void gen_src_data(std::vector<int>& src_data, uint32_t size) {
  src_data.resize(size);
  for (uint32_t i = 0; i < size; ++i) {
    int value = std::rand();
    src_data[i] = value;
    //std::cout << std::dec << i << ": value=0x" << std::hex << value << std::endl;
  }
}

void gen_ref_data(std::vector<int>& ref_data, const std::vector<int>& src_data, uint32_t size) {
  ref_data.resize(size);
  for (int i = 0; i < (int)size; ++i) {
    int value = src_data.at(i);

    key_t key;
    uint32_t samples = size;
    while (samples--) {
      if ((i & 0x1) == 0) {
        value += 1;
      }
    }

    // none taken
    if (i >= 0x7fffffff) {
      value = 0;
    } else {
      value += 2;
    }

    // diverge
    if (i > 1) {
      if (i > 2) {
        value += 6;
      } else {
        value += 5;
      }
    } else {
      if (i > 0) {
        value += 4;
      } else {
        value += 3;
      }
    }

    // all taken
    if (i >= 0) {
      value += 7;
    } else {
      value = 0;
    }

    // loop
    for (int j = 0, n = i; j < n; ++j) {
      value += src_data.at(j);
    }

    // switch
    switch (i) {
    case 0:
      value += 1;
      break;
    case 1:
      value -= 1;
      break;
    case 2:
      value *= 3;
      break;
    case 3:
      value *= 5;
      break;
    default:
      assert(i < (int)size);
      break;
    }

    // select
    value += (i >= 0) ? ((i > 5) ? src_data.at(0) : i) : ((i < 5) ? src_data.at(1) : -i);

    // min/max
	  value += std::min(src_data.at(i), value);
	  value += std::max(src_data.at(i), value);

    ref_data[i] = value;
  }
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  if (count == 0) {
    count = 1;
  }

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));

  uint32_t total_threads = num_cores * num_warps * num_threads;
  uint32_t num_points = count * total_threads;
  uint32_t buf_size = num_points * sizeof(int32_t);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  kernel_arg.num_points = num_points;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_mem_address(src_buffer, &kernel_arg.src_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_mem_address(dst_buffer, &kernel_arg.dst_addr));

  std::cout << "dev_src=0x" << std::hex << kernel_arg.src_addr << std::endl;
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::endl;

  // allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<int32_t> h_src;
  std::vector<int32_t> h_dst(num_points);
  gen_src_data(h_src, num_points);

  // upload source buffer
  std::cout << "upload source buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(src_buffer, h_src.data(), 0, buf_size));

  // upload program
  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_dst.data(), dst_buffer, 0, buf_size));

  // verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<int32_t> h_ref;
    gen_ref_data(h_ref, h_src, num_points);

    for (uint32_t i = 0; i < num_points; ++i) {
      int ref = h_ref[i];
      int cur = h_dst[i];
      if (cur != ref) {
        std::cout << "error at result #" << std::dec << i
                  << std::hex << ": actual 0x" << cur << ", expected 0x" << ref << std::endl;
        ++errors;
      }
    }
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}