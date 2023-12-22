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

const char* kernel_file = "kernel.bin";
uint32_t count = 0;

std::vector<int> src_data;
std::vector<int> ref_data;

vx_device_h device = nullptr;
std::vector<uint8_t> staging_buf;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n words] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:h?")) != -1) {
    switch (c) {
    case 'n':
      count = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'h':
    case '?': {
      show_usage();
      exit(0);
    } break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(device, kernel_arg.src_addr);
    vx_mem_free(device, kernel_arg.dst_addr);
    vx_dev_close(device);
  }
}

void gen_input_data(uint32_t num_points) {
  src_data.resize(num_points);

  for (uint32_t i = 0; i < src_data.size(); ++i) {
    int value = std::rand();
    src_data[i] = value;
    //std::cout << std::dec << i << ": value=0x" << std::hex << value << std::endl;
  }
}

void gen_ref_data(uint32_t num_points) {
  ref_data.resize(num_points);

  for (int i = 0; i < (int)ref_data.size(); ++i) {
    int value = src_data.at(i);

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
      assert(i < (int)num_points);
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

int run_test(const kernel_arg_t& kernel_arg,
             uint32_t buf_size, 
             uint32_t num_points) {
  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(device, staging_buf.data(), kernel_arg.dst_addr, buf_size));

  // verify result
  std::cout << "verify result" << std::endl;  
  {
    int errors = 0;
    auto buf_ptr = (int32_t*)staging_buf.data();
    for (uint32_t i = 0; i < num_points; ++i) {
      int ref = ref_data.at(i);
      int cur = buf_ptr[i];
      if (cur != ref) {
        std::cout << "error at result #" << std::dec << i
                  << std::hex << ": actual 0x" << cur << ", expected 0x" << ref << std::endl;
        ++errors;
      }
    }
    if (errors != 0) {
      std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
      std::cout << "FAILED!" << std::endl;
      return 1;  
    }
  }

  return 0;
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

  uint32_t num_points = count;

  // generate input data
  gen_input_data(num_points);

  // generate reference data
  gen_ref_data(num_points);

  uint32_t src_buf_size = src_data.size() * sizeof(int32_t);  
  uint32_t dst_buf_size = ref_data.size() * sizeof(int32_t);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << dst_buf_size << " bytes" << std::endl;

  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;  
  RT_CHECK(vx_mem_alloc(device, src_buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.src_addr));
  RT_CHECK(vx_mem_alloc(device, dst_buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.dst_addr));

  kernel_arg.num_points = num_points;

  std::cout << "dev_src=0x" << std::hex << kernel_arg.src_addr << std::endl;
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::endl;
  
  // allocate staging buffer  
  std::cout << "allocate staging buffer" << std::endl;    
  uint32_t staging_buf_size = std::max<uint32_t>(src_buf_size,
                                  std::max<uint32_t>(dst_buf_size, 
                                    sizeof(kernel_arg_t)));
  staging_buf.resize(staging_buf_size);
  
  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  memcpy(staging_buf.data(), &kernel_arg, sizeof(kernel_arg_t));
  RT_CHECK(vx_copy_to_dev(device, KERNEL_ARG_DEV_MEM_ADDR, staging_buf.data(), sizeof(kernel_arg_t)));

  // upload source buffer
  {
    std::cout << "upload source buffer" << std::endl;
    auto buf_ptr = staging_buf.data();
    memcpy(buf_ptr, src_data.data(), num_points * sizeof(int32_t));          
    RT_CHECK(vx_copy_to_dev(device, kernel_arg.src_addr, staging_buf.data(), src_buf_size));
  }

  // clear destination buffer
  {
    std::cout << "clear destination buffer" << std::endl;
    auto buf_ptr = (int32_t*)staging_buf.data();
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = 0xdeadbeef;
    }          
    RT_CHECK(vx_copy_to_dev(device, kernel_arg.dst_addr, staging_buf.data(), dst_buf_size));  
  }

  // run tests
  std::cout << "run tests" << std::endl;
  RT_CHECK(run_test(kernel_arg, dst_buf_size, num_points));

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "PASSED!" << std::endl;

  return 0;
}