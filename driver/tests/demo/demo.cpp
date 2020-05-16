#include <iostream>
#include <unistd.h>
#include <string.h>
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

const char* program_file = "kernel.bin";
uint32_t data_stride = 0;

static void show_usage() {
   std::cout << "Vortex Driver Test." << std::endl;
   std::cout << "Usage: [-f: program] [-n stride] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:f:h?")) != -1) {
    switch (c) {
    case 'n':
      data_stride = atoi(optarg);
      break;
    case 'f':
      program_file = optarg;
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

  if (nullptr == program_file) {
    show_usage();
    exit(-1);
  }
}

vx_device_h device = nullptr;
vx_buffer_h buffer = nullptr;

void cleanup() {
  if (buffer) {
    vx_buf_release(buffer);
  }
  if (device) {
    vx_dev_close(device);
  }
}

int run_test(vx_device_h device, 
             vx_buffer_h buffer, 
             const kernel_arg_t& kernel_arg,
             uint32_t buf_size, 
             uint32_t num_points) {
  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, -1));

  // flush the destination buffer caches
  std::cout << "flush the destination buffer caches" << std::endl;
  RT_CHECK(vx_flush_caches(device, kernel_arg.dst_ptr, buf_size));

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(buffer, kernel_arg.dst_ptr, buf_size, 0));

  // verify result
  std::cout << "verify result" << std::endl;  
  {
    int errors = 0;
    auto buf_ptr = (int*)vx_host_ptr(buffer);
    for (uint32_t i = 0; i < num_points; ++i) {
      int ref = i + i; 
      int cur = buf_ptr[i];
      if (cur != ref) {
        std::cout << "error at 0x" << std::hex << (buf_ptr + i)
                << ": actual 0x" << cur << ", expected 0x" << ref << std::endl;
        ++errors;
      }
    }
    if (errors != 0) {
      std::cout << "Found " << errors << " errors!" << std::endl;
      std::cout << "FAILED!" << std::endl;
      return 1;  
    }
  }

  return 0;
}

int main(int argc, char *argv[]) {
  size_t value; 
  kernel_arg_t kernel_arg;
  
  // parse command arguments
  parse_args(argc, argv);

  uint32_t max_cores   = vx_dev_caps(VX_CAPS_MAX_CORES);
  uint32_t max_warps   = vx_dev_caps(VX_CAPS_MAX_WARPS);
  uint32_t max_threads = vx_dev_caps(VX_CAPS_MAX_THREADS);

  if (data_stride == 0) {
    data_stride = 1;
  }

  kernel_arg.stride = data_stride;

  uint32_t num_points = max_cores * max_warps * max_threads;
  uint32_t buf_size = num_points * data_stride * sizeof(uint32_t);

  std::cout << "number of workitems: " << num_points << std::endl;
  std::cout << "workitem size: " << data_stride * sizeof(uint32_t) << " bytes" << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // open device connection
  std::cout << "open device connection" << std::endl;  
  RT_CHECK(vx_dev_open(&device));

  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, program_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;  

  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.src0_ptr = value;

  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.src1_ptr = value;

  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.dst_ptr = value;

  // allocate shared memory  
  std::cout << "allocate shared memory" << std::endl;    
  uint32_t alloc_size = std::max<uint32_t>(buf_size, sizeof(kernel_arg_t));
  RT_CHECK(vx_alloc_shared_mem(device, alloc_size, &buffer));

  // populate source buffer values
  std::cout << "populate source buffer values" << std::endl;    
  {
    auto buf_ptr = (int*)vx_host_ptr(buffer);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = i;
    }
  }

  // upload source buffers
  std::cout << "upload source buffers" << std::endl;      
  RT_CHECK(vx_copy_to_dev(buffer, kernel_arg.src0_ptr, buf_size, 0));
  RT_CHECK(vx_copy_to_dev(buffer, kernel_arg.src1_ptr, buf_size, 0));

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  {
    auto buf_ptr = (int*)vx_host_ptr(buffer);
    memcpy(buf_ptr, &kernel_arg, sizeof(kernel_arg_t));
    RT_CHECK(vx_copy_to_dev(buffer, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0));
  }

  // run tests
  std::cout << "run tests" << std::endl;
  RT_CHECK(run_test(device, buffer, kernel_arg, buf_size, num_points));

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "PASSED!" << std::endl;

  return 0;
}