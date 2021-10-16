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

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.bin";
uint32_t count = 4;

vx_device_h device = nullptr;
vx_buffer_h staging_buf = nullptr;

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
  if (staging_buf) {
    vx_buf_release(staging_buf);
  }
  if (device) {
    vx_dev_close(device);
  }
}

int run_test() {
  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, -1));

  return 0;
}

int main(int argc, char *argv[]) {
  size_t value; 
  kernel_arg_t kernel_arg;
  
  // parse command arguments
  parse_args(argc, argv);

  if (count == 0) {
    count = 1;
  }

  // open device connection
  std::cout << "open device connection" << std::endl;  
  RT_CHECK(vx_dev_open(&device));

  unsigned max_cores, max_warps, max_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_CORES, &max_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_WARPS, &max_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_THREADS, &max_threads));

  uint32_t num_points = count;
  uint32_t buf_size = count * sizeof(int32_t);

  std::cout << "number of points: " << count << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;  

  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.src_ptr = value;

  kernel_arg.num_points = num_points;

  std::cout << "dev_src=" << std::hex << kernel_arg.src_ptr << std::endl;
  
  // allocate shared memory  
  std::cout << "allocate shared memory" << std::endl;    
  uint32_t alloc_size = std::max<uint32_t>(buf_size, sizeof(kernel_arg_t));
  RT_CHECK(vx_alloc_shared_mem(device, alloc_size, &staging_buf));
  
  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  {
    auto buf_ptr = (void*)vx_host_ptr(staging_buf);
    memcpy(buf_ptr, &kernel_arg, sizeof(kernel_arg_t));
    RT_CHECK(vx_copy_to_dev(staging_buf, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0));
  }

  // upload source buffer0
  {
    auto buf_ptr = (float*)vx_host_ptr(staging_buf);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = i-1;
    }
  }
  std::cout << "upload source buffer" << std::endl;      
  RT_CHECK(vx_copy_to_dev(staging_buf, kernel_arg.src_ptr, buf_size, 0));  

  // run tests
  std::cout << "run tests" << std::endl;
  RT_CHECK(run_test());

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "PASSED!" << std::endl;

  return 0;
}