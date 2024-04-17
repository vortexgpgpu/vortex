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
uint32_t count = 0;

vx_device_h device = nullptr;
uint64_t kernel_prog_addr;
uint64_t kernel_args_addr;
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
    vx_mem_free(device, kernel_arg.src0_addr);
    vx_mem_free(device, kernel_arg.src1_addr);
    vx_mem_free(device, kernel_arg.dst_addr);
    vx_mem_free(device, kernel_prog_addr);
    vx_mem_free(device, kernel_args_addr);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {  
  // parse command arguments
  parse_args(argc, argv);

  if (count == 0) {
    count = 1;
  }

  // open device connection
  std::cout << "open device connection" << std::endl;  
  RT_CHECK(vx_dev_open(&device));

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));

  uint32_t num_tasks  = num_cores * num_warps * num_threads;
  uint32_t num_points = count * num_tasks;
  uint32_t buf_size   = num_points * sizeof(int32_t);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  kernel_arg.num_tasks = num_tasks;
  kernel_arg.task_size = count;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, buf_size, &kernel_arg.src0_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, &kernel_arg.src1_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, &kernel_arg.dst_addr));

  std::cout << "dev_src0=0x" << std::hex << kernel_arg.src0_addr << std::endl;
  std::cout << "dev_src1=0x" << std::hex << kernel_arg.src1_addr << std::endl;
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::endl;
  
  // allocate host buffers  
  std::cout << "allocate host buffers" << std::endl;
  std::vector<int32_t> h_src0(num_points);
  std::vector<int32_t> h_src1(num_points);
  std::vector<int32_t> h_dst(num_points);

  // generate source data
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src0[i] = i-1;
    h_src1[i] = i+1;
  }  

  // upload source buffer0
  std::cout << "upload source buffer0" << std::endl;    
  RT_CHECK(vx_copy_to_dev(device, kernel_arg.src0_addr, h_src0.data(), buf_size));

  // upload source buffer1
  std::cout << "upload source buffer1" << std::endl;
  RT_CHECK(vx_copy_to_dev(device, kernel_arg.src1_addr, h_src1.data(), buf_size));
  
  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &kernel_prog_addr));
  
  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &kernel_args_addr));

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, kernel_prog_addr, kernel_args_addr));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(device, h_dst.data(), kernel_arg.dst_addr, buf_size));

  // verify result
  std::cout << "verify result" << std::endl;  
  int errors = 0;      
  for (uint32_t i = 0; i < num_points; ++i) {
    int ref = i + i; 
    int cur = h_dst[i];
    if (cur != ref) {
      std::cout << "error at result #" << std::dec << i
                << std::hex << ": actual 0x" << cur << ", expected 0x" << ref << std::endl;
      ++errors;
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