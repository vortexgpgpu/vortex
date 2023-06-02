#include <iostream>
#include <vector>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include <VX_config.h>
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
int count    = 0;
bool stop_on_error = true;

vx_device_h device   = nullptr;
vx_buffer_h arg_buf  = nullptr;
vx_buffer_h src1_buf = nullptr;
vx_buffer_h src2_buf = nullptr;
vx_buffer_h src3_buf = nullptr;
vx_buffer_h src4_buf = nullptr;
vx_buffer_h dst_buf  = nullptr;
bool use_sw = false;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-n words] [-c] [-z no_hw]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:czh?")) != -1) {
    switch (c) {
    case 'n':
      count = atoi(optarg);
      break;
    case 'k':
      kernel_file = optarg;
      break;
    case 'c':
      stop_on_error = false;
      break;
    case 'z':
      use_sw = true;
      break;
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
  if (arg_buf) {
    vx_buf_free(arg_buf);
  }
   if (src1_buf) {
    vx_buf_free(src1_buf);
  }
  if (src2_buf) {
    vx_buf_free(src2_buf);
  }
  if (src3_buf) {
    vx_buf_free(src3_buf);
  }
  if (src4_buf) {
    vx_buf_free(src4_buf);
  }
  if (dst_buf) {
    vx_buf_free(dst_buf);
  }
  if (device) {
    vx_mem_free(device, kernel_arg.src0_addr);
    vx_mem_free(device, kernel_arg.src1_addr);
    vx_mem_free(device, kernel_arg.src2_addr);
    vx_mem_free(device, kernel_arg.src3_addr);
    vx_mem_free(device, kernel_arg.dst_addr);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  // open device connection
  std::cout << "open device connection" << std::endl;  
  RT_CHECK(vx_dev_open(&device));

  uint64_t isa_flags;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if (0 == (isa_flags & (VX_ISA_EXT_IMADD))) {
    std::cout << "IMADD extensions not supported!" << std::endl;
    cleanup();
    return -1;
  }  

  if (count == 0) {
    count = 1;
  }

  std::cout << std::dec;

  std::cout << "workitem size: " << count << std::endl;
  std::cout << "using kernel: " << kernel_file << std::endl;

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));

  uint32_t num_tasks  = num_cores * num_warps * num_threads;
  uint32_t num_points = count * num_tasks;
  size_t buf_size = num_points * sizeof(uint32_t);
  
  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // upload program
  std::cout << "upload kernel" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, buf_size, &kernel_arg.src0_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, &kernel_arg.src1_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, &kernel_arg.src2_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, &kernel_arg.src3_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, &kernel_arg.dst_addr));

  kernel_arg.num_tasks = num_tasks;
  kernel_arg.task_size = count;
  kernel_arg.use_sw    = use_sw;

  std::cout << "dev_src0=" << std::hex << kernel_arg.src0_addr << std::dec << std::endl;
  std::cout << "dev_src1=" << std::hex << kernel_arg.src1_addr << std::dec << std::endl;
  std::cout << "dev_src2=" << std::hex << kernel_arg.src2_addr << std::dec << std::endl;
  std::cout << "dev_src3=" << std::hex << kernel_arg.src3_addr << std::dec << std::endl;
  std::cout << "dev_dst="  << std::hex << kernel_arg.dst_addr << std::dec << std::endl;
  
  // allocate staging buffer  
  std::cout << "allocate staging buffer" << std::endl;
  RT_CHECK(vx_buf_alloc(device, sizeof(kernel_arg_t), &arg_buf));
  RT_CHECK(vx_buf_alloc(device, buf_size, &src1_buf));
  RT_CHECK(vx_buf_alloc(device, buf_size, &src2_buf));
  RT_CHECK(vx_buf_alloc(device, buf_size, &src3_buf));
  RT_CHECK(vx_buf_alloc(device, buf_size, &src4_buf));
  RT_CHECK(vx_buf_alloc(device, buf_size, &dst_buf));

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  memcpy((void*)vx_host_ptr(arg_buf), &kernel_arg, sizeof(kernel_arg_t));
  RT_CHECK(vx_copy_to_dev(arg_buf, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0));

  // get test arguments
  std::cout << "get test arguments" << std::endl;
  auto a = (int32_t*)vx_host_ptr(src1_buf);
  auto b = (int32_t*)vx_host_ptr(src2_buf);
  auto c = (int32_t*)vx_host_ptr(src3_buf);
  auto s = (int32_t*)vx_host_ptr(src4_buf);
  for (uint32_t i = 0; i < num_points; ++i) {
    a[i] = num_points/2 + i;
    b[i] = num_points/2 + i;
    c[i] = (num_points + i) / 2;
    s[i] = (i % 4);
  }

  // upload source buffer0
  std::cout << "upload source buffer0" << std::endl;      
  RT_CHECK(vx_copy_to_dev(src1_buf, kernel_arg.src0_addr, buf_size, 0));
  
  // upload source buffer1
  std::cout << "upload source buffer1" << std::endl;      
  RT_CHECK(vx_copy_to_dev(src2_buf, kernel_arg.src1_addr, buf_size, 0));

  // upload source buffer2
  std::cout << "upload source buffer2" << std::endl;      
  RT_CHECK(vx_copy_to_dev(src3_buf, kernel_arg.src2_addr, buf_size, 0));

  // upload source buffer3
  std::cout << "upload source buffer2" << std::endl;      
  RT_CHECK(vx_copy_to_dev(src4_buf, kernel_arg.src3_addr, buf_size, 0));

  // clear destination buffer    
  {
    std::cout << "clear destination buffer" << std::endl;   
    auto buf_ptr = (int32_t*)vx_host_ptr(dst_buf);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = 0xdeadbeef;
    }
    RT_CHECK(vx_copy_to_dev(dst_buf, kernel_arg.dst_addr, buf_size, 0));
  }

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, MAX_TIMEOUT));

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(dst_buf, kernel_arg.dst_addr, buf_size, 0));

  // verify destination
  std::cout << "verify test result" << std::endl;
  uint32_t errors = 0;
  auto d = (int32_t*)vx_host_ptr(dst_buf);
  for (uint32_t i = 0; i < num_points; ++i) {
    auto ref = ((a[i] * b[i]) >> (s[i] * 8)) + c[i];
    if (d[i] != ref) {
      std::cout << "error at result #" << i << ": expected=" << ref << ", actual=" << c[i] << ", a=" << a[i] << ", b=" << b[i] << ", c=" << c[i] << ", s=" << s[i] << std::endl;
      ++errors;
    }
  }
  
  int exitcode = 0;

  if (errors != 0) {
    std::cout << "found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "Test FAILED!" << std::endl << std::flush;
    if (stop_on_error) {
      cleanup();
      exit(1);  
    }
    exitcode = 1;
  } else {
    std::cout << "Test PASSED!" << std::endl << std::flush;
  }

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  return exitcode;
}