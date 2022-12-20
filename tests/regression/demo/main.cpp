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
uint32_t count = 0;

vx_device_h device = nullptr;
cmdbuffer *cmdBuf = nullptr;
vx_buffer_h staging_buf = nullptr;
vx_buffer_h buf1 = nullptr;
vx_buffer_h buf2 = nullptr;
vx_buffer_h buf3 = nullptr;
kernel_arg_t kernel_arg;

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
    vx_buf_free(staging_buf);
  }
  if (device) {    
    vx_mem_free(device, kernel_arg.src0_addr);
    vx_mem_free(device, kernel_arg.src1_addr);
    vx_mem_free(device, kernel_arg.dst_addr);
    vx_dev_close(device);
  }
}

int run_test(const kernel_arg_t& kernel_arg,
             uint32_t buf_size, 
             uint32_t num_points, cmdbuffer* cmdBuf) {

  vx_flush(cmdBuf);
  
  // start device
  std::cout << "start device" << std::endl;
  vx_new_start(device, cmdBuf);
  RT_CHECK(vx_start(device));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, MAX_TIMEOUT));

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  vx_new_copy_to_dev(buf3, kernel_arg.dst_addr, buf_size, 0, cmdBuf, 1);
  RT_CHECK(vx_copy_from_dev(buf3, kernel_arg.dst_addr, buf_size, 0));

  // verify result
  std::cout << "verify result" << std::endl;  
  {
    int errors = 0;
    auto buf_ptr = (int32_t*)vx_host_ptr(buf3);
    for (uint32_t i = 0; i < num_points; ++i) {
      int ref = i + i; 
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
  size_t value; 
  
  // parse command arguments
  parse_args(argc, argv);

  if (count == 0) {
    count = 1;
  }
  
  // open device connection
  std::cout << "open device connection" << std::endl;  
  RT_CHECK(vx_dev_open(&device));

  uint64_t max_cores, max_warps, max_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_CORES, &max_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_WARPS, &max_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_THREADS, &max_threads));

  uint32_t num_tasks  = max_cores * max_warps * max_threads;
  uint32_t num_points = count * num_tasks;
  uint32_t buf_size   = num_points * sizeof(int32_t);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  cmdBuf = vx_create_command_buffer(8);
  RT_CHECK(vx_buf_alloc(device, buf_size, &cmdBuf->buffer));
  cmdBuf->createHeaderPacket(cmdBuf, 0);

  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, cmdBuf));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;  

  RT_CHECK(vx_mem_alloc(device, buf_size, &value));
  kernel_arg.src0_addr = value;
  RT_CHECK(vx_mem_alloc(device, buf_size, &value));
  kernel_arg.src1_addr = value;
  RT_CHECK(vx_mem_alloc(device, buf_size, &value));
  kernel_arg.dst_addr = value;

  kernel_arg.num_tasks = num_tasks;
  kernel_arg.task_size = count;

  std::cout << "dev_src0=" << std::hex << kernel_arg.src0_addr << std::endl;
  std::cout << "dev_src1=" << std::hex << kernel_arg.src1_addr << std::endl;
  std::cout << "dev_dst=" << std::hex << kernel_arg.dst_addr << std::endl;
  
  // allocate shared memory  
  std::cout << "allocate shared memory" << std::endl;    
  uint32_t alloc_size = std::max<uint32_t>(buf_size, sizeof(kernel_arg_t));
  RT_CHECK(vx_buf_alloc(device, alloc_size, &staging_buf));
  RT_CHECK(vx_buf_alloc(device, alloc_size, &buf1));
  RT_CHECK(vx_buf_alloc(device, alloc_size, &buf2));
  RT_CHECK(vx_buf_alloc(device, alloc_size, &buf3));
  
  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  {
    auto buf_ptr = (int*)vx_host_ptr(staging_buf);
    memcpy(buf_ptr, &kernel_arg, sizeof(kernel_arg_t));
    vx_new_copy_to_dev(staging_buf, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0, cmdBuf, 2); // append to command buffer
    RT_CHECK(vx_copy_to_dev(staging_buf, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0));
  }

  // upload source buffer0
  {
    auto buf_ptr = (int32_t*)vx_host_ptr(buf1);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = i-1;
    }
  }
  std::cout << "upload source buffer0" << std::endl;   
  vx_new_copy_to_dev(buf1, kernel_arg.src0_addr, buf_size, 0, cmdBuf, 2); // append to command buffer   
  RT_CHECK(vx_copy_to_dev(buf1, kernel_arg.src0_addr, buf_size, 0));

  // upload source buffer1
  {
    auto buf_ptr = (int32_t*)vx_host_ptr(buf2);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = i+1;
    }
  }
  std::cout << "upload source buffer1" << std::endl;    
  vx_new_copy_to_dev(buf2, kernel_arg.src1_addr, buf_size, 0, cmdBuf, 2);  
  RT_CHECK(vx_copy_to_dev(buf2, kernel_arg.src1_addr, buf_size, 0));

  // clear destination buffer
  {
    auto buf_ptr = (int32_t*)vx_host_ptr(buf3);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = 0xdeadbeef;
    }
  }
  std::cout << "clear destination buffer" << std::endl;
  vx_new_copy_to_dev(buf3, kernel_arg.dst_addr, buf_size, 0, cmdBuf, 2);   
  RT_CHECK(vx_copy_to_dev(buf3, kernel_arg.dst_addr, buf_size, 0));

  //vx_flush(cmdBuf);

  // run tests
  std::cout << "run tests" << std::endl;
  RT_CHECK(run_test(kernel_arg, buf_size, num_points, cmdBuf));

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "PASSED!" << std::endl;

  return 0;
}
