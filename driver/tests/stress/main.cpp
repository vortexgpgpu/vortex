#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include "common.h"
#include <assert.h>
#include <vector>

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

std::vector<int> test_data;
std::vector<uint32_t> addr_table;

vx_device_h device = nullptr;
vx_buffer_h staging_buf = nullptr;

static void show_usage() {
   std::cout << "Vortex Driver Test." << std::endl;
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

void gen_input_data(uint32_t num_points) {
  test_data.resize(num_points);
  addr_table.resize(num_points + NUM_LOADS - 1);

  for (uint32_t i = 0; i < test_data.size(); ++i) {
    test_data[i] = std::rand();
  }

  for (uint32_t i = 0; i < addr_table.size(); ++i) {
    float r = static_cast<float>(std::rand()) / RAND_MAX;
    uint32_t index = static_cast<uint32_t>(r * num_points);
    assert(index < num_points);
    addr_table[i] = index;
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
  RT_CHECK(vx_ready_wait(device, -1));

  // download destination staging_buf
  std::cout << "download destination staging_buf" << std::endl;
  RT_CHECK(vx_copy_from_dev(staging_buf, kernel_arg.dst_ptr, buf_size, 0));

  // verify result
  std::cout << "verify result" << std::endl;  
  {
    int errors = 0;
    auto buf_ptr = (int32_t*)vx_host_ptr(staging_buf);

    uint32_t size_m1 = num_points - 1;
    
    for (uint32_t i = 0; i < num_points; ++i) {

      int ref = 0;
      for (uint32_t j = 0; j < NUM_LOADS; ++j) {
        uint32_t addr = i + j;
        uint32_t index = addr_table.at(addr);
        int value = test_data.at(index);
        printf("*** [%d] addr=%d, index=%d, value=%d\n", i, addr, index, value);
        ref += value;
      }
      
      int cur = buf_ptr[i];
      if (cur != ref) {
        std::cout << "error at result #" << std::dec << i
                  << ": actual " << cur << ", expected " << ref << std::endl;
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
  kernel_arg_t kernel_arg;
  
  // parse command arguments
  parse_args(argc, argv);

  if (count == 0) {
    count = 1;
  }

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;  
  RT_CHECK(vx_dev_open(&device));

  unsigned max_cores, max_warps, max_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_CORES, &max_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_WARPS, &max_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_THREADS, &max_threads));

  uint32_t num_tasks  = max_cores * max_warps * max_threads;
  uint32_t num_points = count * num_tasks;
  uint32_t buf_size   = num_points * sizeof(uint32_t);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "staging_buf size: " << buf_size << " bytes" << std::endl;

  // generate input data
  gen_input_data(num_points);

  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;  

  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.src0_ptr = value;
  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.src1_ptr = value;
  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.dst_ptr = value;

  kernel_arg.num_tasks = num_tasks;
  kernel_arg.stride = count;

  std::cout << "dev_src0=" << std::hex << kernel_arg.src0_ptr << std::endl;
  std::cout << "dev_src1=" << std::hex << kernel_arg.src1_ptr << std::endl;
  std::cout << "dev_dst=" << std::hex << kernel_arg.dst_ptr << std::endl;
  
  // allocate shared memory  
  std::cout << "allocate shared memory" << std::endl;    
  uint32_t alloc_size = std::max<uint32_t>(buf_size, sizeof(kernel_arg_t));
  RT_CHECK(vx_alloc_shared_mem(device, alloc_size, &staging_buf));
  
  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  {
    auto buf_ptr = (int*)vx_host_ptr(staging_buf);
    memcpy(buf_ptr, &kernel_arg, sizeof(kernel_arg_t));
    RT_CHECK(vx_copy_to_dev(staging_buf, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0));
  }

  // upload source buffer0
  {
    auto buf_ptr = (int32_t*)vx_host_ptr(staging_buf);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = test_data.at(i);
    }
  }
  std::cout << "upload source buffer0" << std::endl;      
  RT_CHECK(vx_copy_to_dev(staging_buf, kernel_arg.src0_ptr, buf_size, 0));

  // upload source buffer1
  {
    auto buf_ptr = (int32_t*)vx_host_ptr(staging_buf);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = addr_table.at(i);
    }
  }
  std::cout << "upload source buffer1" << std::endl;      
  RT_CHECK(vx_copy_to_dev(staging_buf, kernel_arg.src1_ptr, buf_size, 0));

  // clear destination staging_buf
  {
    auto buf_ptr = (int32_t*)vx_host_ptr(staging_buf);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = 0xdeadbeef;
    }
  }
  std::cout << "clear destination staging_buf" << std::endl;      
  RT_CHECK(vx_copy_to_dev(staging_buf, kernel_arg.dst_ptr, buf_size, 0));  

  // run tests
  std::cout << "run tests" << std::endl;
  RT_CHECK(run_test(kernel_arg, buf_size, num_points));

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "PASSED!" << std::endl;

  return 0;
}