#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include <vector>
#include <VX_config.h>
#include "common.h"

#define NUM_ADDRS 16

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

static uint64_t io_base_addr = IO_CSR_ADDR + IO_CSR_SIZE;

uint64_t usr_test_mem;

std::vector<uint64_t> src_addrs;
std::vector<int32_t> ref_data;

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
    vx_mem_free(device, usr_test_mem);
    vx_dev_close(device);
  }
}

void gen_src_addrs(uint32_t num_points) {
  src_addrs.resize(num_points);

  uint32_t u = 0, k = 0;
  for (uint32_t i = 0; i < num_points; ++i) {
    if (0 ==(i % 4)) {      
      k = (i + u) % NUM_ADDRS;
      ++u;
    }
    uint32_t j = i % NUM_ADDRS;    
    uint64_t a = ((j == k) ? usr_test_mem : io_base_addr) + j * sizeof(uint32_t);    
    std::cout << std::dec << i << "," << k << ": value=0x" << std::hex << a << std::endl;
    src_addrs[i] = a;
  }
}

void gen_ref_data(uint32_t num_points) {
  ref_data.resize(num_points);

  for (uint32_t i = 0; i < num_points; ++i) {
    int32_t j = i % NUM_ADDRS;
    ref_data[i] = j * j;
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
  uint64_t value;
  
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

  RT_CHECK(vx_mem_alloc(device, NUM_ADDRS * sizeof(int32_t), VX_MEM_TYPE_GLOBAL, &usr_test_mem));

  // generate input data
  gen_src_addrs(num_points);

  // generate reference data
  gen_ref_data(num_points);

  uint32_t src_buf_size = num_points * sizeof(uint64_t);  
  uint32_t dst_buf_size = num_points * sizeof(int32_t);

  std::cout << "number of points: " << std::dec << num_points << std::endl;

  // upload program
  std::cout << "upload program" << std::endl;  
  RT_CHECK(vx_upload_kernel_file(device, kernel_file));

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;  

  RT_CHECK(vx_mem_alloc(device, src_buf_size, VX_MEM_TYPE_GLOBAL, &value));
  kernel_arg.src_addr = value;
  RT_CHECK(vx_mem_alloc(device, dst_buf_size, VX_MEM_TYPE_GLOBAL, &value));
  kernel_arg.dst_addr = value;
  kernel_arg.num_points = num_points;

  std::cout << "dev_src=0x" << std::hex << kernel_arg.src_addr << std::endl;
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::endl;
  
  // allocate staging buffer  
  std::cout << "allocate staging buffer" << std::endl;    
  uint32_t staging_buf_size = std::max<uint32_t>(NUM_ADDRS * sizeof(uint64_t),
                                std::max<uint32_t>(src_buf_size,
                                  std::max<uint32_t>(dst_buf_size, 
                                    sizeof(kernel_arg_t))));
  staging_buf.resize(staging_buf_size);
  
  // upload kernel argument  
  std::cout << "upload kernel argument" << std::endl;
  memcpy(staging_buf.data(), &kernel_arg, sizeof(kernel_arg_t));
  RT_CHECK(vx_copy_to_dev(device, KERNEL_ARG_DEV_MEM_ADDR, staging_buf.data(), sizeof(kernel_arg_t)));
  
  // upload test address data
  {
    std::cout << "upload test address data" << std::endl;
    auto buf_ptr = (int32_t*)staging_buf.data();
    for (uint32_t i = 0; i < NUM_ADDRS; ++i) {
      buf_ptr[i] = i * i;
    }
    RT_CHECK(vx_copy_to_dev(device, io_base_addr, staging_buf.data(), NUM_ADDRS * sizeof(int32_t)));
    RT_CHECK(vx_copy_to_dev(device, usr_test_mem, staging_buf.data(), NUM_ADDRS * sizeof(int32_t)));
  }

  // upload source buffer
  {
    std::cout << "upload source buffer" << std::endl;      
    auto buf_ptr = (uint64_t*)staging_buf.data();
    memcpy(buf_ptr, src_addrs.data(), src_buf_size);
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