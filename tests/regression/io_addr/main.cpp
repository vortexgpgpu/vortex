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

const char* kernel_file = "kernel.vxbin";
uint32_t count = 0;

static uint64_t io_base_addr = IO_CSR_ADDR + IO_CSR_SIZE;

vx_device_h device = nullptr;
uint64_t usr_test_mem;
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
    vx_mem_free(device, kernel_arg.src_addr);
    vx_mem_free(device, kernel_arg.dst_addr);
    vx_mem_free(device, kernel_prog_addr);
    vx_mem_free(device, kernel_args_addr);
    vx_mem_free(device, usr_test_mem);
    vx_dev_close(device);
  }
}

void gen_src_addrs(std::vector<uint64_t>& src_addrs, uint32_t size) {
  src_addrs.resize(size);
  uint32_t u = 0, k = 0;
  for (uint32_t i = 0; i < size; ++i) {
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

void gen_ref_data(std::vector<int32_t>& ref_data, uint32_t size) {
  ref_data.resize(size);
  for (uint32_t i = 0; i < size; ++i) {
    int32_t j = i % NUM_ADDRS;
    ref_data[i] = j * j;
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

  uint32_t num_tasks = num_cores * num_warps * num_threads;
  uint32_t num_points = count * num_tasks;

  uint32_t src_buf_size = NUM_ADDRS * sizeof(int32_t);
  uint32_t addr_buf_size = num_points * sizeof(uint64_t);
  uint32_t dst_buf_size = num_points * sizeof(int32_t);

  std::cout << "number of points: " << std::dec << num_points << std::endl;
  std::cout << "usr buffer size: " << src_buf_size << " bytes" << std::endl;
  std::cout << "addr buffer size: " << addr_buf_size << " bytes" << std::endl;
  std::cout << "dst buffer size: " << dst_buf_size << " bytes" << std::endl;
  
  kernel_arg.num_points = num_points;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;  
  RT_CHECK(vx_mem_alloc(device, src_buf_size, &usr_test_mem));
  RT_CHECK(vx_mem_alloc(device, addr_buf_size, &kernel_arg.src_addr));
  RT_CHECK(vx_mem_alloc(device, dst_buf_size, &kernel_arg.dst_addr));

  std::cout << "dev_src=0x" << std::hex << kernel_arg.src_addr << std::endl;
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::endl;
  
  // allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<uint64_t> h_addr;
  std::vector<uint32_t> h_src(NUM_ADDRS);  
  std::vector<int32_t> h_dst(num_points);

  // generate source data
  gen_src_addrs(h_addr, num_points);
  for (uint32_t i = 0; i < NUM_ADDRS; ++i) {
    h_src[i] = i * i;
  }
  
  // upload user address data
  std::cout << "upload source buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(device, usr_test_mem, h_src.data(), src_buf_size));
  RT_CHECK(vx_copy_to_dev(device, io_base_addr, h_src.data(), src_buf_size));
  
  // upload source buffer
  std::cout << "upload address buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(device, kernel_arg.src_addr, h_addr.data(), addr_buf_size));
  
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
  RT_CHECK(vx_copy_from_dev(device, h_dst.data(), kernel_arg.dst_addr, dst_buf_size));

  // verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  {
    std::vector<int32_t> h_ref;
    gen_ref_data(h_ref, num_points);

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