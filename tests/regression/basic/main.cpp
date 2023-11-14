#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include <chrono>
#include <vector>
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
int test = -1;
uint32_t count = 0;

vx_device_h device = nullptr;
std::vector<uint8_t> staging_buf;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-t testno][-k: kernel][-n words][-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:t:k:h?")) != -1) {
    switch (c) {
    case 'n':
      count = atoi(optarg);
      break;
    case 't':
      test = atoi(optarg);
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

uint64_t shuffle(int i, uint64_t value) {
  return (value << i) | (value & ((1 << i)-1));;
}

int run_memcopy_test(uint32_t dev_addr, uint64_t value, int num_blocks) {
  int errors = 0;
  
  auto time_start = std::chrono::high_resolution_clock::now();

  int num_blocks_8 = (64 * num_blocks) / 8;

  // update source buffer  
  for (int i = 0; i < num_blocks_8; ++i) {
    ((uint64_t*)staging_buf.data())[i] = shuffle(i, value);
  }

  /*for (int i = 0; i < num_blocks; ++i) {
    std::cout << "data[" << i << "]=0x";
    for (int j = 7; j >= 0; --j) {
      std::cout << std::hex << ((uint64_t*)staging_buf.data())[i * 8 +j];
    }
    std::cout << std::endl;
  }*/
  
  // write source buffer to local memory
  std::cout << "write source buffer to local memory" << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_copy_to_dev(device, dev_addr, staging_buf.data(), 64 * num_blocks));
  auto t1 = std::chrono::high_resolution_clock::now();

  // clear destination buffer
  for (int i = 0; i < num_blocks_8; ++i) {
    ((uint64_t*)staging_buf.data())[i] = 0;
  }

  // read destination buffer from local memory
  std::cout << "read destination buffer from local memory" << std::endl;
  auto t2 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_copy_from_dev(device, staging_buf.data(), dev_addr, 64 * num_blocks));
  auto t3 = std::chrono::high_resolution_clock::now();

  // verify result
  std::cout << "verify result" << std::endl;
  for (int i = 0; i < num_blocks_8; ++i) {
    auto curr = ((uint64_t*)staging_buf.data())[i];
    auto ref = shuffle(i, value);
    if (curr != ref) {
      std::cout << "error at 0x" << std::hex << (dev_addr + 8 * i)
                << ": actual 0x" << curr << ", expected 0x" << ref << std::endl;
      ++errors;
    }
  } 
  
  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  auto time_end = std::chrono::high_resolution_clock::now();

  double elapsed;
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();  
  printf("upload time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();  
  printf("download time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();  
  printf("Total elapsed time: %lg ms\n", elapsed);

  return 0;
}

int run_kernel_test(const kernel_arg_t& kernel_arg, 
                    uint32_t buf_size, 
                    uint32_t num_points) {
  int errors = 0; 

  auto time_start = std::chrono::high_resolution_clock::now();
  
  // update source buffer
  {
    std::cout << "upload source buffer" << std::endl;
    auto buf_ptr = (int32_t*)staging_buf.data();
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = i;
    }
  }  
  auto t0 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_copy_to_dev(device, kernel_arg.src_addr, staging_buf.data(), buf_size));
  auto t1 = std::chrono::high_resolution_clock::now();

  // clear destination buffer
  {
    std::cout << "clear destination buffer" << std::endl;
    auto buf_ptr = (int32_t*)staging_buf.data();
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = 0xdeadbeef;
    }
    RT_CHECK(vx_copy_to_dev(device, kernel_arg.dst_addr, staging_buf.data(), buf_size));
  }

  // start device
  std::cout << "start execution" << std::endl;
  auto t2 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_start(device));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  auto t3 = std::chrono::high_resolution_clock::now();

  // read destination buffer from local memory
  std::cout << "read destination buffer from local memory" << std::endl;
  auto t4 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_copy_from_dev(device, staging_buf.data(), kernel_arg.dst_addr, buf_size));
  auto t5 = std::chrono::high_resolution_clock::now();

  
  // verify result
  std::cout << "verify result" << std::endl;
  for (uint32_t i = 0; i < num_points; ++i) {
    int32_t curr = ((int32_t*)staging_buf.data())[i];
    int32_t ref = i;
    if (curr != ref) {
      std::cout << "error at result #" << std::dec << i
                << std::hex << ": actual 0x" << curr << ", expected 0x" << ref << std::endl;
      ++errors;
    }
  } 
  
  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  auto time_end = std::chrono::high_resolution_clock::now();

  double elapsed;
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();  
  printf("upload time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();  
  printf("execute time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();  
  printf("download time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();  
  printf("Total elapsed time: %lg ms\n", elapsed);

  return 0;
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

  uint64_t num_cores;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));

  uint32_t num_points = count * num_cores;
  uint32_t num_blocks = (num_points * sizeof(int32_t) + 63) / 64;
  uint32_t buf_size   = num_blocks * 64;

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.src_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_TYPE_GLOBAL, &kernel_arg.dst_addr));

  kernel_arg.count = num_points;

  std::cout << "dev_src=0x" << std::hex << kernel_arg.src_addr << std::endl;
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::endl;

  // allocate staging buffer  
  std::cout << "allocate staging buffer" << std::endl;
  uint32_t alloc_size = std::max<uint32_t>(buf_size, sizeof(kernel_arg_t));
  staging_buf.resize(alloc_size);

  // run tests  
  if (0 == test || -1 == test) {
    std::cout << "run memcopy test" << std::endl;
    RT_CHECK(run_memcopy_test(kernel_arg.src_addr, 0x0badf00d40ff40ff, num_blocks));
  }

  if (1 == test || -1 == test) {
    // upload program
    std::cout << "upload program" << std::endl;  
    RT_CHECK(vx_upload_kernel_file(device, kernel_file));

    // upload kernel argument
    std::cout << "upload kernel argument" << std::endl;
    memcpy(staging_buf.data(), &kernel_arg, sizeof(kernel_arg_t));
    RT_CHECK(vx_copy_to_dev(device, KERNEL_ARG_DEV_MEM_ADDR, staging_buf.data(), sizeof(kernel_arg_t)));

    std::cout << "run kernel test" << std::endl;
    RT_CHECK(run_kernel_test(kernel_arg, buf_size, num_points));
  }

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "Test PASSED" << std::endl;  
  
  return 0;
}