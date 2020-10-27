#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include <chrono>
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
vx_buffer_h buffer = nullptr;

static void show_usage() {
   std::cout << "Vortex Driver Test." << std::endl;
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
  if (buffer) {
    vx_buf_release(buffer);
  }
  if (device) {
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
    ((uint64_t*)vx_host_ptr(buffer))[i] = shuffle(i, value);
  }

  /*for (int i = 0; i < num_blocks; ++i) {
    std::cout << "data[" << i << "]=0x";
    for (int j = 7; j >= 0; --j) {
      std::cout << std::hex << ((uint64_t*)vx_host_ptr(buffer))[i * 8 +j];
    }
    std::cout << std::endl;
  }*/
  
  // write buffer to local memory
  std::cout << "write buffer to local memory" << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_copy_to_dev(buffer, dev_addr, 64 * num_blocks, 0));
  auto t1 = std::chrono::high_resolution_clock::now();

  // clear destination buffer
  for (int i = 0; i < num_blocks_8; ++i) {
    ((uint64_t*)vx_host_ptr(buffer))[i] = 0;
  }

  // read buffer from local memory
  std::cout << "read buffer from local memory" << std::endl;
  auto t2 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_copy_from_dev(buffer, dev_addr, 64 * num_blocks, 0));
  auto t3 = std::chrono::high_resolution_clock::now();

  // verify result
  std::cout << "verify result" << std::endl;
  for (int i = 0; i < num_blocks_8; ++i) {
    auto curr = ((uint64_t*)vx_host_ptr(buffer))[i];
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
    auto buf_ptr = (int32_t*)vx_host_ptr(buffer);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = i;
    }
  }
  std::cout << "upload source buffer" << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_copy_to_dev(buffer, kernel_arg.src_ptr, buf_size, 0));
  auto t1 = std::chrono::high_resolution_clock::now();

  // clear destination buffer
  {
    auto buf_ptr = (int32_t*)vx_host_ptr(buffer);
    for (uint32_t i = 0; i < num_points; ++i) {
      buf_ptr[i] = 0xdeadbeef;
    }
  }  
  std::cout << "clear destination buffer" << std::endl;
  RT_CHECK(vx_copy_to_dev(buffer, kernel_arg.dst_ptr, buf_size, 0));

  // start device
  std::cout << "start execution" << std::endl;
  auto t2 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_start(device));
  RT_CHECK(vx_ready_wait(device, -1));
  auto t3 = std::chrono::high_resolution_clock::now();

  // flush the caches
  std::cout << "flush the caches" << std::endl;
  auto t4 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_flush_caches(device, kernel_arg.dst_ptr, buf_size));
  auto t5 = std::chrono::high_resolution_clock::now();

  // read buffer from local memory
  std::cout << "read buffer from local memory" << std::endl;
  auto t6 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_copy_from_dev(buffer, kernel_arg.dst_ptr, buf_size, 0));
  auto t7 = std::chrono::high_resolution_clock::now();

  
  // verify result
  std::cout << "verify result" << std::endl;
  for (uint32_t i = 0; i < num_points; ++i) {
    int32_t curr = ((int32_t*)vx_host_ptr(buffer))[i];
    int32_t ref = i;
    if (curr != ref) {
      std::cout << "error at result #" << i
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
  printf("execute time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t5 - t4).count();  
  printf("flush time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t7 - t6).count();  
  printf("download time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();  
  printf("Total elapsed time: %lg ms\n", elapsed);

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
  
  unsigned max_cores;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_MAX_CORES, &max_cores));
  uint32_t num_points = 1 * count;
  uint32_t num_blocks = (num_points * sizeof(uint32_t) + 63) / 64;
  uint32_t buf_size = num_blocks * 64;

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // allocate device memory
  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.src_ptr = value;
  RT_CHECK(vx_alloc_dev_mem(device, buf_size, &value));
  kernel_arg.dst_ptr = value;

  kernel_arg.count = count;

  std::cout << "dev_src=" << std::hex << kernel_arg.src_ptr << std::endl;
  std::cout << "dev_dst=" << std::hex << kernel_arg.dst_ptr << std::endl;

  // allocate shared memory  
  std::cout << "allocate shared memory" << std::endl;
  uint32_t alloc_size = std::max<uint32_t>(buf_size, sizeof(kernel_arg_t));
  RT_CHECK(vx_alloc_shared_mem(device, alloc_size, &buffer));

  // run tests  
  if (0 == test || -1 == test) {
    std::cout << "run memcopy test" << std::endl;
    RT_CHECK(run_memcopy_test(kernel_arg.src_ptr, 0x0badf00d40ff40ff, num_blocks));
  }

  if (1 == test || -1 == test) {
    // upload program
    std::cout << "upload program" << std::endl;  
    RT_CHECK(vx_upload_kernel_file(device, kernel_file));

    // upload kernel argument
    std::cout << "upload kernel argument" << std::endl;
    {
      auto buf_ptr = (void*)vx_host_ptr(buffer);
      memcpy(buf_ptr, &kernel_arg, sizeof(kernel_arg_t));
      RT_CHECK(vx_copy_to_dev(buffer, KERNEL_ARG_DEV_MEM_ADDR, sizeof(kernel_arg_t), 0));
    }

    std::cout << "run kernel test" << std::endl;
    RT_CHECK(run_kernel_test(kernel_arg, buf_size, num_points));
  }

  // cleanup
  std::cout << "cleanup" << std::endl;  
  cleanup();

  std::cout << "Test PASSED" << std::endl;  
  
  return 0;
}