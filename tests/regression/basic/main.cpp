#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include <chrono>
#include <vector>
#include "common.h"

#define NONCE  0xdeadbeef

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
int test = -1;
uint32_t count = 0;

vx_device_h device = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-t testno][-k: kernel][-n words][-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:t:k:h")) != -1) {
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
      show_usage();
      exit(0);
      break;
    default:
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(src_buffer);
    vx_mem_free(dst_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

inline uint32_t shuffle(int i, uint32_t value) {
  return (value << i) | (value & ((1 << i)-1));;
}

int run_memcopy_test(const kernel_arg_t& kernel_arg) {
  uint32_t num_points = kernel_arg.count;
  uint32_t buf_size = num_points * sizeof(int32_t);

  std::vector<uint32_t> h_src(num_points);
  std::vector<uint32_t> h_dst(num_points);

  // update source buffer
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src[i] = shuffle(i, NONCE);
  }

  auto time_start = std::chrono::high_resolution_clock::now();

  // upload source buffer
  std::cout << "write source buffer to local memory" << std::endl;
  auto t0 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_copy_to_dev(dst_buffer, h_src.data(), 0, buf_size));
  auto t1 = std::chrono::high_resolution_clock::now();

  // download destination buffer
  std::cout << "read destination buffer from local memory" << std::endl;
  auto t2 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_copy_from_dev(h_dst.data(), dst_buffer, 0, buf_size));
  auto t3 = std::chrono::high_resolution_clock::now();

  // verify result
  int errors = 0;
  std::cout << "verify result" << std::endl;
  for (uint32_t i = 0; i < num_points; ++i) {
    auto cur = h_dst[i];
    auto ref = shuffle(i, NONCE);
    if (cur != ref) {
      printf("*** error: [%d] expected=%d, actual=%d\n", i, ref, cur);
      ++errors;
    }
  }

  auto time_end = std::chrono::high_resolution_clock::now();

  double elapsed;
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
  printf("upload time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count();
  printf("download time: %lg ms\n", elapsed);
  elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
  printf("Total elapsed time: %lg ms\n", elapsed);

  return errors;
}

int run_kernel_test(const kernel_arg_t& kernel_arg) {
  uint32_t num_points = kernel_arg.count;
  uint32_t buf_size = num_points * sizeof(int32_t);

  std::vector<uint32_t> h_src(num_points);
  std::vector<uint32_t> h_dst(num_points);

  // update source buffer
  for (uint32_t i = 0; i < num_points; ++i) {
    h_src[i] = shuffle(i, NONCE);
  }

  // upload program
  std::cout << "upload program" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  auto time_start = std::chrono::high_resolution_clock::now();

  // upload source buffer
  auto t0 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_copy_to_dev(src_buffer, h_src.data(), 0, buf_size));
  auto t1 = std::chrono::high_resolution_clock::now();

  // start device
  std::cout << "start execution" << std::endl;
  auto t2 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  auto t3 = std::chrono::high_resolution_clock::now();

  // download destination buffer
  std::cout << "read destination buffer from local memory" << std::endl;
  auto t4 = std::chrono::high_resolution_clock::now();
  RT_CHECK(vx_copy_from_dev(h_dst.data(), dst_buffer, 0, buf_size));
  auto t5 = std::chrono::high_resolution_clock::now();

  // verify result
  int errors = 0;
  std::cout << "verify result" << std::endl;
  for (uint32_t i = 0; i < num_points; ++i) {
    auto cur = h_dst[i];
    auto ref = shuffle(i, NONCE);
    if (cur != ref) {
      printf("*** error: [%d] expected=%d, actual=%d\n", i, ref, cur);
      ++errors;
    }
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

  return errors;
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
  uint32_t buf_size = num_points * sizeof(int32_t);

  std::cout << "number of points: " << num_points << std::endl;
  std::cout << "buffer size: " << buf_size << " bytes" << std::endl;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_mem_address(src_buffer, &kernel_arg.src_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_mem_address(dst_buffer, &kernel_arg.dst_addr));

  kernel_arg.count = num_points;

  std::cout << "dev_src=0x" << std::hex << kernel_arg.src_addr << std::endl;
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::endl;

  int errors = 0;

  // run tests
  if (0 == test || -1 == test) {
    std::cout << "run memcopy test" << std::endl;
    errors = run_memcopy_test(kernel_arg);
  }

  if (1 == test || -1 == test) {
    std::cout << "run kernel test" << std::endl;
    errors = run_kernel_test(kernel_arg);
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "Test PASSED" << std::endl;

  return 0;
}