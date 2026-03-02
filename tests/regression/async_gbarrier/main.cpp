#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <vector>
#include <vortex.h>
#include "common.h"

#define RT_CHECK(_expr)                                         \
   do {                                                         \
     int _ret = _expr;                                          \
     if (0 == _ret)                                             \
       break;                                                   \
     printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
     cleanup();                                                 \
     exit(-1);                                                  \
   } while (false)

const char* kernel_file = "kernel.vxbin";

vx_device_h device = nullptr;
vx_buffer_h pre_buffer = nullptr;
vx_buffer_h post_buffer = nullptr;
vx_buffer_h status_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex Test." << std::endl;
  std::cout << "Usage: [-k: kernel] [-h: help]" << std::endl;
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:h")) != -1) {
    switch (c) {
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
    vx_mem_free(pre_buffer);
    vx_mem_free(post_buffer);
    vx_mem_free(status_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));

  if (num_cores < 2) {
    std::cout << "Device does not have enough cores to run the test (need at least 2)" << std::endl;
    cleanup();
    return -1;
  }

  kernel_arg.num_cores  = static_cast<uint32_t>(num_cores);
  kernel_arg.num_groups = static_cast<uint32_t>(num_cores * num_warps);
  kernel_arg.group_size = static_cast<uint32_t>(num_threads);

  uint32_t buf_size = kernel_arg.num_cores * sizeof(uint32_t);

  std::cout << "num_cores=" << num_cores
            << ", num_warps=" << num_warps
            << ", num_threads=" << num_threads << std::endl;
  std::cout << "num_groups=" << kernel_arg.num_groups << std::endl;

  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ_WRITE, &pre_buffer));
  RT_CHECK(vx_mem_address(pre_buffer, &kernel_arg.pre_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ_WRITE, &post_buffer));
  RT_CHECK(vx_mem_address(post_buffer, &kernel_arg.post_addr));
  RT_CHECK(vx_mem_alloc(device, sizeof(uint32_t), VX_MEM_READ_WRITE, &status_buffer));
  RT_CHECK(vx_mem_address(status_buffer, &kernel_arg.status_addr));

  std::vector<uint32_t> h_pre(kernel_arg.num_cores, 0);
  std::vector<uint32_t> h_post(kernel_arg.num_cores, 0);
  uint32_t h_status = 0xdeadbeef;

  RT_CHECK(vx_copy_to_dev(pre_buffer, h_pre.data(), 0, buf_size));
  RT_CHECK(vx_copy_to_dev(post_buffer, h_post.data(), 0, buf_size));
  RT_CHECK(vx_copy_to_dev(status_buffer, &h_status, 0, sizeof(uint32_t)));

  std::cout << "upload kernel" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  std::cout << "upload args" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::cout << "download results" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_pre.data(), pre_buffer, 0, buf_size));
  RT_CHECK(vx_copy_from_dev(h_post.data(), post_buffer, 0, buf_size));
  RT_CHECK(vx_copy_from_dev(&h_status, status_buffer, 0, sizeof(uint32_t)));

  uint32_t expected = (kernel_arg.num_cores * (kernel_arg.num_cores + 1)) / 2;
  int errors = 0;

  for (uint32_t i = 0; i < kernel_arg.num_cores; ++i) {
    uint32_t expected_pre = i + 1;
    if (h_pre[i] != expected_pre) {
      std::cout << "pre mismatch at core " << i << ": got " << h_pre[i]
                << ", expected " << expected_pre << std::endl;
      ++errors;
    }
    if (h_post[i] != expected) {
      std::cout << "post mismatch at core " << i << ": got " << h_post[i]
                << ", expected " << expected << std::endl;
      ++errors;
    }
  }

  if (h_status != 0) {
    std::cout << "kernel reported status errors=" << h_status << std::endl;
    ++errors;
  }

  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return errors;
  }

  std::cout << "PASSED!" << std::endl;
  return 0;
}
