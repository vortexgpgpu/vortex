#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include "common.h"

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret) break;                                       \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);    \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

const char* kernel_file = "kernel.vxbin";

vx_device_h device = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h parent_args_buffer = nullptr;
vx_buffer_h child_args_buffer = nullptr;

static void show_usage() {
  std::cout << "Vortex dyn_launch (KMU device launch, hello-world)." << std::endl;
  std::cout << "Usage: [-k: kernel] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "k:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (krnl_buffer) vx_mem_free(krnl_buffer);
    if (parent_args_buffer) vx_mem_free(parent_args_buffer);
    if (child_args_buffer) vx_mem_free(child_args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  std::cout << "open device" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  std::cout << "upload kernel" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  uint64_t krnl_addr = 0;
  RT_CHECK(vx_mem_address(krnl_buffer, &krnl_addr));

  RT_CHECK(vx_mem_alloc(device, sizeof(kernel_arg_t), VX_MEM_READ_WRITE, &child_args_buffer));
  uint64_t child_addr = 0;
  RT_CHECK(vx_mem_address(child_args_buffer, &child_addr));

  kernel_arg_t parent_arg = {};
  parent_arg.role = DL_ROLE_PARENT;
  parent_arg.child_pc = krnl_addr;
  parent_arg.child_arg_addr = child_addr;

  std::cout << "upload parent args" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &parent_arg, sizeof(parent_arg), &parent_args_buffer));

  std::cout << "start device (expect device prints: Hello World!)" << std::endl;
  uint32_t grid_dim[1]  = { 1 };
  uint32_t block_dim[1] = { 1 };
  RT_CHECK(vx_start_g(device, krnl_buffer, parent_args_buffer, 1, grid_dim, block_dim, 0));

  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  cleanup();
  std::cout << "PASSED!" << std::endl;
  return 0;
}
