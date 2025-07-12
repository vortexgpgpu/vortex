#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vector>
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

template <typename Type>
class Comparator {};

template <>
class Comparator<int> {
public:
  static const char* type_str() {
    return "integer";
  }
  static int generate() {
    return rand();
  }
  static bool compare(int a, int b, int index, int errors) {
    if (a != b) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%d, actual=%d\n", index, b, a);
      }
      return false;
    }
    return true;
  }
};

const char* kernel_file = "kernel.vxbin";
uint32_t grd_x = 4;
uint32_t grd_y = 4;
uint32_t grd_z = 1;
uint32_t blk_x = 1;
uint32_t blk_y = 1;
uint32_t blk_z = 1;

vx_device_h device = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
   std::cout << "Vortex Test." << std::endl;
   std::cout << "Usage: [-k: kernel] [-x grid.x] [-y grid.y] [-z grid.z] [-a block.x] [-b block.y] [-c block.z] [-h: help]" << std::endl;
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "k:x:y:z:a:b:c:h")) != -1) {
    switch (c) {
    case 'a':
      blk_x = atoi(optarg);
      break;
    case 'b':
      blk_y = atoi(optarg);
      break;
    case 'c':
      blk_z = atoi(optarg);
      break;
    case 'x':
      grd_x = atoi(optarg);
      break;
    case 'y':
      grd_y = atoi(optarg);
      break;
    case 'z':
      grd_z = atoi(optarg);
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

int main(int argc, char *argv[]) {
  // parse command arguments
  parse_args(argc, argv);

  std::srand(50);

  // open device connection
  std::cout << "open device connection" << std::endl;
  RT_CHECK(vx_dev_open(&device));

  uint32_t cta_size = blk_x * blk_y * blk_z;
  uint32_t cta_count = grd_x * grd_y * grd_z;
  uint32_t total_threads  = cta_count * cta_size;
  uint32_t src_buf_size = cta_size * sizeof(int);
  uint32_t dst_buf_size = total_threads * sizeof(int);

  std::cout << "CTA size: " << cta_size << std::endl;
  std::cout << "number of CTAs: " << cta_count << std::endl;
  std::cout << "number of threads: " << total_threads << std::endl;
  std::cout << "source buffer size: " << src_buf_size << " bytes" << std::endl;
  std::cout << "destination buffer size: " << dst_buf_size << " bytes" << std::endl;

  kernel_arg.block_dim[0] = blk_x;
  kernel_arg.block_dim[1] = blk_y;
  kernel_arg.block_dim[2] = blk_z;
  kernel_arg.grid_dim[0]  = grd_x;
  kernel_arg.grid_dim[1]  = grd_y;
  kernel_arg.grid_dim[2]  = grd_z;

  // allocate device memory
  std::cout << "allocate device memory" << std::endl;
  RT_CHECK(vx_mem_alloc(device, src_buf_size, VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_mem_address(src_buffer, &kernel_arg.src_addr));
  RT_CHECK(vx_mem_alloc(device, dst_buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_mem_address(dst_buffer, &kernel_arg.dst_addr));

  std::cout << "src_dst=0x" << std::hex << kernel_arg.src_addr << std::endl;
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << std::endl;

  // allocate host buffers
  std::cout << "allocate host buffers" << std::endl;
  std::vector<int> h_src(cta_size);
  std::vector<int> h_dst(total_threads);

  for (uint32_t i = 0; i < cta_size; ++i) {
    h_src[i] = Comparator<int>::generate();
  }

  // upload source buffer0
  std::cout << "upload source buffer0" << std::endl;
  RT_CHECK(vx_copy_to_dev(src_buffer, h_src.data(), 0, src_buf_size));

  // Upload kernel binary
  std::cout << "Upload kernel binary" << std::endl;
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  // upload kernel argument
  std::cout << "upload kernel argument" << std::endl;
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  // start device
  std::cout << "start device" << std::endl;
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  // wait for completion
  std::cout << "wait for completion" << std::endl;
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // download destination buffer
  std::cout << "download destination buffer" << std::endl;
  RT_CHECK(vx_copy_from_dev(h_dst.data(), dst_buffer, 0, dst_buf_size));

  // verify result
  std::cout << "verify result" << std::endl;
  int errors = 0;
  for (uint32_t i = 0; i < total_threads; ++i) {
    auto ref = i + h_src[i % cta_size];
    auto cur = h_dst[i];
    if (!Comparator<int>::compare(cur, ref, i, errors)) {
      ++errors;
    }
  }

  // cleanup
  std::cout << "cleanup" << std::endl;
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!" << std::endl;
    std::cout << "FAILED!" << std::endl;
    return 1;
  }

  std::cout << "PASSED!" << std::endl;

  return 0;
}