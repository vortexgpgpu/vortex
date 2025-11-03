#include <iostream>
#include <unistd.h>
#include <string.h>
#include <vortex.h>
#include <vector>
#include <assert.h>
#include "common.h"

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret)                                              \
      break;                                                    \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);    \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

///////////////////////////////////////////////////////////////////////////////

const char* kernel_file = "kernel.vxbin";
uint32_t count = 0;
uint32_t branch_depth = 8; // NEW: default nested depth

vx_device_h device = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex Test.\n";
  std::cout << "Usage: [-k kernel] [-n words] [-d depth(1..8)] [-h]\n";
}

static void parse_args(int argc, char **argv) {
  int c;
  while ((c = getopt(argc, argv, "n:k:d:h")) != -1) {
    switch (c) {
      case 'n':
        count = atoi(optarg);
        break;
      case 'k':
        kernel_file = optarg;
        break;
      case 'd':
        branch_depth = std::max(1, std::min(8, atoi(optarg)));
        break;
      case 'h':
        show_usage();
        exit(0);
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

void gen_src_data(std::vector<int>& src_data, uint32_t size) {
  src_data.resize(size);
  for (uint32_t i = 0; i < size; ++i) {
    int value = std::rand();
    src_data[i] = value;
  }
}

// Host mirror of kernel's nested chain (must be identical to device logic)
static inline int nested_chain_1to8_host(int v, uint32_t id, uint32_t depth) {
  if (depth == 0) return v;

  if ((id >> 0) & 1) v += 1; else v -= 1;
  if (depth <= 1) return v;

  if ((id >> 1) & 1) v += 2; else v -= 2;
  if (depth <= 2) return v;

  if ((id >> 2) & 1) v += 3; else v -= 3;
  if (depth <= 3) return v;

  if ((id >> 3) & 1) v += 4; else v -= 4;
  if (depth <= 4) return v;

  if ((id >> 4) & 1) v += 5; else v -= 5;
  if (depth <= 5) return v;

  if ((id >> 5) & 1) v += 6; else v -= 6;
  if (depth <= 6) return v;

  if ((id >> 6) & 1) v += 7; else v -= 7;
  if (depth <= 7) return v;

  if ((id >> 7) & 1) v += 8; else v -= 8;
  return v;
}

void gen_ref_data(std::vector<int>& ref_data,
                  const std::vector<int>& src_data,
                  uint32_t size,
                  uint32_t depth) {
  ref_data.resize(size);
  for (int i = 0; i < (int)size; ++i) {
    int value = src_data.at(i);

    uint32_t samples = size;
    while (samples--) {
      if ((i & 0x1) == 0) {
        value += 1;
      }
    }

    // none taken
    if (i >= 0x7fffffff) {
      value = 0;
    } else {
      value += 2;
    }

    // diverge
    if (i > 1) {
      if (i > 2) {
        value += 6;
      } else {
        value += 5;
      }
    } else {
      if (i > 0) {
        value += 4;
      } else {
        value += 3;
      }
    }

    // all taken
    if (i >= 0) {
      value += 7;
    } else {
      value = 0;
    }

    // loop
    for (int j = 0, n = i; j < n; ++j) {
      value += src_data.at(j);
    }

    // switch
    switch (i) {
      case 0: value += 1; break;
      case 1: value -= 1; break;
      case 2: value *= 3; break;
      case 3: value *= 5; break;
      default: break;
    }

    // select
    value += (i >= 0) ? ((i > 5) ? src_data.at(0) : i)
                      : ((i < 5) ? src_data.at(1) : -i);

    // min/max
    value += std::min(src_data.at(i), value);
    value += std::max(src_data.at(i), value);

    // NEW: deep nested branch test (mirrors kernel)
    uint32_t d = (depth > 8) ? 8 : depth;
    value = nested_chain_1to8_host(value, (uint32_t)i, d);

    ref_data[i] = value;
  }
}

int main(int argc, char *argv[]) {
  parse_args(argc, argv);

  if (count == 0) count = 1;

  std::srand(50);

  std::cout << "open device connection\n";
  RT_CHECK(vx_dev_open(&device));

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));

  uint32_t total_threads = num_cores * num_warps * num_threads;
  uint32_t num_points = count * total_threads;
  uint32_t buf_size = num_points * sizeof(int32_t);

  std::cout << "number of points: " << num_points << "\n";
  std::cout << "buffer size: " << buf_size << " bytes\n";
  std::cout << "nested branch depth: " << branch_depth << "\n";

  kernel_arg.num_points   = num_points;
  kernel_arg.branch_depth = branch_depth; // NEW

  std::cout << "allocate device memory\n";
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_mem_address(src_buffer, &kernel_arg.src_addr));
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_mem_address(dst_buffer, &kernel_arg.dst_addr));

  std::cout << "dev_src=0x" << std::hex << kernel_arg.src_addr << "\n";
  std::cout << "dev_dst=0x" << std::hex << kernel_arg.dst_addr << "\n";

  std::cout << "allocate host buffers\n";
  std::vector<int32_t> h_src;
  std::vector<int32_t> h_dst(num_points);
  gen_src_data(h_src, num_points);

  std::cout << "upload source buffer\n";
  RT_CHECK(vx_copy_to_dev(src_buffer, h_src.data(), 0, buf_size));

  std::cout << "Upload kernel binary\n";
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));

  std::cout << "upload kernel argument\n";
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device\n";
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));

  std::cout << "wait for completion\n";
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::cout << "download destination buffer\n";
  RT_CHECK(vx_copy_from_dev(h_dst.data(), dst_buffer, 0, buf_size));

  std::cout << "verify result\n";
  int errors = 0;
  {
    std::vector<int32_t> h_ref;
    gen_ref_data(h_ref, h_src, num_points, branch_depth);

    for (uint32_t i = 0; i < num_points; ++i) {
      int ref = h_ref[i];
      int cur = h_dst[i];
      if (cur != ref) {
        std::cout << "error at result #" << std::dec << i
                  << std::hex << ": actual 0x" << cur << ", expected 0x" << ref << "\n";
        ++errors;
      }
    }
  }

  std::cout << "cleanup\n";
  cleanup();

  if (errors != 0) {
    std::cout << "Found " << std::dec << errors << " errors!\nFAILED!\n";
    return errors;
  }

  std::cout << "PASSED!\n";
  return 0;
}
