#include <iostream>
#include <unistd.h>
#include <string.h>
#include <cstdlib>
#include <vector>
#include <cstdint>
#include <cmath>
#include <vortex.h>
#include "common.h"

#define RT_CHECK(_expr)                                                   \
  do {                                                                    \
    int _ret = (_expr);                                                   \
    if (0 == _ret) break;                                                 \
    printf("Error: '%s' returned %d!\n", #_expr, _ret);                  \
    cleanup();                                                            \
    exit(-1);                                                             \
  } while (false)

static const char* kernel_file = "kernel.vxbin";

static vx_device_h device      = nullptr;
static vx_buffer_h src_buffer  = nullptr;
static vx_buffer_h dst_lb_buf  = nullptr;
static vx_buffer_h dst_lh_buf  = nullptr;
static vx_buffer_h krnl_buffer = nullptr;
static vx_buffer_h args_buffer = nullptr;
static kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Vortex packld test.\n"
            << "Usage: [-k kernel] [-h help]\n";
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'h': show_usage(); exit(0);
    default:  show_usage(); exit(-1);
    }
  }
}

static void cleanup() {
  if (device) {
    vx_mem_free(src_buffer);
    vx_mem_free(dst_lb_buf);
    vx_mem_free(dst_lh_buf);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  std::cout << "open device connection\n";
  RT_CHECK(vx_dev_open(&device));

  uint64_t num_cores, num_warps, num_threads;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES,   &num_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS,   &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));

  uint32_t num_tasks = (uint32_t)(num_cores * num_warps * num_threads);

  // src layout: num_tasks * NUM_POINTS * 4 bytes
  uint32_t src_bytes  = num_tasks * NUM_POINTS * 4;
  uint32_t dst_bytes  = num_tasks * NUM_POINTS * sizeof(float);

  std::cout << "num_tasks=" << num_tasks
            << "  src=" << src_bytes << "B"
            << "  dst_lb=" << dst_bytes << "B"
            << "  dst_lh=" << dst_bytes << "B\n";

  kernel_arg.num_tasks  = num_tasks;

  // allocate device memory
  RT_CHECK(vx_mem_alloc(device, src_bytes,  VX_MEM_READ,  &src_buffer));
  RT_CHECK(vx_mem_alloc(device, dst_bytes,  VX_MEM_WRITE, &dst_lb_buf));
  RT_CHECK(vx_mem_alloc(device, dst_bytes,  VX_MEM_WRITE, &dst_lh_buf));
  RT_CHECK(vx_mem_address(src_buffer,  &kernel_arg.src_addr));
  RT_CHECK(vx_mem_address(dst_lb_buf,  &kernel_arg.dst_lb_addr));
  RT_CHECK(vx_mem_address(dst_lh_buf,  &kernel_arg.dst_lh_addr));

  // generate random byte source data
  std::vector<uint8_t> h_src(src_bytes);
  for (auto& b : h_src) b = (uint8_t)(std::rand() & 0xFF);

  RT_CHECK(vx_copy_to_dev(src_buffer, h_src.data(), 0, src_bytes));

  // upload kernel
  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device\n";
  RT_CHECK(vx_start(device, krnl_buffer, args_buffer));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // download results
  std::vector<float> h_dst_lb(num_tasks * NUM_POINTS);
  std::vector<float> h_dst_lh(num_tasks * NUM_POINTS);
  RT_CHECK(vx_copy_from_dev(h_dst_lb.data(), dst_lb_buf, 0, dst_bytes));
  RT_CHECK(vx_copy_from_dev(h_dst_lh.data(), dst_lh_buf, 0, dst_bytes));

  // verify
  int errors = 0;
  for (uint32_t t = 0; t < num_tasks; ++t) {
    for (uint32_t p = 0; p < NUM_POINTS; ++p) {
      uint32_t base_off = (t * NUM_POINTS + p) * 4;

      // PACKLB: pack 4 bytes with stride 1
      uint32_t ref_lb = (uint32_t)h_src[base_off + 0]
                      | ((uint32_t)h_src[base_off + 1] << 8)
                      | ((uint32_t)h_src[base_off + 2] << 16)
                      | ((uint32_t)h_src[base_off + 3] << 24);
      uint32_t got_lb;
      memcpy(&got_lb, &h_dst_lb[t * NUM_POINTS + p], 4);
      if (got_lb != ref_lb) {
        if (errors < 8)
          printf("PACKLB error t=%u p=%u: expected=0x%08x got=0x%08x\n",
                 t, p, ref_lb, got_lb);
        ++errors;
      }

      // PACKLH: pack 2 halfwords with stride 2
      uint16_t h0, h1;
      memcpy(&h0, &h_src[base_off + 0], 2);
      memcpy(&h1, &h_src[base_off + 2], 2);
      uint32_t ref_lh = (uint32_t)h0 | ((uint32_t)h1 << 16);
      uint32_t got_lh;
      memcpy(&got_lh, &h_dst_lh[t * NUM_POINTS + p], 4);
      if (got_lh != ref_lh) {
        if (errors < 8)
          printf("PACKLH error t=%u p=%u: expected=0x%08x got=0x%08x\n",
                 t, p, ref_lh, got_lh);
        ++errors;
      }
    }
  }

  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors!\nFAILED!\n";
    return 1;
  }
  std::cout << "PASSED!\n";
  return 0;
}
