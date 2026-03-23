#include <iostream>
#include <unistd.h>
#include <vector>
#include <vortex.h>
#include "common.h"

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret)                                              \
      break;                                                    \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

const char* kernel_file  = "kernel.vxbin";
uint32_t ctas_per_core   = 4;  // CTAs per core; > max_concurrent to force stalls
uint32_t max_concurrent  = 2;  // max CTAs that share local memory simultaneously

vx_device_h device      = nullptr;
vx_buffer_h out_buffer  = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-c ctas_per_core] [-h]\n";
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "k:c:h")) != -1) {
    switch (c) {
    case 'k': kernel_file = optarg; break;
    case 'c': ctas_per_core = atoi(optarg); break;
    case 'h': show_usage(); exit(0); break;
    default:  show_usage(); exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    vx_mem_free(out_buffer);
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char** argv) {
  parse_args(argc, argv);

  RT_CHECK(vx_dev_open(&device));

  uint64_t num_cores, num_warps, num_threads, local_mem_size;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_CORES,       &num_cores));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS,       &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS,     &num_threads));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_LOCAL_MEM_SIZE,  &local_mem_size));

  // Reserve half of local memory per CTA so only max_concurrent CTAs fit per core.
  uint32_t lmem_size  = (uint32_t)(local_mem_size / max_concurrent);
  uint32_t lmem_words = lmem_size / sizeof(uint32_t);
  uint32_t num_ctas   = (uint32_t)num_cores * ctas_per_core;

  std::cout << "num_cores=" << num_cores
            << " num_warps=" << num_warps
            << " num_threads=" << num_threads << "\n";
  std::cout << "local_mem_size=" << local_mem_size
            << " lmem_size=" << lmem_size
            << " lmem_words=" << lmem_words << "\n";
  std::cout << "num_ctas=" << num_ctas
            << " ctas_per_core=" << ctas_per_core
            << " max_concurrent=" << max_concurrent << "\n";

  // 1 warp per CTA: block_dim = {num_threads}
  uint32_t block_dim[1] = {(uint32_t)num_threads};
  uint32_t grid_dim[1]  = {num_ctas};

  kernel_arg.lmem_words = lmem_words;

  RT_CHECK(vx_mem_alloc(device, num_ctas * sizeof(uint32_t), VX_MEM_WRITE, &out_buffer));
  RT_CHECK(vx_mem_address(out_buffer, &kernel_arg.out_addr));

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start device\n";
  RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 1, grid_dim, block_dim, lmem_size));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  std::vector<uint32_t> h_out(num_ctas);
  RT_CHECK(vx_copy_from_dev(h_out.data(), out_buffer, 0, num_ctas * sizeof(uint32_t)));

  int errors = 0;
  for (uint32_t cta_id = 0; cta_id < num_ctas; cta_id++) {
    uint32_t expected = cta_id * lmem_words + lmem_words * (lmem_words - 1) / 2;
    if (h_out[cta_id] != expected) {
      if (errors < 10)
        printf("*** error: cta[%u] expected=0x%08x actual=0x%08x\n", cta_id, expected, h_out[cta_id]);
      ++errors;
    }
  }

  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors\n";
    std::cout << "FAILED\n";
    return 1;
  }

  std::cout << "PASSED\n";
  return 0;
}
