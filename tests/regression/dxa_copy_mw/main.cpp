// Host driver for dxa_copy_mw — intra-core multicast.
//
// num_recv = NUM_WARPS single-warp CTAs co-resident on one core, each
// receiving the same tile via DXA multicast.

#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <VX_types.h>
#include <vortex.h>
#include <dxa.h>

#include "common.h"

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret) break;                                       \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

static const char* kernel_file = "kernel.vxbin";
static uint32_t    tile_rows   = 16;
static uint32_t    tile_cols   = 4;
static uint32_t    src_rows    = 16;
static uint32_t    src_cols    = 16;

static vx_device_h device     = nullptr;
static vx_buffer_h src_buffer = nullptr;
static vx_buffer_h dst_buffer = nullptr;
static vx_buffer_h krnl_buffer = nullptr;
static vx_buffer_h args_buffer = nullptr;
static kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-r src_rows] [-c src_cols] "
               "[-R tile_rows] [-C tile_cols] [-h]\n";
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "r:c:R:C:k:h")) != -1) {
    switch (c) {
      case 'r': src_rows  = atoi(optarg); break;
      case 'c': src_cols  = atoi(optarg); break;
      case 'R': tile_rows = atoi(optarg); break;
      case 'C': tile_cols = atoi(optarg); break;
      case 'k': kernel_file = optarg; break;
      case 'h': show_usage(); exit(0);
      default:  show_usage(); exit(-1);
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

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  RT_CHECK(vx_dev_open(&device));

  // Single-warp CTAs: block = (tile_cols, 1) so each CTA uses exactly one
  // warp. num_recv = NUM_WARPS so all receivers fit co-resident on one core.
  uint64_t num_warps = 0, num_threads = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads));

  if (tile_cols != (uint32_t)num_threads) {
    std::cout << "Error: tile_cols (" << tile_cols
              << ") must equal NUM_THREADS (" << num_threads
              << ") for single-warp CTAs\n";
    cleanup();
    return -1;
  }

  const uint32_t num_recv   = (uint32_t)num_warps;
  const uint32_t tile_elems = tile_rows * tile_cols;
  const uint32_t local_mem  = tile_elems * sizeof(TYPE);
  const uint32_t src_elems  = src_rows * src_cols;
  const uint32_t src_bytes  = src_elems * sizeof(TYPE);
  const uint32_t dst_bytes  = num_recv * tile_elems * sizeof(TYPE);

  std::cout << "dxa_copy_mw (intra-core multicast)\n";
  std::cout << "  source: " << src_rows << " x " << src_cols
            << ", tile: " << tile_rows << " x " << tile_cols << "\n";
  std::cout << "  block=" << tile_cols << "x1 (single warp), grid="
            << num_recv << "x1, num_recv=" << num_recv << "\n";
  std::cout << "  local_mem=" << local_mem << " bytes\n";

  uint32_t max_localmem = 0;
  RT_CHECK(vx_check_occupancy(device, tile_cols, &max_localmem));
  if (local_mem > max_localmem) {
    std::cout << "Error: tile too large for local memory ("
              << local_mem << " > " << max_localmem << ")\n";
    cleanup();
    return -1;
  }

  RT_CHECK(vx_mem_alloc(device, src_bytes, VX_MEM_READ,  &src_buffer));
  RT_CHECK(vx_mem_address(src_buffer, &kernel_arg.src_addr));
  RT_CHECK(vx_mem_alloc(device, dst_bytes, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_mem_address(dst_buffer, &kernel_arg.dst_addr));

  // Distinctive source pattern (linear index + 1).
  std::vector<TYPE> h_src(src_elems);
  for (uint32_t i = 0; i < src_elems; ++i)
    h_src[i] = static_cast<TYPE>(i + 1);
  RT_CHECK(vx_copy_to_dev(src_buffer, h_src.data(), 0, src_bytes));

  // Program DXA descriptor for the tile.
  constexpr uint32_t kDescSrc = 0;
  RT_CHECK(vx_dxa_program_desc_2d(device, kDescSrc, kernel_arg.src_addr,
    /*size0=*/src_cols, /*size1=*/src_rows,
    /*stride0_bytes=*/src_cols * sizeof(TYPE),
    /*tile0=*/tile_cols, /*tile1=*/tile_rows,
    /*elem_bytes=*/sizeof(TYPE)));
  // Multicast stride = local_mem (per-CTA SMEM size). Dispatcher allocates
  // sequential LMEM regions so receiver k's region starts at k*local_mem.
  RT_CHECK(vx_dxa_program_desc_multicast(device, kDescSrc, local_mem));

  kernel_arg.tile_rows      = tile_rows;
  kernel_arg.tile_cols      = tile_cols;
  kernel_arg.src_row_stride = src_cols;
  kernel_arg.num_recv       = num_recv;

  uint32_t grid_dim[2]  = { num_recv, 1 };
  uint32_t block_dim[2] = { tile_cols, 1 };

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start\n";
  RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid_dim, block_dim, local_mem));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // Verify each receiver got the same tile.
  std::vector<TYPE> h_dst(num_recv * tile_elems, 0);
  RT_CHECK(vx_copy_from_dev(h_dst.data(), dst_buffer, 0, dst_bytes));

  int errors = 0;
  for (uint32_t rcv = 0; rcv < num_recv; ++rcv) {
    for (uint32_t r = 0; r < tile_rows; ++r) {
      for (uint32_t c = 0; c < tile_cols; ++c) {
        TYPE expected = h_src[r * src_cols + c];
        TYPE actual   = h_dst[rcv * tile_elems + r * tile_cols + c];
        if (expected != actual) {
          if (errors < 20)
            printf("*** error: recv=%u [r=%u,c=%u] expected=%f actual=%f\n",
                   rcv, r, c, (float)expected, (float)actual);
          ++errors;
        }
      }
    }
  }

  cleanup();
  if (errors) {
    std::cout << "FAILED (" << errors << " errors)\n";
    return errors;
  }
  std::cout << "PASSED\n";
  return 0;
}
