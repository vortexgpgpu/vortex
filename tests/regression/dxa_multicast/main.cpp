#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <VX_types.h>
#include <vortex.h>

#ifdef EXT_DXA_ENABLE
#include <dxa.h>
#endif

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

const char* kernel_file = "kernel.vxbin";
uint32_t nrows = 16;
uint32_t ncols = 16;
uint32_t tile_rows = 16;  // tall tile: all rows
uint32_t tile_cols = 4;   // narrow: 4 cols
uint32_t active_ctas = 0; // 0 = all CTAs active

vx_device_h device = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h dst_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-r nrows] [-c ncols] [-R tile_rows] [-C tile_cols] [-a active_ctas] [-h]\n";
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "r:c:R:C:a:k:h")) != -1) {
    switch (c) {
    case 'r': nrows = atoi(optarg); break;
    case 'c': ncols = atoi(optarg); break;
    case 'R': tile_rows = atoi(optarg); break;
    case 'C': tile_cols = atoi(optarg); break;
    case 'a': active_ctas = atoi(optarg); break;
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

  // CTA thread block: tile_cols × tile_cols (square, small)
  // DXA fetches a larger tile_rows × tile_cols region
  const uint32_t block_x = tile_cols;
  const uint32_t block_y = tile_cols;  // square block
  const uint32_t group_size = block_x * block_y;

  // All CTAs fetch the same tile → grid can be anything.
  // Use ncols/tile_cols CTAs so multicast covers them all.
  const uint32_t grid_x = ncols / tile_cols;
  const uint32_t grid_y = 1;
  const uint32_t total_ctas = grid_x * grid_y;

  const uint32_t src_elems = nrows * ncols;
  const uint32_t src_bytes = src_elems * sizeof(TYPE);
  const uint32_t tile_elems = tile_rows * tile_cols;
  const uint32_t local_mem = tile_elems * sizeof(TYPE);

  // Dst buffer: each CTA writes its SMEM to a different row strip
  const uint32_t dst_rows = total_ctas * tile_rows;
  const uint32_t dst_elems = dst_rows * ncols;
  const uint32_t dst_bytes = dst_elems * sizeof(TYPE);

#ifdef EXT_DXA_ENABLE
  std::cout << "mode: DXA MULTICAST\n";
#else
  std::cout << "mode: LSU\n";
#endif
  std::cout << "matrix: " << nrows << " x " << ncols << "\n";
  std::cout << "tile: " << tile_rows << " x " << tile_cols << "\n";
  std::cout << "block: " << block_x << " x " << block_y << "\n";
  std::cout << "grid: " << grid_x << " x " << grid_y << " (" << total_ctas << " CTAs)\n";
  std::cout << "local_mem: " << local_mem << " bytes\n";

  RT_CHECK(vx_dev_open(&device));

  uint32_t max_localmem = 0;
  RT_CHECK(vx_check_occupancy(device, group_size, &max_localmem));
  if (local_mem > max_localmem) {
    std::cout << "Error: tile too large for local memory (" << local_mem << " > " << max_localmem << ")\n";
    cleanup();
    return -1;
  }

  // Get hardware config for num_ctas calculation
  uint64_t num_warps_val = 0, num_threads_val = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_WARPS, &num_warps_val));
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &num_threads_val));
  uint32_t warps_per_cta = group_size / (uint32_t)num_threads_val;
  if (warps_per_cta == 0) warps_per_cta = 1;
  uint32_t ctas_per_core = (uint32_t)num_warps_val / warps_per_cta;
  if (ctas_per_core > total_ctas) ctas_per_core = total_ctas;
  std::cout << "ctas_per_core: " << ctas_per_core << "\n";

  uint32_t grid_dim[2]  = {grid_x, grid_y};
  uint32_t block_dim[2] = {block_x, block_y};
  kernel_arg.tile_rows    = tile_rows;
  kernel_arg.tile_cols    = tile_cols;
  kernel_arg.ncols        = ncols;
  kernel_arg.nrows        = nrows;
  kernel_arg.num_ctas     = ctas_per_core;
  kernel_arg.active_ctas  = active_ctas;

  uint32_t verify_ctas = (active_ctas == 0) ? total_ctas : active_ctas;
  std::cout << "active_ctas: " << verify_ctas << " / " << total_ctas << "\n";

  // Allocate source
  RT_CHECK(vx_mem_alloc(device, src_bytes, VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_mem_address(src_buffer, &kernel_arg.src_addr));

  std::vector<TYPE> h_src(src_elems);
  for (uint32_t i = 0; i < src_elems; ++i) {
    h_src[i] = static_cast<TYPE>(i + 1);
  }
  RT_CHECK(vx_copy_to_dev(src_buffer, h_src.data(), 0, src_bytes));

  // Allocate destination
  RT_CHECK(vx_mem_alloc(device, dst_bytes, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_mem_address(dst_buffer, &kernel_arg.dst_addr));

#ifdef EXT_DXA_ENABLE
  // Program DXA descriptor for the shared tile.
  constexpr uint32_t kDescSrc = 0;
  RT_CHECK(vx_dxa_program_desc_2d(device, kDescSrc, kernel_arg.src_addr,
    ncols, nrows, ncols * sizeof(TYPE),
    tile_cols, tile_rows, sizeof(TYPE)));
  // Program multicast SMEM stride: byte distance between consecutive CTAs' SMEM bases.
  RT_CHECK(vx_dxa_program_desc_multicast(device, kDescSrc, local_mem));
  std::cout << "smem_stride: " << local_mem << " bytes\n";
#endif

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start\n";
  RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid_dim, block_dim, local_mem));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  int errors = 0;

#ifdef VERIFY_WRITEBACK
  // Verify: each CTA should have the same tile data.
  // CTA N writes to dst rows [N*tile_rows .. (N+1)*tile_rows-1]
  std::vector<TYPE> h_dst(dst_elems, 0);
  RT_CHECK(vx_copy_from_dev(h_dst.data(), dst_buffer, 0, dst_bytes));

  // Verify only active CTAs (inactive CTAs don't write dst)
  for (uint32_t cta = 0; cta < verify_ctas; ++cta) {
    for (uint32_t r = 0; r < tile_rows; ++r) {
      for (uint32_t c = 0; c < tile_cols; ++c) {
        uint32_t src_idx = r * ncols + c;
        uint32_t dst_idx = (cta * tile_rows + r) * ncols + c;
        if (h_dst[dst_idx] != h_src[src_idx]) {
          if (errors < 20) {
            printf("*** error: CTA%d [r=%d,c=%d] expected=%f, actual=%f\n",
                   cta, r, c, (float)h_src[src_idx], (float)h_dst[dst_idx]);
          }
          ++errors;
        }
      }
    }
  }
#endif

  if (errors != 0) {
    std::cout << "Found " << errors << " errors\n";
    std::cout << "FAILED\n";
    cleanup();
    return errors;
  }

  std::cout << "PASSED\n";
  cleanup();
  return 0;
}
