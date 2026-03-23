#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <VX_types.h>
#include <vortex.h>

#ifdef USE_DXA
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
uint32_t nrows = 32;
uint32_t ncols = 32;
uint32_t tile_rows = 8;
uint32_t tile_cols = 8;

vx_device_h device = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_buffer_h krnl_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-r nrows] [-c ncols] [-R tile_rows] [-C tile_cols] [-h]\n";
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "r:c:R:C:k:h")) != -1) {
    switch (c) {
    case 'r': nrows = atoi(optarg); break;
    case 'c': ncols = atoi(optarg); break;
    case 'R': tile_rows = atoi(optarg); break;
    case 'C': tile_cols = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
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
    vx_mem_free(krnl_buffer);
    vx_mem_free(args_buffer);
    vx_dev_close(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  if ((nrows % tile_rows) != 0) {
    std::cout << "Error: nrows must be divisible by tile_rows\n";
    return -1;
  }
  if ((ncols % tile_cols) != 0) {
    std::cout << "Error: ncols must be divisible by tile_cols\n";
    return -1;
  }

  std::srand(42);

#ifdef USE_DXA
  std::cout << "mode: DXA\n";
#else
  std::cout << "mode: LSU\n";
#endif
  std::cout << "array: " << nrows << " x " << ncols << "\n";
  std::cout << "tile: " << tile_rows << " x " << tile_cols << "\n";

  const uint32_t total_elems = nrows * ncols;
  const uint32_t buf_size = total_elems * sizeof(TYPE);
  const uint32_t group_size = tile_rows * tile_cols;
  const uint32_t grid_x = ncols / tile_cols;
  const uint32_t grid_y = nrows / tile_rows;

  std::cout << "grid: " << grid_x << " x " << grid_y << "\n";

  RT_CHECK(vx_dev_open(&device));

#ifdef USE_DXA
  // Check DXA ISA support.
  uint64_t isa_flags = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
#ifdef ISA_EXT_DXA
  const uint64_t dxa_isa_bit = (1ull << (32 + ISA_EXT_DXA));
  if ((isa_flags & dxa_isa_bit) == 0) {
    std::cerr << "Error: DXA ISA extension is disabled.\n";
    cleanup();
    return -1;
  }
#endif
#endif

  uint32_t max_localmem = 0;
  RT_CHECK(vx_check_occupancy(device, group_size, &max_localmem));
  const uint32_t local_mem = tile_rows * tile_cols * sizeof(TYPE);
  if (local_mem > max_localmem) {
    std::cout << "Error: tile too large for local memory\n";
    cleanup();
    return -1;
  }

  uint32_t grid_dim[2]  = {grid_x, grid_y};
  uint32_t block_dim[2] = {tile_cols, tile_rows};
  kernel_arg.tile_rows    = tile_rows;
  kernel_arg.tile_cols    = tile_cols;
  kernel_arg.ncols        = ncols;
  kernel_arg.nrows        = nrows;

  // Allocate and fill source array.
  RT_CHECK(vx_mem_alloc(device, buf_size, VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_mem_address(src_buffer, &kernel_arg.src_addr));

  std::vector<TYPE> h_src(total_elems);
  for (uint32_t i = 0; i < total_elems; ++i) {
    h_src[i] = static_cast<TYPE>(i + 1);
  }
  RT_CHECK(vx_copy_to_dev(src_buffer, h_src.data(), 0, buf_size));

#ifdef USE_DXA
  // Program DXA descriptor for 2D source tile.
  constexpr uint32_t kDescSrc = 0;
  RT_CHECK(vx_dxa_program_desc_2d(device, kDescSrc, kernel_arg.src_addr,
    /*size0=*/ncols, /*size1=*/nrows,
    /*stride0_bytes=*/ncols * sizeof(TYPE),
    /*tile0=*/tile_cols, /*tile1=*/tile_rows,
    /*elem_bytes=*/sizeof(TYPE)));
#endif

  RT_CHECK(vx_upload_kernel_file(device, kernel_file, &krnl_buffer));
  RT_CHECK(vx_upload_bytes(device, &kernel_arg, sizeof(kernel_arg_t), &args_buffer));

  std::cout << "start\n";
  RT_CHECK(vx_start_g(device, krnl_buffer, args_buffer, 2, grid_dim, block_dim, local_mem));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));

  // Kernel execution time is reported by the runtime (cycles, instrs, IPC).
  std::cout << "PASSED\n";

  cleanup();
  return 0;
}
