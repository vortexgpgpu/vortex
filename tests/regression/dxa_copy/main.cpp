#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <VX_types.h>
#include <vortex.h>

#ifdef VX_CFG_EXT_DXA_ENABLE
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
uint32_t ndim = 2;

// Default sizes/tiles per dimension (overridable via CLI).
uint32_t sizes[DXA_MAX_DIMS] = {32, 32, 1, 1, 1};
uint32_t tiles[DXA_MAX_DIMS] = {4, 4, 1, 1, 1};

vx_device_h device = nullptr;
vx_buffer_h src_buffer = nullptr;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-d1|-d2|-d3|-d4|-d5] [-s<d> size] [-t<d> tile] [-h]\n";
  std::cout << "  -d1..-d5: set number of dimensions (default: -d2)\n";
  std::cout << "  -s0 N:    set size of dimension 0 (innermost)\n";
  std::cout << "  -t0 N:    set tile size of dimension 0\n";
}

static void set_defaults() {
  // Set reasonable defaults based on ndim.
  switch (ndim) {
  case 1: sizes[0] = 128; tiles[0] = 16; break;
  case 2: sizes[0] = 32; sizes[1] = 32; tiles[0] = 4; tiles[1] = 4; break;
  case 3: sizes[0] = 8; sizes[1] = 8; sizes[2] = 8; tiles[0] = 2; tiles[1] = 2; tiles[2] = 2; break;
  case 4: sizes[0] = 4; sizes[1] = 4; sizes[2] = 4; sizes[3] = 4;
          tiles[0] = 2; tiles[1] = 2; tiles[2] = 2; tiles[3] = 2; break;
  case 5: sizes[0] = 4; sizes[1] = 4; sizes[2] = 4; sizes[3] = 2; sizes[4] = 2;
          tiles[0] = 2; tiles[1] = 2; tiles[2] = 2; tiles[3] = 2; tiles[4] = 1; break;
  }
  // Unused dims = 1.
  for (uint32_t d = ndim; d < DXA_MAX_DIMS; ++d) {
    sizes[d] = 1;
    tiles[d] = 1;
  }
}

static void parse_args(int argc, char** argv) {
  // First pass: find -d flag.
  for (int i = 1; i < argc; ++i) {
    if (strlen(argv[i]) == 3 && argv[i][0] == '-' && argv[i][1] == 'd'
        && argv[i][2] >= '1' && argv[i][2] <= '5') {
      ndim = argv[i][2] - '0';
    }
  }
  set_defaults();

  // Second pass: manual parse for all flags (getopt doesn't handle -d1, -s0 N, -t0 N well).
  for (int i = 1; i < argc; ++i) {
    if (argv[i][0] != '-') continue;
    if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) { kernel_file = argv[++i]; continue; }
    if (strcmp(argv[i], "-h") == 0) { show_usage(); exit(0); }
    // -s<d> N and -t<d> N
    if (strlen(argv[i]) == 3 && i + 1 < argc) {
      int d = argv[i][2] - '0';
      if (d >= 0 && d < (int)DXA_MAX_DIMS) {
        if (argv[i][1] == 's') { sizes[d] = atoi(argv[++i]); continue; }
        if (argv[i][1] == 't') { tiles[d] = atoi(argv[++i]); continue; }
      }
    }
  }
}

void cleanup() {
  if (device) {
    if (src_buffer) vx_buffer_release(src_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  // Validate sizes are divisible by tiles.
  uint32_t total_elems = 1;
  uint32_t group_size = 1;
  uint32_t total_groups = 1;
  for (uint32_t d = 0; d < ndim; ++d) {
    if (tiles[d] == 0 || sizes[d] == 0) {
      std::cout << "Error: size and tile must be > 0 for dim " << d << "\n";
      return -1;
    }
    if ((sizes[d] % tiles[d]) != 0) {
      std::cout << "Error: size[" << d << "]=" << sizes[d]
                << " must be divisible by tile[" << d << "]=" << tiles[d] << "\n";
      return -1;
    }
    total_elems *= sizes[d];
    group_size *= tiles[d];
    total_groups *= sizes[d] / tiles[d];
  }

  std::srand(42);

#ifdef VX_CFG_EXT_DXA_ENABLE
  std::cout << "mode: DXA\n";
#else
  std::cout << "mode: LSU\n";
#endif
  std::cout << "ndim: " << ndim << "\n";
  std::cout << "sizes:";
  for (uint32_t d = 0; d < ndim; ++d) std::cout << " " << sizes[d];
  std::cout << "\ntiles:";
  for (uint32_t d = 0; d < ndim; ++d) std::cout << " " << tiles[d];
  std::cout << "\ntotal_elems: " << total_elems << ", groups: " << total_groups << "\n";

  const uint32_t buf_size = total_elems * sizeof(TYPE);
  const uint32_t local_mem = group_size * sizeof(TYPE);

  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

#ifdef VX_CFG_EXT_DXA_ENABLE
  uint64_t isa_flags = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_EXT_DXA) == 0) {
    std::cerr << "Error: DXA ISA extension is disabled.\n";
    cleanup();
    return -1;
  }
#endif

  RT_CHECK(vx_check_occupancy(device, group_size, local_mem));

  // Set up kernel args.
  kernel_arg.ndim = ndim;
  for (uint32_t d = 0; d < DXA_MAX_DIMS; ++d) {
    kernel_arg.sizes[d] = sizes[d];
    kernel_arg.tiles[d] = tiles[d];
    kernel_arg.grids[d] = (d < ndim) ? sizes[d] / tiles[d] : 1;
  }

  // Allocate and fill source array.
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &src_buffer));
  RT_CHECK(vx_buffer_address(src_buffer, &kernel_arg.src_addr));

  std::vector<TYPE> h_src(total_elems);
  for (uint32_t i = 0; i < total_elems; ++i)
    h_src[i] = static_cast<TYPE>(i + 1);
  RT_CHECK(vx_enqueue_write(queue, src_buffer, 0, h_src.data(), buf_size, 0, nullptr, nullptr));

#ifdef VX_CFG_EXT_DXA_ENABLE
  // Program DXA descriptor for N-D source tile.
  constexpr uint32_t kDescSrc = 0;

  // Compute byte strides for row-major layout.
  // stride[d] = product(sizes[0..d]) * sizeof(TYPE)
  uint32_t byte_strides[4] = {};
  uint32_t stride_acc = sizes[0] * sizeof(TYPE);
  for (uint32_t d = 0; d < 4 && d + 1 < ndim; ++d) {
    byte_strides[d] = stride_acc;
    stride_acc *= sizes[d + 1];
  }

  switch (ndim) {
  case 1:
    RT_CHECK(vortex::dxa::program_1d(device, kDescSrc, kernel_arg.src_addr,
      sizes[0], tiles[0], sizeof(TYPE)));
    break;
  case 2:
    RT_CHECK(vortex::dxa::program_2d(device, kDescSrc, kernel_arg.src_addr,
      sizes[0], sizes[1], byte_strides[0], tiles[0], tiles[1], sizeof(TYPE)));
    break;
  case 3:
    RT_CHECK(vortex::dxa::program_3d(device, kDescSrc, kernel_arg.src_addr,
      sizes[0], sizes[1], sizes[2],
      byte_strides[0], byte_strides[1],
      tiles[0], tiles[1], tiles[2], sizeof(TYPE)));
    break;
  case 4:
    RT_CHECK(vortex::dxa::program_4d(device, kDescSrc, kernel_arg.src_addr,
      sizes[0], sizes[1], sizes[2], sizes[3],
      byte_strides[0], byte_strides[1], byte_strides[2],
      tiles[0], tiles[1], tiles[2], tiles[3], sizeof(TYPE)));
    break;
  case 5:
    RT_CHECK(vortex::dxa::program_5d(device, kDescSrc, kernel_arg.src_addr,
      sizes[0], sizes[1], sizes[2], sizes[3], sizes[4],
      byte_strides[0], byte_strides[1], byte_strides[2], byte_strides[3],
      tiles[0], tiles[1], tiles[2], tiles[3], tiles[4], sizeof(TYPE)));
    break;
  }
#endif

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  // Launch with flattened 1D grid; kernel decomposes flat group ID.
  uint32_t grid_dim[1] = {total_groups};
  uint32_t block_dim[1] = {group_size};

  std::cout << "start\n";
  vx_event_h launch_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 1;
    li.grid_dim[0]  = grid_dim[0];
    li.block_dim[0] = block_dim[0];
    li.lmem_size    = local_mem;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }
  RT_CHECK(vx_event_wait_value(launch_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(launch_ev);

  std::cout << "PASSED\n";

  cleanup();
  return 0;
}
