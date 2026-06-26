// Host driver for dxa_copy_mcast — intra-core multicast.
//
// num_recv = VX_CFG_NUM_WARPS single-warp CTAs co-resident on one core, each
// receiving the same tile via DXA multicast.

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
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
static uint32_t    mode        = DXA_COPY_MCAST_MODE_SMOKE;
static uint32_t    writeback_mode = DXA_COPY_MCAST_WRITEBACK_FULL;
static uint32_t    pipeline_depth = 1;
static uint32_t    num_ctas    = 0;
static uint32_t    verify      = 1;

static vx_device_h device     = nullptr;
static vx_buffer_h src_buffer = nullptr;
static vx_buffer_h dst_buffer = nullptr;
static vx_queue_h  queue      = nullptr;
static vx_module_h module_    = nullptr;
static vx_kernel_h kernel     = nullptr;
static kernel_arg_t kernel_arg = {};

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-r src_rows] [-c src_cols] "
               "[-R tile_rows] [-C tile_cols] "
               "[--mode=smoke|percta|mcast] [--writeback=full|sample|none] "
               "[--pipeline-depth N] [--num-ctas N] [--verify=0|1] [-h]\n";
}

static const char* mode_name(uint32_t value) {
  switch (value) {
  case DXA_COPY_MCAST_MODE_SMOKE:  return "smoke";
  case DXA_COPY_MCAST_MODE_PERCTA: return "percta";
  case DXA_COPY_MCAST_MODE_MCAST:  return "mcast";
  default:                         return "unknown";
  }
}

static const char* writeback_name(uint32_t value) {
  switch (value) {
  case DXA_COPY_MCAST_WRITEBACK_FULL:   return "full";
  case DXA_COPY_MCAST_WRITEBACK_SAMPLE: return "sample";
  case DXA_COPY_MCAST_WRITEBACK_NONE:   return "none";
  default:                              return "unknown";
  }
}

static bool parse_mode_value(const char* value) {
  if (std::strcmp(value, "smoke") == 0) {
    mode = DXA_COPY_MCAST_MODE_SMOKE;
    return true;
  }
  if (std::strcmp(value, "percta") == 0) {
    mode = DXA_COPY_MCAST_MODE_PERCTA;
    return true;
  }
  if (std::strcmp(value, "mcast") == 0) {
    mode = DXA_COPY_MCAST_MODE_MCAST;
    return true;
  }
  return false;
}

static bool parse_writeback_value(const char* value) {
  if (std::strcmp(value, "full") == 0) {
    writeback_mode = DXA_COPY_MCAST_WRITEBACK_FULL;
    return true;
  }
  if (std::strcmp(value, "sample") == 0) {
    writeback_mode = DXA_COPY_MCAST_WRITEBACK_SAMPLE;
    return true;
  }
  if (std::strcmp(value, "none") == 0) {
    writeback_mode = DXA_COPY_MCAST_WRITEBACK_NONE;
    return true;
  }
  return false;
}

static const char* require_arg(int argc, char** argv, int& i, const char* opt) {
  if (i + 1 >= argc) {
    std::cout << "Error: missing value for " << opt << "\n";
    show_usage();
    exit(-1);
  }
  return argv[++i];
}

static void parse_args(int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "-h") == 0 || std::strcmp(arg, "--help") == 0) {
      show_usage();
      exit(0);
    } else if (std::strcmp(arg, "-k") == 0) {
      kernel_file = require_arg(argc, argv, i, "-k");
    } else if (std::strcmp(arg, "-r") == 0) {
      src_rows = std::atoi(require_arg(argc, argv, i, "-r"));
    } else if (std::strncmp(arg, "-r", 2) == 0 && arg[2] != '\0') {
      src_rows = std::atoi(arg + 2);
    } else if (std::strcmp(arg, "-c") == 0) {
      src_cols = std::atoi(require_arg(argc, argv, i, "-c"));
    } else if (std::strncmp(arg, "-c", 2) == 0 && arg[2] != '\0') {
      src_cols = std::atoi(arg + 2);
    } else if (std::strcmp(arg, "-R") == 0) {
      tile_rows = std::atoi(require_arg(argc, argv, i, "-R"));
    } else if (std::strncmp(arg, "-R", 2) == 0 && arg[2] != '\0') {
      tile_rows = std::atoi(arg + 2);
    } else if (std::strcmp(arg, "-C") == 0) {
      tile_cols = std::atoi(require_arg(argc, argv, i, "-C"));
    } else if (std::strncmp(arg, "-C", 2) == 0 && arg[2] != '\0') {
      tile_cols = std::atoi(arg + 2);
    } else if (std::strncmp(arg, "--mode=", 7) == 0) {
      if (!parse_mode_value(arg + 7)) {
        std::cout << "Error: invalid mode " << (arg + 7) << "\n";
        show_usage();
        exit(-1);
      }
    } else if (std::strcmp(arg, "--mode") == 0) {
      if (!parse_mode_value(require_arg(argc, argv, i, "--mode"))) {
        std::cout << "Error: invalid mode\n";
        show_usage();
        exit(-1);
      }
    } else if (std::strncmp(arg, "--writeback=", 12) == 0) {
      if (!parse_writeback_value(arg + 12)) {
        std::cout << "Error: invalid writeback mode " << (arg + 12) << "\n";
        show_usage();
        exit(-1);
      }
    } else if (std::strcmp(arg, "--writeback") == 0) {
      if (!parse_writeback_value(require_arg(argc, argv, i, "--writeback"))) {
        std::cout << "Error: invalid writeback mode\n";
        show_usage();
        exit(-1);
      }
    } else if (std::strncmp(arg, "--pipeline-depth=", 17) == 0) {
      pipeline_depth = std::atoi(arg + 17);
    } else if (std::strcmp(arg, "--pipeline-depth") == 0) {
      pipeline_depth = std::atoi(require_arg(argc, argv, i, "--pipeline-depth"));
    } else if (std::strncmp(arg, "--num-ctas=", 11) == 0) {
      num_ctas = std::atoi(arg + 11);
    } else if (std::strcmp(arg, "--num-ctas") == 0) {
      num_ctas = std::atoi(require_arg(argc, argv, i, "--num-ctas"));
    } else if (std::strncmp(arg, "--verify=", 9) == 0) {
      verify = std::atoi(arg + 9);
    } else if (std::strcmp(arg, "--verify") == 0) {
      verify = std::atoi(require_arg(argc, argv, i, "--verify"));
    } else {
      std::cout << "Error: unknown argument " << arg << "\n";
      show_usage();
      exit(-1);
    }
  }
}

void cleanup() {
  if (device) {
    if (src_buffer) vx_buffer_release(src_buffer);
    if (dst_buffer) vx_buffer_release(dst_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_dump_perf(device, stdout);
    vx_device_release(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);
  std::srand(42);

  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  // Single-warp CTAs: block = (tile_cols, 1) so each CTA uses exactly one
  // warp. num_recv = VX_CFG_NUM_WARPS so all receivers fit co-resident on one core.
  uint64_t num_warps = 0, num_threads = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  if (tile_rows == 0 || tile_cols == 0 || src_rows == 0 || src_cols == 0) {
    std::cout << "Error: source and tile dimensions must be non-zero\n";
    cleanup();
    return -1;
  }
  if (tile_rows > src_rows || tile_cols > src_cols) {
    std::cout << "Error: tile must fit within source matrix\n";
    cleanup();
    return -1;
  }

  const bool is_smoke = (mode == DXA_COPY_MCAST_MODE_SMOKE);
  verify = verify ? 1 : 0;
  if (is_smoke && tile_cols != (uint32_t)num_threads) {
    std::cout << "Error: smoke mode tile_cols (" << tile_cols
              << ") must equal VX_CFG_NUM_THREADS (" << num_threads
              << ") for single-warp CTAs\n";
    cleanup();
    return -1;
  }

  if (!is_smoke && num_ctas == 0)
    num_ctas = 4;

  if (!is_smoke && ((src_rows % tile_rows) != 0 || (src_cols % tile_cols) != 0)) {
    std::cout << "Error: benchmark mode requires tile dimensions to divide the source matrix\n";
    cleanup();
    return -1;
  }

  const uint32_t num_recv = is_smoke ? (uint32_t)num_warps : num_ctas;
  if (num_recv == 0 || num_recv > (uint32_t)num_warps) {
    std::cout << "Error: num_ctas/num_recv (" << num_recv
              << ") must be in 1.." << num_warps << "\n";
    cleanup();
    return -1;
  }
  if (is_smoke)
    pipeline_depth = 1;
  if (pipeline_depth == 0)
    pipeline_depth = 1;
  if (pipeline_depth > DXA_COPY_MCAST_MAX_PIPELINE_DEPTH)
    pipeline_depth = DXA_COPY_MCAST_MAX_PIPELINE_DEPTH;

  const uint32_t tile_elems = tile_rows * tile_cols;
  const uint32_t tile_local_mem = tile_elems * sizeof(TYPE);
  const uint32_t src_elems  = src_rows * src_cols;
  const uint32_t src_bytes  = src_elems * sizeof(TYPE);
  const uint32_t tile_grid_rows = is_smoke ? 1 : (src_rows / tile_rows);
  const uint32_t tile_grid_cols = is_smoke ? 1 : (src_cols / tile_cols);
  const uint32_t tile_count = tile_grid_rows * tile_grid_cols;
  const uint32_t block_threads = is_smoke ? tile_cols : (uint32_t)num_threads;
  const uint32_t sample_elems_per_tile =
    tile_elems < block_threads ? tile_elems : block_threads;
  const uint32_t dst_elems_per_recv = is_smoke ? tile_elems :
    (writeback_mode == DXA_COPY_MCAST_WRITEBACK_FULL ? src_elems :
     writeback_mode == DXA_COPY_MCAST_WRITEBACK_SAMPLE ?
       tile_count * sample_elems_per_tile : 1);
  const uint32_t dst_bytes  = num_recv * dst_elems_per_recv * sizeof(TYPE);

  uint64_t local_mem_size = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_LOCAL_MEM_SIZE, &local_mem_size));
  while (pipeline_depth > 1) {
    const uint64_t requested = (uint64_t)tile_local_mem * pipeline_depth;
    const uint64_t aligned =
      (requested + VX_CFG_MEM_BLOCK_SIZE - 1) & ~(uint64_t)(VX_CFG_MEM_BLOCK_SIZE - 1);
    if (aligned * num_recv <= local_mem_size)
      break;
    --pipeline_depth;
  }
  const uint32_t local_mem  = tile_local_mem * pipeline_depth;

  std::cout << "dxa_copy_mcast (intra-core multicast)\n";
  std::cout << "  mode: " << mode_name(mode) << "\n";
  std::cout << "  writeback: " << writeback_name(writeback_mode) << "\n";
  std::cout << "  pipeline_depth: " << pipeline_depth << "\n";
  std::cout << "  source: " << src_rows << " x " << src_cols
            << ", tile: " << tile_rows << " x " << tile_cols << "\n";
  std::cout << "  tile_grid: " << tile_grid_rows << " x " << tile_grid_cols << "\n";
  std::cout << "  block=" << block_threads << "x1 (single warp), grid="
            << num_recv << "x1, num_recv=" << num_recv << "\n";
  std::cout << "  local_mem=" << local_mem << " bytes\n";

  RT_CHECK(vx_check_occupancy(device, block_threads, 0));
  {
    const uint64_t aligned_local_mem =
      ((uint64_t)local_mem + VX_CFG_MEM_BLOCK_SIZE - 1) & ~(uint64_t)(VX_CFG_MEM_BLOCK_SIZE - 1);
    const uint64_t required_local_mem = aligned_local_mem * num_recv;
    if (required_local_mem > local_mem_size) {
      std::cout << "Error: multicast group local-memory request exceeds capacity ("
                << required_local_mem << " > " << local_mem_size << ")\n";
      cleanup();
      return -1;
    }
  }

  RT_CHECK(vx_buffer_create(device, src_bytes, VX_MEM_READ,  &src_buffer));
  RT_CHECK(vx_buffer_address(src_buffer, &kernel_arg.src_addr));
  RT_CHECK(vx_buffer_create(device, dst_bytes, VX_MEM_WRITE, &dst_buffer));
  RT_CHECK(vx_buffer_address(dst_buffer, &kernel_arg.dst_addr));

  // Distinctive source pattern (linear index + 1).
  std::vector<TYPE> h_src(src_elems);
  for (uint32_t i = 0; i < src_elems; ++i)
    h_src[i] = static_cast<TYPE>(i + 1);
  RT_CHECK(vx_enqueue_write(queue, src_buffer, 0, h_src.data(), src_bytes, 0, nullptr, nullptr));

  // Program DXA descriptor for the tile.
  constexpr uint32_t kDescSrc = 0;
  RT_CHECK(vortex::dxa::program_2d(device, kDescSrc, kernel_arg.src_addr,
    /*size0=*/src_cols, /*size1=*/src_rows,
    /*stride0_bytes=*/src_cols * sizeof(TYPE),
    /*tile0=*/tile_cols, /*tile1=*/tile_rows,
    /*elem_bytes=*/sizeof(TYPE)));
  // Multicast stride = local_mem (per-CTA SMEM size). Dispatcher allocates
  // sequential LMEM regions so receiver k's region starts at k*local_mem.
  RT_CHECK(vortex::dxa::set_multicast(device, kDescSrc, local_mem));

  kernel_arg.tile_rows      = tile_rows;
  kernel_arg.tile_cols      = tile_cols;
  kernel_arg.src_rows       = src_rows;
  kernel_arg.src_cols       = src_cols;
  kernel_arg.src_row_stride = src_cols;
  kernel_arg.num_recv       = num_recv;
  kernel_arg.mode           = mode;
  kernel_arg.writeback_mode = writeback_mode;
  kernel_arg.pipeline_depth = pipeline_depth;

  uint32_t grid_dim[2]  = { num_recv, 1 };
  uint32_t block_dim[2] = { block_threads, 1 };

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  // Host result buffer — must outlive the async read enqueued below.
  std::vector<TYPE> h_dst(num_recv * dst_elems_per_recv, 0);

  std::cout << "start\n";
  vx_event_h launch_ev = nullptr;
  {
    vx_launch_info_t li = {};
    li.struct_size  = sizeof(li);
    li.kernel       = kernel;
    li.args_host    = &kernel_arg;
    li.args_size    = sizeof(kernel_arg);
    li.ndim         = 2;
    li.grid_dim[0]  = grid_dim[0];
    li.grid_dim[1]  = grid_dim[1];
    li.block_dim[0] = block_dim[0];
    li.block_dim[1] = block_dim[1];
    li.lmem_size    = local_mem;
    // Multicast group: all num_recv receiver CTAs share one B tile and must be
    // co-resident on one core in contiguous CTA slots. The grid is num_recv×1,
    // so the group spans the X axis. Without this, cluster_size defaults to 1,
    // get_cluster_rank() is 0 for every CTA, and every CTA (not just rank-0)
    // fires the multicast — over-releasing receivers' event barriers and
    // walking the barrier-id space past its bounds.
    li.cluster_dim[0] = num_recv;
    li.cluster_dim[1] = 1;
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  int errors = 0;
  if (verify && writeback_mode != DXA_COPY_MCAST_WRITEBACK_NONE) {
    vx_event_h read_ev = nullptr;
    RT_CHECK(vx_enqueue_read(queue, h_dst.data(), dst_buffer, 0, dst_bytes, 1, &launch_ev, &read_ev));
    RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(read_ev);

    if (is_smoke) {
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
    } else if (writeback_mode == DXA_COPY_MCAST_WRITEBACK_FULL) {
      for (uint32_t rcv = 0; rcv < num_recv; ++rcv) {
        for (uint32_t r = 0; r < src_rows; ++r) {
          for (uint32_t c = 0; c < src_cols; ++c) {
            TYPE expected = h_src[r * src_cols + c];
            TYPE actual   = h_dst[rcv * src_elems + r * src_cols + c];
            if (expected != actual) {
              if (errors < 20)
                printf("*** error: recv=%u [r=%u,c=%u] expected=%f actual=%f\n",
                       rcv, r, c, (float)expected, (float)actual);
              ++errors;
            }
          }
        }
      }
    } else {
      for (uint32_t rcv = 0; rcv < num_recv; ++rcv) {
        for (uint32_t tile_y = 0; tile_y < tile_grid_rows; ++tile_y) {
          for (uint32_t tile_x = 0; tile_x < tile_grid_cols; ++tile_x) {
            const uint32_t tile_idx = tile_y * tile_grid_cols + tile_x;
            for (uint32_t idx = 0; idx < sample_elems_per_tile; ++idx) {
              const uint32_t local_r = idx / tile_cols;
              const uint32_t local_c = idx - local_r * tile_cols;
              const uint32_t src_idx =
                (tile_y * tile_rows + local_r) * src_cols +
                tile_x * tile_cols + local_c;
              TYPE expected = h_src[src_idx];
              TYPE actual = h_dst[(rcv * tile_count + tile_idx) *
                                  sample_elems_per_tile + idx];
              if (expected != actual) {
                if (errors < 20)
                  printf("*** error: recv=%u tile=[%u,%u] idx=%u expected=%f actual=%f\n",
                         rcv, tile_y, tile_x, idx, (float)expected, (float)actual);
                ++errors;
              }
            }
          }
        }
      }
    }
  } else {
    RT_CHECK(vx_event_wait_value(launch_ev, 1, VX_TIMEOUT_INFINITE));
  }
  vx_event_release(launch_ev);

  cleanup();
  std::cout << "DXA_COPY_MCAST_RESULT mode=" << mode_name(mode)
            << " writeback=" << writeback_name(writeback_mode)
            << " pipeline_depth=" << pipeline_depth
            << " rows=" << src_rows
            << " cols=" << src_cols
            << " tile_rows=" << tile_rows
            << " tile_cols=" << tile_cols
            << " tile_grid_rows=" << tile_grid_rows
            << " tile_grid_cols=" << tile_grid_cols
            << " num_ctas=" << num_recv
            << " verify=" << verify
            << " status=" << (errors ? "FAIL" : "PASS")
            << "\n";
  if (errors) {
    std::cout << "FAILED (" << errors << " errors)\n";
    return errors;
  }
  std::cout << "PASSED\n";
  return 0;
}
