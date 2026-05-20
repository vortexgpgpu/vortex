#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <VX_types.h>
#include <vortex.h>
#include <dxa.h>

#include "common.h"

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret)                                              \
      break;                                                    \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

template <typename Type>
class Comparator {};

template <>
class Comparator<float> {
public:
  static const char* type_str() {
    return "float";
  }
  static float generate() {
    return static_cast<float>(rand()) / RAND_MAX;
  }
  static bool compare(float a, float b, int index, int errors) {
    union fi_t {
      float f;
      int32_t i;
    };
    fi_t fa, fb;
    fa.f = a;
    fb.f = b;
    auto d = std::abs(fa.i - fb.i);
    if (d > FLOAT_ULP) {
      if (errors < 100) {
        printf("*** error: [%d] expected=%f, actual=%f\n", index, a, b);
      }
      return false;
    }
    return true;
  }
};

static void matmul_cpu(TYPE* out, const TYPE* A, const TYPE* B, uint32_t n) {
  for (uint32_t row = 0; row < n; ++row) {
    for (uint32_t col = 0; col < n; ++col) {
      TYPE sum(0);
      for (uint32_t k = 0; k < n; ++k) {
        sum += A[row * n + k] * B[k * n + col];
      }
      out[row * n + col] = sum;
    }
  }
}

const char* kernel_file = "kernel.vxbin";
uint32_t size = 64;
uint32_t tile_size = 8;
uint32_t chunk_k = 0;
uint32_t mode = 2;
uint32_t verify = 1;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr;
vx_buffer_h B_buffer = nullptr;
vx_buffer_h C_buffer = nullptr;
vx_queue_h  queue = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel = nullptr;
kernel_arg_t kernel_arg = {};
constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-n matrix_size] [-t tile_size] [-c chunk_k] "
               "[-m mode(1=single,2=double)] [-q skip_verify] [-h]\n"
            << "  chunk_k=0 means auto full-K bounded by local memory\n";
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:t:c:m:k:qh")) != -1) {
    switch (c) {
    case 'n': size = atoi(optarg); break;
    case 't': tile_size = atoi(optarg); break;
    case 'c': chunk_k = atoi(optarg); break;
    case 'm': mode = atoi(optarg); break;
    case 'k': kernel_file = optarg; break;
    case 'q': verify = 0; break;
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
    if (A_buffer) vx_buffer_release(A_buffer);
    if (B_buffer) vx_buffer_release(B_buffer);
    if (C_buffer) vx_buffer_release(C_buffer);
    if (kernel)  vx_kernel_release(kernel);
    if (module_) vx_module_release(module_);
    if (queue)   vx_queue_release(queue);
    vx_device_release(device);
  }
}

int main(int argc, char* argv[]) {
  parse_args(argc, argv);

  if ((size % tile_size) != 0) {
    std::cout << "Error: size must be divisible by tile_size\n";
    return -1;
  }
  if (mode != 1 && mode != 2) {
    std::cout << "Error: mode must be 1(single) or 2(double)\n";
    return -1;
  }

  std::srand(50);

  std::cout << "open device connection\n";
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t isa_flags = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
#ifdef ISA_EXT_DXA
  const uint64_t dxa_isa_bit = (1ull << (32 + ISA_EXT_DXA));
  if ((isa_flags & dxa_isa_bit) == 0) {
    std::cerr << "Error: DXA ISA extension is disabled in runtime CONFIGS.\n";
    cleanup();
    return -1;
  }
#endif

  const uint32_t size_sq = size * size;
  const uint32_t buf_size = size_sq * sizeof(TYPE);
  const uint32_t group_size = tile_size * tile_size;

  std::cout << "data type: " << Comparator<TYPE>::type_str() << "\n";
  std::cout << "matrix size: " << size << "x" << size << "\n";
  std::cout << "mode: " << mode << " (1=single/full-K, 2=double/chunked-K)\n";

  // Derive the per-block local-memory budget (total LMEM split across the
  // blocks resident per core) so chunk_k can be auto-sized to fit. The
  // occupancy math mirrors vx_check_occupancy; that runtime call is now a
  // pure validator and is invoked below once local_mem is known.
  uint32_t max_localmem = 0;
  {
    uint64_t warps_per_core = 0, threads_per_warp = 0, lmem_size = 0;
    RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &warps_per_core));
    RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &threads_per_warp));
    RT_CHECK(vx_device_query(device, VX_CAPS_LOCAL_MEM_SIZE, &lmem_size));
    uint32_t warps_per_block = (group_size + threads_per_warp - 1) / threads_per_warp;
    uint32_t blocks_per_core = warps_per_core / warps_per_block;
    max_localmem = uint32_t(lmem_size / blocks_per_core);
  }
  const uint32_t stage_count = (mode == 2) ? 2u : 1u;
  const uint32_t bytes_per_k = stage_count * (2u * tile_size * sizeof(TYPE));
  if (bytes_per_k == 0) {
    std::cout << "Error: invalid bytes_per_k\n";
    cleanup();
    return -1;
  }

  // Single-buffer mode always uses full-K (one shot, no K-loop).
  // Double-buffer mode uses chunk_k (user-specified or auto-bounded by SMEM).
  if (mode == 1) {
    chunk_k = size;
  }

  // Determine chunk_k bounded by SMEM capacity and divisibility.
  uint32_t chunk_k_target = (chunk_k != 0) ? std::min(chunk_k, size) : size;
  uint32_t chunk_k_cap = max_localmem / bytes_per_k;
  chunk_k_cap = std::max(1u, std::min(chunk_k_cap, size));
  uint32_t chosen_chunk_k = std::min(chunk_k_target, chunk_k_cap);
  while (chosen_chunk_k > 1 && (size % chosen_chunk_k) != 0) {
    --chosen_chunk_k;
  }
  if ((size % chosen_chunk_k) != 0) {
    std::cout << "Error: unable to find valid chunk_k divisor\n";
    cleanup();
    return -1;
  }
  chunk_k = chosen_chunk_k;

  const uint32_t tile_elems_a = tile_size * chunk_k;
  const uint32_t tile_elems_b = chunk_k * tile_size;
  const uint32_t local_mem = stage_count * (tile_elems_a + tile_elems_b) * sizeof(TYPE);

  std::cout << "tile size: " << tile_size
            << ", chunk_k(requested)=" << chunk_k_target
            << ", chunk_k(selected)=" << chunk_k << "\n";
  std::cout << "local memory: " << local_mem << " bytes\n";
  std::cout << "occupancy: max_localmem=" << max_localmem << " bytes\n";
  RT_CHECK(vx_check_occupancy(device, group_size, local_mem));

  uint32_t grid_dim[2]  = {size / tile_size, size / tile_size};
  uint32_t block_dim[2] = {tile_size, tile_size};
  kernel_arg.size = size;
  kernel_arg.tile_size = tile_size;
  kernel_arg.chunk_k = chunk_k;
  kernel_arg.mode = mode;

  std::cout << "allocate device memory\n";
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &A_buffer));
  RT_CHECK(vx_buffer_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_READ, &B_buffer));
  RT_CHECK(vx_buffer_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_buffer_create(device, buf_size, VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_buffer_address(C_buffer, &kernel_arg.C_addr));

  std::vector<TYPE> h_A(size_sq);
  std::vector<TYPE> h_B(size_sq);
  std::vector<TYPE> h_C(size_sq);
  for (uint32_t i = 0; i < size_sq; ++i) {
    h_A[i] = Comparator<TYPE>::generate();
    h_B[i] = Comparator<TYPE>::generate();
  }

  RT_CHECK(vx_enqueue_write(queue, A_buffer, 0, h_A.data(), buf_size, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, B_buffer, 0, h_B.data(), buf_size, 0, nullptr, nullptr));

  // Descriptor A: dim0=k, dim1=row => A[row, k]
  RT_CHECK(vx_dxa_program_desc_2d(device, kDescA, kernel_arg.A_addr,
    /*size0=*/size, /*size1=*/size,
    /*stride0_bytes=*/size * sizeof(TYPE),
    /*tile0=*/chunk_k, /*tile1=*/tile_size,
    /*elem_bytes=*/sizeof(TYPE)));

  // Descriptor B: dim0=col, dim1=k => B[k, col]
  RT_CHECK(vx_dxa_program_desc_2d(device, kDescB, kernel_arg.B_addr,
    /*size0=*/size, /*size1=*/size,
    /*stride0_bytes=*/size * sizeof(TYPE),
    /*tile0=*/tile_size, /*tile1=*/chunk_k,
    /*elem_bytes=*/sizeof(TYPE)));

  std::cout << "load kernel module\n";
  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  std::cout << "launch kernel\n";
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
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  vx_event_h read_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, h_C.data(), C_buffer, 0, buf_size, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  if (verify) {
    std::vector<TYPE> h_ref(size_sq);
    matmul_cpu(h_ref.data(), h_A.data(), h_B.data(), size);
    for (uint32_t i = 0; i < h_ref.size(); ++i) {
      if (!Comparator<TYPE>::compare(h_C[i], h_ref[i], i, errors)) {
        ++errors;
      }
    }
  }

  cleanup();

  if (errors != 0) {
    std::cout << "Found " << errors << " errors\n";
    std::cout << "FAILED\n";
    return errors;
  }

  std::cout << "PASSED\n";
  return 0;
}
