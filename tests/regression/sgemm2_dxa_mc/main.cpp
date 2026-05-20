// Host driver for sgemm2_dxa_mc (inter-core multicast).
// Adapted from sgemm2_dxa: one CTA per core, NUM_CORES CTAs share each B
// column-block via global-barrier-routed multicast. Requires:
//   - NUM_CORES ≥ 2 to demonstrate inter-core behavior.
//   - cross-core LMEM write fabric (socket-level) — see proposal Phase 2.
// (This test won't pass until that fabric lands; structure of the kernel +
//  descriptor programming is the artifact under review.)

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h>
#include <vector>

#include <VX_types.h>
#include <vortex2.h>
#include <dxa.h>

#include "common.h"

#define FLOAT_ULP 6

#define RT_CHECK(_expr)                                         \
  do {                                                          \
    int _ret = _expr;                                           \
    if (0 == _ret) break;                                       \
    printf("Error: '%s' returned %d!\n", #_expr, (int)_ret);   \
    cleanup();                                                  \
    exit(-1);                                                   \
  } while (false)

static bool fp_close(float a, float b) {
  union { float f; int32_t i; } fa, fb;
  fa.f = a; fb.f = b;
  return std::abs(fa.i - fb.i) <= FLOAT_ULP;
}

static void matmul_cpu(TYPE* out, const TYPE* A, const TYPE* B, uint32_t n) {
  for (uint32_t r = 0; r < n; ++r)
    for (uint32_t c = 0; c < n; ++c) {
      TYPE s(0);
      for (uint32_t k = 0; k < n; ++k) s += A[r * n + k] * B[k * n + c];
      out[r * n + c] = s;
    }
}

const char* kernel_file = "kernel.vxbin";
uint32_t size      = 16;
uint32_t tile_size = 4;
uint32_t verify    = 1;

vx_device_h device = nullptr;
vx_buffer_h A_buffer = nullptr, B_buffer = nullptr, C_buffer = nullptr;
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;
kernel_arg_t kernel_arg = {};
constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

static void show_usage() {
  std::cout << "Usage: [-k kernel] [-n size] [-t tile_size] [-q skip_verify] [-h]\n";
}

static void parse_args(int argc, char** argv) {
  int c;
  while ((c = getopt(argc, argv, "n:t:k:qh")) != -1) {
    switch (c) {
      case 'n': size = atoi(optarg); break;
      case 't': tile_size = atoi(optarg); break;
      case 'k': kernel_file = optarg; break;
      case 'q': verify = 0; break;
      case 'h': show_usage(); exit(0);
      default:  show_usage(); exit(-1);
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
  std::srand(50);

  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t num_cores = 0, num_warps = 0, num_threads = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_CORES, &num_cores));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_WARPS, &num_warps));
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &num_threads));

  if (num_cores < 2) {
    std::cout << "Error: sgemm2_dxa_mc requires NUM_CORES >= 2 "
                 "(have " << num_cores << "); use sgemm2_dxa_mw on single-core\n";
    cleanup();
    return -1;
  }
  if (size % tile_size != 0) {
    std::cout << "Error: size must be divisible by tile_size\n";
    cleanup();
    return -1;
  }

  // Multicast group spans NUM_CORES cores. Grid arranged so consecutive CTAs
  // in row order land on consecutive cores (KMU's default round-robin).
  const uint32_t mc_group_size = (uint32_t)num_cores;
  if ((size / tile_size) % mc_group_size != 0) {
    std::cout << "Error: (size/tile_size) must be divisible by mc_group_size ("
              << mc_group_size << ")\n";
    cleanup();
    return -1;
  }

  const uint32_t chunk_k = size;
  const uint32_t buf_bytes = size * size * sizeof(TYPE);
  const uint32_t local_mem = (chunk_k * tile_size + chunk_k * tile_size) * sizeof(TYPE);

  std::cout << "sgemm2_dxa_mc (inter-core multicast)\n";
  std::cout << "  size=" << size << ", tile_size=" << tile_size
            << ", chunk_k=" << chunk_k << ", mc_group_size=" << mc_group_size << "\n";
  std::cout << "  block=" << tile_size << "x" << tile_size
            << ", grid=" << (size/tile_size) << "x" << (size/tile_size) << "\n";
  std::cout << "  local_mem=" << local_mem << " bytes\n";

  RT_CHECK(vx_buffer_create(device, buf_bytes, VX_MEM_READ,  &A_buffer));
  RT_CHECK(vx_buffer_address(A_buffer, &kernel_arg.A_addr));
  RT_CHECK(vx_buffer_create(device, buf_bytes, VX_MEM_READ,  &B_buffer));
  RT_CHECK(vx_buffer_address(B_buffer, &kernel_arg.B_addr));
  RT_CHECK(vx_buffer_create(device, buf_bytes, VX_MEM_WRITE, &C_buffer));
  RT_CHECK(vx_buffer_address(C_buffer, &kernel_arg.C_addr));

  std::vector<TYPE> hA(size*size), hB(size*size), hC(size*size, 0);
  for (uint32_t i = 0; i < size*size; ++i) {
    hA[i] = (float)std::rand() / RAND_MAX;
    hB[i] = (float)std::rand() / RAND_MAX;
  }
  RT_CHECK(vx_enqueue_write(queue, A_buffer, 0, hA.data(), buf_bytes, 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, B_buffer, 0, hB.data(), buf_bytes, 0, nullptr, nullptr));

  RT_CHECK(vx_dxa_program_desc_2d(device, kDescA, kernel_arg.A_addr,
    /*size0=*/size, /*size1=*/size,
    /*stride0_bytes=*/size * sizeof(TYPE),
    /*tile0=*/chunk_k, /*tile1=*/tile_size,
    /*elem_bytes=*/sizeof(TYPE)));

  RT_CHECK(vx_dxa_program_desc_2d(device, kDescB, kernel_arg.B_addr,
    /*size0=*/size, /*size1=*/size,
    /*stride0_bytes=*/size * sizeof(TYPE),
    /*tile0=*/tile_size, /*tile1=*/chunk_k,
    /*elem_bytes=*/sizeof(TYPE)));

  // For inter-core multicast, the SMEM offset on each receiver core is the
  // same — each core's __local_mem() has the same layout. The cross-core
  // LMEM bus routes by core_id.
  RT_CHECK(vx_dxa_program_desc_multicast(device, kDescB, local_mem));

  kernel_arg.size          = size;
  kernel_arg.tile_size     = tile_size;
  kernel_arg.chunk_k       = chunk_k;
  kernel_arg.mc_group_size = mc_group_size;

  uint32_t grid_dim[2]  = { size / tile_size, size / tile_size };
  uint32_t block_dim[2] = { tile_size, tile_size };

  RT_CHECK(vx_module_load_file(device, kernel_file, &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

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
    RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  }

  vx_event_h read_ev = nullptr;
  RT_CHECK(vx_enqueue_read(queue, hC.data(), C_buffer, 0, buf_bytes, 1, &launch_ev, &read_ev));
  RT_CHECK(vx_event_wait_value(read_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(read_ev);
  vx_event_release(launch_ev);

  int errors = 0;
  if (verify) {
    std::vector<TYPE> hRef(size*size);
    matmul_cpu(hRef.data(), hA.data(), hB.data(), size);
    for (uint32_t i = 0; i < size*size; ++i) {
      if (!fp_close(hC[i], hRef[i])) {
        if (errors < 20)
          printf("*** error: [%u] expected=%f, actual=%f\n", i, hRef[i], hC[i]);
        ++errors;
      }
    }
  }

  cleanup();
  if (errors) { std::cout << "FAILED (" << errors << " errors)\n"; return errors; }
  std::cout << "PASSED\n";
  return 0;
}
