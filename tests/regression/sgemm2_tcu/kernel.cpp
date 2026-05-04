#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::ITYPE, vt::OTYPE>;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  uint32_t M = arg->M;
  uint32_t N = arg->N;
  uint32_t K = arg->K;
  uint32_t cta_M = arg->cta_M;

  uint32_t tid = threadIdx.x;
  uint32_t num_threads = blockDim.x; // num_warps * NUM_THREADS
  uint32_t warp_rank = tid / NUM_THREADS;

  uint32_t tile_row = blockIdx.y * cta_M;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  // Shared memory: A [cta_M x tileK] row-major, B [tileK x tileN] row-major
  auto smem   = reinterpret_cast<ctx::input_t *>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + cta_M * ctx::tileK;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    // Cooperatively load A tile [cta_M x tileK] from DRAM (row-major).
    uint32_t a_size = cta_M * ctx::tileK;
    for (uint32_t i = 0; i < a_size; i += num_threads) {
      uint32_t idx = i + tid;
      uint32_t r = idx / ctx::tileK;
      uint32_t c = idx % ctx::tileK;
      A_smem[r * ctx::tileK + c] = pA[(tile_row + r) * K + (k + c)];
    }

    // Cooperatively load B tile [tileK x tileN] from DRAM (col-major source)
    // and store row-major in shared memory.
    uint32_t b_size = ctx::tileK * ctx::tileN;
    for (uint32_t i = 0; i < b_size; i += num_threads) {
      uint32_t idx = i + tid;
      uint32_t r = idx / ctx::tileN;
      uint32_t c = idx % ctx::tileN;
      B_smem[r * ctx::tileN + c] = pB[(tile_col + c) * K + (k + r)];
    }

    __syncthreads();

    // Per-warp WMMA from shared memory.
    auto A_warp = A_smem + warp_rank * ctx::tileM * ctx::tileK;
    ctx::load_matrix_sync(fragA, A_warp, ctx::tileK);
    ctx::load_matrix_sync(fragB, B_smem, ctx::tileN);
    ctx::mma_sync(fragC, fragA, fragB, fragC);

    __syncthreads();
  }

  // Each warp stores its (tileM x tileN) output tile.
  auto pTileC = pC + (tile_row + warp_rank * ctx::tileM) * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
