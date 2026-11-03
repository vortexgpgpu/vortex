#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, 8, 0, LDST_MATRIX>;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  size_t N = arg->N;
  size_t K = arg->K;
  uint32_t cta_M = arg->cta_M;

  uint32_t tid = threadIdx.x;
  uint32_t num_threads = blockDim.x; // num_warps * VX_CFG_NUM_THREADS
  uint32_t warp_rank = tid / VX_CFG_NUM_THREADS;

  size_t tile_row = blockIdx.y * cta_M;
  size_t tile_col = blockIdx.x * ctx::tileN;

  // Two shared-memory stages, each A [cta_M x tileK] row-major then
  // B [tileK x tileN] col-major: the next K tile is copied into one while the
  // warps compute on the other, so each K step needs a single barrier.
  uint32_t a_size = cta_M * ctx::tileK;
  uint32_t b_size = ctx::tileK * ctx::tileN;
  auto smem = reinterpret_cast<ctx::input_t *>(__local_mem());

  auto load_stage = [&](size_t k, ctx::input_t* A_smem) __attribute__((always_inline)) {
    // Cooperatively load A tile [cta_M x tileK] from DRAM (row-major).
    for (size_t i = 0; i < a_size; i += num_threads) {
      size_t idx = i + tid;
      size_t r = idx / ctx::tileK;
      size_t c = idx % ctx::tileK;
      A_smem[r * ctx::tileK + c] = pA[(tile_row + r) * K + (k + c)];
    }
    // Cooperatively load B tile [tileK x tileN] from DRAM (col-major source)
    // and keep it col-major in shared memory: K-contiguous, each B fragment
    // register is one word rather than a gather across K rows.
    auto B_smem = A_smem + a_size;
    for (size_t i = 0; i < b_size; i += num_threads) {
      size_t idx = i + tid;
      size_t c = idx / ctx::tileK;
      size_t r = idx % ctx::tileK;
      B_smem[c * ctx::tileK + r] = pB[(tile_col + c) * K + (k + r)];
    }
  };

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  size_t a_lane_off = ctx::lane_offset<ctx::fragment_a>(ctx::tileK);
  size_t b_lane_off = ctx::lane_offset<ctx::fragment_b, vt::col_major>(ctx::tileK);

  // One K step: copy the next tile into the other stage, which was last read
  // before the previous barrier, while computing on this one.
  auto step = [&](size_t k, ctx::input_t* A_smem, ctx::input_t* next) __attribute__((always_inline)) {
    if (k + ctx::tileK < K) {
      load_stage(k + ctx::tileK, next);
    }
    // Per-warp WMMA from shared memory.
    auto A_warp = A_smem + warp_rank * ctx::tileM * ctx::tileK;
    ctx::load_matrix_sync(fragA, A_warp, ctx::tileK, a_lane_off);
    ctx::load_matrix_sync<vt::col_major>(fragB, A_smem + a_size, ctx::tileK, b_lane_off);
    ctx::mma_sync(fragC, fragA, fragB, fragC);
    __syncthreads();
  };

  auto stage0 = smem;
  auto stage1 = smem + (a_size + b_size);
  load_stage(0, stage0);
  __syncthreads();

  // Two K steps per iteration, one per stage, so each stage's address is fixed
  // in the loop body rather than selected by a running index.
#if KUNROLL > 1
#pragma unroll KUNROLL
#endif
  for (size_t k = 0; k < K; k += 2 * ctx::tileK) {
    step(k, stage0, stage1);
    if (k + ctx::tileK < K) {
      step(k + ctx::tileK, stage1, stage0);
    }
  }

  // Each warp stores its (tileM x tileN) output tile.
  auto pTileC = pC + (tile_row + warp_rank * ctx::tileM) * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
