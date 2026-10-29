#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;

using ctx = vt::umma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, UMMA_NRC>;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  uint32_t N = arg->N;
  uint32_t K = arg->K;

  uint32_t tid = threadIdx.x;
  uint32_t num_threads = blockDim.x;  // warps * VX_CFG_NUM_THREADS
  [[maybe_unused]] uint32_t warp_rank = tid / VX_CFG_NUM_THREADS;
  uint32_t num_warps = num_threads / VX_CFG_NUM_THREADS;

  // CTA tile dimensions
  uint32_t cta_M = num_warps * ctx::xtileM;
  uint32_t tile_row = blockIdx.y * cta_M;
  uint32_t tile_col = blockIdx.x * ctx::xtileN;

  // Shared memory layout: A [cta_M × tileK] then B [tileK × per_warp_N]
  auto smem   = reinterpret_cast<ctx::input_t *>(__local_mem());
  [[maybe_unused]] auto A_smem = smem;
  auto B_smem = smem + cta_M * ctx::tileK;

  // Every warp calls tmem_alloc directly — the allocator is CTA-scoped and
  // idempotent (see tcu_unit.cpp's tmem_alloc), so every warp gets the same
  // handle back with no elected-thread/shared-memory broadcast needed.
#ifdef UMMA_A_TMEM
  // One CTA-scoped allocation covers both: C at [handle, handle+xtileN) and
  // A's k-stripe at [a_col, a_col+a_tmem_words). The allocator is idempotent
  // per CTA, so two separate alloc calls would hand back the same handle.
  uint32_t handle = vt::vx_tmem_alloc(ctx::xtileN + ctx::a_tmem_words);
  uint32_t a_col  = handle + ctx::xtileN;
#else
  uint32_t handle = vt::vx_tmem_alloc(ctx::xtileN);
#endif
  ctx::fill_tmem(handle, 0);
  __syncthreads();

  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    // Cooperatively load A [cta_M × tileK] into smem.
    //   Default (block-major):  A_smem[a_blockmajor_idx(r, c)].
    //   WGMMA_RMAJOR_A:         A_smem[r * tileK + c] — row-major M-outer
    //   K-inner, matching the abuf row-major fetch (ldm = tileK).
  #ifndef UMMA_A_TMEM
    uint32_t a_size = cta_M * ctx::tileK;
    for (uint32_t i = 0; i < a_size; i += num_threads) {
      uint32_t idx = i + tid;
      uint32_t r = idx / ctx::tileK;
      uint32_t c = idx % ctx::tileK;
    #ifdef WGMMA_RMAJOR_A
      A_smem[r * ctx::tileK + c] = pA[(tile_row + r) * K + (k + c)];
    #else
      A_smem[ctx::a_blockmajor_idx(r, c)] = pA[(tile_row + r) * K + (k + c)];
    #endif
    }
  #endif

    // Cooperatively load B into smem (block-major):
    //   B_smem[(k_blk*N_STEPS + n_blk)*BW + n_in*tcK + k_in]
    uint32_t b_size = ctx::tileK * ctx::xtileN;
    for (uint32_t i = 0; i < b_size; i += num_threads) {
      uint32_t idx = i + tid;
      uint32_t r = idx / ctx::xtileN;
      uint32_t c = idx % ctx::xtileN;
      B_smem[ctx::b_blockmajor_idx(r, c)] = pB[(k + r) * N + (tile_col + c)];
    }

  #ifdef UMMA_A_TMEM
    ctx::store_a_tmem(a_col, pA, tile_row, k, K);
  #endif

    __syncthreads();

    auto desc_b = vt::vx_make_smem_desc(B_smem, 0); // block-major: stride field unused

  #ifdef UMMA_A_TMEM
    ctx::umma_sync(vt::vx_make_tmem_ref(a_col), desc_b, handle);
  #else
    // Each warp's A slice — base differs by layout:
    //   block-major: warp_rank * a_warp_elems (per a_blockmajor_idx packing).
    //   row-major:   warp_rank * xtileM * tileK  (per kernel's m*tileK + k).
  #ifdef WGMMA_RMAJOR_A
    auto A_warp = A_smem + warp_rank * ctx::xtileM * ctx::tileK;
    constexpr uint32_t a_ldm = ctx::tileK;  // row-major A: ldm = tileK elements
  #else
    auto A_warp = A_smem + warp_rank * ctx::a_warp_elems;
    constexpr uint32_t a_ldm = 0;           // block-major A
  #endif
    auto desc_a = vt::vx_make_smem_desc(A_warp, a_ldm * sizeof(ctx::input_t));
    ctx::umma_sync(desc_a, desc_b, handle);
  #endif

    __syncthreads();
  }

  // Epilogue: TMEM -> global memory.
  ctx::store_output(handle, pC, tile_row, tile_col, N);
  __syncthreads();

  vt::vx_tmem_dealloc(handle);
}
