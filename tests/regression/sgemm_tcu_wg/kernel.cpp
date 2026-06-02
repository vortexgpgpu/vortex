#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>

namespace vt = vortex::tensor;

using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pA = reinterpret_cast<ctx::input_t *>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t *>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  uint32_t N = arg->N;
  uint32_t K = arg->K;

  uint32_t tid = threadIdx.x;
  uint32_t num_threads = blockDim.x;  // warps * VX_CFG_NUM_THREADS
  uint32_t warp_rank = tid / VX_CFG_NUM_THREADS;
  uint32_t num_warps = num_threads / VX_CFG_NUM_THREADS;

  // CTA tile dimensions
  uint32_t cta_M = num_warps * ctx::xtileM;
  uint32_t tile_row = blockIdx.y * cta_M;
  uint32_t tile_col = blockIdx.x * ctx::xtileN;

  // Shared memory layout: A [cta_M × tileK] then B [tileK × per_warp_N]
  auto smem   = reinterpret_cast<ctx::input_t *>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + cta_M * ctx::tileK;

  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    // Cooperatively load A [cta_M × tileK] into smem.
    //   Default (block-major):  A_smem[a_blockmajor_idx(r, c)].
    //   WGMMA_RMAJOR_A:         A_smem[r * tileK + c] — row-major M-outer
    //   K-inner, matching the abuf row-major fetch (ldm = tileK). This is
    //   the same layout the DXA worker produces, used to isolate whether
    //   the abuf row-major path is correct independent of the DXA writer.
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

    // Cooperatively load B into smem.
    //   Default (block-major):  B_smem[(k_blk*N_STEPS + n_blk)*BW + n_in*tcK + k_in]
    //   WGMMA_KMAJOR_B:         B_smem[n*tileK + k] — N outer, K inner; the
    //   "K-major" layout NVIDIA Hopper WGMMA SS-descriptors expect, consumed
    //   when desc_b carries a non-zero stride.
    uint32_t b_size = ctx::tileK * ctx::xtileN;
    for (uint32_t i = 0; i < b_size; i += num_threads) {
      uint32_t idx = i + tid;
      uint32_t r = idx / ctx::xtileN;
      uint32_t c = idx % ctx::xtileN;
    #ifdef WGMMA_KMAJOR_B
      B_smem[c * ctx::tileK + r] = pB[(k + r) * N + (tile_col + c)];
    #else
      B_smem[ctx::b_blockmajor_idx(r, c)] = pB[(k + r) * N + (tile_col + c)];
    #endif
    }

    __syncthreads();

    // Each warp's A slice — base differs by layout:
    //   block-major: warp_rank * a_warp_elems (per a_blockmajor_idx packing).
    //   row-major:   warp_rank * xtileM * tileK  (per kernel's m*tileK + k).
  #ifdef WGMMA_RMAJOR_A
    auto A_warp = A_smem + warp_rank * ctx::xtileM * ctx::tileK;
  #else
    auto A_warp = A_smem + warp_rank * ctx::a_warp_elems;
  #endif
  #ifdef WGMMA_KMAJOR_B
    auto desc_b = vt::vx_make_smem_desc(B_smem, ctx::tileK * sizeof(ctx::input_t));
  #else
    auto desc_b = vt::vx_make_smem_desc(B_smem, 0); // stride field unused under block-major
  #endif

  #ifdef WGMMA_RMAJOR_A
    constexpr uint32_t a_ldm = ctx::tileK;  // row-major A: ldm = tileK elements
  #else
    constexpr uint32_t a_ldm = 0;           // block-major A
  #endif
  #if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
    // RS: A from registers, B from smem (NRC <= 16 only).
    ctx::fragment_a fragA;
    ctx::load_matrix_sync(fragA, A_warp, a_ldm);
    ctx::wgmma_sync(fragC, fragA, desc_b, fragC);
  #else
    // SS: both from smem
    auto desc_a = vt::vx_make_smem_desc(A_warp, a_ldm * sizeof(ctx::input_t));
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);
  #endif

    __syncthreads();
  }

  // Store C tile using wgmma_context's n-major store
  auto out = pC + (tile_row + warp_rank * ctx::xtileM) * N + tile_col;
  ctx::store_matrix_sync(out, fragC, N);
}
