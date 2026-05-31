#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;

// DXA descriptor slots (programmed by host in main.cpp).
[[maybe_unused]] constexpr uint32_t kDescA = 0;
[[maybe_unused]] constexpr uint32_t kDescB = 1;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);
#ifdef SW_LOAD_B
  auto pB = reinterpret_cast<const ctx::input_t *>(arg->B_addr);
#endif
#ifdef SW_LOAD_A
  auto pA = reinterpret_cast<const ctx::input_t *>(arg->A_addr);
#endif

  uint32_t N = arg->N;
  uint32_t K = arg->K;

  uint32_t tid = threadIdx.x;
  uint32_t num_threads = blockDim.x;
  uint32_t warp_rank = tid / VX_CFG_NUM_THREADS;
  uint32_t num_warps = num_threads / VX_CFG_NUM_THREADS;

  // CTA tile dimensions
  uint32_t cta_M = num_warps * ctx::xtileM;
  uint32_t tile_row = blockIdx.y * cta_M;
  uint32_t tile_col = blockIdx.x * ctx::xtileN;

  // Shared memory layout: A [cta_M x tileK] then B [tileK x xtileN]
  auto smem   = reinterpret_cast<ctx::input_t *>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + cta_M * ctx::tileK;

  // Initialize accumulator tile to zero.
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  // Transaction barrier for DXA completion + CTA synchronization.
  vortex::barrier bar(0);

  // Only the first warp in the CTA issues DXA commands.
  const bool is_dxa_warp = (get_sub_group_id() == 0);

  // Loop over K tiles.
  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    // DXA: load A tile [tile_row .. tile_row+cta_M, k .. k+tileK] into A_smem
    // DXA: load B tile [k .. k+tileK, tile_col .. tile_col+tileN] into B_smem
    // Bisection switches: SW_LOAD_B replaces B's DXA with cooperative SW
    // load (K-major); SW_LOAD_A replaces A's. Used to isolate whether the
    // failing config's bug is in DXA-written A, DXA-written B, or neither.
    {
    #if defined(SW_LOAD_A) && defined(SW_LOAD_B)
      // both via SW — no DXA needed
    #elif defined(SW_LOAD_A)
      if (is_dxa_warp) { bar.expect_tx(1); vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem, tile_col, k); }
    #elif defined(SW_LOAD_B)
      if (is_dxa_warp) { bar.expect_tx(1); vx_dxa_issue_2d_wg(kDescA, bar.id(), A_smem, k, tile_row); }
    #else
      if (is_dxa_warp) {
        bar.expect_tx(2);  // Two pending transactions: A + B
        vx_dxa_issue_2d_wg(kDescA, bar.id(), A_smem, k, tile_row);
        vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem, tile_col, k);
      }
    #endif
    #ifdef SW_LOAD_A
      // Cooperative load A row-major (matches DXA row-major layout).
      uint32_t a_size = cta_M * ctx::tileK;
      for (uint32_t i = 0; i < a_size; i += num_threads) {
        uint32_t idx = i + tid;
        uint32_t r = idx / ctx::tileK;
        uint32_t c = idx % ctx::tileK;
        A_smem[r * ctx::tileK + c] = pA[(tile_row + r) * K + (k + c)];
      }
    #endif
    #ifdef SW_LOAD_B
      // Cooperative load B K-major (matches DXA K-major LAYOUT).
      uint32_t b_size = ctx::tileK * ctx::xtileN;
      for (uint32_t i = 0; i < b_size; i += num_threads) {
        uint32_t idx = i + tid;
        uint32_t r = idx / ctx::xtileN;
        uint32_t c = idx % ctx::xtileN;
        B_smem[c * ctx::tileK + r] = pB[(k + r) * N + (tile_col + c)];
      }
    #endif
    }

    // Wait for DXA completion (all warps participate).
    bar.arrive_and_wait();

    // Each warp's A slice starts at warp_rank * xtileM * tileK
    auto A_warp = A_smem + warp_rank * ctx::xtileM * ctx::tileK;
    // B layout in SMEM: K-major (N-outer, K-inner) — written by DXA worker
    // in scatter mode (descriptor LAYOUT bit set to K_MAJOR in main.cpp).
    // Per-N-row stride = tileK elements.
    auto desc_b = vt::vx_make_smem_desc(B_smem, ctx::tileK * sizeof(ctx::input_t));

  #if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
    // RS: A from registers, B from smem (NRC <= 16 only)
    ctx::fragment_a fragA;
    ctx::load_matrix_sync(fragA, A_warp, ctx::tileK);
    ctx::wgmma_sync(fragC, fragA, desc_b, fragC);
  #else
    // SS: both from smem
    auto desc_a = vt::vx_make_smem_desc(A_warp, ctx::tileK * sizeof(ctx::input_t));
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);
  #endif

    // Sync after WGMMA before next DXA overwrites smem.
    bar.arrive_and_wait();
  }

  // Store the computed C tile to global memory.
  auto pTileC = pC + (tile_row + warp_rank * ctx::xtileM) * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
