// sgemm_tcu_wg_dxa_mcast — Intra-core DXA multicast variant of sgemm_tcu_wg_dxa.
//
// WGMMA pattern: each CTA uses (num_warps_per_cta) warps in a warp group.
// `mc_group_size` such CTAs are co-resident on one core, sharing the same
// B tile via intra-core DXA multicast.
//
// Constraint: mc_group_size * warps_per_cta ≤ NUM_WARPS_per_core
// (e.g., VX_CFG_NUM_WARPS=16 + 4-warp CTAs → 4 co-resident CTAs).

#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;

constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  const uint32_t N = arg->N;
  const uint32_t K = arg->K;
  const uint32_t mc_group_size = arg->mc_group_size;

  const uint32_t tid          = threadIdx.x;
  const uint32_t num_threads  = blockDim.x;
  const uint32_t warp_rank    = tid / VX_CFG_NUM_THREADS;
  const uint32_t num_warps    = num_threads / VX_CFG_NUM_THREADS;

  // CTA tile geometry (same as parent).
  const uint32_t cta_M    = num_warps * ctx::xtileM;
  const uint32_t tile_row = blockIdx.y * cta_M;
  const uint32_t tile_col = blockIdx.x * ctx::xtileN;

  // CTAs at the same blockIdx.x (= same tile_col) form a multicast group
  // sharing B. The helper owns the rank-0-only issue decision internally
  // via vortex::get_local_group_id() == 0.

  auto smem   = reinterpret_cast<ctx::input_t *>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + cta_M * ctx::tileK;

  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  // Barriers:
  //   local_A — per-CTA bar for A (each CTA fetches its own slice).
  //   local_B — per-CTA bar for B (receives the multicast release).
  //   group_B — local-group bar across the mc_group; the
  //             vortex::dxa_multicast_2d helper rendezvouses on it before
  //             rank-0 fires the multicast each iteration.
  vortex::barrier       local_A(0);
  vortex::barrier       local_B(1);
  vortex::group_barrier group_B(2, mc_group_size);
  const bool is_loader_warp = (get_sub_group_id() == 0);

  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    if (is_loader_warp) {
      local_A.expect_tx(1);                        // A: per-CTA, manual

      // dxa_multicast_2d ctor does local_B.expect_tx(1) + pre-computes the
      // full-local-group mask. sync_and_issue() rendezvouses K members on
      // group_B then fires the B multicast from rank-0 only.
      vortex::dxa_multicast_2d mc_B(kDescB, mc_group_size, local_B, group_B);

      // Each CTA fetches its own A slice (per-CTA, not multicast).
      vx_dxa_issue_2d_wg(kDescA, local_A.id(), A_smem, k, tile_row);

      // K-way rendezvous + rank-0 fires the B multicast.
      mc_B.sync_and_issue(B_smem, /*coord0=*/tile_col, /*coord1=*/k);
    }

    local_A.arrive_and_wait();
    local_B.arrive_and_wait();

    // WGMMA compute.
    auto A_warp = A_smem + warp_rank * ctx::xtileM * ctx::tileK;
    auto desc_b = vt::vx_make_smem_desc(B_smem, ctx::xtileN * sizeof(ctx::input_t));

  #if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
    ctx::fragment_a fragA;
    ctx::load_matrix_sync(fragA, A_warp, ctx::tileK);
    ctx::wgmma_sync(fragC, fragA, desc_b, fragC);
  #else
    auto desc_a = vt::vx_make_smem_desc(A_warp, ctx::tileK * sizeof(ctx::input_t));
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);
  #endif

    // Sync before next iteration's DXA can overwrite SMEM.
    local_A.arrive_and_wait();
    local_B.arrive_and_wait();
  }

  auto pTileC = pC + (tile_row + warp_rank * ctx::xtileM) * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
