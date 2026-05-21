// sgemm_tcu_wg_dxa_mc — Inter-core DXA multicast variant of sgemm_tcu_wg_dxa.
//
// WGMMA pattern: each CTA uses (num_warps_per_cta) warps. `mc_group_size`
// (= VX_CFG_NUM_CORES) CTAs on distinct cores share the same B tile via inter-core
// DXA multicast routed through a global barrier.

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

  const uint32_t cta_M    = num_warps * ctx::xtileM;
  const uint32_t tile_row = blockIdx.y * cta_M;
  const uint32_t tile_col = blockIdx.x * ctx::xtileN;

  // mc_rank = this CTA's core index within the multicast group.
  const uint32_t my_core = vx_core_id();
  const uint32_t mc_rank = my_core % mc_group_size;

  auto smem   = reinterpret_cast<ctx::input_t *>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + cta_M * ctx::tileK;

  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  // Barriers:
  //   bar_A    — per-CTA local event bar for A.
  //   gbar_B   — per-core gbar event slot for B multicast receiver.
  //   sync_B   — cross-core sync gbar; primes B on every core before
  //              core-0 fires the multicast.
  vortex::barrier  bar_A (0);
  vortex::gbarrier gbar_B(1);
  vortex::gbarrier sync_B(2);
  const bool is_dxa_warp = (get_sub_group_id() == 0);

  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    if (is_dxa_warp) {
      bar_A.expect_tx(1);              // local A event
      gbar_B.expect_tx(1);             // global B event (per-core slot)
      sync_B.arrive_and_wait();        // sync across cores

      vx_dxa_issue_2d_wg(kDescA, bar_A.id(), A_smem, k, tile_row);

      if (mc_rank == 0) {
        const uint32_t core_mask = (1u << mc_group_size) - 1;
        vx_dxa_issue_2d_multicast_wg(kDescB, gbar_B.id(), B_smem,
                                      tile_col, k, core_mask);
      }
    }

    bar_A.arrive_and_wait();
    gbar_B.arrive_and_wait();

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

    bar_A.arrive_and_wait();
    gbar_B.arrive_and_wait();
  }

  auto pTileC = pC + (tile_row + warp_rank * ctx::xtileM) * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
