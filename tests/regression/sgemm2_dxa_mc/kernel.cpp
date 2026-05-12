// sgemm2_dxa_mc — Inter-core DXA multicast variant of sgemm2_dxa.
//
// Layout:
//   - `mc_group_size` (= NUM_CORES) CTAs, one per core.
//   - Each CTA is full-sized (tile_size × tile_size threads, multi-warp).
//   - CTAs at the same blockIdx.y (column-block) form an inter-core multicast
//     group sharing the same B column block. A is per-CTA (one row block per CTA).
//
// Multicast pattern (two-barrier idiom):
//   - bar_A    per-CTA local event bar for A.
//   - gbar_B   per-core gbar event slot — receives the multicast release.
//   - sync_B   cross-core sync gbar; every participating core arrives here
//              after priming gbar_B, so core-0 can only fire the multicast
//              when every receiver's gbar_B is set up.
// The global-bar handle in vx_dxa_issue_2d_multicast_wg selects the gbar
// release path; core_mask names participating cores for the cross-core
// SMEM fabric.

#include <vx_spawn2.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include <vx_intrinsics.h>

#include "common.h"

constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

__kernel void kernel_main(kernel_arg_t* arg) {
  auto C = reinterpret_cast<TYPE*>(arg->C_addr);

  const uint32_t size           = arg->size;
  const uint32_t tile_size      = arg->tile_size;
  const uint32_t chunk_k        = arg->chunk_k;
  const uint32_t mc_group_size  = arg->mc_group_size;

  // CTA tile position.
  const uint32_t row_base = blockIdx.x * tile_size;
  const uint32_t col_base = blockIdx.y * tile_size;
  const uint32_t l_row    = threadIdx.x;
  const uint32_t l_col    = threadIdx.y;
  const uint32_t g_row    = row_base + l_row;
  const uint32_t g_col    = col_base + l_col;

  // Core position within the multicast group.
  // The KMU dispatches CTAs in grid order across cores round-robin;
  // mc_rank == this CTA's core index within the multicast group.
  const uint32_t my_core = vx_core_id();
  const uint32_t mc_rank = my_core % mc_group_size;

  const uint32_t tile_elems_a = tile_size * chunk_k;
  const uint32_t tile_elems_b = chunk_k * tile_size;
  auto shmem = reinterpret_cast<TYPE*>(__local_mem());
  TYPE* shA = shmem;
  TYPE* shB = shmem + tile_elems_a;

  // Barriers:
  //   bar_A    — per-CTA local event bar for A (each CTA fetches its own).
  //   gbar_B   — per-core gbar event slot; receives B multicast release.
  //   sync_B   — cross-core sync gbar; guarantees every receiver core has
  //              primed its gbar_B.expect_tx before core-0 fires multicast.
  vortex::barrier  bar_A (0);
  vortex::gbarrier gbar_B(1);
  vortex::gbarrier sync_B(2);
  const bool is_dxa_warp = (get_sub_group_id() == 0);

  if (is_dxa_warp) {
    bar_A.expect_tx(1);              // local A event
    gbar_B.expect_tx(1);             // global B event (per-core slot)
    sync_B.arrive_and_wait();        // sync across cores

    // Each CTA fetches its own A tile (no multicast).
    vx_dxa_issue_2d_wg(kDescA, bar_A.id(), shA, /*col=*/0, /*row=*/row_base);

    // Rank-0 core issues inter-core multicast. The global-bar handle selects
    // the gbar release path; core_mask names participating cores.
    if (mc_rank == 0) {
      const uint32_t core_mask = (1u << mc_group_size) - 1;
      vx_dxa_issue_2d_multicast_wg(kDescB, gbar_B.id(), shB,
                                    /*col=*/col_base, /*row=*/0,
                                    core_mask);
    }
  }

  // Wait for A (local) and B (global) to land.
  bar_A.arrive_and_wait();
  gbar_B.arrive_and_wait();

  // ── Compute: tile_size × tile_size per CTA ────────────────────────────────
  TYPE sum(0);
  for (uint32_t k = 0; k < chunk_k; ++k) {
    sum += shA[l_row * chunk_k + k] * shB[k * tile_size + l_col];
  }

  C[g_row * size + g_col] = sum;

  (void)arg->A_addr; (void)arg->B_addr;
}
