// sgemm2_dxa_mw — Intra-core DXA multicast variant of sgemm2_dxa.
//
// Layout:
//   - `mc_group_size` (= VX_CFG_NUM_WARPS per core) CTAs co-resident on one core.
//   - Each CTA is single-warp (block_dim = tile_size × 1, exactly VX_CFG_NUM_THREADS).
//   - Each CTA computes one row strip of C: (1 × tile_size) elements.
//   - CTAs in the same column-block share the same B tile (chunk_k × tile_size)
//     via intra-core multicast. A is per-CTA (one row per CTA).
//
// Multicast pattern (two-barrier idiom):
//   - bar_A   per-CTA event bar for A (each CTA fetches its own A row).
//   - bar_B   per-CTA event bar for B (multicast receiver slot).
//   - sync_B  shared sync bar across the mc_group — every CTA arrives here
//             after priming its bar_B, so rank-0 can only fire the multicast
//             when every receiver's events_r is set.

#include <vx_spawn2.h>
#include <VX_config.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

#include "common.h"

constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

__kernel void kernel_main(kernel_arg_t* arg) {
  auto C = reinterpret_cast<TYPE*>(arg->C_addr);

  const uint32_t size           = arg->size;
  const uint32_t tile_size      = arg->tile_size;
  const uint32_t chunk_k        = arg->chunk_k;
  const uint32_t mc_group_size  = arg->mc_group_size;

  // CTA position. blockIdx.x walks rows; blockIdx.y walks column-blocks.
  // 4 CTAs (mc_group_size) in adjacent blockIdx.x values are co-resident
  // and form a multicast group sharing one B column-block.
  const uint32_t row      = blockIdx.x;
  const uint32_t col_blk  = blockIdx.y;
  const uint32_t mc_rank  = row % mc_group_size;    // 0..mc_group_size-1

  // SMEM tile layout: A row [chunk_k] then B column-block [chunk_k × tile_size].
  auto shmem = reinterpret_cast<TYPE*>(__local_mem());
  TYPE* shA = shmem;
  TYPE* shB = shmem + chunk_k;

  // Barriers:
  //   bar_A  — per-CTA event bar for A (each CTA fetches its own row).
  //   bar_B  — per-CTA event bar for B (receives the multicast release).
  //   sync_B — shared sync bar across mc_group peers; guarantees every
  //            receiver has primed its bar_B.expect_tx before rank-0 fires
  //            the multicast (otherwise late receivers race the release).
  vortex::barrier         bar_A(0);
  vortex::barrier         bar_B(1);
  vortex::shared_barrier  sync_B(2, mc_group_size);
  const bool is_dxa_warp = (get_sub_group_id() == 0);

  if (is_dxa_warp) {
    bar_A.expect_tx(1);              // A event (per-CTA, no sync needed)
    bar_B.expect_tx(1);              // B event (per-CTA slot, multicast tgt)
    sync_B.arrive_and_wait();        // wait for all mc_group peers to prime B

    // Each CTA fetches its own A row.
    vx_dxa_issue_2d_wg(kDescA, bar_A.id(), shA, /*col=*/0, /*row=*/row);

    // Rank-0 issues the shared B multicast.
    if (mc_rank == 0) {
      const uint32_t mc_mask = (1u << mc_group_size) - 1;
      vx_dxa_issue_2d_multicast_wg(kDescB, bar_B.id(), shB,
                                    /*col=*/col_blk * tile_size, /*row=*/0,
                                    mc_mask);
    }
  }

  // Wait for A (per-CTA) and B (multicast) to land in SMEM.
  bar_A.arrive_and_wait();
  bar_B.arrive_and_wait();

  // ── Compute: 1 row × tile_size cols ───────────────────────────────────────
  const uint32_t l_col = threadIdx.x;  // 0..tile_size-1 (single warp)
  TYPE sum(0);
  for (uint32_t k = 0; k < chunk_k; ++k) {
    sum += shA[k] * shB[k * tile_size + l_col];
  }

  // Store C[row, col_blk*tile_size + l_col]
  C[row * size + col_blk * tile_size + l_col] = sum;

  (void)arg->A_addr; (void)arg->B_addr;  // descriptors carry addresses
}
