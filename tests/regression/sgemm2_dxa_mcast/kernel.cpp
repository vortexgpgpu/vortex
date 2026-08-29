// sgemm2_dxa_mcast — Intra-core DXA multicast variant of sgemm2_dxa.
//
// Layout:
//   - `mc_group_size` (= VX_CFG_NUM_WARPS per core) CTAs co-resident on one core.
//   - Each CTA is single-warp (block_dim = tile_size × 1, exactly VX_CFG_NUM_THREADS).
//   - Each CTA computes one row strip of C: (1 × tile_size) elements.
//   - CTAs in the same column-block share the same B tile (chunk_k × tile_size)
//     via intra-core multicast. A is per-CTA (one row per CTA).
//
// Multicast pattern:
//   - local_A — per-CTA bar for A (each CTA fetches its own row, no peers).
//   - local_B — per-CTA bar for B (receives the multicast release).
//   - group_B — local-group bar across the mc_group; the
//               vortex::dxa_multicast_2d helper rendezvouses on it before
//               rank-0 fires the multicast.

#include <vx_spawn2.h>
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
  // (= one local group) and share one B column-block via multicast.
  const uint32_t row      = blockIdx.x;
  const uint32_t col_blk  = blockIdx.y;

  // SMEM tile layout: A row [chunk_k] then B column-block [chunk_k × tile_size].
  auto shmem = reinterpret_cast<TYPE*>(__local_mem());
  TYPE* shA = shmem;
  TYPE* shB = shmem + chunk_k;

  vortex::barrier       local_A(0);
  vortex::barrier       local_B(1);
  vortex::group_barrier group_B(2, mc_group_size);
  const bool is_loader_warp = (get_sub_group_id() == 0);

  if (is_loader_warp) {
    local_A.expect_tx(1);                         // A is per-CTA (no helper)

    // dxa_multicast_2d ctor performs local_B.expect_tx(1) and pre-computes
    // the full-local-group mask. sync_and_issue() rendezvouses K members
    // on group_B then (only on rank-0) fires the B multicast.
    vortex::dxa_multicast_2d mc_B(kDescB, mc_group_size, local_B, group_B);

    // Each CTA fetches its own A row (per-CTA, not multicast).
    vx_dxa_issue_2d_wg(kDescA, local_A.id(), shA, /*col=*/0, /*row=*/row);

    // K-way rendezvous + rank-0 fires the B multicast.
    mc_B.sync_and_issue(shB, /*coord0=*/col_blk * tile_size, /*coord1=*/0);
  }

  // Wait for A (per-CTA) and B (multicast) to land in SMEM. All warps of
  // this CTA participate so the per-CTA arrival count reaches W.
  local_A.arrive_and_wait();
  local_B.arrive_and_wait();

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
