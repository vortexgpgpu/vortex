// dxa_copy_mw — Intra-core DXA multicast copy test.
//
// `num_recv` single-warp CTAs co-resident on one core all fetch the SAME tile
// from GMEM via a single DXA multicast issued by CTA 0. Each receiver then
// writes its SMEM copy to a distinct dst region for host verification.

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

#include "common.h"

constexpr uint32_t kDescSrc = 0;

__kernel void kernel_main(kernel_arg_t* arg) {
  const uint32_t cta_id    = get_local_group_id();
  const uint32_t tile_rows = arg->tile_rows;
  const uint32_t tile_cols = arg->tile_cols;
  const uint32_t num_recv  = arg->num_recv;

  auto shmem = reinterpret_cast<TYPE*>(__local_mem());

  // Two-barrier idiom:
  //   evt_bar  — per-CTA event bar (receives the multicast release).
  //   sync_bar — shared sync bar (all peer CTAs see same bar_addr); guarantees
  //              every receiver has primed evt_bar.expect_tx before CTA 0
  //              fires the multicast.
  vortex::barrier        evt_bar(0);
  vortex::shared_barrier sync_bar(1, num_recv);
  const bool is_dxa_warp = (get_sub_group_id() == 0);

  if (is_dxa_warp) {
    evt_bar.expect_tx(1);              // register the multicast event
    sync_bar.arrive_and_wait();        // wait for all peers to prime
    if (cta_id == 0) {
      const uint32_t mc_mask = (1u << num_recv) - 1;
      vx_dxa_issue_2d_multicast_wg(kDescSrc, evt_bar.id(), shmem,
                                    /*col=*/0, /*row=*/0, mc_mask);
    }
  }
  evt_bar.arrive_and_wait();           // wait for my DXA release

  // Store the received tile to dst region for host verification.
  // Layout: dst[cta_id * tile_elems + r * tile_cols + c]
  auto dst = reinterpret_cast<TYPE*>(arg->dst_addr);
  const uint32_t tile_elems = tile_rows * tile_cols;
  const uint32_t l_col = threadIdx.x;
  for (uint32_t r = 0; r < tile_rows; ++r) {
    if (l_col < tile_cols) {
      dst[cta_id * tile_elems + r * tile_cols + l_col] =
          shmem[r * tile_cols + l_col];
    }
  }

  (void)arg->src_addr;
  (void)arg->src_row_stride;
}
