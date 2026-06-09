// dxa_copy_mcast — Intra-core DXA multicast copy test.
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

  // Naming:
  //   local_bar — per-CTA bar (vortex::barrier); receives MY multicast event.
  //   group_bar — local-group-shared bar (vortex::group_barrier); K members
  //               rendezvous here so every receiver's expect_tx is visible
  //               before rank-0 fires the multicast.
  vortex::barrier        local_bar(0);
  vortex::group_barrier  group_bar(1, num_recv);
  const bool is_loader_warp = (get_sub_group_id() == 0);

  if (is_loader_warp) {
    // Helper bundles expect_tx + sync_and_issue so the mask <-> expect_tx
    // invariant cannot be violated by hand. Mask = (1<<num_recv) - 1.
    vortex::dxa_multicast_2d mc(kDescSrc, num_recv, local_bar, group_bar);
    mc.sync_and_issue(shmem, /*coord0=*/0, /*coord1=*/0);
  }
  // ALL warps of THIS CTA wait for the multicast release to land in LMEM.
  local_bar.arrive_and_wait();

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
