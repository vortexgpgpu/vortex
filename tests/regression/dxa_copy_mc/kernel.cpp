// dxa_copy_mc — Inter-core DXA multicast copy test.
//
// One CTA per core (`num_recv` cores total). Each core's CTA receives the same
// tile via DXA multicast issued by core 0. Sync across cores uses the global
// barrier (gbar); the cross-core LMEM fabric routes the SMEM payload per
// receiver core.

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

#include "common.h"

constexpr uint32_t kDescSrc = 0;

__kernel void kernel_main(kernel_arg_t* arg) {
  const uint32_t tile_rows = arg->tile_rows;
  const uint32_t tile_cols = arg->tile_cols;
  const uint32_t num_recv  = arg->num_recv;
  const uint32_t my_core   = vx_core_id();

  auto shmem = reinterpret_cast<TYPE*>(__local_mem());

  // Per-core event gbar: receives the multicast release on each core's slot.
  vortex::gbarrier evt_gbar (0);
  // Cross-core sync gbar: gbar IDs alias across cores by construction, so all
  // receiver cores rendezvous on this slot before the issuer fires multicast.
  vortex::gbarrier sync_gbar(1);
  const bool is_dxa_warp = (get_sub_group_id() == 0);

  if (is_dxa_warp) {
    evt_gbar.expect_tx(1);          // register multicast event on my gbar slot
    sync_gbar.arrive_and_wait();    // sync across cores
    if (my_core == 0) {
      const uint32_t core_mask = (1u << num_recv) - 1;
      vx_dxa_issue_2d_multicast_wg(kDescSrc, evt_gbar.id(), shmem,
                                    /*col=*/0, /*row=*/0, core_mask);
    }
  }
  evt_gbar.arrive_and_wait();       // wait for my DXA release

  // Write the received tile to a distinct dst region for verification.
  // Layout: dst[my_core * tile_elems + r * tile_cols + c]
  auto dst = reinterpret_cast<TYPE*>(arg->dst_addr);
  const uint32_t tile_elems = tile_rows * tile_cols;
  const uint32_t l_col = threadIdx.x;
  const uint32_t l_row = threadIdx.y;
  const uint32_t block_rows = blockDim.y;
  for (uint32_t r = l_row; r < tile_rows; r += block_rows) {
    if (l_col < tile_cols) {
      dst[my_core * tile_elems + r * tile_cols + l_col] =
          shmem[r * tile_cols + l_col];
    }
  }

  (void)arg->src_addr;
  (void)arg->src_row_stride;
}
