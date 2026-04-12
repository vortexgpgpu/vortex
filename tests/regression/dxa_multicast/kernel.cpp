#include <vx_spawn2.h>
#include <vx_intrinsics.h>

#include "common.h"

#include <vx_dxa.h>
#include <vx_barrier.h>

constexpr uint32_t kDescSrc = 0;

__kernel void kernel_main(kernel_arg_t* arg) {
  const uint32_t cta_id = get_local_group_id();

  // Early exit for inactive CTAs (active_ctas == 0 means all active)
  const uint32_t active = arg->active_ctas;
  if (active != 0 && cta_id >= active)
    return;

  const uint32_t tile_rows = arg->tile_rows;
  const uint32_t tile_cols = arg->tile_cols;

  // All CTAs fetch the SAME tile (same coordinates) — this is the multicast scenario.
  const uint32_t row_base = 0;
  const uint32_t col_base = 0;

  auto shmem = reinterpret_cast<TYPE*>(__local_mem());

  vortex::barrier bar(0);
  const bool is_dxa_warp = (get_sub_group_id() == 0);

#ifdef EXT_DXA_ENABLE
  // ── Multicast path: CTA 0 issues one DXA copy, data replayed to all CTAs ──
  if (is_dxa_warp) {
    if (cta_id == 0) {
      uint32_t cta_mask = (1u << arg->num_ctas) - 1;
      vx_dxa_issue_2d_multicast_wg(kDescSrc, bar.id(), shmem,
                                    col_base, row_base, cta_mask);
    }
  }
  bar.arrive_and_wait();

#else
  // ── LSU path: each thread loads elements, k-iteration for tall tiles ──
  auto src = reinterpret_cast<const TYPE*>(arg->src_addr);
  const uint32_t ncols = arg->ncols;
  const uint32_t l_col = threadIdx.x;
  const uint32_t l_row = threadIdx.y;
  const uint32_t block_rows = blockDim.y;
  for (uint32_t k = 0; k < tile_rows; k += block_rows) {
    uint32_t local_r = k + l_row;
    if (local_r < tile_rows) {
      uint32_t g_row = row_base + local_r;
      uint32_t g_col = col_base + l_col;
      shmem[local_r * tile_cols + l_col] = src[g_row * ncols + g_col];
    }
  }
  __syncthreads();
#endif

  // ── Store-back for verification (disabled for pure copy measurement) ──
#ifdef VERIFY_WRITEBACK
  {
    auto dst = reinterpret_cast<TYPE*>(arg->dst_addr);
    const uint32_t ncols_out = arg->ncols;
    const uint32_t dst_row_base = cta_id * tile_rows;
    const uint32_t l_col2 = threadIdx.x;
    const uint32_t l_row2 = threadIdx.y;
    const uint32_t block_rows2 = blockDim.y;
    for (uint32_t k = 0; k < tile_rows; k += block_rows2) {
      uint32_t local_r = k + l_row2;
      if (local_r < tile_rows) {
        dst[(dst_row_base + local_r) * ncols_out + l_col2] = shmem[local_r * tile_cols + l_col2];
      }
    }
  }
#endif

  (void)arg->nrows;
}
