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

static inline __attribute__((always_inline)) vortex::barrier& select_barrier(
    uint32_t index,
    vortex::barrier& bar0,
    vortex::barrier& bar1,
    vortex::barrier& bar2,
    vortex::barrier& bar3,
    vortex::barrier& bar4,
    vortex::barrier& bar5,
    vortex::barrier& bar6,
    vortex::barrier& bar7) {
  switch (index) {
  case 0: return bar0;
  case 1: return bar1;
  case 2: return bar2;
  case 3: return bar3;
  case 4: return bar4;
  case 5: return bar5;
  case 6: return bar6;
  default: return bar7;
  }
}

static inline __attribute__((always_inline)) void issue_tile(
    uint32_t mode,
    bool is_loader_warp,
    uint32_t num_recv,
    TYPE* shmem,
    uint32_t tile_elems,
    uint32_t buffer_id,
    uint32_t coord0,
    uint32_t coord1,
    vortex::barrier& local_bar) {
  if (!is_loader_warp)
    return;

  TYPE* tile_smem = shmem + buffer_id * tile_elems;
  if (mode == DXA_COPY_MCAST_MODE_PERCTA) {
    local_bar.expect_tx(1);
    vx_dxa_issue_2d_wg(kDescSrc, local_bar.id(), tile_smem, coord0, coord1);
  } else {
    vortex::dxa_multicast_2d mc(kDescSrc, num_recv, local_bar);
    mc.sync_and_issue(tile_smem, coord0, coord1);
  }
}

static inline __attribute__((always_inline)) void writeback_tile(
    uint32_t writeback_mode,
    TYPE* dst,
    TYPE* shmem,
    uint32_t buffer_id,
    uint32_t tile_elems,
    uint32_t tile_rows,
    uint32_t tile_cols,
    uint32_t src_rows,
    uint32_t src_cols,
    uint32_t tile_grid_cols,
    uint32_t cta_id,
    uint32_t tile_y,
    uint32_t tile_x) {
  TYPE* tile_smem = shmem + buffer_id * tile_elems;
  const uint32_t tid = threadIdx.x;
  const uint32_t coord0 = tile_x * tile_cols;
  const uint32_t coord1 = tile_y * tile_rows;

  if (writeback_mode == DXA_COPY_MCAST_WRITEBACK_FULL) {
    for (uint32_t idx = tid; idx < tile_elems; idx += blockDim.x) {
      const uint32_t local_r = idx / tile_cols;
      const uint32_t local_c = idx - local_r * tile_cols;
      const uint32_t dst_idx = (coord1 + local_r) * src_cols + coord0 + local_c;
      dst[cta_id * src_rows * src_cols + dst_idx] = tile_smem[idx];
    }
  } else if (writeback_mode == DXA_COPY_MCAST_WRITEBACK_SAMPLE) {
    const uint32_t sample_elems =
      tile_elems < blockDim.x ? tile_elems : blockDim.x;
    const uint32_t tile_idx = tile_y * tile_grid_cols + tile_x;
    const uint32_t dst_base =
      (cta_id * ((src_rows / tile_rows) * tile_grid_cols) + tile_idx) * sample_elems;
    for (uint32_t idx = tid; idx < sample_elems; idx += blockDim.x) {
      dst[dst_base + idx] = tile_smem[idx];
    }
  }
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t cta_id    = get_local_group_id();
  const uint32_t mode      = arg->mode;
  const uint32_t writeback_mode = arg->writeback_mode;
  const uint32_t tile_rows = arg->tile_rows;
  const uint32_t tile_cols = arg->tile_cols;
  const uint32_t src_rows  = arg->src_rows;
  const uint32_t src_cols  = arg->src_cols;
  const uint32_t num_recv  = arg->num_recv;
  uint32_t pipeline_depth = arg->pipeline_depth;
  if (pipeline_depth == 0)
    pipeline_depth = 1;
  if (pipeline_depth > DXA_COPY_MCAST_MAX_PIPELINE_DEPTH)
    pipeline_depth = DXA_COPY_MCAST_MAX_PIPELINE_DEPTH;

  auto shmem = reinterpret_cast<TYPE*>(__local_mem());

  // local_bar is per-CTA; each receiver records its own multicast event.
  vortex::barrier        local_bar(0);
  const bool is_loader_warp = (get_sub_group_id() == 0);
  const uint32_t tile_elems = tile_rows * tile_cols;
  const uint32_t tid = threadIdx.x;
  auto dst = reinterpret_cast<TYPE*>(arg->dst_addr);

  if (mode == DXA_COPY_MCAST_MODE_SMOKE) {
    if (is_loader_warp) {
      vortex::dxa_multicast_2d mc(kDescSrc, num_recv, local_bar);
      mc.sync_and_issue(shmem, /*coord0=*/0, /*coord1=*/0);
    }
    local_bar.arrive_and_wait();

    // Smoke mode keeps the original compact tile-only result layout.
    for (uint32_t idx = tid; idx < tile_elems; idx += blockDim.x) {
      dst[cta_id * tile_elems + idx] = shmem[idx];
    }
  } else {
    const uint32_t tile_grid_rows = src_rows / tile_rows;
    const uint32_t tile_grid_cols = src_cols / tile_cols;

    if (pipeline_depth <= 1) {
      for (uint32_t tile_y = 0; tile_y < tile_grid_rows; ++tile_y) {
        for (uint32_t tile_x = 0; tile_x < tile_grid_cols; ++tile_x) {
          const uint32_t coord0 = tile_x * tile_cols;
          const uint32_t coord1 = tile_y * tile_rows;
          issue_tile(mode, is_loader_warp, num_recv, shmem, tile_elems,
                     0, coord0, coord1, local_bar);
          local_bar.arrive_and_wait();
          writeback_tile(writeback_mode, dst, shmem, 0, tile_elems,
                         tile_rows, tile_cols, src_rows, src_cols,
                         tile_grid_cols, cta_id, tile_y, tile_x);
        }
      }
    } else {
      vortex::barrier bar1(1);
      vortex::barrier bar2(2);
      vortex::barrier bar3(3);
      vortex::barrier bar4(4);
      vortex::barrier bar5(5);
      vortex::barrier bar6(6);
      vortex::barrier bar7(7);
      const uint32_t tile_count = tile_grid_rows * tile_grid_cols;
      const uint32_t preload =
        pipeline_depth < tile_count ? pipeline_depth : tile_count;

      for (uint32_t i = 0; i < preload; ++i) {
        const uint32_t tile_y = i / tile_grid_cols;
        const uint32_t tile_x = i - tile_y * tile_grid_cols;
        const uint32_t coord0 = tile_x * tile_cols;
        const uint32_t coord1 = tile_y * tile_rows;
        auto& bar = select_barrier(i, local_bar, bar1, bar2, bar3,
                                   bar4, bar5, bar6, bar7);
        issue_tile(mode, is_loader_warp, num_recv, shmem, tile_elems,
                   i, coord0, coord1, bar);
      }

      for (uint32_t tile_idx = 0; tile_idx < tile_count; ++tile_idx) {
        const uint32_t buffer_id = tile_idx % pipeline_depth;
        auto& bar = select_barrier(buffer_id, local_bar, bar1, bar2, bar3,
                                   bar4, bar5, bar6, bar7);
        bar.arrive_and_wait();

        const uint32_t tile_y = tile_idx / tile_grid_cols;
        const uint32_t tile_x = tile_idx - tile_y * tile_grid_cols;
        writeback_tile(writeback_mode, dst, shmem, buffer_id, tile_elems,
                       tile_rows, tile_cols, src_rows, src_cols,
                       tile_grid_cols, cta_id, tile_y, tile_x);

        const uint32_t next = tile_idx + pipeline_depth;
        if (next < tile_count) {
          const uint32_t next_y = next / tile_grid_cols;
          const uint32_t next_x = next - next_y * tile_grid_cols;
          const uint32_t coord0 = next_x * tile_cols;
          const uint32_t coord1 = next_y * tile_rows;
          issue_tile(mode, is_loader_warp, num_recv, shmem, tile_elems,
                     buffer_id, coord0, coord1, bar);
        }
      }
    }
  }

  (void)arg->src_addr;
  (void)arg->src_row_stride;
}
