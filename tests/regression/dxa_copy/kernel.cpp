#include <vx_spawn2.h>
#include <vx_intrinsics.h>

#include "common.h"

#if defined(EXT_DXA_ENABLE) && !defined(LSU_ONLY)
#define USE_DXA_PATH 1
#include <vx_dxa.h>
#include <vx_barrier.h>

// DXA descriptor slot (programmed by host).
constexpr uint32_t kDescSrc = 0;
#endif

__kernel void kernel_main(kernel_arg_t* arg) {
  const uint32_t ndim = arg->ndim;
  const uint32_t num_elems = ({
    uint32_t n = 1;
    for (uint32_t d = 0; d < ndim; ++d) n *= arg->tiles[d];
    n;
  });

  // Decompose flat blockIdx.x into per-dimension block coords.
  uint32_t coords[DXA_MAX_DIMS] = {};
  {
    uint32_t rem = blockIdx.x;
    for (uint32_t d = 0; d < ndim; ++d) {
      uint32_t grid_d = arg->grids[d];
      coords[d] = (rem % grid_d) * arg->tiles[d]; // element coord = block_coord * tile_size
      rem /= grid_d;
    }
  }

  // Allocate shared memory for one tile.
  auto shmem = reinterpret_cast<TYPE*>(__local_mem());

#ifdef USE_DXA_PATH
  // ── DXA path: issue N-D tile copy, barrier wait ──
  vortex::barrier bar(0);
  const bool is_dxa_warp = (get_sub_group_id() == 0);
  if (is_dxa_warp) {
    bar.arrive_tx(1);  // Pre-register 1 pending DXA transaction
    switch (ndim) {
    case 1:
      vx_dxa_issue_1d_wg(kDescSrc, bar.id(), shmem, coords[0]);
      break;
    case 2:
      vx_dxa_issue_2d_wg(kDescSrc, bar.id(), shmem, coords[0], coords[1]);
      break;
    case 3:
      vx_dxa_issue_3d_wg(kDescSrc, bar.id(), shmem, coords[0], coords[1], coords[2]);
      break;
    case 4:
      vx_dxa_issue_4d_wg(kDescSrc, bar.id(), shmem,
        coords[0], coords[1], coords[2], coords[3]);
      break;
    case 5:
      vx_dxa_issue_5d_wg(kDescSrc, bar.id(), shmem,
        coords[0], coords[1], coords[2], coords[3], coords[4]);
      break;
    }
  }
  bar.arrive_and_wait();

#else
  auto src = reinterpret_cast<const TYPE*>(arg->src_addr);
#ifdef LSU_KLOOP
  // ── 2D LSU k-loop path: block_dim = block_x cols; loop over tile_rows ──
  // Allows tile_rows × tile_cols >> block_dim. Used for the LSU-vs-DXA paper
  // sweep where block_dim is sized by NT, independent of tile area.
  if (ndim == 2) {
    const uint32_t tile_cols_d = arg->tiles[0];
    const uint32_t tile_rows_d = arg->tiles[1];
    const uint32_t block_w = blockDim.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t s0 = arg->sizes[0];
    for (uint32_t r = 0; r < tile_rows_d; ++r) {
      for (uint32_t c = tid; c < tile_cols_d; c += block_w) {
        uint32_t g_row = coords[1] + r;
        uint32_t g_col = coords[0] + c;
        shmem[r * tile_cols_d + c] = src[g_row * s0 + g_col];
      }
    }
    __syncthreads();
  } else
#endif
  {
    // ── Legacy LSU path: each thread loads one element, syncthreads ──
    const uint32_t tid = threadIdx.x;
    if (tid < num_elems) {
      // Decompose tid into per-dim local offsets and compute global linear index.
      uint32_t local[DXA_MAX_DIMS] = {};
      uint32_t rem = tid;
      for (uint32_t d = 0; d < ndim; ++d) {
        local[d] = rem % arg->tiles[d];
        rem /= arg->tiles[d];
      }
      // Global linear index in row-major order.
      uint32_t gidx = 0;
      uint32_t stride = 1;
      for (uint32_t d = 0; d < ndim; ++d) {
        gidx += (coords[d] + local[d]) * stride;
        stride *= arg->sizes[d];
      }
      shmem[tid] = src[gidx];
    }
    __syncthreads();
  }
#endif
  (void)num_elems;
}
