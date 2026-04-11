#include <vx_spawn2.h>
#include <vx_intrinsics.h>

#include "common.h"

#ifdef EXT_DXA_ENABLE
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

#ifdef EXT_DXA_ENABLE
  // ── DXA path: issue N-D tile copy, barrier wait ──
  vortex::barrier bar(0);
  const bool is_dxa_warp = (csr_read(VX_CSR_CTA_RANK) == 0);
  if (is_dxa_warp) {
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
  // ── LSU path: each thread loads one element, syncthreads ──
  auto src = reinterpret_cast<const TYPE*>(arg->src_addr);
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
#endif
  (void)num_elems;
}
