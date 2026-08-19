// Persistent-CTA variant of dxa_copy: a fixed launch grid of CTAs iterates
// over all tiles of the tensor (grid-stride loop), the deployment shape used
// by persistent-scheduler GPU kernels. One tile in flight per CTA, plain
// fixed-table barrier, single LMEM buffer — deliberately no double buffering
// so the comparison against per-tile CTAs isolates launch/retire and
// prologue amortization only.
#include <vx_spawn2.h>
#include <vx_intrinsics.h>

#include "common.h"

#ifdef VX_CFG_EXT_DXA_ENABLE
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
  const uint32_t total_groups = ({
    uint32_t n = 1;
    for (uint32_t d = 0; d < ndim; ++d) n *= arg->grids[d];
    n;
  });

  auto shmem = reinterpret_cast<TYPE*>(__local_mem());
  const uint32_t tid = threadIdx.x;
  auto dst = reinterpret_cast<TYPE*>(arg->dst_addr);

  // Grid-stride loop over tiles: CTA b owns tiles b, b+G, b+2G, ...
  for (uint32_t grp = blockIdx.x; grp < total_groups; grp += gridDim.x) {
    // Decompose flat group id into per-dimension element coords.
    uint32_t coords[DXA_MAX_DIMS] = {};
    {
      uint32_t rem = grp;
      for (uint32_t d = 0; d < ndim; ++d) {
        uint32_t grid_d = arg->grids[d];
        coords[d] = (rem % grid_d) * arg->tiles[d];
        rem /= grid_d;
      }
    }

#ifdef VX_CFG_EXT_DXA_ENABLE
    // ── DXA path: issue N-D tile copy, barrier wait ──
    vortex::barrier bar(0);
    const bool is_dxa_warp = (get_sub_group_id() == 0);
    if (is_dxa_warp) {
      bar.expect_tx(1);
      switch (ndim) {
      case 1:
        vx_dxa_issue_1d_wg(kDescSrc, bar.id(), shmem, coords[0]);
        break;
      case 2:
        vx_dxa_issue_2d_wg(kDescSrc, bar.id(), shmem, coords[0], coords[1]);
        break;
      case 3:
        vx_dxa_issue_3d_wg(kDescSrc, bar.id(), shmem, coords[0], coords[1],
                           coords[2]);
        break;
      case 4:
        vx_dxa_issue_4d_wg(kDescSrc, bar.id(), shmem, coords[0], coords[1],
                           coords[2], coords[3]);
        break;
      case 5:
        vx_dxa_issue_5d_wg(kDescSrc, bar.id(), shmem, coords[0], coords[1],
                           coords[2], coords[3], coords[4]);
        break;
      }
    }
    bar.arrive_and_wait();
#else
    // ── LSU path: block-stride loop, each thread loads its share ──
    auto src = reinterpret_cast<const TYPE*>(arg->src_addr);
    for (uint32_t e = tid; e < num_elems; e += blockDim.x) {
      uint32_t local[DXA_MAX_DIMS] = {};
      uint32_t rem = e;
      for (uint32_t d = 0; d < ndim; ++d) {
        local[d] = rem % arg->tiles[d];
        rem /= arg->tiles[d];
      }
      uint32_t gidx = 0;
      uint32_t stride = 1;
      for (uint32_t d = 0; d < ndim; ++d) {
        gidx += (coords[d] + local[d]) * stride;
        stride *= arg->sizes[d];
      }
      shmem[e] = src[gidx];
    }
    __syncthreads();
#endif

    // Write the tile back so the host can verify (block-stride loop so
    // tiles larger than the thread block work in both paths).
    for (uint32_t e = tid; e < num_elems; e += blockDim.x) {
      uint32_t local[DXA_MAX_DIMS] = {};
      uint32_t rem = e;
      for (uint32_t d = 0; d < ndim; ++d) {
        local[d] = rem % arg->tiles[d];
        rem /= arg->tiles[d];
      }
      uint32_t gidx = 0;
      uint32_t stride = 1;
      for (uint32_t d = 0; d < ndim; ++d) {
        gidx += (coords[d] + local[d]) * stride;
        stride *= arg->sizes[d];
      }
      dst[gidx] = shmem[e];
    }

    // Single-buffer discipline: the whole CTA must finish reading the LMEM
    // tile before the next iteration overwrites it (DXA write vs. store
    // read-back race otherwise). Applied to both paths symmetrically.
    __syncthreads();
  }
}
