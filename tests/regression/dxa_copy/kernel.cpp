#include <vx_spawn2.h>
#include <vx_intrinsics.h>

#include "common.h"

#ifdef VX_CFG_EXT_DXA_ENABLE
#include <vx_dxa.h>
#include <vx_barrier.h>

// DXA descriptor slot (programmed by host).
constexpr uint32_t kDescSrc = 0;
#endif

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
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

  auto lmem = reinterpret_cast<uint32_t*>(__local_mem());
  auto shmem = reinterpret_cast<TYPE*>(lmem);

#ifdef VX_CFG_EXT_DXA_ENABLE
  // ── DXA path: issue N-D tile copy, barrier wait ──
  vortex::barrier bar(0);
  auto soft_state = reinterpret_cast<vortex::smem_barrier_state*>(lmem);
  vortex::smem_barrier soft_bar(soft_state);
  if (arg->use_softbar) {
    shmem = reinterpret_cast<TYPE*>(lmem + 16);
    soft_bar.init();
    __syncthreads();
  }
  const bool is_dxa_warp = (get_sub_group_id() == 0);
  const bool do_timing = (arg->results_addr != 0);
  uint64_t t0 = 0;
  uint64_t t1 = 0;
  uint64_t t2 = 0;
  if (do_timing) {
    t0 = vx_rdcycle_sync();
  }
  if (is_dxa_warp) {
    uint32_t bar_id = bar.id();
    if (arg->use_softbar) {
      soft_bar.expect_tx(1);
      bar_id = soft_bar.id();
    } else {
      bar.expect_tx(1);
    }
    if (do_timing) {
      t1 = vx_rdcycle_sync();
    }
    switch (ndim) {
    case 1:
      vx_dxa_issue_1d_wg(kDescSrc, bar_id, shmem, coords[0]);
      break;
    case 2:
      vx_dxa_issue_2d_wg(kDescSrc, bar_id, shmem, coords[0], coords[1]);
      break;
    case 3:
      vx_dxa_issue_3d_wg(kDescSrc, bar_id, shmem, coords[0], coords[1], coords[2]);
      break;
    case 4:
      vx_dxa_issue_4d_wg(kDescSrc, bar_id, shmem,
        coords[0], coords[1], coords[2], coords[3]);
      break;
    case 5:
      vx_dxa_issue_5d_wg(kDescSrc, bar_id, shmem,
        coords[0], coords[1], coords[2], coords[3], coords[4]);
      break;
    }
    if (do_timing) {
      t2 = vx_rdcycle_sync();
    }
  }

  uint32_t spin = 0;
  uint64_t t3 = 0;
  if (do_timing) {
    if (arg->use_softbar) {
      uint32_t phase = soft_bar.arrive();
      spin = soft_bar.wait(phase);
    } else {
      uint32_t phase = bar.arrive();
      bar.wait(phase);
    }
    t3 = vx_rdcycle_sync();

    if (is_dxa_warp && blockIdx.x == 0) {
      uint32_t active = (uint32_t)vx_active_threads();
      vx_tmc_one();
      auto results = reinterpret_cast<dxa_barrier_result_t*>(arg->results_addr);
      auto shmem_words = reinterpret_cast<volatile uint32_t*>(shmem);
      uint32_t checksum = 0;
      for (uint32_t i = 0; i < num_elems; i += blockDim.x) {
        checksum ^= shmem_words[i];
      }
      dxa_barrier_result_t result = {};
      result.register_cycles = static_cast<uint32_t>(t1 - t0);
      result.issue_cycles = static_cast<uint32_t>(t2 - t1);
      result.release_cycles = static_cast<uint32_t>(t3 - t0);
      result.wait_iters = spin;
      result.checksum = checksum;
      results[0] = result;
      vx_tmc(active);
    }
  } else {
    if (arg->use_softbar) {
      soft_bar.arrive_and_wait();
    } else {
      bar.arrive_and_wait();
    }
  }

#else
  // LSU path: threads cooperatively cover the whole tile. This keeps the copy
  // benchmark usable when tile_elems exceeds one warp.
  auto src = reinterpret_cast<const TYPE*>(arg->src_addr);
  if (ndim == 2) {
    const uint32_t tile_cols = arg->tiles[0];
    const uint32_t tile_rows = arg->tiles[1];
    const uint32_t src_cols = arg->sizes[0];
    if (tile_cols <= blockDim.x) {
      const uint32_t col = threadIdx.x;
      if (col < tile_cols) {
        for (uint32_t row = 0; row < tile_rows; ++row) {
          shmem[row * tile_cols + col] =
            src[(coords[1] + row) * src_cols + coords[0] + col];
        }
      }
    } else {
      for (uint32_t row = 0; row < tile_rows; ++row) {
        const uint32_t src_base = (coords[1] + row) * src_cols + coords[0];
        const uint32_t dst_base = row * tile_cols;
        for (uint32_t col = threadIdx.x; col < tile_cols; col += blockDim.x) {
          shmem[dst_base + col] = src[src_base + col];
        }
      }
    }
  } else {
    for (uint32_t tid = threadIdx.x; tid < num_elems; tid += blockDim.x) {
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
  }
  __syncthreads();
#endif
  (void)num_elems;
}
