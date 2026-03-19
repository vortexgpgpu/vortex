#include <vx_spawn.h>
#include <vx_intrinsics.h>

#include "common.h"

#ifdef USE_DXA
#include <vx_dxa.h>
#include <vx_barrier.h>

// DXA descriptor slot (programmed by host).
constexpr uint32_t kDescSrc = 0;
#endif

void kernel_body(kernel_arg_t* arg) {
  const uint32_t tile_rows = arg->tile_rows;
  const uint32_t tile_cols = arg->tile_cols;
  const uint32_t ncols     = arg->ncols;
  const uint32_t num_elems = tile_rows * tile_cols;

  // CTA tile origin in the global array.
  const uint32_t row_base = blockIdx.y * tile_rows;
  const uint32_t col_base = blockIdx.x * tile_cols;

  // Allocate shared memory for one tile.
  auto shmem = reinterpret_cast<TYPE*>(__local_mem(num_elems * sizeof(TYPE)));

#ifdef USE_DXA
  // ── DXA path: issue 2D tile copy, barrier wait ──
  vortex::barrier bar(0);
  const bool is_dxa_warp = (__warps_per_group == 0) ? false : ((vx_warp_id() & (__warps_per_group - 1)) == 0);
  if (is_dxa_warp) {
    vx_dxa_issue_2d_wg(kDescSrc, bar.id(),
                        (uint32_t)(uintptr_t)shmem,
                        col_base, row_base);
  }
  bar.arrive_and_wait();

#else
  // ── LSU path: each thread loads one element, syncthreads ──
  auto src = reinterpret_cast<const TYPE*>(arg->src_addr);
  const uint32_t l_col = threadIdx.x;
  const uint32_t l_row = threadIdx.y;
  const uint32_t g_row = row_base + l_row;
  const uint32_t g_col = col_base + l_col;

  shmem[l_row * tile_cols + l_col] = src[g_row * ncols + g_col];
  __syncthreads();
#endif
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim,
                          (vx_kernel_func_cb)kernel_body, arg);
}
