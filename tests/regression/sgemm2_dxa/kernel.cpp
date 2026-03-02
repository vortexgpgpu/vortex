#include <vx_spawn.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

#include "common.h"

// DXA descriptor slots (programmed by host in main.cpp).
constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

// Accumulate partial sum from shared memory tiles.
// shA layout: [tile_size × chunk_k], row-major (row = l_row, col = k)
// shB layout: [chunk_k × tile_size], row-major (row = k, col = l_col)
static inline void gemm_accumulate(TYPE& sum, const TYPE* shA, const TYPE* shB, uint32_t tile_size,
                                   uint32_t chunk_k, uint32_t l_row, uint32_t l_col) {
  for (uint32_t k = 0; k < chunk_k; ++k) {
    sum += shA[l_row * chunk_k + k] * shB[k * tile_size + l_col];
  }
}

void kernel_body(kernel_arg_t* arg) {
  auto C = reinterpret_cast<TYPE*>(arg->C_addr);

  const uint32_t size      = arg->size;
  const uint32_t tile_size = arg->tile_size;
  const uint32_t chunk_k   = arg->chunk_k;
  const uint32_t mode      = arg->mode;

  // Global and local thread coordinates (same as sgemm2 baseline).
  const uint32_t row_base = blockIdx.x * tile_size;
  const uint32_t col_base = blockIdx.y * tile_size;
  const uint32_t l_row    = threadIdx.x;
  const uint32_t l_col    = threadIdx.y;
  const uint32_t g_row    = row_base + l_row;
  const uint32_t g_col    = col_base + l_col;

  TYPE sum(0);

  // SMEM tile layout: A tile is [tile_size x chunk_k], B tile is [chunk_k x tile_size].
  const uint32_t tile_elems_a = tile_size * chunk_k;
  const uint32_t tile_elems_b = chunk_k * tile_size;
  const uint32_t stage_elems  = tile_elems_a + tile_elems_b;
  const uint32_t stage_count  = (mode == 2) ? 2u : 1u;

  // Allocate shared memory for tile buffers.
  auto local_ptr = __local_mem(stage_count * stage_elems * sizeof(TYPE));
  auto shmem = reinterpret_cast<TYPE*>(local_ptr);

  // Stage 0 and stage 1 tile pointers.
  TYPE* shA[2] = { shmem, shmem + stage_elems };
  TYPE* shB[2] = { shmem + tile_elems_a, shmem + stage_elems + tile_elems_a };

  // Transaction barriers: each barrier independently tracks one pipeline
  // stage's DXA completion + CTA synchronization. Double-buffer uses 2
  // barriers (one per stage); single-buffer uses only bar[0].
  vortex::barrier bar[2] = { vortex::barrier(0), vortex::barrier(1) };

  // Only the first hardware warp (warp 0 within the CTA) issues DXA commands.
  // group_warp_id = vx_warp_id() % __warps_per_group identifies the warp's
  // position within its CTA (0 = first warp).  __local_group_id is the CTA
  // group index within the core, NOT the warp-within-CTA index.
  const bool is_dxa_warp    = (__warps_per_group == 0) ? false : (vx_warp_id() % __warps_per_group == 0);

  if (mode == 2) {
    // ── Double-buffered pipeline ──────────────────────────────────────
    // Two barriers: bar[0] tracks buffer 0, bar[1] tracks buffer 1.
    // DXA issue registers on bar[stage], warps arrive_and_wait on bar[cur].
    uint32_t cur = 0;

    // Prologue: issue DXA copy for first tiles into buffer 0, on bar[0].
    if (is_dxa_warp) {
      vx_dxa_issue_2d_wg(kDescA, bar[0].id(), (uint32_t)(uintptr_t)shA[0], 0, row_base);
      vx_dxa_issue_2d_wg(kDescB, bar[0].id(), (uint32_t)(uintptr_t)shB[0], col_base, 0);
    }

    // K-loop: issue next on bar[nxt] → wait current on bar[cur] → compute.
    for (uint32_t k_base = 0; k_base < size; k_base += chunk_k) {
      const uint32_t next_k   = k_base + chunk_k;
      const bool     has_next = (next_k < size);
      const uint32_t nxt      = cur ^ 1u;

      // (1) Issue DXA for next iteration's tiles on bar[nxt].
      if (has_next && is_dxa_warp) {
        vx_dxa_issue_2d_wg(kDescA, bar[nxt].id(), (uint32_t)(uintptr_t)shA[nxt], next_k, row_base);
        vx_dxa_issue_2d_wg(kDescB, bar[nxt].id(), (uint32_t)(uintptr_t)shB[nxt], col_base, next_k);
      }

      // (2) Wait for current tiles on bar[cur] (DXA completion + CTA sync).
      bar[cur].arrive_and_wait();

      // (3) Compute on current tiles.
      gemm_accumulate(sum, shA[cur], shB[cur], tile_size, chunk_k, l_row, l_col);

      bar[cur].arrive_and_wait();
      
      cur = nxt;
      
    }
  } else {
    // ── Single-buffered: full-K in one shot ───────────────────────────
    // DXA fetches the entire A and B tiles (tile_size × size) at once.
    if (is_dxa_warp) {
      vx_dxa_issue_2d_wg(kDescA, bar[0].id(), (uint32_t)(uintptr_t)shA[0], 0, row_base);
      vx_dxa_issue_2d_wg(kDescB, bar[0].id(), (uint32_t)(uintptr_t)shB[0], col_base, 0);
    }
    bar[0].arrive_and_wait();

    // Accumulate over the full K dimension.
    gemm_accumulate(sum, shA[0], shB[0], tile_size, chunk_k, l_row, l_col);

    bar[0].arrive_and_wait();
  }

  // Store result to global memory.
  C[g_row * size + g_col] = sum;
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim,
                          (vx_kernel_func_cb)kernel_body, arg);
}
