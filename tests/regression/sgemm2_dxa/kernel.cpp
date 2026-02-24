#include <vx_spawn.h>
#include <vx_dxa.h>

#include "common.h"

constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;
constexpr uint32_t kFlagsG2S = 0;

static inline uint32_t local_barrier_id(uint32_t logical_id) {
  return __local_group_id + (logical_id << 16);
}

static inline void dxa_issue_2d_leader(TYPE* smem_ptr,
                                       uint32_t desc_slot,
                                       uint32_t barrier_id,
                                       uint32_t coord0,
                                       uint32_t coord1) {
  vx_dxa_issue_5d_leader(desc_slot,
                         barrier_id,
                         (uint32_t)(uintptr_t)smem_ptr,
                         kFlagsG2S,
                         coord0,
                         coord1,
                         0,
                         0,
                         0);
}

static inline void gemm_accumulate(TYPE& sum,
                                   const TYPE* shA,
                                   const TYPE* shB,
                                   uint32_t tile_size,
                                   uint32_t chunk_k,
                                   uint32_t l_row,
                                   uint32_t l_col) {
  for (uint32_t k = 0; k < chunk_k; ++k) {
    sum += shA[l_row * chunk_k + k] * shB[k * tile_size + l_col];
  }
}

void kernel_body(kernel_arg_t* arg) {
  auto C = reinterpret_cast<TYPE*>(arg->C_addr);

  const uint32_t size = arg->size;
  const uint32_t tile_size = arg->tile_size;
  const uint32_t chunk_k = arg->chunk_k;
  const uint32_t mode = arg->mode;

  const uint32_t row_base = blockIdx.x * tile_size;
  const uint32_t col_base = blockIdx.y * tile_size;

  const uint32_t l_row = threadIdx.x;
  const uint32_t l_col = threadIdx.y;
  const uint32_t g_row = row_base + l_row;
  const uint32_t g_col = col_base + l_col;

  TYPE sum(0);

  const uint32_t tile_elems_a = tile_size * chunk_k;
  const uint32_t tile_elems_b = chunk_k * tile_size;
  const uint32_t stage_elems = tile_elems_a + tile_elems_b;
  const uint32_t stage_count = (mode == 2) ? 2u : 1u;

  auto local_ptr = __local_mem(stage_count * stage_elems * sizeof(TYPE));
  auto shmem = reinterpret_cast<TYPE*>(local_ptr);

  TYPE* shA0 = shmem;
  TYPE* shB0 = shA0 + tile_elems_a;
  TYPE* shA1 = shB0 + tile_elems_b;
  TYPE* shB1 = shA1 + tile_elems_a;

  // Single txbar barrier for DXA completion.
  // Only one leader warp executes barrier arrive/wait (count=1).
  // CTA-level visibility is provided by __syncthreads().
  const uint32_t barDXA = local_barrier_id(1);

  const uint32_t hw_threads = static_cast<uint32_t>(vx_num_threads());
  const uint32_t local_tid = threadIdx.x + threadIdx.y * blockDim.x;
  const bool dxa_leader = (local_tid < hw_threads);

  if (mode == 2) {
    // Prologue: load first tiles into stage 0
    if (dxa_leader) {
      dxa_issue_2d_leader(shA0, kDescA, barDXA, 0, row_base);
      dxa_issue_2d_leader(shB0, kDescB, barDXA, col_base, 0);
    }
    uint32_t phase = 0;
    if (dxa_leader) {
      phase = vx_barrier_arrive((int)barDXA, 1);
      vx_barrier_wait((int)barDXA, (int)phase);
    }
    __syncthreads();

    uint32_t stage_now = 0;
    for (uint32_t k_base = 0; k_base < size; k_base += chunk_k) {
      const uint32_t next_k = k_base + chunk_k;
      const bool has_next = (next_k < size);
      const uint32_t stage_next = stage_now ^ 1u;

      TYPE* srcA = (stage_now == 0) ? shA0 : shA1;
      TYPE* srcB = (stage_now == 0) ? shB0 : shB1;
      TYPE* dstA = (stage_next == 0) ? shA0 : shA1;
      TYPE* dstB = (stage_next == 0) ? shB0 : shB1;

      // Prefetch NEXT stage (non-blocking while computing current stage)
      if (has_next && dxa_leader) {
        dxa_issue_2d_leader(dstA, kDescA, barDXA, next_k, row_base);
        dxa_issue_2d_leader(dstB, kDescB, barDXA, col_base, next_k);
        phase = vx_barrier_arrive((int)barDXA, 1);
      }

      // Compute on CURRENT stage (overlaps with DXA prefetch for next)
      gemm_accumulate(sum, srcA, srcB, tile_size, chunk_k, l_row, l_col);

      // Wait for prefetch + warp sync before swapping to next stage
      if (has_next) {
        if (dxa_leader) {
          vx_barrier_wait((int)barDXA, (int)phase);
        }
        __syncthreads();
        stage_now = stage_next;
      }
    }
  } else {
    for (uint32_t k_base = 0; k_base < size; k_base += chunk_k) {
      uint32_t phase = 0;
      if (dxa_leader) {
        dxa_issue_2d_leader(shA0, kDescA, barDXA, k_base, row_base);
        dxa_issue_2d_leader(shB0, kDescB, barDXA, col_base, k_base);
        phase = vx_barrier_arrive((int)barDXA, 1);
        vx_barrier_wait((int)barDXA, (int)phase);
      }
      __syncthreads();

      gemm_accumulate(sum, shA0, shB0, tile_size, chunk_k, l_row, l_col);

      if (k_base + chunk_k < size) {
        __syncthreads();
      }
    }
  }

  C[g_row * size + g_col] = sum;
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
