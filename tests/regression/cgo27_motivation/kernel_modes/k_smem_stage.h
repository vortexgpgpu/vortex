#ifndef _CGO27_K_SMEM_STAGE_H_
#define _CGO27_K_SMEM_STAGE_H_

// LSU operand staging, shared by modes 3 and 4 (the DXA control pair). Kept here rather
// than duplicated so the two depths differ only in stage count, which is the whole point
// of having both.

#include "wmma_common.h"
#include <vx_spawn2.h>   // vx_barrier.h needs get_num_sub_groups() from here
#include <vx_barrier.h>

// Cooperative global -> LMEM copy of the A and B tiles for one K step. This is what
// vx_dxa_issue_2d_wg does in modes 5/6, done by the block instead of by the engine.
//
// Layouts, matching the DXA descriptors programmed host-side for mode 2:
//   A: row-major [M x K]              -> tile [tileM x tileK] at (tile_row, k_off)
//   B: col-major [K x N] as [N x K]   -> tile [tileN x tileK] at (tile_col, k_off)
// Both land contiguous in smem with a tileK row stride, which is the layout
// load_matrix_sync expects.
//
// always_inline for the reason wmma_common.h documents: an out-of-line helper here
// leaks the smem pointers and pushes them to the stack.
//
// Grid-strided over the flattened tile so the split across threads does not depend on
// tileK dividing the block size.
static __attribute__((always_inline)) void smem_stage_fill(
    ctx::input_t* A_dst, ctx::input_t* B_dst,
    const ctx::input_t* pA, const ctx::input_t* pB,
    uint32_t k_off, uint32_t tile_row, uint32_t tile_col, uint32_t K) {
  const uint32_t lane  = threadIdx.x;
  const uint32_t nlane = blockDim.x;

  for (uint32_t i = lane; i < ctx::tileM * ctx::tileK; i += nlane) {
    const uint32_t r = i / ctx::tileK, c = i % ctx::tileK;
    A_dst[i] = pA[(tile_row + r) * K + k_off + c];
  }
  for (uint32_t i = lane; i < ctx::tileN * ctx::tileK; i += nlane) {
    const uint32_t r = i / ctx::tileK, c = i % ctx::tileK;
    B_dst[i] = pB[(tile_col + r) * K + k_off + c];
  }
}

// Consume one filled stage. Identical to kernel_m6.cpp's p3_consume apart from where the
// operands came from: the leading wait is the RAW edge against the fill, the trailing
// one the WAR edge against the next refill of this stage.
static __attribute__((always_inline)) void smem_stage_consume(
    ctx::fragment_acc& fragD, uint32_t bar_no,
    const ctx::input_t* A_src, const ctx::input_t* B_src) {
  vortex::barrier b(bar_no);
  b.arrive_and_wait();                       // stage filled
  ctx::fragment_a fragA;
  ctx::fragment_b fragB;
  ctx::load_matrix_sync(fragA, A_src, ctx::tileK);
  ctx::load_matrix_sync<vt::col_major>(fragB, B_src, ctx::tileK);
  ctx::mma_sync(fragD, fragA, fragB, fragD);
  b.arrive_and_wait();                       // stage free to refill
}

#endif // _CGO27_K_SMEM_STAGE_H_
