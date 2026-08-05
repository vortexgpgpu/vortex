#ifndef _CGO27_K_TCU_H_
#define _CGO27_K_TCU_H_

// In-core TCU (WMMA) paths — one block (one warp) per output tile.
//
//   moti_tcu           mode 1  operands loaded straight from global    <- dtcu_compare mode 0
//   moti_tcu_dxa       mode 2  operands staged into smem by DXA (naive) <- sgemm_tcu_wg_dxa
//   moti_tcu_dxa_pipe3 mode 5  WMMA + DXA, 3-stage smem pipeline (deeper than mode 6)
//   moti_tcu_dxa_pipe  mode 6  WMMA + DXA, 2-stage smem pipeline        <- sgemm2_dxa
//
// 1/2 are the NAIVE variants (fetch tile K, compute tile K, repeat); 5/6 are the
// software-pipelined variants, differing only in prefetch DEPTH: mode 6 runs one
// tile ahead (2 stages), mode 5 runs two tiles ahead (3 stages).
// Deliberately NO dtensor_start anywhere in this file — the WMMA entries must not
// be able to poke the cluster DTCU.

// NOTE on `app = arg->app` at kernel entry: read the epilogue selector UP FRONT,
// never at the store. Loading it after the compute/barrier phase leaves a lone
// global load on the tail of the kernel, and the warp cannot store D or retire
// until it returns — measured on mode 2 (DXA/barrier-bound): 15,456 cycles with no
// epilogue call, 17,492 with the selector read at entry, 23,986 with it read at the
// tail. Reading it at entry lets the load overlap the GEMM.
#include "wmma_common.h"
#include <vx_spawn2.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

// ---- mode 1: WMMA, operands loaded directly from global ----
__kernel void moti_tcu(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N = arg->N, K = arg->K, app = arg->app;
  auto pA = reinterpret_cast<ctx::input_t*>(arg->A_addr);
  auto pB = reinterpret_cast<ctx::input_t*>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t*>(arg->C_addr);
  auto pD = reinterpret_cast<ctx::output_t*>(arg->D_addr);

  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragD;
  wmma_seed_C(fragD, pC, tile_row, tile_col, N);

  for (uint32_t i = 0; i < K; i += ctx::tileK) {
    ctx::load_matrix_sync(fragA, pA + tile_row * K + i, K);                 // A row-major
    ctx::load_matrix_sync<vt::col_major>(fragB, pB + tile_col * K + i, K);  // B col-major
    ctx::mma_sync(fragD, fragA, fragB, fragD);
  }
  wmma_fuse_epilogue(fragD, app);
  wmma_store_D(pD, fragD, tile_row, tile_col, N);
}

// ---- mode 2: WMMA, operands staged into smem by DXA (naive, single-buffered) ----
__kernel void moti_tcu_dxa(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N = arg->N, K = arg->K, app = arg->app;
  auto pC = reinterpret_cast<ctx::output_t*>(arg->C_addr);
  auto pD = reinterpret_cast<ctx::output_t*>(arg->D_addr);

  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  auto smem   = reinterpret_cast<ctx::input_t*>(__local_mem());
  auto A_smem = smem;                            // [tileM x tileK]
  auto B_smem = smem + ctx::tileM * ctx::tileK;  // [tileN x tileK]

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragD;
  wmma_seed_C(fragD, pC, tile_row, tile_col, N);

  vortex::barrier bar(0);
  for (uint32_t i = 0; i < K; i += ctx::tileK) {
    bar.expect_tx(2);
    vx_dxa_issue_2d_wg(DESC_A, bar.id(), A_smem, i, tile_row);
    vx_dxa_issue_2d_wg(DESC_B, bar.id(), B_smem, i, tile_col);
    bar.arrive_and_wait();

    ctx::load_matrix_sync(fragA, A_smem, ctx::tileK);
    ctx::load_matrix_sync<vt::col_major>(fragB, B_smem, ctx::tileK);
    ctx::mma_sync(fragD, fragA, fragB, fragD);

    bar.arrive_and_wait(); // WMMA done before the next DXA overwrites smem
  }
  wmma_fuse_epilogue(fragD, app);
  wmma_store_D(pD, fragD, tile_row, tile_col, N);
}

// ---- mode 5: WMMA + DXA, THREE-stage smem pipeline (SW pipelined, deeper) ----
//
// Replaces the old register-double-buffered mode 5, which is not implementable on
// this target: mma_sync pins its operands to fixed physical f-registers (C/D f0-f7,
// A f10-f17, B f24-f31), so one MMA reserves 24 of the 32 f-registers and only 8 are
// free while a second operand set needs 16 — and FragA/FragC/D are hard-wired to 8
// registers regardless of NR, so the tile cannot be shrunk to buy room. Details in
// 260718_moti_RFC.md. The pipelining that DOES work here goes through shared memory,
// so mode 5 is now the DEEPER version of mode 6: three smem stages instead of two,
// i.e. the DXA runs up to two tiles ahead of the TCU rather than one.
//
// Stage for tile t is t mod 3, so the loop is unrolled by 3 and every stage/barrier
// selection is a compile-time constant (a runtime `A_smem[cur]` index would push the
// pointer array to the stack — that bug cost mode 6 6.2x, see its comment).
//
// COST vs mode 6: 3x stage_bytes of lmem and 3 barriers per CTA instead of 2, so
// occupancy can drop (lmem 64KB / 3*2048B = 10 CTAs; NUM_BARRIERS=32 / 3 = 10 CTAs).
// Whether the extra prefetch depth beats that occupancy loss is what the sweep must
// answer — it is the point of having both depths.
// Scoped helpers for the 3-stage pipeline: each keeps its barrier object alive only
// for the operation that needs it. Three long-lived `vortex::barrier` objects cost 6
// live integer registers (each holds bar_id_ + num_warps_), and together with the
// deeper pipeline's bookkeeping that pushed the kernel into integer spills.
static __attribute__((always_inline)) void p3_fill(uint32_t bar_no,
                                                   ctx::input_t* A_dst, ctx::input_t* B_dst,
                                                   uint32_t k_off,
                                                   uint32_t tile_row, uint32_t tile_col) {
  vortex::barrier b(bar_no);
  b.expect_tx(2);
  vx_dxa_issue_2d_wg(DESC_A, b.id(), A_dst, k_off, tile_row);
  vx_dxa_issue_2d_wg(DESC_B, b.id(), B_dst, k_off, tile_col);
}
static __attribute__((always_inline)) void p3_consume(ctx::fragment_acc& fragD, uint32_t bar_no,
                                                      ctx::input_t* A_src, ctx::input_t* B_src) {
  vortex::barrier b(bar_no);
  b.arrive_and_wait();                       // stage filled
  ctx::fragment_a fragA;
  ctx::fragment_b fragB;
  ctx::load_matrix_sync(fragA, A_src, ctx::tileK);
  ctx::load_matrix_sync<vt::col_major>(fragB, B_src, ctx::tileK);
  ctx::mma_sync(fragD, fragA, fragB, fragD);
  b.arrive_and_wait();                       // stage free to refill
}

__kernel void moti_tcu_dxa_pipe3(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N = arg->N, K = arg->K, app = arg->app;
  auto pC = reinterpret_cast<ctx::output_t*>(arg->C_addr);
  auto pD = reinterpret_cast<ctx::output_t*>(arg->D_addr);

  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  // Stage offsets are constexpr so each stage address folds into a constant
  // displacement off one base pointer instead of six live pointers.
  constexpr uint32_t elemsA = ctx::tileM * ctx::tileK;
  constexpr uint32_t elemsB = ctx::tileN * ctx::tileK;
  constexpr uint32_t stage  = elemsA + elemsB;
  auto smem = reinterpret_cast<ctx::input_t*>(__local_mem());
  #define A_S(n) (smem + (n) * stage)
  #define B_S(n) (smem + (n) * stage + elemsA)

  ctx::fragment_acc fragD;
  wmma_seed_C(fragD, pC, tile_row, tile_col, N);

  const bool is_dxa = (get_sub_group_id() == 0);
  const uint32_t numK = K / ctx::tileK;
  const uint32_t tK = ctx::tileK;

  // prologue: fill the first two stages so the TCU starts two tiles ahead
  if (is_dxa) {
    p3_fill(0, A_S(0), B_S(0), 0, tile_row, tile_col);
    if (numK > 1) p3_fill(1, A_S(1), B_S(1), tK, tile_row, tile_col);
  }
  for (uint32_t kk = 0; kk < numK; kk += 3) {
    // stage for tile t is t mod 3, so a by-3 unroll keeps every selection constant
    if (kk + 2 < numK && is_dxa) p3_fill(2, A_S(2), B_S(2), (kk + 2) * tK, tile_row, tile_col);
    p3_consume(fragD, 0, A_S(0), B_S(0));                       // tile kk
    if (kk + 1 < numK) {
      if (kk + 3 < numK && is_dxa) p3_fill(0, A_S(0), B_S(0), (kk + 3) * tK, tile_row, tile_col);
      p3_consume(fragD, 1, A_S(1), B_S(1));                     // tile kk+1
    }
    if (kk + 2 < numK) {
      if (kk + 4 < numK && is_dxa) p3_fill(1, A_S(1), B_S(1), (kk + 4) * tK, tile_row, tile_col);
      p3_consume(fragD, 2, A_S(2), B_S(2));                     // tile kk+2
    }
  }
  wmma_fuse_epilogue(fragD, app);
  wmma_store_D(pD, fragD, tile_row, tile_col, N);
  #undef A_S
  #undef B_S
}

// ---- mode 6: WMMA + DXA, smem double-buffered (SW pipelined) ----
// ping-pong across two smem stages: issue the DXA fill for tile K+1 into the idle
// stage before waiting on / computing tile K from the other one. Structure from
// sgemm2_dxa; issue coords match the mode-2 descriptors.
__kernel void moti_tcu_dxa_pipe(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N = arg->N, K = arg->K, app = arg->app;
  auto pC = reinterpret_cast<ctx::output_t*>(arg->C_addr);
  auto pD = reinterpret_cast<ctx::output_t*>(arg->D_addr);

  uint32_t tile_row = blockIdx.y * ctx::tileM;
  uint32_t tile_col = blockIdx.x * ctx::tileN;

  uint32_t elemsA = ctx::tileM * ctx::tileK;
  uint32_t elemsB = ctx::tileN * ctx::tileK;
  uint32_t stage  = elemsA + elemsB;
  // Named stage pointers + named barriers, K loop unrolled by 2 — same reason as
  // mode 5: `A_smem[cur]` / `bar[cur]` with a runtime `cur` makes those arrays
  // addressable and pushes them to the stack (measured 9 sp-relative accesses vs 1
  // in the non-pipelined mode 2). Unrolling makes every selection compile-time.
  auto smem = reinterpret_cast<ctx::input_t*>(__local_mem());
  ctx::input_t* A_smem0 = smem;                          // stage 0
  ctx::input_t* B_smem0 = smem + elemsA;
  ctx::input_t* A_smem1 = smem + stage;                  // stage 1
  ctx::input_t* B_smem1 = smem + stage + elemsA;

  ctx::fragment_a   fragA;
  ctx::fragment_b   fragB;
  ctx::fragment_acc fragD;
  wmma_seed_C(fragD, pC, tile_row, tile_col, N);

  vortex::barrier bar0(0), bar1(1);
  const bool is_dxa = (get_sub_group_id() == 0);
  uint32_t numK = K / ctx::tileK;

  if (is_dxa) {
    bar0.expect_tx(2);
    vx_dxa_issue_2d_wg(DESC_A, bar0.id(), A_smem0, 0, tile_row);
    vx_dxa_issue_2d_wg(DESC_B, bar0.id(), B_smem0, 0, tile_col);
  }
  for (uint32_t kk = 0; kk < numK; kk += 2) {
    // even step: stage 1 prefetches tile kk+1 while stage 0 is consumed
    if (kk + 1 < numK && is_dxa) {
      uint32_t i1 = (kk + 1) * ctx::tileK;
      bar1.expect_tx(2);
      vx_dxa_issue_2d_wg(DESC_A, bar1.id(), A_smem1, i1, tile_row);
      vx_dxa_issue_2d_wg(DESC_B, bar1.id(), B_smem1, i1, tile_col);
    }
    bar0.arrive_and_wait();
    ctx::load_matrix_sync(fragA, A_smem0, ctx::tileK);
    ctx::load_matrix_sync<vt::col_major>(fragB, B_smem0, ctx::tileK);
    ctx::mma_sync(fragD, fragA, fragB, fragD);
    bar0.arrive_and_wait();   // stage 0 free to refill
    // odd step: stage 0 prefetches tile kk+2 while stage 1 is consumed
    if (kk + 1 < numK) {
      if (kk + 2 < numK && is_dxa) {
        uint32_t i2 = (kk + 2) * ctx::tileK;
        bar0.expect_tx(2);
        vx_dxa_issue_2d_wg(DESC_A, bar0.id(), A_smem0, i2, tile_row);
        vx_dxa_issue_2d_wg(DESC_B, bar0.id(), B_smem0, i2, tile_col);
      }
      bar1.arrive_and_wait();
      ctx::load_matrix_sync(fragA, A_smem1, ctx::tileK);
      ctx::load_matrix_sync<vt::col_major>(fragB, B_smem1, ctx::tileK);
      ctx::mma_sync(fragD, fragA, fragB, fragD);
      bar1.arrive_and_wait();
    }
  }
  wmma_fuse_epilogue(fragD, app);
  wmma_store_D(pD, fragD, tile_row, tile_col, N);
}

#endif // _CGO27_K_TCU_H_
