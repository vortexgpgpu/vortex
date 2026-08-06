// GPU-side program for mode 6 -- WMMA, 3-stage smem pipeline, DXA-staged.
//
// This is the DEVICE program: RISC-V clang compiles it to kernel_m6.vxbin, which runs on
// the GPU. It has no main(); its entry point is the __kernel below. The host program that
// opens the device, uploads A/B/C, launches this and reads D back is main.cpp, built for
// x86 -- that is where main() lives.
//
// It contains this mode's kernel and NOTHING else, which is the point. In the old
// all-in-one kernel.vxbin every mode occupied address space even though only one ran, and
// address decides icache set: adding modes 3/4 moved mode 2 from 15,468 to 24,106 cycles
// with a byte-identical kernel body. Here every mode's code starts at 0x180000034
// whatever else is in the tree.

#include "wmma_common.h"
#include <vx_spawn2.h>
#include <vx_barrier.h>
#include <vx_dxa.h>

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
