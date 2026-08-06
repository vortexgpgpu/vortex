// GPU-side program for mode 4 -- WMMA, 3-stage smem pipeline, LSU-staged.
//
// This is the DEVICE program: RISC-V clang compiles it to kernel_m4.vxbin, which runs on
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
#include "k_smem_stage.h"
#include <vx_spawn2.h>
#include <vx_barrier.h>

// ---- mode 4: WMMA, 3-stage smem pipeline, LSU-staged (control for mode 6) ----
//
// Stage for tile t is t mod 3, so the loop is unrolled by 3 — same reason as mode 6.
__kernel void moti_tcu_pipe3(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N = arg->N, K = arg->K, app = arg->app;
  auto pA = reinterpret_cast<const ctx::input_t*>(arg->A_addr);
  auto pB = reinterpret_cast<const ctx::input_t*>(arg->B_addr);
  auto pC = reinterpret_cast<ctx::output_t*>(arg->C_addr);
  auto pD = reinterpret_cast<ctx::output_t*>(arg->D_addr);

  const uint32_t tile_row = blockIdx.y * ctx::tileM;
  const uint32_t tile_col = blockIdx.x * ctx::tileN;

  constexpr uint32_t elemsA = ctx::tileM * ctx::tileK;
  constexpr uint32_t elemsB = ctx::tileN * ctx::tileK;
  constexpr uint32_t stage  = elemsA + elemsB;
  auto smem = reinterpret_cast<ctx::input_t*>(__local_mem());
  #define A_S(n) (smem + (n) * stage)
  #define B_S(n) (smem + (n) * stage + elemsA)

  ctx::fragment_acc fragD;
  wmma_seed_C(fragD, pC, tile_row, tile_col, N);

  const uint32_t numK = K / ctx::tileK;
  const uint32_t tK   = ctx::tileK;

  // prologue: fill the first two stages so the TCU starts two tiles ahead
  smem_stage_fill(A_S(0), B_S(0), pA, pB, 0, tile_row, tile_col, K);
  if (numK > 1)
    smem_stage_fill(A_S(1), B_S(1), pA, pB, tK, tile_row, tile_col, K);

  for (uint32_t kk = 0; kk < numK; kk += 3) {
    if (kk + 2 < numK)
      smem_stage_fill(A_S(2), B_S(2), pA, pB, (kk + 2) * tK, tile_row, tile_col, K);
    smem_stage_consume(fragD, 0, A_S(0), B_S(0));                    // tile kk
    if (kk + 1 < numK) {
      if (kk + 3 < numK)
        smem_stage_fill(A_S(0), B_S(0), pA, pB, (kk + 3) * tK, tile_row, tile_col, K);
      smem_stage_consume(fragD, 1, A_S(1), B_S(1));                  // tile kk+1
    }
    if (kk + 2 < numK) {
      if (kk + 4 < numK)
        smem_stage_fill(A_S(1), B_S(1), pA, pB, (kk + 4) * tK, tile_row, tile_col, K);
      smem_stage_consume(fragD, 2, A_S(2), B_S(2));                  // tile kk+2
    }
  }
  wmma_fuse_epilogue(fragD, app);
  wmma_store_D(pD, fragD, tile_row, tile_col, N);
  #undef A_S
  #undef B_S
}
