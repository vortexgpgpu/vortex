// GPU-side program for mode 3 -- WMMA, 2-stage smem pipeline, LSU-staged.
//
// This is the DEVICE program: RISC-V clang compiles it to kernel_m3.vxbin, which runs on
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

// ---- mode 3: WMMA, 2-stage smem pipeline, LSU-staged (control for mode 5) ----
//
// Structure mirrors moti_tcu_dxa_pipe exactly, including the by-2 unroll that keeps
// every stage and barrier selection a compile-time constant.
//
// One thing genuinely differs and it is the finding, not a flaw: a DXA fill is ASYNC,
// so mode 5 issues the fill for tile k+1 and computes tile k while it runs. An LSU
// fill is the block's own instruction stream, so nothing overlaps inside a CTA — the
// prefetch distance only buys overlap ACROSS resident CTAs. Whether two stages are
// worth their lmem and barriers without an engine behind them is the question.
__kernel void moti_tcu_pipe2(kernel_arg_t* __UNIFORM__ arg) {
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
  ctx::input_t* A_smem0 = smem;                  // stage 0
  ctx::input_t* B_smem0 = smem + elemsA;
  ctx::input_t* A_smem1 = smem + stage;          // stage 1
  ctx::input_t* B_smem1 = smem + stage + elemsA;

  ctx::fragment_acc fragD;
  wmma_seed_C(fragD, pC, tile_row, tile_col, N);

  const uint32_t numK = K / ctx::tileK;
  const uint32_t tK   = ctx::tileK;

  smem_stage_fill(A_smem0, B_smem0, pA, pB, 0, tile_row, tile_col, K);
  for (uint32_t kk = 0; kk < numK; kk += 2) {
    // even step: stage 1 takes tile kk+1 while stage 0 is consumed
    if (kk + 1 < numK)
      smem_stage_fill(A_smem1, B_smem1, pA, pB, (kk + 1) * tK, tile_row, tile_col, K);
    smem_stage_consume(fragD, 0, A_smem0, B_smem0);
    // odd step: stage 0 takes tile kk+2 while stage 1 is consumed
    if (kk + 1 < numK) {
      if (kk + 2 < numK)
        smem_stage_fill(A_smem0, B_smem0, pA, pB, (kk + 2) * tK, tile_row, tile_col, K);
      smem_stage_consume(fragD, 1, A_smem1, B_smem1);
    }
  }
  wmma_fuse_epilogue(fragD, app);
  wmma_store_D(pD, fragD, tile_row, tile_col, N);
}
