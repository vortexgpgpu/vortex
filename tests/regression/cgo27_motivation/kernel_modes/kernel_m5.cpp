// GPU-side program for mode 5 -- WMMA, 2-stage smem pipeline, DXA-staged.
//
// This is the DEVICE program: RISC-V clang compiles it to kernel_m5.vxbin, which runs on
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
