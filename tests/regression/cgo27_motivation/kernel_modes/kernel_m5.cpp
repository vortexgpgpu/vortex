// GPU-side program for mode 5 -- workgroup WGMMA + DXA, A resident in Local Memory.
//
// Mode 3 with ONE thing changed: the CTA owns MOTI_WG_NCOLS column tiles instead of one,
// and stages its A rows for the WHOLE K range once, up front, instead of a K-slice per
// step. That is the N-axis reuse the harness has been missing, and it is the only form of
// it this machine can express.
//
// WHY NOT REGISTERS. The obvious way to reuse A along N is a wider accumulator: raise
// WGMMA_NRC so one fragment-A load feeds a wider output tile. Measured, and it loses
// badly -- at NRC=16 the accumulator is 16 f-registers and the compiler spills every
// iteration (stack references 7 -> 37, loads and stores both 6,016, i.e. exactly equal;
// 128x64x32 went 25,386 -> 136,677 and 256x128x64 103,400 -> 932,115 with the grid at a
// full 16 CTAs, so it is not occupancy). N-axis reuse cannot be bought with registers
// here. Local Memory, on the other hand, is 96 % idle: mode 3 uses 2,560 B of 65,536.
//
// WHAT IT BUYS. Mode 3 fetches its A rows from global once per column tile, because each
// column tile is a different CTA:
//
//     A traffic = (M/cta_M x N/xtileN) x cta_M x K  =  M*N*K / xtileN
//
// Sweeping NCOLS column tiles inside one CTA divides that by NCOLS -- 4x less global A
// traffic at the default. B traffic is unchanged; it has no reuse to find.
//
// WHAT IT COSTS. A[cta_M x K] is 16 KB at cta_M=64, K=128, against mode 3's 2.5 KB, so
// CtaDispatcher's usable_slots() drops from 16 to 3 and the resident CTAs go 4 -> 3 (the
// ceiling is otherwise NUM_WARPS/warps-per-CTA = 4). A quarter of the occupancy for a
// quarter of the A traffic: which way that lands is the measurement, and it is expected
// to depend on the shape -- large N gives more column tiles to amortise over, large K
// makes A too big for Local Memory to hold. That shape dependence is the point of having
// the mode, not a defect of it.
//
// The accumulator is still ONE tile: the column loop finishes and stores each tile before
// starting the next, so nothing here reintroduces the register pressure that killed
// NRC=16.

#include "k_wg_common.h"
#include <vx_spawn2.h>
#include <vx_barrier.h>
#include <vx_dxa.h>
// The standalone epilogue passes. k_epilogue.h compiles ONLY the one this build's
// MOTI_APP needs -- nothing at all at MOTI_APP=1 -- so no unused kernel lands in the
// binary and no mode's address moves. See common.h.
#include "k_epilogue.h"

// Column tiles swept by one CTA against one resident A. MOTI_WG_NCOLS is defined in
// common.h and shared with the host, which derives the grid width and the lmem size from
// the same macro -- one definition, so the two cannot disagree.
static constexpr uint32_t kNCols = MOTI_WG_NCOLS;

__kernel void moti_tcu_wg_acol(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t N = arg->N, K = arg->K, app = arg->app;
  auto pC = reinterpret_cast<wgctx::output_t*>(arg->C_addr);
  auto pD = reinterpret_cast<wgctx::output_t*>(arg->D_addr);

  const uint32_t tid       = threadIdx.x;
  const uint32_t warp_rank = tid / VX_CFG_NUM_THREADS;
  const uint32_t num_warps = blockDim.x / VX_CFG_NUM_THREADS;

  const uint32_t cta_M    = num_warps * wgctx::xtileM;
  const uint32_t tile_row = blockIdx.y * cta_M;
  const uint32_t col_base = blockIdx.x * (kNCols * wgctx::xtileN);

  // A [cta_M x K] row-major, stride K -- the whole K range, staged once. B [xtileN x kStK]
  // K-major behind it, restaged per (column tile, K step) exactly as mode 3 does.
  auto smem   = reinterpret_cast<wgctx::input_t*>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + cta_M * K;

  vortex::barrier bar(0);
  const bool is_dxa_warp = (get_sub_group_id() == 0);

  // One descriptor for the CTA's entire A block. DESC_A is programmed with tile0 = K for
  // this mode, so a single issue at k=0 brings all of it.
  if (is_dxa_warp) {
    bar.expect_tx(1);
    vx_dxa_issue_2d_wg(DESC_A, bar.id(), A_smem, 0, tile_row);
  }
  bar.arrive_and_wait();

  for (uint32_t n = 0; n < kNCols; ++n) {
    const uint32_t tile_col = col_base + n * wgctx::xtileN;

    wgctx::fragment_acc fragC;
    wgctx::fill_fragment(fragC, 0);

    for (uint32_t k = 0; k < K; k += kStK) {
      if (is_dxa_warp) {
        bar.expect_tx(1);
        vx_dxa_issue_2d_wg(DESC_B, bar.id(), B_smem, k, tile_col);
      }
      bar.arrive_and_wait();                              // B staged

      for (uint32_t s = 0; s < kS; ++s) {
        // A never moves: read straight out of the resident block at row stride K.
        auto A_warp = A_smem + warp_rank * wgctx::xtileM * K + k + s * wgctx::tileK;
        auto desc_b = vt::vx_make_smem_desc(B_smem + s * wgctx::tileK,
                                            kStK * sizeof(wgctx::input_t));
        wgctx::fragment_a fragA;
        wgctx::load_matrix_sync(fragA, A_warp, K);
        wgctx::wgmma_sync(fragC, fragA, desc_b, fragC);
      }

      bar.arrive_and_wait();                              // B free to refill
    }

    // Same fused epilogue as mode 3: C folded in while the accumulator is in registers.
    wg_store_epilogue(fragC, pC, pD, tile_row + warp_rank * wgctx::xtileM,
                      tile_col, N, app);
  }
}
