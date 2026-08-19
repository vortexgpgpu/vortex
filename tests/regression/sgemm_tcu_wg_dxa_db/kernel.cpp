// Double-buffered WGMMA GEMM: the deployment-shaped pipeline.
//
// The single-buffered sibling (sgemm_tcu_wg_dxa) serializes the K loop —
// issue, wait, compute, wait — so the engine is idle while the tensor pipe
// runs and vice versa. That measures the instruction saving of engine-driven
// tile loads but can never show overlap.
//
// Here two SMEM stages alternate and each stage owns its own transaction
// barrier, so the engine fills stage k+1 while WGMMA consumes stage k. The
// only wait is on the barrier of the stage about to be consumed.
//
//   prologue : issue stage 0
//   loop k   : issue stage k+1  →  wait(stage k)  →  WGMMA(stage k)
//
// SW_LOAD_A / SW_LOAD_B replace the engine with cooperative software loads
// for that operand; with both set there is no DXA traffic at all, which is
// the honest software baseline for the same pipeline structure.
#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;

// DXA descriptor slots (programmed by host in main.cpp).
[[maybe_unused]] constexpr uint32_t kDescA = 0;
[[maybe_unused]] constexpr uint32_t kDescB = 1;

#if defined(SW_LOAD_A) && defined(SW_LOAD_B)
#define DB_ALL_SW 1
#endif

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);
#ifdef SW_LOAD_B
  auto pB = reinterpret_cast<const ctx::input_t *>(arg->B_addr);
#endif
#ifdef SW_LOAD_A
  auto pA = reinterpret_cast<const ctx::input_t *>(arg->A_addr);
#endif

  uint32_t N = arg->N;
  uint32_t K = arg->K;

  uint32_t tid = threadIdx.x;
  uint32_t num_threads = blockDim.x;
  uint32_t warp_rank = tid / VX_CFG_NUM_THREADS;
  uint32_t num_warps = num_threads / VX_CFG_NUM_THREADS;

  uint32_t cta_M = num_warps * ctx::xtileM;
  uint32_t tile_row = blockIdx.y * cta_M;
  uint32_t tile_col = blockIdx.x * ctx::xtileN;

  // Two stages back to back: [A0 B0][A1 B1].
  const uint32_t a_elems = cta_M * ctx::tileK;
  const uint32_t b_elems = ctx::tileK * ctx::xtileN;
  const uint32_t stage_elems = a_elems + b_elems;

  auto smem = reinterpret_cast<ctx::input_t *>(__local_mem());
  ctx::input_t* A_st[2] = { smem,                    smem + stage_elems };
  ctx::input_t* B_st[2] = { smem + a_elems,          smem + stage_elems + a_elems };

  // One barrier per stage: waiting on stage k must not be satisfied by the
  // in-flight fill of stage k+1.
  vortex::barrier bar0(0);
  vortex::barrier bar1(1);
  vortex::barrier* bar_st[2] = { &bar0, &bar1 };
  // Reuse is fenced by a SECOND arrive_and_wait on the stage's OWN barrier
  // after the compute (see the loop). Using a third, independent barrier
  // here would advance its generation out of step with the stage barriers
  // and make warps wait on mismatched phases.

#ifndef DB_ALL_SW
  const bool is_dxa_warp = (get_sub_group_id() == 0);
#endif

  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  // ---- stage fill: engine issue (arming its barrier) and/or SW copy -------
  auto fill_stage = [&](uint32_t s, uint32_t kk) {
#ifndef DB_ALL_SW
    if (is_dxa_warp) {
  #if defined(SW_LOAD_A)
      bar_st[s]->expect_tx(1);
      vx_dxa_issue_2d_wg(kDescB, bar_st[s]->id(), B_st[s], tile_col, kk);
  #elif defined(SW_LOAD_B)
      bar_st[s]->expect_tx(1);
      vx_dxa_issue_2d_wg(kDescA, bar_st[s]->id(), A_st[s], kk, tile_row);
  #else
      bar_st[s]->expect_tx(2);
      vx_dxa_issue_2d_wg(kDescA, bar_st[s]->id(), A_st[s], kk, tile_row);
      vx_dxa_issue_2d_wg(kDescB, bar_st[s]->id(), B_st[s], tile_col, kk);
  #endif
    }
#endif
#ifdef SW_LOAD_A
    for (uint32_t i = 0; i < a_elems; i += num_threads) {
      uint32_t idx = i + tid;
      if (idx >= a_elems) break;
      uint32_t r = idx / ctx::tileK;
      uint32_t c = idx % ctx::tileK;
      A_st[s][r * ctx::tileK + c] = pA[(tile_row + r) * K + (kk + c)];
    }
#endif
#ifdef SW_LOAD_B
    for (uint32_t i = 0; i < b_elems; i += num_threads) {
      uint32_t idx = i + tid;
      if (idx >= b_elems) break;
      uint32_t r = idx / ctx::xtileN;
      uint32_t c = idx % ctx::xtileN;
      B_st[s][ctx::b_blockmajor_idx(r, c)] = pB[(kk + r) * N + (tile_col + c)];
    }
#endif
  };

  // ---- prologue: start filling stage 0 -----------------------------------
  uint32_t stage = 0;
  fill_stage(stage, 0);

  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    const uint32_t next_k = k + ctx::tileK;
    const uint32_t nstage = stage ^ 1u;

    // (1) Issue the NEXT stage first, so the engine fills stage k+1 while
    // this iteration's WGMMA consumes stage k.
    if (next_k < K)
      fill_stage(nstage, next_k);

    // (2) Wait for the current stage: DXA completion + CTA sync.
    bar_st[stage]->arrive_and_wait();

    auto A_warp = A_st[stage] + warp_rank * ctx::xtileM * ctx::tileK;
    auto desc_b = vt::vx_make_smem_desc(B_st[stage], 0);

  #if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
    ctx::fragment_a fragA;
    ctx::load_matrix_sync(fragA, A_warp, ctx::tileK);
    ctx::wgmma_sync(fragC, fragA, desc_b, fragC);
  #else
    auto desc_a = vt::vx_make_smem_desc(A_warp, ctx::tileK * sizeof(ctx::input_t));
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);
  #endif

    // (3) Release the current stage on its OWN barrier, so its buffer is
    // only refilled after every warp has consumed it. Same barrier as the
    // wait above, so the two rendezvous stay phase-matched.
    bar_st[stage]->arrive_and_wait();

    stage = nstage;
  }

  auto pTileC = pC + (tile_row + warp_rank * ctx::xtileM) * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
