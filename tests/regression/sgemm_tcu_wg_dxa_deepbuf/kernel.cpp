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

  // CTA tile dimensions
  uint32_t cta_M = num_warps * ctx::xtileM;
  uint32_t tile_row = blockIdx.y * cta_M;
  uint32_t tile_col = blockIdx.x * ctx::xtileN;

  // Shared memory layout: A [cta_M x tileK] then B [tileK x xtileN]
  auto smem   = reinterpret_cast<ctx::input_t *>(__local_mem());
  auto A_smem = smem;
  auto B_smem = smem + cta_M * ctx::tileK;
  (void)A_smem; (void)B_smem;

  // Initialize accumulator tile to zero.
  ctx::fragment_acc fragC;
  ctx::fill_fragment(fragC, 0);

  // Only the first warp in the CTA issues DXA commands.
  const bool is_dxa_warp = (get_sub_group_id() == 0);
  (void)is_dxa_warp; // unused under WARP_SPEC, which derives is_producer instead

  // Guards the final store below for WARP_SPEC's dedicated producer warp,
  // which never accumulates into fragC and must not write pC.
  bool skip_store = false;
  (void)skip_store; // only read/written under WARP_SPEC

#if defined(WARP_SPEC) && !defined(SW_LOAD_A) && !defined(SW_LOAD_B)
  // ── Warp-specialized pipeline ───────────────────────────────────────────
  // Unlike DOUBLE_BUF (every warp both issues DXA *and* computes, ping-
  // ponging through an array of buffer pointers), here warp roles are
  // strictly separated: the last warp in the CTA is a dedicated producer
  // that only ever issues DXA, never runs WGMMA; every other warp is a
  // dedicated consumer that only ever runs WGMMA, never touches DXA. This
  // mirrors CUTLASS/Hopper-style warp-specialized kernels.
  //
  // This also sidesteps the root cause of DOUBLE_BUF's regression (measured
  // separately): that kernel indexed `A_sm[cur]`/`B_sm[cur]` through a
  // runtime-variable array, which the compiler spilled to the stack and
  // reloaded every iteration (verified in disassembly: 4 extra `lw`/iter).
  // Here each stage's address is instead computed by plain arithmetic
  // (`smem + stage*stage_elems + ...`), so there is no array for the
  // compiler to spill -- `stage` is just a scalar multiplied in, and
  // producer/consumer state now also lives in different warps' register
  // files instead of being doubled up in one.
  const uint32_t num_consumers = num_warps - 1;
  const bool     is_producer   = (warp_rank == num_consumers);
  skip_store = is_producer;

  // Recompute this CTA's M-band for the reduced consumer count (main.cpp's
  // cta_M/grid already account for this under WARP_SPEC).
  const uint32_t cta_M2    = num_consumers * ctx::xtileM;
  const uint32_t tile_row2 = blockIdx.y * cta_M2;

  const uint32_t stage_elems = cta_M2 * ctx::tileK + ctx::tileK * ctx::xtileN;
  vortex::barrier bar[2] = { vortex::barrier(0), vortex::barrier(1) };

  auto stage_A = [&](uint32_t s) { return smem + s * stage_elems; };
  auto stage_B = [&](uint32_t s) { return smem + s * stage_elems + cta_M2 * ctx::tileK; };

  uint32_t cur = 0;

  // Prologue: the producer prefetches tile k=0 into stage 0.
  if (is_producer) {
    bar[0].expect_tx(2);
    vx_dxa_issue_2d_wg(kDescA, bar[0].id(), stage_A(0), 0, tile_row2);
    vx_dxa_issue_2d_wg(kDescB, bar[0].id(), stage_B(0), tile_col, 0);
  }

  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    const uint32_t next_k   = k + ctx::tileK;
    const bool     has_next = (next_k < K);
    const uint32_t nxt      = cur ^ 1u;

    // Producer: issue the next stage's DXA fetch (runs ahead of compute).
    if (has_next && is_producer) {
      bar[nxt].expect_tx(2);
      vx_dxa_issue_2d_wg(kDescA, bar[nxt].id(), stage_A(nxt), next_k, tile_row2);
      vx_dxa_issue_2d_wg(kDescB, bar[nxt].id(), stage_B(nxt), tile_col, next_k);
    }

    // Both roles rendezvous here: producer's arrival plus its expect_tx'd
    // DMA completions must both land before this phase advances.
    bar[cur].arrive_and_wait();

    // Consumer: compute on the current stage. Producer does nothing here.
    if (!is_producer) {
      auto A_warp = stage_A(cur) + warp_rank * ctx::xtileM * ctx::tileK;
      auto desc_b = vt::vx_make_smem_desc(stage_B(cur), 0);
    #if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
      ctx::fragment_a fragA;
      ctx::load_matrix_sync(fragA, A_warp, ctx::tileK);
      ctx::wgmma_sync(fragC, fragA, desc_b, fragC);
    #else
      auto desc_a = vt::vx_make_smem_desc(A_warp, ctx::tileK * sizeof(ctx::input_t));
      ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);
    #endif
    }

    // Gate buffer reuse: all warps (producer included) rendezvous again
    // before the producer's next-next issue could overwrite this stage.
    bar[cur].arrive_and_wait();
    cur = nxt;
  }

  // Store uses the consumer-band tile_row, not the outer (unused) one.
  if (!skip_store) {
    auto pTileC = pC + (tile_row2 + warp_rank * ctx::xtileM) * N + tile_col;
    ctx::store_matrix_sync(pTileC, fragC, N);
  }
  return;
#elif defined(DEEP_BUF) && !defined(SW_LOAD_A) && !defined(SW_LOAD_B)
  // ── N-stage software-pipelined DXA fetch ────────────────────────────────
  // Generalizes DOUBLE_BUF (N=2) to PIPE_N stages (common.h, default 4) so
  // up to PIPE_N-1 tiles' worth of DXA transfers can be genuinely queued/
  // in-flight at once, instead of only ever having one iteration's fetch
  // outstanding. The goal is interconnect *utilization*, not per-iteration
  // latency hiding: compute already has slack to absorb one transfer's
  // ~500-cycle latency at this tile size, but with only 2 requests queued
  // per iteration the DXA channel is idle most of each ~10,000-cycle
  // iteration window. Deeper queuing keeps more requests in flight so the
  // channel (and the DXA_MAX_INFLIGHT=8 budget) is actually exercised.
  //
  // Same register-spill-free arithmetic addressing as DOUBLE_BUF
  // (stage_A/stage_B/bar_id, no array indexed by a runtime variable). The
  // fill-wait + gate-wait discipline (two barrier rendezvous per stage
  // *visit*) is unchanged and generalizes safely to any N: every warp still
  // passes through the same sequential loop in program order, so a stage's
  // refetch-issue (N-1 iterations after its previous visit) can only be
  // reached after that visit's gate-wait -- a real CTA-wide barrier -- has
  // been passed by every warp, including the issuing one.
  constexpr uint32_t kPipeN = PIPE_N;
  const uint32_t stage_elems = cta_M * ctx::tileK + ctx::tileK * ctx::xtileN;
  auto stage_A = [&](uint32_t s) { return smem + s * stage_elems; };
  auto stage_B = [&](uint32_t s) { return smem + s * stage_elems + cta_M * ctx::tileK; };
  const uint32_t cta_id = get_local_group_id();
  auto bar_id = [&](uint32_t s) { return cta_id | (s << 8); };

  const uint32_t num_steps = (K + ctx::tileK - 1) / ctx::tileK;

  // Prologue: fill the pipeline kPipeN-1 stages ahead of the first compute.
  if (is_dxa_warp) {
    for (uint32_t s = 0; s < kPipeN - 1 && s < num_steps; ++s) {
      vx_barrier_expect_tx(bar_id(s), 2);
      vx_dxa_issue_2d_wg(kDescA, bar_id(s), stage_A(s), s * ctx::tileK, tile_row);
      vx_dxa_issue_2d_wg(kDescB, bar_id(s), stage_B(s), tile_col, s * ctx::tileK);
    }
  }

  for (uint32_t i = 0; i < num_steps; ++i) {
    const uint32_t cur         = i % kPipeN;
    const uint32_t issue_i     = i + kPipeN - 1;
    const bool     has_issue   = (issue_i < num_steps);
    const uint32_t issue_stage = issue_i % kPipeN;
    const uint32_t issue_k     = issue_i * ctx::tileK;

    // (1) Keep the pipeline full: issue the fetch kPipeN-1 steps ahead.
    if (has_issue && is_dxa_warp) {
      vx_barrier_expect_tx(bar_id(issue_stage), 2);
      vx_dxa_issue_2d_wg(kDescA, bar_id(issue_stage), stage_A(issue_stage), issue_k, tile_row);
      vx_dxa_issue_2d_wg(kDescB, bar_id(issue_stage), stage_B(issue_stage), tile_col, issue_k);
    }

    // (2) Wait for the current stage's DXA completion.
    vx_barrier(bar_id(cur), num_warps);

    // (3) Compute on the current stage.
    auto A_warp = stage_A(cur) + warp_rank * ctx::xtileM * ctx::tileK;
    auto desc_b = vt::vx_make_smem_desc(stage_B(cur), 0);
  #if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
    ctx::fragment_a fragA;
    ctx::load_matrix_sync(fragA, A_warp, ctx::tileK);
    ctx::wgmma_sync(fragC, fragA, desc_b, fragC);
  #else
    auto desc_a = vt::vx_make_smem_desc(A_warp, ctx::tileK * sizeof(ctx::input_t));
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);
  #endif

    // (4) Gate stage reuse: all warps done reading before its next refetch.
    vx_barrier(bar_id(cur), num_warps);
  }
#elif defined(DOUBLE_BUF) && !defined(SW_LOAD_A) && !defined(SW_LOAD_B)
  // ── Double-buffered pipeline (arithmetic addressing) ────────────────────
  // Two SMEM buffers + two transaction barriers ping-pong so the DXA fetch of
  // tile k+1 overlaps the WGMMA compute of tile k, exactly as the original
  // DOUBLE_BUF did -- but unlike that version, every buffer address here is
  // computed by plain arithmetic (`smem + stage*stage_elems + ...`) from the
  // scalar `cur`/`nxt`, never through a `ctx::input_t* A_sm[2]`-style array
  // indexed by a runtime variable. That array was the confirmed root cause
  // of the original version's regression: the compiler spilled it to the
  // stack and reloaded it every iteration (4 extra `lw`/iter, verified in
  // disassembly). This version keeps all `num_warps` warps computing (no
  // warp is sacrificed as a dedicated producer, unlike WARP_SPEC) -- it is
  // the "no known bug" double-buffer: same overlap, same warp count as
  // baseline, register-spill-free addressing.
  const uint32_t stage_elems = cta_M * ctx::tileK + ctx::tileK * ctx::xtileN;

  auto stage_A = [&](uint32_t s) { return smem + s * stage_elems; };
  auto stage_B = [&](uint32_t s) { return smem + s * stage_elems + cta_M * ctx::tileK; };

  // Second register-spill fix, same idea as stage_A/stage_B above: the
  // original `vortex::barrier bar[2] = {barrier(0), barrier(1)}` was itself
  // a 2-element array of objects indexed by the runtime `cur`/`nxt`, and the
  // compiler spilled *that* to the stack the same way it spilled the raw
  // A_sm/B_sm pointer array (confirmed in disassembly: 2 `lw`/iter reading a
  // {bar_id_, num_warps_} struct off a stack table). vortex::barrier's own
  // id encoding is `(ctor_id << 8) + get_local_group_id()` -- since
  // ctor_id is exactly what `cur`/`nxt` already is (0 or 1), the id for
  // stage `s` is just `cta_id | (s << 8)`, computed inline with no array
  // and no class object at all. Calls the same underlying intrinsics
  // vortex::barrier::expect_tx()/arrive_and_wait() wrap.
  const uint32_t cta_id = get_local_group_id();
  auto bar_id = [&](uint32_t s) { return cta_id | (s << 8); };

  uint32_t cur = 0;

  // Prologue: prefetch tile k=0 into buffer 0.
  if (is_dxa_warp) {
    vx_barrier_expect_tx(bar_id(0), 2);
    vx_dxa_issue_2d_wg(kDescA, bar_id(0), stage_A(0), 0, tile_row);
    vx_dxa_issue_2d_wg(kDescB, bar_id(0), stage_B(0), tile_col, 0);
  }

  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    const uint32_t next_k   = k + ctx::tileK;
    const bool     has_next = (next_k < K);
    const uint32_t nxt      = cur ^ 1u;

    // (1) Prefetch next tile into the other buffer (async, overlaps compute).
    if (has_next && is_dxa_warp) {
      vx_barrier_expect_tx(bar_id(nxt), 2);
      vx_dxa_issue_2d_wg(kDescA, bar_id(nxt), stage_A(nxt), next_k, tile_row);
      vx_dxa_issue_2d_wg(kDescB, bar_id(nxt), stage_B(nxt), tile_col, next_k);
    }

    // (2) Wait for the current buffer's DXA completion.
    vx_barrier(bar_id(cur), num_warps);

    // (3) Compute on the current buffer.
    auto A_warp = stage_A(cur) + warp_rank * ctx::xtileM * ctx::tileK;
    auto desc_b = vt::vx_make_smem_desc(stage_B(cur), 0);
  #if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
    ctx::fragment_a fragA;
    ctx::load_matrix_sync(fragA, A_warp, ctx::tileK);
    ctx::wgmma_sync(fragC, fragA, desc_b, fragC);
  #else
    auto desc_a = vt::vx_make_smem_desc(A_warp, ctx::tileK * sizeof(ctx::input_t));
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);
  #endif

    // (4) Gate buffer reuse: all warps done reading before next DXA overwrites.
    vx_barrier(bar_id(cur), num_warps);
    cur = nxt;
  }
#else
  // Transaction barrier for DXA completion + CTA synchronization.
  vortex::barrier bar(0);

  // Loop over K tiles.
  for (uint32_t k = 0; k < K; k += ctx::tileK) {
    // DXA: load A tile [tile_row .. tile_row+cta_M, k .. k+tileK] into A_smem
    // DXA: load B tile [k .. k+tileK, tile_col .. tile_col+tileN] into B_smem
    // SW_LOAD_B replaces B's DXA with a cooperative SW load (K-major);
    // SW_LOAD_A replaces A's DXA with a cooperative SW load (row-major).
    {
    #if defined(SW_LOAD_A) && defined(SW_LOAD_B)
      // both via SW — no DXA needed
    #elif defined(SW_LOAD_A)
      if (is_dxa_warp) { bar.expect_tx(1); vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem, tile_col, k); }
    #elif defined(SW_LOAD_B)
      if (is_dxa_warp) { bar.expect_tx(1); vx_dxa_issue_2d_wg(kDescA, bar.id(), A_smem, k, tile_row); }
    #else
      if (is_dxa_warp) {
        bar.expect_tx(2);  // Two pending transactions: A + B
        vx_dxa_issue_2d_wg(kDescA, bar.id(), A_smem, k, tile_row);
        vx_dxa_issue_2d_wg(kDescB, bar.id(), B_smem, tile_col, k);
      }
    #endif
    #ifdef SW_LOAD_A
      // Cooperative load A row-major (matches DXA row-major layout).
      uint32_t a_size = cta_M * ctx::tileK;
      for (uint32_t i = 0; i < a_size; i += num_threads) {
        uint32_t idx = i + tid;
        uint32_t r = idx / ctx::tileK;
        uint32_t c = idx % ctx::tileK;
        A_smem[r * ctx::tileK + c] = pA[(tile_row + r) * K + (k + c)];
      }
    #endif
    #ifdef SW_LOAD_B
      // Cooperative load B block-major (matches DXA BlockMajor LAYOUT / bbuf).
      uint32_t b_size = ctx::tileK * ctx::xtileN;
      for (uint32_t i = 0; i < b_size; i += num_threads) {
        uint32_t idx = i + tid;
        uint32_t r = idx / ctx::xtileN;
        uint32_t c = idx % ctx::xtileN;
        B_smem[ctx::b_blockmajor_idx(r, c)] = pB[(k + r) * N + (tile_col + c)];
      }
    #endif
    }

    // Wait for DXA completion (all warps participate).
    bar.arrive_and_wait();

    auto A_warp = A_smem + warp_rank * ctx::xtileM * ctx::tileK;
    // B layout in SMEM: block-major (bbuf-native); stride field unused.
    auto desc_b = vt::vx_make_smem_desc(B_smem, 0);

  #if defined(WGMMA_RS) && (WGMMA_NRC <= 16)
    // RS: A from registers, B from smem (NRC <= 16 only)
    ctx::fragment_a fragA;
    ctx::load_matrix_sync(fragA, A_warp, ctx::tileK);
    ctx::wgmma_sync(fragC, fragA, desc_b, fragC);
  #else
    // SS: both from smem
    auto desc_a = vt::vx_make_smem_desc(A_warp, ctx::tileK * sizeof(ctx::input_t));
    ctx::wgmma_sync(fragC, desc_a, desc_b, fragC);
  #endif

    // Sync after WGMMA before next DXA overwrites smem.
    bar.arrive_and_wait();
  }
#endif

  // Store the computed C tile to global memory.
  auto pTileC = pC + (tile_row + warp_rank * ctx::xtileM) * N + tile_col;
  ctx::store_matrix_sync(pTileC, fragC, N);
}
