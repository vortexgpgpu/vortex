#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_tmem.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

namespace vt = vortex::tensor;
namespace tm = vortex::tmem;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;

// DXA descriptor slots (programmed by host in main.cpp).
constexpr uint32_t kDescA = 0;
constexpr uint32_t kDescB = 1;

#if (KB > 1) && !defined(B_KMAJOR)
#error "KB > 1 requires B_KMAJOR"
#endif

static inline __attribute__((always_inline)) uint32_t b_coord0(uint32_t k, uint32_t col) {
#ifdef B_KMAJOR
  (void)col; return k;
#else
  (void)k; return col;
#endif
}
static inline __attribute__((always_inline)) uint32_t b_coord1(uint32_t k, uint32_t col) {
#ifdef B_KMAJOR
  (void)k; return col;
#else
  (void)col; return k;
#endif
}

static_assert(sizeof(ctx::output_t) == sizeof(float), "the TMEM epilogue stores fp32 lanes");

constexpr uint32_t kPipeN = PIPE_N;
constexpr uint32_t kTileN = ctx::xtileN * TMEM_NCHUNK;  // CTA tile N
constexpr uint32_t kBK    = KB * ctx::tileK;  // K per pipeline stage
#ifdef B_KMAJOR
constexpr uint32_t kBLdmBytes = kBK * sizeof(ctx::input_t);
#else
constexpr uint32_t kBLdmBytes = 0;
#endif

// Full, empty and done barriers of a CTA.
static_assert(2 * kPipeN + 1 <= VX_CFG_NUM_BARRIERS, "not enough barriers for PIPE_N");


__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto pC = reinterpret_cast<ctx::output_t *>(arg->C_addr);

  uint32_t N = arg->N;
  uint32_t K = arg->K;

  uint32_t tid = threadIdx.x;
  uint32_t num_threads = blockDim.x;
  uint32_t warp_rank = tid / VX_CFG_NUM_THREADS;
  uint32_t num_warps = num_threads / VX_CFG_NUM_THREADS;

  // One warp per TCU block: block b computes rows [b*xtileM, (b+1)*xtileM).
  uint32_t cta_M = num_warps * ctx::xtileM;
  uint32_t tile_row = blockIdx.y * cta_M;
  uint32_t tile_col = blockIdx.x * kTileN;

  auto smem = reinterpret_cast<ctx::input_t *>(__local_mem());
  const uint32_t stage_elems = cta_M * kBK + kBK * kTileN;
  auto stage_A = [&](uint32_t s) { return smem + s * stage_elems; };
  auto stage_B = [&](uint32_t s) { return smem + s * stage_elems + cta_M * kBK; };

  const uint32_t cta_id = get_local_group_id();
  auto full_bar  = [&](uint32_t s) { return cta_id | (s << 8); };
  auto empty_bar = [&](uint32_t s) { return cta_id | ((kPipeN + s) << 8); };
  const uint32_t done_bar = cta_id | ((2 * kPipeN) << 8);

  // main.cpp sizes the shared-memory window to cover kTileN TMEM columns.
  const uint32_t taddr     = tm::cta_col_base(smem);
  const uint32_t num_steps = K / kBK;

  if (warp_rank == 0) {
    tm::tj_desc_t desc = {};
    desc.idesc        = tm::make_idesc(vt::ITYPE::id, vt::OTYPE::id, kTileN / ctx::tcN, true);
    desc.ctrl         = tm::make_ctrl(tm::TJ_CONSUME, kPipeN, KB, K / ctx::tileK);
    desc.taddr        = taddr;
    desc.a_base       = tm::smem_offset(stage_A(0));
    desc.b_base       = tm::smem_offset(stage_B(0));
    desc.stage_stride = stage_elems * sizeof(ctx::input_t);
    desc.ld           = (kBK * sizeof(ctx::input_t)) | (kBLdmBytes << 16);
    desc.full_bar     = full_bar(0);
    desc.empty_bar    = empty_bar(0);
    desc.done_bar     = done_bar;

  #ifdef TMEM_CLOSED
    // The TCU issues every stage load itself; no warp runs in the K loop.
    desc.ctrl      = tm::make_ctrl(tm::TJ_CLOSED, kPipeN, KB, K / ctx::tileK);
    desc.dxa_slots = kDescA | (kDescB << 4);
    desc.a_coord   = 0 | (tile_row << 16);
    desc.b_coord   = b_coord0(0, tile_col) | (b_coord1(0, tile_col) << 16);
  #ifdef B_KMAJOR
    desc.k_advance = kBK;                // K is coordinate 0 of A and of B
  #else
    desc.k_advance = kBK | (1u << 17);   // K is coordinate 1 of B
  #endif
    vx_barrier_expect_tx(done_bar, 1);
    tm::tj_post(desc);
    (void)num_steps;
  #else
  #ifndef TMEM_PERSTAGE
    vx_barrier_expect_tx(done_bar, 1);
    tm::tj_post(desc);
  #endif

    // Producer: this warp and the TCU are the two participants of every full
    // and empty barrier. The TCU arrives on empty[s] once it has fetched s.
    for (uint32_t i = 0; i < num_steps; ++i) {
      uint32_t s = i % kPipeN;
      if (i >= kPipeN) {
        int phase = vx_barrier_arrive(empty_bar(s), 2);
        vx_barrier_wait(empty_bar(s), phase);
      }
      uint32_t k = i * kBK;
      vx_barrier_expect_tx(full_bar(s), 2);
      vx_dxa_issue_2d_wg(kDescA, full_bar(s), stage_A(s), k, tile_row);
      vx_dxa_issue_2d_wg(kDescB, full_bar(s), stage_B(s), b_coord0(k, tile_col), b_coord1(k, tile_col));
      vx_barrier_arrive(full_bar(s), 2);
    }
    // Close the last generation of every empty barrier the TCU arrived on.
    uint32_t first = (num_steps > kPipeN) ? (num_steps - kPipeN) : 0;
    for (uint32_t i = first; i < num_steps; ++i) {
      uint32_t s = i % kPipeN;
      int phase = vx_barrier_arrive(empty_bar(s), 2);
      vx_barrier_wait(empty_bar(s), phase);
    }
  #endif
  }
#ifdef TMEM_PERSTAGE
  else if (warp_rank == 1) {
    // Blackwell-style MMA warp: wait on each stage, then post a one-stage job
    // that consumes it and releases it to the producer when fetched.
    tm::tj_desc_t desc = {};
    desc.taddr        = taddr;
    desc.a_base       = tm::smem_offset(stage_A(0));
    desc.b_base       = tm::smem_offset(stage_B(0));
    desc.stage_stride = stage_elems * sizeof(ctx::input_t);
    desc.ld           = (kBK * sizeof(ctx::input_t)) | (kBLdmBytes << 16);
    desc.full_bar     = full_bar(0);
    desc.empty_bar    = empty_bar(0);
    desc.done_bar     = done_bar;
    desc.ctrl         = tm::make_ctrl(tm::TJ_CONSUME, kPipeN, KB, KB, true);
    for (uint32_t i = 0; i < num_steps; ++i) {
      uint32_t s = i % kPipeN;
      int phase = vx_barrier_arrive(full_bar(s), 2);
      vx_barrier_wait(full_bar(s), phase);
      desc.idesc = tm::make_idesc(vt::ITYPE::id, vt::OTYPE::id, kTileN / ctx::tcN, i == 0);
      desc.stage = s;
      vx_barrier_expect_tx(done_bar, 1);
      tm::tj_post(desc);
    }
  }
#endif

  // Epilogue: wait for the job; warp r drains TMEM lanes (M rows)
  // [r*xtileM, (r+1)*xtileM), xtileN columns per load.
  vx_barrier(done_bar, num_warps);

  const uint32_t lane = warp_rank * ctx::xtileM;
  const uint32_t t = vx_thread_id();
  for (uint32_t c = 0; c < TMEM_NCHUNK; ++c) {
    ctx::fragment_acc fragC;
    uint32_t col = taddr + c * ctx::xtileN;
    auto pChunkC = pC + (tile_row + lane) * N + tile_col + c * ctx::xtileN;
  #ifdef TMEM_LANE_PER_THREAD
    // Thread t holds one M row; a store writes NT rows.
    constexpr uint32_t kThreadsPerLane = VX_CFG_NUM_THREADS / ctx::xtileM;
    tm::load<tm::LANE_PER_THREAD>(fragC, tm::make_taddr(lane, col));
    auto pRowC = pChunkC + (t % ctx::xtileM) * N + t / ctx::xtileM;
    for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r) {
      pRowC[r * kThreadsPerLane] = fragC.data[r];
    }
  #else
    // Register r holds NT contiguous columns of one M row; a store writes one row.
    tm::load<tm::LANE_PER_REG>(fragC, tm::make_taddr(lane, col));
    #pragma unroll
    for (uint32_t g = 0; g < ctx::fragment_acc::NR / ctx::xtileM; ++g) {
      auto p = pChunkC + g * VX_CFG_NUM_THREADS + t;
      #pragma unroll
      for (uint32_t i = 0; i < ctx::xtileM; ++i) {
        *p = fragC.data[g * ctx::xtileM + i];
        p += N;
      }
    }
  #endif
  }
}
