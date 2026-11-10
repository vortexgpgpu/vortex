#include "common.h"
#include <vx_spawn2.h>
#include <vx_tensor.h>
#include <vx_tmem.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include <vx_mufu.h>

namespace vt = vortex::tensor;
namespace tm = vortex::tmem;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, false, WGMMA_NRC>;
using wg_cfg = vt::wgmma_config_t<VX_CFG_NUM_THREADS, vt::ITYPE, vt::OTYPE, WGMMA_NRC>;

constexpr uint32_t kDescQ  = 0;
constexpr uint32_t kDescKT = 1;
constexpr uint32_t kDescVT = 2;

constexpr uint32_t kPipeN = PIPE_N;
constexpr uint32_t kBK    = KB * ctx::tileK;  // K per stage
static_assert(VX_CFG_NUM_THREADS == ctx::xtileM, "one column per register");

// tmem.ld/st use fixed registers f0.., so the softmax moves 8 columns at a time
// to leave the rest of the FP registers to the compiler.
struct frag8_t {
  static constexpr uint32_t NR = 8;
  float data[NR];
};
constexpr uint32_t kNR = frag8_t::NR;

// The control warps (CTRL_WARPS) stage the operands and issue tc.mma; each of
// the two Q tiles has a softmax group of one warp per TCU block, so an MMA's M
// rows are the group's TMEM lanes.
constexpr uint32_t kTiles      = 2;
constexpr uint32_t kGroupWarps = VX_CFG_NUM_TCU_BLOCKS;
constexpr uint32_t kTileM      = kGroupWarps * ctx::xtileM;
constexpr uint32_t kTileCols   = 128;  // S, P and O of a tile
constexpr uint32_t kWarps      = CTRL_WARPS + kTiles * kGroupWarps;

// Stages (full, empty), then per tile: its MMAs done, its softmax done.
constexpr uint32_t kBarDone = 2 * kPipeN;
constexpr uint32_t kBarSm   = kBarDone + kTiles;
static_assert(kBarSm + kTiles <= VX_CFG_NUM_BARRIERS, "not enough barriers");

// P's fp16 conversion needs a normal input; attention below this weight is negligible.
constexpr float kPMin = 6.2e-5f;
// Softmax runs in base 2. O is rescaled only once a row maximum has grown by
// more than log2(256); until then P is relative to the stale maximum and stays
// below 256.
constexpr float kLog2e      = 1.44269504088896341f;
constexpr float kRescaleTau = 8.0f;

static inline float fmax32(float a, float b) {
  float r;
  __asm__ volatile ("fmax.s %0, %1, %2" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

static inline float max8(const float* v) {
  float a = fmax32(fmax32(v[0], v[1]), fmax32(v[2], v[3]));
  float b = fmax32(fmax32(v[4], v[5]), fmax32(v[6], v[7]));
  return fmax32(a, b);
}

// fp32 -> fp16 bits for a positive normal fp16 value, rounding half up.
static inline uint32_t f32_to_f16_pos(float x) {
  uint32_t b;
  __builtin_memcpy(&b, &x, 4);
  return ((b + 0x1000u) >> 13) - 0x1C000u;
}

// Per stage of K: wait for it, multiply, and commit it back.
static inline __attribute__((always_inline)) void mma_steps(
    uint32_t& i, uint32_t K, uint32_t cta_id, bool acc_init, uint32_t taddr,
    uint32_t a_col, bool a_tmem, uint32_t smem_base, uint32_t stage_bytes, uint32_t b_off) {
  for (uint32_t k = 0; k < K; k += kBK, ++i) {
    const uint32_t s = i % kPipeN;
    const uint32_t full = cta_id | (s << 8);
    int phase = vx_barrier_arrive(full, 2);
    vx_barrier_wait(full, phase);
    const uint32_t b = smem_base + s * stage_bytes + b_off;
    if (a_tmem) {
      tm::mma<1>(taddr, a_col + (k / kBK) * KB * wg_cfg::k_steps * wg_cfg::fedpK, b, acc_init && k == 0);
    } else {
      tm::mma<0>(taddr, smem_base + s * stage_bytes, b, acc_init && k == 0);
    }
    tm::commit(cta_id | ((kPipeN + s) << 8), 2);
  }
}

static inline __attribute__((always_inline)) void wait_empty(uint32_t i, uint32_t cta_id) {
  if (i >= kPipeN) {
    uint32_t e = cta_id | ((kPipeN + i % kPipeN) << 8);
    int phase = vx_barrier_arrive(e, 2);
    vx_barrier_wait(e, phase);
  }
}

// S stages: Q rows [q_row, +kTileM) and K_T rows [kv_row, +L), over D.
static inline __attribute__((always_inline)) void load_s(
    uint32_t& i, uint32_t D, uint32_t cta_id, ctx::input_t* smem, uint32_t stage_elems,
    uint32_t q_row, uint32_t kv_row) {
  for (uint32_t k = 0; k < D; k += kBK, ++i) {
    wait_empty(i, cta_id);
    const uint32_t s = i % kPipeN;
    const uint32_t full = cta_id | (s << 8);
    auto stage = smem + s * stage_elems;
    vx_barrier_expect_tx(full, 2);
    vx_dxa_issue_2d_wg(kDescQ, full, stage, k, q_row);
    vx_dxa_issue_2d_wg(kDescKT, full, stage + kTileM * kBK, k, kv_row);
    vx_barrier_arrive(full, 2);
  }
}

// P.V stages: V_T columns [kv_col, +L).
static inline __attribute__((always_inline)) void load_pv(
    uint32_t& i, uint32_t L, uint32_t cta_id, ctx::input_t* smem, uint32_t stage_elems, uint32_t kv_col) {
  for (uint32_t k = 0; k < L; k += kBK, ++i) {
    wait_empty(i, cta_id);
    const uint32_t s = i % kPipeN;
    const uint32_t full = cta_id | (s << 8);
    vx_barrier_expect_tx(full, 1);
    vx_dxa_issue_2d_wg(kDescVT, full, smem + s * stage_elems, kv_col + k, 0);
    vx_barrier_arrive(full, 2);
  }
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t L = arg->L, N = arg->N, D = arg->D, NB = arg->NB;
  const float scale = arg->scale;
  auto pO = reinterpret_cast<ctx::output_t*>(arg->O_addr);

  const uint32_t warp_rank = threadIdx.x / VX_CFG_NUM_THREADS;
  const uint32_t t = vx_thread_id();

  auto lmem = reinterpret_cast<uint8_t*>(__local_mem());
  auto tmem_slot = reinterpret_cast<volatile uint32_t*>(lmem);
  auto smem = reinterpret_cast<ctx::input_t*>(lmem + TMEM_SLOT_BYTES);
  const uint32_t cta_row = blockIdx.x * kTiles * kTileM;

  const uint32_t cta_id = get_local_group_id();
  auto bar = [&](uint32_t n) { return cta_id | (n << 8); };

  // Every warp meets on the first softmax barrier before its first use, to
  // share the TMEM columns and, at the end, to free them.
  const uint32_t all_bar = bar(kBarSm);
  if (warp_rank == 0) {
    // Every thread holds the same column; a store per thread avoids a divergent branch.
    *tmem_slot = tm::alloc(kTiles * kTileCols);
  }
  vx_barrier(all_bar, kWarps);
  const uint32_t col_base = *tmem_slot;
  auto s_col = [&](uint32_t tile) { return col_base + tile * kTileCols; };
  auto p_col = [&](uint32_t tile) { return s_col(tile) + L; };
  auto o_col = [&](uint32_t tile) { return p_col(tile) + L / 2; };

  // Both MMAs share one stage ring: S = Q.K_T stages A (Q) then B (K_T), and
  // O += P.V stages only B (V_T), as P is in TMEM.
  const uint32_t stage_elems = kTileM * kBK + kBK * (L > N ? L : N);
  const uint32_t smem_base   = tm::smem_offset(smem);
  const uint32_t q_bytes     = kTileM * kBK * sizeof(ctx::input_t);
  const uint32_t stage_bytes = stage_elems * sizeof(ctx::input_t);

  // MMAs in issue order. Ping-pong: while one tile's softmax runs, the TCU
  // works on the other's. The steps take plain values so the per-K-tile code
  // stays in registers across the asm memory clobbers.
  if (warp_rank == 0) {
    // Producer: it and the MMA warp are the two participants of every full and
    // empty barrier. Loads run ahead of the softmax, bounded by the ring.
    const uint32_t q_row = cta_row;
    uint32_t i = 0;
    for (uint32_t tile = 0; tile < kTiles; ++tile) {
      load_s(i, D, cta_id, smem, stage_elems, q_row + tile * kTileM, 0);
    }
    for (uint32_t b = 0; b < NB; ++b) {
      for (uint32_t tile = 0; tile < kTiles; ++tile) {
        load_pv(i, L, cta_id, smem, stage_elems, b * L);
        if (b + 1 < NB) {
          load_s(i, D, cta_id, smem, stage_elems, q_row + tile * kTileM, (b + 1) * L);
        }
      }
    }
    // Close the last generation of every empty barrier the MMA warp committed.
    for (uint32_t j = (i > kPipeN) ? (i - kPipeN) : 0; j < i; ++j) {
      uint32_t e = cta_id | ((kPipeN + j % kPipeN) << 8);
      int phase = vx_barrier_arrive(e, 2);
      vx_barrier_wait(e, phase);
    }
    vx_barrier(all_bar, kWarps);
    tm::dealloc(col_base, kTiles * kTileCols);
    return;
  }
  if (warp_rank == 1) {
    // A tile's softmax waits on the commit after its S MMA (or its last P.V).
    constexpr uint32_t kLd = tm::make_ld(kBK * sizeof(ctx::input_t), kBK * sizeof(ctx::input_t));
    tm::set_idesc<0>(tm::make_idesc(vt::ITYPE::id, vt::OTYPE::id, L / ctx::tcN, KB * wg_cfg::k_steps), kLd);
    tm::set_idesc<1>(tm::make_idesc(vt::ITYPE::id, vt::OTYPE::id, N / ctx::tcN, KB * wg_cfg::k_steps, false, true), kLd);
    uint32_t i = 0;
    for (uint32_t tile = 0; tile < kTiles; ++tile) {
      mma_steps(i, D, cta_id, true, s_col(tile), 0, false, smem_base, stage_bytes, q_bytes);
      tm::commit(bar(kBarDone + tile), 1 + kGroupWarps);
    }
    for (uint32_t b = 0; b < NB; ++b) {
      for (uint32_t tile = 0; tile < kTiles; ++tile) {
        vx_barrier(bar(kBarSm + tile), 1 + kGroupWarps);
        mma_steps(i, L, cta_id, b == 0, o_col(tile), p_col(tile), true, smem_base, stage_bytes, 0);
        if (b + 1 < NB) {
          mma_steps(i, D, cta_id, true, s_col(tile), 0, false, smem_base, stage_bytes, q_bytes);
        }
        tm::commit(bar(kBarDone + tile), 1 + kGroupWarps);
      }
    }
    vx_barrier(all_bar, kWarps);
    return;
  }

  const uint32_t tile = (warp_rank - CTRL_WARPS) / kGroupWarps;
  const uint32_t lane = ((warp_rank - CTRL_WARPS) % kGroupWarps) * ctx::xtileM;
  const uint32_t done_bar = bar(kBarDone + tile);
  const uint32_t sm_bar   = bar(kBarSm + tile);

  // The scale is positive, so the row maximum of S scales with it.
  const float scale2 = scale * kLog2e;
  float row_max = 0.0f;
  float row_sum = 0.0f;

  for (uint32_t b = 0; b < NB; ++b) {
    // S of block b, and O through block b - 1.
    vx_barrier(done_bar, 1 + kGroupWarps);

    float smax = -3.0e38f;
    for (uint32_t c = 0; c < L / kNR; ++c) {
      frag8_t s;
      tm::load<tm::LANE_PER_THREAD>(s, tm::make_taddr(lane, s_col(tile) + c * kNR));
      smax = fmax32(smax, max8(s.data));
    }
    const float cand = (b == 0) ? smax * scale2 : fmax32(row_max, smax * scale2);
    if (b == 0) {
      row_max = cand;
    } else if (vx_vote_any(cand - row_max > kRescaleTau)) {
      const float rescale = vx_ex2f(row_max - cand);
      row_max = cand;
      row_sum *= rescale;
      for (uint32_t c = 0; c < N / kNR; ++c) {
        frag8_t o;
        uint32_t taddr = tm::make_taddr(lane, o_col(tile) + c * kNR);
        tm::load<tm::LANE_PER_THREAD>(o, taddr);
        #pragma unroll
        for (uint32_t r = 0; r < kNR; ++r) {
          o.data[r] *= rescale;
        }
        tm::store<tm::LANE_PER_THREAD>(taddr, o);
      }
    }

    // P = exp(S - max) as fp16, K-packed for the P.V MMA.
    for (uint32_t c = 0; c < L / (2 * kNR); ++c) {
      frag8_t s0, s1, p;
      uint32_t col = s_col(tile) + c * 2 * kNR;
      tm::load<tm::LANE_PER_THREAD>(s0, tm::make_taddr(lane, col));
      tm::load<tm::LANE_PER_THREAD>(s1, tm::make_taddr(lane, col + kNR));
      float psum[kNR / 2] = {};
      #pragma unroll
      for (uint32_t r = 0; r < kNR; ++r) {
        float lo = (r < kNR / 2) ? s0.data[2 * r] : s1.data[2 * r - kNR];
        float hi = (r < kNR / 2) ? s0.data[2 * r + 1] : s1.data[2 * r + 1 - kNR];
        lo = fmax32(vx_ex2f(lo * scale2 - row_max), kPMin);
        hi = fmax32(vx_ex2f(hi * scale2 - row_max), kPMin);
        psum[r % (kNR / 2)] += lo + hi;
        uint32_t word = f32_to_f16_pos(lo) | (f32_to_f16_pos(hi) << 16);
        __builtin_memcpy(&p.data[r], &word, 4);
      }
      tm::store<tm::LANE_PER_THREAD>(tm::make_taddr(lane, p_col(tile) + c * kNR), p);
      row_sum += (psum[0] + psum[1]) + (psum[2] + psum[3]);
    }
    tm::wait_st();
    vx_barrier_arrive(sm_bar, 1 + kGroupWarps);
  }

  vx_barrier(done_bar, 1 + kGroupWarps);
  const float inv_sum = vx_rcpf(row_sum);
  const uint32_t row = cta_row + tile * kTileM + lane + t;
  for (uint32_t c = 0; c < N / kNR; ++c) {
    frag8_t o;
    tm::load<tm::LANE_PER_THREAD>(o, tm::make_taddr(lane, o_col(tile) + c * kNR));
    #pragma unroll
    for (uint32_t r = 0; r < kNR; ++r) {
      pO[row * N + c * kNR + r] = o.data[r] * inv_sum;
    }
  }
  vx_barrier(all_bar, kWarps);
}
