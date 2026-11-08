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

constexpr uint32_t kPipeN = 2;
constexpr uint32_t kBK    = KB * ctx::tileK;  // K per stage
constexpr uint32_t kNR    = ctx::fragment_acc::NR;
static_assert(VX_CFG_NUM_THREADS == ctx::xtileM, "one column per register");
static_assert(2 * kPipeN + 2 <= VX_CFG_NUM_BARRIERS, "not enough barriers");

// P's fp16 conversion needs a normal input; attention below this weight is negligible.
constexpr float kPMin = 6.2e-5f;

// The threads of a warp run in lockstep, so per-thread selection is branch-free.
static inline float fmax32(float a, float b) {
  float r;
  __asm__ volatile ("fmax.s %0, %1, %2" : "=f"(r) : "f"(a), "f"(b));
  return r;
}

static inline uint32_t f32_to_f16(uint32_t x) {
  uint32_t r = (((x >> 23) & 0xff) - 127 + 15) << 10 | ((x & 0x7fffff) >> 13);
  r += ((x & 0x1fff) + (r & 1) + 0xfff) >> 13;
  return ((x >> 16) & 0x8000) | r;
}

static inline uint32_t full_bar(uint32_t cta_id, uint32_t s) {
  return cta_id | (s << 8);
}

static inline uint32_t empty_bar(uint32_t cta_id, uint32_t s) {
  return cta_id | ((kPipeN + s) << 8);
}

static inline __attribute__((always_inline)) void wait_bar(uint32_t bar, uint32_t count) {
  int phase = vx_barrier_arrive(bar, count);
  vx_barrier_wait(bar, phase);
}

// K-tile j into stage j % kPipeN: Q rows from q_row and K_T rows from kv for S,
// V_T columns from kv for P.V.
static inline __attribute__((always_inline)) void load_stage(
    uint32_t cta_id, uint32_t j, bool pv, ctx::input_t* smem, uint32_t stage_elems,
    uint32_t q_elems, uint32_t q_row, uint32_t kv) {
  const uint32_t s = j % kPipeN;
  const uint32_t k = j * kBK;
  const uint32_t full = full_bar(cta_id, s);
  auto stage = smem + s * stage_elems;
  vx_barrier_expect_tx(full, pv ? 1 : 2);
  if (pv) {
    vx_dxa_issue_2d_wg(kDescVT, full, stage, kv + k, 0);
  } else {
    vx_dxa_issue_2d_wg(kDescQ, full, stage, k, q_row);
    vx_dxa_issue_2d_wg(kDescKT, full, stage + q_elems, k, kv);
  }
}

// One warp's K loop for an MMA into TMEM: S = Q.K_T stages Q then K_T, and
// O += P.V stages only V_T, as P is in TMEM. A stage is reloaded once the
// tc.mma that read it commits, while the next tc.mma runs.
static inline __attribute__((always_inline)) void mma_loop(
    uint32_t cta_id, uint32_t n, bool pv, uint32_t idesc, bool acc_init, uint32_t taddr,
    uint32_t p_col, ctx::input_t* smem, uint32_t stage_elems, uint32_t q_elems,
    uint32_t q_row, uint32_t kv) {
  constexpr uint32_t kIdescAccInit = 1u << 11;
  const uint32_t smem_base = tm::smem_offset(smem);
  const uint32_t stage_bytes = stage_elems * sizeof(ctx::input_t);
  for (uint32_t j = 0; j < n && j < kPipeN; ++j) {
    load_stage(cta_id, j, pv, smem, stage_elems, q_elems, q_row, kv);
  }
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t s = i % kPipeN;
    wait_bar(full_bar(cta_id, s), 1);
    tm::mma_desc_t d = {};
    d.idesc   = idesc | ((acc_init && i == 0) ? kIdescAccInit : 0);
    d.k_tiles = KB;
    d.taddr   = taddr;
    d.a_base  = pv ? (p_col + i * KB * wg_cfg::fedpK) : (smem_base + s * stage_bytes);
    d.b_base  = smem_base + s * stage_bytes + (pv ? 0 : q_elems * sizeof(ctx::input_t));
    d.ld      = (kBK * sizeof(ctx::input_t)) | ((kBK * sizeof(ctx::input_t)) << 16);
    tm::mma(d);
    tm::commit(empty_bar(cta_id, s), 2);
    if (i >= 1 && i - 1 + kPipeN < n) {
      wait_bar(empty_bar(cta_id, (i - 1) % kPipeN), 2);
      load_stage(cta_id, i - 1 + kPipeN, pv, smem, stage_elems, q_elems, q_row, kv);
    }
  }
  // Close the empty generations no reload waited on; the last ends with the last MMA.
  for (uint32_t i = (n > kPipeN) ? (n - kPipeN) : 0; i < n; ++i) {
    wait_bar(empty_bar(cta_id, i % kPipeN), 2);
  }
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t L = arg->L, N = arg->N, D = arg->D, NB = arg->NB;
  const float scale = arg->scale;
  auto pO = reinterpret_cast<ctx::output_t*>(arg->O_addr);

  const uint32_t warp_rank = threadIdx.x / VX_CFG_NUM_THREADS;
  const uint32_t num_warps = blockDim.x / VX_CFG_NUM_THREADS;
  const uint32_t lane = warp_rank * ctx::xtileM;
  const uint32_t t = vx_thread_id();

  auto lmem = reinterpret_cast<uint8_t*>(__local_mem());
  auto tmem_slot = reinterpret_cast<volatile uint32_t*>(lmem);
  auto smem = reinterpret_cast<ctx::input_t*>(lmem + TMEM_SLOT_BYTES);
  const uint32_t cta_M = num_warps * ctx::xtileM;
  const uint32_t cta_row = blockIdx.x * cta_M;
  const uint32_t q_elems = cta_M * kBK;
  const uint32_t stage_elems = q_elems + kBK * (L > N ? L : N);

  const uint32_t cta_id = get_local_group_id();
  const uint32_t done_bar  = cta_id | ((2 * kPipeN) << 8);
  const uint32_t step_bar  = cta_id | ((2 * kPipeN + 1) << 8);

  // TMEM: S, then P (fp16 pairs), then O.
  const uint32_t tmem_cols = tm::alloc_cols(L + L / 2 + N);
  if (warp_rank == 0) {
    // Every thread holds the same column; a store per thread avoids a divergent branch.
    *tmem_slot = tm::alloc(tmem_cols);
  }
  vx_barrier(step_bar, num_warps);
  const uint32_t s_col = *tmem_slot;
  const uint32_t p_col = s_col + L;
  const uint32_t o_col = p_col + (L / 2);

  // One GEMM into TMEM, run by warp 0: S = Q . K^T, or O += P . V^T with A = P.
  const uint32_t s_idesc = tm::make_idesc(vt::ITYPE::id, vt::OTYPE::id, L / ctx::tcN, false);
  const uint32_t o_idesc = tm::make_idesc(vt::ITYPE::id, vt::OTYPE::id, N / ctx::tcN, false, false, true);
  auto gemm = [&](bool pv, uint32_t b) {
    if (warp_rank == 0) {
      mma_loop(cta_id, (pv ? L : D) / kBK, pv, pv ? o_idesc : s_idesc, !pv || b == 0, pv ? o_col : s_col,
               p_col, smem, stage_elems, q_elems, cta_row, b * L);
    }
    vx_barrier(done_bar, num_warps);
  };

  float row_max = 0.0f;
  float row_sum = 0.0f;

  for (uint32_t b = 0; b < NB; ++b) {
    gemm(false, b);

    // Online softmax: this thread's row maximum, over every block so far.
    float new_max = row_max;
    for (uint32_t c = 0; c < L / kNR; ++c) {
      ctx::fragment_acc s;
      tm::load<tm::LANE_PER_THREAD>(s, tm::make_taddr(lane, s_col + c * kNR));
      for (uint32_t r = 0; r < kNR; ++r) {
        float v = s.data[r] * scale;
        new_max = (c == 0 && r == 0 && b == 0) ? v : fmax32(new_max, v);
      }
    }
    const float rescale = (b == 0) ? 1.0f : vx_expf_fast(row_max - new_max);
    row_max = new_max;
    row_sum *= rescale;

    // P = exp(S - max) as fp16, K-packed for the P.V MMA.
    for (uint32_t c = 0; c < L / (2 * kNR); ++c) {
      ctx::fragment_acc s0, s1, p;
      uint32_t col = s_col + c * 2 * kNR;
      tm::load<tm::LANE_PER_THREAD>(s0, tm::make_taddr(lane, col));
      tm::load<tm::LANE_PER_THREAD>(s1, tm::make_taddr(lane, col + kNR));
      for (uint32_t r = 0; r < kNR; ++r) {
        float lo = (r < kNR / 2) ? s0.data[2 * r] : s1.data[2 * r - kNR];
        float hi = (r < kNR / 2) ? s0.data[2 * r + 1] : s1.data[2 * r + 1 - kNR];
        lo = fmax32(vx_expf_fast(lo * scale - new_max), kPMin);
        hi = fmax32(vx_expf_fast(hi * scale - new_max), kPMin);
        row_sum += lo + hi;
        uint32_t lo_bits, hi_bits;
        __builtin_memcpy(&lo_bits, &lo, 4);
        __builtin_memcpy(&hi_bits, &hi, 4);
        uint32_t word = f32_to_f16(lo_bits) | (f32_to_f16(hi_bits) << 16);
        __builtin_memcpy(&p.data[r], &word, 4);
      }
      tm::store<tm::LANE_PER_THREAD>(tm::make_taddr(lane, p_col + c * kNR), p);
    }

    if (b != 0) {
      for (uint32_t c = 0; c < N / kNR; ++c) {
        ctx::fragment_acc o;
        uint32_t taddr = tm::make_taddr(lane, o_col + c * kNR);
        tm::load<tm::LANE_PER_THREAD>(o, taddr);
        for (uint32_t r = 0; r < kNR; ++r) {
          o.data[r] *= rescale;
        }
        tm::store<tm::LANE_PER_THREAD>(taddr, o);
      }
    }
    tm::wait_st();
    vx_barrier(step_bar, num_warps);

    // O += P . V^T: A from TMEM, only V is staged.
    gemm(true, b);
  }

  const float inv_sum = vx_rcpf(row_sum);
  for (uint32_t c = 0; c < N / kNR; ++c) {
    ctx::fragment_acc o;
    tm::load<tm::LANE_PER_THREAD>(o, tm::make_taddr(lane, o_col + c * kNR));
    for (uint32_t r = 0; r < kNR; ++r) {
      pO[(cta_row + lane + t) * N + c * kNR + r] = o.data[r] * inv_sum;
    }
  }

  vx_barrier(step_bar, num_warps);
  if (warp_rank == 0) {
    tm::dealloc(s_col, tmem_cols);
  }
}
