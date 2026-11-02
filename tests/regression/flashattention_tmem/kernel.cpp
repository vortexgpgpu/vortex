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

constexpr uint32_t kDescQ  = 0;
constexpr uint32_t kDescKT = 1;
constexpr uint32_t kDescVT = 2;

constexpr uint32_t kPipeN = 2;
constexpr uint32_t kBK    = ctx::tileK;
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

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t L = arg->L, N = arg->N, D = arg->D, NB = arg->NB;
  const float scale = arg->scale;
  auto pO = reinterpret_cast<ctx::output_t*>(arg->O_addr);

  const uint32_t warp_rank = threadIdx.x / VX_CFG_NUM_THREADS;
  const uint32_t num_warps = blockDim.x / VX_CFG_NUM_THREADS;
  const uint32_t lane = warp_rank * ctx::xtileM;
  const uint32_t t = vx_thread_id();

  auto smem = reinterpret_cast<ctx::input_t*>(__local_mem());
  const uint32_t cta_M = num_warps * ctx::xtileM;
  const uint32_t s1_elems = cta_M * kBK + kBK * L;
  const uint32_t s2_elems = kBK * N;

  const uint32_t cta_id = get_local_group_id();
  const uint32_t full_bar  = cta_id;
  const uint32_t empty_bar = cta_id | (kPipeN << 8);
  const uint32_t done_bar  = cta_id | ((2 * kPipeN) << 8);
  const uint32_t step_bar  = cta_id | ((2 * kPipeN + 1) << 8);

  const uint32_t s_col = tm::cta_col_base(smem);
  const uint32_t p_col = s_col + L;
  const uint32_t o_col = p_col + (L / 2);

  float row_max = 0.0f;
  float row_sum = 0.0f;

  for (uint32_t b = 0; b < NB; ++b) {
    // S = Q . K^T
    if (warp_rank == 0) {
      tm::tj_desc_t d = {};
      d.idesc        = tm::make_idesc(vt::ITYPE::id, vt::OTYPE::id, L / ctx::tcN, true);
      d.ctrl         = tm::make_ctrl(tm::TJ_CLOSED, kPipeN, 1, D / ctx::tileK);
      d.taddr        = s_col;
      d.a_base       = tm::smem_offset(smem);
      d.b_base       = tm::smem_offset(smem + cta_M * kBK);
      d.stage_stride = s1_elems * sizeof(ctx::input_t);
      d.ld           = (kBK * sizeof(ctx::input_t)) | ((kBK * sizeof(ctx::input_t)) << 16);
      d.full_bar     = full_bar;
      d.empty_bar    = empty_bar;
      d.done_bar     = done_bar;
      d.dxa_slots    = kDescQ | (kDescKT << 4);
      d.b_coord      = (b * L) << 16;
      d.k_advance    = kBK;
      vx_barrier_expect_tx(done_bar, 1);
      tm::tj_post(d);
    }
    vx_barrier(done_bar, num_warps);

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

    // P = exp(S - max) as fp16, K-packed for the next job.
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
    vx_barrier(step_bar, num_warps);

    // O += P . V^T: A from TMEM, only V is staged.
    if (warp_rank == 0) {
      tm::tj_desc_t d = {};
      d.idesc        = tm::make_idesc(vt::ITYPE::id, vt::OTYPE::id, N / ctx::tcN, b == 0, false, true);
      d.ctrl         = tm::make_ctrl(tm::TJ_CLOSED, kPipeN, 1, L / ctx::tileK);
      d.taddr        = o_col;
      d.a_base       = p_col;
      d.b_base       = tm::smem_offset(smem);
      d.stage_stride = s2_elems * sizeof(ctx::input_t);
      d.ld           = (kBK * sizeof(ctx::input_t)) | ((kBK * sizeof(ctx::input_t)) << 16);
      d.full_bar     = full_bar;
      d.empty_bar    = empty_bar;
      d.done_bar     = done_bar;
      d.dxa_slots    = (kDescVT << 4);
      d.b_coord      = b * L;
      d.k_advance    = kBK;
      vx_barrier_expect_tx(done_bar, 1);
      tm::tj_post(d);
    }
    vx_barrier(done_bar, num_warps);
  }

  const float inv_sum = vx_rcpf(row_sum);
  for (uint32_t c = 0; c < N / kNR; ++c) {
    ctx::fragment_acc o;
    tm::load<tm::LANE_PER_THREAD>(o, tm::make_taddr(lane, o_col + c * kNR));
    for (uint32_t r = 0; r < kNR; ++r) {
      pO[(lane + t) * N + c * kNR + r] = o.data[r] * inv_sum;
    }
  }
}
