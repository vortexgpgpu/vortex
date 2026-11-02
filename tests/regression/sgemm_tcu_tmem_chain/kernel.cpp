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

constexpr uint32_t kDescQ  = 0;
constexpr uint32_t kDescKT = 1;
constexpr uint32_t kDescVT = 2;

constexpr uint32_t kPipeN = 2;
constexpr uint32_t kBK    = ctx::tileK;
constexpr uint32_t kNR    = ctx::fragment_acc::NR;
static_assert(VX_CFG_NUM_THREADS == ctx::xtileM, "one column per register");
static_assert(2 * kPipeN + 2 <= VX_CFG_NUM_BARRIERS, "not enough barriers");

// fp32 -> fp16, round to nearest even, for S in fp16's normal range. Branch-free:
// the threads of a warp run this in lockstep.
static inline uint32_t f32_to_f16(uint32_t x) {
  uint32_t r = (((x >> 23) & 0xff) - 127 + 15) << 10 | ((x & 0x7fffff) >> 13);
  r += ((x & 0x1fff) + (r & 1) + 0xfff) >> 13;
  return ((x >> 16) & 0x8000) | r;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t M = arg->M, L = arg->L, N = arg->N, K1 = arg->K1;
  auto pS = reinterpret_cast<ctx::output_t*>(arg->S_addr);
  auto pO = reinterpret_cast<ctx::output_t*>(arg->O_addr);
  (void)M;

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

  // S = Q . K^T, into TMEM.
  if (warp_rank == 0) {
    tm::tj_desc_t d = {};
    d.idesc        = tm::make_idesc(vt::ITYPE::id, vt::OTYPE::id, L / ctx::tcN, true);
    d.ctrl         = tm::make_ctrl(tm::TJ_CLOSED, kPipeN, 1, K1 / ctx::tileK);
    d.taddr        = s_col;
    d.a_base       = tm::smem_offset(smem);
    d.b_base       = tm::smem_offset(smem + cta_M * kBK);
    d.stage_stride = s1_elems * sizeof(ctx::input_t);
    d.ld           = (kBK * sizeof(ctx::input_t)) | ((kBK * sizeof(ctx::input_t)) << 16);
    d.full_bar     = full_bar;
    d.empty_bar    = empty_bar;
    d.done_bar     = done_bar;
    d.dxa_slots    = kDescQ | (kDescKT << 4);
    d.k_advance    = kBK;
    vx_barrier_expect_tx(done_bar, 1);
    tm::tj_post(d);
  }
  vx_barrier(done_bar, num_warps);

  // P = fp16(S), K-packed into TMEM; S is checked by the host. Thread t holds M row
  // lane + t; register r holds column base + r.
  for (uint32_t c = 0; c < L / (2 * kNR); ++c) {
    ctx::fragment_acc s0, s1, p;
    uint32_t col = s_col + c * 2 * kNR;
    tm::load<tm::LANE_PER_THREAD>(s0, tm::make_taddr(lane, col));
    tm::load<tm::LANE_PER_THREAD>(s1, tm::make_taddr(lane, col + kNR));
    for (uint32_t r = 0; r < kNR; ++r) {
      uint32_t lo_bits, hi_bits;
      float lo = (r < kNR / 2) ? s0.data[2 * r] : s1.data[2 * r - kNR];
      float hi = (r < kNR / 2) ? s0.data[2 * r + 1] : s1.data[2 * r + 1 - kNR];
      __builtin_memcpy(&lo_bits, &lo, 4);
      __builtin_memcpy(&hi_bits, &hi, 4);
      uint32_t row = lane + t;
      uint32_t base = c * 2 * kNR + 2 * r;
      pS[row * L + base]     = lo;
      pS[row * L + base + 1] = hi;
      uint32_t word = f32_to_f16(lo_bits) | (f32_to_f16(hi_bits) << 16);
      __builtin_memcpy(&p.data[r], &word, 4);
    }
    tm::store<tm::LANE_PER_THREAD>(tm::make_taddr(lane, p_col + c * kNR), p);
  }
  vx_barrier(step_bar, num_warps);

  // O = P . V^T: A from TMEM, only V is staged.
  if (warp_rank == 0) {
    tm::tj_desc_t d = {};
    d.idesc        = tm::make_idesc(vt::ITYPE::id, vt::OTYPE::id, N / ctx::tcN, true, false, true);
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
    d.k_advance    = kBK;
    vx_barrier_expect_tx(done_bar, 1);
    tm::tj_post(d);
  }
  vx_barrier(done_bar, num_warps);

  for (uint32_t c = 0; c < N / kNR; ++c) {
    ctx::fragment_acc o;
    tm::load<tm::LANE_PER_THREAD>(o, tm::make_taddr(lane, o_col + c * kNR));
    for (uint32_t r = 0; r < kNR; ++r) {
      pO[(lane + t) * N + c * kNR + r] = o.data[r];
    }
  }
}
