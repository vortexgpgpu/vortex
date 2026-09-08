#include "common.h"

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::fp16, vt::fp32, false, WGMMA_NRC>;

enum : uint32_t { kDescQ = 0, kDescK = 1, kDescV = 2, kDescOut = 3, kDescStats = 4 };

static constexpr uint32_t kM = ATT_COMPUTE_WARPS * ctx::xtileM;
static constexpr uint32_t kN = ctx::xtileN;
static constexpr uint32_t kK = ATT_K_TILES * ctx::tileK;
static constexpr uint32_t kQKStageWords = kM * ctx::tileK + ctx::tileK * kN;
static constexpr uint32_t kD = kN;
static constexpr uint32_t kVWords = kN * kD;
static constexpr uint32_t kScoreWords = kM * kN;
static constexpr uint32_t kOutputWords = kM * kD;
static constexpr uint32_t kOutputStageWords = kOutputWords + kM;
static constexpr uint32_t kInputBytes = 2 * kQKStageWords * sizeof(ctx::input_t);
static constexpr uint32_t kVOffsetBytes = kInputBytes;
static constexpr uint32_t kScoreOffsetBytes = kVOffsetBytes + kVWords * sizeof(float);
static constexpr uint32_t kOutputOffsetBytes = kScoreOffsetBytes + kScoreWords * sizeof(float);

// A compact FA-like epilogue: QK scores get a positive linear-softmax proxy,
// then weight a separately loaded V tile. The row stream records the weight
// denominator. No exp approximation is hidden in this readability-oriented
// regression; the data path and source-lifetime behavior are the point.
static inline float attention_weight(float score) { return score * 0.0625f + 1.0f; }

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t tid = threadIdx.x;
  const uint32_t warp = get_sub_group_id();
  const uint32_t lane = tid % VX_CFG_NUM_THREADS;
  const bool use_s2g = arg->mode == 0;
  const bool use_lsu_load = arg->mode == 2;
  auto smem = reinterpret_cast<uint8_t*>(__local_mem());
  auto qk_base = reinterpret_cast<ctx::input_t*>(smem);
  ctx::input_t* q_stage[2] = {qk_base, qk_base + kQKStageWords};
  ctx::input_t* k_stage[2] = {qk_base + kM * ctx::tileK, qk_base + kQKStageWords + kM * ctx::tileK};
  // Keep V in fp32: the current low-level kernel ABI exposes fp16 as its raw
  // uint16_t storage type (there is no implicit half->float conversion).
  // Q/K still exercise fp16 WGMMA; V is an independently transferred,
  // accumulation-domain value tile.
  auto v_stage = reinterpret_cast<float*>(smem + kVOffsetBytes);
  auto score_base = reinterpret_cast<float*>(smem + kScoreOffsetBytes);
  auto out_base = reinterpret_cast<float*>(smem + kOutputOffsetBytes);
  float* out_stage[ATT_OUT_STAGES];
  for (uint32_t s = 0; s < ATT_OUT_STAGES; ++s) out_stage[s] = out_base + s * kOutputStageWords;

  // CTA barrier 0 is reserved by __syncthreads() (its id is the CTA slot).
  // Keep the asynchronous Q/K/V transaction barrier in a disjoint slot;
  // otherwise expect_tx() contributes events to the CTA rendezvous itself.
  constexpr uint32_t role_participants = ATT_COMPUTE_WARPS + 1;
  vortex::barrier qk_bar0(1, role_participants), qk_bar1(2, role_participants);
  vortex::barrier* qk_bar[2] = {&qk_bar0, &qk_bar1};
  vortex::barrier out_free(3, role_participants), out_ready(4, role_participants);

  auto lsu_load_input = [&](uint32_t stage, uint32_t kt, uint32_t row, bool with_v) {
    if (warp != ATT_LOAD_WARP) return;
    const auto* q_global = reinterpret_cast<const ctx::input_t*>(arg->q_addr);
    const auto* k_global = reinterpret_cast<const ctx::input_t*>(arg->k_addr);
    for (uint32_t i = lane; i < kM * ctx::tileK; i += VX_CFG_NUM_THREADS) {
      const uint32_t r = i / ctx::tileK;
      const uint32_t c = i % ctx::tileK;
      q_stage[stage][i] = q_global[(row + r) * kK + kt + c];
    }
    for (uint32_t i = lane; i < ctx::tileK * kN; i += VX_CFG_NUM_THREADS) {
      const uint32_t r = i / kN;
      const uint32_t c = i % kN;
      k_stage[stage][ctx::b_blockmajor_idx(r, c)] = k_global[(kt + r) * kN + c];
    }
    if (with_v) {
      const auto* v_global = reinterpret_cast<const float*>(arg->v_addr);
      for (uint32_t i = lane; i < kVWords; i += VX_CFG_NUM_THREADS)
        v_stage[i] = v_global[i];
    }
  };

  auto issue_input = [&](uint32_t stage, uint32_t kt, uint32_t row, bool with_v) {
    if (warp != ATT_LOAD_WARP) return;
    if (use_lsu_load) {
      lsu_load_input(stage, kt, row, with_v);
      return;
    }
    qk_bar[stage]->expect_tx(with_v ? 3 : 2);
    vx_dxa_issue_2d_wg(kDescQ, qk_bar[stage]->id(), q_stage[stage], kt, row);
    vx_dxa_issue_2d_wg(kDescK, qk_bar[stage]->id(), k_stage[stage], 0, kt);
    if (with_v) vx_dxa_issue_2d_wg(kDescV, qk_bar[stage]->id(), v_stage, 0, row);
  };

  ctx::fragment_acc score;
  for (uint32_t iter = 0; iter < ATT_ITERS; ++iter) {
    const uint32_t row = iter * kM;
    issue_input(0, 0, row, iter == 0);
    // CTA rendezvous drains the loader's issue path before the transaction
    // barrier is sampled by the compute warps.
    __syncthreads();

    if (warp < ATT_COMPUTE_WARPS) ctx::fill_fragment(score, 0.0f);
    uint32_t stage = 0;
    for (uint32_t kt = 0; kt < ATT_K_TILES; ++kt) {
      const uint32_t next = stage ^ 1u;
      if (kt + 1 < ATT_K_TILES) issue_input(next, (kt + 1) * ctx::tileK, row, false);
      __syncthreads();
      if (use_lsu_load) {
        __syncthreads();
      } else if (warp < ATT_COMPUTE_WARPS || warp == ATT_LOAD_WARP) {
        qk_bar[stage]->arrive_and_wait();
      }
      if (warp < ATT_COMPUTE_WARPS) {
        auto q = q_stage[stage] + warp * ctx::xtileM * ctx::tileK;
        auto desc_k = vt::vx_make_smem_desc(k_stage[stage], 0);
#if defined(WGMMA_RS)
        ctx::fragment_a frag_q;
        ctx::load_matrix_sync(frag_q, q, ctx::tileK);
        ctx::wgmma_sync(score, frag_q, desc_k, score);
#else
        auto desc_q = vt::vx_make_smem_desc(q, ctx::tileK * sizeof(ctx::input_t));
        ctx::wgmma_sync(score, desc_q, desc_k, score);
#endif
      }
      __syncthreads();
      if (use_lsu_load) {
        __syncthreads();
      } else if (warp < ATT_COMPUTE_WARPS || warp == ATT_LOAD_WARP) {
        qk_bar[stage]->arrive_and_wait();
      }
      stage = next;
    }

    const uint32_t output_stage = iter % ATT_OUT_STAGES;
    if (use_s2g && warp == ATT_STORE_WARP && iter >= ATT_OUT_STAGES) vx_dxa_wait_group_read<1>();
    if (use_s2g && (warp < ATT_COMPUTE_WARPS || warp == ATT_STORE_WARP)) out_free.arrive_and_wait();

    if (warp < ATT_COMPUTE_WARPS) ctx::store_matrix_sync(score_base + warp * ctx::xtileM * kN, score, kN);
    __syncthreads();
    if (warp < ATT_COMPUTE_WARPS) {
      const bool stage_global = use_s2g || (ATT_STAGE_GLOBAL_STORE != 0);
      float* out = stage_global ? out_stage[output_stage] : reinterpret_cast<float*>(arg->out_addr) + iter * kOutputWords;
      float* stats = stage_global ? out_stage[output_stage] + kOutputWords : reinterpret_cast<float*>(arg->stats_addr) + row;
      float* score_tile = score_base + warp * ctx::xtileM * kN;
      float* out_tile = out + warp * ctx::xtileM * kD;
      // First write one statistic per row with a uniform lane-owned loop.
      // This avoids a divergent predicate on a global FP store.
      for (uint32_t r = lane; r < ctx::xtileM; r += VX_CFG_NUM_THREADS) {
        float denom = 0.0f;
        for (uint32_t n = 0; n < kN; ++n) denom += attention_weight(score_tile[r * kN + n]);
        stats[warp * ctx::xtileM + r] = denom;
      }
      // All lanes then cooperate on each row's output feature dimension.
      for (uint32_t r = 0; r < ctx::xtileM; ++r) {
        float denom = 0.0f;
        for (uint32_t n = 0; n < kN; ++n) denom += attention_weight(score_tile[r * kN + n]);
        const float inv_denom = 1.0f / denom;
        for (uint32_t d = lane; d < kD; d += VX_CFG_NUM_THREADS) {
          float value = 0.0f;
          for (uint32_t n = 0; n < kN; ++n)
            value += attention_weight(score_tile[r * kN + n]) * v_stage[n * kD + d];
          out_tile[r * kD + d] = value * inv_denom;
        }
      }
    }

    if (!use_s2g && (ATT_STAGE_GLOBAL_STORE != 0)) {
      __syncthreads();
      if (warp == ATT_STORE_WARP) {
        auto* out_global = reinterpret_cast<float*>(arg->out_addr) + iter * kOutputWords;
        auto* stats_global = reinterpret_cast<float*>(arg->stats_addr) + row;
        auto* staged = out_stage[output_stage];
        for (uint32_t i = lane; i < kOutputWords; i += VX_CFG_NUM_THREADS)
          out_global[i] = staged[i];
        for (uint32_t i = lane; i < kM; i += VX_CFG_NUM_THREADS)
          stats_global[i] = staged[kOutputWords + i];
      }
    }

    if (use_s2g) {
      if (warp < ATT_COMPUTE_WARPS || warp == ATT_STORE_WARP) out_ready.arrive_and_wait();
      if (warp == ATT_STORE_WARP) {
        vx_dxa_s2g_issue_grouped_2d(kDescOut, out_stage[output_stage], 0, row);
        vx_dxa_s2g_issue_grouped_2d(kDescStats, out_stage[output_stage] + kOutputWords, 0, row);
        vx_dxa_commit_group();
      }
    }
    __syncthreads();
  }

  if (use_s2g && warp == ATT_STORE_WARP) vx_dxa_wait_group_read<0>();
  __syncthreads();
}

extern "C" void __vx_kentry_kernel_main(kernel_arg_t*) __attribute__((alias("kernel_main"), used));
