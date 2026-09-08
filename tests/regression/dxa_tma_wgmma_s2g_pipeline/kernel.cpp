#include "common.h"
#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_tensor.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::fp16, vt::fp32, false, WGMMA_NRC>;
enum : uint32_t { kDescA, kDescB, kDescOut, kDescBias, kDescStats, kDescState, kDescMeta };
constexpr uint32_t kM = TMA_PIPE_COMPUTE_WARPS * ctx::xtileM, kN = ctx::xtileN;
constexpr uint32_t kK = TMA_PIPE_K_TILES * ctx::tileK;
constexpr uint32_t kAWords = kM * ctx::tileK, kBWords = ctx::tileK * kN;
constexpr uint32_t kInputWords = kAWords + kBWords, kOutputWords = kM * kN;
constexpr uint32_t kOutStride = kOutputWords + kN + TMA_PIPE_CANARY_WORDS;
constexpr uint32_t kStateStride = 3 * kM + TMA_PIPE_META_WORDS + TMA_PIPE_CANARY_WORDS;
constexpr uint32_t kInputBytes = TMA_PIPE_IN_STAGES * kInputWords * sizeof(ctx::input_t);
constexpr uint32_t kOutBytes = TMA_PIPE_OUT_STAGES * kOutStride * sizeof(float);
constexpr uint32_t kInReady = 1, kInFree = kInReady + TMA_PIPE_IN_STAGES;
constexpr uint32_t kOutReady = kInFree + TMA_PIPE_IN_STAGES, kOutFree = kOutReady + TMA_PIPE_OUT_STAGES;
constexpr uint32_t kStateReady = kOutFree + TMA_PIPE_OUT_STAGES, kStateFree = kStateReady + TMA_PIPE_STATE_STAGES;
constexpr uint32_t kOutFormatted = kStateFree + TMA_PIPE_STATE_STAGES, kStateFormatted = kOutFormatted + TMA_PIPE_OUT_STAGES;
static_assert(kStateFormatted + TMA_PIPE_STATE_STAGES <= VX_CFG_NUM_BARRIERS, "role barriers exceed hardware slots");

static inline vortex::barrier role_barrier(uint32_t base, uint32_t stage, uint32_t participants = 3) {
  return vortex::barrier(base + stage, participants);
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t warp = get_sub_group_id(), lane = threadIdx.x % VX_CFG_NUM_THREADS;
  const bool s2g = arg->mode == TMA_PIPE_TMA_S2G || arg->mode == TMA_PIPE_SERIAL_S2G;
  const bool serial = arg->mode == TMA_PIPE_SERIAL_S2G;
  auto memory = reinterpret_cast<uint8_t*>(__local_mem());
  auto input = reinterpret_cast<ctx::input_t*>(memory);
  auto output = reinterpret_cast<float*>(memory + kInputBytes);
  auto state = reinterpret_cast<float*>(memory + kInputBytes + kOutBytes);

  if (warp == 0) {
    for (uint32_t s = 0; s < TMA_PIPE_OUT_STAGES; ++s) {
      auto guard = reinterpret_cast<uint32_t*>(output + s * kOutStride + kOutputWords + kN);
      for (uint32_t i = lane; i < TMA_PIPE_CANARY_WORDS; i += VX_CFG_NUM_THREADS) { guard[i] = TMA_PIPE_CANARY; }
    }
    for (uint32_t s = 0; s < TMA_PIPE_STATE_STAGES; ++s) {
      auto guard = reinterpret_cast<uint32_t*>(state + s * kStateStride + 3 * kM + TMA_PIPE_META_WORDS);
      for (uint32_t i = lane; i < TMA_PIPE_CANARY_WORDS; i += VX_CFG_NUM_THREADS) { guard[i] = TMA_PIPE_CANARY; }
    }
  }
  __syncthreads();

  if (warp == TMA_PIPE_LOAD_WARP) {
    for (uint32_t t = 0; t < TMA_PIPE_ITERS * TMA_PIPE_K_TILES; ++t) {
      const uint32_t s = t % TMA_PIPE_IN_STAGES, kt = t % TMA_PIPE_K_TILES;
      auto a = input + s * kInputWords;
      auto b = a + kAWords;
      if (t >= TMA_PIPE_IN_STAGES) { role_barrier(kInFree, s).arrive_and_wait(); }
      if (arg->mode == TMA_PIPE_LSU_GLOBAL) {
        const auto ga = reinterpret_cast<const ctx::input_t*>(arg->a_addr);
        const auto gb = reinterpret_cast<const ctx::input_t*>(arg->b_addr);
        for (uint32_t i = lane; i < kAWords; i += VX_CFG_NUM_THREADS) { a[i] = ga[(i / ctx::tileK) * kK + kt * ctx::tileK + i % ctx::tileK]; }
        for (uint32_t i = lane; i < kBWords; i += VX_CFG_NUM_THREADS) { b[ctx::b_blockmajor_idx(i / kN, i % kN)] = gb[(kt * ctx::tileK + i / kN) * kN + i % kN]; }
        vx_fence();
      } else {
        auto ready = role_barrier(kInReady, s);
        ready.expect_tx(2);
        vx_dxa_issue_2d_wg(kDescA, ready.id(), a, kt * ctx::tileK, 0);
        vx_dxa_issue_2d_wg(kDescB, ready.id(), b, 0, kt * ctx::tileK);
      }
      role_barrier(kInReady, s).arrive();
    }
  } else if (warp < TMA_PIPE_COMPUTE_WARPS) {
    for (uint32_t iter = 0; iter < TMA_PIPE_ITERS; ++iter) {
      ctx::fragment_acc accum;
      ctx::fill_fragment(accum, 0.0f);
      for (uint32_t kt = 0; kt < TMA_PIPE_K_TILES; ++kt) {
        const uint32_t t = iter * TMA_PIPE_K_TILES + kt, s = t % TMA_PIPE_IN_STAGES;
        role_barrier(kInReady, s).arrive_and_wait();
        auto a = input + s * kInputWords + warp * ctx::xtileM * ctx::tileK;
        auto b = vt::vx_make_smem_desc(input + s * kInputWords + kAWords, 0);
        ctx::fragment_a frag_a;
        ctx::load_matrix_sync(frag_a, a, ctx::tileK);
        ctx::wgmma_sync(accum, frag_a, b, accum);
        vx_wsync(); // TCU must finish reading this input stage before reuse.
        if (t + TMA_PIPE_IN_STAGES < TMA_PIPE_ITERS * TMA_PIPE_K_TILES) { role_barrier(kInFree, s).arrive(); }
      }
      const uint32_t os = iter % TMA_PIPE_OUT_STAGES, ss = iter % TMA_PIPE_STATE_STAGES;
      if (iter >= TMA_PIPE_OUT_STAGES) { role_barrier(kOutFree, os, 4).arrive_and_wait(); }
      if (iter >= TMA_PIPE_STATE_STAGES) { role_barrier(kStateFree, ss, 4).arrive_and_wait(); }
      auto out = output + os * kOutStride;
      auto st = state + ss * kStateStride;
      for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r) { accum.data[r] = accum.data[r] * 0.5f + float(iter + 1); }
      ctx::store_matrix_sync(out + warp * ctx::xtileM * kN, accum, kN);
      vx_fence();
      for (uint32_t r = lane; r < ctx::xtileM; r += VX_CFG_NUM_THREADS) {
        const uint32_t row = warp * ctx::xtileM + r;
        float sum = 0.0f, maximum = 0.0f;
        for (uint32_t c = 0; c < kN; ++c) { const float x = out[row * kN + c]; sum += x; maximum = __builtin_fmaxf(maximum, x); }
        st[row] = sum;
        st[kM + row] = maximum;
        st[2 * kM + row] = sum / kN;
      }
      vx_fence(); // Drain producer LSU writes before publishing to a different warp/DXA.
      role_barrier(kOutReady, os, 4).arrive();
      role_barrier(kStateReady, ss, 4).arrive();
    }
  } else if (warp == 5 || warp == 6) {
    const bool output_role = warp == 5;
    const uint32_t stages = output_role ? TMA_PIPE_OUT_STAGES : TMA_PIPE_STATE_STAGES;
    const uint32_t ready_base = output_role ? kOutReady : kStateReady;
    const uint32_t free_base = output_role ? kOutFree : kStateFree;
    const uint32_t formatted_base = output_role ? kOutFormatted : kStateFormatted;
    for (uint32_t iter = 0; iter < TMA_PIPE_ITERS; ++iter) {
      const uint32_t s = iter % stages;
      if (iter >= stages) { role_barrier(free_base, s, 4).arrive_and_wait(); }
      role_barrier(ready_base, s, 4).arrive_and_wait();
      if (output_role) {
        auto bias = output + s * kOutStride + kOutputWords;
        for (uint32_t i = lane; i < kN; i += VX_CFG_NUM_THREADS) { bias[i] = float(iter + i) + 0.25f; }
      } else {
        auto meta = state + s * kStateStride + 3 * kM;
        for (uint32_t i = lane; i < TMA_PIPE_META_WORDS; i += VX_CFG_NUM_THREADS) { meta[i] = float(iter * TMA_PIPE_META_WORDS + i); }
      }
      vx_fence();
      role_barrier(formatted_base, s, 2).arrive();
    }
  } else if (warp == TMA_PIPE_STORE_WARP || warp == TMA_PIPE_STATE_WARP) {
    const bool output_role = warp == TMA_PIPE_STORE_WARP;
    const uint32_t stages = output_role ? TMA_PIPE_OUT_STAGES : TMA_PIPE_STATE_STAGES;
    const uint32_t ready_base = output_role ? kOutReady : kStateReady;
    const uint32_t free_base = output_role ? kOutFree : kStateFree;
    const uint32_t formatted_base = output_role ? kOutFormatted : kStateFormatted;
    for (uint32_t iter = 0; iter < TMA_PIPE_ITERS; ++iter) {
      const uint32_t s = iter % stages;
      role_barrier(ready_base, s, 4).arrive_and_wait();
      role_barrier(formatted_base, s, 2).arrive_and_wait();
      auto out = output + s * kOutStride;
      auto st = state + s * kStateStride;
      if (s2g) {
        if (output_role) {
          vx_dxa_issue_s2g_2d_wg(kDescOut, out, 0, iter * kM);
          vx_dxa_issue_s2g_2d_wg(kDescBias, out + kOutputWords, 0, iter);
        } else {
          vx_dxa_issue_s2g_3d_wg(kDescStats, st, 0, 0, 2 * iter);
          if ((iter & 1u) == 0) { vx_dxa_issue_s2g_2d_wg(kDescState, st + 2 * kM, 0, iter); }
          if ((iter & 3u) == 0) { vx_dxa_issue_s2g_2d_wg(kDescMeta, st + 3 * kM, 0, iter); }
        }
        vx_dxa_commit_group();
        if (serial) { vx_dxa_wait_group_read<0>(); }
      } else {
        if (output_role) {
          auto dst = reinterpret_cast<float*>(arg->out_addr) + iter * kOutputWords;
          auto bias = reinterpret_cast<float*>(arg->bias_addr) + iter * kN;
          for (uint32_t i = lane; i < kOutputWords; i += VX_CFG_NUM_THREADS) { dst[i] = out[i]; }
          for (uint32_t i = lane; i < kN; i += VX_CFG_NUM_THREADS) { bias[i] = out[kOutputWords + i]; }
        } else {
          auto stats = reinterpret_cast<float*>(arg->stats_addr) + iter * 2 * kM;
          auto mean = reinterpret_cast<float*>(arg->state_addr) + iter * kM;
          auto meta = reinterpret_cast<float*>(arg->meta_addr) + iter * TMA_PIPE_META_WORDS;
          for (uint32_t i = lane; i < 2 * kM; i += VX_CFG_NUM_THREADS) { stats[i] = st[i]; }
          if ((iter & 1u) == 0) { for (uint32_t i = lane; i < kM; i += VX_CFG_NUM_THREADS) { mean[i] = st[2 * kM + i]; } }
          if ((iter & 3u) == 0) { for (uint32_t i = lane; i < TMA_PIPE_META_WORDS; i += VX_CFG_NUM_THREADS) { meta[i] = st[3 * kM + i]; } }
        }
        vx_fence();
      }
      if (iter + 1 >= stages && iter + 1 < TMA_PIPE_ITERS) {
        if (s2g && !serial) {
          if (output_role) { vx_dxa_wait_group_read<TMA_PIPE_OUT_STAGES - 1>(); }
          else { vx_dxa_wait_group_read<TMA_PIPE_STATE_STAGES - 1>(); }
        }
        role_barrier(free_base, (iter + 1) % stages, 4).arrive();
      }
    }
    if (s2g) { vx_dxa_wait_group_read<0>(); }
  }
  __syncthreads();
  if (warp == 0) {
    auto dst = reinterpret_cast<uint32_t*>(arg->canary_addr);
    for (uint32_t s = 0; s < TMA_PIPE_OUT_STAGES + TMA_PIPE_STATE_STAGES; ++s) {
      auto guard = s < TMA_PIPE_OUT_STAGES ? output + s * kOutStride + kOutputWords + kN
                                         : state + (s - TMA_PIPE_OUT_STAGES) * kStateStride + 3 * kM + TMA_PIPE_META_WORDS;
      for (uint32_t i = lane; i < TMA_PIPE_CANARY_WORDS; i += VX_CFG_NUM_THREADS) { dst[s * TMA_PIPE_CANARY_WORDS + i] = reinterpret_cast<uint32_t*>(guard)[i]; }
    }
  }
}

extern "C" void __vx_kentry_kernel_main(kernel_arg_t*) __attribute__((alias("kernel_main"), used));
