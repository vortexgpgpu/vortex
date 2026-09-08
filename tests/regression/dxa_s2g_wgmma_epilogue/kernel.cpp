#include "common.h"

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_tensor.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

namespace vt = vortex::tensor;
using ctx = vt::wgmma_context<VX_CFG_NUM_THREADS, vt::fp16, vt::fp32, false, WGMMA_NRC>;

enum : uint32_t { kDescOut = 0, kDescBias = 1, kDescStats = 2 };

static constexpr uint32_t kCtaM = S2G_TE_COMPUTE_WARPS * ctx::xtileM;
static constexpr uint32_t kOutWords = kCtaM * ctx::xtileN;
static constexpr uint32_t kBiasWords = ctx::xtileN;
static constexpr uint32_t kStatsWords = S2G_TE_STATS_X * S2G_TE_STATS_Y * S2G_TE_STATS_Z;
static constexpr uint32_t kOutStageWords = kOutWords + kBiasWords;
static constexpr uint32_t kOutBase = 0;
static constexpr uint32_t kStatsBase = 2 * kOutStageWords;
static constexpr uint32_t kAOffset = kStatsBase + S2G_TE_STATS_STAGES * kStatsWords;
static constexpr uint32_t kAWords = kCtaM * ctx::tileK;
static constexpr uint32_t kBWords = ctx::tileK * ctx::xtileN;

static inline float epilogue(float value, uint32_t iter) {
  return (value < 0.0f ? 0.0f : value) * 0.5f + static_cast<float>(iter + 1);
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t tid = threadIdx.x;
  const uint32_t warp = get_sub_group_id();
  const uint32_t lane = tid % VX_CFG_NUM_THREADS;
  const uint32_t threads = blockDim.x;
  const uint32_t n = ctx::xtileN;
  const uint32_t k = S2G_TE_K_TILES * ctx::tileK;
  auto smem = reinterpret_cast<uint32_t*>(__local_mem());
  auto a_smem = reinterpret_cast<ctx::input_t*>(smem + kAOffset);
  auto b_smem = a_smem + kAWords;
  const auto a_global = reinterpret_cast<const ctx::input_t*>(arg->a_addr);
  const auto b_global = reinterpret_cast<const ctx::input_t*>(arg->b_addr);
  const bool serial = arg->mode == 0;
  vortex::barrier out_free(1, 3), out_ready(2, 3);
  vortex::barrier stats_free(3, 3), stats_ready(4, 3);

  for (uint32_t iter = 0; iter < S2G_TE_ITERS; ++iter) {
    const uint32_t out_stage = iter & 1u;
    const uint32_t stats_stage = iter % S2G_TE_STATS_STAGES;
    if (warp == 2) {
      if (serial && iter != 0) vx_dxa_wait_group_read<0>();
      if (!serial && iter >= 2) vx_dxa_wait_group_read<1>();
    }
    if (warp == 4) {
      if (serial && iter != 0) vx_dxa_wait_group_read<0>();
      if (!serial && iter >= 3) vx_dxa_wait_group_read<2>();
    }
    if (warp < 2 || warp == 2) out_free.arrive_and_wait();
    if (warp < 2 || warp == 4) stats_free.arrive_and_wait();

    ctx::fragment_acc frag_c;
    if (warp < S2G_TE_COMPUTE_WARPS) ctx::fill_fragment(frag_c, 0.0f);
    for (uint32_t kt = 0; kt < S2G_TE_K_TILES; ++kt) {
      for (uint32_t i = tid; i < kAWords; i += threads) {
        const uint32_t row = i / ctx::tileK;
        const uint32_t col = i % ctx::tileK;
        a_smem[ctx::a_blockmajor_idx(row, col)] = a_global[row * k + kt * ctx::tileK + col];
      }
      for (uint32_t i = tid; i < kBWords; i += threads) {
        const uint32_t row = i / n;
        const uint32_t col = i % n;
        b_smem[ctx::b_blockmajor_idx(row, col)] = b_global[(kt * ctx::tileK + row) * n + col];
      }
      __syncthreads();
      if (warp < S2G_TE_COMPUTE_WARPS) {
        auto a_warp = a_smem + warp * ctx::a_warp_elems;
        auto desc_b = vt::vx_make_smem_desc(b_smem, 0);
        ctx::fragment_a frag_a;
        ctx::load_matrix_sync(frag_a, a_warp, 0);
        ctx::wgmma_sync(frag_c, frag_a, desc_b, frag_c);
      }
      __syncthreads();
    }

    if (warp < S2G_TE_COMPUTE_WARPS) {
      for (uint32_t r = 0; r < ctx::fragment_acc::NR; ++r)
        frag_c.data[r] = epilogue(frag_c.data[r], iter);
      auto out = reinterpret_cast<float*>(smem + kOutBase + out_stage * kOutStageWords);
      ctx::store_matrix_sync(out + warp * ctx::xtileM * n, frag_c, n);
      if (warp == 0) {
        for (uint32_t i = lane; i < kBiasWords; i += VX_CFG_NUM_THREADS)
          out[kOutWords + i] = static_cast<float>(iter * n + i);
      }
      auto stats = smem + kStatsBase + stats_stage * kStatsWords;
      const uint32_t base = warp * VX_CFG_NUM_THREADS + lane;
      for (uint32_t i = base; i < kStatsWords;
           i += S2G_TE_COMPUTE_WARPS * VX_CFG_NUM_THREADS)
        stats[i] = 0x70000000u + iter * 0x1000u + i;
    }
    if (warp < 2 || warp == 2) out_ready.arrive_and_wait();
    if (warp < 2 || warp == 4) stats_ready.arrive_and_wait();

    if (warp == 2) {
      const auto out = reinterpret_cast<const uint32_t*>(smem + kOutBase + out_stage * kOutStageWords);
      vx_dxa_s2g_issue_grouped_2d(kDescOut, out, 0, iter * kCtaM);
      vx_dxa_s2g_issue_grouped_2d(kDescBias, out + kOutWords, 0, iter);
      vx_dxa_commit_group();
    }
    if (warp == 4) {
      const auto stats = smem + kStatsBase + stats_stage * kStatsWords;
      vx_dxa_s2g_issue_grouped_3d(kDescStats, stats, 0, 0, iter * S2G_TE_STATS_Z);
      vx_dxa_commit_group();
    }
  }
  if (warp == 2 || warp == 4) vx_dxa_wait_group_read<0>();
  __syncthreads();
}

extern "C" void __vx_kentry_kernel_main(kernel_arg_t*) __attribute__((alias("kernel_main"), used));
