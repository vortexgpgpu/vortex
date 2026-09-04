#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_dxa.h>
#include <vx_barrier.h>

#include "common.h"

// Descriptor slots are programmed by the host.  Each descriptor points at a
// separate destination stream; coord0 selects this iteration's disjoint
// segment, so the same descriptor can be reused by every group.
enum : uint32_t {
  kDescOutput = 0,
  kDescScale  = 1,
  kDescState0 = 2,
  kDescState1 = 3,
  kDescMeta   = 4,
};

static constexpr uint32_t kOutStageWords =
    S2G_WS_OUT_WORDS + S2G_WS_SCALE_WORDS;
static constexpr uint32_t kStateStageWords =
    S2G_WS_STATE0_WORDS + S2G_WS_STATE1_WORDS + S2G_WS_META_WORDS;
static constexpr uint32_t kOutBase = 0;
static constexpr uint32_t kStateBase =
    2 * kOutStageWords;

static inline uint32_t out_value(uint32_t iter, uint32_t word) {
  return 0x10000000u + iter * 0x10000u + word;
}

static inline uint32_t scale_value(uint32_t iter, uint32_t word) {
  return 0x20000000u + iter * 0x10000u + word;
}

static inline uint32_t state0_value(uint32_t iter, uint32_t word) {
  return 0x30000000u + iter * 0x10000u + word;
}

static inline uint32_t state1_value(uint32_t iter, uint32_t word) {
  return 0x40000000u + iter * 0x10000u + word;
}

static inline uint32_t meta_value(uint32_t iter, uint32_t word) {
  return 0x50000000u + iter * 0x10000u + word;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t warp = get_sub_group_id();
  const uint32_t tid = threadIdx.x;
  // threadIdx.x is the flat CTA thread id (not a lane id).  The two
  // producer warps therefore use a compact warp-local lane and stride by the
  // producer-WG width; otherwise each producer would touch only its first
  // eight words on the four-thread/warp SimX configuration.
  const uint32_t threads_per_warp = blockDim.x / S2G_WS_WARPS;
  const uint32_t lane = tid % threads_per_warp;
  const uint32_t producer_index = warp * threads_per_warp + lane;
  const uint32_t producer_stride = 2 * threads_per_warp;
  const uint32_t iterations = arg->iterations < S2G_WS_ITERS
                            ? arg->iterations : S2G_WS_ITERS;
  volatile uint32_t* smem =
      reinterpret_cast<volatile uint32_t*>(__local_mem());

  // Raw role hand-off barriers.  Each barrier has exactly three participants:
  // the two format/compute warps and the corresponding S2G issuer.  The
  // free barrier is entered only after the issuer's source-consumed wait;
  // consequently a producer can never overwrite a stage whose DXA engine is
  // still allowed to read.  The ready barrier orders the producers' SMEM
  // writes before the issuer's grouped issue.  These are deliberately raw
  // barriers rather than a high-level pipeline wrapper.
  vortex::barrier out_free_bar(1, 3);
  vortex::barrier out_ready_bar(2, 3);
  vortex::barrier state_free_bar(3, 3);
  vortex::barrier state_ready_bar(4, 3);

  // Roles 0/1 are the format/compute producer WG. They fill both rings;
  // warps 2 and 4 are independent S2G issuers. Warps 3/5/6/7 remain active
  // CTA participants, making the hand-off a real warp-specialized pattern.
  for (uint32_t iter = 0; iter < iterations; ++iter) {
    const uint32_t out_stage = iter & 1u;
    const uint32_t state_stage = iter % 3u;
    const bool serial = (arg->mode == 0);

    // Wait before a producer reuses a stage.  In pipelined mode the newest
    // N groups remain in flight (output N=1, state N=2), preserving overlap.
    // The issuer performs this wait before entering its stream's free
    // barrier, so the producer-side stage reuse is ordered by the same
    // source-consumed event that the tracker observes.
    if (warp == 2) {
      if (serial && iter != 0)
        vx_dxa_wait_group_read<0>();
      if (!serial && iter >= 2)
        vx_dxa_wait_group_read<1>();
    }
    if (warp == 4) {
      if (serial && iter != 0)
        vx_dxa_wait_group_read<0>();
      if (!serial && iter >= 3)
        vx_dxa_wait_group_read<2>();
    }

    if (warp < 2 || warp == 2)
      out_free_bar.arrive_and_wait();
    if (warp < 2 || warp == 4)
      state_free_bar.arrive_and_wait();

    // Only producer warps write source stages.  The role-ready barriers below
    // order these SMEM stores before the independent issuer warps enter DXA.
    if (warp < 2) {
      volatile uint32_t* out =
          smem + kOutBase + out_stage * kOutStageWords;
      volatile uint32_t* state =
          smem + kStateBase + state_stage * kStateStageWords;
      for (uint32_t i = producer_index; i < S2G_WS_OUT_WORDS;
           i += producer_stride)
        out[i] = out_value(iter, i);
      for (uint32_t i = producer_index; i < S2G_WS_SCALE_WORDS;
           i += producer_stride)
        out[S2G_WS_OUT_WORDS + i] = scale_value(iter, i);
      for (uint32_t i = producer_index; i < S2G_WS_STATE0_WORDS;
           i += producer_stride)
        state[i] = state0_value(iter, i);
      for (uint32_t i = producer_index; i < S2G_WS_STATE1_WORDS;
           i += producer_stride)
        state[S2G_WS_STATE0_WORDS + i] = state1_value(iter, i);
      for (uint32_t i = producer_index; i < S2G_WS_META_WORDS;
           i += producer_stride)
        state[S2G_WS_STATE0_WORDS + S2G_WS_STATE1_WORDS + i]
            = meta_value(iter, i);
    }
    if (warp < 2 || warp == 2)
      out_ready_bar.arrive_and_wait();
    if (warp < 2 || warp == 4)
      state_ready_bar.arrive_and_wait();

    if (warp == 2) {
      volatile uint32_t* out =
          smem + kOutBase + out_stage * kOutStageWords;
      // Two heterogeneous operations belong to one group: a large tile and
      // a compact scale vector have different source pointers/descriptors.
      vx_dxa_s2g_issue_grouped_1d(
          kDescOutput, const_cast<const uint32_t*>(out),
          iter * S2G_WS_OUT_STRIDE);
      vx_dxa_s2g_issue_grouped_1d(kDescScale,
          const_cast<const uint32_t*>(out + S2G_WS_OUT_WORDS),
          iter * S2G_WS_SCALE_STRIDE);
      vx_dxa_commit_group();
    }

    if (warp == 4) {
      volatile uint32_t* state =
          smem + kStateBase + state_stage * kStateStageWords;
      // State groups intentionally vary between one, two, and three ops.
      vx_dxa_s2g_issue_grouped_1d(
          kDescState0, const_cast<const uint32_t*>(state),
          iter * S2G_WS_STATE0_STRIDE);
      if ((iter & 1u) == 0)
        vx_dxa_s2g_issue_grouped_1d(kDescState1,
            const_cast<const uint32_t*>(state + S2G_WS_STATE0_WORDS),
            iter * S2G_WS_STATE1_STRIDE);
      if ((iter & 3u) == 0)
        vx_dxa_s2g_issue_grouped_1d(kDescMeta,
            const_cast<const uint32_t*>(
                state + S2G_WS_STATE0_WORDS + S2G_WS_STATE1_WORDS),
            iter * S2G_WS_META_STRIDE);
      vx_dxa_commit_group();
    }
  }

  // Source-consumed wait is sufficient for SMEM lifetime, but deliberately
  // does not claim destination/cache visibility. The runtime's final DXA
  // drain provides the host-observation ordering for this regression.
  if (warp == 2)
    vx_dxa_wait_group_read<0>();
  if (warp == 4)
    vx_dxa_wait_group_read<0>();
  __syncthreads();
}

extern "C" void __vx_kentry_kernel_main(kernel_arg_t*)
  __attribute__((alias("kernel_main"), used));
