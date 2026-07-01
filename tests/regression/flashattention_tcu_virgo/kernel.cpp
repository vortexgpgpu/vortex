#include "common.h"

#include <vx_intrinsics.h>
#include <vx_spawn2.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::fp32, vt::fp32>;

static_assert(ctx::tileM == kQueryRows, "query tile must match the TCU M tile");
static_assert(kKeyGroupSize == ctx::tileN, "key group must cover one N tile");
static_assert(kKeyGroupSize == ctx::tileK, "P*V tile must match the TCU K tile");
static_assert(ctx::tileK == kHeadDimension, "head dimension must match the TCU K tile");
static_assert(kSequenceLength % kKeyGroupSize == 0,
              "sequence length must be grouped exactly");
static_assert(kTotalQueryRows % kQueryRows == 0,
              "total query rows must be an exact number of query tiles");
static_assert(kKeyGroupSize % ctx::tileN == 0,
              "key group must be an exact number of N tiles");
static_assert(kHeadDimension % ctx::tileN == 0,
              "head dimension must be an exact number of N tiles");

static constexpr uint32_t kKeyGroups = kSequenceLength / kKeyGroupSize;
static constexpr uint32_t kScoreTiles = kKeyGroupSize / ctx::tileN;
static constexpr uint32_t kOutputTiles = kHeadDimension / ctx::tileN;

#ifdef FA_ENABLE_TIMING
static inline uint64_t timing_now() {
  return vx_rdcycle();
}
#endif

static inline void thread_block_init_sharedmem(float *output_accumulator,
                                               uint32_t row) 
{
  #pragma unroll
  for (uint32_t column = 0; column < kHeadDimension; ++column) 
  {
    output_accumulator[row * kHeadDimension + column] = 0.0f;
  }
}

static inline float thread_block_online_softmax(
    float *scores,
    float *probabilities,
    float *running_max,
    float *running_sum,
    uint32_t row) 
{
  float group_max = -3.402823466e+38f;
  constexpr float scale = 0.25f;
  const uint32_t row_offset = row * kKeyGroupSize;

  #pragma unroll 16
  for (uint32_t group_key = 0; group_key < kKeyGroupSize; ++group_key) 
  {
    const uint32_t offset = row_offset + group_key;
    const float score = scores[offset] * scale;
    scores[offset] = score;
    group_max = score > group_max ? score : group_max;
  }

  const float new_max = *running_max > group_max ? *running_max : group_max;
  const float previous_scale =
      *running_sum == 0.0f ? 0.0f : exp2_taylor(*running_max - new_max);

  float group_sum = 0.0f;

  #pragma unroll 8
  for (uint32_t group_key = 0; group_key < kKeyGroupSize; ++group_key) 
  {
    const uint32_t offset = row_offset + group_key;
    const float probability = exp2_taylor(scores[offset] - new_max);
    probabilities[offset] = probability;
    group_sum += probability;
  }

  *running_sum = (*running_sum * previous_scale) + group_sum;
  *running_max = new_max;
  return previous_scale;
}

static inline void thread_block_O_rescale(float *output_accumulator,
                                          float previous_scale,
                                          uint32_t row) 
{
  #pragma unroll 16
  for (uint32_t column = 0; column < kHeadDimension; ++column) {
    output_accumulator[row * kHeadDimension + column] *= previous_scale;
  }
}

extern "C" void kernel_main(kernel_arg_t *__UNIFORM__ arg) 
{
  auto q = reinterpret_cast<const ctx::input_t *>(arg->q_addr);
  auto k = reinterpret_cast<const ctx::input_t *>(arg->k_addr);
  auto v = reinterpret_cast<const ctx::input_t *>(arg->v_addr);
  auto output = reinterpret_cast<ctx::output_t *>(arg->output_addr);
#ifdef FA_ENABLE_TIMING
  auto timing = reinterpret_cast<uint64_t *>(arg->timing_addr) +
      blockIdx.x * kTimingCount;
#endif
  const uint32_t query_tile_offset =
      blockIdx.x * kQueryRows * kHeadDimension;
  q += query_tile_offset;
  output += query_tile_offset;

  auto lmem = reinterpret_cast<uint8_t *>(__local_mem());
  auto scores = reinterpret_cast<float *>(lmem);
  auto probabilities = reinterpret_cast<float *>(
      scores + kQueryRows * kKeyGroupSize);
  auto output_accumulator = reinterpret_cast<float *>(
      probabilities + kQueryRows * kKeyGroupSize);

  ctx::fragment_a frag_q;
  ctx::fragment_b frag_k;
  ctx::fragment_acc frag_scores;

  float running_max = -3.402823466e+38f;
  float running_sum = 0.0f;
  const uint32_t row = threadIdx.x;
#ifdef FA_ENABLE_TIMING
  uint64_t init_cycles = 0;
  uint64_t qk_cycles = 0;
  uint64_t softmax_cycles = 0;
  uint64_t pv_cycles = 0;
  uint64_t final_cycles = 0;
  uint64_t total_start = 0;
  uint64_t phase_start = 0;

  if (row == 0) {
    total_start = timing_now();
    phase_start = total_start;
  }
#endif

  if (row < kQueryRows) 
  {
    thread_block_init_sharedmem(output_accumulator, row);
  }
    __syncthreads();

#ifdef FA_ENABLE_TIMING
  if (row == 0) {
    init_cycles += timing_now() - phase_start;
  }
#endif

  #pragma unroll kSequenceLength/kKeyGroupSize
  for (uint32_t group_base = 0; group_base < kSequenceLength;
       group_base += kKeyGroupSize) 
  {
    // GEMM I: S = Q * K^T.
#ifdef FA_ENABLE_TIMING
    if (row == 0) {
      phase_start = timing_now();
    }
#endif
    ctx::load_matrix_sync(frag_q, q, kHeadDimension);

    #pragma unroll kScoreTiles
    for (uint32_t tile_idx = 0; tile_idx < kScoreTiles; ++tile_idx) 
    {
      constexpr uint32_t key_tile_stride = ctx::tileN;
      const uint32_t tile_base = tile_idx * key_tile_stride;
      ctx::fill_fragment(frag_scores, 0.0f);
      ctx::load_matrix_sync<vt::col_major>(
          frag_k, k + (group_base + tile_base) * kHeadDimension,
          kHeadDimension);
      ctx::mma_sync(frag_scores, frag_q, frag_k, frag_scores);
      ctx::store_matrix_sync(scores + tile_base, frag_scores, kKeyGroupSize);
    }
    __syncthreads();
#ifdef FA_ENABLE_TIMING
    if (row == 0) {
      qk_cycles += timing_now() - phase_start;
      phase_start = timing_now();
    }
#endif

    float previous_scale = 0.0f;
    if (row < kQueryRows) 
    {
      previous_scale =
          thread_block_online_softmax(scores, probabilities, &running_max,
                                      &running_sum, row);
      thread_block_O_rescale(output_accumulator, previous_scale, row);
    }
    __syncthreads();
#ifdef FA_ENABLE_TIMING
    if (row == 0) {
      softmax_cycles += timing_now() - phase_start;
      phase_start = timing_now();
    }
#endif

    // GEMM II: O = O + P * V.
    ctx::fragment_a frag_p;
    ctx::fragment_b frag_v;
    ctx::fragment_acc frag_output;

    ctx::load_matrix_sync(frag_p, probabilities, kKeyGroupSize);

    #pragma unroll kOutputTiles
    for (uint32_t output_idx = 0; output_idx < kOutputTiles; ++output_idx) 
    {
      constexpr uint32_t output_tile_stride = ctx::tileN;
      const uint32_t output_base = output_idx * output_tile_stride;
      ctx::load_matrix_sync(
          frag_output, output_accumulator + output_base, kHeadDimension);
      ctx::load_matrix_sync(
          frag_v, v + group_base * kHeadDimension + output_base,
          kHeadDimension);
      ctx::mma_sync(frag_output, frag_p, frag_v, frag_output);
      ctx::store_matrix_sync(
          output_accumulator + output_base, frag_output, kHeadDimension);
    }
    __syncthreads();
#ifdef FA_ENABLE_TIMING
    if (row == 0) {
      pv_cycles += timing_now() - phase_start;
    }
#endif
  }

#ifdef FA_ENABLE_TIMING
  if (row == 0) {
    phase_start = timing_now();
  }
#endif
  if (row < kQueryRows) 
  {
    float inv_running_sum = 1.0f / running_sum;

    #pragma unroll kHeadDimension
    for (uint32_t column = 0; column < kHeadDimension; ++column) 
    {
      output[row * kHeadDimension + column] =
          output_accumulator[row * kHeadDimension + column] * inv_running_sum;
    }
  }
#ifdef FA_ENABLE_TIMING
  __syncthreads();
  if (row == 0) {
    final_cycles += timing_now() - phase_start;
    timing[kTimingTotal] = timing_now() - total_start;
    timing[kTimingInit] = init_cycles;
    timing[kTimingQk] = qk_cycles;
    timing[kTimingSoftmax] = softmax_cycles;
    timing[kTimingPv] = pv_cycles;
    timing[kTimingFinal] = final_cycles;
  }
#endif
}
