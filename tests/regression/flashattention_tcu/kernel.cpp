#include "common.h"

#include <math.h>
#include <vx_intrinsics.h>
#include <vx_spawn2.h>
#include <vx_tensor.h>

namespace vt = vortex::tensor;
using ctx = vt::wmma_context<NUM_THREADS, vt::fp16, vt::fp32>;

static_assert(NUM_THREADS == 8, "flashattention_tcu currently requires 8 threads");
static_assert(ctx::tileM == kQueryRows, "query tile must match the TCU M tile");
static_assert(ctx::tileN == 8, "key/output tile must match the TCU N tile");
static_assert(ctx::tileK == kHeadDimension, "head dimension must match the TCU K tile");

extern "C" void kernel_main(kernel_arg_t *__UNIFORM__ arg) {
  auto q = reinterpret_cast<const ctx::input_t *>(arg->q_addr);
  auto k = reinterpret_cast<const ctx::input_t *>(arg->k_addr);
  auto v = reinterpret_cast<const ctx::input_t *>(arg->v_addr);
  auto output = reinterpret_cast<ctx::output_t *>(arg->output_addr);

  auto lmem = reinterpret_cast<uint8_t *>(__local_mem());
  auto scores = reinterpret_cast<float *>(lmem);
  auto probabilities_fp16 = reinterpret_cast<uint16_t *>(
      scores + kQueryRows * kKeyGroupSize);
  auto output_accumulator = reinterpret_cast<float *>(
      probabilities_fp16 + kQueryRows * kKeyGroupSize);

  ctx::fragment_a frag_q;
  ctx::fragment_b frag_k;
  ctx::fragment_acc frag_scores;

  float running_max = -3.402823466e+38f;
  float running_sum = 0.0f;
  const uint32_t row = threadIdx.x;
  const float scale = 0.25f;

  for (uint32_t column = 0; column < kHeadDimension; ++column) {
    output_accumulator[row * kHeadDimension + column] = 0.0f;
  }
  __syncthreads();

  // Process two 16-key groups with the online FlashAttention recurrence.
  for (uint32_t group_base = 0; group_base < kSequenceLength;
       group_base += kKeyGroupSize) {
    // Compute QK^T as two regular 8x8 TCU operations.
    for (uint32_t tile_base = 0; tile_base < kKeyGroupSize;
         tile_base += ctx::tileN) {
      ctx::fill_fragment(frag_scores, 0.0f);
      ctx::load_matrix_sync(frag_q, q, kHeadDimension);
      ctx::load_matrix_sync<vt::col_major>(
          frag_k, k + (group_base + tile_base) * kHeadDimension,
          kHeadDimension);
      ctx::mma_sync(frag_scores, frag_q, frag_k, frag_scores);
      ctx::store_matrix_sync(
          scores + tile_base, frag_scores, kKeyGroupSize);
    }
    __syncthreads();

    float group_max = -3.402823466e+38f;
    for (uint32_t group_key = 0; group_key < kKeyGroupSize; ++group_key) {
      const uint32_t key = group_base + group_key;
      const bool valid = !arg->causal || key <= (kQueryStart + row);
      if (valid) {
        const float score = scores[row * kKeyGroupSize + group_key] * scale;
        scores[row * kKeyGroupSize + group_key] = score;
        group_max = score > group_max ? score : group_max;
      }
    }

    const float new_max = running_max > group_max ? running_max : group_max;
    const float previous_scale =
        running_sum == 0.0f ? 0.0f : expf(running_max - new_max);

    // Rescale the previous numerator when the running maximum changes.
    for (uint32_t column = 0; column < kHeadDimension; ++column) {
      output_accumulator[row * kHeadDimension + column] *= previous_scale;
    }

    float group_sum = 0.0f;
    for (uint32_t group_key = 0; group_key < kKeyGroupSize; ++group_key) {
      const uint32_t key = group_base + group_key;
      const bool valid = !arg->causal || key <= (kQueryStart + row);
      const float probability =
          valid
              ? expf(scores[row * kKeyGroupSize + group_key] - new_max)
              : 0.0f;
      probabilities_fp16[row * kKeyGroupSize + group_key] =
          float_to_half(probability);
      group_sum += probability;
    }

    running_sum = running_sum * previous_scale + group_sum;
    running_max = new_max;
    __syncthreads();

    // Accumulate exp(S - m) V using two regular TCU operations.
    ctx::fragment_a frag_p;
    ctx::fragment_b frag_v;
    ctx::fragment_acc frag_output;

    for (uint32_t output_base = 0; output_base < kHeadDimension;
         output_base += ctx::tileN) {
      ctx::load_matrix_sync(
          frag_output, output_accumulator + output_base, kHeadDimension);
      ctx::load_matrix_sync(
          frag_p, probabilities_fp16, kKeyGroupSize);
      ctx::load_matrix_sync(
          frag_v, v + group_base * kHeadDimension + output_base,
          kHeadDimension);
      ctx::mma_sync(frag_output, frag_p, frag_v, frag_output);
      ctx::store_matrix_sync(
          output_accumulator + output_base, frag_output, kHeadDimension);
    }
    __syncthreads();
  }

  for (uint32_t column = 0; column < kHeadDimension; ++column) {
    output[row * kHeadDimension + column] =
        output_accumulator[row * kHeadDimension + column] / running_sum;
  }
}
