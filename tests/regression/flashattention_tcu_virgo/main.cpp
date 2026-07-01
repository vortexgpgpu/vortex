#include "common.h"

#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>
#include <vortex.h>

#define RT_CHECK(expr)                                                   \
  do {                                                                   \
    const int status = (expr);                                           \
    if (status != 0) {                                                   \
      std::cerr << "Error: " << #expr << " returned " << status << '\n'; \
      cleanup();                                                         \
      return 1;                                                          \
    }                                                                    \
  } while (false)

static constexpr float kTolerance = 3.0e-3f;

vx_device_h device = nullptr;
vx_buffer_h q_buffer = nullptr;
vx_buffer_h k_buffer = nullptr;
vx_buffer_h v_buffer = nullptr;
vx_buffer_h output_buffer = nullptr;
#ifdef FA_ENABLE_TIMING
vx_buffer_h timing_buffer = nullptr;
#endif
vx_buffer_h kernel_buffer = nullptr;
vx_buffer_h args_buffer = nullptr;

static void cleanup() {
  if (q_buffer) vx_mem_free(q_buffer);
  if (k_buffer) vx_mem_free(k_buffer);
  if (v_buffer) vx_mem_free(v_buffer);
  if (output_buffer) vx_mem_free(output_buffer);
#ifdef FA_ENABLE_TIMING
  if (timing_buffer) vx_mem_free(timing_buffer);
#endif
  if (kernel_buffer) vx_mem_free(kernel_buffer);
  if (args_buffer) vx_mem_free(args_buffer);
  if (device) vx_dev_close(device);
}

static float input_value(uint32_t row, uint32_t column, uint32_t salt) {
  const int32_t value =
      static_cast<int32_t>((row * 7 + column * 3 + salt) % 17) - 8;
  return static_cast<float>(value) * 0.0625f;
}

int main() {
  RT_CHECK(vx_dev_open(&device));

  uint64_t isa_flags = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_EXT_TCU) == 0) {
    std::cerr << "TCU extension not supported\n";
    cleanup();
    return 1;
  }

  uint64_t device_threads = 0;
  RT_CHECK(vx_dev_caps(device, VX_CAPS_NUM_THREADS, &device_threads));
  if (device_threads != NUM_THREADS) {
    std::cerr << "Device warp size " << device_threads
              << " does not match NUM_THREADS=" << NUM_THREADS << '\n';
    cleanup();
    return 1;
  }

  const size_t q_elements = kTotalQueryRows * kHeadDimension;
  const size_t kv_elements = kSequenceLength * kHeadDimension;
  const size_t output_elements = kTotalQueryRows * kHeadDimension;

  std::vector<float> q(q_elements);
  std::vector<float> k(kv_elements);
  std::vector<float> v(kv_elements);
  std::vector<float> output(output_elements, 0.0f);
  std::vector<float> reference(output_elements, 0.0f);
#ifdef FA_ENABLE_TIMING
  std::vector<uint64_t> timing(kQueryTiles * kTimingCount, 0);
#endif

  for (uint32_t row = 0; row < kTotalQueryRows; ++row) {
    for (uint32_t column = 0; column < kHeadDimension; ++column) {
      q[row * kHeadDimension + column] =
          input_value(row + kQueryStart, column, 1);
    }
  }
  for (uint32_t row = 0; row < kSequenceLength; ++row) {
    for (uint32_t column = 0; column < kHeadDimension; ++column) {
      k[row * kHeadDimension + column] =
          input_value(row, column, 5);
      v[row * kHeadDimension + column] =
          input_value(row, column, 11);
    }
  }

  for (uint32_t row = 0; row < kTotalQueryRows; ++row) {
    float running_max = -3.402823466e+38f;
    float running_sum = 0.0f;
    for (uint32_t group_base = 0; group_base < kSequenceLength;
         group_base += kKeyGroupSize) {
      float scores[kKeyGroupSize] = {};
      float group_max = -3.402823466e+38f;
      for (uint32_t group_key = 0; group_key < kKeyGroupSize; ++group_key) {
        const uint32_t key = group_base + group_key;
        float score = 0.0f;
        for (uint32_t column = 0; column < kHeadDimension; ++column) {
          score += q[row * kHeadDimension + column]
                 * k[key * kHeadDimension + column];
        }
        score *= 0.25f;
        scores[group_key] = score;
        group_max = score > group_max ? score : group_max;
      }

      const float new_max =
          running_max > group_max ? running_max : group_max;
      const float previous_scale =
          running_sum == 0.0f ? 0.0f : exp2_taylor(running_max - new_max);
      for (uint32_t column = 0; column < kHeadDimension; ++column) {
        reference[row * kHeadDimension + column] *= previous_scale;
      }

      float group_sum = 0.0f;
      for (uint32_t group_key = 0; group_key < kKeyGroupSize; ++group_key) {
        const uint32_t key = group_base + group_key;
        const float probability =
            exp2_taylor(scores[group_key] - new_max);
        group_sum += probability;
        for (uint32_t column = 0; column < kHeadDimension; ++column) {
          reference[row * kHeadDimension + column] +=
              probability * v[key * kHeadDimension + column];
        }
      }

      running_sum = running_sum * previous_scale + group_sum;
      running_max = new_max;
    }

    for (uint32_t column = 0; column < kHeadDimension; ++column) {
      reference[row * kHeadDimension + column] /= running_sum;
    }
  }

  kernel_arg_t args = {};

  RT_CHECK(vx_mem_alloc(
      device, q.size() * sizeof(float), VX_MEM_READ, &q_buffer));
  RT_CHECK(vx_mem_address(q_buffer, &args.q_addr));
  RT_CHECK(vx_mem_alloc(
      device, k.size() * sizeof(float), VX_MEM_READ, &k_buffer));
  RT_CHECK(vx_mem_address(k_buffer, &args.k_addr));
  RT_CHECK(vx_mem_alloc(
      device, v.size() * sizeof(float), VX_MEM_READ, &v_buffer));
  RT_CHECK(vx_mem_address(v_buffer, &args.v_addr));
  RT_CHECK(vx_mem_alloc(
      device, output.size() * sizeof(float), VX_MEM_WRITE, &output_buffer));
  RT_CHECK(vx_mem_address(output_buffer, &args.output_addr));
#ifdef FA_ENABLE_TIMING
  RT_CHECK(vx_mem_alloc(
      device, timing.size() * sizeof(uint64_t), VX_MEM_WRITE, &timing_buffer));
  RT_CHECK(vx_mem_address(timing_buffer, &args.timing_addr));
#endif

  RT_CHECK(vx_copy_to_dev(
      q_buffer, q.data(), 0, q.size() * sizeof(float)));
  RT_CHECK(vx_copy_to_dev(
      k_buffer, k.data(), 0, k.size() * sizeof(float)));
  RT_CHECK(vx_copy_to_dev(
      v_buffer, v.data(), 0, v.size() * sizeof(float)));
  RT_CHECK(vx_upload_kernel_file(device, "kernel.vxbin", &kernel_buffer));
  RT_CHECK(vx_upload_bytes(device, &args, sizeof(args), &args_buffer));

  uint32_t grid_dim[1] = {kQueryTiles};
  uint32_t block_dim[1] = {NUM_THREADS};
  RT_CHECK(vx_start_g(device, kernel_buffer, args_buffer, 1,
                      grid_dim, block_dim, kLocalMemoryBytes));
  RT_CHECK(vx_ready_wait(device, VX_MAX_TIMEOUT));
  RT_CHECK(vx_copy_from_dev(
      output.data(), output_buffer, 0, output.size() * sizeof(float)));
#ifdef FA_ENABLE_TIMING
  RT_CHECK(vx_copy_from_dev(
      timing.data(), timing_buffer, 0, timing.size() * sizeof(uint64_t)));
#endif

  uint32_t errors = 0;
  float max_error = 0.0f;
  for (size_t index = 0; index < output.size(); ++index) {
    const float error = std::fabs(output[index] - reference[index]);
    max_error = error > max_error ? error : max_error;
    if (error > kTolerance) {
      if (errors < 16) {
        std::cerr << "Mismatch at " << index << ": expected "
                  << reference[index] << ", got " << output[index]
                  << ", error " << error << '\n';
      }
      ++errors;
    }
  }

  if (errors != 0) {
    std::cerr << "FAILED: " << errors << " mismatches, max error "
              << max_error << '\n';
    cleanup();
    return 1;
  }

#ifdef FA_ENABLE_TIMING
  uint64_t timing_totals[kTimingCount] = {};
  for (uint32_t tile = 0; tile < kQueryTiles; ++tile) {
    for (uint32_t counter = 0; counter < kTimingCount; ++counter) {
      timing_totals[counter] += timing[tile * kTimingCount + counter];
    }
  }
  const char *timing_names[kTimingCount] = {
      "total", "init", "qk", "softmax_rescale", "pv", "final_store"};
  const uint64_t measured_work =
      timing_totals[kTimingInit] + timing_totals[kTimingQk] +
      timing_totals[kTimingSoftmax] + timing_totals[kTimingPv] +
      timing_totals[kTimingFinal];

  std::cout << "Timing cycles across " << kQueryTiles << " query tiles:\n";
  for (uint32_t counter = 0; counter < kTimingCount; ++counter) {
    const uint64_t denominator =
        counter == kTimingTotal ? timing_totals[kTimingTotal] : measured_work;
    const double percent = denominator == 0
        ? 0.0
        : (100.0 * static_cast<double>(timing_totals[counter]) /
           static_cast<double>(denominator));
    std::cout << "  " << std::setw(15) << timing_names[counter] << ": "
              << std::setw(10) << timing_totals[counter]
              << " cycles (" << std::fixed << std::setprecision(1)
              << percent << "%)\n" << std::defaultfloat;
  }
#endif

  cleanup();

  std::cout << "PASSED: Virgo-style FP32 non-causal FlashAttention TCU forward"
            << ", Q=" << kTotalQueryRows
            << ", KV=" << kSequenceLength
            << ", D=" << kHeadDimension
            << ", max error=" << max_error << '\n';
  return 0;
}
