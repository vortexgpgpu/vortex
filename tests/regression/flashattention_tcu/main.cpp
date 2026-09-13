#include "common.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>
#include <vortex2.h>

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
vx_queue_h  queue   = nullptr;
vx_module_h module_ = nullptr;
vx_kernel_h kernel  = nullptr;

static void cleanup() {
  if (q_buffer) vx_buffer_release(q_buffer);
  if (k_buffer) vx_buffer_release(k_buffer);
  if (v_buffer) vx_buffer_release(v_buffer);
  if (output_buffer) vx_buffer_release(output_buffer);
  if (kernel)  vx_kernel_release(kernel);
  if (module_) vx_module_release(module_);
  if (queue)   vx_queue_release(queue);
  if (device) vx_device_release(device);
}

static float input_value(uint32_t row, uint32_t column, uint32_t salt) {
  const int32_t value =
      static_cast<int32_t>((row * 7 + column * 3 + salt) % 17) - 8;
  return static_cast<float>(value) * 0.0625f;
}

int main() {
  RT_CHECK(vx_device_open(0, &device));

  vx_queue_info_t qi = { sizeof(qi), nullptr, VX_QUEUE_PRIORITY_NORMAL, 0 };
  RT_CHECK(vx_queue_create(device, &qi, &queue));

  uint64_t isa_flags = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_ISA_FLAGS, &isa_flags));
  if ((isa_flags & VX_ISA_EXT_TCU) == 0) {
    std::cerr << "TCU extension not supported\n";
    cleanup();
    return 1;
  }

  uint64_t device_threads = 0;
  RT_CHECK(vx_device_query(device, VX_CAPS_NUM_THREADS, &device_threads));
  if (device_threads != NUM_THREADS) {
    std::cerr << "Device warp size " << device_threads
              << " does not match NUM_THREADS=" << NUM_THREADS << '\n';
    cleanup();
    return 1;
  }

  const size_t q_elements = kQueryRows * kHeadDimension;
  const size_t kv_elements = kSequenceLength * kHeadDimension;
  const size_t output_elements = kQueryRows * kHeadDimension;

  std::vector<uint16_t> q(q_elements);
  std::vector<uint16_t> k(kv_elements);
  std::vector<uint16_t> v(kv_elements);
  std::vector<float> output(output_elements, 0.0f);
  std::vector<float> reference(output_elements, 0.0f);

  for (uint32_t row = 0; row < kQueryRows; ++row) {
    for (uint32_t column = 0; column < kHeadDimension; ++column) {
      q[row * kHeadDimension + column] =
          float_to_half(input_value(row + kQueryStart, column, 1));
    }
  }
  for (uint32_t row = 0; row < kSequenceLength; ++row) {
    for (uint32_t column = 0; column < kHeadDimension; ++column) {
      k[row * kHeadDimension + column] =
          float_to_half(input_value(row, column, 5));
      v[row * kHeadDimension + column] =
          float_to_half(input_value(row, column, 11));
    }
  }

  for (uint32_t row = 0; row < kQueryRows; ++row) {
    float running_max = -3.402823466e+38f;
    float running_sum = 0.0f;
    for (uint32_t group_base = 0; group_base < kSequenceLength;
         group_base += kKeyGroupSize) {
      float scores[kKeyGroupSize] = {};
      float group_max = -3.402823466e+38f;
      for (uint32_t group_key = 0; group_key < kKeyGroupSize; ++group_key) {
        const uint32_t key = group_base + group_key;
        if (key <= kQueryStart + row) {
          float score = 0.0f;
          for (uint32_t column = 0; column < kHeadDimension; ++column) {
            score += half_to_float(q[row * kHeadDimension + column])
                   * half_to_float(k[key * kHeadDimension + column]);
          }
          score *= 0.25f;
          scores[group_key] = score;
          group_max = score > group_max ? score : group_max;
        }
      }

      const float new_max =
          running_max > group_max ? running_max : group_max;
      const float previous_scale =
          running_sum == 0.0f ? 0.0f : std::exp(running_max - new_max);
      for (uint32_t column = 0; column < kHeadDimension; ++column) {
        reference[row * kHeadDimension + column] *= previous_scale;
      }

      float group_sum = 0.0f;
      for (uint32_t group_key = 0; group_key < kKeyGroupSize; ++group_key) {
        const uint32_t key = group_base + group_key;
        if (key <= kQueryStart + row) {
          const float probability = std::exp(scores[group_key] - new_max);
          const float probability_fp16 =
              half_to_float(float_to_half(probability));
          group_sum += probability;
          for (uint32_t column = 0; column < kHeadDimension; ++column) {
            reference[row * kHeadDimension + column] +=
                probability_fp16
              * half_to_float(v[key * kHeadDimension + column]);
          }
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
  args.causal = 1;

  RT_CHECK(vx_buffer_create(
      device, q.size() * sizeof(uint16_t), VX_MEM_READ, &q_buffer));
  RT_CHECK(vx_buffer_address(q_buffer, &args.q_addr));
  RT_CHECK(vx_buffer_create(
      device, k.size() * sizeof(uint16_t), VX_MEM_READ, &k_buffer));
  RT_CHECK(vx_buffer_address(k_buffer, &args.k_addr));
  RT_CHECK(vx_buffer_create(
      device, v.size() * sizeof(uint16_t), VX_MEM_READ, &v_buffer));
  RT_CHECK(vx_buffer_address(v_buffer, &args.v_addr));
  RT_CHECK(vx_buffer_create(
      device, output.size() * sizeof(float), VX_MEM_WRITE, &output_buffer));
  RT_CHECK(vx_buffer_address(output_buffer, &args.output_addr));

  RT_CHECK(vx_enqueue_write(queue, q_buffer, 0, q.data(), q.size() * sizeof(uint16_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, k_buffer, 0, k.data(), k.size() * sizeof(uint16_t), 0, nullptr, nullptr));
  RT_CHECK(vx_enqueue_write(queue, v_buffer, 0, v.data(), v.size() * sizeof(uint16_t), 0, nullptr, nullptr));
  RT_CHECK(vx_module_load_file(device, "kernel.vxbin", &module_));
  RT_CHECK(vx_module_get_kernel(module_, "main", &kernel));

  uint32_t grid_dim[1] = {1};
  uint32_t block_dim[1] = {NUM_THREADS};
  vx_launch_info_t li = {};
  li.struct_size  = sizeof(li);
  li.kernel       = kernel;
  li.args_host    = &args;
  li.args_size    = sizeof(args);
  li.ndim         = 1;
  li.grid_dim[0]  = grid_dim[0];
  li.block_dim[0] = block_dim[0];
  li.lmem_size    = kLocalMemoryBytes;
  vx_event_h launch_ev = nullptr;
  RT_CHECK(vx_enqueue_launch(queue, &li, 0, nullptr, &launch_ev));
  RT_CHECK(vx_event_wait_value(launch_ev, 1, VX_TIMEOUT_INFINITE));
  vx_event_release(launch_ev);
  {
    vx_event_h rd_ev = nullptr;
    RT_CHECK(vx_enqueue_read(queue, output.data(), output_buffer, 0, output.size() * sizeof(float), 0, nullptr, &rd_ev));
    RT_CHECK(vx_event_wait_value(rd_ev, 1, VX_TIMEOUT_INFINITE));
    vx_event_release(rd_ev);
  }

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

  cleanup();
  if (errors != 0) {
    std::cerr << "FAILED: " << errors << " mismatches, max error "
              << max_error << '\n';
    return 1;
  }

  std::cout << "PASSED: causal FlashAttention forward, Q=8, KV=32, D=16"
            << ", max error=" << max_error << '\n';
  return 0;
}
