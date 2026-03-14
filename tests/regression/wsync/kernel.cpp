#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

static constexpr uint32_t kMinDrainExtra = 16;

static inline uint32_t mix32(uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}

typedef struct {
  uint32_t numerators[8];
  uint32_t denominators[8];
} div_inputs_t;

static inline void fill_div_inputs(div_inputs_t* inputs, uint32_t seed) {
  inputs->numerators[0] = mix32(seed ^ 0x13579bdfu);
  inputs->numerators[1] = mix32(seed ^ 0x2468ace0u);
  inputs->numerators[2] = mix32(seed ^ 0xfdb97531u);
  inputs->numerators[3] = mix32(seed ^ 0x89abcdefu);
  inputs->denominators[0] = mix32(seed ^ 0x31415926u) | 1u;
  inputs->denominators[1] = mix32(seed ^ 0x27182818u) | 1u;
  inputs->denominators[2] = mix32(seed ^ 0xfeedfaceu) | 1u;
  inputs->denominators[3] = mix32(seed ^ 0xc001d00du) | 1u;

  seed = mix32(seed ^ 0x9e3779b9u);
  inputs->numerators[4] = mix32(seed ^ 0xa5a5a5a5u);
  inputs->numerators[5] = mix32(seed ^ 0x5a5a5a5au);
  inputs->numerators[6] = mix32(seed ^ 0xdeadc0deu);
  inputs->numerators[7] = mix32(seed ^ 0xbaadf00du);
  inputs->denominators[4] = mix32(seed ^ 0x01234567u) | 1u;
  inputs->denominators[5] = mix32(seed ^ 0x76543210u) | 1u;
  inputs->denominators[6] = mix32(seed ^ 0x0f0f0f0fu) | 1u;
  inputs->denominators[7] = mix32(seed ^ 0xf0f0f0f0u) | 1u;
}

static inline uint32_t div_batch(const div_inputs_t& inputs) {
  uint32_t r0, r1, r2, r3;

  __asm__ volatile(
    "divu %0, %4, %8\n\t"
    "divu %1, %5, %9\n\t"
    "divu %2, %6, %10\n\t"
    "divu %3, %7, %11\n\t"
    : "=&r"(r0), "=&r"(r1), "=&r"(r2), "=&r"(r3)
    : "r"(inputs.numerators[0]), "r"(inputs.numerators[1]),
      "r"(inputs.numerators[2]), "r"(inputs.numerators[3]),
      "r"(inputs.denominators[0]), "r"(inputs.denominators[1]),
      "r"(inputs.denominators[2]), "r"(inputs.denominators[3]));

  uint32_t q0, q1, q2, q3;
  __asm__ volatile(
    "divu %0, %4, %8\n\t"
    "divu %1, %5, %9\n\t"
    "divu %2, %6, %10\n\t"
    "divu %3, %7, %11\n\t"
    : "=&r"(q0), "=&r"(q1), "=&r"(q2), "=&r"(q3)
    : "r"(inputs.numerators[4]), "r"(inputs.numerators[5]),
      "r"(inputs.numerators[6]), "r"(inputs.numerators[7]),
      "r"(inputs.denominators[4]), "r"(inputs.denominators[5]),
      "r"(inputs.denominators[6]), "r"(inputs.denominators[7]));

  return r0 ^ r1 ^ r2 ^ r3 ^ q0 ^ q1 ^ q2 ^ q3;
}

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t tid = threadIdx.x;
  uint32_t iterations = arg->iterations;
  auto results = reinterpret_cast<lane_result_t*>(arg->results_addr);
  lane_result_t result = {};

  vx_wsync();
  uint32_t baseline_raw = csr_read(VX_CSR_MCYCLE);
  __asm__ volatile("" : : "r"(baseline_raw) : "memory");
  uint32_t baseline_sync = static_cast<uint32_t>(vx_rdcycle_sync());
  __asm__ volatile("" : : "r"(baseline_sync) : "memory");
  result.baseline_gap = baseline_sync - baseline_raw;

  for (uint32_t iter = 0; iter < iterations; ++iter) {
    uint32_t seed = mix32((tid + 1) * 0x10001u + iter * 0x45d9f3bu);
    div_inputs_t batch0, batch1, batch2, batch3;
    fill_div_inputs(&batch0, seed);
    fill_div_inputs(&batch1, seed ^ 0x9e3779b9u);
    fill_div_inputs(&batch2, seed ^ 0x85ebca6bu);
    fill_div_inputs(&batch3, seed ^ 0xc2b2ae35u);

    // Queue older long-latency warp instructions, then compare an unsynchronized
    // cycle sample against a synchronized sample that drains the backlog first.
    uint32_t batch_checksum = div_batch(batch0)
                            ^ div_batch(batch1)
                            ^ div_batch(batch2)
                            ^ div_batch(batch3);
    __asm__ volatile("" : : "r"(batch_checksum) : "memory");
    uint32_t raw_cycle = csr_read(VX_CSR_MCYCLE);
    __asm__ volatile("" : : "r"(raw_cycle) : "memory");
    uint32_t sync_cycle = static_cast<uint32_t>(vx_rdcycle_sync());
    uint32_t gap = sync_cycle - raw_cycle;
    __asm__ volatile("" : : "r"(sync_cycle) : "memory");

    result.checksum ^= batch_checksum ^ raw_cycle ^ sync_cycle ^ gap;

    if (gap <= result.baseline_gap + kMinDrainExtra) {
      if (0 == result.failures) {
        result.first_iteration = iter;
        result.raw_cycle = raw_cycle;
        result.sync_cycle = sync_cycle;
        result.gap = gap;
      }
      ++result.failures;
    }
  }

  results[tid] = result;
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  uint32_t grid_dim = 1;
  uint32_t block_dim = arg->num_threads;
  return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
