#include <vx_spawn2.h>
#include "common.h"

// Each warp owns a group of four regions (one word per lane each, one cache
// line per group) and, per iteration, issues in program order:
//
//   load  r2   - misses: heads the group's miss chain (answered from the fill)
//   store r3   - chained behind it
//   store r1   - chained behind it
//   store r0   - chained behind it
//   delay      - warp-uniform countdown, swept per iteration
//   load  r0   - the checked load: must return the store to r0 above
//
// On a write-through cache the stores leave for memory at once and update
// the array when they replay behind the fill. The sweep lands the checked
// load at varying points of the chain's drain, including after the head has
// replayed and before the store to r0 has. The sequence is inline assembly
// so no compiler scheduling moves the requests relative to each other.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto data = reinterpret_cast<uint32_t*>(arg->data_addr);
  auto nval = reinterpret_cast<uint32_t*>(arg->new_addr);
  auto sink = reinterpret_cast<uint32_t*>(arg->sink_addr);
  const uint32_t W    = arg->warp_size;
  const uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t lane = tid % W;
  const uint32_t warp = tid / W;
  const uint32_t num_warps = arg->num_threads / W;
  const uint32_t group_words = 4 * W;
  uint32_t acc = 0;
  uint32_t k = 0;
  for (uint32_t g = warp; (g + 1) * group_words <= arg->num_words; g += num_warps, ++k) {
    uint32_t* r0 = data + g * group_words + lane;
    uint32_t* r1 = r0 + W;
    uint32_t* r2 = r0 + 2 * W;
    uint32_t* r3 = r0 + 3 * W;
    uint32_t i0 = g * group_words + lane;
    uint32_t mark0 = RAW_MARK + i0;
    uint32_t mark1 = RAW_MARK + i0 + W;
    uint32_t mark3 = RAW_MARK + i0 + 3 * W;
    uint32_t delay = (k % arg->delay_sweep) * arg->delay_step;
    uint32_t a, v;
    __asm__ volatile(
      "lw   %[a], 0(%[r2])\n\t"
      "sw   %[m3], 0(%[r3])\n\t"
      "sw   %[m1], 0(%[r1])\n\t"
      "sw   %[m0], 0(%[r0])\n\t"
      "beqz %[n], 2f\n\t"
      "1:\n\t"
      "addi %[n], %[n], -1\n\t"
      "bnez %[n], 1b\n\t"
      "2:\n\t"
      "lw   %[v], 0(%[r0])\n\t"
      : [a] "=&r"(a), [v] "=&r"(v), [n] "+r"(delay)
      : [r0] "r"(r0), [r1] "r"(r1), [r2] "r"(r2), [r3] "r"(r3),
        [m0] "r"(mark0), [m1] "r"(mark1), [m3] "r"(mark3)
      : "memory");
    acc += a;
    nval[i0] = v;
  }
  sink[tid] = acc;
}
