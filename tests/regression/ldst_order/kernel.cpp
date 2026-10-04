#include <vx_spawn2.h>
#include "common.h"

// Each warp owns a group of four regions (one word per lane each) and, per
// iteration, issues in program order:
//
//   load  r2   - misses: heads the group's miss chain (answered from the fill)
//   store r3   - chained behind it, so the requests after it replay through
//                the data array once the fill lands
//   load  r0   - the checked load, chained behind that store
//   delay      - warp-uniform countdown, swept per iteration
//   store r0   - must not be visible to the load of r0 above
//
// The sweep lands the final store at the cache at varying points of the
// chain's drain, including after the fill and before the r0 load replays.
// Regions 1 and 2 are only read (r1 afterwards, to keep the group's lines
// busy). The sequence is inline assembly so no compiler scheduling or
// divergence handling moves the requests relative to each other.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto data = reinterpret_cast<uint32_t*>(arg->data_addr);
  auto old  = reinterpret_cast<uint32_t*>(arg->old_addr);
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
    uint32_t mark0 = LDST_MARK + i0;
    uint32_t mark3 = LDST_MARK + i0 + 3 * W;
    uint32_t delay = (k % arg->delay_sweep) * arg->delay_step;
    uint32_t a, b, v;
    __asm__ volatile(
      "lw   %[a], 0(%[r2])\n\t"
      "sw   %[m3], 0(%[r3])\n\t"
      "lw   %[v], 0(%[r0])\n\t"
      "beqz %[n], 2f\n\t"
      "1:\n\t"
      "addi %[n], %[n], -1\n\t"
      "bnez %[n], 1b\n\t"
      "2:\n\t"
      "sw   %[m0], 0(%[r0])\n\t"
      "lw   %[b], 0(%[r1])\n\t"
      : [a] "=&r"(a), [b] "=&r"(b), [v] "=&r"(v), [n] "+r"(delay)
      : [r0] "r"(r0), [r1] "r"(r1), [r2] "r"(r2), [r3] "r"(r3),
        [m0] "r"(mark0), [m3] "r"(mark3)
      : "memory");
    acc += a + b;
    old[i0] = v;
  }
  sink[tid] = acc;
}
