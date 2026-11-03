#include <vx_spawn2.h>
#include "common.h"

// A warm chain holds absolute device pointers, so each step is a bare
// load-to-use. A cold chain is never uploaded: its next element comes from an
// LCG that runs independently of the loads, and only the loaded value masked
// to zero at run time joins the address, keeping each load dependent on the
// last. The loops step CHASE_UNROLL loads per branch, keeping the branch's
// stall out of the measured latency.
#define HOT_STEP   p = *reinterpret_cast<const volatile uintptr_t*>(p)
#define HOT_STEP8  HOT_STEP; HOT_STEP; HOT_STEP; HOT_STEP; \
                   HOT_STEP; HOT_STEP; HOT_STEP; HOT_STEP
#define COLD_STEP  i = (i * CHASE_LCG_A + CHASE_LCG_C) & last; \
                   p = *reinterpret_cast<const volatile uintptr_t*>(base + i * CHASE_STRIDE + (p & mask))
#define COLD_STEP8 COLD_STEP; COLD_STEP; COLD_STEP; COLD_STEP; \
                   COLD_STEP; COLD_STEP; COLD_STEP; COLD_STEP

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uintptr_t p;
  if (arg->cold) {
    uintptr_t base = arg->next_addr;
    uintptr_t mask = arg->mask;
    uint32_t last = arg->elements - 1;
    uint32_t i = 0;
    p = 0;
    for (uint32_t n = 0; n < arg->iters; n += CHASE_UNROLL) {
      COLD_STEP8;
    }
    p = i;
  } else {
    p = arg->first_addr;
    for (uint32_t n = 0; n < arg->warm; ++n) {
      HOT_STEP;
    }
    for (uint32_t n = 0; n < arg->iters; n += CHASE_UNROLL) {
      HOT_STEP8;
    }
  }
  *reinterpret_cast<uint64_t*>(arg->out_addr) = p;
}
