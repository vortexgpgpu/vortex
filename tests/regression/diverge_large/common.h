#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
  uint32_t num_points;
  uint64_t out_addr;    // out: large_cfg(i)
} kernel_arg_t;

// Shared by the kernel and the host reference.

static __attribute__((noinline)) uint32_t mix(uint32_t acc, uint32_t n) {
  return (acc ^ (n * 0x9e3779b9u)) * 33u + n;
}

// A function far past the divergence lowering's 100-basic-block complexity
// threshold, made of lane-divergent branches.
#define LCFG_STEP(n) if (((x * (2 * (n) + 1)) >> ((n) % 5)) & 1) { acc = mix(acc, n); }
#define LCFG_STEP8(n) LCFG_STEP(n) LCFG_STEP(n + 1) LCFG_STEP(n + 2) LCFG_STEP(n + 3) \
                      LCFG_STEP(n + 4) LCFG_STEP(n + 5) LCFG_STEP(n + 6) LCFG_STEP(n + 7)

static __attribute__((noinline)) uint32_t large_cfg(uint32_t x) {
  uint32_t acc = x;
  LCFG_STEP8(0)  LCFG_STEP8(8)  LCFG_STEP8(16) LCFG_STEP8(24)
  LCFG_STEP8(32) LCFG_STEP8(40) LCFG_STEP8(48) LCFG_STEP8(56)
  return acc;
}

#endif
