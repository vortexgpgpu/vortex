#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
  uint32_t failures;
  uint32_t observed_pending;
  uint32_t observed_phase;
  uint32_t observed_arrived;
  uint32_t register_cycles;
  uint32_t event_cycles;
  uint32_t release_cycles;
  uint32_t wait_iters;
  uint32_t checksum;
} barrier_result_t;

typedef struct {
  uint32_t payload_bytes;
  uint32_t iterations;
  uint32_t num_warps;
  uint32_t mode;
  uint64_t results_addr;
} kernel_arg_t;

enum {
  BARRIER_OVERHEAD_MODE_SOFT = 0,
  BARRIER_OVERHEAD_MODE_HARD = 1,
};

#endif
