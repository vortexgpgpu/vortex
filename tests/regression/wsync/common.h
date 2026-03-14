#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
  uint32_t failures;
  uint32_t first_iteration;
  uint32_t baseline_gap;
  uint32_t raw_cycle;
  uint32_t sync_cycle;
  uint32_t gap;
  uint32_t checksum;
} lane_result_t;

typedef struct {
  uint32_t num_threads;
  uint32_t iterations;
  uint64_t results_addr;
} kernel_arg_t;

#endif
