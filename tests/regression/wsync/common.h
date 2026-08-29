#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Source buffer layout: WSYNC_BUF_LINES cache lines, each WSYNC_LINE_WORDS
// uint32_t words (= one 64-byte cache line).  Total buffer = 64 KiB, which
// is 4× the 16 KiB L1 D-cache, guaranteeing L1 misses on every access.
#define WSYNC_BUF_LINES  1024u
#define WSYNC_LINE_WORDS 16u

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
  uint64_t src_addr;
  uint64_t results_addr;
} kernel_arg_t;

#endif
