#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef NUM_THREADS
#define NUM_THREADS 32
#endif


static constexpr uint32_t kQueryRows = 16;
#ifndef K_TOTAL_QUERY_ROWS
#define K_TOTAL_QUERY_ROWS 32
#endif
static constexpr uint32_t kTotalQueryRows = K_TOTAL_QUERY_ROWS;
#ifndef K_SEQUENCE_LENGTH
#define K_SEQUENCE_LENGTH 1024
#endif
static constexpr uint32_t kSequenceLength = K_SEQUENCE_LENGTH;
static constexpr uint32_t kHeadDimension = 16;
static constexpr uint32_t kQueryStart =
    kSequenceLength > kTotalQueryRows ? kSequenceLength - kTotalQueryRows : 0;
static constexpr uint32_t kKeyGroupSize = 16;
static constexpr uint32_t kLocalMemoryBytes = 8192;

static constexpr uint32_t kQueryTiles = kTotalQueryRows / kQueryRows;
static_assert(kTotalQueryRows % kQueryRows == 0,
              "total query rows must be an exact number of query tiles");

#ifdef FA_ENABLE_TIMING
enum timing_counter_t {
  kTimingTotal,
  kTimingInit,
  kTimingQk,
  kTimingSoftmax,
  kTimingPv,
  kTimingFinal,
  kTimingCount
};
#endif

typedef struct {
  uint64_t q_addr;
  uint64_t k_addr;
  uint64_t v_addr;
  uint64_t output_addr;
#ifdef FA_ENABLE_TIMING
  uint64_t timing_addr;
#endif
} kernel_arg_t;

static inline float exp2_taylor(float x) {
  return 1.0f + x + (0.5f * x * x);
}

#endif
