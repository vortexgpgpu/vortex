#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define FENCE_MP_MAGIC 0x5a5a0000u

typedef struct {
  uint32_t num_tasks;  // threads launched
  uint32_t count;      // data words passed from producer to consumer
  uint32_t spin_limit; // consumer polls before giving up
  uint32_t pad;
  uint64_t data_addr;  // count words, zero on entry
  uint64_t flag_addr;  // one word, zero on entry
  uint64_t out_addr;   // [0] mismatches, [1] 1 if the consumer timed out, [2] pre-read sum
} kernel_arg_t;

#endif
