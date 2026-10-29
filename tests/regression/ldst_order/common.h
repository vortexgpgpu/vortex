#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define LDST_MARK 0x80000000u

typedef struct {
  uint32_t num_words;   // words in the data buffer
  uint32_t num_threads; // total threads launched
  uint32_t warp_size;   // threads per warp
  uint32_t delay_sweep; // iteration k delays the final store (k % delay_sweep) * delay_step steps
  uint32_t delay_step;
  uint64_t data_addr;   // in: initial words; out: regions 0 and 3 overwritten with LDST_MARK + index
  uint64_t old_addr;    // out: the value each region-0 load returned
  uint64_t sink_addr;   // out: per-thread sum of the other loads (keeps them alive)
} kernel_arg_t;

#endif
