#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define FENCE_MP_MAGIC 0x5a5a0000u

// How the producer publishes the flag and the consumer observes it.
enum {
  MP_FENCE   = 0, // full fences on both sides
  MP_ACQ_REL = 1, // a release fence before the flag store, an acquire fence after each flag load
  MP_AMO     = 2  // the flag swapped in by an amoswap.rl, read by an amoor.aq
};

typedef struct {
  uint32_t num_tasks;  // threads launched
  uint32_t count;      // data words passed from producer to consumer
  uint32_t spin_limit; // consumer polls before giving up
  uint32_t mode;       // MP_*
  uint64_t data_addr;  // count words, zero on entry
  uint64_t flag_addr;  // one word, zero on entry
  uint64_t out_addr;   // [0] mismatches, [1] 1 if the consumer timed out, [2] pre-read sum
} kernel_arg_t;

#endif
