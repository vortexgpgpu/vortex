#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define LMEM_MARK 0x80000000u

// Local-memory layout per block, in words: read-only words first, then one
// RAW and one WAR word per warp and iteration.
#define RO_WORDS 128

typedef struct {
  uint32_t num_iters;   // iterations per warp
  uint32_t warp_size;   // threads per warp
  uint32_t lmem_words;  // local-memory words per block
  uint32_t raw_base;    // first RAW word
  uint32_t war_base;    // first WAR word
  uint64_t err_addr;    // out: per-thread error count
  uint64_t dump_addr;   // out: each block's final local memory
} kernel_arg_t;

#endif
