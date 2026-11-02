#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Local memory per block: RO_WORDS read-only words, then WIDE_WORDS words per
// thread that it stores to and reads back. RO_WORDS keeps them 16-byte aligned.
#define RO_WORDS   1024
#define WIDE_WORDS 4

// Initial value of local word i: words differ from each other and about half
// of them have the sign bit set.
static inline uint32_t ro_word(uint32_t i) {
  return (i * 0x9E3779B1u) ^ 0xA5C30000u;
}

// The words a thread stores to its wide slot.
static inline uint32_t wide_word(uint32_t tid, uint32_t k) {
  return 0x6C000000u | (tid << 4) | k;
}

typedef struct {
  uint32_t lmem_words;  // local-memory words per block
  uint64_t err_addr;    // out: per-thread error count
  uint64_t dump_addr;   // out: each block's final local memory
  uint64_t wide_addr;   // out: 1 if the kernel reads with wide loads
} kernel_arg_t;

#endif
