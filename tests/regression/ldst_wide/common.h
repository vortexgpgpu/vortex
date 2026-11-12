#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Read-only words the wide loads read, in local memory and in a global buffer,
// followed in local memory by a SLOT_WORDS-word slot per thread that the wide
// stores fill; each thread has a slot in a global buffer too.
#define RO_WORDS   1024
#define SLOT_WORDS 16

// The global read-only words differ from the local ones.
#define GLOBAL_SALT 0x0F0F1234u

// Read-only word i: words differ from each other and about half of them have
// the sign bit set.
static inline uint32_t ro_word(uint32_t i) {
  return (i * 0x9E3779B1u) ^ 0xA5C30000u;
}

static inline uint32_t wide_word(uint32_t tid, uint32_t k) {
  return 0x6C000000u | (tid << 4) | k;
}

static inline uint32_t alt_word(uint32_t tid, uint32_t k) {
  return 0x3A000000u | (tid << 4) | k;
}

// A slot's final word k: odd lanes overwrite words 0..3 with a second wide
// store, and a scalar store overwrites word 7 after the wide store of 5..8.
static inline uint32_t slot_word(uint32_t tid, uint32_t lid, uint32_t k) {
  if ((k < 4 && (lid & 1)) || k == 7) {
    return alt_word(tid, k);
  }
  return wide_word(tid, k);
}

typedef struct {
  uint32_t lmem_words;  // local-memory words per block
  uint64_t src_addr;    // global read-only words
  uint64_t dst_addr;    // global slots
  uint64_t err_addr;    // out: per-thread error count
  uint64_t dump_addr;   // out: each block's final local memory
  uint64_t wide_addr;   // out: 1 if the kernel uses the wide forms
} kernel_arg_t;

#endif
