#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Local memory per block: RO_WORDS read-only words, then SLOT_WORDS words per
// thread that it stores to, then, from a 16-byte boundary, WIDE_WORDS words per
// thread that it stores to and reads back with wide loads.
#define RO_WORDS   1024
#define SLOT_WORDS 5
#define WIDE_WORDS 4

static inline uint32_t wide_base(uint32_t block) {
  return (RO_WORDS + block * SLOT_WORDS + 3) & ~3u;
}

// Global input words read past each thread's index (the largest LDG offset).
#define IN_PAD 8

// Initial value of local word i and global input word i: bytes differ from
// word to word and about half of them have the sign bit set, so sub-word loads
// see both sign- and zero-extended values.
static inline uint32_t ro_word(uint32_t i) {
  return (i * 0x9E3779B1u) ^ 0xA5C30000u;
}
static inline uint32_t in_word(uint32_t i) {
  return (i * 0x85EBCA77u) ^ 0x5AC30000u;
}

// The words a thread stores: local slot word k, and global output word k.
static inline uint32_t slot_word(uint32_t tid, uint32_t k) {
  return 0xC0000000u | (tid << 4) | k;
}
static inline uint32_t out_word(uint32_t tid, uint32_t k) {
  return 0x30000000u | (tid << 4) | k;
}
static inline uint32_t wide_word(uint32_t tid, uint32_t k) {
  return 0x6C000000u | (tid << 4) | k;
}

typedef struct {
  uint32_t lmem_words;  // local-memory words per block
  uint64_t in_addr;     // global input words, one per thread + IN_PAD
  uint64_t out_addr;    // global output, SLOT_WORDS words per thread
  uint64_t err_addr;    // out: per-thread error count
  uint64_t dump_addr;   // out: each block's final local memory
} kernel_arg_t;

#endif
