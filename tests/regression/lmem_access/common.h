#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define LMEM_MARK 0x80000000u

// Local-memory layout per block, in words: read-only words first, then one
// RAW, one WAR and one AMO word per warp and iteration, then a read-only
// bank-conflict region of warp_size^2 words, then per warp and iteration
// SUB_WORDS(warp_size) words of sub-word stores and one same-word store word,
// then one cross-warp word per iteration, then the atomic counter and the
// LR/SC counter that every thread of the block updates.
#define RO_WORDS 128

// retry bound of the LR/SC increment
#define LRSC_MAX_RETRIES 4096

// words holding one byte store of each lane, four lanes to a word
#define SUB_WORDS(w) (((w) + 3) / 4)

// The byte a lane stores into its sub-word store word, and the word a lane
// stores into the warp's same-word store word.
static inline uint8_t sub_byte(uint32_t slot, uint32_t lane) {
  return (uint8_t)(0x40 + slot * 7 + lane);
}
static inline uint32_t same_word(uint32_t slot, uint32_t lane) {
  return 0x80000000u | (slot << 8) | lane;
}

// The value one warp stores into iteration k's cross-warp word.
static inline uint32_t xwarp_word(uint32_t k) {
  return 0xC0000000u | (k << 12) | 0x5A5u;
}

// Initial value of word i: every byte differs and the high bytes are set, so
// sub-word loads see distinct, sign-extended values.
static inline uint32_t ro_word(uint32_t i) {
  return (i * 0x9E3779B1u) ^ 0xA5C30000u;
}

typedef struct {
  uint32_t num_iters;   // iterations per warp
  uint32_t warp_size;   // threads per warp
  uint32_t lmem_words;  // local-memory words per block
  uint32_t raw_base;    // first RAW word
  uint32_t war_base;    // first WAR word
  uint32_t amo_base;    // first AMO word
  uint32_t cf_base;     // first bank-conflict word
  uint32_t sub_base;    // first sub-word store word
  uint32_t same_base;   // first same-word store word
  uint32_t xwarp_base;  // first cross-warp word
  uint32_t count_word;  // atomic counter
  uint32_t lrsc_word;   // LR/SC counter
  uint32_t lrsc;        // run the LR/SC phase
  uint64_t err_addr;    // out: per-thread error count
  uint64_t dump_addr;   // out: each block's final local memory
} kernel_arg_t;

#endif
