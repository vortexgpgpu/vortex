#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Each thread owns NUM_CASES regions per round, one cache line (REGION_WORDS
// words) each, laid out round-major so neighbouring lines belong to different
// threads and the buffer outgrows the cache across rounds.
#define REGION_WORDS 16
#define NUM_CASES    6
#define LANE_WORDS   8
// Same-set lines: one per thread and round, a page apart, so they compete for
// one cache set and a line being refetched is often evicted before its fill.
#define SET_STRIDE_WORDS 1024

enum {
  CASE_HALF  = 0, // store half the line, then load all of it (same thread)
  CASE_BYTES = 1, // byte/halfword stores, then load the words holding them
  CASE_AMO   = 2, // stores, then AMO add and LR/SC on written and unwritten words
  CASE_STORE = 3, // byte stores only: checked after the kernel
  CASE_WARP  = 4, // one warp stores half the line, the next warp loads all of it
  CASE_LANE  = 5  // a warp's lanes store into one line, then load its unwritten half
};

typedef struct {
  uint64_t buf_addr;
  uint64_t set_addr;
  uint64_t err_addr;
  uint32_t rounds;
  uint32_t num_threads;  // threads in the grid
  uint32_t warp_size;
  uint32_t cross_warp;   // blocks hold an even number of warps
  uint32_t amo;          // the device implements the A extension
  uint32_t phase;        // 0: exercise the cases, 1: re-check every region
  uint32_t rotate;       // phase 1: check the regions of thread (gid + rotate)
} kernel_arg_t;

// Pre-pattern the host writes over the whole buffer.
static inline uint8_t pattern_byte(uint32_t i) {
  return (uint8_t)(i * 37u + (i >> 9) + 0x5Au);
}

static inline uint32_t pattern_word(uint32_t word_idx) {
  uint32_t b = word_idx * 4;
  return (uint32_t)pattern_byte(b)
       | ((uint32_t)pattern_byte(b + 1) << 8)
       | ((uint32_t)pattern_byte(b + 2) << 16)
       | ((uint32_t)pattern_byte(b + 3) << 24);
}

static inline uint32_t store_value(uint32_t region, uint32_t w) {
  return 0xC0000000u | (region << 4) | w;
}

static inline uint32_t region_index(uint32_t round, uint32_t num_threads, uint32_t gid, uint32_t c) {
  return (round * num_threads + gid) * NUM_CASES + c;
}

static inline uint32_t set_line(uint32_t round, uint32_t num_threads, uint32_t gid) {
  return round * num_threads + gid;
}

// Expected word `word_idx` of the same-set buffer once the kernel has run:
// each line's word 0 is stored, the rest keeps the pre-pattern.
static inline uint32_t expected_set_word(uint32_t word_idx) {
  if ((word_idx % SET_STRIDE_WORDS) == 0) {
    return 0xA0000000u | ((word_idx / SET_STRIDE_WORDS) << 4);
  }
  return pattern_word(word_idx);
}

// Owner thread of a CASE_WARP region writes it when it sits in an even warp
// of its block; the thread one warp up reads it.
static inline bool warp_producer(uint32_t gid, uint32_t warp_size) {
  return ((gid / warp_size) & 1) == 0;
}

// Expected content of word w of region `region` once the kernel has run.
static inline uint32_t expected_word(uint32_t region, uint32_t w, const kernel_arg_t* arg) {
  uint32_t p = pattern_word(region * REGION_WORDS + w);
  uint32_t v = store_value(region, w);
  switch (region % NUM_CASES) {
  case CASE_HALF:
    return (w < REGION_WORDS / 2) ? v : p;
  case CASE_BYTES:
    if (w == 1) {
      return (p & ~0x0000ff00u) | 0x0000ee00u;
    }
    if (w == 10) {
      return (p & ~0x0000ffffu) | 0x0000beefu;
    }
    return p;
  case CASE_AMO:
    if (w < 4 && !arg->amo) {
      return v;
    }
    if (!arg->amo) {
      return p;
    }
    if (w == 1) {
      return v + 3;
    }
    if (w < 4) {
      return v;
    }
    if (w == 8) {
      return p + 5;
    }
    if (w == 12) {
      return p + 1;
    }
    return p;
  case CASE_STORE:
    return ((w & 1) == 0) ? ((p & ~0xffu) | (0x30u + w)) : p;
  case CASE_WARP: {
    uint32_t gid = (region / NUM_CASES) % arg->num_threads;
    if (arg->cross_warp && warp_producer(gid, arg->warp_size) && w >= REGION_WORDS / 2) {
      return v;
    }
    return p;
  }
  default: {
    // the region of each warp's lane 0 is shared by the warp's first lanes
    uint32_t gid = (region / NUM_CASES) % arg->num_threads;
    uint32_t lanes = (arg->warp_size < LANE_WORDS) ? arg->warp_size : LANE_WORDS;
    if ((gid % arg->warp_size) == 0 && w < lanes) {
      return v;
    }
    return p;
  }
  }
}

#endif
