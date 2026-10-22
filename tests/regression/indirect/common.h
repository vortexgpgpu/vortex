#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define PRODUCE_MAX_WORDS 32

// produce_k: copies words[] to dst (device-produced indirect arguments).
typedef struct {
  uint64_t dst_addr;
  uint32_t num_words;
  uint32_t words[PRODUCE_MAX_WORDS];
} produce_arg_t;

// vecadd_k: c = a + b.  scale_k: c = a * k.  poison_k: c = POISON.
typedef struct {
  uint64_t a_addr;
  uint64_t b_addr;
  uint64_t c_addr;
  uint32_t n;
  uint32_t k;
} vec_arg_t;

// mark_k: hits[linear block] += 1; dims = {gridDim.xyz, launches}.
typedef struct {
  uint64_t hits_addr;
  uint64_t dims_addr;
} mark_arg_t;

#define POISON 0xDEADBEEFu

#endif
