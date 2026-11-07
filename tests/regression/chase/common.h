#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Elements are 128 B apart; the LCG has full period over a power-of-two
// element count (c odd, a = 1 mod 4).
#define CHASE_STRIDE 128
#define CHASE_LCG_A  1664525u
#define CHASE_LCG_C  1013904223u
#define CHASE_UNROLL 8

typedef struct {
  uint32_t warm;
  uint32_t iters;
  uint32_t cold;
  uint32_t elements;
  uint64_t mask;
  uint64_t first_addr;
  uint64_t next_addr;
  uint64_t out_addr;
} kernel_arg_t;

#endif
