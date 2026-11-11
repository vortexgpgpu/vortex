#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// results[] written by the uniform instructions, checked by the host
enum {
  RES_SHL40,     // 64-bit: 1 << 40 (two words)
  RES_SHL40_HI,
  RES_SRL,       // (1 << 40) >> 38
  RES_DIV,       // (n * scale) / scale
  RES_LOOP,      // a uniform loop counting to n
  RES_LUI_HI,    // lui 0x80000 sign-extends to 64 bits: upper word
  RES_ADDW,      // addw wraps at 32 bits and sign-extends
  RES_NT,        // csrr NUM_THREADS
  RES_XSTORE,    // n stored per thread through a uniform store operand
  RES_COUNT
};

typedef struct {
  uint32_t num_points;
  uint32_t scale;
  uint64_t src_addr;
  uint64_t dst_addr;
  uint64_t res_addr;
} kernel_arg_t;

#define ARG_N     0
#define ARG_SCALE 4
#define ARG_SRC   8
#define ARG_DST   16
#define ARG_RES   24

#endif
