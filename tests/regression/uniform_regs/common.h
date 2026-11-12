#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// results[] written by the uniform instructions, checked by the host
enum {
  RES_SHIFT,     // (1 << (XLEN-1)) >> (XLEN-3) = 4
  RES_DIV,       // (n * scale) / scale
  RES_LOOP,      // a uniform loop counting to n
  RES_LUI,       // lui 0x80000, then srai XLEN-1: all ones
  RES_SUB,       // n - scale
  RES_NT,        // csrr NUM_THREADS
  RES_XSTORE,    // n stored per thread through a uniform store operand
  RES_XLOAD,     // n loaded per thread from a uniform address
  RES_FLOAD,     // scale's bits through flw/fsw at uniform addresses
  RES_SUBF,      // n - 1, the uniform value first
  RES_SLTF,      // scale < n, the uniform value first
  RES_BRF,       // branches on a uniform first operand
  RES_WCTL,      // tmc, bar and arrive/wait on uniform operands passed
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
