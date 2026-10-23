#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Ops 0..6 are the raw MUFU functions (same numbering as the instruction's fn
// field); the rest are the fast-math compositions from vx_mufu.h.
enum {
  OP_RCP = 0,
  OP_RSQRT,
  OP_EX2,
  OP_LG2,
  OP_SINPI,
  OP_COSPI,
  OP_TANH,
  OP_EXPF,
  OP_LOGF,
  OP_SINF,
  OP_COSF,
  OP_SQRTF,
  OP_SIGMOID,
  OP_SILU,
  OP_GELU,
  OP_COUNT
};

typedef struct {
  uint32_t op;
  uint32_t num_points;
  uint32_t repeat;
  uint32_t padding;
  uint64_t src_addr;
  uint64_t dst_addr;
} kernel_arg_t;

#endif
