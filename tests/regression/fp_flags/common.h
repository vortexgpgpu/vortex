#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Sign-injection, min/max and compare ops, single then double precision.
enum {
  FF_OP_FSGNJ = 0,
  FF_OP_FSGNJN,
  FF_OP_FSGNJX,
  FF_OP_FMIN,
  FF_OP_FMAX,
  FF_OP_FEQ,
  FF_OP_FLT,
  FF_OP_FLE,
  FF_NUM_OPS,
};

#define FF_OP_D(op) ((op) + FF_NUM_OPS)

// raw: 64-bit FP registers only; a single-precision op reads its operands as
// the raw 64-bit register values, which may be improperly NaN-boxed, and the
// result is checked in full, boxing included.
typedef struct {
  uint32_t op;
  uint32_t raw;
  uint64_t a;
  uint64_t b;
} ff_case_t;

typedef struct {
  uint64_t result;
  uint32_t fflags;
  uint32_t pad;
} ff_out_t;

typedef struct {
  uint32_t count;
  uint32_t pad;
  uint64_t src_addr;
  uint64_t dst_addr;
} kernel_arg_t;

#endif
