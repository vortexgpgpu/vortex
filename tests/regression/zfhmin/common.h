#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define ZF_NUM_RM 6

enum {
  ZF_MODE_NARROW_S = 0,
  ZF_MODE_WIDEN_S  = 1,
  ZF_MODE_NARROW_D = 2,
  ZF_MODE_WIDEN_D  = 3,
  ZF_MODE_FLAGS    = 4,
  ZF_MODE_BOX      = 5,
};

enum {
  ZF_OP_H_S = 0,
  ZF_OP_S_H = 1,
  ZF_OP_H_D = 2,
  ZF_OP_D_H = 3,
};

#define ZF_BOX_OUTS 5

typedef struct {
  uint32_t op;
  uint32_t rm;
  uint64_t bits;
} zf_flag_case_t;

typedef struct {
  uint64_t result;
  uint32_t fflags;
  uint32_t pad;
} zf_flag_out_t;

typedef struct {
  uint32_t mode;
  uint32_t count;
  uint32_t frm;
  uint32_t pad;
  uint64_t src_addr;
  uint64_t dst_addr;
  uint64_t aux_addr;
} kernel_arg_t;

#endif
