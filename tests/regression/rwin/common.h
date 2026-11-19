#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

enum {
  CHK_PREFIX = 0,   // a + b through x37/x69 written and read under vx_rpfx
  CHK_STICKY,       // (a + 1) + b * b through x102/x103 under vx_rwin
  CHK_X0,           // x0 reads zero with a nonzero rs1 window
  CHK_FP,           // a through f225
  CHK_ISOLATED,     // a ^ 0x55 kept in x200 across compiler-generated code
  CHK_COUNT
};

typedef struct {
  uint32_t num_points;
  uint32_t padding;
  uint64_t src_addr;
  uint64_t dst_addr;
} kernel_arg_t;

#endif
