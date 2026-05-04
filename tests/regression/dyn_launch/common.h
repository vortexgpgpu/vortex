#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#define DL_ROLE_PARENT 0u
#define DL_ROLE_CHILD  1u
#define DL_ROLE_TAIL   2u

typedef struct {
  uint32_t role;
  uint32_t _pad;
  uint64_t child_pc;
  uint64_t child_arg_addr;
} kernel_arg_t;

#endif
