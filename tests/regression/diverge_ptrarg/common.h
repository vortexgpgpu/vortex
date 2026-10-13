#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
  uint32_t num_points;
  uint64_t src_addr;   // float[num_points]
  uint64_t dst_addr;   // int32_t[num_points]
} kernel_arg_t;

#endif
