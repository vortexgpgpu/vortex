#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t num_points;
  uint64_t src_addr;
  uint64_t dst_addr;
  uint32_t branch_depth;
} kernel_arg_t;

#endif