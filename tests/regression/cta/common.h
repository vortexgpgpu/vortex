#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t block_dim[3];
  uint32_t grid_dim[3];
  uint64_t src_addr;
  uint64_t dst_addr;
} kernel_arg_t;

#endif
