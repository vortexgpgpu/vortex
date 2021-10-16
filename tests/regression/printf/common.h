#ifndef _COMMON_H_
#define _COMMON_H_

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

typedef struct {
  uint32_t num_points;
  uint32_t src_ptr;
} kernel_arg_t;

#endif