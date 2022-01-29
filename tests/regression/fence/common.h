#ifndef _COMMON_H_
#define _COMMON_H_

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

typedef struct {
  uint32_t num_tasks;
  uint32_t task_size;
  uint32_t src0_addr;
  uint32_t src1_addr;
  uint32_t dst_addr;
} kernel_arg_t;

#endif