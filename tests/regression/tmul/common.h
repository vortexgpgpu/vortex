#ifndef _COMMON_H_
#define _COMMON_H_

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

#ifndef TYPE
#define TYPE int
#endif

typedef struct {
  uint32_t num_tasks;
  uint32_t size;
  uint32_t A_addr;
  uint32_t B_addr;
  uint32_t C_addr;
} kernel_arg_t;

#endif