#ifndef _COMMON_H_
#define _COMMON_H_

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

#ifndef TYPE
#define TYPE float
#endif

typedef struct {
  uint32_t num_tasks;
  uint32_t size;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;  
} kernel_arg_t;

#endif
