#ifndef _COMMON_H_
#define _COMMON_H_

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

typedef struct {
  uint32_t size;
  uint64_t src_addr;
  uint64_t dst_addr;  
} kernel_arg_t;

#endif