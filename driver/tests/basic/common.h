#ifndef _COMMON_H_
#define _COMMON_H_

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

struct kernel_arg_t {
  uint32_t count;
  uint32_t src_ptr;
  uint32_t dst_ptr;  
};

#endif