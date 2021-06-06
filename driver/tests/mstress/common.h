#ifndef _COMMON_H_
#define _COMMON_H_

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

#define NUM_LOADS 8

struct kernel_arg_t {
  uint32_t num_tasks;
  uint32_t size;
  uint32_t stride;  
  uint32_t addr_ptr;
  uint32_t src_ptr;
  uint32_t dst_ptr;  
};

#endif