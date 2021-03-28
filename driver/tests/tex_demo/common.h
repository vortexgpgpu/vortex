#ifndef _COMMON_H_
#define _COMMON_H_

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

struct kernel_arg_t {
  uint32_t num_tasks;
  uint32_t src_width;
  uint32_t src_height;
  uint32_t src_stride;
  uint32_t src_pitch;
  uint32_t src_ptr;
  uint32_t dst_width;
  uint32_t dst_height;
  uint32_t dst_stride;  
  uint32_t dst_pitch;
  uint32_t dst_ptr;  
};

#endif