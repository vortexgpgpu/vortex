#ifndef _COMMON_H_
#define _COMMON_H_

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

typedef struct {
  uint32_t num_tasks;
  uint8_t format;
  uint8_t filter;
  uint8_t wrap;
  uint8_t use_sw;
  uint32_t lod;
  uint8_t src_logWidth;
  uint8_t src_logHeight;
  uint8_t src_stride;
  uint8_t src_pitch;
  uint32_t src_ptr;
  uint32_t dst_width;
  uint32_t dst_height;
  uint8_t dst_stride;  
  uint32_t dst_pitch;
  uint32_t dst_ptr;  
} kernel_arg_t;

#endif