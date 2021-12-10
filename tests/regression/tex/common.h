#ifndef _COMMON_H_
#define _COMMON_H_

#include <VX_config.h>

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

typedef struct {
  bool      use_sw;
  uint32_t  num_tasks;
  uint8_t   format;
  uint8_t   filter;
  uint8_t   wrapu;
  uint8_t   wrapv;  
  uint8_t   src_logwidth;
  uint8_t   src_logheight;
  uint32_t  src_addr;
  uint32_t  mip_offs[TEX_LOD_MAX+1];  
  uint32_t  dst_width;
  uint32_t  dst_height;
  uint8_t   dst_stride;  
  uint32_t  dst_pitch;
  uint32_t  dst_addr;  
} kernel_arg_t;

#endif