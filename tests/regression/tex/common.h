#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <VX_config.h>
#include <VX_types.h>
#include <graphics.h>

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

using FloatX = float;
// using FloatX = cocogfx::TFixed<24, int64_t>;

typedef struct {
  bool      use_sw;  
  uint32_t  num_tasks;  
  uint32_t  dst_width;
  uint32_t  dst_height;
  uint8_t   dst_stride;  
  uint32_t  dst_pitch;
  uint64_t  dst_addr;
  uint8_t   filter;
  graphics::TexDCRS dcrs; 
} kernel_arg_t;

#endif