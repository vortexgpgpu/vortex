#ifndef _COMMON_H_
#define _COMMON_H_

#include <VX_config.h>
#include <VX_types.h>
#include <stdint.h>

#define KERNEL_ARG_DEV_MEM_ADDR 0x7ffff000

typedef struct {
  uint32_t  num_tasks;
  uint32_t  dst_width;
  uint32_t  dst_height;  
  uint32_t  color;
  uint32_t  depth;
  bool      backface;
  bool      blend_enable;
  bool      use_sw;
} kernel_arg_t;

#endif