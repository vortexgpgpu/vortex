#ifndef _COMMON_H_
#define _COMMON_H_

#include <VX_config.h>
#include <VX_types.h>
#include <stdint.h>

typedef struct {
  uint32_t  dst_width;
  uint32_t  dst_height;
  uint32_t  color;        // packed ARGB src color (used as base before scaling)
  uint32_t  depth;
  uint32_t  backface;     // 0/1
  uint32_t  blend_enable; // 0/1
  uint32_t  a_scale_q16;  // 16.16 fixed-point
  uint32_t  r_scale_q16;
  uint32_t  g_scale_q16;
  uint32_t  b_scale_q16;
} kernel_arg_t;

#endif
