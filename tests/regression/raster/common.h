#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
  uint32_t dst_width;
  uint32_t dst_height;
  uint64_t cbuf_addr;
  uint8_t  cbuf_stride;
  uint32_t cbuf_pitch;
  uint64_t prim_addr;
} kernel_arg_t;

#endif
