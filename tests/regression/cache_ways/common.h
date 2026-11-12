#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
  uint64_t addr;
  uint32_t lines;
  uint32_t line_stride;  // bytes between lines
  uint32_t sectors;      // sectors read per line
  uint32_t sector_bytes;
} kernel_arg_t;

#endif
