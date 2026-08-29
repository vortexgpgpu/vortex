#pragma once

#include <stdint.h>

#define NUM_DIMS     4u
#define NUM_MARKERS  16u
#define MARKER_BASE  0x100u

typedef struct {
  uint32_t count;
  uint32_t diverge;
  uint64_t dst_addr;
} kernel_arg_t;
