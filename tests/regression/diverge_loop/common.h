#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

typedef struct {
  uint32_t num_points;
  uint32_t num_samples;
  uint32_t num_shadows;
  uint64_t t_addr;
  uint64_t color_addr;
  uint64_t occl_addr;
  uint64_t dst_addr;
} kernel_arg_t;

#endif
