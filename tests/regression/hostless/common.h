#ifndef _COMMON_H_
#define _COMMON_H_

#include <cstdint>

#define KERNEL_ARG_DEV_MEM_ADDR 0x9fff0000
#define DEV_SMEM_START_ADDR 0xff000000

#ifndef TYPE
#define TYPE float
#endif

typedef struct {
  uint32_t dim;
  uint64_t addr_a;
  uint64_t addr_b;
  uint64_t addr_dst;
} kernel_arg_t;

#endif
