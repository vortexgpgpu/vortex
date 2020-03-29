#ifndef _COMMON_H_
#define _COMMON_H_

#include "../../../runtime/config.h"

#define MAX_CORES NUMBER_CORES

#define MAX_WARPS NW

#define MAX_THREADS NT

#define BLOCK_SIZE GLOBAL_BLOCK_SIZE_BYTES

#define KERNEL_ARG_DEV_MEM_ADDR 0x7fffff00

struct kernel_arg_t {
  uint32_t src0_ptr;
  uint32_t src1_ptr;
  uint32_t dst_ptr;
  uint32_t stride;
};

#endif