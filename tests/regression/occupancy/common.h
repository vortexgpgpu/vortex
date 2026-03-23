#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
  uint32_t lmem_words;
  uint64_t out_addr;
} kernel_arg_t;

#endif
