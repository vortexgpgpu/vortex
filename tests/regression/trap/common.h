#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// What one trap round records per thread.
#define TRAP_WORDS 3 // site, mepc, mcause | (handler tag << 8)
#define TRAP_ROUNDS 2

typedef struct {
  uint32_t num_points;
  uint64_t dst_addr;
} kernel_arg_t;

#endif
