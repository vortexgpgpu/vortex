#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

#ifndef UMMA_NRC
  #define UMMA_NRC 8
#endif

// Columns between the TMEM allocation's base and the accumulator
#ifndef UMMA_ACC_OFFSET
  #define UMMA_ACC_OFFSET 0
#endif

#ifndef ITYPE
#define ITYPE fp16
#endif

#ifndef OTYPE
#define OTYPE fp32
#endif

typedef struct {
  uint32_t M, N, K;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif
