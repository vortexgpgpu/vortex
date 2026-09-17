#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

typedef struct {
  uint32_t N;
  uint32_t d;
  uint64_t Q_addr;
  uint64_t K_addr;
  uint64_t S_addr;
  uint64_t P_addr;
  uint64_t V_addr;
  uint64_t O_addr;
} kernel_arg_t;

#endif