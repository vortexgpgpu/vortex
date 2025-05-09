#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

typedef struct {
  uint32_t grid_dim[2];
  uint64_t A_addr;
  uint64_t x_addr;
  uint64_t y_addr;
  uint32_t M;
  uint32_t N;
} kernel_arg_t;

#endif
