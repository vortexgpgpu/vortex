#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

typedef struct {
  uint32_t grid_dim[2];
  uint32_t width;
  uint64_t I_addr;
  uint64_t W_addr;
  uint64_t O_addr;
  bool     use_lmem;
} kernel_arg_t;

#endif
