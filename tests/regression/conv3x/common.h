#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

typedef struct {
  uint32_t num_tasks;
  uint32_t width;
  uint32_t log2_width;
  uint64_t lmem_addr;
  uint64_t I_addr;
  uint64_t W_addr;
  uint64_t O_addr;
} kernel_arg_t;

#endif
