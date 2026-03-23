#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

typedef struct {
  uint32_t num_points;
  uint64_t src0_addr;
  uint64_t src1_addr;
  uint64_t dst_addr;  
} kernel_arg_t;

#endif
