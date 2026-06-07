#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE int64_t
#endif

#define SHIFT_BITS 47

typedef struct {
  uint32_t num_points;
  uint64_t src0_addr;
  uint64_t dst_addr;  
} kernel_arg_t;

#endif
