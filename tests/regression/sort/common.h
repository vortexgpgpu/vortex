#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE int
#endif

typedef struct {
  uint32_t num_points;
  uint64_t src_addr;
  uint64_t dst_addr;  
} kernel_arg_t;

#endif
