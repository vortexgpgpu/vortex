#ifndef _COMMON_H_
#define _COMMON_H_

#define NUM_LOADS 8

typedef struct {
  uint32_t num_tasks;
  uint32_t size;
  uint32_t stride;  
  uint64_t src0_addr;
  uint64_t src1_addr;
  uint64_t dst_addr;  
} kernel_arg_t;

#endif