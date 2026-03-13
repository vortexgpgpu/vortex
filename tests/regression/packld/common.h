#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// Number of test vectors per thread
#ifndef NUM_POINTS
#define NUM_POINTS 16
#endif

typedef struct {
  uint32_t num_tasks;   // total thread count
  uint64_t src_addr;    // byte array: 4*num_tasks*NUM_POINTS elements
  uint64_t dst_lb_addr; // float output for vx_packlb_f: num_tasks*NUM_POINTS floats
  uint64_t dst_lh_addr; // float output for vx_packlh_f: num_tasks*NUM_POINTS floats
} kernel_arg_t;

#endif // _COMMON_H_
