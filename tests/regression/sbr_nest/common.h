#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t num_points;   // number of threads to write
  uint32_t depth;        // nesting depth to exercise (>= NUM_THREADS to overflow)
  uint64_t dst_addr;
} kernel_arg_t;

#endif
