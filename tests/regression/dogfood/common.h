#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint32_t testid;
  uint32_t num_tasks;  
  uint32_t task_size;  
  uint64_t src0_addr;
  uint64_t src1_addr;
  uint64_t dst_addr;  
} kernel_arg_t;

// The fflags-lanes test masks these lanes off while their operands would raise
// a flag.
inline bool fflags_lane_masked(uint32_t lane, uint32_t i, uint32_t num_threads) {
  return (num_threads >= 4) && (((i % 4) == 0 && lane == 1) || ((i % 4) == 2 && lane == 0)
                             || ((i % 4) == 3 && lane == 1));
}

#endif