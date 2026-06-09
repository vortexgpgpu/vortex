#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint64_t dst_addr; // num_warps * threads_per_warp uint32_t output words
  uint64_t tp_addr;  // 4 * num_warps * threads_per_warp uint32_t transpose output words
} kernel_arg_t;

#endif
