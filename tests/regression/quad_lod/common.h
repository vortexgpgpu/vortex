#ifndef _COMMON_H_
#define _COMMON_H_

typedef struct {
  uint64_t dst_addr;  // num_warps * threads_per_warp uint32_t output words
  uint32_t logw;      // log2 texture width
  uint32_t logh;      // log2 texture height
} kernel_arg_t;

#endif
