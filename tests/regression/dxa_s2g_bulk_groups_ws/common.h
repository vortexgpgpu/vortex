#ifndef _DXA_S2G_BULK_GROUPS_WS_COMMON_H_
#define _DXA_S2G_BULK_GROUPS_WS_COMMON_H_

#include <stdint.h>

// Eight logical warps make the role split explicit.  The Makefile overrides
// NUM_WARPS for this regression; the kernel still checks the runtime device
// capability and skips cleanly on smaller configurations.
#define S2G_WS_WARPS       8
#define S2G_WS_ITERS       6

#define S2G_WS_OUT_WORDS   64
#define S2G_WS_SCALE_WORDS 8
#define S2G_WS_STATE0_WORDS 32
#define S2G_WS_STATE1_WORDS 16
#define S2G_WS_META_WORDS  4

#define S2G_WS_OUT_STRIDE   (S2G_WS_OUT_WORDS + 8)
#define S2G_WS_SCALE_STRIDE (S2G_WS_SCALE_WORDS + 4)
#define S2G_WS_STATE0_STRIDE (S2G_WS_STATE0_WORDS + 4)
#define S2G_WS_STATE1_STRIDE (S2G_WS_STATE1_WORDS + 4)
#define S2G_WS_META_STRIDE  (S2G_WS_META_WORDS + 4)

typedef struct {
  uint32_t mode;       // 0 = serial wait<0>, 1 = pipelined partial waits
  uint32_t iterations;
} kernel_arg_t;

#endif
