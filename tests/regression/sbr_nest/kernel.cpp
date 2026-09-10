#include <vx_spawn2.h>
#include "common.h"

// Known-bad gate for the fused split-branch (vx_sbr) reconvergence fix.
//
// A TRULY-NESTED divergent chain: level k lives inside every level < k, so the
// reconvergence stack sees real nesting depth = `depth`, not a sequence of
// independent diamonds. Each level's guard `tid >= k+1` peels one more lane, so
// with NUM_THREADS lanes the first NUM_THREADS-1 levels are genuinely divergent
// and every deeper level is uniform (no lane qualifies) yet still nests.
//
// That deep-uniform tail is exactly what overflowed vx_sbr's old always-push
// model (which pushed an IPDOM entry per fused branch regardless of divergence):
// nesting `depth` > NUM_THREADS-1 aborts with "IPDOM stack is full". The fix
// pushes only on real divergence, so the costly stack tops out at NUM_THREADS-1
// and this passes. `depth` is uniform, so `depth > k` never itself diverges.
static int __attribute__((noinline)) deep_nest(uint32_t tid, uint32_t depth) {
  int v = (int)tid;
#define LVL(k) if (depth > (k) && tid >= ((k) + 1u)) { v += (int)((k) + 1);
#define END }
  LVL(0) LVL(1) LVL(2)  LVL(3)  LVL(4)  LVL(5)  LVL(6)  LVL(7)
  LVL(8) LVL(9) LVL(10) LVL(11) LVL(12) LVL(13) LVL(14) LVL(15)
  END END END END END END END END END END END END END END END END
#undef LVL
#undef END
  return v;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  int32_t* dst_ptr = (int32_t*)arg->dst_addr;
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  dst_ptr[tid] = deep_nest(tid, arg->depth);
}
