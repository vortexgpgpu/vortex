#include "common.h"

// Atomic reduction: every work-item accumulates its element into a single
// global counter. Maximum-contention atomic_add, lowering to a hardware RVA
// amoadd.w on Vortex (requires the A extension).
__kernel void atomicreduce(__global const int* data,
                           __global int* result) {
  int gid = get_global_id(0);
  atomic_add(&result[0], data[gid]);
}
