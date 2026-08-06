#include "common.h"

// Histogram: each work-item atomically increments the bin its element maps
// to. atomic_add on a __global int lowers to a hardware RVA amoadd.w on
// Vortex (requires the A extension).
__kernel void histogram(__global const int* data,
                        __global int* bins) {
  int gid = get_global_id(0);
  int bin = data[gid] % NUM_BINS;
  atomic_add(&bins[bin], 1);
}
