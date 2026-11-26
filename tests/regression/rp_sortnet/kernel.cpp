#include <vx_spawn2.h>
#include "common.h"

// Each thread sorts one row of KEYS keys with a bitonic network held in
// registers: KEYS loads and stores against about KEYS/4 * log2(KEYS)^2
// compare-exchanges, so the kernel is compute-bound and its cost is set by
// how many of the keys fit in registers.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t nr = arg->num_rows;
  if (id >= nr)
    return;
  auto p = reinterpret_cast<uint32_t*>(arg->keys_addr) + id;
  uint32_t a[KEYS];
  SORT_UNROLL
  for (int k = 0; k < KEYS; ++k)
    a[k] = p[k * nr];
  bitonic_sort(a);
  SORT_UNROLL
  for (int k = 0; k < KEYS; ++k)
    p[k * nr] = a[k];
}
