#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid != 0) {
    return;
  }
  auto bad = reinterpret_cast<volatile uint32_t*>(arg->bad_addr);
  auto dst = reinterpret_cast<volatile uint32_t*>(arg->dst_addr);
  if (arg->mode == FAULT_MODE_READONLY) {
    // The page is mapped, so translation succeeds; the store must still be
    // refused because the leaf PTE grants no write access. Reading first
    // caches the translation, so the check has to hold against a TLB hit
    // and not only against a fresh walk.
    dst[0] = bad[0];
    bad[0] = 0xBADF00Du;
  } else {
    // No mapping at all: the walk must fault and terminate the launch
    // instead of returning garbage.
    dst[0] = bad[0];
  }
}
