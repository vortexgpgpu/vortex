#include <vx_spawn2.h>
#include "common.h"

// Each thread multiplies its x by its y num_iters times, modulo a shared
// LIMBS-word modulus (Montgomery form). x, the modulus and the accumulator are
// about 3 x LIMBS live words, so the kernel is compute-bound and its cost is
// set by how much of that fits in registers.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t ni = arg->num_items;
  if (id >= ni)
    return;
  auto xp = reinterpret_cast<uint32_t*>(arg->x_addr) + id;
  auto yp = reinterpret_cast<const uint32_t*>(arg->y_addr) + id;
  auto np = reinterpret_cast<const uint32_t*>(arg->mod_addr);
  uint32_t x[LIMBS], n[LIMBS];
  BIG_UNROLL
  for (int j = 0; j < LIMBS; ++j) {
    x[j] = xp[j * ni];
    n[j] = np[j];
  }
  uint32_t n0inv = arg->n0inv;
  for (uint32_t it = 0; it < arg->num_iters; ++it)
    mont_mul(x, [&](int i) { return yp[i * ni]; }, n, n0inv);
  BIG_UNROLL
  for (int j = 0; j < LIMBS; ++j)
    xp[j * ni] = x[j];
}
