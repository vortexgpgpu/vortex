#include <vx_spawn2.h>
#include "common.h"

// Each thread applies Keccak-f[1600] num_perms times to its own state. The
// state is 50 words and a round needs about 30 more, so the kernel is
// compute-bound and its cost is set by how much of that fits in registers.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t ns = arg->num_states;
  if (id >= ns)
    return;
  auto st = reinterpret_cast<uint64_t*>(arg->state_addr) + id;
  uint64_t A[25];
  KECCAK_UNROLL
  for (int i = 0; i < 25; ++i)
    A[i] = st[i * ns];
  for (uint32_t p = 0; p < arg->num_perms; ++p)
    keccak_f(A);
  KECCAK_UNROLL
  for (int i = 0; i < 25; ++i)
    st[i * ns] = A[i];
}
