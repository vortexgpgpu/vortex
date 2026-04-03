#include <vx_spawn2.h>
#include <vx_barrier.h>
#include <stdint.h>
#include <math.h>
#include "common.h"

static inline uint32_t xor_0_to_n(uint32_t n) {
  switch (n & 3) {
  case 0: return n;
  case 1: return 1;
  case 2: return n + 1;
  default: return 0;
  }
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto A_ptr = reinterpret_cast<TYPE*>(arg->A_addr);
  auto B_ptr = reinterpret_cast<TYPE*>(arg->B_addr);
  auto C_ptr = reinterpret_cast<TYPE*>(arg->C_addr);

  // Lay out local memory: local_A0, local_B0, then local_u32 (for fingerprint)
  auto lmem      = reinterpret_cast<TYPE*>(__local_mem());
  auto local_A0  = lmem;
  auto local_B0  = local_A0 + blockDim.x * blockDim.y;
  auto local_u32 = reinterpret_cast<uint32_t*>(local_B0 + blockDim.x * blockDim.y);

  auto size      = arg->size;
  auto tile_size = arg->tile_size;

  auto g_row = blockIdx.x * blockDim.x + threadIdx.x;
  auto g_col = blockIdx.y * blockDim.y + threadIdx.y;

  auto l_row = threadIdx.x;
  auto l_col = threadIdx.y;

  TYPE sum(0);
  TYPE local_acc(0);

  vortex::barrier bar1(1), bar2(2);

  uint32_t token_compute;

  for (uint32_t k = 0; k < size; k += tile_size) {
    if (k > 0) {
      bar1.wait(token_compute);
    }
    local_A0[l_row * tile_size + l_col] = A_ptr[g_row * size + (k + l_col)];
    local_B0[l_row * tile_size + l_col] = B_ptr[(k + l_row) * size + g_col];

    uint32_t token_load = bar1.arrive();

    TYPE my_val   = local_A0[l_row * tile_size + l_col];
    TYPE my_val_b = local_B0[l_row * tile_size + l_col];

    for (int iter = 0; iter < tile_size; ++iter) {
      my_val   = sqrt(my_val   * my_val   + 0.01f);
      my_val_b = sqrt(my_val_b * my_val_b + 0.01f);
    }
    local_acc += my_val + my_val_b;

    bar1.wait(token_load);

    for (uint32_t j = 0; j < tile_size; ++j) {
      sum += local_A0[l_row * tile_size + j] * local_B0[j * tile_size + l_col];
    }
    token_compute = bar1.arrive();
  }

  // Barrier-2 correctness check
  auto threads_per_block = blockDim.x * blockDim.y;
  auto l_tid = threadIdx.y * blockDim.x + threadIdx.x;
  local_u32[l_tid] = l_tid;
  bar2.arrive_and_wait();

  if (0 == l_tid) {
    uint32_t fingerprint = 0;
    for (uint32_t i = 0; i < threads_per_block; ++i) {
      fingerprint ^= local_u32[i];
    }
    local_u32[threads_per_block] = fingerprint;
  }

  bar2.arrive_and_wait();
  uint32_t expected_fp = xor_0_to_n(threads_per_block - 1);
  if (local_u32[threads_per_block] != expected_fp) {
    sum += TYPE(1);
  }

  C_ptr[g_row * size + g_col] = sum + local_acc - local_acc;
}
