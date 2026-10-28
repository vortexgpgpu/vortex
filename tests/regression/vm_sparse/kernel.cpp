#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

#define WORDS_PER_PAGE (4096u / sizeof(uint32_t))

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  if (arg->count_addr) {
    __atomic_fetch_add(reinterpret_cast<uint32_t*>(arg->count_addr), 1u, __ATOMIC_RELAXED);
    return;
  }
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= arg->num_threads) {
    return;
  }
  auto sparse = reinterpret_cast<uint32_t*>(arg->sparse_addr);
  auto reads = reinterpret_cast<volatile uint32_t*>(arg->read_addr);
  auto amos = reinterpret_cast<volatile uint32_t*>(arg->amo_addr);
  uint32_t page = tid / SLOTS;
  uint32_t slot = tid % SLOTS;
  uint32_t* w = sparse + page * WORDS_PER_PAGE + slot * SLOT_STRIDE;
  reads[tid] = *reinterpret_cast<volatile uint32_t*>(w);
  *reinterpret_cast<volatile uint32_t*>(w + 1) = arg->tag | tid;
  amos[tid] = __atomic_fetch_add(w + 2, 1u, __ATOMIC_RELAXED);
}
