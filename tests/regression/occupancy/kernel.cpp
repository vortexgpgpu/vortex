#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  if (threadIdx.x != 0)
    return;

  uint32_t cta_id    = blockIdx.x;
  uint32_t lmem_words = arg->lmem_words;
  auto lmem = reinterpret_cast<uint32_t*>(__local_mem());

  for (uint32_t i = 0; i < lmem_words; i++) {
    lmem[i] = cta_id + i;
  }

  uint32_t sum = 0;
  for (uint32_t i = 0; i < lmem_words; i++) {
    sum += lmem[i];
  }

  auto out = reinterpret_cast<uint32_t*>(arg->out_addr);
  out[cta_id] = sum;
}
