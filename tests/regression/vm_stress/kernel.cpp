#include <vx_spawn2.h>
#include "common.h"

// Each task touches one word in each of its pages, with an odd page stride
// so consecutive touches spread across TLB sets and defeat streaming reuse.
// The physical buffer is identity-mapped (VX_MEM_PHYS) and exercises the
// pinned-slab translation path alongside the paged mappings.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto src_ptr  = reinterpret_cast<uint32_t*>(arg->src_addr);
  auto dst_ptr  = reinterpret_cast<uint32_t*>(arg->dst_addr);
  auto phys_ptr = reinterpret_cast<uint32_t*>(arg->phys_addr);

  uint32_t task_id = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t bias = phys_ptr[task_id % arg->phys_words];

  for (uint32_t k = 0; k < arg->pages_per_task; ++k) {
    uint32_t page = ((task_id * arg->pages_per_task + k) * arg->stride_pages) % arg->total_pages;
    uint32_t word = page * WORDS_PER_PAGE + (task_id % WORDS_PER_PAGE);
    dst_ptr[word] = src_ptr[word] + bias;
  }
}
