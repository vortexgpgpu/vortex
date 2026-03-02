#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"
#include <vx_barrier.h>

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
  vortex::gbarrier bar1(1), bar2(2);

  auto cid = vx_core_id();
  auto wid = vx_warp_id();
  auto tid = vx_thread_id();

  auto pre_ptr = reinterpret_cast<uint32_t*>(arg->pre_addr);
  auto post_ptr = reinterpret_cast<uint32_t*>(arg->post_addr);
  auto status_ptr = reinterpret_cast<uint32_t*>(arg->status_addr);

  bool leader = (wid == 0 && tid == 0);

  if (leader) {
    pre_ptr[cid] = cid + 1;
  }

  vx_fence();

  uint32_t phase = bar1.arrive();

  // Useful independent work that overlaps with barrier progress.
  // This demonstrates why split arrive/wait is valuable.
  uint32_t overlap_work = ((cid + 1) * 0x9e3779b9u) ^ ((wid + 1) * 0x85ebca6bu) ^ (tid + 1);
  for (uint32_t i = 0; i < 32; ++i) {
    overlap_work = overlap_work * 1664525u + 1013904223u;
    overlap_work ^= (overlap_work >> 13);
  }

  bar1.wait(phase);

  if (leader) {
    uint32_t sum = 0;
    for (uint32_t i = 0; i < arg->num_cores; ++i) {
      sum += pre_ptr[i];
    }
    // Consume overlap_work in a runtime-dependent way while keeping expected
    // output deterministic for normal execution.
    if (overlap_work == 0xffffffffu) {
      post_ptr[cid] = sum + 1;
    } else {
      post_ptr[cid] = sum;
    }
  }

  vx_fence();

  bar2.arrive_and_wait();

  if (cid == 0 && leader) {
    uint32_t expected = (arg->num_cores * (arg->num_cores + 1)) / 2;
    uint32_t errors = 0;

    for (uint32_t i = 0; i < arg->num_cores; ++i) {
      if (post_ptr[i] != expected) {
        ++errors;
      }
    }

    status_ptr[0] = errors;
  }
}

int main() {
  auto arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
  uint32_t grid_dim = {arg->num_groups};
  uint32_t block_dim = {arg->group_size};
  return vx_spawn_threads(1, &grid_dim, &block_dim, (vx_kernel_func_cb)kernel_body, arg);
}
