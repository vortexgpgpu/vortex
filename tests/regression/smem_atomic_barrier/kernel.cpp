#include <vx_spawn2.h>
#include <vx_barrier.h>
#include <vx_intrinsics.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto results = reinterpret_cast<barrier_result_t*>(arg->results_addr);
  auto lmem = reinterpret_cast<uint32_t*>(__local_mem());
  auto state = reinterpret_cast<vortex::smem_barrier_state*>(lmem);
  auto payload = reinterpret_cast<volatile uint32_t*>(lmem + 16);

  vortex::barrier hard_bar(0, arg->num_warps);
  vortex::smem_barrier soft_bar(state, arg->num_warps);
  if (arg->mode == BARRIER_OVERHEAD_MODE_SOFT) {
    soft_bar.init();
  }
  __syncthreads();

  barrier_result_t local = {};
  uint32_t tid = threadIdx.x;
  uint32_t words = arg->payload_bytes / sizeof(uint32_t);

  for (uint32_t iter = 0; iter < arg->iterations; ++iter) {
    uint64_t t0 = vx_rdcycle_sync();
    if (arg->mode == BARRIER_OVERHEAD_MODE_SOFT) {
      soft_bar.expect_tx(1);
    }
    uint64_t t1 = vx_rdcycle_sync();

    for (uint32_t i = tid; i < words; i += blockDim.x) {
      payload[i] = payload[i] + i + iter + tid;
      local.checksum ^= payload[i];
    }

    uint64_t t2 = vx_rdcycle_sync();
    uint32_t spin = 0;
    if (arg->mode == BARRIER_OVERHEAD_MODE_SOFT) {
      soft_bar.complete_tx(1);
      uint32_t phase = soft_bar.arrive();
      spin = soft_bar.wait(phase);
    } else {
      uint32_t phase = hard_bar.arrive();
      hard_bar.wait(phase);
    }
    uint64_t t3 = vx_rdcycle_sync();

    local.register_cycles += static_cast<uint32_t>(t1 - t0);
    local.event_cycles += static_cast<uint32_t>(t2 - t0);
    local.release_cycles += static_cast<uint32_t>(t3 - t0);
    local.wait_iters += spin;
  }

  if (get_sub_group_id() == 0) {
    uint32_t active = (uint32_t)vx_active_threads();
    vx_tmc_one();
    if (arg->mode == BARRIER_OVERHEAD_MODE_SOFT) {
      local.observed_pending = static_cast<uint32_t>(state->events);
      local.observed_phase = state->phase;
      local.observed_arrived = state->arrived;
      if (state->events != 0)
        ++local.failures;
      if (state->phase != arg->iterations)
        ++local.failures;
      if (state->arrived != (arg->iterations * arg->num_warps))
        ++local.failures;
    } else {
      local.observed_pending = 0;
      local.observed_phase = arg->iterations;
      local.observed_arrived = arg->iterations * arg->num_warps;
    }
    results[0] = local;
    vx_tmc(active);
  }
}
