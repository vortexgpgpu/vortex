#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <VX_types.h>
#include "common.h"

// A fence publishes every older store of its warp, whichever thread made it.
// The upper half of block 0's threads load from lines of one data-cache bank
// that miss, so that bank's misses back up, then store a line further: the
// last thread's store, the payload, waits on its lane. The lower half, on
// lanes with nothing ahead of them (a coalescing LSU merges neighbouring
// threads onto one data-cache lane), fences next, stores the guard,
// and evicts the guard's line from its data cache with loads to the same set,
// so the guard reaches the next level with no further fence. Block 1's first
// thread, on another core, waits for the guard and reads the payload: with a
// write-back data cache that the fence did not walk after the store, the
// payload never left the writer's cache.
static void lane_release(kernel_arg_t* arg) {
  uint32_t lane = threadIdx.x;
  auto base    = (volatile uint32_t*)arg->flood_addr;
  auto payload = base + LANE_PAYLOAD_WORD;
  auto guard   = base + LANE_GUARD_WORD;
  auto result  = (uint32_t*)arg->result_addr;
  if (blockIdx.x == 0) {
    if (lane >= LANE_THREADS / 2) {
      auto flood = base + lane * LANE_BANK_WORDS;
      #pragma unroll
      for (uint32_t k = 0; k < LANE_FLOOD_ROUNDS; ++k) {
        (void)flood[k * LANE_ROUND_WORDS];
      }
      flood[LANE_FLOOD_ROUNDS * LANE_ROUND_WORDS] = LANE_PAYLOAD;
    }
    // after the stores in program order: the paths of an if-else may run in
    // either order
    if (lane < LANE_THREADS / 2) {
      vx_fence();
      guard[0] = 1;
      // enough lines to fill the set and then replace every way of it
      for (uint32_t w = 1; w <= 2 * LANE_CACHE_WAYS; ++w) {
        (void)guard[w * LANE_WAY_WORDS];
      }
    }
  } else if (blockIdx.x == 1 && lane == 0) {
    uint32_t seen = 0;
    for (uint32_t i = 0; i < (1u << 12) && !seen; ++i) {
      vx_fence();
      seen = guard[0];
    }
    vx_fence();
    result[0] = seen;
    result[1] = payload[0];
  }
}

// A fence publishes the stores a data cache with several memory ports sent
// out on all of them. Block 0's threads store to lines that leave its data
// cache through the second memory port: first rounds of lines of the second
// next-level bank, which blocks 2 and up, on other cores, keep busy with
// loads that miss; then the payload, to the first next-level bank, which so
// queues behind those stores on its way. The block then
// fences, and its first thread sets the guard, on a line of the first port
// and bank, with an atomic. Block 1, on its own core, waits for the guard and
// reads the payload with atomics: its read finds the first bank free, so a
// fence that left through the first port only would let it in ahead of the
// payload.
static void port_release(kernel_arg_t* arg) {
  uint32_t lane = threadIdx.x;
  auto base   = (uint8_t*)arg->flood_addr;
  auto line   = [&](uint32_t port, uint32_t bank, uint32_t k) {
    return (uint32_t*)(base + PORT_LINE_OFFSET(port, bank, k));
  };
  auto payload = line(1, 0, 0);
  auto guard   = line(0, 0, 1);
  auto result  = (uint32_t*)arg->result_addr;
  if (blockIdx.x == 0) {
    // a first fence brings the fence's line into the next level; then let
    // the loads of the other blocks fill the second next-level bank
    vx_fence();
    uint32_t start = csr_read(VX_CSR_MCYCLE);
    while (csr_read(VX_CSR_MCYCLE) - start < PORT_WARMUP_CYCLES) {}
    #pragma unroll
    for (uint32_t j = 0; j < PORT_STORE_ROUNDS; ++j) {
      *(volatile uint32_t*)line(1, 1, j * PORT_THREADS + lane) = j;
    }
    if (lane == PORT_THREADS - 1) {
      *(volatile uint32_t*)payload = PORT_PAYLOAD;
    }
    vx_fence();
    if (lane == 0) {
      __atomic_exchange_n(guard, 1u, __ATOMIC_RELAXED);
    }
  } else if (blockIdx.x == 1) {
    if (lane == 0) {
      uint32_t seen = 0;
      for (uint32_t i = 0; i < (1u << 16) && !seen; ++i) {
        seen = __atomic_fetch_add(guard, 0u, __ATOMIC_RELAXED);
      }
      result[0] = seen;
      result[1] = __atomic_fetch_add(payload, 0u, __ATOMIC_RELAXED);
    }
  } else {
    uint32_t first = PORT_FLOOD_LINE + ((blockIdx.x - 2) * PORT_THREADS + lane) * PORT_FLOOD_ROUNDS;
    for (uint32_t i = 0; i < PORT_FLOOD_ROUNDS; ++i) {
      (void)*(volatile uint32_t*)line(1, 1, first + i);
    }
  }
}

// Message passing between mirrored tasks, which land on different cores: each
// round a task publishes its payload word with a plain store, fences, and
// releases its guard word with an atomic; it then reads its partner's guard
// with an atomic, fences, and reads the partner's payload with a plain load,
// which must hold the partner's value if the guard showed the round released. Before publishing, the task
// reads the partner's payload slot, so its data cache holds the line the
// partner is about to write: only a fence that reaches across the caches makes
// the second read see the partner's store. The check is branch-free, so the
// retired instructions do not depend on the interleaving. With plain_guard the
// guard is a plain word too, published and observed through the same fences.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  if (arg->mode == FENCE_MP_LANES) {
    lane_release(arg);
    return;
  }
  if (arg->mode == FENCE_MP_PORTS) {
    port_release(arg);
    return;
  }
  uint32_t n = arg->num_tasks;
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) {
    return;
  }
  uint32_t partner = n - 1 - gid;
  auto guard   = (uint32_t*)arg->guard_addr;
  auto payload = (volatile uint32_t*)arg->payload_addr;
  auto result  = (uint32_t*)arg->result_addr;

  uint32_t errors = 0, checked = 0;
  for (uint32_t r = 0; r < arg->rounds; ++r) {
    volatile uint32_t* slot = payload + r * n;
    (void)slot[partner];
    slot[gid] = payload_value(r, gid);
    vx_fence();
    uint32_t seen;
    if (arg->plain_guard) {
      ((volatile uint32_t*)guard)[gid] = r + 1;
      seen = ((volatile uint32_t*)guard)[partner];
    } else {
      __atomic_exchange_n(&guard[gid], r + 1, __ATOMIC_RELAXED);
      seen = __atomic_fetch_add(&guard[partner], 0, __ATOMIC_RELAXED);
    }
    vx_fence();
    uint32_t value = slot[partner];
    uint32_t released = (seen >= r + 1);
    checked += released;
    errors  += released & (value != payload_value(r, partner));
  }
  result[2 * gid + 0] = errors;
  result[2 * gid + 1] = checked;
}
