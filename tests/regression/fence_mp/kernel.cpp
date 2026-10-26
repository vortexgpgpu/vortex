#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// Message passing across cores through plain memory: the producer (the first
// thread) writes data, then raises a flag; the consumer (the last thread, on
// the last core) reads the data once beforehand so stale copies sit in its
// caches, polls the flag, and checks the data. The release must make the data
// visible across every cache the two cores do not share before the flag, and
// the acquire must drop the consumer's stale copies, whether they are full
// fences, acquire/release fences, or the .aq/.rl bits of atomics.
static void publish(volatile uint32_t* flag, uint32_t mode) {
  switch (mode) {
  case MP_ACQ_REL:
    __atomic_thread_fence(__ATOMIC_RELEASE);
    *flag = 1;
    break;
  case MP_AMO:
    __atomic_exchange_n(const_cast<uint32_t*>(flag), 1u, __ATOMIC_RELEASE);
    break;
  default:
    vx_fence();
    *flag = 1;
    vx_fence();
    break;
  }
}

static bool observe(volatile uint32_t* flag, uint32_t mode) {
  switch (mode) {
  case MP_ACQ_REL: {
    uint32_t f = *flag;
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    return f != 0;
  }
  case MP_AMO:
    return __atomic_fetch_or(const_cast<uint32_t*>(flag), 0u, __ATOMIC_ACQUIRE) != 0;
  default:
    vx_fence();
    return *flag != 0;
  }
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  auto data = reinterpret_cast<volatile uint32_t*>(arg->data_addr);
  auto flag = reinterpret_cast<volatile uint32_t*>(arg->flag_addr);
  auto out  = reinterpret_cast<volatile uint32_t*>(arg->out_addr);
  if (gid == 0) {
    for (uint32_t k = 0; k < arg->count; ++k) {
      data[k] = FENCE_MP_MAGIC + k;
    }
    publish(flag, arg->mode);
  } else if (gid == arg->num_tasks - 1) {
    uint32_t sum = 0;
    for (uint32_t k = 0; k < arg->count; ++k) {
      sum += data[k];
    }
    out[2] = sum;
    uint32_t timeout = 1;
    for (uint32_t s = 0; s < arg->spin_limit; ++s) {
      if (observe(flag, arg->mode)) {
        timeout = 0;
        break;
      }
    }
    if (arg->mode == MP_FENCE) {
      vx_fence();
    }
    uint32_t errors = 0;
    for (uint32_t k = 0; k < arg->count; ++k) {
      errors += (data[k] != FENCE_MP_MAGIC + k);
    }
    out[0] = errors;
    out[1] = timeout;
  }
}
