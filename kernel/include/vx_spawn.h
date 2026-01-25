// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __VX_SPAWN_H__
#define __VX_SPAWN_H__

#include <vx_intrinsics.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef union {
  struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
  };
  uint32_t m[3];
} dim3_t;

extern __thread dim3_t blockIdx;
extern __thread dim3_t threadIdx;
extern dim3_t gridDim;
extern dim3_t blockDim;

extern __thread uint32_t __local_group_id;
extern uint32_t __warps_per_group;

static inline uint32_t __vx_local_group_id() {
  if (__warps_per_group != 0) {
    return (uint32_t)(vx_warp_id() / (int)__warps_per_group);
  }
  return 0;
}

typedef void (*vx_kernel_func_cb)(void *arg);

typedef void (*vx_serial_cb)(void *arg);

#define __local_mem(size) \
  (void*)((int8_t*)csr_read(VX_CSR_LOCAL_MEM_BASE) + __vx_local_group_id() * size)

// Barrier ID mapping:
// - Use an even barrier ID for CTA-wide sync barriers (__syncthreads / __sync_threads)
// - Use an odd  barrier ID for user async barriers (barrier class, __asyncthreads_*)
//
// This avoids interference between sync and async barrier phases when a CTA slot
// (i.e., __local_group_id) is reused across multiple logical CTAs.
#define __sync_barrier_id() \
  ((__warps_per_group > 1) ? (uint32_t)(__vx_local_group_id() << 1) : (uint32_t)__vx_local_group_id())

#define __async_barrier_id() \
  ((__warps_per_group > 1) ? (uint32_t)((__vx_local_group_id() << 1) | 1u) : (uint32_t)__vx_local_group_id())

// CTA-level sync barrier (__syncthreads) implemented using async barrier arrive+wait.
// This keeps the ISA surface minimal (ARRIVE+WAIT) while hiding the token at the API level.
#define __sync_threads() do { \
  if (__warps_per_group > 1) { \
    uint32_t __bar_id = __sync_barrier_id(); \
    uint32_t __token = vx_barrier_arrive(__bar_id, __warps_per_group); \
    vx_barrier_wait(__bar_id, __token); \
  } \
} while (0)

#define __syncthreads() __sync_threads()

#define __asyncthreads_arrive() \
  vx_barrier_arrive(__async_barrier_id(), __warps_per_group)

#define __asyncthreads_wait(token) \
  vx_barrier_wait(__async_barrier_id(), token)

// launch a kernel function with a grid of blocks and block of threads
int vx_spawn_threads(uint32_t dimension,
                     const uint32_t* grid_dim,
                     const uint32_t* block_dim,
                     vx_kernel_func_cb kernel_func,
                     const void* arg);

// function call serialization
void vx_serial(vx_serial_cb callback, const void * arg);

#ifdef __cplusplus
// }
// #endif

}  // extern "C"

//////////////////////////////////////////////////////////////////////////////
// Simple CTA-level async barrier class
//////////////////////////////////////////////////////////////////////////////

// CTA-level async barrier
class barrier {
public:
  // Constructor
  barrier() {
    bar_id_ = 0;
    num_warps_ = 0;
  }

  // Initialize barrier with expected warp count
  void init(uint32_t num_warps) {
    bar_id_ = (num_warps > 1) ? (uint32_t)((__vx_local_group_id() << 1) | 1u) : (uint32_t)__vx_local_group_id();
    num_warps_ = num_warps;
  }

  // Arrive at barrier (non-blocking)
  // Returns: token (current generation number)
  uint32_t arrive() {
    return vx_barrier_arrive(bar_id_, num_warps_);
  }

  // Wait for barrier phase to complete
  // Blocks until generation > token
  void wait(uint32_t token) {
    vx_barrier_wait(bar_id_, token);
  }

  // Convenience: arrive and wait in one call
  void arrive_and_wait() {
    uint32_t token = arrive();
    wait(token);
  }

private:
  uint32_t bar_id_;
  uint32_t num_warps_;
};

#endif  // __cplusplus

#endif // __VX_SPAWN_H__
