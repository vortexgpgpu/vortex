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

typedef void (*vx_kernel_func_cb)(void *arg);

typedef void (*vx_serial_cb)(void *arg);

#define __local_mem(size) \
  (void*)((int8_t*)csr_read(VX_CSR_LOCAL_MEM_BASE) + __local_group_id * size)

#define __syncthreads() \
  vx_barrier(__local_group_id, __warps_per_group)

#define __syncthreads_arrive() \
  vx_barrier_arrive(__local_group_id, __warps_per_group)

#define __syncthreads_wait(token) \
  vx_barrier_wait(__local_group_id, token)

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
// Usage:
//   barrier bar;
//   bar.init(num_warps);
//   uint32_t token = bar.arrive();  // returns token (generation number)
//   bar.wait(token);                // wait until generation > token
class barrier {
public:
  // Constructor
  barrier() {
    bar_id_ = 0;
    num_warps_ = 0;
  }

  // Initialize barrier with expected warp count
  void init(uint32_t num_warps) {
    bar_id_ = __local_group_id;
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
