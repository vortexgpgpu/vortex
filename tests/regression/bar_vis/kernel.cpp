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

#include "common.h"
#include <vx_intrinsics.h>
#include <vx_spawn2.h>

// Every round each thread stores its round number, the CTA meets at a barrier,
// and each thread loads the word its lane stored in the neighbouring warp. Warp
// w issues on slice w % ISSUE_WIDTH, so with two LSU blocks that warp is on the
// other one. A load that sees an older round read past a store the barrier
// should have ordered before it.
// Stores to other lines ahead of the exchange (junk) keep the memory path busy
// when the barrier is reached.
// Mode 1 is the barrier exchange with the backlog on one side: only the even
// warps (one LSU block) store ahead of the exchange, 1 KiB apart from their
// exchange line so every store queues for the same data-cache bank, while the
// odd warps that read their words after the barrier find their own port idle.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  const uint32_t tid = threadIdx.x, T = blockDim.x;
  uint32_t* buf = arg->shared ? reinterpret_cast<uint32_t*>(__local_mem())
                              : reinterpret_cast<uint32_t*>(arg->buf_addr) + blockIdx.x * 2 * T;
  const uint32_t W = (uint32_t)vx_num_threads();
  const uint32_t peer = tid ^ W;
  const bool writer = ((tid / W) & 1u) == 0;
  auto junk = reinterpret_cast<volatile uint32_t*>(arg->junk_addr);
  uint32_t stale = 0;
  for (uint32_t r = 1; r <= arg->rounds; ++r) {
    if (arg->mode == 1) {
      if (writer) {
        // Past every CTA's exchange words, in whole KiB.
        const uintptr_t past = (gridDim.x * 2 * T * 4 + 1023u) & ~uintptr_t(1023);
        const uintptr_t line = reinterpret_cast<uintptr_t>(&buf[tid]) & ~uintptr_t(63);
        for (uint32_t j = 0; j < arg->junk; ++j) {
          *reinterpret_cast<volatile uint32_t*>(line + past + 1024u * j) = r + j;
        }
      }
    } else {
      for (uint32_t j = 0; j < arg->junk; ++j) {
        junk[((blockIdx.x * T + tid) * arg->junk + j) * 16u] = r + j;
      }
    }
    buf[tid] = r;
    __syncthreads();
    stale += (buf[peer] != r) ? 1u : 0u;
    __syncthreads();
  }
  reinterpret_cast<uint32_t*>(arg->err_addr)[blockIdx.x * T + tid] = stale;
}
