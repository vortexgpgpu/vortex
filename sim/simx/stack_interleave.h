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

#pragma once

#include "types.h"
#include "constants.h"

namespace vortex {

// Thread t's stack is the STACK_SIZE bytes below STACK_TOP - t*STACK_SIZE.
// Within each group of NUM_THREADS stacks, offset {thread, word, byte} is
// stored as {word ^ group, thread, byte}, so one frame slot across a warp's
// threads is contiguous. The XOR keeps equal frame slots of different warps
// out of the same cache set, since a group spans a power of two. The map
// depends on the address alone, so any thread may dereference another
// thread's stack pointer.
class StackInterleave {
public:
#ifdef VX_CFG_LSU_STACK_INTERLEAVE_ENABLE
  static constexpr bool ENABLED = (VX_CFG_NUM_THREADS > 1);
#else
  static constexpr bool ENABLED = false;
#endif

  static constexpr uint64_t WORD_SIZE   = sizeof(Word);
  static constexpr uint64_t STACK_SIZE  = uint64_t(1) << VX_MEM_STACK_LOG2_SIZE;
  static constexpr uint64_t NUM_THREADS = VX_CFG_NUM_THREADS;
  static constexpr uint64_t NUM_STACKS  = uint64_t(VX_CFG_NUM_CLUSTERS) * NUM_SOCKETS * VX_CFG_SOCKET_SIZE
                                        * VX_CFG_NUM_WARPS * NUM_THREADS;
  static constexpr uint64_t TOP    = VX_MEM_STACK_BASE_ADDR;
  static constexpr uint64_t SPAN   = NUM_STACKS * STACK_SIZE;
  static constexpr uint64_t BOTTOM = TOP - SPAN;

  static_assert(ispow2(NUM_THREADS), "stack interleave needs a power-of-two thread count");
  static_assert(SPAN <= TOP, "stack window underflows the address space");

  static bool contains(uint64_t addr) {
    return ENABLED && addr >= BOTTOM && addr < TOP;
  }

  static uint64_t map(uint64_t addr) {
    if (!contains(addr)) {
      return addr;
    }
    uint64_t off    = addr - BOTTOM;
    uint64_t stack  = off / STACK_SIZE;
    uint64_t thread = stack % NUM_THREADS;
    uint64_t group  = stack / NUM_THREADS;
    uint64_t words  = STACK_SIZE / WORD_SIZE;
    uint64_t word   = ((off % STACK_SIZE) / WORD_SIZE) ^ (group % words);
    return BOTTOM + ((group * words + word) * NUM_THREADS + thread) * WORD_SIZE + (off % WORD_SIZE);
  }
};

}
