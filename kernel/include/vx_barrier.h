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

#include <vx_intrinsics.h>
#include <stdint.h>

namespace vortex {

class barrier {
public:
  barrier(uint32_t id, uint32_t num_warps = __warps_per_group) {
    bar_id_ = __local_group_id + (id << 16);
    num_warps_ = num_warps;
  }

  // Notify arrival at barrier (non-blocking)
  // Returns: phase (current generation number)
  uint32_t arrive() {
    return vx_barrier_arrive(bar_id_, num_warps_);
  }

  // Wait for barrier phase to complete
  // Blocks until generation > token
  void wait(uint32_t phase) {
    vx_barrier_wait(bar_id_, phase);
  }

  // Convenience: arrive and wait in one call
  void arrive_and_wait() {
    uint32_t phase = arrive();
    wait(phase);
  }

private:
  uint32_t bar_id_;
  uint32_t num_warps_;
};

}