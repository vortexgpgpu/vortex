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

#include <cstdint>
#include <functional>
#include <unordered_map>
#include "VX_types.h"

namespace vortex {

struct kmu_req_t {
  uint64_t PC;
  uint64_t param;
  uint32_t cta_id;
  uint32_t block_idx[3];
  uint32_t block_dim[3];
  uint32_t grid_dim[3];
  uint32_t lmem_size;
  uint32_t block_size;
  uint32_t warp_step[3];
};

class Kmu {
public:
  using mem_reader_t = std::function<void(void*, uint64_t, uint32_t)>;

  Kmu();

  void reset();

  void dcr_write(uint32_t addr, uint32_t value);

  // Called by ProcessorImpl::run() to arm a kernel launch.
  void start();

  // Device-initiated re-arm (dynamic parallelism). Overrides the current KMU
  // state in one shot and starts dispatching CTAs. The caller must ensure the
  // KMU is idle (no CTAs left to dispatch) before invoking this.
  void arm_child(uint64_t pc,
                 uint64_t param,
                 const uint32_t grid_dim[3],
                 const uint32_t block_dim[3],
                 uint32_t block_size,
                 const uint32_t warp_step[3],
                 uint32_t lmem_size);

  // Attach the memory read path used when a device-side launch request arrives
  // from a given core.
  void attach_mem_reader(uint32_t core_id, const mem_reader_t& mem_read);

  // Device-initiated launch request through VX_CSR_KMU_LAUNCH. This models the
  // CSR write as a launch signal into the KMU, carrying the descriptor address
  // and source core rather than descriptor decoding logic.
  void request_child_launch(uint64_t desc_addr, uint32_t core_id);

  // True while CTAs remain to be issued.
  bool running() const { return running_; }

  // Called by CtaDispatcher when ready for the next CTA.
  // Fills *req with the next CTA's parameters and advances the iterator.
  // Returns false when the grid is exhausted.
  bool step(kmu_req_t* req);

private:
  uint64_t PC_;
  uint64_t param_;
  uint32_t block_dim_[3];
  uint32_t grid_dim_[3];
  uint32_t lmem_size_;
  uint32_t block_size_;
  uint32_t warp_step_[3];
  bool     running_;
  uint32_t cta_id_;
  uint32_t block_idx_[3];
  std::unordered_map<uint32_t, mem_reader_t> mem_readers_;

  void launch_child(uint64_t desc_addr, const mem_reader_t& mem_read);
};

} // namespace vortex
