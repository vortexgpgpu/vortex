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

#ifdef EXT_DXA_ENABLE

#include <array>
#include <deque>
#include <cstdint>

namespace vortex {

class Core;

// Cycle-approximate DXA engine for simx.
//
// Models the next-gen AG→RRS→WBC pipeline timing:
//   - g2s: pipelined gmem reads with MAX_OUTSTANDING concurrent slots,
//     so the dominant cost is cache-line fill latency amortized across
//     the pipeline depth, plus per-element write-back.
//   - s2g: serial per-element (read smem, write gmem).
//
// The model counts down modeled cycles.  When the countdown expires it
// performs the actual data copy via core_->dxa_copy() and releases the
// barrier via core_->barrier_event_release().
class DxaEngine {
public:
  explicit DxaEngine(Core* core);

  void reset();

  bool issue(uint32_t desc_slot,
             uint32_t smem_addr,
             const uint32_t coords[5],
             uint32_t bar_id);

  void tick();

private:
  struct Request {
    uint32_t desc_slot = 0;
    uint32_t smem_addr = 0;
    uint32_t bar_id = 0;
    std::array<uint32_t, 5> coords = {0, 0, 0, 0, 0};
  };

  struct ActiveTransfer {
    Request req;
    uint32_t total_elems = 0;
    uint32_t elem_bytes = 0;
    uint32_t cycles_left = 0;
  };

  Core* core_;
  std::deque<Request> queue_;
  static constexpr uint32_t kQueueDepth = 8;

  bool has_active_;
  ActiveTransfer active_xfer_;

  bool start_next_request();
  bool decode_request(const Request& req, uint32_t* total_elems, uint32_t* elem_bytes, uint32_t* total_cycles) const;
  void progress_active_request();
  void complete_active_request();
};

} // namespace vortex

#endif // EXT_DXA_ENABLE
