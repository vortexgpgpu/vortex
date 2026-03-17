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
// Models the line-granularity setup→tile_iter→rd_ctrl→wr_ctrl pipeline:
//   - g2s: pipelined GMEM line reads → width adaptation → SMEM word writes.
//     1 SMEM word/cycle at full bank bandwidth.
//   - s2g: serial per-element (read smem, write gmem).
//
// The model counts down modeled cycles.  When the countdown expires it
// performs the actual data copy and releases the barrier.
// Core interaction is limited to dcache_read/write + barrier_event_release.
class DxaEngine {
public:
  struct Descriptor {
    uint64_t base_addr;
    std::array<uint32_t, 5> sizes;
    std::array<uint32_t, 4> strides;
    uint32_t meta;
    std::array<uint32_t, 5> element_strides;
    std::array<uint16_t, 5> tile_sizes;
    uint32_t cfill;
  };

  explicit DxaEngine(Core* core);

  void reset();

  int dcr_write(uint32_t addr, uint32_t value);

  bool issue(uint32_t desc_slot,
             uint32_t smem_addr,
             const uint32_t coords[5],
             uint32_t bar_id);

  void tick();

private:
  // Decoded copy geometry, derived from a Descriptor.
  struct CopyCfg {
    uint32_t rank;
    uint32_t elem_bytes;
    uint32_t tile0;
    uint32_t tile1;
  };

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
  std::array<Descriptor, VX_DCR_DXA_DESC_COUNT> descriptors_;
  std::deque<Request> queue_;
  static constexpr uint32_t kQueueDepth = 8;

  bool has_active_;
  ActiveTransfer active_xfer_;

  const Descriptor& read_descriptor(uint32_t slot) const;
  bool build_copy_cfg(const Descriptor& desc, CopyCfg* cfg) const;
  bool estimate_transfer(uint32_t slot, uint32_t* total_elems, uint32_t* elem_bytes) const;
  bool execute_copy(uint32_t slot, uint32_t smem_addr, const uint32_t coords[5], uint32_t* bytes_copied);

  bool start_next_request();
  bool decode_request(const Request& req, uint32_t* total_elems, uint32_t* elem_bytes, uint32_t* total_cycles,
                      uint64_t* gmem_reads_out, uint64_t* smem_writes_out,
                      uint64_t* gmem_rsp_blk_out, uint64_t* gmem_req_blk_out, uint64_t* smem_wr_blk_out) const;
  void progress_active_request();
  void complete_active_request();
};

} // namespace vortex

#endif // EXT_DXA_ENABLE
