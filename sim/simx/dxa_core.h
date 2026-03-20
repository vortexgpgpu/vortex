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
#include <vector>
#include <cstdint>
#include <simobject.h>
#include <VX_types.h>

namespace vortex {

class Core;
class Cluster;

// Cycle-approximate DXA engine for simx, placed at Cluster scope matching RTL.
//
// Instantiates NUM_DXA_UNITS parallel DxaSlices that each independently
// process copy requests.  Requests arrive via submit() called from SfuUnit
// and are dispatched round-robin to idle slices.
//
// Timing model: countdown-based, derived from the g2s pipelined model
// (setup → tile_iter → rd_ctrl → wr_ctrl).  Each slice counts down modeled
// cycles, then executes the actual data copy and releases the barrier.
//
// TODO(phase-2): connect each DxaSlice to real L2 SimChannels so that
// GMEM reads experience actual CacheSim latency and MSHR contention.
class DxaCore : public SimObject<DxaCore> {
public:
  // Descriptor mirrors RTL VX_dxa_desc_t, written via DCR.
  struct Descriptor {
    uint64_t base_addr = 0;
    std::array<uint32_t, 5> sizes = {};
    std::array<uint32_t, 4> strides = {};
    uint32_t meta = 0;
    std::array<uint32_t, 5> element_strides = {};
    std::array<uint16_t, 5> tile_sizes = {};
    uint32_t cfill = 0;
  };

  struct PerfStats {
    uint64_t transfers = 0;
    uint64_t gmem_reads = 0;
    uint64_t smem_writes = 0;
    uint64_t total_latency = 0;

    PerfStats& operator+=(const PerfStats& rhs) {
      transfers    += rhs.transfers;
      gmem_reads   += rhs.gmem_reads;
      smem_writes  += rhs.smem_writes;
      total_latency += rhs.total_latency;
      return *this;
    }
  };

  DxaCore(const SimContext& ctx, const char* name, Cluster* cluster);
  virtual ~DxaCore() = default;

  virtual void reset();
  virtual void tick();

  // Write a descriptor field via DCR.
  int dcr_write(uint32_t addr, uint32_t value);

  // Called by SfuUnit (via core->socket()->cluster()->dxa_core()) to submit
  // a DXA copy request.  Returns false (backpressure) if the queue is full.
  bool submit(Core* core,
              uint32_t desc_slot,
              uint32_t smem_addr,
              const uint32_t coords[5],
              uint32_t bar_id);

  const PerfStats& perf_stats() const { return perf_stats_; }

private:
  // Decoded copy geometry, derived from a Descriptor.
  struct CopyCfg {
    uint32_t rank;
    uint32_t elem_bytes;
    uint32_t tile0;
    uint32_t tile1;
  };

  struct Request {
    Core*    core      = nullptr;
    uint32_t desc_slot = 0;
    uint32_t smem_addr = 0;
    uint32_t bar_id    = 0;
    std::array<uint32_t, 5> coords = {0, 0, 0, 0, 0};
  };

  struct ActiveTransfer {
    Request  req;
    uint32_t total_elems  = 0;
    uint32_t elem_bytes   = 0;
    uint32_t cycles_left  = 0;
    uint64_t issue_cycle  = 0;
  };

  // One DxaSlice per NUM_DXA_UNITS worker.
  struct DxaSlice {
    bool           has_active = false;
    ActiveTransfer active_xfer;
  };

  bool build_copy_cfg(const Descriptor& desc, CopyCfg* cfg) const;
  bool decode_request(const Request& req,
                      uint32_t* total_elems,
                      uint32_t* elem_bytes,
                      uint32_t* total_cycles,
                      uint64_t* gmem_reads_out,
                      uint64_t* smem_writes_out) const;
  bool execute_copy(const Request& req);

  bool start_slice(DxaSlice& slice);
  void tick_slice(DxaSlice& slice);

  Cluster* cluster_;
  std::array<Descriptor, VX_DCR_DXA_DESC_COUNT> descriptors_;
  std::deque<Request> queue_;
  static constexpr uint32_t kQueueDepth = 8;
  std::vector<DxaSlice> slices_;   // size = NUM_DXA_UNITS
  uint64_t cycle_;
  PerfStats perf_stats_;
};

} // namespace vortex

#endif // EXT_DXA_ENABLE
