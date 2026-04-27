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

#include <memory>
#include <simobject.h>
#include "types.h"
#include "instr_trace.h"

namespace vortex {

class Core;
class Cluster;

// cycle-accurate DXA engine; timing and emulation are decoupled
class DxaCore : public SimObject<DxaCore> {
public:
  // timing-only request to LocalMem DXA channel (no data payload)
  struct SmemReq {
    uint64_t addr;    // smem address (for bank-conflict modeling)
    uint32_t size;    // element size in bytes
    uint32_t bar_id;  // barrier to release when is_last
    Core*    core;    // core that owns the barrier
    bool     is_last; // triggers barrier_event_release in LocalMem::tick()
  };

  struct PerfStats {
    uint64_t transfers      = 0;
    uint64_t gmem_reads     = 0;
    uint64_t gmem_dedup     = 0;
    uint64_t lmem_writes    = 0;
    uint64_t total_latency  = 0;

    PerfStats& operator+=(const PerfStats& rhs) {
      transfers     += rhs.transfers;
      gmem_reads    += rhs.gmem_reads;
      gmem_dedup    += rhs.gmem_dedup;
      lmem_writes   += rhs.lmem_writes;
      total_latency += rhs.total_latency;
      return *this;
    }
  };

  // trace packet: routing fields set at execute(), emulation fields at execute_copy()
  struct TraceData : public ITraceData {
    using Ptr = std::shared_ptr<TraceData>;
    // inputs
    uint32_t desc_slot  = 0;
    uint64_t smem_addr  = 0;
    uint32_t coords[5]  = {};
    uint32_t bar_id     = 0;
    uint32_t cta_mask   = 0;
    // emulation result
    uint32_t tile0      = 0;
    uint32_t tile1      = 0;
    uint32_t elem_bytes = 0;
    std::vector<uint64_t> gmem_lines;  // unique CL addrs
    std::vector<uint64_t> smem_blocks; // DXA_SMEM_WORD_SIZE-aligned SMEM addrs
    uint64_t gmem_dedup_hits = 0;      // cross-row CL sharing
  };

  // GMEM ports bound to L2 DXA ports by Cluster (size = kDxaMemPorts)
  std::vector<SimChannel<MemReq>>     gmem_req_out;
  std::vector<SimChannel<MemRsp>>     gmem_rsp_in;
  MemArbiter::Ptr                     arb_;
  // SMEM timing channel, one per core (size = NUM_SOCKETS * SOCKET_SIZE)
  std::vector<SimChannel<SmemReq>> lmem_req_out;

  DxaCore(const SimContext& ctx, const char* name, Cluster* cluster);
  virtual ~DxaCore();

  // Write a descriptor field via DCR.
  int dcr_write(uint32_t addr, uint32_t value);

  // emulation: read tile from GMEM into SMEM, fill TraceData emulation fields
  TraceData::Ptr execute_copy(Core* core,
                              uint32_t desc_slot,
                              uint64_t smem_addr,
                              const uint32_t coords[5]);

  // timing: enqueue precomputed TraceData; returns false on backpressure
  bool submit(Core* core, TraceData::Ptr td);

  // emulation: read size bytes from global memory
  void gmem_read(Core* core, uint64_t addr, void* data, uint32_t size);

  const PerfStats& perf_stats() const;

protected:
  virtual void on_reset();
  virtual void on_tick();

private:
  class Impl;
  Impl* impl_;

  friend class SimObject<DxaCore>;
};

} // namespace vortex
