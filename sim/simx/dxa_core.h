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
#include "arch.h"

namespace vortex {

class Core;
class Cluster;

// Decode the packed barrier-address word (meta[30:4]) back to a flat bar_id.
inline uint32_t dxa_decode_barrier_id(uint32_t bar_addr_raw, const Arch& arch) {
  uint32_t cta_no = bar_addr_raw & 0xffffu;
  uint32_t bar_no = (bar_addr_raw >> 16) & 0x7fffu;
  return (cta_no * arch.num_barriers() + bar_no) | (bar_addr_raw & 0x80000000u);
}

// Cycle-accurate DXA engine for simx, placed at Cluster scope matching RTL.
//
// Timing and emulation are fully decoupled:
//   submit()    — timing only; enqueues without touching memory.
//   gmem_read() — emulation only; reads GMEM with no timing side effects.
//   execute_copy() (private) — called once at GMEM→SMEM transition;
//                              writes all tile data to LocalMem RAM directly,
//                              before any timing signals go out on lmem_req_out.
//
// GMEM ports (gmem_req_out/gmem_rsp_in):
//   Count = kDxaMemPorts (= DXA_MEM_PORTS from config, ≤ NUM_DXA_UNITS).
//   An internal MemArbiter (TxRxArbiter<MemReq,MemRsp>) maps the NUM_DXA_UNITS
//   per-slice request channels onto kDxaMemPorts shared L2 ports; responses are
//   routed back to the originating slice via tag encoding in the arbiter.
//
// SMEM timing channel (lmem_req_out → LocalMem::dxa_req_in):
//   One DxaSmemReq (no data) per element per cycle; LocalMem stalls its
//   normal Inputs and releases the barrier on is_last.
class DxaCore : public SimObject<DxaCore> {
public:
  // Timing-only request from DxaCore → LocalMem DXA channel.
  // Carries no data — functional writes happen via execute_copy() before the
  // first timing signal is sent.  LocalMem stalls its Inputs for each received
  // request and releases the barrier on is_last.
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
    uint64_t smem_writes    = 0;
    uint64_t total_latency  = 0;

    PerfStats& operator+=(const PerfStats& rhs) {
      transfers     += rhs.transfers;
      gmem_reads    += rhs.gmem_reads;
      gmem_dedup    += rhs.gmem_dedup;
      smem_writes   += rhs.smem_writes;
      total_latency += rhs.total_latency;
      return *this;
    }
  };

  // Single trace packet built by execute() and consumed by tick().
  // Routing fields (desc_slot, smem_addr, coords, bar_id) are set at execute()
  // time; emulation fields (tile0..gmem_dedup_hits) are filled by execute_copy().
  struct TraceData : public ITraceData {
    using Ptr = std::shared_ptr<TraceData>;
    // routing
    uint32_t desc_slot = 0;
    uint64_t smem_addr = 0;
    uint32_t coords[5] = {};
    uint32_t bar_id    = 0;
    // emulation result
    uint32_t tile0      = 0;
    uint32_t tile1      = 0;
    uint32_t elem_bytes = 0;
    std::vector<uint64_t> gmem_lines;  // unique CL addrs (matches RTL addr_gen+dedup)
    std::vector<uint64_t> smem_blocks; // DXA_SMEM_WORD_SIZE-aligned SMEM addrs (matches RTL VX_dxa_wr_ctrl)
    uint64_t gmem_dedup_hits = 0;      // cross-row CL sharing (matches RTL VX_dxa_dedup)
  };

  // GMEM channels — size = kDxaMemPorts.  Bound to L2 DXA ports by Cluster.
  // Requests flow: arb_->ReqOut → gmem_req_out → L2.
  // Responses flow: L2 → gmem_rsp_in → arb_->RspIn → arb_->RspOut (per slice).
  std::vector<SimChannel<MemReq>>     gmem_req_out;
  std::vector<SimChannel<MemRsp>>     gmem_rsp_in;
  // Internal arbiter: NUM_DXA_UNITS slice inputs → kDxaMemPorts L2-facing outputs.
  MemArbiter::Ptr                     arb_;
  // SMEM timing channel — size = NUM_SOCKETS * SOCKET_SIZE (one per core).
  // Bound to each core's LocalMem::dxa_req_in by Cluster.
  std::vector<SimChannel<SmemReq>> lmem_req_out;

  DxaCore(const SimContext& ctx, const char* name, Cluster* cluster);
  virtual ~DxaCore();

  virtual void reset();
  virtual void tick();

  // Write a descriptor field via DCR.
  int dcr_write(uint32_t addr, uint32_t value);

  // Emulation only: read tile from GMEM into SMEM; fill TraceData emulation fields.
  // No SimChannel activity; no timing side effects.
  TraceData::Ptr execute_copy(Core* core,
                              uint32_t desc_slot,
                              uint64_t smem_addr,
                              const uint32_t coords[5]);

  // Timing only: enqueue the precomputed TraceData packet.
  // Returns false (backpressure) when the internal queue is full.
  bool submit(Core* core, TraceData::Ptr td);

  // Emulation only: read size bytes from global memory into data.
  // No SimChannel activity; no timing side effects.
  void gmem_read(Core* core, uint64_t addr, void* data, uint32_t size);

  const PerfStats& perf_stats() const;

private:
  class Impl;
  Impl* impl_;
};

} // namespace vortex
