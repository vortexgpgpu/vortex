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
#include "dxa_unit.h"

namespace vortex {

class Core;
class Cluster;

// Cluster-shared DXA engine. Aggregates DxaReq packets from per-core
// DxaUnits, dispatches to VX_CFG_NUM_DXA_CORES workers, and drives GMEM reads
// (through L2) and LMEM writes (per-core direct channels carrying real
// MemReq packets with TLM payload + completion flag).
class DxaCore : public SimObject<DxaCore> {
public:
  using Ptr = std::shared_ptr<DxaCore>;

  struct PerfStats {
    uint64_t transfers      = 0;
    uint64_t gmem_reads     = 0;
    uint64_t gmem_dedup     = 0;
    uint64_t lmem_writes    = 0;
    uint64_t total_latency  = 0;
    // DXA.STORE side (RFC 260904): full-line GMEM writes, LMEM word reads,
    // completed store transfers and their issue->ack latency sum.
    uint64_t gmem_writes    = 0;
    uint64_t lmem_reads     = 0;
    uint64_t store_transfers = 0;
    uint64_t store_latency  = 0;
    // Occupancy (RFC 260904 §5.6): cycles any worker was RUNNING (summed over
    // workers, so busy / (cycles * workers) is utilisation), and the time requests
    // sat in the queue before a worker took them, split by direction.
    uint64_t busy_cycles    = 0;
    uint64_t load_qwait     = 0;
    uint64_t store_qwait    = 0;

    PerfStats& operator+=(const PerfStats& rhs) {
      transfers     += rhs.transfers;
      gmem_reads    += rhs.gmem_reads;
      gmem_dedup    += rhs.gmem_dedup;
      lmem_writes   += rhs.lmem_writes;
      total_latency += rhs.total_latency;
      gmem_writes   += rhs.gmem_writes;
      lmem_reads    += rhs.lmem_reads;
      store_transfers += rhs.store_transfers;
      store_latency += rhs.store_latency;
      busy_cycles   += rhs.busy_cycles;
      load_qwait    += rhs.load_qwait;
      store_qwait   += rhs.store_qwait;
      return *this;
    }
  };

  // Per-core DxaReq inputs (size = NUM_CORES_PER_CLUSTER). Cluster binds
  // each core's DxaUnit::req_out here.
  std::vector<SimChannel<DxaReq>>  dxa_req_in;

  // GMEM ports to L2 (size = kDxaMemPorts). Internally fed by gmem_arb_
  // from VX_CFG_NUM_DXA_CORES worker outputs.
  std::vector<SimChannel<MemReq>>  gmem_req_out;
  std::vector<SimChannel<MemRsp>>  gmem_rsp_in;
  MemArbiter::Ptr                  gmem_arb_;

  // Per-core LMEM ports (size = NUM_CORES_PER_CLUSTER). Cluster binds each
  // core's LocalMem::Inputs[port_dxa] to lmem_req_out (loads write LMEM,
  // stores read it) and LocalMem::Outputs[port_dxa] to lmem_rsp_in (read data
  // for DXA.STORE; the load side never receives a response here).
  std::vector<SimChannel<MemReq>>  lmem_req_out;
  std::vector<SimChannel<MemRsp>>  lmem_rsp_in;

  DxaCore(const SimContext& ctx, const char* name, Cluster* cluster);
  virtual ~DxaCore();

  int dcr_write(uint32_t addr, uint32_t value);

  const PerfStats& perf_stats() const;

protected:
  void on_reset();
  void on_tick();

private:
  class Impl;
  Impl* impl_;

  friend class SimObject<DxaCore>;
};

} // namespace vortex
