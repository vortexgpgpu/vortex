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
class Socket;

// Socket-local DXA engine. Aggregates DxaReq packets from the socket's cores,
// dispatches to VX_CFG_NUM_DXA_CORES workers, and drives GMEM reads through
// socket-local L2-facing ports plus LMEM writes to local cores. The worker
// count is independent of VX_CFG_SOCKET_SIZE.
class DxaCore : public SimObject<DxaCore> {
public:
  using Ptr = std::shared_ptr<DxaCore>;

  struct PerfStats {
    uint64_t transfers      = 0;
    uint64_t gmem_reads     = 0;
    uint64_t gmem_dedup     = 0;
    uint64_t lmem_writes    = 0;
    uint64_t total_latency  = 0;
#ifdef VX_CFG_EXT_DXA_S2G_ENABLE
    uint64_t s2g_transfers   = 0;
    uint64_t s2g_lmem_reads  = 0;
    uint64_t s2g_gmem_writes = 0;
    uint64_t s2g_read_events = 0;
#endif

    PerfStats& operator+=(const PerfStats& rhs) {
      transfers     += rhs.transfers;
      gmem_reads    += rhs.gmem_reads;
      gmem_dedup    += rhs.gmem_dedup;
      lmem_writes   += rhs.lmem_writes;
      total_latency += rhs.total_latency;
#ifdef VX_CFG_EXT_DXA_S2G_ENABLE
      s2g_transfers += rhs.s2g_transfers;
      s2g_lmem_reads += rhs.s2g_lmem_reads;
      s2g_gmem_writes += rhs.s2g_gmem_writes;
      s2g_read_events += rhs.s2g_read_events;
#endif
      return *this;
    }
  };

  // Per-core DxaReq inputs (size = VX_CFG_SOCKET_SIZE). Socket binds each
  // local core's DxaUnit::req_out here.
  std::vector<SimChannel<DxaReq>>  dxa_req_in;

  // GMEM ports to L2 (size = kDxaMemPorts). Internally fed by gmem_arb_
  // from VX_CFG_NUM_DXA_CORES worker outputs.
  std::vector<SimChannel<MemReq>>  gmem_req_out;
  std::vector<SimChannel<MemRsp>>  gmem_rsp_in;
  MemArbiter::Ptr                  gmem_arb_;

  // Per-core LMEM DMA ports (size = VX_CFG_SOCKET_SIZE). G2S writes through
  // these ports; S2G reads and receives the matching response below.
  std::vector<SimChannel<MemReq>>  lmem_req_out;
#ifdef VX_CFG_EXT_DXA_S2G_ENABLE
  std::vector<SimChannel<MemRsp>>  lmem_rsp_in;
  // Source completions route to the issuing core's per-warp group counters.
  std::vector<SimChannel<DxaReadCompletion>> completion_out;
#endif

  DxaCore(const SimContext& ctx, const char* name, Socket* socket);
  virtual ~DxaCore();

  int dcr_write(uint32_t addr, uint32_t value);

  bool running() const;

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
