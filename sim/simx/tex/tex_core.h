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
#include "tex_unit.h"

namespace vortex {

class Cluster;

// Cluster-shared TEX engine. Mirrors the RTL VX_tex_core pipeline:
//   tex_arb (cluster) → VX_tex_core { dcr, tex_addr, tex_mem, tex_sampler }
// All texel reads flow through the tcache via MemReq/MemRsp; functional and
// timing meet at MemRsp arrival per simx_v3 §3.3.
class TexCore : public SimObject<TexCore> {
public:
  using Ptr = std::shared_ptr<TexCore>;

  struct PerfStats {
    uint64_t mem_reads     = 0;
    uint64_t mem_latency   = 0;
    uint64_t stall_cycles  = 0;

    PerfStats& operator+=(const PerfStats& rhs) {
      mem_reads    += rhs.mem_reads;
      mem_latency  += rhs.mem_latency;
      stall_cycles += rhs.stall_cycles;
      return *this;
    }
  };

  // Inputs from per-socket TexBus arbiter (cluster-level VX_tex_arb).
  // Size = NUM_TEX_CORES (after the cluster arb collapses NUM_SOCKETS → NUM_TEX_CORES).
  std::vector<SimChannel<TexReq>>  tex_req_in;
  std::vector<SimChannel<TexRsp>>  tex_rsp_out;

  // Memory ports to the tcache. Size = TCACHE_NUM_REQS.
  std::vector<SimChannel<MemReq>>  tcache_req_out;
  std::vector<SimChannel<MemRsp>>  tcache_rsp_in;

  TexCore(const SimContext& ctx, const char* name, Cluster* cluster);
  virtual ~TexCore();

  int dcr_write(uint32_t addr, uint32_t value);

  const PerfStats& perf_stats() const;

protected:
  void on_reset();
  void on_tick();

private:
  class Impl;
  Impl* impl_;

  friend class SimObject<TexCore>;
};

} // namespace vortex
