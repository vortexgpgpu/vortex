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

#include <simobject.h>
#include <VX_types.h>
#include "pipeline.h"
#include "graphics.h"
#include "types.h"

namespace vortex {

class RAM;

class OMUnit : public SimObject<OMUnit> {
public:
  struct PerfStats {
    uint64_t reads;
    uint64_t writes;
    uint64_t latency;
    uint64_t stalls;

    PerfStats()
      : reads(0)
      , writes(0)
      , latency(0)
      , stalls(0)
    {}

    PerfStats& operator+=(const PerfStats& rhs) {
      this->reads   += rhs.reads;
      this->writes  += rhs.writes;
      this->latency += rhs.latency;
      this->stalls  += rhs.stalls;
      return *this;
    }
  };

  struct TraceData : public ITraceData {
    using Ptr = std::shared_ptr<TraceData>;
    std::vector<uint64_t> mem_rd_addrs;
    std::vector<uint64_t> mem_wr_addrs;
    uint32_t om_idx;
  };

  using DCRS = graphics::OMDCRS;

  std::vector<SimPort<MemReq>> MemReqs;
  std::vector<SimPort<MemRsp>> MemRsps;

  SimPort<instr_trace_t*> Input;
  SimPort<instr_trace_t*> Output;

  OMUnit(const SimContext& ctx,
          const char* name,
          const Arch &arch,
          const DCRS& dcrs);

  ~OMUnit();

  void reset();

  void tick();

  void attach_ram(RAM* mem);

  void write(uint32_t x, uint32_t y, bool is_backface, uint32_t color, uint32_t depth,
             OMUnit::TraceData::Ptr trace_data);

  const PerfStats& perf_stats() const;

private:

  class Impl;
  Impl* impl_;
};

}