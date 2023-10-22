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
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>
#include "types.h"
#include "graphics.h"
#include "pipeline.h"

#define FIXEDPOINT_RASTERIZER

namespace vortex {

class RAM;

class RasterUnit : public SimObject<RasterUnit> {
public:

  struct Config {
    uint32_t tile_logsize;
    uint32_t block_logsize;
  };
  
  struct PerfStats {        
    uint64_t reads;
    uint64_t latency;
    uint64_t stalls;

    PerfStats() 
      : reads(0)
      , latency(0)
      , stalls(0)
    {}
    
    PerfStats& operator+=(const PerfStats& rhs) {
      this->reads   += rhs.reads;
      this->latency += rhs.latency;
      this->stalls  += rhs.stalls;
      return *this;
    }
  };

  struct TraceData : public ITraceData {
    using Ptr = std::shared_ptr<TraceData>;
    uint32_t raster_idx;
  };
  
  using DCRS = graphics::RasterDCRS;

  SimPort<MemReq> MemReqs;
  SimPort<MemRsp> MemRsps;

  SimPort<pipeline_trace_t*> Input;
  SimPort<pipeline_trace_t*> Output;
  
  RasterUnit(const SimContext& ctx, 
            const char* name,
            uint32_t raster_index,
            uint32_t raster_count,
            const Arch &arch, 
            const DCRS& dcrs,            
            const Config& config);    

  ~RasterUnit();

  void reset();

  void tick();

  uint32_t id() const;

  void attach_ram(RAM* mem);

  uint32_t fetch(uint32_t cid, uint32_t wid, uint32_t tid, CSRs& csrs);

  const PerfStats& perf_stats() const;

private:

  class Impl;
  Impl* impl_;
};

}