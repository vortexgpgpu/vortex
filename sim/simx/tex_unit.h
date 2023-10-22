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

namespace vortex {

class RAM;

class TexUnit : public SimObject<TexUnit> {
public:
    struct Config {
        uint32_t address_latency;
        uint32_t sampler_latency;
    };

    struct TraceData : public ITraceData {
        using Ptr = std::shared_ptr<TraceData>;
        std::vector<std::vector<mem_addr_size_t>> mem_addrs;
        uint32_t tex_idx;
        TraceData(uint32_t num_lanes) : mem_addrs(num_lanes) {
            for (uint32_t i = 0; i < num_lanes; ++i) {
                mem_addrs.at(i).reserve(4);
            }
        }
    };

    using DCRS = graphics::TexDCRS;

    struct PerfStats {
        uint64_t stalls;
        uint64_t reads;
        uint64_t latency;

        PerfStats() 
            : stalls(0)
            , reads(0)
            , latency(0)
        {}

        PerfStats& operator+=(const PerfStats& rhs) {
            this->reads   += rhs.reads;
            this->latency += rhs.latency;
            this->stalls  += rhs.stalls;
            return *this;
        }
    };

    std::vector<SimPort<MemReq>> MemReqs;
    std::vector<SimPort<MemRsp>> MemRsps;

    SimPort<pipeline_trace_t*> Input;
    SimPort<pipeline_trace_t*> Output;

    TexUnit(const SimContext& ctx,
            const char* name,
            const Arch &arch, 
            const DCRS& dcrs,      
            const Config& config);

    ~TexUnit();

    void reset();

    void tick();

    void attach_ram(RAM* mem);

    uint32_t read(uint32_t cid, uint32_t wid, uint32_t tid,
                  uint32_t stage, int32_t u, int32_t v, uint32_t lod, 
                  const CSRs& csrs, TraceData::Ptr trace_data);

    const PerfStats& perf_stats() const;

private:

    class Impl;
    Impl* impl_;
};

}