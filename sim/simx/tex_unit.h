#pragma once

#include <simobject.h>
#include <VX_types.h>
#include "pipeline.h"

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
    };

    class DCRS {
    private:
        std::array<std::array<uint32_t, (DCR_TEX_STATE_COUNT-1)>, TEX_STAGE_COUNT> states_;
        uint32_t stage_;

    public:
        uint32_t read(uint32_t stage, uint32_t addr) const {
            uint32_t state = DCR_TEX_STATE(addr-1);
            return states_.at(stage).at(state);
        }

        uint32_t read(uint32_t addr) const {
            if (addr == DCR_TEX_STAGE) {
                return stage_;
            }
            uint32_t state = DCR_TEX_STATE(addr-1);
            return states_.at(stage_).at(state);
        }
    
        void write(uint32_t addr, uint32_t value) {
            if (addr == DCR_TEX_STAGE) {
                stage_ = value;
                return;
            }
            uint32_t state = DCR_TEX_STATE(addr-1);
            states_.at(stage_).at(state) = value;
        }
    };
    
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
            uint32_t cores_per_unit,
            const Arch &arch, 
            const DCRS& dcrs,      
            const Config& config);

    ~TexUnit();

    void reset();

    void tick();

    void attach_ram(RAM* mem);

    uint32_t read(uint32_t stage, int32_t u, int32_t v, uint32_t lod, TraceData::Ptr trace_data);

    const PerfStats& perf_stats() const;

private:

    class Impl;
    Impl* impl_;
};

}