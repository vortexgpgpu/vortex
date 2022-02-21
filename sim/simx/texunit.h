#pragma once

#include <simobject.h>
#include <vector>
#include "pipeline.h"

namespace vortex {

class Core;

class TexUnit : public SimObject<TexUnit> {
public:
    struct Config {
        uint32_t address_latency;
        uint32_t sampler_latency;
    };

    struct TraceData : public vortex::TraceData {
        std::vector<std::vector<mem_addr_size_t>> mem_addrs;
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
    };

    SimPort<pipeline_trace_t*> Input;
    SimPort<pipeline_trace_t*> Output;

    TexUnit(const SimContext& ctx, const char* name, const Config& config, Core* core);    
    ~TexUnit();

    void reset();

    uint32_t csr_read(uint32_t addr);
  
    void csr_write(uint32_t addr, uint32_t value);

    uint32_t read(uint32_t stage, int32_t u, int32_t v, int32_t lod, TraceData* trace_data);
    
    void tick();

    const PerfStats& perf_stats() const;

private:

    class Impl;
    Impl* impl_;
};

}