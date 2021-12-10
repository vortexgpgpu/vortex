#pragma once

#include <simobject.h>
#include <bitmanip.h>
#include <vector>
#include "types.h"

namespace vortex {

class Core;

class SharedMem : public SimObject<SharedMem> {
public:
    struct Config {
        uint32_t num_reqs;
        uint32_t num_banks; 
        uint32_t bank_offset;
        uint32_t latency;
        bool     write_reponse;
    };

    struct PerfStats {
        uint64_t reads;
        uint64_t writes;
        uint64_t bank_stalls;

        PerfStats() 
            : reads(0)
            , writes(0)
            , bank_stalls(0)
        {}
    };

    std::vector<SimPort<MemReq>> Inputs;
    std::vector<SimPort<MemRsp>> Outputs;

    SharedMem(const SimContext& ctx, const char* name, const Config& config) 
        : SimObject<SharedMem>(ctx, name)
        , Inputs(config.num_reqs, this)
        , Outputs(config.num_reqs, this)
        , config_(config)
        , bank_sel_addr_start_(config.bank_offset)
        , bank_sel_addr_end_(config.bank_offset + log2up(config.num_banks)-1)
    {}    
    
    virtual ~SharedMem() {}

    void reset() {
        perf_stats_ = PerfStats();
    }

    void tick() {
        std::vector<bool> in_used_banks(config_.num_banks);
        for (uint32_t req_id = 0; req_id < config_.num_reqs; ++req_id) {
            auto& core_req_port = this->Inputs.at(req_id);            
            if (core_req_port.empty())
                continue;

            auto& core_req = core_req_port.front();

            uint32_t bank_id = (uint32_t)bit_getw(
                core_req.addr, bank_sel_addr_start_, bank_sel_addr_end_);

            // bank conflict check
            if (in_used_banks.at(bank_id))
                continue;

            in_used_banks.at(bank_id) = true;

            if (!core_req.write || config_.write_reponse) {
                // send response
                MemRsp core_rsp{core_req.tag, core_req.core_id};
                this->Outputs.at(req_id).send(core_rsp, 1);
            }

            // update perf counters
            perf_stats_.reads += !core_req.write;            
            perf_stats_.writes += core_req.write;

            // remove input
            core_req_port.pop();
        }
    }

    const PerfStats& perf_stats() const { 
        return perf_stats_; 
    }

protected:
    Config    config_;
    uint32_t  bank_sel_addr_start_;
    uint32_t  bank_sel_addr_end_;
    PerfStats perf_stats_;
};

}