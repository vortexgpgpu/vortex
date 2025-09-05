// mem_sim.cpp
#include "mem_sim.h"
#include <vector>
#include <stdlib.h>
#include <math.h>

#include "constants.h"
#include "types.h"
#include "debug.h"
#include "mem_backend.h"
#include "mem_backend_sst.h"
#include "mem_backend_dram.h"

using namespace vortex;

class MemSim::Impl {
private:
    MemSim*   simobject_;
    Config    config_;
    MemCrossBar::Ptr mem_xbar_;
    std::unique_ptr<IMemBackend> backend_;
    mutable PerfStats perf_stats_;

public:
    Impl(MemSim* simobject, const Config& config)
        : simobject_(simobject)
        , config_(config)
    {
        char sname[100];
        snprintf(sname, 100, "%s-xbar", simobject->name().c_str());
        mem_xbar_ = MemCrossBar::Create(
            sname,
            ArbiterType::RoundRobin,
            config.num_ports,
            config.num_banks,
            [lg2_block_size = log2ceil(config.block_size), num_banks = config.num_banks](const MemCrossBar::ReqType& req) {
                // Bank interleaving: choose the output index based on address bits
                return static_cast<uint32_t>((req.addr >> lg2_block_size) & (num_banks - 1));
            });

        for (uint32_t i = 0; i < config.num_ports; ++i) {
            simobject->MemReqPorts.at(i).bind(&mem_xbar_->ReqIn.at(i));
            mem_xbar_->RspIn.at(i).bind(&simobject->MemRspPorts.at(i));
        }

        #ifdef USE_SST_MEM_BACKEND
        backend_ = std::make_unique<MemBackendSST>();
        #else
        backend_ = std::make_unique<MemBackendDram>(config.num_banks, config.block_size, config.clock_ratio);
        #endif

        if (backend_) {
            backend_->mem_xbar_rsp_cb_ = [this](uint32_t bank, const MemRsp& rsp) {
                // Push the response into the appropriate crossbar output queue
                if (bank < mem_xbar_->RspOut.size())
                    mem_xbar_->RspOut.at(bank).push(rsp, 1);
            };
        }
    }

    const PerfStats& perf_stats() const {
        perf_stats_.bank_stalls = mem_xbar_->collisions();
        return perf_stats_;
    }

    void reset() {
        if (backend_)
            backend_->reset();
    }

    void tick() {
        // Advance the selected memory backend
        if (backend_)
            backend_->tick();

        // Drain requests from each bank and send to the backend
        for (uint32_t bank = 0; bank < config_.num_banks; ++bank) {
            if (mem_xbar_->ReqOut.at(bank).empty())
                continue;
            auto& mem_req = mem_xbar_->ReqOut.at(bank).front();
            if (backend_) {
                backend_->send_request(
                    mem_req.addr,
                    mem_req.write,
                    config_.block_size,
                    mem_req.tag,
                    mem_req.cid,
                    mem_req.uuid);
            }
            DT(3, simobject_->name() << "-mem-req" << bank << ": " << mem_req);
            mem_xbar_->ReqOut.at(bank).pop();
        }
    }
};

MemSim::MemSim(const SimContext& ctx, const char* name, const Config& config)
    : SimObject<MemSim>(ctx, name)
    , MemReqPorts(config.num_ports, this)
    , MemRspPorts(config.num_ports, this)
    , impl_(new Impl(this, config))
{}

MemSim::~MemSim() {
    delete impl_;
}

void MemSim::reset() {
    impl_->reset();
}

void MemSim::tick() {
    impl_->tick();
}

const MemSim::PerfStats& MemSim::perf_stats() const {
    return impl_->perf_stats();
}