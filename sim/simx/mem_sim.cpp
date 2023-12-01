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

#include "mem_sim.h"
#include <vector>
#include <queue>
#include <stdlib.h>

DISABLE_WARNING_PUSH
DISABLE_WARNING_UNUSED_PARAMETER
#define RAMULATOR
#include <ramulator/src/Gem5Wrapper.h>
#include <ramulator/src/Request.h>
#include <ramulator/src/Statistics.h>
DISABLE_WARNING_POP

#include "constants.h"
#include "types.h"
#include "debug.h"

using namespace vortex;

class MemSim::Impl {
private:
    MemSim* simobject_;
    Config config_;
    PerfStats perf_stats_;
    ramulator::Gem5Wrapper* dram_;

public:

    Impl(MemSim* simobject, const Config& config) 
        : simobject_(simobject)
        , config_(config)
    {
        ramulator::Config ram_config;
        ram_config.add("standard", "DDR4");
        ram_config.add("channels", std::to_string(config.channels));
        ram_config.add("ranks", "1");
        ram_config.add("speed", "DDR4_2400R");
        ram_config.add("org", "DDR4_4Gb_x8");
        ram_config.add("mapping", "defaultmapping");
        ram_config.set_core_num(config.num_cores);
        dram_ = new ramulator::Gem5Wrapper(ram_config, MEM_BLOCK_SIZE);
        Stats::statlist.output("ramulator.ddr4.log");
    }

    ~Impl() {
        dram_->finish();
        Stats::statlist.printall();
        delete dram_;
    }

    const PerfStats& perf_stats() const {
        return perf_stats_;
    }

    void dram_callback(ramulator::Request& req, uint32_t tag, uint64_t uuid) {
        if (req.type == ramulator::Request::Type::WRITE)
            return;
        MemRsp mem_rsp{tag, (uint32_t)req.coreid, uuid};
        simobject_->MemRspPort.send(mem_rsp, 1);
        DT(3, simobject_->name() << "-" << mem_rsp);
    }

    void reset() {
        perf_stats_ = PerfStats();
    }

    void tick() {
        if (MEM_CYCLE_RATIO > 0) { 
            auto cycle = SimPlatform::instance().cycles();
            if ((cycle % MEM_CYCLE_RATIO) == 0)
                dram_->tick();
        } else {
            for (int i = MEM_CYCLE_RATIO; i <= 0; ++i)
                dram_->tick();            
        }
              
        if (simobject_->MemReqPort.empty())
            return;
        
        auto& mem_req = simobject_->MemReqPort.front();

        ramulator::Request dram_req( 
            mem_req.addr,
            mem_req.write ? ramulator::Request::Type::WRITE : ramulator::Request::Type::READ,
            std::bind(&Impl::dram_callback, this, placeholders::_1, mem_req.tag, mem_req.uuid),
            mem_req.cid
        );

        if (!dram_->send(dram_req))
            return;
        
        if (mem_req.write) {
            ++perf_stats_.writes;
        } else {
            ++perf_stats_.reads;
        }
        
        DT(3, simobject_->name() << "-" << mem_req);

        simobject_->MemReqPort.pop();        
    }
};

///////////////////////////////////////////////////////////////////////////////

MemSim::MemSim(const SimContext& ctx, const char* name, const Config& config) 
    : SimObject<MemSim>(ctx, name)
    , MemReqPort(this) 
    , MemRspPort(this)
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