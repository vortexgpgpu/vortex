#include "memsim.h"
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

    void dram_callback(ramulator::Request& req, uint32_t tag) {
        MemRsp mem_rsp{tag, (uint32_t)req.coreid};
        simobject_->MemRspPort.send(mem_rsp, 1);
    }

    void step(uint64_t /*cycle*/) {
        dram_->tick();
              
        if (simobject_->MemReqPort.empty())
            return;
        
        auto& mem_req = simobject_->MemReqPort.front();

        if (mem_req.write) {      
            ramulator::Request dram_req( 
                mem_req.addr,
                ramulator::Request::Type::WRITE,
                mem_req.core_id
            );
            dram_->send(dram_req);
            ++perf_stats_.writes;
        } else {
            ramulator::Request dram_req( 
                mem_req.addr,
                ramulator::Request::Type::READ,
                std::bind(&Impl::dram_callback, this, placeholders::_1, mem_req.tag),
                mem_req.core_id
            );
            dram_->send(dram_req);
            ++perf_stats_.reads;
        }

        simobject_->MemReqPort.pop();        
    }
};

///////////////////////////////////////////////////////////////////////////////

MemSim::MemSim(const SimContext& ctx, const Config& config) 
    : SimObject<MemSim>(ctx, "MemSim")
    , MemReqPort(this) 
    , MemRspPort(this)
    , impl_(new Impl(this, config))
{}

MemSim::~MemSim() {
    delete impl_;
}

void MemSim::step(uint64_t cycle) {
    impl_->step(cycle);
}