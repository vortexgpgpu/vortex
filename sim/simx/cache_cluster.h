#pragma once

#include "cache_sim.h"

namespace vortex {

class CacheCluster : public SimObject<CacheCluster> {
public:
    std::vector<std::vector<SimPort<MemReq>>> CoreReqPorts;
    std::vector<std::vector<SimPort<MemRsp>>> CoreRspPorts;
    SimPort<MemReq> MemReqPort;
    SimPort<MemRsp> MemRspPort;

    CacheCluster(const SimContext& ctx, 
                 const char* name, 
                 uint32_t num_inputs, 
                 uint32_t num_caches, 
                 const CacheSim::Config& config) 
        : SimObject(ctx, name)        
        , CoreReqPorts(num_inputs, std::vector<SimPort<MemReq>>(config.num_inputs, this))
        , CoreRspPorts(num_inputs, std::vector<SimPort<MemRsp>>(config.num_inputs, this))
        , MemReqPort(this)
        , MemRspPort(this)
        , caches_(MAX(num_caches, 0x1)) {

        CacheSim::Config config2(config);
        if (0 == num_caches) {
            num_caches = 1;
            config2.bypass = true;
        }

        char sname[100];
        
        std::vector<Switch<MemReq, MemRsp>::Ptr> mem_arbs(config.num_inputs);   

        for (uint32_t i = 0; i < config.num_inputs; ++i) {
            snprintf(sname, 100, "%s-mem-arb%d", name, i);
            mem_arbs.at(i) = Switch<MemReq, MemRsp>::Create(sname, ArbiterType::RoundRobin, num_inputs, num_caches);

            for (uint32_t j = 0; j < num_inputs; ++j) {
                this->CoreReqPorts.at(j).at(i).bind(&mem_arbs.at(i)->ReqIn.at(j));
                mem_arbs.at(i)->RspIn.at(j).bind(&this->CoreRspPorts.at(j).at(i));
            }            
        }

        snprintf(sname, 100, "%s-cache-arb", name);
        auto cache_arb = Switch<MemReq, MemRsp>::Create(sname, ArbiterType::RoundRobin, num_caches, 1);

        for (uint32_t i = 0; i < num_caches; ++i) {
            snprintf(sname, 100, "%s-cache%d", name, i);
            caches_.at(i) = CacheSim::Create(sname, config2);

            for (uint32_t j = 0; j < config.num_inputs; ++j) {
                mem_arbs.at(j)->ReqOut.at(i).bind(&caches_.at(i)->CoreReqPorts.at(j));
                caches_.at(i)->CoreRspPorts.at(j).bind(&mem_arbs.at(j)->RspOut.at(i));
            }

            caches_.at(i)->MemReqPort.bind(&cache_arb->ReqIn.at(i));
            cache_arb->RspIn.at(i).bind(&caches_.at(i)->MemRspPort);
        }

        cache_arb->ReqOut.at(0).bind(&this->MemReqPort);
        this->MemRspPort.bind(&cache_arb->RspOut.at(0));
    }

    ~CacheCluster() {}

    void reset() {}
    
    void tick() {}

    CacheSim::PerfStats perf_stats() const {
        CacheSim::PerfStats perf;
        for (auto cache : caches_) {
            perf += cache->perf_stats();
        }   
        return perf;
    }
    
private:
    std::vector<CacheSim::Ptr> caches_;
};

}
