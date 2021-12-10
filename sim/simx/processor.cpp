#include "processor.h"
#include "core.h"
#include "constants.h"

using namespace vortex;

class Processor::Impl {
private:
  std::vector<Core::Ptr> cores_;
  std::vector<Cache::Ptr> l2caches_;
  std::vector<Switch<MemReq, MemRsp>::Ptr> l2_mem_switches_;
  Cache::Ptr l3cache_;
  Switch<MemReq, MemRsp>::Ptr l3_mem_switch_;

public:
  Impl(const ArchDef& arch) 
    : cores_(arch.num_cores())
    , l2caches_(NUM_CLUSTERS)
    , l2_mem_switches_(NUM_CLUSTERS)
  {
    SimPlatform::instance().initialize();

    uint32_t num_cores = arch.num_cores();
    uint32_t cores_per_cluster = num_cores / NUM_CLUSTERS;

    // create cores
    for (uint32_t i = 0; i < num_cores; ++i) {
        cores_.at(i) = Core::Create(arch, i);
    }

     // setup memory simulator
    auto memsim = MemSim::Create("dram", MemSim::Config{
      MEMORY_BANKS,
      arch.num_cores()
    });
    
    std::vector<SimPort<MemReq>*> mem_req_ports(1, &memsim->MemReqPort);
    std::vector<SimPort<MemRsp>*> mem_rsp_ports(1, &memsim->MemRspPort);

    if (L3_ENABLE) {
      l3cache_ = Cache::Create("l3cache", Cache::Config{
        log2ceil(L3_CACHE_SIZE),  // C
        log2ceil(MEM_BLOCK_SIZE), // B
        2,                      // W
        0,                      // A
        32,                     // address bits  
        L3_NUM_BANKS,           // number of banks
        L3_NUM_PORTS,           // number of ports
        NUM_CLUSTERS,           // request size 
        true,                   // write-through
        false,                  // write response
        0,                      // victim size
        L3_MSHR_SIZE,           // mshr
        2,                      // pipeline latency
        }
      );        
      l3cache_->MemReqPort.bind(mem_req_ports.at(0));
      mem_rsp_ports.at(0)->bind(&l3cache_->MemRspPort);

      mem_req_ports.resize(NUM_CLUSTERS);
      mem_rsp_ports.resize(NUM_CLUSTERS);

      for (uint32_t i = 0; i < NUM_CLUSTERS; ++i) {
        mem_req_ports.at(i) = &l3cache_->CoreReqPorts.at(i);
        mem_rsp_ports.at(i) = &l3cache_->CoreRspPorts.at(i);
      }
    } else if (NUM_CLUSTERS > 1) {
      l3_mem_switch_ = Switch<MemReq, MemRsp>::Create("l3_arb", ArbiterType::RoundRobin, NUM_CLUSTERS);
      l3_mem_switch_->ReqOut.bind(mem_req_ports.at(0));      
      mem_rsp_ports.at(0)->bind(&l3_mem_switch_->RspIn);

      mem_req_ports.resize(NUM_CLUSTERS);
      mem_rsp_ports.resize(NUM_CLUSTERS);

      for (uint32_t i = 0; i < NUM_CLUSTERS; ++i) {
        mem_req_ports.at(i) = &l3_mem_switch_->ReqIn.at(i);
        mem_rsp_ports.at(i) = &l3_mem_switch_->RspOut.at(i);
      }
    }

    for (uint32_t i = 0; i < NUM_CLUSTERS; ++i) {
      std::vector<SimPort<MemReq>*> cluster_mem_req_ports(cores_per_cluster); 
      std::vector<SimPort<MemRsp>*> cluster_mem_rsp_ports(cores_per_cluster);

      if (L2_ENABLE) {
        auto& l2cache = l2caches_.at(i);
        l2cache = Cache::Create("l2cache", Cache::Config{
          log2ceil(L2_CACHE_SIZE),  // C
          log2ceil(MEM_BLOCK_SIZE), // B
          2,                      // W
          0,                      // A
          32,                     // address bits  
          L2_NUM_BANKS,           // number of banks
          L2_NUM_PORTS,           // number of ports
          (uint8_t)cores_per_cluster, // request size 
          true,                   // write-through
          false,                  // write response
          0,                      // victim size
          L2_MSHR_SIZE,           // mshr
          2,                      // pipeline latency
        });
        l2cache->MemReqPort.bind(mem_req_ports.at(i));
        mem_rsp_ports.at(i)->bind(&l2cache->MemRspPort);

        for (uint32_t j = 0; j < cores_per_cluster; ++j) {
          cluster_mem_req_ports.at(j) = &l2cache->CoreReqPorts.at(j);
          cluster_mem_rsp_ports.at(j) = &l2cache->CoreRspPorts.at(j);
        }
      } else {
        auto& l2_mem_switch = l2_mem_switches_.at(i);
        l2_mem_switch = Switch<MemReq, MemRsp>::Create("l2_arb", ArbiterType::RoundRobin, cores_per_cluster);
        l2_mem_switch->ReqOut.bind(mem_req_ports.at(i));
        mem_rsp_ports.at(i)->bind(&l2_mem_switch->RspIn);

        for (uint32_t j = 0; j < cores_per_cluster; ++j) {
          cluster_mem_req_ports.at(j) = &l2_mem_switch->ReqIn.at(j);
          cluster_mem_rsp_ports.at(j) = &l2_mem_switch->RspOut.at(j);
        }
      }

      for (uint32_t j = 0; j < cores_per_cluster; ++j) {
        auto& core = cores_.at((i * cores_per_cluster) + j);
        core->MemReqPort.bind(cluster_mem_req_ports.at(j));
        cluster_mem_rsp_ports.at(j)->bind(&core->MemRspPort);
      }
    }
  }

  ~Impl() {
    SimPlatform::instance().finalize();
  }

  void attach_ram(RAM* ram) {
    for (auto core : cores_) {
      core->attach_ram(ram);
    }
  }

  int run() {
    SimPlatform::instance().reset();
    bool running;
    int exitcode = 0;
    do {
      SimPlatform::instance().tick();
      running = false;
      for (auto& core : cores_) {
        if (core->running()) {
          running = true;
        }
        if (core->check_exit()) {
          exitcode = core->getIRegValue(3);
          running = false;
          break;
        }
      }
    } while (running);

    return exitcode;
  }
};

///////////////////////////////////////////////////////////////////////////////

Processor::Processor(const ArchDef& arch) 
  : impl_(new Impl(arch))
{}

Processor::~Processor() {
  delete impl_;
}

void Processor::attach_ram(RAM* mem) {
  impl_->attach_ram(mem);
}

int Processor::run() {
  return impl_->run();
}