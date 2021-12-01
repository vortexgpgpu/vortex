#include "processor.h"
#include "constants.h"

using namespace vortex;

Processor::Processor(const ArchDef& arch) 
  : cores_(arch.num_cores())
  , l2caches_(NUM_CLUSTERS)
  , l2_mem_switches_(NUM_CLUSTERS)
{
  uint32_t num_cores = arch.num_cores();
  uint32_t cores_per_cluster = num_cores / NUM_CLUSTERS; 

  // create cores
  for (uint32_t i = 0; i < num_cores; ++i) {
      cores_.at(i) = Core::Create(arch, i);
  }

  // connect memory sub-systen
  memsim_ = MemSim::Create(1, MEM_LATENCY);
  std::vector<SimPort<MemReq>*> mem_req_ports(1); 
  std::vector<SimPort<MemRsp>*> mem_rsp_ports(1);

  mem_req_ports.at(0) = &memsim_->MemReqPorts.at(0);
  mem_rsp_ports.at(0) = &memsim_->MemRspPorts.at(0);

  if (L3_ENABLE) {
    l3cache_ = Cache::Create("l3cache", Cache::Config{
      log2ceil(L3_CACHE_SIZE),  // C
      log2ceil(MEM_BLOCK_SIZE), // B
      2,                      // W
      0,                      // A
      32,                    // address bits    
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
      
    mem_rsp_ports.at(0)->bind(&l3cache_->MemRspPort);
    l3cache_->MemReqPort.bind(mem_req_ports.at(0));

    mem_req_ports.resize(NUM_CLUSTERS);
    mem_rsp_ports.resize(NUM_CLUSTERS);

    for (uint32_t i = 0; i < NUM_CLUSTERS; ++i) {
      mem_req_ports.at(i) = &l3cache_->CoreReqPorts.at(i);
      mem_rsp_ports.at(i) = &l3cache_->CoreRspPorts.at(i);
    }
  } else if (NUM_CLUSTERS > 1) {
    l3_mem_switch_ = Switch<MemReq, MemRsp>::Create("l3_arb", ArbiterType::RoundRobin, NUM_CLUSTERS);
    mem_rsp_ports.at(0)->bind(&l3_mem_switch_->RspIn);
    l3_mem_switch_->ReqOut.bind(mem_req_ports.at(0));      

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

      mem_rsp_ports.at(i)->bind(&l2cache->MemRspPort);
      l2cache->MemReqPort.bind(mem_req_ports.at(i));

      for (uint32_t j = 0; j < cores_per_cluster; ++j) {
        cluster_mem_req_ports.at(j) = &l2cache->CoreReqPorts.at(j);
        cluster_mem_rsp_ports.at(j) = &l2cache->CoreRspPorts.at(j);
      }
    } else {
      auto& l2_mem_switch = l2_mem_switches_.at(i);
      l2_mem_switch = Switch<MemReq, MemRsp>::Create("l2_arb", ArbiterType::RoundRobin, cores_per_cluster);

      mem_rsp_ports.at(i)->bind(&l2_mem_switch->RspIn);
      l2_mem_switch->ReqOut.bind(mem_req_ports.at(i));

      for (uint32_t j = 0; j < cores_per_cluster; ++j) {
        cluster_mem_req_ports.at(j) = &l2_mem_switch->ReqIn.at(j);
        cluster_mem_rsp_ports.at(j) = &l2_mem_switch->RspOut.at(j);
      }
    }

    for (uint32_t j = 0; j < cores_per_cluster; ++j) {
      auto& core = cores_.at((i * cores_per_cluster) + j);        
      cluster_mem_rsp_ports.at(j)->bind(&core->MemRspPort);
      core->MemReqPort.bind(cluster_mem_req_ports.at(j));
    }
  }
}

void Processor::attach_ram(RAM* ram) {
  for (auto core : cores_) {
    core->attach_ram(ram);
  }
}

Processor::~Processor() {}

int Processor::run() {
  bool running;
  int exitcode = 0;
  do {
    SimPlatform::instance().step();
    
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

  std::cout << std::flush;

  return exitcode;
}