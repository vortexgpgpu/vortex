#pragma once

#include "constants.h"
#include "debug.h"
#include "types.h"
#include "core.h"

namespace vortex {

class Processor {
private:
  ArchDef arch_; 
  Decoder decoder_;
  MemoryUnit mu_;
  RAM ram_;
  std::vector<Core::Ptr> cores_;  
  std::vector<Cache::Ptr> l2caches_;  
  std::vector<Switch<MemReq, MemRsp>::Ptr> l2_mem_switches_;
  Cache::Ptr l3cache_;
  Switch<MemReq, MemRsp>::Ptr l3_mem_switch_;
  MemSim::Ptr memsim_;

public:
  Processor(const ArchDef& arch) 
    : arch_(arch)
    , decoder_(arch)
    , mu_(0, arch.wsize(), true)
    , ram_((1<<12), (1<<20)) 
    , cores_(arch.num_cores())
    , l2caches_(NUM_CLUSTERS)
    , l2_mem_switches_(NUM_CLUSTERS)
  {
    uint32_t num_cores = arch.num_cores();
    uint32_t cores_per_cluster = num_cores / NUM_CLUSTERS;
    
    // bind RAM to memory unit
    mu_.attach(ram_, 0, 0xFFFFFFFF);    

    // create cores
    for (uint32_t i = 0; i < num_cores; ++i) {
      cores_.at(i) = Core::Create(arch, decoder_, mu_, i);
    }
    
    // connect memory sub-systen
    memsim_ = MemSim::Create(1, MEM_LATENCY);
    std::vector<SlavePort<MemReq>*>  mem_req_ports(1); 
    std::vector<MasterPort<MemRsp>*> mem_rsp_ports(1);
    mem_req_ports.at(0) = &memsim_->MemReqPorts.at(0);
    mem_rsp_ports.at(0) = &memsim_->MemRspPorts.at(0);

    if (L3_ENABLE) {
      l3cache_ = Cache::Create("l3cache", CacheConfig{
        log2ceil(L3_CACHE_SIZE),  // C
        log2ceil(MEM_BLOCK_SIZE), // B
        2,                      // W
        0,                      // A
        32,                    // address bits    
        L3_NUM_BANKS,           // number of banks
        L3_NUM_PORTS,           // number of ports
        NUM_CLUSTERS,           // request size   
        true,                   // write-throught
        0,                      // victim size
        L3_MSHR_SIZE,           // mshr
        2,                      // pipeline latency
      });
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
      if (L2_ENABLE) {
        auto& l2cache = l2caches_.at(i);
        l2cache = Cache::Create("l2cache", CacheConfig{
          log2ceil(L2_CACHE_SIZE),  // C
          log2ceil(MEM_BLOCK_SIZE), // B
          2,                      // W
          0,                      // A
          32,                     // address bits    
          L2_NUM_BANKS,           // number of banks
          L2_NUM_PORTS,           // number of ports
          NUM_CORES,              // request size   
          true,                   // write-throught
          0,                      // victim size
          L2_MSHR_SIZE,           // mshr
          2,                      // pipeline latency
        });
        mem_rsp_ports.at(i)->bind(&l2cache->MemRspPort);
        l2cache->MemReqPort.bind(mem_req_ports.at(i));

        mem_req_ports.resize(cores_per_cluster);
        mem_rsp_ports.resize(cores_per_cluster);
        for (uint32_t j = 0; j < cores_per_cluster; ++j) {
          mem_req_ports.at(j) = &l2cache->CoreReqPorts.at(j);
          mem_rsp_ports.at(j) = &l2cache->CoreRspPorts.at(j);
        }
      } else if (cores_per_cluster > 1) {
        auto& l2_mem_switch = l2_mem_switches_.at(i);
        l2_mem_switch = Switch<MemReq, MemRsp>::Create("l2_arb", ArbiterType::RoundRobin, NUM_CORES);
        mem_rsp_ports.at(i)->bind(&l2_mem_switch->RspIn);
        l2_mem_switch->ReqOut.bind(mem_req_ports.at(i));  

        mem_req_ports.resize(cores_per_cluster);
        mem_rsp_ports.resize(cores_per_cluster);
        for (uint32_t j = 0; j < cores_per_cluster; ++j) {
          mem_req_ports.at(j) = &l2_mem_switch->ReqIn.at(j);
          mem_rsp_ports.at(j) = &l2_mem_switch->RspOut.at(j);
        }
      }

      for (uint32_t j = 0; j < cores_per_cluster; ++j) {
        auto& core = cores_.at((i * NUM_CLUSTERS) + j);        
        mem_rsp_ports.at(i)->bind(&core->MemRspPort);
        core->MemReqPort.bind(mem_req_ports.at(j));
      }
    }
  }

  ~Processor() {}

  int run(const std::string& program, bool riscv_test, bool /*showStats*/) {
    {
      std::string program_ext(fileExtension(program.c_str()));
      if (program_ext == "bin") {
        ram_.loadBinImage(program.c_str(), STARTUP_ADDR);
      } else if (program_ext == "hex") {
        ram_.loadHexImage(program.c_str());
      } else {
        std::cout << "*** error: only *.bin or *.hex images supported." << std::endl;
        return -1;
      }
    }

    bool running;
    int exitcode = 0;
    do {
      SimPlatform::instance().step();
      
      running = false;
      for (auto& core : cores_) {
        if (core->running()) {
          running = true;
        }
        if (core->check_ebreak()) {
          exitcode = core->getIRegValue(3);
          running = false;
          break;
        }
      }
    } while (running);

    // get error status

    if (riscv_test) {
      if (1 == exitcode) {
        std::cout << "Passed." << std::endl;
        exitcode = 0;
      } else {
        std::cout << "Failed." << std::endl;
      }
    } else {
      if (exitcode != 0) {
        std::cout << "*** error: exitcode=" << exitcode << std::endl;
      }
    }

    return exitcode;
  }

};

}