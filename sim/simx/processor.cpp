#include "processor.h"
#include "core.h"
#include "constants.h"

using namespace vortex;

class Processor::Impl {
private:
  std::vector<Core::Ptr> cores_;
  std::vector<RasterUnit::Ptr> raster_units_;
  std::vector<RopUnit::Ptr> rop_units_;
  DCRS dcrs_;

public:
  Impl(const Arch& arch) 
    : cores_(arch.num_cores())
    , raster_units_(NUM_CLUSTERS)
    , rop_units_(NUM_CLUSTERS)
  {
    SimPlatform::instance().initialize();

    uint32_t num_cores = arch.num_cores();
    uint32_t cores_per_cluster = num_cores / NUM_CLUSTERS;
    
    std::vector<CacheSim::Ptr> raster_caches(NUM_CLUSTERS);
    std::vector<CacheSim::Ptr> rop_caches(NUM_CLUSTERS);

    // create raster blocks
    for (uint32_t i = 0; i < NUM_CLUSTERS; ++i) {
      raster_units_.at(i) = RasterUnit::Create("raster_unit", arch, dcrs_.raster_dcrs, i, RASTER_TILE_LOGSIZE, RASTER_BLOCK_LOGSIZE);
      raster_caches.at(i) = CacheSim::Create("raster_cache", CacheSim::Config{
        log2ceil(RCACHE_SIZE),  // C
        log2ceil(MEM_BLOCK_SIZE), // B
        log2ceil(sizeof(uint32_t)), // W
        log2ceil(RCACHE_NUM_WAYS), // A
        32,                     // address bits    
        RCACHE_NUM_BANKS,       // number of banks
        RCACHE_NUM_PORTS,       // number of ports
        1,                      // number of requests 
        false,                  // write-through
        false,                  // write response
        0,                      // victim size
        RCACHE_MSHR_SIZE,       // mshr
        4,                      // pipeline latency
      });
      // connect cache
      raster_units_.at(i)->MemReqs.bind(&raster_caches.at(i)->CoreReqPorts.at(0));
      raster_caches.at(i)->CoreRspPorts.at(0).bind(&raster_units_.at(i)->MemRsps);
    }

    // create rop blocks
    for (uint32_t i = 0; i < NUM_CLUSTERS; ++i) {
      rop_units_.at(i) = RopUnit::Create("rop_unit", arch, dcrs_.rop_dcrs);
      rop_caches.at(i) = CacheSim::Create("rop_cache", CacheSim::Config{
        log2ceil(OCACHE_SIZE),  // C
        log2ceil(MEM_BLOCK_SIZE), // B
        log2ceil(sizeof(uint32_t)), // W
        log2ceil(OCACHE_NUM_WAYS), // A 
        32,                     // address bits    
        OCACHE_NUM_BANKS,       // number of banks
        OCACHE_NUM_PORTS,       // number of ports
        (uint8_t)arch.num_threads(), // number of requests
        false,                  // write-through
        false,                  // write response
        0,                      // victim size
        OCACHE_MSHR_SIZE,       // mshr
        4,                      // pipeline latency
      });
      // connect cache
      for (uint32_t j = 0; j < arch.num_threads(); ++j) {
        rop_units_.at(i)->MemReqs.at(j).bind(&rop_caches.at(i)->CoreReqPorts.at(j));
        rop_caches.at(i)->CoreRspPorts.at(j).bind(&rop_units_.at(i)->MemRsps.at(j));
      }
    }

    // create cores
    for (uint32_t i = 0; i < num_cores; ++i) {
      auto j = i / cores_per_cluster;
      cores_.at(i) = Core::Create(i, arch, dcrs_, raster_units_.at(j), rop_units_.at(j), raster_caches.at(j), rop_caches.at(j));
    }

     // setup memory simulator
    auto memsim = MemSim::Create("dram", MemSim::Config{
      MEMORY_BANKS,
      arch.num_cores()
    });
    
    std::vector<SimPort<MemReq>*> mem_req_ports(1, &memsim->MemReqPort);
    std::vector<SimPort<MemRsp>*> mem_rsp_ports(1, &memsim->MemRspPort);

    if (L3_ENABLED) {
      auto l3cache = CacheSim::Create("l3cache", CacheSim::Config{
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
      
      l3cache->MemReqPort.bind(mem_req_ports.at(0));
      mem_rsp_ports.at(0)->bind(&l3cache->MemRspPort);

      mem_req_ports.resize(NUM_CLUSTERS);
      mem_rsp_ports.resize(NUM_CLUSTERS);

      for (uint32_t i = 0; i < NUM_CLUSTERS; ++i) {
        mem_req_ports.at(i) = &l3cache->CoreReqPorts.at(i);
        mem_rsp_ports.at(i) = &l3cache->CoreRspPorts.at(i);
      }
    } else if (NUM_CLUSTERS > 1) {
      auto l3_mem_switch = Switch<MemReq, MemRsp>::Create("l3_arb", ArbiterType::RoundRobin, NUM_CLUSTERS);
      l3_mem_switch->ReqOut.bind(mem_req_ports.at(0));      
      mem_rsp_ports.at(0)->bind(&l3_mem_switch->RspIn);

      mem_req_ports.resize(NUM_CLUSTERS);
      mem_rsp_ports.resize(NUM_CLUSTERS);

      for (uint32_t i = 0; i < NUM_CLUSTERS; ++i) {
        mem_req_ports.at(i) = &l3_mem_switch->ReqIn.at(i);
        mem_rsp_ports.at(i) = &l3_mem_switch->RspOut.at(i);
      }
    }

    for (uint32_t i = 0; i < NUM_CLUSTERS; ++i) {
      std::vector<SimPort<MemReq>*> cluster_mem_req_ports(cores_per_cluster); 
      std::vector<SimPort<MemRsp>*> cluster_mem_rsp_ports(cores_per_cluster);

      auto gpu_switch = Switch<MemReq, MemRsp>::Create("gpu_switch", ArbiterType::RoundRobin, 3);
      gpu_switch->ReqOut.bind(mem_req_ports.at(i));
      mem_rsp_ports.at(i)->bind(&gpu_switch->RspIn);

      if (L2_ENABLED) {
        auto l2cache = CacheSim::Create("l2cache", CacheSim::Config{
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
        l2cache->MemReqPort.bind(&gpu_switch->ReqIn.at(0));
        gpu_switch->RspOut.at(0).bind(&l2cache->MemRspPort);

        for (uint32_t j = 0; j < cores_per_cluster; ++j) {
          cluster_mem_req_ports.at(j) = &l2cache->CoreReqPorts.at(j);
          cluster_mem_rsp_ports.at(j) = &l2cache->CoreRspPorts.at(j);
        }
      } else {
        auto l2_mem_switch = Switch<MemReq, MemRsp>::Create("l2_arb", ArbiterType::RoundRobin, cores_per_cluster);
        l2_mem_switch->ReqOut.bind(&gpu_switch->ReqIn.at(0));
        gpu_switch->RspOut.at(0).bind(&l2_mem_switch->RspIn);

        for (uint32_t j = 0; j < cores_per_cluster; ++j) {
          cluster_mem_req_ports.at(j) = &l2_mem_switch->ReqIn.at(j);
          cluster_mem_rsp_ports.at(j) = &l2_mem_switch->RspOut.at(j);
        }
      }

      raster_caches.at(i)->MemReqPort.bind(&gpu_switch->ReqIn.at(1));
      gpu_switch->RspOut.at(1).bind(&raster_caches.at(i)->MemRspPort);

      rop_caches.at(i)->MemReqPort.bind(&gpu_switch->ReqIn.at(2));
      gpu_switch->RspOut.at(2).bind(&rop_caches.at(i)->MemRspPort);

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
    for (auto raster_unit : raster_units_) {
      raster_unit->attach_ram(ram);
    }
    for (auto rop_unit : rop_units_) {
      rop_unit->attach_ram(ram);
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

  void write_dcr(uint32_t addr, uint64_t value) {
    dcrs_.write(addr, value);
  }
};

///////////////////////////////////////////////////////////////////////////////

Processor::Processor(const Arch& arch) 
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

void Processor::write_dcr(uint32_t addr, uint64_t value) {
  return impl_->write_dcr(addr, value);
}