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

#include "cluster.h"

using namespace vortex;

Cluster::Cluster(const SimContext& ctx, 
                 uint32_t cluster_id,
                 ProcessorImpl* processor, 
                 const Arch &arch, const 
                 DCRS &dcrs) 
  : SimObject(ctx, "cluster")
  , mem_req_port(this)
  , mem_rsp_port(this)
  , cluster_id_(cluster_id)
  , cores_(arch.num_cores())  
  , barriers_(arch.num_barriers(), 0)
  , raster_units_(NUM_RASTER_UNITS)
  , rop_units_(NUM_ROP_UNITS)
  , tex_units_(NUM_TEX_UNITS)
  , sharedmems_(arch.num_cores())
  , processor_(processor)
{
  auto num_cores = arch.num_cores();
  
  char sname[100];
  snprintf(sname, 100, "cluster%d-l2cache", cluster_id);
  l2cache_ = CacheSim::Create(sname, CacheSim::Config{
    !L2_ENABLED,
    log2ceil(L2_CACHE_SIZE), // C
    log2ceil(MEM_BLOCK_SIZE), // B
    log2ceil(L2_NUM_WAYS),  // W
    0,                      // A
    XLEN,                   // address bits  
    L2_NUM_BANKS,           // number of banks
    1,                      // number of ports
    5,                      // request size 
    true,                   // write-through
    false,                  // write response
    0,                      // victim size
    L2_MSHR_SIZE,           // mshr
    2,                      // pipeline latency
  });

  l2cache_->MemReqPort.bind(&this->mem_req_port);
  this->mem_rsp_port.bind(&l2cache_->MemRspPort);

  snprintf(sname, 100, "cluster%d-icaches", cluster_id);
  icaches_ = CacheCluster::Create(sname, num_cores, NUM_ICACHES, 1, CacheSim::Config{
    !ICACHE_ENABLED,
    log2ceil(ICACHE_SIZE),  // C
    log2ceil(L1_LINE_SIZE), // B
    log2ceil(sizeof(uint32_t)), // W
    log2ceil(ICACHE_NUM_WAYS),// A
    XLEN,                   // address bits    
    1,                      // number of banks
    1,                      // number of ports
    1,                      // number of inputs
    true,                   // write-through
    false,                  // write response
    0,                      // victim size
    (uint8_t)arch.num_warps(), // mshr
    2,                      // pipeline latency
  });

  icaches_->MemReqPort.bind(&l2cache_->CoreReqPorts.at(0));
  l2cache_->CoreRspPorts.at(0).bind(&icaches_->MemRspPort);

  snprintf(sname, 100, "cluster%d-dcaches", cluster_id);
  dcaches_ = CacheCluster::Create(sname, num_cores, NUM_DCACHES, NUM_LSU_LANES, CacheSim::Config{
    !DCACHE_ENABLED,
    log2ceil(DCACHE_SIZE),  // C
    log2ceil(L1_LINE_SIZE), // B
    log2ceil(sizeof(Word)), // W
    log2ceil(DCACHE_NUM_WAYS),// A
    XLEN,                   // address bits    
    DCACHE_NUM_BANKS,       // number of banks
    1,                      // number of ports
    DCACHE_NUM_BANKS,       // number of inputs
    true,                   // write-through
    false,                  // write response
    0,                      // victim size
    DCACHE_MSHR_SIZE,       // mshr
    4,                      // pipeline latency
  });

  dcaches_->MemReqPort.bind(&l2cache_->CoreReqPorts.at(1));
  l2cache_->CoreRspPorts.at(1).bind(&dcaches_->MemRspPort);
  
  snprintf(sname, 100, "cluster%d-tcaches", cluster_id);
  tcaches_ = CacheCluster::Create(sname, NUM_TEX_UNITS, NUM_TCACHES, NUM_SFU_LANES, CacheSim::Config{
    !TCACHE_ENABLED,
    log2ceil(TCACHE_SIZE),  // C
    log2ceil(L1_LINE_SIZE), // B
    log2ceil(sizeof(uint32_t)), // W
    log2ceil(TCACHE_NUM_WAYS),// A
    XLEN,                   // address bits    
    TCACHE_NUM_BANKS,       // number of banks
    1,                      // number of ports
    TCACHE_NUM_BANKS,       // number of inputs
    true,                   // write-through
    false,                  // write response
    0,                      // victim size
    TCACHE_MSHR_SIZE,       // mshr
    4,                      // pipeline latency
  });

  tcaches_->MemReqPort.bind(&l2cache_->CoreReqPorts.at(2));
  l2cache_->CoreRspPorts.at(2).bind(&tcaches_->MemRspPort);

  snprintf(sname, 100, "cluster%d-ocaches", cluster_id);
  ocaches_ = CacheCluster::Create(sname, NUM_ROP_UNITS, NUM_OCACHES, NUM_SFU_LANES, CacheSim::Config{
    !OCACHE_ENABLED,
    log2ceil(OCACHE_SIZE),  // C
    log2ceil(MEM_BLOCK_SIZE), // B
    log2ceil(sizeof(uint32_t)), // W
    log2ceil(OCACHE_NUM_WAYS), // A 
    XLEN,                   // address bits    
    OCACHE_NUM_BANKS,       // number of banks
    1,                      // number of ports
    OCACHE_NUM_BANKS,       // number of inputs
    true,                   // write-through
    false,                  // write response
    0,                      // victim size
    OCACHE_MSHR_SIZE,       // mshr
    4,                      // pipeline latency
  });

  ocaches_->MemReqPort.bind(&l2cache_->CoreReqPorts.at(3));
  l2cache_->CoreRspPorts.at(3).bind(&ocaches_->MemRspPort);

  snprintf(sname, 100, "cluster%d-rcaches", cluster_id);
  rcaches_ = CacheCluster::Create(sname, NUM_RASTER_UNITS, NUM_RCACHES, 1, CacheSim::Config{
    !RCACHE_ENABLED,
    log2ceil(RCACHE_SIZE),  // C
    log2ceil(MEM_BLOCK_SIZE), // B
    log2ceil(sizeof(uint32_t)), // W
    log2ceil(RCACHE_NUM_WAYS), // A
    XLEN,                   // address bits    
    RCACHE_NUM_BANKS,       // number of banks
    1,                      // number of ports
    RCACHE_NUM_BANKS,       // number of inputs 
    true,                   // write-through
    false,                  // write response
    0,                      // victim size
    RCACHE_MSHR_SIZE,       // mshr
    4,                      // pipeline latency
  });

  rcaches_->MemReqPort.bind(&l2cache_->CoreReqPorts.at(4));
  l2cache_->CoreRspPorts.at(4).bind(&rcaches_->MemRspPort);

  ///////////////////////////////////////////////////////////////////////////

  // create raster units    
  for (uint32_t i = 0; i < NUM_RASTER_UNITS; ++i) {
    snprintf(sname, 100, "cluster%d-raster_unit%d", cluster_id, i);
    uint32_t raster_idx = cluster_id * NUM_RASTER_UNITS + i;      
    uint32_t raster_count = arch.num_clusters() * NUM_RASTER_UNITS;     
    raster_units_.at(i) = RasterUnit::Create(sname, raster_idx, raster_count, arch, dcrs.raster_dcrs, RasterUnit::Config{
      RASTER_TILE_LOGSIZE, 
      RASTER_BLOCK_LOGSIZE
    });
    raster_units_.at(i)->MemReqs.bind(&rcaches_->CoreReqPorts.at(i).at(0));
    rcaches_->CoreRspPorts.at(i).at(0).bind(&raster_units_.at(i)->MemRsps);
  }

  // create rop units
  for (uint32_t i = 0; i < NUM_ROP_UNITS; ++i) {
    snprintf(sname, 100, "cluster%d-rop_unit%d", cluster_id, i);      
    rop_units_.at(i) = RopUnit::Create(sname, arch, dcrs.rop_dcrs);
    for (uint32_t j = 0; j < NUM_SFU_LANES; ++j) {
      rop_units_.at(i)->MemReqs.at(j).bind(&ocaches_->CoreReqPorts.at(i).at(j));
      ocaches_->CoreRspPorts.at(i).at(j).bind(&rop_units_.at(i)->MemRsps.at(j));
    }
  }

  // create tex units
  for (uint32_t i = 0; i < NUM_TEX_UNITS; ++i) {
    snprintf(sname, 100, "cluster%d-tex_unit%d", cluster_id, i);      
    tex_units_.at(i) = TexUnit::Create(sname, arch, dcrs.tex_dcrs, TexUnit::Config{
      2, // address latency
      6, // sampler latency
    });      
    for (uint32_t j = 0; j < NUM_SFU_LANES; ++j) {
      tex_units_.at(i)->MemReqs.at(j).bind(&tcaches_->CoreReqPorts.at(i).at(j));
      tcaches_->CoreRspPorts.at(i).at(j).bind(&tex_units_.at(i)->MemRsps.at(j));
    }
  }

  // create shared memory blocks
  for (uint32_t i = 0; i < num_cores; ++i) {
    snprintf(sname, 100, "cluster%d-shared_mem%d", cluster_id, i);
    sharedmems_.at(i) = SharedMem::Create(sname, SharedMem::Config{
      (1 << SMEM_LOG_SIZE),
      sizeof(Word),
      NUM_LSU_LANES, 
      NUM_LSU_LANES,
      false
    });
  }

  // create cores 

  for (uint32_t raster_idx = 0, rop_idx = 0, tex_idx = 0, i = 0; i < num_cores; ++i) {  
    auto per_core_raster_units = std::max<uint32_t>((NUM_RASTER_UNITS + num_cores - 1 - i) / num_cores, 1);
    auto per_core_rop_units = std::max<uint32_t>((NUM_ROP_UNITS + num_cores - 1 - i) / num_cores, 1);
    auto per_core_tex_units = std::max<uint32_t>((NUM_TEX_UNITS + num_cores - 1 - i) / num_cores, 1);

    std::vector<RasterUnit::Ptr> raster_units(per_core_raster_units);
    std::vector<RopUnit::Ptr> rop_units(per_core_rop_units);
    std::vector<TexUnit::Ptr> tex_units(per_core_tex_units);

    for (uint32_t j = 0; j < per_core_raster_units; ++j) {
      raster_units.at(j) = raster_units_.at(raster_idx++ % NUM_RASTER_UNITS);
    }

    for (uint32_t j = 0; j < per_core_rop_units; ++j) {
      rop_units.at(j) = rop_units_.at(rop_idx++ % NUM_ROP_UNITS);
    }

    for (uint32_t j = 0; j < per_core_tex_units; ++j) {
      tex_units.at(j) = tex_units_.at(tex_idx++ % NUM_TEX_UNITS);
    }

    uint32_t core_id = cluster_id * num_cores + i;

    cores_.at(i) = Core::Create(core_id, 
                                this, 
                                arch, 
                                dcrs, 
                                sharedmems_.at(i), 
                                raster_units, 
                                rop_units, 
                                tex_units);

    cores_.at(i)->icache_req_ports.at(0).bind(&icaches_->CoreReqPorts.at(i).at(0));
    icaches_->CoreRspPorts.at(i).at(0).bind(&cores_.at(i)->icache_rsp_ports.at(0));      

    for (uint32_t j = 0; j < NUM_LSU_LANES; ++j) {
      snprintf(sname, 100, "cluster%d-smem_demux%d_%d", cluster_id, i, j);
      auto smem_demux = SMemDemux::Create(sname);
      
      cores_.at(i)->dcache_req_ports.at(j).bind(&smem_demux->ReqIn);
      smem_demux->RspIn.bind(&cores_.at(i)->dcache_rsp_ports.at(j));        
      
      smem_demux->ReqDc.bind(&dcaches_->CoreReqPorts.at(i).at(j));
      dcaches_->CoreRspPorts.at(i).at(j).bind(&smem_demux->RspDc);

      smem_demux->ReqSm.bind(&sharedmems_.at(i)->Inputs.at(j));
      sharedmems_.at(i)->Outputs.at(j).bind(&smem_demux->RspSm);
    }
  }
}

Cluster::~Cluster() {
  //--
}

void Cluster::reset() {  
  for (auto& barrier : barriers_) {
    barrier.reset();
  }
}

void Cluster::tick() {
  //--
}

void Cluster::attach_ram(RAM* ram) {
  for (auto core : cores_) {
    core->attach_ram(ram);
  }
  for (auto raster_unit : raster_units_) {
    raster_unit->attach_ram(ram);
  }
  for (auto rop_unit : rop_units_) {
    rop_unit->attach_ram(ram);
  }
  for (auto tex_unit : tex_units_) {
    tex_unit->attach_ram(ram);
  }
}

bool Cluster::running() const {
  for (auto& core : cores_) {
    if (core->running())
      return true;
  }
  return false;
}

bool Cluster::check_exit(Word* exitcode, bool riscv_test) const {
  bool done = true;
  Word exitcode_ = 0;
  for (auto& core : cores_) {
    Word ec;
    if (core->check_exit(&ec, riscv_test)) {
      exitcode_ |= ec;
    } else {
      done = false;
    }
  }
  *exitcode = exitcode_;
  return done;
}

void Cluster::barrier(uint32_t bar_id, uint32_t count, uint32_t core_id) {
  auto& barrier = barriers_.at(bar_id);

  uint32_t local_core_id = core_id % cores_.size();
  barrier.set(local_core_id);

  DP(3, "*** Suspend core #" << core_id << " at barrier #" << bar_id);

  if (barrier.count() == (size_t)count) {
      // resume all suspended cores
      for (uint32_t i = 0; i < cores_.size(); ++i) {
        if (barrier.test(i)) {
          DP(3, "*** Resume core #" << i << " at barrier #" << bar_id);
          cores_.at(i)->resume();
        }
      }
      barrier.reset();
    }
}

ProcessorImpl* Cluster::processor() const {
  return processor_;
}

Cluster::PerfStats Cluster::perf_stats() const {
  Cluster::PerfStats perf;
  perf.icache = icaches_->perf_stats();
  perf.dcache = dcaches_->perf_stats();    
  perf.tcache = tcaches_->perf_stats();
  perf.ocache = ocaches_->perf_stats();
  perf.rcache = rcaches_->perf_stats();
  perf.l2cache = l2cache_->perf_stats();

  for (auto sharedmem : sharedmems_) {
    perf.sharedmem += sharedmem->perf_stats();
  }
  
  for (uint32_t i = 0; i < NUM_RASTER_UNITS; ++i) {
    perf.raster_unit += raster_units_.at(i)->perf_stats();
  }
  
  for (uint32_t i = 0; i < NUM_ROP_UNITS; ++i) {
    perf.rop_unit += rop_units_.at(i)->perf_stats();
  }

  for (uint32_t i = 0; i < NUM_TEX_UNITS; ++i) {
    perf.tex_unit += tex_units_.at(i)->perf_stats();
  }    
  
  return perf;
}