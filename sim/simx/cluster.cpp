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
                 const Arch &arch, 
                 const DCRS &dcrs) 
  : SimObject(ctx, "cluster")
  , mem_req_port(this)
  , mem_rsp_port(this)
  , cluster_id_(cluster_id)
  , processor_(processor)
  , sockets_(NUM_SOCKETS)
  , barriers_(arch.num_barriers(), 0)
  , raster_units_(NUM_RASTER_UNITS)
  , tex_units_(NUM_TEX_UNITS)
  , om_units_(NUM_OM_UNITS)
  , cores_per_socket_(arch.socket_size())
{
  char sname[100];

  uint32_t sockets_per_cluster = sockets_.size();

  // create raster units    
  for (uint32_t i = 0; i < NUM_RASTER_UNITS; ++i) {
    snprintf(sname, 100, "cluster%d-raster_unit%d", cluster_id, i);
    uint32_t raster_idx = cluster_id * NUM_RASTER_UNITS + i;      
    uint32_t raster_count = arch.num_clusters() * NUM_RASTER_UNITS;     
    raster_units_.at(i) = RasterUnit::Create(sname, raster_idx, raster_count, arch, dcrs.raster_dcrs, RasterUnit::Config{
      RASTER_TILE_LOGSIZE, 
      RASTER_BLOCK_LOGSIZE
    });
  }

  // create om units
  for (uint32_t i = 0; i < NUM_OM_UNITS; ++i) {
    snprintf(sname, 100, "cluster%d-om_unit%d", cluster_id, i);      
    om_units_.at(i) = OMUnit::Create(sname, arch, dcrs.om_dcrs);
  }

  // create tex units
  for (uint32_t i = 0; i < NUM_TEX_UNITS; ++i) {
    snprintf(sname, 100, "cluster%d-tex_unit%d", cluster_id, i);      
    tex_units_.at(i) = TexUnit::Create(sname, arch, dcrs.tex_dcrs, TexUnit::Config{
      2, // address latency
      6, // sampler latency
    });
  }

  // create sockets

  snprintf(sname, 100, "cluster%d-icache-arb", cluster_id);
  auto icache_switch = MemSwitch::Create(sname, ArbiterType::RoundRobin, sockets_per_cluster);

  snprintf(sname, 100, "cluster%d-dcache-arb", cluster_id);
  auto dcache_switch = MemSwitch::Create(sname, ArbiterType::RoundRobin, sockets_per_cluster);

  for (uint32_t i = 0, raster_idx = 0, om_idx = 0, tex_idx = 0; i < sockets_per_cluster; ++i) {
    auto per_socket_raster_units = std::max<uint32_t>((NUM_RASTER_UNITS + sockets_per_cluster - 1 - i) / sockets_per_cluster, 1);
    auto per_socket_om_units = std::max<uint32_t>((NUM_OM_UNITS + sockets_per_cluster - 1 - i) / sockets_per_cluster, 1);
    auto per_socket_tex_units = std::max<uint32_t>((NUM_TEX_UNITS + sockets_per_cluster - 1 - i) / sockets_per_cluster, 1);

    std::vector<RasterUnit::Ptr> raster_units(per_socket_raster_units);
    std::vector<TexUnit::Ptr> tex_units(per_socket_tex_units);
    std::vector<OMUnit::Ptr> om_units(per_socket_om_units);

    for (uint32_t j = 0; j < per_socket_raster_units; ++j) {
      raster_units.at(j) = raster_units_.at(raster_idx++ % NUM_RASTER_UNITS);
    }

    for (uint32_t j = 0; j < per_socket_tex_units; ++j) {
      tex_units.at(j) = tex_units_.at(tex_idx++ % NUM_TEX_UNITS);
    }

    for (uint32_t j = 0; j < per_socket_om_units; ++j) {
      om_units.at(j) = om_units_.at(om_idx++ % NUM_OM_UNITS);
    }

    uint32_t socket_id = cluster_id * sockets_per_cluster + i;

    auto socket = Socket::Create(socket_id, 
                                 this, 
                                 arch, 
                                 dcrs, 
                                 raster_units, 
                                 tex_units, 
                                 om_units);

    socket->icache_mem_req_port.bind(&icache_switch->ReqIn.at(i));
    icache_switch->RspIn.at(i).bind(&socket->icache_mem_rsp_port);

    socket->dcache_mem_req_port.bind(&dcache_switch->ReqIn.at(i));
    dcache_switch->RspIn.at(i).bind(&socket->dcache_mem_rsp_port);

    sockets_.at(i) = socket;
  }

  // Create l2cache
  
  snprintf(sname, 100, "cluster%d-l2cache", cluster_id);
  l2cache_ = CacheSim::Create(sname, CacheSim::Config{
    !L2_ENABLED,
    log2ceil(L2_CACHE_SIZE),// C
    log2ceil(MEM_BLOCK_SIZE),// L
    log2ceil(L1_LINE_SIZE), // W
    log2ceil(L2_NUM_WAYS),  // A
    log2ceil(L2_NUM_BANKS), // B
    XLEN,                   // address bits  
    1,                      // number of ports
    5,                      // request size 
    true,                   // write-through
    false,                  // write response
    L2_MSHR_SIZE,           // mshr size
    2,                      // pipeline latency
  });

  l2cache_->MemReqPort.bind(&this->mem_req_port);
  this->mem_rsp_port.bind(&l2cache_->MemRspPort);

  icache_switch->ReqOut.at(0).bind(&l2cache_->CoreReqPorts.at(0));
  l2cache_->CoreRspPorts.at(0).bind(&icache_switch->RspOut.at(0));

  dcache_switch->ReqOut.at(0).bind(&l2cache_->CoreReqPorts.at(1));
  l2cache_->CoreRspPorts.at(1).bind(&dcache_switch->RspOut.at(0));

  // Create tcache

  snprintf(sname, 100, "cluster%d-tcaches", cluster_id);
  tcaches_ = CacheCluster::Create(sname, NUM_TEX_UNITS, NUM_TCACHES, NUM_SFU_LANES, CacheSim::Config{
    !TCACHE_ENABLED,
    log2ceil(TCACHE_SIZE),  // C
    log2ceil(L1_LINE_SIZE), // L
    log2ceil(sizeof(uint32_t)), // W
    log2ceil(TCACHE_NUM_WAYS), // A
    log2ceil(TCACHE_NUM_BANKS), // B
    XLEN,                   // address bits
    1,                      // number of ports
    TCACHE_NUM_BANKS,       // number of inputs
    true,                   // write-through
    false,                  // write response
    TCACHE_MSHR_SIZE,       // mshr
    4,                      // pipeline latency
  });

  tcaches_->MemReqPort.bind(&l2cache_->CoreReqPorts.at(2));
  l2cache_->CoreRspPorts.at(2).bind(&tcaches_->MemRspPort);

  for (uint32_t i = 0; i < NUM_TEX_UNITS; ++i) {
    for (uint32_t j = 0; j < NUM_SFU_LANES; ++j) {
      tex_units_.at(i)->MemReqs.at(j).bind(&tcaches_->CoreReqPorts.at(i).at(j));
      tcaches_->CoreRspPorts.at(i).at(j).bind(&tex_units_.at(i)->MemRsps.at(j));
    }
  }

  // Create rcache

  snprintf(sname, 100, "cluster%d-rcaches", cluster_id);
  rcaches_ = CacheCluster::Create(sname, NUM_RASTER_UNITS, NUM_RCACHES, 1, CacheSim::Config{
    !RCACHE_ENABLED,
    log2ceil(RCACHE_SIZE),  // C
    log2ceil(MEM_BLOCK_SIZE), // L
    log2ceil(sizeof(uint32_t)), // W
    log2ceil(RCACHE_NUM_WAYS), // A
    log2ceil(RCACHE_NUM_BANKS), // B
    XLEN,                   // address bits
    1,                      // number of ports
    RCACHE_NUM_BANKS,       // number of inputs 
    true,                   // write-through
    false,                  // write response
    RCACHE_MSHR_SIZE,       // mshr
    4,                      // pipeline latency
  });

  rcaches_->MemReqPort.bind(&l2cache_->CoreReqPorts.at(4));
  l2cache_->CoreRspPorts.at(4).bind(&rcaches_->MemRspPort);
  
  for (uint32_t i = 0; i < NUM_RASTER_UNITS; ++i) {
    raster_units_.at(i)->MemReqs.bind(&rcaches_->CoreReqPorts.at(i).at(0));
    rcaches_->CoreRspPorts.at(i).at(0).bind(&raster_units_.at(i)->MemRsps);
  }

  // Create ocache

  snprintf(sname, 100, "cluster%d-ocaches", cluster_id);
  ocaches_ = CacheCluster::Create(sname, NUM_OM_UNITS, NUM_OCACHES, NUM_SFU_LANES, CacheSim::Config{
    !OCACHE_ENABLED,
    log2ceil(OCACHE_SIZE),  // C
    log2ceil(MEM_BLOCK_SIZE), // L
    log2ceil(sizeof(uint32_t)), // W
    log2ceil(OCACHE_NUM_WAYS), // A
    log2ceil(OCACHE_NUM_BANKS), // B
    XLEN,                   // address bits
    1,                      // number of ports
    OCACHE_NUM_BANKS,       // number of inputs
    true,                   // write-through
    false,                  // write response
    OCACHE_MSHR_SIZE,       // mshr
    4,                      // pipeline latency
  });

  ocaches_->MemReqPort.bind(&l2cache_->CoreReqPorts.at(3));
  l2cache_->CoreRspPorts.at(3).bind(&ocaches_->MemRspPort);

  for (uint32_t i = 0; i < NUM_OM_UNITS; ++i) {
    for (uint32_t j = 0; j < NUM_SFU_LANES; ++j) {
      om_units_.at(i)->MemReqs.at(j).bind(&ocaches_->CoreReqPorts.at(i).at(j));
      ocaches_->CoreRspPorts.at(i).at(j).bind(&om_units_.at(i)->MemRsps.at(j));
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
  for (auto& socket : sockets_) {
    socket->attach_ram(ram);
  }
  for (auto raster_unit : raster_units_) {
    raster_unit->attach_ram(ram);
  }
  for (auto tex_unit : tex_units_) {
    tex_unit->attach_ram(ram);
  }
  for (auto om_unit : om_units_) {
    om_unit->attach_ram(ram);
  }
}

bool Cluster::running() const {
  for (auto& socket : sockets_) {
    if (socket->running())
      return true;
  }
  return false;
}

int Cluster::get_exitcode() const {
  int exitcode = 0;
  for (auto& socket : sockets_) {
    exitcode |= socket->get_exitcode();
  }
  return exitcode;
}

void Cluster::barrier(uint32_t bar_id, uint32_t count, uint32_t core_id) {
  auto& barrier = barriers_.at(bar_id);

  auto sockets_per_cluster = sockets_.size();
  auto cores_per_socket = cores_per_socket_;

  uint32_t cores_per_cluster = sockets_per_cluster * cores_per_socket;
  uint32_t local_core_id = core_id % cores_per_cluster;
  barrier.set(local_core_id);

  DP(3, "*** Suspend core #" << core_id << " at barrier #" << bar_id);

  if (barrier.count() == (size_t)count) {
      // resume all suspended cores
      for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
        for (uint32_t c = 0; c < cores_per_socket; ++c) {
          uint32_t i = s * cores_per_socket + c;
          if (barrier.test(i)) {
            DP(3, "*** Resume core #" << i << " at barrier #" << bar_id);
            sockets_.at(s)->resume(c);
          }
        }
      }
      barrier.reset();
    }
}

Cluster::PerfStats Cluster::perf_stats() const {
  PerfStats perf_stats;
  perf_stats.l2cache = l2cache_->perf_stats();
  perf_stats.rcache = rcaches_->perf_stats();
  perf_stats.tcache = tcaches_->perf_stats();
  perf_stats.ocache = ocaches_->perf_stats();
  return perf_stats;
}