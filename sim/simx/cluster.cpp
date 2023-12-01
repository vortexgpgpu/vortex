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
  , sharedmems_(arch.num_cores())
  , processor_(processor)
{
  auto num_cores = arch.num_cores();
  
  char sname[100];
  snprintf(sname, 100, "cluster%d-l2cache", cluster_id);
  l2cache_ = CacheSim::Create(sname, CacheSim::Config{
    !L2_ENABLED,
    log2ceil(L2_CACHE_SIZE), // C
    log2ceil(MEM_BLOCK_SIZE), // L
    log2ceil(L2_NUM_WAYS),  // W
    0,                      // A
    log2ceil(L2_NUM_BANKS), // B
    XLEN,                   // address bits  
    1,                      // number of ports
    5,                      // request size 
    true,                   // write-through
    false,                  // write response
    L2_MSHR_SIZE,           // mshr
    2,                      // pipeline latency
  });

  l2cache_->MemReqPort.bind(&this->mem_req_port);
  this->mem_rsp_port.bind(&l2cache_->MemRspPort);

  snprintf(sname, 100, "cluster%d-icaches", cluster_id);
  icaches_ = CacheCluster::Create(sname, num_cores, NUM_ICACHES, 1, CacheSim::Config{
    !ICACHE_ENABLED,
    log2ceil(ICACHE_SIZE),  // C
    log2ceil(L1_LINE_SIZE), // L
    log2ceil(sizeof(uint32_t)), // W
    log2ceil(ICACHE_NUM_WAYS),// A
    1,                      // B
    XLEN,                   // address bits
    1,                      // number of ports
    1,                      // number of inputs
    true,                   // write-through
    false,                  // write response
    (uint8_t)arch.num_warps(), // mshr
    2,                      // pipeline latency
  });

  icaches_->MemReqPort.bind(&l2cache_->CoreReqPorts.at(0));
  l2cache_->CoreRspPorts.at(0).bind(&icaches_->MemRspPort);

  snprintf(sname, 100, "cluster%d-dcaches", cluster_id);
  dcaches_ = CacheCluster::Create(sname, num_cores, NUM_DCACHES, NUM_LSU_LANES, CacheSim::Config{
    !DCACHE_ENABLED,
    log2ceil(DCACHE_SIZE),  // C
    log2ceil(L1_LINE_SIZE), // L
    log2ceil(sizeof(Word)), // W
    log2ceil(DCACHE_NUM_WAYS),// A
    log2ceil(DCACHE_NUM_BANKS), // B
    XLEN,                   // address bits
    1,                      // number of ports
    DCACHE_NUM_BANKS,       // number of inputs
    true,                   // write-through
    false,                  // write response
    DCACHE_MSHR_SIZE,       // mshr
    4,                      // pipeline latency
  });

  dcaches_->MemReqPort.bind(&l2cache_->CoreReqPorts.at(1));
  l2cache_->CoreRspPorts.at(1).bind(&dcaches_->MemRspPort);

  ///////////////////////////////////////////////////////////////////////////

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

  for (uint32_t i = 0; i < num_cores; ++i) {  
    uint32_t core_id = cluster_id * num_cores + i;
    cores_.at(i) = Core::Create(core_id, 
                                this, 
                                arch, 
                                dcrs, 
                                sharedmems_.at(i));

    cores_.at(i)->icache_req_ports.at(0).bind(&icaches_->CoreReqPorts.at(i).at(0));
    icaches_->CoreRspPorts.at(i).at(0).bind(&cores_.at(i)->icache_rsp_ports.at(0));      

    for (uint32_t j = 0; j < NUM_LSU_LANES; ++j) {
      snprintf(sname, 100, "cluster%d-smem_demux%d_%d", cluster_id, i, j);
      auto smem_demux = SMemDemux::Create(sname);
      
      cores_.at(i)->dcache_req_ports.at(j).bind(&smem_demux->ReqIn);
      smem_demux->RspIn.bind(&cores_.at(i)->dcache_rsp_ports.at(j));        
      
      smem_demux->ReqDC.bind(&dcaches_->CoreReqPorts.at(i).at(j));
      dcaches_->CoreRspPorts.at(i).at(j).bind(&smem_demux->RspDC);

      smem_demux->ReqSM.bind(&sharedmems_.at(i)->Inputs.at(j));
      sharedmems_.at(i)->Outputs.at(j).bind(&smem_demux->RspSM);
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
  perf.l2cache = l2cache_->perf_stats();

  for (auto sharedmem : sharedmems_) {
    perf.sharedmem += sharedmem->perf_stats();
  }
  
  return perf;
}