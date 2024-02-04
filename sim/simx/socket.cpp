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

#include "socket.h"
#include "cluster.h"

using namespace vortex;

Socket::Socket(const SimContext& ctx, 
                uint32_t socket_id,
                Cluster* cluster, 
                const Arch &arch, 
                const DCRS &dcrs) 
  : SimObject(ctx, "socket")
  , icache_mem_req_port(this)
  , icache_mem_rsp_port(this)
  , dcache_mem_req_port(this)
  , dcache_mem_rsp_port(this)
  , socket_id_(socket_id)
  , cluster_(cluster)
  , cores_(arch.socket_size())  
{
  auto cores_per_socket = cores_.size();
  
  char sname[100];
  snprintf(sname, 100, "socket%d-icaches", socket_id);
  icaches_ = CacheCluster::Create(sname, cores_per_socket, NUM_ICACHES, 1, CacheSim::Config{
    !ICACHE_ENABLED,
    log2ceil(ICACHE_SIZE),  // C
    log2ceil(L1_LINE_SIZE), // L
    log2ceil(sizeof(uint32_t)), // W
    log2ceil(ICACHE_NUM_WAYS),// A
    1,                      // B
    XLEN,                   // address bits
    1,                      // number of ports
    1,                      // number of inputs
    false,                  // write-through
    false,                  // write response
    (uint8_t)arch.num_warps(), // mshr
    2,                      // pipeline latency
  });

  icaches_->MemReqPort.bind(&icache_mem_req_port);
  icache_mem_rsp_port.bind(&icaches_->MemRspPort);

  snprintf(sname, 100, "socket%d-dcaches", socket_id);
  dcaches_ = CacheCluster::Create(sname, cores_per_socket, NUM_DCACHES, NUM_LSU_LANES, CacheSim::Config{
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
    2,                      // pipeline latency
  });

  dcaches_->MemReqPort.bind(&dcache_mem_req_port);
  dcache_mem_rsp_port.bind(&dcaches_->MemRspPort);

  // create cores

  for (uint32_t i = 0; i < cores_per_socket; ++i) {  
    uint32_t core_id = socket_id * cores_per_socket + i;
    cores_.at(i) = Core::Create(core_id, 
                                this, 
                                arch, 
                                dcrs);

    cores_.at(i)->icache_req_ports.at(0).bind(&icaches_->CoreReqPorts.at(i).at(0));
    icaches_->CoreRspPorts.at(i).at(0).bind(&cores_.at(i)->icache_rsp_ports.at(0));      

    for (uint32_t j = 0; j < NUM_LSU_LANES; ++j) {
      cores_.at(i)->dcache_req_ports.at(j).bind(&dcaches_->CoreReqPorts.at(i).at(j));
      dcaches_->CoreRspPorts.at(i).at(j).bind(&cores_.at(i)->dcache_rsp_ports.at(j));
    }
  }
}

Socket::~Socket() {
  //--
}

void Socket::reset() {  
  //--
}

void Socket::tick() {
  //--
}

void Socket::attach_ram(RAM* ram) {
  for (auto core : cores_) {
    core->attach_ram(ram);
  }
}

bool Socket::running() const {
  for (auto& core : cores_) {
    if (core->running())
      return true;
  }
  return false;
}

bool Socket::check_exit(Word* exitcode, bool riscv_test) const {
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

void Socket::barrier(uint32_t bar_id, uint32_t count, uint32_t core_id) {
  cluster_->barrier(bar_id, count, socket_id_ * cores_.size() + core_id);
}

void Socket::resume(uint32_t core_index) {
  cores_.at(core_index)->resume();
}

Socket::PerfStats Socket::perf_stats() const {
  PerfStats perf_stats;
  perf_stats.icache = icaches_->perf_stats();
  perf_stats.dcache = dcaches_->perf_stats();  
  return perf_stats;
}