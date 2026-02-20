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
                const char* name,
                uint32_t socket_id,
                Cluster* cluster,
                const Arch &arch,
                const DCRS &dcrs)
  : SimObject(ctx, name)
  , mem_req_out(L1_MEM_PORTS, this)
  , mem_rsp_in(L1_MEM_PORTS, this)
  , socket_id_(socket_id)
  , cluster_(cluster)
  , cores_(arch.socket_size())
{
  auto cores_per_socket = cores_.size();

  char sname[100];
  snprintf(sname, 100, "%s-icache", name);
  icaches_ = CacheCluster::Create(sname, cores_per_socket, NUM_ICACHES, CacheSim::Config{
    !ICACHE_ENABLED,
    log2ceil(ICACHE_SIZE),  // C
    log2ceil(L1_LINE_SIZE), // L
    log2ceil(sizeof(uint32_t)), // W
    log2ceil(ICACHE_NUM_WAYS),// A
    log2ceil(1),            // B
    XLEN,                   // address bits
    1,                      // number of inputs
    ICACHE_MEM_PORTS,       // memory ports
    false,                  // write-back
    false,                  // write response
    ICACHE_MSHR_SIZE,       // mshr size
    1,                      // pipeline latency
  });

  snprintf(sname, 100, "%s-dcache", name);
  dcaches_ = CacheCluster::Create(sname, cores_per_socket, NUM_DCACHES, CacheSim::Config{
    !DCACHE_ENABLED,
    log2ceil(DCACHE_SIZE),  // C
    log2ceil(L1_LINE_SIZE), // L
    log2ceil(DCACHE_WORD_SIZE), // W
    log2ceil(DCACHE_NUM_WAYS),// A
    log2ceil(DCACHE_NUM_BANKS), // B
    XLEN,                   // address bits
    DCACHE_NUM_REQS,        // number of inputs
    L1_MEM_PORTS,           // memory ports
    DCACHE_WRITEBACK,       // write-back
    false,                  // write response
    DCACHE_MSHR_SIZE,       // mshr size
    1,                      // pipeline latency
  });

  // find overlap
  uint32_t overlap = __MIN(ICACHE_MEM_PORTS, L1_MEM_PORTS);

  // connect l1 caches to outgoing memory interfaces
  for (uint32_t i = 0; i < L1_MEM_PORTS; ++i) {
    snprintf(sname, 100, "%s-l1_arb%d", name, i);
    auto l1_arb = MemArbiter::Create(sname, ArbiterType::RoundRobin, 2 * overlap, overlap);

    if (i < overlap) {
      icaches_->mem_req_out.at(i).bind(&l1_arb->ReqIn.at(i));
      l1_arb->RspOut.at(i).bind(&icaches_->mem_rsp_in.at(i));

      dcaches_->mem_req_out.at(i).bind(&l1_arb->ReqIn.at(overlap + i));
      l1_arb->RspOut.at(overlap + i).bind(&dcaches_->mem_rsp_in.at(i));

      l1_arb->ReqOut.at(i).bind(&this->mem_req_out.at(i));
      this->mem_rsp_in.at(i).bind(&l1_arb->RspIn.at(i));
    } else {
      if (L1_MEM_PORTS > ICACHE_MEM_PORTS) {
        // if more dcache ports
        dcaches_->mem_req_out.at(i).bind(&this->mem_req_out.at(i));
        this->mem_rsp_in.at(i).bind(&dcaches_->mem_rsp_in.at(i));
      } else {
        // if more icache ports
        icaches_->mem_req_out.at(i).bind(&this->mem_req_out.at(i));
        this->mem_rsp_in.at(i).bind(&icaches_->mem_rsp_in.at(i));
      }
    }
  }

  // create cores
  for (uint32_t i = 0; i < cores_per_socket; ++i) {
    uint32_t core_id = socket_id * cores_per_socket + i;
    snprintf(sname, 100, "%s-core%d", name, i);
    cores_.at(i) = Core::Create(sname, core_id, this, arch, dcrs);
  }

  // connect cores to caches
  for (uint32_t i = 0; i < cores_per_socket; ++i) {
      cores_.at(i)->icache_req_out.at(0).bind(&icaches_->core_req_in.at(i).at(0));
      icaches_->core_rsp_out.at(i).at(0).bind(&cores_.at(i)->icache_rsp_in.at(0));

    for (uint32_t j = 0; j < DCACHE_NUM_REQS; ++j) {
      cores_.at(i)->dcache_req_out.at(j).bind(&dcaches_->core_req_in.at(i).at(j));
      dcaches_->core_rsp_out.at(i).at(j).bind(&cores_.at(i)->dcache_rsp_in.at(j));
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

#ifdef VM_ENABLE
void Socket::set_satp(uint64_t satp) {
  for (auto core : cores_) {
    core->set_satp(satp);
  }
}
#endif

bool Socket::running() const {
  for (auto& core : cores_) {
    if (core->running())
      return true;
  }
  return false;
}

int Socket::get_exitcode() const {
  int exitcode = 0;
  for (auto& core : cores_) {
    exitcode |= core->get_exitcode();
  }
  return exitcode;
}

uint32_t Socket::get_barrier_phase(uint32_t bar_id) const {
  return cluster_->get_barrier_phase(bar_id);
}

void Socket::barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t core_id) {
  cluster_->barrier_arrive(bar_id, count, core_id);
}

void Socket::resume(uint32_t core_index) {
  cores_.at(core_index)->resume(-1);
}

Socket::PerfStats Socket::perf_stats() const {
  PerfStats perf_stats;
  perf_stats.icache = icaches_->perf_stats();
  perf_stats.dcache = dcaches_->perf_stats();
  return perf_stats;
}
