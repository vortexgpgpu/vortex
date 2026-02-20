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
                 const char* name,
                 uint32_t cluster_id,
                 ProcessorImpl* processor,
                 const Arch &arch,
                 const DCRS &dcrs)
  : SimObject(ctx, name)
  , mem_req_out(L2_MEM_PORTS, this)
  , mem_rsp_in(L2_MEM_PORTS, this)
  , cluster_id_(cluster_id)
  , processor_(processor)
  , sockets_(NUM_SOCKETS)
  , barriers_(arch.socket_size() * arch.num_barriers())
  , cores_per_socket_(arch.socket_size())
{
  char sname[100];

  uint32_t sockets_per_cluster = sockets_.size();

  // create sockets

  for (uint32_t i = 0; i < sockets_per_cluster; ++i) {
    uint32_t socket_id = cluster_id * sockets_per_cluster + i;
    snprintf(sname, 100, "%s-socket%d", name, i);
    sockets_.at(i) = Socket::Create(sname, socket_id, this, arch, dcrs);
  }

  // Create l2cache

  snprintf(sname, 100, "%s-l2cache", name);
  l2cache_ = CacheSim::Create(sname, CacheSim::Config{
    !L2_ENABLED,
    log2ceil(L2_CACHE_SIZE),// C
    log2ceil(MEM_BLOCK_SIZE),// L
    log2ceil(L1_LINE_SIZE), // W
    log2ceil(L2_NUM_WAYS),  // A
    log2ceil(L2_NUM_BANKS), // B
    XLEN,                   // address bits
    L2_NUM_REQS,            // request size
    L2_MEM_PORTS,           // memory ports
    L2_WRITEBACK,           // write-back
    false,                  // write response
    L2_MSHR_SIZE,           // mshr size
    2,                      // pipeline latency
  });

  // connect l2cache core interface
  for (uint32_t i = 0; i < sockets_per_cluster; ++i) {
    for (uint32_t j = 0; j < L1_MEM_PORTS; ++j) {
      sockets_.at(i)->mem_req_out.at(j).bind(&l2cache_->core_req_in.at(i * L1_MEM_PORTS + j));
      l2cache_->core_rsp_out.at(i * L1_MEM_PORTS + j).bind(&sockets_.at(i)->mem_rsp_in.at(j));
    }
  }

  // connect l2cache memory interface
  for (uint32_t i = 0; i < L2_MEM_PORTS; ++i) {
    l2cache_->mem_req_out.at(i).bind(&this->mem_req_out.at(i));
    this->mem_rsp_in.at(i).bind(&l2cache_->mem_rsp_in.at(i));
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
}

#ifdef VM_ENABLE
void Cluster::set_satp(uint64_t satp) {
  for (auto& socket : sockets_) {
    socket->set_satp(satp);
  }
}
#endif

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

uint32_t Cluster::get_barrier_phase(uint32_t bar_id) const {
  return barriers_.at(bar_id).phase;
}

void Cluster::barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t core_id) {
  auto& barrier = barriers_.at(bar_id);

  auto sockets_per_cluster = sockets_.size();
  auto cores_per_socket = cores_per_socket_;

  uint32_t cores_per_cluster = sockets_per_cluster * cores_per_socket;
  uint32_t local_core_id = core_id % cores_per_cluster;

  // set core arrival bit
  barrier.mask.set(local_core_id);

  if (barrier.mask.count() == (size_t)count) {
    // resume all suspended cores
    for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
      for (uint32_t c = 0; c < cores_per_socket; ++c) {
        uint32_t i = s * cores_per_socket + c;
        if (barrier.mask.test(i)) {
          DP(3, "*** Resume core #" << i << " at barrier #" << bar_id);
          sockets_.at(s)->resume(c);
        }
      }
    }
    // reset mask and advance phase
    barrier.mask.reset();
    barrier.phase++;
  }
}

Cluster::PerfStats Cluster::perf_stats() const {
  PerfStats perf_stats;
  perf_stats.l2cache = l2cache_->perf_stats();
  return perf_stats;
}
