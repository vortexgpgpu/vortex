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
#include "cache_cluster.h"
#include "core.h"
#include "cluster.h"
#include "constants.h"

using namespace vortex;

class Socket::Impl {
public:
  Impl(Socket* simobject)
    : simobject_(simobject)
    , cores_(SOCKET_SIZE)
  {
    auto cores_per_socket = cores_.size();

    const std::string& name = simobject->name();
    char sname[100];

    snprintf(sname, 100, "%s-icache", name.c_str());
    icaches_ = CacheCluster::Create(sname, cores_per_socket, NUM_ICACHES, Cache::Config{
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
      ICACHE_REPL_POLICY,     // replacement policy
    });

    snprintf(sname, 100, "%s-dcache", name.c_str());
    dcaches_ = CacheCluster::Create(sname, cores_per_socket, NUM_DCACHES, Cache::Config{
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
      DCACHE_REPL_POLICY,     // replacement policy
    });

    // find overlap
    uint32_t overlap = __MIN(ICACHE_MEM_PORTS, L1_MEM_PORTS);

    // connect l1 caches to outgoing memory interfaces
    for (uint32_t i = 0; i < L1_MEM_PORTS; ++i) {
      snprintf(sname, 100, "%s-l1_arb%d", name.c_str(), i);
      auto l1_arb = MemArbiter::Create(sname, ArbiterType::RoundRobin, 2 * overlap, overlap);

      if (i < overlap) {
        icaches_->mem_req_out.at(i).bind(&l1_arb->ReqIn.at(i));
        l1_arb->RspOut.at(i).bind(&icaches_->mem_rsp_in.at(i));

        dcaches_->mem_req_out.at(i).bind(&l1_arb->ReqIn.at(overlap + i));
        l1_arb->RspOut.at(overlap + i).bind(&dcaches_->mem_rsp_in.at(i));

        l1_arb->ReqOut.at(i).bind(&simobject->mem_req_out.at(i));
        simobject->mem_rsp_in.at(i).bind(&l1_arb->RspIn.at(i));
      } else {
        if (L1_MEM_PORTS > ICACHE_MEM_PORTS) {
          // if more dcache ports
          dcaches_->mem_req_out.at(i).bind(&simobject->mem_req_out.at(i));
          simobject->mem_rsp_in.at(i).bind(&dcaches_->mem_rsp_in.at(i));
        } else {
          // if more icache ports
          icaches_->mem_req_out.at(i).bind(&simobject->mem_req_out.at(i));
          simobject->mem_rsp_in.at(i).bind(&icaches_->mem_rsp_in.at(i));
        }
      }
    }

    // create cores
    for (uint32_t i = 0; i < cores_per_socket; ++i) {
      uint32_t core_id = simobject_->id() * cores_per_socket + i;
      snprintf(sname, 100, "%s-core%d", name.c_str(), i);
      cores_.at(i) = Core::Create(sname, core_id, simobject_);
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

  bool running() const {
    for (auto& core : cores_) {
      if (core->running())
        return true;
    }
    return false;
  }

  int get_exitcode() const {
    int exitcode = 0;
    for (auto& core : cores_) {
      exitcode |= core->get_exitcode();
    }
    return exitcode;
  }

  void global_barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t core_id) {
    simobject_->cluster()->global_barrier_arrive(bar_id, count, core_id);
  }

  void global_barrier_resume(uint32_t bar_id, uint32_t core_index) {
    cores_.at(core_index)->global_barrier_resume(bar_id);
  }

  Socket::PerfStats perf_stats() const {
    Socket::PerfStats perf_stats;
    perf_stats.icache = icaches_->perf_stats();
    perf_stats.dcache = dcaches_->perf_stats();
    return perf_stats;
  }

  int dcr_write(uint32_t addr, uint32_t value) {
    for (auto& core : cores_) {
      int ret = core->dcr_write(addr, value);
      if (ret != 0)
        return ret;
    }
    return 0;
  }

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
    for (auto& core : cores_) {
      uint16_t core_id = tag & 0xfffff;
      if (core_id != core->id())
        continue; // skip cores that don't match the tag
      uint32_t tag_value = tag >> 16;
      int ret = core->dcr_read(addr, tag_value, value);
      if (ret != 0)
        return ret;
    }
    return 0;
  }

  Core::Ptr& core(uint32_t idx) {
    return cores_.at(idx);
  }

  void dcache_flush_begin() { dcaches_->flush_begin(); }
  bool dcache_flush_done() const { return dcaches_->flush_done(); }

private:
  Socket*                 simobject_;
  std::vector<Core::Ptr>  cores_;
  CacheCluster::Ptr       icaches_;
  CacheCluster::Ptr       dcaches_;
};

///////////////////////////////////////////////////////////////////////////////

Socket::Socket(const SimContext& ctx,
                const char* name,
                uint32_t socket_id,
                Cluster* cluster)
  : SimObject(ctx, name)
  , mem_req_out(L1_MEM_PORTS, this)
  , mem_rsp_in(L1_MEM_PORTS, this)
  , socket_id_(socket_id)
  , cluster_(cluster)
  , impl_(new Impl(this))
{}

Socket::~Socket() {
  delete impl_;
}

void Socket::on_reset() {
  // Cores are SimObjects; reset by SimPlatform.
}

void Socket::on_tick() {
  //--
}

bool Socket::running() const {
  return impl_->running();
}

int Socket::get_exitcode() const {
  return impl_->get_exitcode();
}

void Socket::global_barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t core_id) {
  impl_->global_barrier_arrive(bar_id, count, core_id);
}

void Socket::global_barrier_resume(uint32_t bar_id, uint32_t core_index) {
  impl_->global_barrier_resume(bar_id, core_index);
}

Socket::PerfStats Socket::perf_stats() const {
  return impl_->perf_stats();
}

int Socket::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

int Socket::dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
  return impl_->dcr_read(addr, tag, value);
}

Core::Ptr& Socket::core(uint32_t idx) {
  return impl_->core(idx);
}

void Socket::dcache_flush_begin() {
  impl_->dcache_flush_begin();
}

bool Socket::dcache_flush_done() const {
  return impl_->dcache_flush_done();
}
