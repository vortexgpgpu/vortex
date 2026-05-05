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
#include "socket.h"
#include "core.h"
#include "local_mem.h"
#include "constants.h"
#include "types.h"
#include "debug.h"
#ifdef EXT_DXA_ENABLE
#include "dxa_core.h"
#include "sfu_unit.h"
#endif

using namespace vortex;

class Cluster::Impl {
public:
  Impl(Cluster* simobject)
    : simobject_(simobject)
    , sockets_(NUM_SOCKETS)
    , gbarriers_(NUM_BARRIERS)
    , cores_per_socket_(SOCKET_SIZE)
  {
    const std::string& name = simobject_->name();
    char sname[100];

    uint32_t sockets_per_cluster = sockets_.size();
    uint32_t cluster_id = simobject_->id();

    // create sockets

    for (uint32_t i = 0; i < sockets_per_cluster; ++i) {
      uint32_t socket_id = cluster_id * sockets_per_cluster + i;
      snprintf(sname, 100, "%s-socket%d", name.c_str(), i);
      sockets_.at(i) = Socket::Create(sname, socket_id, simobject_);
    }

    // Create l2cache

    snprintf(sname, 100, "%s-l2cache", name.c_str());
    l2cache_ = Cache::Create(sname, Cache::Config{
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
      L2_REPL_POLICY,         // replacement policy
    });


    // connect l2cache memory interface
    for (uint32_t i = 0; i < L2_MEM_PORTS; ++i) {
      l2cache_->mem_req_out.at(i).bind(&simobject_->mem_req_out.at(i));
      simobject_->mem_rsp_in.at(i).bind(&l2cache_->mem_rsp_in.at(i));
    }

#ifdef EXT_DXA_ENABLE
    // Create DxaCore at cluster scope
    snprintf(sname, 100, "%s-dxa-core", name.c_str());
    dxa_core_ = DxaCore::Create(sname, simobject_);

    // merge socket + DXA GMEM ports into L2 via a 2:1 priority arbiter
    //   input 2k   = socket port k  (high priority)
    //   input 2k+1 = DXA gmem port k (low priority)
    uint32_t kDxaMemPorts = dxa_core_->gmem_req_out.size();
    snprintf(sname, 100, "%s-l2arb", name.c_str());
    auto l2arb = MemArbiter::Create(sname, ArbiterType::Priority, 2 * L2_NUM_REQS, L2_NUM_REQS);
    for (uint32_t i = 0; i < sockets_per_cluster; ++i) {
      for (uint32_t j = 0; j < L1_MEM_PORTS; ++j) {
        uint32_t port = i * L1_MEM_PORTS + j;
        sockets_.at(i)->mem_req_out.at(j).bind(&l2arb->ReqIn.at(2 * port));
        l2arb->RspOut.at(2 * port).bind(&sockets_.at(i)->mem_rsp_in.at(j));
      }
    }
    for (uint32_t i = 0; i < kDxaMemPorts; ++i) {
      dxa_core_->gmem_req_out.at(i).bind(&l2arb->ReqIn.at(2 * i + 1));
      l2arb->RspOut.at(2 * i + 1).bind(&dxa_core_->gmem_rsp_in.at(i));
    }
    for (uint32_t i = 0; i < L2_NUM_REQS; ++i) {
      l2arb->ReqOut.at(i).bind(&l2cache_->core_req_in.at(i));
      l2cache_->core_rsp_out.at(i).bind(&l2arb->RspIn.at(i));
    }

    // Per-core SFU.dxa_req_out (DxaUnit decodes onto it) → DxaCore::dxa_req_in[cid].
    for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
      for (uint32_t c = 0; c < cores_per_socket_; ++c) {
        uint32_t cid = s * cores_per_socket_ + c;
        auto sfu = sockets_.at(s)->core(c)->sfu_unit();
        sfu->dxa_req_out.bind(&dxa_core_->dxa_req_in.at(cid));
      }
    }

    // DxaCore::lmem_req_out[cid] → core's LocalMem.Inputs[port_dxa]. The
    // DXA completion event is modelled by a SimChannel tx_callback on the
    // same channel: it snoops every DXA-write packet at delivery (the
    // cycle the LMEM input receives it) and pulses
    // core->barrier_event_release for those carrying notify_done.
    uint32_t port_dxa = LSU_NUM_REQS;
  #ifdef EXT_TCU_ENABLE
    port_dxa += 1;
  #endif
    for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
      for (uint32_t c = 0; c < cores_per_socket_; ++c) {
        uint32_t cid = s * cores_per_socket_ + c;
        Core* core = sockets_.at(s)->core(c).get();
        auto& ch = dxa_core_->lmem_req_out.at(cid);
        ch.bind(&core->local_mem()->Inputs.at(port_dxa));
        ch.tx_callback([core](const MemReq& req, uint64_t /*cycles*/) {
          if (req.write && req.notify_done) {
            core->barrier_event_release(req.notify_bar_id);
          }
        });
      }
    }
#else
    // connect l2cache core interface
    for (uint32_t i = 0; i < sockets_per_cluster; ++i) {
      for (uint32_t j = 0; j < L1_MEM_PORTS; ++j) {
        sockets_.at(i)->mem_req_out.at(j).bind(&l2cache_->core_req_in.at(i * L1_MEM_PORTS + j));
        l2cache_->core_rsp_out.at(i * L1_MEM_PORTS + j).bind(&sockets_.at(i)->mem_rsp_in.at(j));
      }
    }
#endif
  }

  void reset() {
    for (auto& gbar : gbarriers_) {
      gbar.reset();
    }
    // Sockets are SimObjects; reset by SimPlatform.
  }

  bool running() const {
    for (auto& socket : sockets_) {
      if (socket->running())
        return true;
    }
    return false;
  }

  int get_exitcode() const {
    int exitcode = 0;
    for (auto& socket : sockets_) {
      exitcode |= socket->get_exitcode();
    }
    return exitcode;
  }

  void global_barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t core_id) {
    auto bar_index = bar_id % gbarriers_.size();
    auto& gbar = gbarriers_.at(bar_index);

    auto sockets_per_cluster = sockets_.size();
    auto cores_per_socket = cores_per_socket_;

    uint32_t cores_per_cluster = sockets_per_cluster * cores_per_socket;
    uint32_t local_core_id = core_id % cores_per_cluster;

    // set core arrival bit
    gbar.mask.set(local_core_id);

    DT(4, "*** Global barrier arrive: cluster #" << simobject_->id() << ", core #" << core_id << " at barrier #" << bar_id << ", arrived=" << gbar.mask.count());

    if (gbar.mask.count() == (size_t)count) {
      // resume all suspended cores
      for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
        for (uint32_t c = 0; c < cores_per_socket; ++c) {
          uint32_t i = s * cores_per_socket + c;
          if (gbar.mask.test(i)) {
            sockets_.at(s)->global_barrier_resume(bar_id, c);
          }
        }
      }
      // reset mask and advance phase
      gbar.mask.reset();
    }
  }

  Cluster::PerfStats perf_stats() const {
    Cluster::PerfStats perf_stats;
    perf_stats.l2cache = l2cache_->perf_stats();
#ifdef EXT_DXA_ENABLE
    perf_stats.dxa = dxa_core_->perf_stats();
#endif
    return perf_stats;
  }

  int dcr_write(uint32_t addr, uint32_t value) {
#ifdef EXT_DXA_ENABLE
    if (addr >= VX_DCR_DXA_STATE_BEGIN && addr < VX_DCR_DXA_STATE_END) {
      return dxa_core_->dcr_write(addr, value);
    }
#endif
    for (auto& socket : sockets_) {
      int ret = socket->dcr_write(addr, value);
      if (ret != 0)
        return ret;
    }
    return 0;
  }

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
    for (auto& socket : sockets_) {
      int ret = socket->dcr_read(addr, tag, value);
      if (ret != 0)
        return ret;
    }
    return 0;
  }

  void dcache_flush_begin() {
    for (auto& socket : sockets_) {
      socket->dcache_flush_begin();
    }
  }

  bool dcache_flush_done() const {
    for (auto& socket : sockets_) {
      if (!socket->dcache_flush_done()) return false;
    }
    return true;
  }

  void l2_flush_begin() {
    l2cache_->flush_begin();
  }

  bool l2_flush_done() const {
    return l2cache_->flush_done();
  }

  Core* get_core(uint32_t idx) const {
    uint32_t sockets_per_cluster = sockets_.size();
    if (idx >= sockets_per_cluster * cores_per_socket_) return nullptr;
    uint32_t s = idx / cores_per_socket_;
    uint32_t c = idx % cores_per_socket_;
    return sockets_.at(s)->core(c).get();
  }

#ifdef EXT_DXA_ENABLE
  DxaCore::Ptr& dxa_core() { return dxa_core_; }
#endif

private:
  Cluster*                    simobject_;
  std::vector<Socket::Ptr>    sockets_;
  std::vector<core_barrier_t> gbarriers_;
  Cache::Ptr                  l2cache_;
  uint32_t                    cores_per_socket_;
#ifdef EXT_DXA_ENABLE
  DxaCore::Ptr                dxa_core_;
#endif
};

///////////////////////////////////////////////////////////////////////////////

Cluster::Cluster(const SimContext& ctx,
                 const char* name,
                 uint32_t cluster_id,
                 ProcessorImpl* processor)
  : SimObject(ctx, name)
  , mem_req_out(L2_MEM_PORTS, this)
  , mem_rsp_in(L2_MEM_PORTS, this)
  , cluster_id_(cluster_id)
  , processor_(processor)
  , impl_(new Impl(this))
{}

Cluster::~Cluster() {
  delete impl_;
}

void Cluster::on_reset() {
  impl_->reset();
}

void Cluster::on_tick() {
  //--
}

bool Cluster::running() const {
  return impl_->running();
}

int Cluster::get_exitcode() const {
  return impl_->get_exitcode();
}

void Cluster::global_barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t core_id) {
  impl_->global_barrier_arrive(bar_id, count, core_id);
}

Cluster::PerfStats Cluster::perf_stats() const {
  return impl_->perf_stats();
}

int Cluster::dcr_write(uint32_t addr, uint32_t value) {
  return impl_->dcr_write(addr, value);
}

int Cluster::dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
  return impl_->dcr_read(addr, tag, value);
}

Core* Cluster::get_core(uint32_t idx) const {
  return impl_->get_core(idx);
}

void Cluster::dcache_flush_begin() {
  impl_->dcache_flush_begin();
}

bool Cluster::dcache_flush_done() const {
  return impl_->dcache_flush_done();
}

void Cluster::l2_flush_begin() {
  impl_->l2_flush_begin();
}

bool Cluster::l2_flush_done() const {
  return impl_->l2_flush_done();
}

#ifdef EXT_DXA_ENABLE
DxaCore::Ptr& Cluster::dxa_core() {
  return impl_->dxa_core();
}
#endif
