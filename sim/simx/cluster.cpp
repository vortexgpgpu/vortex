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
                 const Arch &arch)
  : SimObject(ctx, name)
  , mem_req_out(L2_MEM_PORTS, this)
  , mem_rsp_in(L2_MEM_PORTS, this)
  , cluster_id_(cluster_id)
  , processor_(processor)
  , sockets_(NUM_SOCKETS)
  , gbarriers_(arch.num_barriers())
  , cores_per_socket_(arch.socket_size())
{
  char sname[100];

  uint32_t sockets_per_cluster = sockets_.size();

  // create sockets

  for (uint32_t i = 0; i < sockets_per_cluster; ++i) {
    uint32_t socket_id = cluster_id * sockets_per_cluster + i;
    snprintf(sname, 100, "%s-socket%d", name, i);
    sockets_.at(i) = Socket::Create(sname, socket_id, this, arch);
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


  // connect l2cache memory interface
  for (uint32_t i = 0; i < L2_MEM_PORTS; ++i) {
    l2cache_->mem_req_out.at(i).bind(&this->mem_req_out.at(i));
    this->mem_rsp_in.at(i).bind(&l2cache_->mem_rsp_in.at(i));
  }

#ifdef EXT_DXA_ENABLE
  // Create DxaCore at cluster scope
  snprintf(sname, 100, "%s-dxa-core", name);
  dxa_core_ = DxaCore::Create(sname, this);

  // Merge socket + DXA GMEM ports into L2 via a 2:1 priority arbiter.
  // Mirrors RTL VX_mem_arb(NUM_INPUTS=2*L2_SOCKET_REQS, NUM_OUTPUTS=L2_SOCKET_REQS).
  // TxArbiter output k arbitrates over inputs {2k, 2k+1}:
  //   input 2k   = socket port k  (high priority, lower index wins in Priority mode)
  //   input 2k+1 = DXA gmem port k (low priority; idle when k >= kDxaMemPorts)
  uint32_t kDxaMemPorts = dxa_core_->gmem_req_out.size();
  snprintf(sname, 100, "%s-l2arb", name);
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
  // Wire DXA SMEM timing channel to each core's LocalMem.
  // LocalMem::dxa_req_in releases the barrier on the is_last element.
  for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
    for (uint32_t c = 0; c < cores_per_socket_; ++c) {
      uint32_t lmem_idx = s * cores_per_socket_ + c;
      dxa_core_->lmem_req_out.at(lmem_idx).bind(
          &sockets_.at(s)->core(c)->local_mem()->dxa_req_in);
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

Cluster::~Cluster() {
  //--
}

void Cluster::reset() {
  for (auto& gbar : gbarriers_) {
    gbar.reset();
  }
  for (auto& socket : sockets_) {
    socket->reset();
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

void Cluster::global_barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t core_id) {
  auto bar_index = bar_id % gbarriers_.size();
  auto& gbar = gbarriers_.at(bar_index);

  auto sockets_per_cluster = sockets_.size();
  auto cores_per_socket = cores_per_socket_;

  uint32_t cores_per_cluster = sockets_per_cluster * cores_per_socket;
  uint32_t local_core_id = core_id % cores_per_cluster;

  // set core arrival bit
  gbar.mask.set(local_core_id);

  DT(4, "*** Global barrier arrive: cluster #" << id() << ", core #" << core_id << " at barrier #" << bar_id << ", arrived=" << gbar.mask.count());

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

Cluster::PerfStats Cluster::perf_stats() const {
  PerfStats perf_stats;
  perf_stats.l2cache = l2cache_->perf_stats();
#ifdef EXT_DXA_ENABLE
  perf_stats.dxa = dxa_core_->perf_stats();
#endif
  return perf_stats;
}

int Cluster::dcr_write(uint32_t addr, uint32_t value) {
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

int Cluster::dcr_read(uint32_t addr, uint32_t tag, uint32_t* value) {
  for (auto& socket : sockets_) {
    int ret = socket->dcr_read(addr, tag, value);
    if (ret != 0)
      return ret;
  }
  return 0;
}
