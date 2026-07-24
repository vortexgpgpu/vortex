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
#include "types.h"
#ifdef VX_CFG_EXT_DXA_ENABLE
#include "dxa_core.h"
#include "local_mem.h"
#include "sfu_unit.h"
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
#include "tex_core.h"
#include "tex_unit.h"
#include "sfu_unit.h"
#endif
#ifdef VX_CFG_EXT_RTU_ENABLE
#include "rtu_core.h"
#include "rtu_unit.h"
#include "sfu_unit.h"
#endif

using namespace vortex;

class Socket::Impl {
public:
  Impl(Socket* simobject)
    : simobject_(simobject)
    , cores_(VX_CFG_SOCKET_SIZE)
    , domain_id_(SimPlatform::instance().alloc_domain())
  {
    // The socket is one execution domain: its cores, L1 cache clusters, and
    // socket-resident engines interact with same-cycle visibility; everything
    // below the socket's memory interface slices is uncore (domain 0).
    SimPlatform::DomainScope domain_scope(domain_id_);

    auto cores_per_socket = cores_.size();

    const std::string& name = simobject->name();
    char sname[100];

    snprintf(sname, 100, "%s-icache", name.c_str());
    icaches_ = CacheCluster::Create(sname, cores_per_socket, VX_CFG_NUM_ICACHES, Cache::Config{
      !VX_CFG_ICACHE_ENABLED,
      log2ceil(VX_CFG_ICACHE_SIZE),  // C
      log2ceil(VX_CFG_L1_LINE_SIZE), // L
      log2ceil(VX_CFG_L1_LINE_SIZE), // S (no sectoring)
      log2ceil(sizeof(uint32_t)), // W
      log2ceil(VX_CFG_ICACHE_NUM_WAYS),// A
      log2ceil(1),            // B
      VX_CFG_XLEN,                   // address bits
      1,                      // number of inputs
      VX_CFG_ICACHE_MEM_PORTS,       // memory ports
      false,                  // write-back
      false,                  // write response
      VX_CFG_ICACHE_MSHR_SIZE,       // mshr size
      VX_CFG_ICACHE_LATENCY,         // pipeline latency (capacity-scaled, matches hardware)
      VX_CFG_ICACHE_REPL_POLICY,     // replacement policy
      false,                  // is_llc (icache never carries AMO state)
    });

    snprintf(sname, 100, "%s-dcache", name.c_str());
    // L1 dcache is the LLC iff neither L2 nor L3 is enabled.
    dcaches_ = CacheCluster::Create(sname, cores_per_socket, VX_CFG_NUM_DCACHES, Cache::Config{
      !VX_CFG_DCACHE_ENABLED,
      log2ceil(VX_CFG_DCACHE_SIZE),  // C
      log2ceil(VX_CFG_DCACHE_LINE_SIZE), // L
      log2ceil(VX_CFG_DCACHE_SECTOR_SIZE), // S
      log2ceil(DCACHE_WORD_SIZE), // W
      log2ceil(VX_CFG_DCACHE_NUM_WAYS),// A
      log2ceil(VX_CFG_DCACHE_NUM_BANKS), // B
      VX_CFG_XLEN,                   // address bits
      VX_CFG_DCACHE_NUM_REQS,        // number of inputs
      VX_CFG_L1_MEM_PORTS,           // memory ports
      VX_CFG_DCACHE_WRITEBACK,       // write-back
      false,                  // write response
      VX_CFG_DCACHE_MSHR_SIZE,       // mshr size
      VX_CFG_DCACHE_LATENCY,         // pipeline latency (capacity-scaled, matches hardware)
      VX_CFG_DCACHE_REPL_POLICY,     // replacement policy
      (VX_CFG_DCACHE_ENABLED != 0) && (VX_CFG_L2_ENABLED == 0) && (VX_CFG_L3_ENABLED == 0), // is_llc
    });

#ifdef VX_CFG_EXT_TEX_ENABLE
    // ── Socket-resident TEX engine + tcache ─────────────────────────────
    snprintf(sname, 100, "%s-tex-core", name.c_str());
    tex_core_ = TexCore::Create(sname, simobject_);

    snprintf(sname, 100, "%s-tcache", name.c_str());
    constexpr uint32_t kTcacheLineSize = VX_CFG_MEM_BLOCK_SIZE; // = TCACHE_LINE_SIZE = VX_CFG_L1_LINE_SIZE
    constexpr uint32_t kTcacheWordSize = 4;              // = TCACHE_WORD_SIZE
    constexpr uint32_t kTcacheNumReqs  = VX_CFG_TCACHE_NUM_BANKS;
    constexpr uint32_t kTcacheMemPorts = 1;              // = TCACHE_MEM_PORTS
    auto tcache = Cache::Create(sname, Cache::Config{
      false,                       // bypass
      log2ceil(VX_CFG_TCACHE_SIZE),       // C
      log2ceil(kTcacheLineSize),   // L
      log2ceil(kTcacheLineSize),   // S (no sectoring)
      log2ceil(kTcacheWordSize),   // W
      log2ceil(VX_CFG_TCACHE_NUM_WAYS),   // A
      log2ceil(VX_CFG_TCACHE_NUM_BANKS),  // B
      VX_CFG_XLEN,                        // address bits
      kTcacheNumReqs,              // request size
      kTcacheMemPorts,             // memory ports
      false,                       // write-back (read-only cache)
      false,                       // write response
      VX_CFG_TCACHE_MSHR_SIZE,            // mshr size
      2,                           // pipeline latency
      uint8_t(VX_CFG_L2_REPL_POLICY),     // replacement policy (use L2 policy as default)
      false,                       // is_llc (TCACHE is auxiliary, not LLC)
    });
    tcache_ = tcache;

    // tex_core ↔ tcache (per-port).
    for (uint32_t i = 0; i < kTcacheNumReqs; ++i) {
      tex_core_->tcache_req_out.at(i).bind(&tcache->core_req_in.at(i));
      tcache->core_rsp_out.at(i).bind(&tex_core_->tcache_rsp_in.at(i));
    }
#endif

#ifdef VX_CFG_EXT_RTU_ENABLE
    // ── Socket-resident RTU engine + rtcache ────────────────────────────
    snprintf(sname, 100, "%s-rtu-core", name.c_str());
    rtu_core_ = RtuCore::Create(sname, simobject_);

    snprintf(sname, 100, "%s-rtcache", name.c_str());
    constexpr uint32_t kRtcacheLineSize = VX_CFG_MEM_BLOCK_SIZE;
    constexpr uint32_t kRtcacheWordSize = 4;
    // num_inputs = NUM_RTU_BLOCKS so each RtuCore memory port gets its own
    // cache input lane; the cache's internal crossbar funnels them onto
    // VX_CFG_RTCACHE_NUM_BANKS banks.
    constexpr uint32_t kRtcacheNumInputs = VX_CFG_NUM_RTU_BLOCKS;
    constexpr uint32_t kRtcacheMemPorts  = 1;
    auto rtcache = Cache::Create(sname, Cache::Config{
      false,                              // bypass
      log2ceil(VX_CFG_RTCACHE_SIZE),      // C
      log2ceil(kRtcacheLineSize),         // L
      log2ceil(kRtcacheLineSize),         // S (no sectoring)
      log2ceil(kRtcacheWordSize),         // W
      log2ceil(VX_CFG_RTCACHE_NUM_WAYS),  // A
      log2ceil(VX_CFG_RTCACHE_NUM_BANKS), // B
      VX_CFG_XLEN,                        // address bits
      kRtcacheNumInputs,                  // num_inputs (1 per RTU port)
      kRtcacheMemPorts,                   // memory ports
      false,                              // write-back (read-only)
      false,                              // write response
      VX_CFG_RTCACHE_MSHR_SIZE,           // mshr size
      2,                                  // pipeline latency
      uint8_t(VX_CFG_L2_REPL_POLICY),     // replacement policy
      false,                              // is_llc
    });
    rtcache_ = rtcache;

    // RtuCore ↔ rtcache (per memory port).
    uint32_t kRtuMemPorts = rtu_core_->dcache_req_out.size();
    for (uint32_t i = 0; i < kRtuMemPorts; ++i) {
      rtu_core_->dcache_req_out.at(i).bind(&rtcache->core_req_in.at(i));
      rtcache->core_rsp_out.at(i).bind(&rtu_core_->dcache_rsp_in.at(i));
    }
#endif

#ifdef VX_CFG_EXT_DXA_ENABLE
    // ── Socket-resident DXA engine ──────────────────────────────────────
    snprintf(sname, 100, "%s-dxa-core", name.c_str());
    dxa_core_ = DxaCore::Create(sname, simobject_);
#endif

    // find overlap
    uint32_t overlap = __MIN(VX_CFG_ICACHE_MEM_PORTS, VX_CFG_L1_MEM_PORTS);

    // Registered socket memory interface: each direction of every socket mem
    // port crosses the execution-domain boundary through a RegSlice owned by
    // its producing domain (requests socket-side, responses uncore-side), so
    // the crossing is never combinational regardless of the arbiter chain
    // behind it.
    auto bind_mem_port = [&](uint32_t port,
                             SimChannel<MemReq>& req_src,
                             SimChannel<MemRsp>& rsp_dst) {
      char bname[100];
      snprintf(bname, 100, "%s-breq%d", name.c_str(), port);
      auto breq = MemReqSlice::Create(bname, 1);
      snprintf(bname, 100, "%s-brsp%d", name.c_str(), port);
      MemRspSlice::Ptr brsp;
      {
        SimPlatform::DomainScope uncore_scope(0u);
        brsp = MemRspSlice::Create(bname, 1);
      }
      req_src.bind(&breq->In);
      breq->Out.bind(&simobject_->mem_req_out.at(port));
      simobject_->mem_rsp_in.at(port).bind(&brsp->In);
      brsp->Out.bind(&rsp_dst);
    };

#if defined(VX_CFG_EXT_TEX_ENABLE) || defined(VX_CFG_EXT_RTU_ENABLE) || defined(VX_CFG_EXT_DXA_ENABLE)
    // Port 0: icache and dcache arbitrate with the socket-resident units'
    // memory ports as peers. Priority order — icache, dcache, tcache,
    // rtcache, DXA gmem — keeps icache first and DXA bulk traffic last so
    // it cannot starve core fetch/load traffic (matches the hardware socket arb).
    __unused(overlap);
    constexpr uint32_t kSocketArbIns = 2
        + VX_CFG_EXT_TEX_ENABLED + VX_CFG_EXT_RTU_ENABLED + VX_CFG_EXT_DXA_ENABLED;
    snprintf(sname, 100, "%s-mem_arb0", name.c_str());
    auto sock_arb = MemArbiter::Create(sname, ArbiterType::Priority, kSocketArbIns, 1);
    icaches_->mem_req_out.at(0).bind(&sock_arb->ReqIn.at(0));
    sock_arb->RspOut.at(0).bind(&icaches_->mem_rsp_in.at(0));
    dcaches_->mem_req_out.at(0).bind(&sock_arb->ReqIn.at(1));
    sock_arb->RspOut.at(1).bind(&dcaches_->mem_rsp_in.at(0));
  #ifdef VX_CFG_EXT_TEX_ENABLE
    constexpr uint32_t kTexArbIdx = 2;
    tcache_->mem_req_out.at(0).bind(&sock_arb->ReqIn.at(kTexArbIdx));
    sock_arb->RspOut.at(kTexArbIdx).bind(&tcache_->mem_rsp_in.at(0));
  #endif
  #ifdef VX_CFG_EXT_RTU_ENABLE
    constexpr uint32_t kRtuArbIdx = 2 + VX_CFG_EXT_TEX_ENABLED;
    rtcache_->mem_req_out.at(0).bind(&sock_arb->ReqIn.at(kRtuArbIdx));
    sock_arb->RspOut.at(kRtuArbIdx).bind(&rtcache_->mem_rsp_in.at(0));
  #endif
  #ifdef VX_CFG_EXT_DXA_ENABLE
    constexpr uint32_t kDxaArbIdx = 2 + VX_CFG_EXT_TEX_ENABLED + VX_CFG_EXT_RTU_ENABLED;
    dxa_core_->gmem_req_out.at(0).bind(&sock_arb->ReqIn.at(kDxaArbIdx));
    sock_arb->RspOut.at(kDxaArbIdx).bind(&dxa_core_->gmem_rsp_in.at(0));
  #endif
    bind_mem_port(0, sock_arb->ReqOut.at(0), sock_arb->RspIn.at(0));

    // Remaining ports: extra dcache banks straight through.
    for (uint32_t i = 1; i < VX_CFG_L1_MEM_PORTS; ++i) {
      bind_mem_port(i, dcaches_->mem_req_out.at(i), dcaches_->mem_rsp_in.at(i));
    }
#else
    // connect l1 caches to outgoing memory interfaces
    for (uint32_t i = 0; i < VX_CFG_L1_MEM_PORTS; ++i) {
      snprintf(sname, 100, "%s-l1_arb%d", name.c_str(), i);
      auto l1_arb = MemArbiter::Create(sname, ArbiterType::RoundRobin, 2 * overlap, overlap);

      if (i < overlap) {
        icaches_->mem_req_out.at(i).bind(&l1_arb->ReqIn.at(i));
        l1_arb->RspOut.at(i).bind(&icaches_->mem_rsp_in.at(i));

        dcaches_->mem_req_out.at(i).bind(&l1_arb->ReqIn.at(overlap + i));
        l1_arb->RspOut.at(overlap + i).bind(&dcaches_->mem_rsp_in.at(i));

        bind_mem_port(i, l1_arb->ReqOut.at(i), l1_arb->RspIn.at(i));
      } else {
        if (VX_CFG_L1_MEM_PORTS > VX_CFG_ICACHE_MEM_PORTS) {
          // if more dcache ports
          bind_mem_port(i, dcaches_->mem_req_out.at(i), dcaches_->mem_rsp_in.at(i));
        } else {
          // if more icache ports
          bind_mem_port(i, icaches_->mem_req_out.at(i), icaches_->mem_rsp_in.at(i));
        }
      }
    }
#endif

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

      for (uint32_t j = 0; j < VX_CFG_DCACHE_NUM_REQS; ++j) {
        cores_.at(i)->dcache_req_out.at(j).bind(&dcaches_->core_req_in.at(i).at(j));
        dcaches_->core_rsp_out.at(i).at(j).bind(&cores_.at(i)->dcache_rsp_in.at(j));
      }
    }

#ifdef VX_CFG_EXT_TEX_ENABLE
    // Socket-level TexBus arbiter: one input per core's SfuUnit → 1 TEX core.
    snprintf(sname, 100, "%s-tex-bus", name.c_str());
    auto tex_bus = TexBusArbiter::Create(sname, ArbiterType::RoundRobin,
                                         cores_per_socket, 1);
    tex_bus_arb_ = tex_bus;
    for (uint32_t c = 0; c < cores_per_socket; ++c) {
      auto sfu = cores_.at(c)->sfu_unit();
      sfu->tex_req_out.bind(&tex_bus->ReqIn.at(c));
      tex_bus->RspOut.at(c).bind(&sfu->tex_rsp_in);
    }
    tex_bus->ReqOut.at(0).bind(&tex_core_->tex_req_in.at(0));
    tex_core_->tex_rsp_out.at(0).bind(&tex_bus->RspIn.at(0));
#endif

#ifdef VX_CFG_EXT_RTU_ENABLE
    // Socket-level RtuBus arbiter: one input per core's SfuUnit → 1 RTU core.
    snprintf(sname, 100, "%s-rtu-bus", name.c_str());
    auto rtu_bus = RtuBusArbiter::Create(sname, ArbiterType::RoundRobin,
                                         cores_per_socket, 1);
    rtu_bus_arb_ = rtu_bus;
    for (uint32_t c = 0; c < cores_per_socket; ++c) {
      auto sfu = cores_.at(c)->sfu_unit();
      sfu->rtu_req_out.bind(&rtu_bus->ReqIn.at(c));
      rtu_bus->RspOut.at(c).bind(&sfu->rtu_rsp_in);
      // Async ray pool: give each SfuUnit a direct pointer to the
      // socket's RtuCore so its RtuUnit can call allocate_slot() /
      // free_slot() without going through the bus.
      sfu->set_rtu_core(rtu_core_.get());
    }
    rtu_bus->ReqOut.at(0).bind(&rtu_core_->rtu_req_in.at(0));
    rtu_core_->rtu_rsp_out.at(0).bind(&rtu_bus->RspIn.at(0));
#endif

#ifdef VX_CFG_EXT_DXA_ENABLE
    // Per-core SFU.dxa_req_out (DxaUnit decodes onto it) → DxaCore::dxa_req_in[c].
    for (uint32_t c = 0; c < cores_per_socket; ++c) {
      auto sfu = cores_.at(c)->sfu_unit();
      sfu->dxa_req_out.bind(&dxa_core_->dxa_req_in.at(c));
    }

    // DxaCore::lmem_req_out[c] → core's LocalMem.Inputs[port_dxa].
    // A tx_callback on the channel fires barrier_event_release for each
    // DXA-write packet carrying notify_done at the cycle LMEM receives it.
    uint32_t port_dxa = LSU_NUM_REQS;
  #ifdef VX_CFG_EXT_TCU_ENABLE
    port_dxa += 1;
  #endif
    for (uint32_t c = 0; c < cores_per_socket; ++c) {
      Core* core = cores_.at(c).get();
      auto& ch = dxa_core_->lmem_req_out.at(c);
      ch.bind(&core->local_mem()->Inputs.at(port_dxa));
      ch.tx_callback([core](const MemReq& req, uint64_t /*cycles*/) {
        if (req.is_write() && req.flags.dxa_notify_done) {
          // notify_bar_id arrives in raw (encoded) form: low byte = cta_no,
          // bits[30:8] = bar_no. Decode to flat barrier index before release.
          uint32_t decoded = bar_decode_id(req.flags.dxa_notify_bar_id, VX_CFG_NUM_BARRIERS);
          core->barrier_event_release(decoded);
        }
      });
    }
#endif
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

  Socket::PerfStats perf_stats() const {
    Socket::PerfStats perf_stats;
    perf_stats.icache = icaches_->perf_stats();
    perf_stats.dcache = dcaches_->perf_stats();
#ifdef VX_CFG_EXT_DXA_ENABLE
    perf_stats.dxa = dxa_core_->perf_stats();
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
    perf_stats.tex    = tex_core_->perf_stats();
    perf_stats.tcache = tcache_->perf_stats();
#endif
#ifdef VX_CFG_EXT_RTU_ENABLE
    perf_stats.rtu     = rtu_core_->perf_stats();
    perf_stats.rtcache = rtcache_->perf_stats();
#endif
    return perf_stats;
  }

  int dcr_write(uint32_t addr, uint32_t value) {
#ifdef VX_CFG_EXT_DXA_ENABLE
    if (addr >= VX_DCR_DXA_STATE_BEGIN && addr < VX_DCR_DXA_STATE_END) {
      return dxa_core_->dcr_write(addr, value);
    }
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
    if (addr >= VX_DCR_TEX_STATE_BEGIN && addr < VX_DCR_TEX_STATE_END) {
      return tex_core_->dcr_write(addr, value);
    }
#endif
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
  void icache_flush_begin() { icaches_->flush_begin(); }
  bool icache_flush_done() const { return icaches_->flush_done(); }
#ifdef VX_CFG_EXT_TEX_ENABLE
  void tcache_flush_begin() { tcache_->flush_begin(); }
  bool tcache_flush_done() const { return tcache_->flush_done(); }
#endif
#ifdef VX_CFG_EXT_RTU_ENABLE
  void rtcache_flush_begin() { rtcache_->flush_begin(); }
  bool rtcache_flush_done() const { return rtcache_->flush_done(); }
#endif

#ifdef VX_CFG_EXT_DXA_ENABLE
  DxaCore::Ptr& dxa_core() { return dxa_core_; }
#endif

#ifdef VX_CFG_EXT_RTU_ENABLE
  RtuCore::Ptr& rtu_core() { return rtu_core_; }
#endif

private:
  Socket*                 simobject_;
  std::vector<Core::Ptr>  cores_;
  uint32_t                domain_id_;
  CacheCluster::Ptr       icaches_;
  CacheCluster::Ptr       dcaches_;
#ifdef VX_CFG_EXT_DXA_ENABLE
  DxaCore::Ptr            dxa_core_;
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
  TexCore::Ptr            tex_core_;
  Cache::Ptr              tcache_;
  TexBusArbiter::Ptr      tex_bus_arb_;
#endif
#ifdef VX_CFG_EXT_RTU_ENABLE
  RtuCore::Ptr            rtu_core_;
  Cache::Ptr              rtcache_;
  RtuBusArbiter::Ptr      rtu_bus_arb_;
#endif
};

///////////////////////////////////////////////////////////////////////////////

Socket::Socket(const SimContext& ctx,
                const char* name,
                uint32_t socket_id,
                Cluster* cluster)
  : SimObject(ctx, name)
  , mem_req_out(VX_CFG_L1_MEM_PORTS, this)
  , mem_rsp_in(VX_CFG_L1_MEM_PORTS, this)
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

bool Socket::running() const {
  return impl_->running();
}

int Socket::get_exitcode() const {
  return impl_->get_exitcode();
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

void Socket::icache_flush_begin() {
  impl_->icache_flush_begin();
}

bool Socket::icache_flush_done() const {
  return impl_->icache_flush_done();
}

#ifdef VX_CFG_EXT_TEX_ENABLE
void Socket::tcache_flush_begin() { impl_->tcache_flush_begin(); }
bool Socket::tcache_flush_done() const { return impl_->tcache_flush_done(); }
#endif

#ifdef VX_CFG_EXT_RTU_ENABLE
void Socket::rtcache_flush_begin() { impl_->rtcache_flush_begin(); }
bool Socket::rtcache_flush_done() const { return impl_->rtcache_flush_done(); }
#endif

#ifdef VX_CFG_EXT_DXA_ENABLE
DxaCore::Ptr& Socket::dxa_core() {
  return impl_->dxa_core();
}
#endif

#ifdef VX_CFG_EXT_RTU_ENABLE
RtuCore::Ptr& Socket::rtu_core() {
  return impl_->rtu_core();
}
#endif
