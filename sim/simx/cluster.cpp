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
#ifdef EXT_TEX_ENABLE
#include "tex_core.h"
#include "tex_unit.h"
#include "sfu_unit.h"
#endif
#ifdef EXT_OM_ENABLE
#include "om_core.h"
#include "om_unit.h"
#include "sfu_unit.h"
#endif
#ifdef EXT_RASTER_ENABLE
#include "raster_core.h"
#include "raster_unit.h"
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

    // ── L2 fan-in: sockets + optional extension caches ─────────────────
    // Row 0 = sockets (high priority).
    // Row 1 = DXA GMEM (if enabled).
    // Row 2 = tcache (if enabled).
    // Row 3 = ocache (if enabled).
    // Row 4 = rcache (if enabled).
    // The priority arbiter lets sockets win over extension traffic on
    // contention, matching the RTL `VX_mem_arb` priority ordering.
#if defined(EXT_DXA_ENABLE) || defined(EXT_TEX_ENABLE) || defined(EXT_OM_ENABLE) || defined(EXT_RASTER_ENABLE)
    constexpr uint32_t kL2Rows = 1
  #ifdef EXT_DXA_ENABLE
        + 1
  #endif
  #ifdef EXT_TEX_ENABLE
        + 1
  #endif
  #ifdef EXT_OM_ENABLE
        + 1
  #endif
  #ifdef EXT_RASTER_ENABLE
        + 1
  #endif
        ;
    snprintf(sname, 100, "%s-l2arb", name.c_str());
    auto l2arb = MemArbiter::Create(sname, ArbiterType::Priority,
                                    kL2Rows * L2_NUM_REQS, L2_NUM_REQS);
    // sockets → row 0
    for (uint32_t i = 0; i < sockets_per_cluster; ++i) {
      for (uint32_t j = 0; j < L1_MEM_PORTS; ++j) {
        uint32_t port = i * L1_MEM_PORTS + j;
        sockets_.at(i)->mem_req_out.at(j).bind(&l2arb->ReqIn.at(kL2Rows * port + 0));
        l2arb->RspOut.at(kL2Rows * port + 0).bind(&sockets_.at(i)->mem_rsp_in.at(j));
      }
    }
#else
    // No L2-sharing extensions: direct sockets → L2.
    for (uint32_t i = 0; i < sockets_per_cluster; ++i) {
      for (uint32_t j = 0; j < L1_MEM_PORTS; ++j) {
        sockets_.at(i)->mem_req_out.at(j).bind(&l2cache_->core_req_in.at(i * L1_MEM_PORTS + j));
        l2cache_->core_rsp_out.at(i * L1_MEM_PORTS + j).bind(&sockets_.at(i)->mem_rsp_in.at(j));
      }
    }
#endif // any L2-sharing extension

#ifdef EXT_DXA_ENABLE
    // Create DxaCore at cluster scope
    snprintf(sname, 100, "%s-dxa-core", name.c_str());
    dxa_core_ = DxaCore::Create(sname, simobject_);

    // DXA gmem → row 1 of l2arb.
    constexpr uint32_t kDxaRow = 1;
    uint32_t kDxaMemPorts = dxa_core_->gmem_req_out.size();
    for (uint32_t i = 0; i < kDxaMemPorts; ++i) {
      dxa_core_->gmem_req_out.at(i).bind(&l2arb->ReqIn.at(kL2Rows * i + kDxaRow));
      l2arb->RspOut.at(kL2Rows * i + kDxaRow).bind(&dxa_core_->gmem_rsp_in.at(i));
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
#endif

#ifdef EXT_TEX_ENABLE
    // ── Cluster-shared TEX engine + tcache ──────────────────────────────
    snprintf(sname, 100, "%s-tex-core", name.c_str());
    tex_core_ = TexCore::Create(sname, simobject_);

    // tcache: TLM `Cache` instance (read-only), config from VX_config.toml's
    // [tcache] section. Mirrors RTL `tcache` (VX_cache_cluster instance).
    snprintf(sname, 100, "%s-tcache", name.c_str());
    constexpr uint32_t kTcacheLineSize = MEM_BLOCK_SIZE; // = TCACHE_LINE_SIZE = L1_LINE_SIZE
    constexpr uint32_t kTcacheWordSize = 4;              // = TCACHE_WORD_SIZE
    constexpr uint32_t kTcacheNumReqs  = TCACHE_NUM_BANKS;
    constexpr uint32_t kTcacheMemPorts = 1;              // = TCACHE_MEM_PORTS
    auto tcache = Cache::Create(sname, Cache::Config{
      false,                       // bypass
      log2ceil(TCACHE_SIZE),       // C
      log2ceil(kTcacheLineSize),   // L
      log2ceil(kTcacheWordSize),   // W
      log2ceil(TCACHE_NUM_WAYS),   // A
      log2ceil(TCACHE_NUM_BANKS),  // B
      XLEN,                        // address bits
      kTcacheNumReqs,              // request size
      kTcacheMemPorts,             // memory ports
      false,                       // write-back (read-only cache)
      false,                       // write response
      TCACHE_MSHR_SIZE,            // mshr size
      2,                           // pipeline latency
      uint8_t(L2_REPL_POLICY),     // replacement policy (use L2 policy as default)
    });
    tcache_ = tcache;

    // tex_core ↔ tcache (per-port).
    for (uint32_t i = 0; i < kTcacheNumReqs; ++i) {
      tex_core_->tcache_req_out.at(i).bind(&tcache->core_req_in.at(i));
      tcache->core_rsp_out.at(i).bind(&tex_core_->tcache_rsp_in.at(i));
    }
    // tcache memory side → l2arb. Row index = kL2Rows-1 if no OM, else
    // kL2Rows-2 (OM occupies the last row when both are present).
    constexpr uint32_t kTexRow = 1
  #ifdef EXT_DXA_ENABLE
        + 1
  #endif
        ;
    for (uint32_t i = 0; i < kTcacheMemPorts; ++i) {
      tcache->mem_req_out.at(i).bind(&l2arb->ReqIn.at(kL2Rows * i + kTexRow));
      l2arb->RspOut.at(kL2Rows * i + kTexRow).bind(&tcache->mem_rsp_in.at(i));
    }

    // Cluster-level TexBus arbiter: NUM_CORES_PER_CLUSTER inputs (one per
    // SfuUnit) → 1 TEX-core lane (kNumTexCores=1 by default).
    snprintf(sname, 100, "%s-tex-bus", name.c_str());
    uint32_t cores_per_cluster = sockets_per_cluster * cores_per_socket_;
    auto tex_bus = TexBusArbiter::Create(sname, ArbiterType::RoundRobin,
                                         cores_per_cluster, 1);
    tex_bus_arb_ = tex_bus;
    for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
      for (uint32_t c = 0; c < cores_per_socket_; ++c) {
        uint32_t cid = s * cores_per_socket_ + c;
        auto sfu = sockets_.at(s)->core(c)->sfu_unit();
        sfu->tex_req_out.bind(&tex_bus->ReqIn.at(cid));
        tex_bus->RspOut.at(cid).bind(&sfu->tex_rsp_in);
      }
    }
    tex_bus->ReqOut.at(0).bind(&tex_core_->tex_req_in.at(0));
    tex_core_->tex_rsp_out.at(0).bind(&tex_bus->RspIn.at(0));
#endif

#if defined(EXT_DXA_ENABLE) || defined(EXT_TEX_ENABLE) || defined(EXT_OM_ENABLE) || defined(EXT_RASTER_ENABLE)
    // L2 arb outputs → l2cache (after all rows are bound).
    for (uint32_t i = 0; i < L2_NUM_REQS; ++i) {
      l2arb->ReqOut.at(i).bind(&l2cache_->core_req_in.at(i));
      l2cache_->core_rsp_out.at(i).bind(&l2arb->RspIn.at(i));
    }
#endif

#ifdef EXT_OM_ENABLE
    // ── Cluster-shared OM engine + ocache ───────────────────────────────
    snprintf(sname, 100, "%s-om-core", name.c_str());
    om_core_ = OmCore::Create(sname, simobject_);

    // ocache: TLM `Cache` instance (write-through), config from
    // VX_config.toml's [ocache] section. Mirrors RTL `ocache` (WRITE_ENABLE=1,
    // WRITEBACK=0 — see hw/rtl/VX_graphics.sv:311-312).
    snprintf(sname, 100, "%s-ocache", name.c_str());
    constexpr uint32_t kOcacheLineSize = MEM_BLOCK_SIZE;
    constexpr uint32_t kOcacheWordSize = 4;
    constexpr uint32_t kOcacheNumReqs  = OCACHE_NUM_BANKS;
    constexpr uint32_t kOcacheMemPorts = 1;
    auto ocache = Cache::Create(sname, Cache::Config{
      false,                        // bypass
      log2ceil(OCACHE_SIZE),        // C
      log2ceil(kOcacheLineSize),    // L
      log2ceil(kOcacheWordSize),    // W
      log2ceil(OCACHE_NUM_WAYS),    // A
      log2ceil(OCACHE_NUM_BANKS),   // B
      XLEN,                         // address bits
      kOcacheNumReqs,               // request size
      kOcacheMemPorts,              // memory ports
      false,                        // write-back (write-through)
      false,                        // write response
      OCACHE_MSHR_SIZE,             // mshr size
      2,                            // pipeline latency
      uint8_t(L2_REPL_POLICY),      // replacement policy
    });
    ocache_ = ocache;

    // om_core ↔ ocache (per-port).
    for (uint32_t i = 0; i < kOcacheNumReqs; ++i) {
      om_core_->ocache_req_out.at(i).bind(&ocache->core_req_in.at(i));
      ocache->core_rsp_out.at(i).bind(&om_core_->ocache_rsp_in.at(i));
    }

    // ocache memory side → l2arb. Row index is sockets + DXA + TEX (if those are present).
    constexpr uint32_t kOmRow = 1
  #ifdef EXT_DXA_ENABLE
        + 1
  #endif
  #ifdef EXT_TEX_ENABLE
        + 1
  #endif
        ;
    for (uint32_t i = 0; i < kOcacheMemPorts; ++i) {
      ocache->mem_req_out.at(i).bind(&l2arb->ReqIn.at(kL2Rows * i + kOmRow));
      l2arb->RspOut.at(kL2Rows * i + kOmRow).bind(&ocache->mem_rsp_in.at(i));
    }

    // Per-core SFU.om_req_out (OmUnit decodes onto it) → OmCore::om_req_in[cid].
    // OM has no return value — no rsp channel back to SfuUnit.
    for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
      for (uint32_t c = 0; c < cores_per_socket_; ++c) {
        uint32_t cid = s * cores_per_socket_ + c;
        auto sfu = sockets_.at(s)->core(c)->sfu_unit();
        sfu->om_req_out.bind(&om_core_->om_req_in.at(cid));
      }
    }
#endif

#ifdef EXT_RASTER_ENABLE
    // ── Cluster-shared RASTER engine + rcache ───────────────────────────
    snprintf(sname, 100, "%s-raster-core", name.c_str());
    raster_core_ = RasterCore::Create(sname, simobject_);

    // rcache: TLM `Cache` instance (read-only, like tcache). Mirrors RTL
    // `rcache` (VX_cache_cluster, WRITE_ENABLE=0).
    snprintf(sname, 100, "%s-rcache", name.c_str());
    constexpr uint32_t kRcacheLineSize = MEM_BLOCK_SIZE;
    constexpr uint32_t kRcacheWordSize = 4;
    constexpr uint32_t kRcacheNumReqs  = RCACHE_NUM_BANKS;
    constexpr uint32_t kRcacheMemPorts = 1;
    auto rcache = Cache::Create(sname, Cache::Config{
      false,                        // bypass
      log2ceil(RCACHE_SIZE),        // C
      log2ceil(kRcacheLineSize),    // L
      log2ceil(kRcacheWordSize),    // W
      log2ceil(RCACHE_NUM_WAYS),    // A
      log2ceil(RCACHE_NUM_BANKS),   // B
      XLEN,                         // address bits
      kRcacheNumReqs,               // request size
      kRcacheMemPorts,              // memory ports
      false,                        // write-back (read-only)
      false,                        // write response
      RCACHE_MSHR_SIZE,             // mshr size
      2,                            // pipeline latency
      uint8_t(L2_REPL_POLICY),      // replacement policy
    });
    rcache_ = rcache;

    // raster_core ↔ rcache (per-port).
    for (uint32_t i = 0; i < kRcacheNumReqs; ++i) {
      raster_core_->rcache_req_out.at(i).bind(&rcache->core_req_in.at(i));
      rcache->core_rsp_out.at(i).bind(&raster_core_->rcache_rsp_in.at(i));
    }

    // rcache memory side → l2arb (RASTER is last row when present).
    constexpr uint32_t kRasterRow = kL2Rows - 1;
    for (uint32_t i = 0; i < kRcacheMemPorts; ++i) {
      rcache->mem_req_out.at(i).bind(&l2arb->ReqIn.at(kL2Rows * i + kRasterRow));
      l2arb->RspOut.at(kL2Rows * i + kRasterRow).bind(&rcache->mem_rsp_in.at(i));
    }

    // Cluster-level RasterBus arbiter: NUM_CORES_PER_CLUSTER inputs (one per
    // SfuUnit) → 1 lane (kNumRasterLanes=1).
    snprintf(sname, 100, "%s-raster-bus", name.c_str());
    uint32_t cores_per_cluster_r = sockets_per_cluster * cores_per_socket_;
    auto raster_bus = RasterBusArbiter::Create(sname, ArbiterType::RoundRobin,
                                               cores_per_cluster_r, 1);
    raster_bus_arb_ = raster_bus;
    for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
      for (uint32_t c = 0; c < cores_per_socket_; ++c) {
        uint32_t cid = s * cores_per_socket_ + c;
        auto sfu = sockets_.at(s)->core(c)->sfu_unit();
        sfu->raster_req_out.bind(&raster_bus->ReqIn.at(cid));
        raster_bus->RspOut.at(cid).bind(&sfu->raster_rsp_in);
      }
    }
    raster_bus->ReqOut.at(0).bind(&raster_core_->raster_req_in.at(0));
    raster_core_->raster_rsp_out.at(0).bind(&raster_bus->RspIn.at(0));
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
#ifdef EXT_TEX_ENABLE
    if (addr >= VX_DCR_TEX_STATE_BEGIN && addr < VX_DCR_TEX_STATE_END) {
      return tex_core_->dcr_write(addr, value);
    }
#endif
#ifdef EXT_OM_ENABLE
    if (addr >= VX_DCR_OM_STATE_BEGIN && addr < VX_DCR_OM_STATE_END) {
      return om_core_->dcr_write(addr, value);
    }
#endif
#ifdef EXT_RASTER_ENABLE
    if (addr >= VX_DCR_RASTER_STATE_BEGIN && addr < VX_DCR_RASTER_STATE_END) {
      return raster_core_->dcr_write(addr, value);
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
#ifdef EXT_TEX_ENABLE
  TexCore::Ptr                tex_core_;
  Cache::Ptr                  tcache_;
  TexBusArbiter::Ptr          tex_bus_arb_;
#endif
#ifdef EXT_OM_ENABLE
  OmCore::Ptr                 om_core_;
  Cache::Ptr                  ocache_;
#endif
#ifdef EXT_RASTER_ENABLE
  RasterCore::Ptr             raster_core_;
  Cache::Ptr                  rcache_;
  RasterBusArbiter::Ptr       raster_bus_arb_;
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

