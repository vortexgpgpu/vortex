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
#ifdef VX_CFG_EXT_OM_ENABLE
#include "om_core.h"
#include "om_unit.h"
#include "sfu_unit.h"
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
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
    , gbarriers_(VX_CFG_NUM_BARRIERS)
    , cores_per_socket_(VX_CFG_SOCKET_SIZE)
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

    // Global-barrier event links: each core's arrive end fans into the
    // cluster's handler; one resume link fans back out per core.
    for (uint32_t i = 0; i < sockets_per_cluster; ++i) {
      for (uint32_t c = 0; c < cores_per_socket_; ++c) {
        auto* core = sockets_.at(i)->core(c).get();
        core->gbar_arrive_out.bind(&simobject_->gbar_arrive_in);
        simobject_->gbar_resume_out.at(i * cores_per_socket_ + c).bind(&core->gbar_resume_in);
      }
    }

    // Create l2cache

    snprintf(sname, 100, "%s-l2cache", name.c_str());
    // L2 is the LLC iff L2 is enabled and L3 is not.
    l2cache_ = Cache::Create(sname, Cache::Config{
      !VX_CFG_L2_ENABLED,
      log2ceil(VX_CFG_L2_SIZE),// C
      log2ceil(VX_CFG_L2_LINE_SIZE),// L
      log2ceil(VX_CFG_L2_SECTOR_SIZE),// S
      log2ceil(VX_CFG_L1_LINE_SIZE), // W
      log2ceil(VX_CFG_L2_NUM_WAYS),  // A
      log2ceil(VX_CFG_L2_NUM_BANKS), // B
      VX_CFG_XLEN,                   // address bits
      VX_CFG_L2_NUM_REQS,            // request size
      VX_CFG_L2_MEM_PORTS,           // memory ports
      VX_CFG_L2_WRITEBACK,           // write-back
      false,                  // write response
      VX_CFG_L2_MSHR_SIZE,           // mshr size
      VX_CFG_L2_LATENCY,             // pipeline latency
      VX_CFG_L2_REPL_POLICY,         // replacement policy
      (VX_CFG_L2_ENABLED != 0) && (VX_CFG_L3_ENABLED == 0), // is_llc
    });


    // connect l2cache memory interface
    for (uint32_t i = 0; i < VX_CFG_L2_MEM_PORTS; ++i) {
      l2cache_->mem_req_out.at(i).bind(&simobject_->mem_req_out.at(i));
      simobject_->mem_rsp_in.at(i).bind(&l2cache_->mem_rsp_in.at(i));
    }

    // ── L2 fan-in: sockets + cluster-resident gfx caches ────────────────
    // Row 0 = sockets (high priority; TEX/RTU/DXA traffic already merged at
    // each socket's L2-facing arb).
    // Row 1 = ocache (if enabled).
    // Row 2 = rcache (if enabled).
    // The priority arbiter lets sockets win over extension traffic on
    // contention, matching the hardware priority ordering.
#if defined(VX_CFG_EXT_OM_ENABLE) || defined(VX_CFG_EXT_RASTER_ENABLE)
    constexpr uint32_t kL2Rows = 1
        + VX_CFG_EXT_OM_ENABLED + VX_CFG_EXT_RASTER_ENABLED;
    snprintf(sname, 100, "%s-l2arb", name.c_str());
    auto l2arb = MemArbiter::Create(sname, ArbiterType::Priority,
                                    kL2Rows * VX_CFG_L2_NUM_REQS, VX_CFG_L2_NUM_REQS);
    // sockets → row 0
    for (uint32_t i = 0; i < sockets_per_cluster; ++i) {
      for (uint32_t j = 0; j < VX_CFG_L1_MEM_PORTS; ++j) {
        uint32_t port = i * VX_CFG_L1_MEM_PORTS + j;
        sockets_.at(i)->mem_req_out.at(j).bind(&l2arb->ReqIn.at(kL2Rows * port + 0));
        l2arb->RspOut.at(kL2Rows * port + 0).bind(&sockets_.at(i)->mem_rsp_in.at(j));
      }
    }
    // L2 arb outputs → l2cache (after all rows are bound).
    for (uint32_t i = 0; i < VX_CFG_L2_NUM_REQS; ++i) {
      l2arb->ReqOut.at(i).bind(&l2cache_->core_req_in.at(i));
      l2cache_->core_rsp_out.at(i).bind(&l2arb->RspIn.at(i));
    }
#else
    // No cluster-resident gfx caches: direct sockets → L2.
    for (uint32_t i = 0; i < sockets_per_cluster; ++i) {
      for (uint32_t j = 0; j < VX_CFG_L1_MEM_PORTS; ++j) {
        sockets_.at(i)->mem_req_out.at(j).bind(&l2cache_->core_req_in.at(i * VX_CFG_L1_MEM_PORTS + j));
        l2cache_->core_rsp_out.at(i * VX_CFG_L1_MEM_PORTS + j).bind(&sockets_.at(i)->mem_rsp_in.at(j));
      }
    }
#endif // cluster-resident gfx caches

#ifdef VX_CFG_EXT_OM_ENABLE
    // ── Cluster-shared OM engine + ocache ───────────────────────────────
    snprintf(sname, 100, "%s-om-core", name.c_str());
    om_core_ = OmCore::Create(sname, simobject_);

    // ocache: write-through TLM Cache, config from VX_config.toml [ocache] section.
    snprintf(sname, 100, "%s-ocache", name.c_str());
    constexpr uint32_t kOcacheLineSize = VX_CFG_MEM_BLOCK_SIZE;
    constexpr uint32_t kOcacheWordSize = 4;
    constexpr uint32_t kOcacheNumReqs  = VX_CFG_OCACHE_NUM_BANKS;
    constexpr uint32_t kOcacheMemPorts = 1;
    auto ocache = Cache::Create(sname, Cache::Config{
      false,                        // bypass
      log2ceil(VX_CFG_OCACHE_SIZE),        // C
      log2ceil(kOcacheLineSize),    // L
      log2ceil(kOcacheLineSize),    // S (no sectoring)
      log2ceil(kOcacheWordSize),    // W
      log2ceil(VX_CFG_OCACHE_NUM_WAYS),    // A
      log2ceil(VX_CFG_OCACHE_NUM_BANKS),   // B
      VX_CFG_XLEN,                         // address bits
      kOcacheNumReqs,               // request size
      kOcacheMemPorts,              // memory ports
      false,                        // write-back (write-through)
      true,                         // write response (OM holds its same-pixel
                                    // R-M-W interlock until writes COMMIT)
      VX_CFG_OCACHE_MSHR_SIZE,             // mshr size
      2,                            // pipeline latency
      uint8_t(VX_CFG_L2_REPL_POLICY),      // replacement policy
      false,                        // is_llc (OCACHE is auxiliary, not LLC)
    });
    ocache_ = ocache;

    // om_core ↔ ocache (per-port).
    for (uint32_t i = 0; i < kOcacheNumReqs; ++i) {
      om_core_->ocache_req_out.at(i).bind(&ocache->core_req_in.at(i));
      ocache->core_rsp_out.at(i).bind(&om_core_->ocache_rsp_in.at(i));
    }

    // ocache memory side → l2arb.
    constexpr uint32_t kOmRow = 1;
    for (uint32_t i = 0; i < kOcacheMemPorts; ++i) {
      ocache->mem_req_out.at(i).bind(&l2arb->ReqIn.at(kL2Rows * i + kOmRow));
      l2arb->RspOut.at(kL2Rows * i + kOmRow).bind(&ocache->mem_rsp_in.at(i));
    }

    // Per-core SFU.om_req_out (OmUnit decodes onto it) → OmCore::om_req_in[cid],
    // crossing into the cluster domain through a registered stage owned by the
    // sending core's partition. OM has no return value — no rsp channel back
    // to SfuUnit.
    for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
      for (uint32_t c = 0; c < cores_per_socket_; ++c) {
        uint32_t cid = s * cores_per_socket_ + c;
        auto* core = sockets_.at(s)->core(c).get();
        snprintf(sname, 100, "%s-om-slice%d", name.c_str(), cid);
        RegSlice<OmReq>::Ptr slice;
        {
          SimPlatform::DomainScope core_scope(core);
          slice = RegSlice<OmReq>::Create(sname, 1);
        }
        core->sfu_unit()->om_req_out.bind(&slice->In);
        slice->Out.bind(&om_core_->om_req_in.at(cid));
      }
    }
#endif

#ifdef VX_CFG_EXT_RASTER_ENABLE
    // ── Cluster-shared RASTER engine + rcache ───────────────────────────
    snprintf(sname, 100, "%s-raster-core", name.c_str());
    raster_core_ = RasterCore::Create(sname, simobject_);

    // rcache: read-only TLM Cache, config from VX_config.toml [rcache] section.
    snprintf(sname, 100, "%s-rcache", name.c_str());
    constexpr uint32_t kRcacheLineSize = VX_CFG_MEM_BLOCK_SIZE;
    constexpr uint32_t kRcacheWordSize = 4;
    constexpr uint32_t kRcacheNumReqs  = VX_CFG_RCACHE_NUM_BANKS;
    constexpr uint32_t kRcacheMemPorts = 1;
    auto rcache = Cache::Create(sname, Cache::Config{
      false,                        // bypass
      log2ceil(VX_CFG_RCACHE_SIZE),        // C
      log2ceil(kRcacheLineSize),    // L
      log2ceil(kRcacheLineSize),    // S (no sectoring)
      log2ceil(kRcacheWordSize),    // W
      log2ceil(VX_CFG_RCACHE_NUM_WAYS),    // A
      log2ceil(VX_CFG_RCACHE_NUM_BANKS),   // B
      VX_CFG_XLEN,                         // address bits
      kRcacheNumReqs,               // request size
      kRcacheMemPorts,              // memory ports
      false,                        // write-back (read-only)
      false,                        // write response
      VX_CFG_RCACHE_MSHR_SIZE,             // mshr size
      2,                            // pipeline latency
      uint8_t(VX_CFG_L2_REPL_POLICY),      // replacement policy
      false,                        // is_llc (RCACHE is auxiliary, not LLC)
    });
    rcache_ = rcache;

    // raster_core ↔ rcache (per-port).
    for (uint32_t i = 0; i < kRcacheNumReqs; ++i) {
      raster_core_->rcache_req_out.at(i).bind(&rcache->core_req_in.at(i));
      rcache->core_rsp_out.at(i).bind(&raster_core_->rcache_rsp_in.at(i));
    }

    // rcache memory side → l2arb.
    constexpr uint32_t kRasterRow = 1 + VX_CFG_EXT_OM_ENABLED;
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
    // Both bus directions cross the core <-> cluster boundary through
    // registered stages owned by their sending side: requests core-side,
    // responses cluster-side.
    for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
      for (uint32_t c = 0; c < cores_per_socket_; ++c) {
        uint32_t cid = s * cores_per_socket_ + c;
        auto* core = sockets_.at(s)->core(c).get();
        auto sfu = core->sfu_unit();
        snprintf(sname, 100, "%s-raster-req-slice%d", name.c_str(), cid);
        RegSlice<RasterReq>::Ptr req_slice;
        {
          SimPlatform::DomainScope core_scope(core);
          req_slice = RegSlice<RasterReq>::Create(sname, 1);
        }
        sfu->raster_req_out.bind(&req_slice->In);
        req_slice->Out.bind(&raster_bus->ReqIn.at(cid));
        snprintf(sname, 100, "%s-raster-rsp-slice%d", name.c_str(), cid);
        auto rsp_slice = RegSlice<RasterRsp>::Create(sname, 1);
        raster_bus->RspOut.at(cid).bind(&rsp_slice->In);
        rsp_slice->Out.bind(&sfu->raster_rsp_in);
        raster_core_->fwd_arm_out.at(cid).bind(&core->fwd_arm_in);
        core->fwd_done_out.bind(&raster_core_->fwd_done_in);
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
            simobject_->gbar_resume_out.at(i).send({bar_id});
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
#ifdef VX_CFG_EXT_RASTER_ENABLE
    perf_stats.raster = raster_core_->perf_stats();
    perf_stats.rcache = rcache_->perf_stats();
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
    perf_stats.om     = om_core_->perf_stats();
    perf_stats.ocache = ocache_->perf_stats();
#endif
#if defined(VX_CFG_EXT_DXA_ENABLE) || defined(VX_CFG_EXT_TEX_ENABLE) || defined(VX_CFG_EXT_RTU_ENABLE)
    // Socket-resident units (TEX/RTU/DXA) aggregate across sockets.
    for (auto& socket : sockets_) {
      auto socket_perf = socket->perf_stats();
#ifdef VX_CFG_EXT_DXA_ENABLE
      perf_stats.dxa += socket_perf.dxa;
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
      perf_stats.tex    += socket_perf.tex;
      perf_stats.tcache += socket_perf.tcache;
#endif
#ifdef VX_CFG_EXT_RTU_ENABLE
      perf_stats.rtu     += socket_perf.rtu;
      perf_stats.rtcache += socket_perf.rtcache;
#endif
    }
#endif
    return perf_stats;
  }

  int dcr_write(uint32_t addr, uint32_t value) {
#ifdef VX_CFG_EXT_OM_ENABLE
    if (addr >= VX_DCR_OM_STATE_BEGIN && addr < VX_DCR_OM_STATE_END) {
#ifdef VX_CFG_EXT_RASTER_ENABLE
      // The depth buffer is shared with the raster early-Z stage; let it snoop
      // the depth config (zbuf addr/pitch, func, early-Z gate).
      raster_core_->om_dcr_snoop(addr, value);
#endif
      return om_core_->dcr_write(addr, value);
    }
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
    // RASTER_FRAG_* (fragment-shader dispatch descriptor): capture the
    // entry/param halves and latch the assembled 64-bit values into RasterCore
    // (descriptor only — the frame is armed by the delegated launch's
    // frame_kick, not by these writes). Do NOT forward these to dcr_write —
    // that path calls reset_load_state(), which is undesirable for a
    // descriptor that only names the FS to launch.
    if (addr == VX_DCR_RASTER_FRAG_ENTRY_LO) {
      frag_entry_ = (frag_entry_ & ~uint64_t(0xffffffff)) | value;
      return 0;
    }
    if (addr == VX_DCR_RASTER_FRAG_ENTRY_HI) {
      frag_entry_ = (frag_entry_ & uint64_t(0xffffffff)) | (uint64_t(value) << 32);
      return 0;
    }
    if (addr == VX_DCR_RASTER_FRAG_PARAM_LO) {
      frag_param_ = (frag_param_ & ~uint64_t(0xffffffff)) | value;
      return 0;
    }
    if (addr == VX_DCR_RASTER_FRAG_PARAM_HI) {
      frag_param_ = (frag_param_ & uint64_t(0xffffffff)) | (uint64_t(value) << 32);
      raster_core_->set_frag_descriptor(frag_entry_, frag_param_);
      return 0;
    }
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

  void icache_flush_begin() {
    for (auto& socket : sockets_) {
      socket->icache_flush_begin();
    }
  }

  bool icache_flush_done() const {
    for (auto& socket : sockets_) {
      if (!socket->icache_flush_done()) return false;
    }
    return true;
  }

#ifdef VX_CFG_EXT_TEX_ENABLE
  void tcache_flush_begin() {
    for (auto& socket : sockets_) {
      socket->tcache_flush_begin();
    }
  }
  bool tcache_flush_done() const {
    for (auto& socket : sockets_) {
      if (!socket->tcache_flush_done()) return false;
    }
    return true;
  }
#endif

#ifdef VX_CFG_EXT_RASTER_ENABLE
  void rcache_flush_begin() { rcache_->flush_begin(); }
  bool rcache_flush_done() const { return rcache_->flush_done(); }
#endif

#ifdef VX_CFG_EXT_OM_ENABLE
  void ocache_flush_begin() { ocache_->flush_begin(); }
  bool ocache_flush_done() const { return ocache_->flush_done(); }
#endif

#ifdef VX_CFG_EXT_RTU_ENABLE
  void rtcache_flush_begin() {
    for (auto& socket : sockets_) {
      socket->rtcache_flush_begin();
    }
  }
  bool rtcache_flush_done() const {
    for (auto& socket : sockets_) {
      if (!socket->rtcache_flush_done()) return false;
    }
    return true;
  }
#endif

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

#ifdef VX_CFG_EXT_RASTER_ENABLE
  RasterCore::Ptr& raster_core() { return raster_core_; }
#endif

private:
  Cluster*                    simobject_;
  std::vector<Socket::Ptr>    sockets_;
  std::vector<core_barrier_t> gbarriers_;
  Cache::Ptr                  l2cache_;
  uint32_t                    cores_per_socket_;
#ifdef VX_CFG_EXT_OM_ENABLE
  OmCore::Ptr                 om_core_;
  Cache::Ptr                  ocache_;
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
  RasterCore::Ptr             raster_core_;
  Cache::Ptr                  rcache_;
  RasterBusArbiter::Ptr       raster_bus_arb_;
  // RASTER_FRAG_* descriptor halves, assembled across the 4 DCR writes.
  uint64_t                    frag_entry_ = 0;
  uint64_t                    frag_param_ = 0;
#endif
};

///////////////////////////////////////////////////////////////////////////////

Cluster::Cluster(const SimContext& ctx,
                 const char* name,
                 uint32_t cluster_id,
                 ProcessorImpl* processor)
  : SimObject(ctx, name)
  , mem_req_out(VX_CFG_L2_MEM_PORTS, this)
  , mem_rsp_in(VX_CFG_L2_MEM_PORTS, this)
  , gbar_arrive_in(this)
  , gbar_resume_out(NUM_SOCKETS * VX_CFG_SOCKET_SIZE, this)
  , cluster_id_(cluster_id)
  , processor_(processor)
  , impl_(new Impl(this))
{
  gbar_arrive_in.bind(this, &Cluster::on_gbar_arrive);
}

Cluster::~Cluster() {
  delete impl_;
}

void Cluster::on_reset() {
  impl_->reset();
}

bool Cluster::running() const {
  return impl_->running();
}

int Cluster::get_exitcode() const {
  return impl_->get_exitcode();
}

void Cluster::on_gbar_arrive(const GbarArrive& msg) {
  impl_->global_barrier_arrive(msg.bar_id, msg.count, msg.core_id);
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

void Cluster::icache_flush_begin() {
  impl_->icache_flush_begin();
}

bool Cluster::icache_flush_done() const {
  return impl_->icache_flush_done();
}

#ifdef VX_CFG_EXT_TEX_ENABLE
void Cluster::tcache_flush_begin() { impl_->tcache_flush_begin(); }
bool Cluster::tcache_flush_done() const { return impl_->tcache_flush_done(); }
#endif

#ifdef VX_CFG_EXT_RASTER_ENABLE
void Cluster::rcache_flush_begin() { impl_->rcache_flush_begin(); }
bool Cluster::rcache_flush_done() const { return impl_->rcache_flush_done(); }
#endif

#ifdef VX_CFG_EXT_OM_ENABLE
void Cluster::ocache_flush_begin() { impl_->ocache_flush_begin(); }
bool Cluster::ocache_flush_done() const { return impl_->ocache_flush_done(); }
#endif

#ifdef VX_CFG_EXT_RTU_ENABLE
void Cluster::rtcache_flush_begin() { impl_->rtcache_flush_begin(); }
bool Cluster::rtcache_flush_done() const { return impl_->rtcache_flush_done(); }
#endif

void Cluster::l2_flush_begin() {
  impl_->l2_flush_begin();
}

bool Cluster::l2_flush_done() const {
  return impl_->l2_flush_done();
}

#ifdef VX_CFG_EXT_RASTER_ENABLE
RasterCore::Ptr& Cluster::raster_core() {
  return impl_->raster_core();
}
#endif

