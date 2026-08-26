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
#ifdef VX_CFG_EXT_DXA_ENABLE
#include "dxa_core.h"
#include "sfu_unit.h"
#endif
#ifdef VX_CFG_EXT_DTCU_CLUSTER_ENABLE
#include "dtcu_tma.h" // complete DtcuTma type (cluster binds tma()->mem_req_out to L2)
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
#include "tex_core.h"
#include "tex_unit.h"
#include "sfu_unit.h"
#endif
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
#ifdef VX_CFG_EXT_RTU_ENABLE
#include "rtu_core.h"
#include "rtu_unit.h"
#include "sfu_unit.h"
#endif

#ifdef VX_CFG_EXT_DTCU_CLUSTER_ENABLE
// What the engine actually needs is a shared cache BELOW the per-socket dcache -- not L2
// specifically. The engine writes the completion flag out its own port, which never
// touches any socket's dcache, while a consumer reads it with an AMO that stops at the
// LAST-LEVEL cache. Those two meet iff the dcache is not itself the LLC, and
// socket.cpp:112 makes the dcache the LLC exactly when both L2 and L3 are off. Either
// one suffices: with L2 off but L3 on, the cluster's l2cache_ is a pure pass-through
// arbiter (cache.cpp:1270) and both the flag and the AMO resolve at L3.
//
// This is the cluster-side mirror of the assert in socket.cpp. Without it a cluster-only
// build (-DVX_CFG_EXT_DTCU_SOCKET_DISABLE, and L2 defaults OFF) compiled clean and then
// spun forever on a flag that could never arrive.
static_assert((VX_CFG_L2_ENABLED != 0) || (VX_CFG_L3_ENABLED != 0),
              "DTCU_cluster requires L2 or L3: without one the socket dcache is the LLC, "
              "so the engine's completion store and a consumer's AMO never meet");
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

    // ── L2 fan-in: sockets + optional extension engines/caches ─────────
    // Row 0 = sockets (high priority).
    // Row 1 = DXA GMEM               (if enabled).
    // Row 2 = DTCU_cluster TMA       (if enabled).
    // Row 3 = DTCU_socket TMA fan-in (if enabled) -- ALL N socket engines share this
    //         ONE row through a private round-robin arbiter, so the row count stays
    //         independent of NUM_SOCKETS.
    // Row 4 = tcache                 (if enabled).
    // Row 5 = ocache                 (if enabled).
    // Row 6 = rcache                 (if enabled).
    // Row 7 = RTU dcache             (if enabled).
    // The priority arbiter lets sockets win over extension traffic on contention,
    // matching the RTL `VX_mem_arb` priority ordering. The two DTCU rows sit directly
    // behind DXA so an engine outranks graphics traffic, which is the ordering 1.6c
    // rests on.
#if defined(VX_CFG_EXT_DXA_ENABLE) || defined(VX_CFG_EXT_DTCU_CLUSTER_ENABLE) || defined(VX_CFG_EXT_DTCU_SOCKET_ENABLE) || defined(VX_CFG_EXT_TEX_ENABLE) || defined(VX_CFG_EXT_OM_ENABLE) || defined(VX_CFG_EXT_RASTER_ENABLE) || defined(VX_CFG_EXT_RTU_ENABLE)
    constexpr uint32_t kL2RowsUsed = 1
        + VX_CFG_EXT_DXA_ENABLED + VX_CFG_EXT_DTCU_CLUSTER_ENABLED + VX_CFG_EXT_DTCU_SOCKET_ENABLED
        + VX_CFG_EXT_TEX_ENABLED + VX_CFG_EXT_OM_ENABLED + VX_CFG_EXT_RASTER_ENABLED
        + VX_CFG_EXT_RTU_ENABLED;
    constexpr uint32_t kDxaRow = 1;
    constexpr uint32_t kDtcuClusterRow = kDxaRow + VX_CFG_EXT_DXA_ENABLED;
    constexpr uint32_t kDtcuSocketRow =
        kDtcuClusterRow + VX_CFG_EXT_DTCU_CLUSTER_ENABLED;
    // TxArbiter groups its inputs in blocks of (1 << log2ceil(num_inputs/num_outputs))
    // and only ever serves input i from output i/R (types.h TxArbiter::on_tick). The
    // `kL2Rows * port + row` indexing used throughout this constructor is therefore
    // only correct when that block size EQUALS the row count, so round up to a power
    // of two. Padded rows stay unbound; the arbiter only grants non-empty inputs, so
    // they are never selected. Without this, rows and ports interleave wrongly: two
    // socket ports share one arbiter output while the top L2 request lanes go
    // permanently undriven -- functionally invisible (the L2 routes by address) but a
    // real contention-modelling error.
    constexpr uint32_t kL2Rows = 1u << log2ceil(kL2RowsUsed);
    static_assert(kL2Rows * VX_CFG_L2_NUM_REQS <= 64,
                  "l2arb exceeds TxArbiter's 64-input limit; reduce rows or L2_NUM_REQS");
    snprintf(sname, 100, "%s-l2arb", name.c_str());
    auto l2arb = MemArbiter::Create(sname, ArbiterType::Priority,
                                    kL2Rows * VX_CFG_L2_NUM_REQS, VX_CFG_L2_NUM_REQS);
#if VX_CFG_L2_ARB_ENGINE_BYPASS_LIMIT != 0
    constexpr uint64_t kEngineRows =
        (uint64_t(VX_CFG_EXT_DXA_ENABLED) << kDxaRow)
      | (uint64_t(VX_CFG_EXT_DTCU_CLUSTER_ENABLED) << kDtcuClusterRow)
      | (uint64_t(VX_CFG_EXT_DTCU_SOCKET_ENABLED) << kDtcuSocketRow);
    l2arb->configure_core_bypass(
        VX_CFG_L2_ARB_ENGINE_BYPASS_LIMIT, kEngineRows,
        [](uint32_t row, const MemReq& req) {
          return row != kDtcuSocketRow || !req.is_write();
        });
#endif
    // sockets → row 0
    for (uint32_t i = 0; i < sockets_per_cluster; ++i) {
      for (uint32_t j = 0; j < VX_CFG_L1_MEM_PORTS; ++j) {
        uint32_t port = i * VX_CFG_L1_MEM_PORTS + j;
        sockets_.at(i)->mem_req_out.at(j).bind(&l2arb->ReqIn.at(kL2Rows * port + 0));
        l2arb->RspOut.at(kL2Rows * port + 0).bind(&sockets_.at(i)->mem_rsp_in.at(j));
      }
    }
#else
    // No L2-sharing extensions: direct sockets → L2.
    for (uint32_t i = 0; i < sockets_per_cluster; ++i) {
      for (uint32_t j = 0; j < VX_CFG_L1_MEM_PORTS; ++j) {
        sockets_.at(i)->mem_req_out.at(j).bind(&l2cache_->core_req_in.at(i * VX_CFG_L1_MEM_PORTS + j));
        l2cache_->core_rsp_out.at(i * VX_CFG_L1_MEM_PORTS + j).bind(&sockets_.at(i)->mem_rsp_in.at(j));
      }
    }
#endif // any L2-sharing extension

#ifdef VX_CFG_EXT_DXA_ENABLE
    // Create DxaCore at cluster scope
    snprintf(sname, 100, "%s-dxa-core", name.c_str());
    dxa_core_ = DxaCore::Create(sname, simobject_);

    // DXA gmem → row 1 of l2arb.
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

    // DxaCore::lmem_req_out[cid] → core's LocalMem.Inputs[port_dxa].
    // A tx_callback on the channel fires barrier_event_release for each
    // DXA-write packet carrying notify_done at the cycle LMEM receives it.
    uint32_t port_dxa = LSU_NUM_REQS;
  #ifdef VX_CFG_EXT_TCU_ENABLE
    port_dxa += 1;
  #endif
    for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
      for (uint32_t c = 0; c < cores_per_socket_; ++c) {
        uint32_t cid = s * cores_per_socket_ + c;
        Core* core = sockets_.at(s)->core(c).get();
        auto& ch = dxa_core_->lmem_req_out.at(cid);
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
    }
#endif

#ifdef VX_CFG_EXT_DTCU_CLUSTER_ENABLE
    // ── Disaggregated tensor core, CLUSTER variant ──────────────────────
    // One engine per cluster, driven by dtensor_cluster_start. Everything it does --
    // operand reads, D stores, the completion flag -- goes out one L2 port, so it has
    // no separate D port and gets a private l2arb row right after DXA. NOT DXA: no
    // per-core SFU dispatch, no LMEM writes; it reads and writes GMEM via TLM.
    snprintf(sname, 100, "%s-dtcu", name.c_str());
    dtcu_ = Dtcu::Create(sname, DTCU_ENGINE_CLUSTER);
    dtcu_->tma()->mem_req_out.bind(&l2arb->ReqIn.at(kL2Rows * 0 + kDtcuClusterRow));
    l2arb->RspOut.at(kL2Rows * 0 + kDtcuClusterRow).bind(&dtcu_->tma()->mem_rsp_in);
#endif

#ifdef VX_CFG_EXT_DTCU_SOCKET_ENABLE
    // ── Disaggregated tensor core, SOCKET variant: shared L2 read port ──
    // The engines themselves live in Socket (they need its dcache for D); only their
    // operand/descriptor READ path arrives here. All N funnel through one private
    // arbiter into a SINGLE l2arb row, because the cluster cannot grow an L2 bypass
    // port per socket -- the sharing is the modelled constraint, not an artifact.
    //
    // Round-robin, not Priority: the engines are peers, and Priority would let socket
    // 0 starve the rest. The arbiter also makes the response path self-routing (it
    // ORs its input index into the tag LSBs on the way down and strips it on the way
    // back), which is why no engine id has to be carried in the tag.
    snprintf(sname, 100, "%s-dtcu-socket-arb", name.c_str());
    auto dtcu_sock_arb = MemArbiter::Create(sname, ArbiterType::RoundRobin,
                                            sockets_per_cluster, 1);
    for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
      sockets_.at(s)->dtcu_mem_req_out.bind(&dtcu_sock_arb->ReqIn.at(s));
      dtcu_sock_arb->RspOut.at(s).bind(&sockets_.at(s)->dtcu_mem_rsp_in);
    }
    // DO NOT PROMOTE THIS ROW ABOVE THE SOCKET ROW. The socket engine's output
    // ordering depends on it, and the dependency is not otherwise expressed anywhere.
    //
    // Its D stores land in the socket's dcache, which is write-through, so they reach
    // L2 by the SOCKET egress path (row 0). Its completion flag leaves on the engine's
    // own read port and reaches L2 by THIS row. Two independent paths, no fence between
    // them -- the only thing stopping the flag from overtaking the data it announces is
    // that l2arb is a PriorityArbiter and PriorityArbiter::grant() returns the lowest
    // requesting index, so row 0 always beats this row. Raising this row for latency
    // (tempting: it is the last row to be granted under sustained core traffic) would
    // let a consumer observe done==1 and then read pre-GEMM bytes.
    //
    // The clean fix is to make strsp mean "acknowledged at the point of coherence"
    // rather than "accepted by the first cache", but that changes upstream semantics in
    // cache.cpp's need_core_rsp, which every level shares. Not worth it while the
    // ordering holds; pinned here so that it cannot be broken silently.
    static_assert(kDtcuSocketRow > 0,
                  "DTCU_socket's L2 row must rank BELOW the socket row (row 0): its "
                  "completion flag would otherwise be able to overtake its own D stores");
    dtcu_sock_arb->ReqOut.at(0).bind(&l2arb->ReqIn.at(kL2Rows * 0 + kDtcuSocketRow));
    l2arb->RspOut.at(kL2Rows * 0 + kDtcuSocketRow).bind(&dtcu_sock_arb->RspIn.at(0));
#endif

#ifdef VX_CFG_EXT_TEX_ENABLE
    // ── Cluster-shared TEX engine + tcache ──────────────────────────────
    snprintf(sname, 100, "%s-tex-core", name.c_str());
    tex_core_ = TexCore::Create(sname, simobject_);

    // tcache: read-only TLM Cache, config from VX_config.toml [tcache] section.
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
    // tcache memory side → l2arb. Row index = kL2Rows-1 if no OM, else
    // kL2Rows-2 (OM occupies the last row when both are present).
    constexpr uint32_t kTexRow = 1 + VX_CFG_EXT_DXA_ENABLED
                               + VX_CFG_EXT_DTCU_CLUSTER_ENABLED + VX_CFG_EXT_DTCU_SOCKET_ENABLED;
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

#if defined(VX_CFG_EXT_DXA_ENABLE) || defined(VX_CFG_EXT_DTCU_CLUSTER_ENABLE) || defined(VX_CFG_EXT_DTCU_SOCKET_ENABLE) || defined(VX_CFG_EXT_TEX_ENABLE) || defined(VX_CFG_EXT_OM_ENABLE) || defined(VX_CFG_EXT_RASTER_ENABLE) || defined(VX_CFG_EXT_RTU_ENABLE)

    // L2 arb outputs → l2cache (after all rows are bound).
    for (uint32_t i = 0; i < VX_CFG_L2_NUM_REQS; ++i) {
      l2arb->ReqOut.at(i).bind(&l2cache_->core_req_in.at(i));
      l2cache_->core_rsp_out.at(i).bind(&l2arb->RspIn.at(i));
    }
#endif

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

    // ocache memory side → l2arb. Row index is sockets + DXA + TEX (if those are present).
    constexpr uint32_t kOmRow = 1 + VX_CFG_EXT_DXA_ENABLED
                              + VX_CFG_EXT_DTCU_CLUSTER_ENABLED + VX_CFG_EXT_DTCU_SOCKET_ENABLED
                              + VX_CFG_EXT_TEX_ENABLED;
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

    // rcache memory side -> l2arb (RASTER is the last USED row before RTU).
    // Spelled out rather than kL2Rows-1: kL2Rows is rounded up to a power of two, so the
    // last index may be a padding row bound to nothing.
    constexpr uint32_t kRasterRow = 1 + VX_CFG_EXT_DXA_ENABLED
                                  + VX_CFG_EXT_DTCU_CLUSTER_ENABLED + VX_CFG_EXT_DTCU_SOCKET_ENABLED
                                  + VX_CFG_EXT_TEX_ENABLED + VX_CFG_EXT_OM_ENABLED;
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

#ifdef VX_CFG_EXT_RTU_ENABLE
    // ── Cluster-shared RTU engine + rtcache (§8.10) ─────────────────────
    snprintf(sname, 100, "%s-rtu-core", name.c_str());
    rtu_core_ = RtuCore::Create(sname, simobject_);

    // rtcache: RTU-private read-only `Cache` instance, mirror of
    // rcache for the raster path. Sits between RtuCore and the L2
    // arb so cross-ray BVH-node reuse (root + upper internals) hits
    // in L1 instead of thrashing L2. Sizing comes from VX_config.toml
    // [rtcache] section.
    snprintf(sname, 100, "%s-rtcache", name.c_str());
    constexpr uint32_t kRtcacheLineSize = VX_CFG_MEM_BLOCK_SIZE;
    constexpr uint32_t kRtcacheWordSize = 4;
    // num_inputs = NUM_RTU_BLOCKS so each RtuCore memory port gets
    // its own cache input lane. The cache's internal bank crossbar
    // arbitrates these inputs onto VX_CFG_RTCACHE_NUM_BANKS banks
    // (default 1, single-bank funnel).
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

    // rtcache memory side → l2arb at the RTU row.
    constexpr uint32_t kRtuRow = 1 + VX_CFG_EXT_DXA_ENABLED
                                   + VX_CFG_EXT_TEX_ENABLED
                                   + VX_CFG_EXT_OM_ENABLED
                                   + VX_CFG_EXT_RASTER_ENABLED;
    for (uint32_t i = 0; i < kRtcacheMemPorts; ++i) {
      rtcache->mem_req_out.at(i).bind(&l2arb->ReqIn.at(kL2Rows * i + kRtuRow));
      l2arb->RspOut.at(kL2Rows * i + kRtuRow).bind(&rtcache->mem_rsp_in.at(i));
    }

    // Cluster-level RtuBus arbiter: NUM_CORES_PER_CLUSTER inputs (one per
    // SfuUnit) → 1 RTU-core lane.
    snprintf(sname, 100, "%s-rtu-bus", name.c_str());
    uint32_t rtu_cores_per_cluster = sockets_per_cluster * cores_per_socket_;
    auto rtu_bus = RtuBusArbiter::Create(sname, ArbiterType::RoundRobin,
                                         rtu_cores_per_cluster, 1);
    rtu_bus_arb_ = rtu_bus;
    for (uint32_t s = 0; s < sockets_per_cluster; ++s) {
      for (uint32_t c = 0; c < cores_per_socket_; ++c) {
        uint32_t cid = s * cores_per_socket_ + c;
        auto sfu = sockets_.at(s)->core(c)->sfu_unit();
        sfu->rtu_req_out.bind(&rtu_bus->ReqIn.at(cid));
        rtu_bus->RspOut.at(cid).bind(&sfu->rtu_rsp_in);
        // §8.6 async ray pool: give each SfuUnit a direct pointer to
        // the cluster's RtuCore so its RtuUnit can call
        // allocate_slot() / free_slot() without going through the bus.
        sfu->set_rtu_core(rtu_core_.get());
      }
    }
    rtu_bus->ReqOut.at(0).bind(&rtu_core_->rtu_req_in.at(0));
    rtu_core_->rtu_rsp_out.at(0).bind(&rtu_bus->RspIn.at(0));
    (void)rtu_cores_per_cluster;
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
#ifdef VX_CFG_EXT_DXA_ENABLE
    perf_stats.dxa = dxa_core_->perf_stats();
#endif
#ifdef VX_CFG_EXT_DTCU_CLUSTER_ENABLE
    perf_stats.dtcu = dtcu_->perf_stats();
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
    perf_stats.tex    = tex_core_->perf_stats();
    perf_stats.tcache = tcache_->perf_stats();
#endif
#ifdef VX_CFG_EXT_RASTER_ENABLE
    perf_stats.raster = raster_core_->perf_stats();
    perf_stats.rcache = rcache_->perf_stats();
#endif
#ifdef VX_CFG_EXT_OM_ENABLE
    perf_stats.om     = om_core_->perf_stats();
    perf_stats.ocache = ocache_->perf_stats();
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
  void tcache_flush_begin() { tcache_->flush_begin(); }
  bool tcache_flush_done() const { return tcache_->flush_done(); }
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
  void rtcache_flush_begin() { rtcache_->flush_begin(); }
  bool rtcache_flush_done() const { return rtcache_->flush_done(); }
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

#ifdef VX_CFG_EXT_DXA_ENABLE
  DxaCore::Ptr& dxa_core() { return dxa_core_; }
#endif

#ifdef VX_CFG_EXT_DTCU_CLUSTER_ENABLE
  Dtcu::Ptr& dtcu() { return dtcu_; }
#endif

#ifdef VX_CFG_EXT_RASTER_ENABLE
  RasterCore::Ptr& raster_core() { return raster_core_; }
#endif

#ifdef VX_CFG_EXT_RTU_ENABLE
  RtuCore::Ptr& rtu_core() { return rtu_core_; }
#endif

private:
  Cluster*                    simobject_;
  std::vector<Socket::Ptr>    sockets_;
  std::vector<core_barrier_t> gbarriers_;
  Cache::Ptr                  l2cache_;
  uint32_t                    cores_per_socket_;
#ifdef VX_CFG_EXT_DXA_ENABLE
  DxaCore::Ptr                dxa_core_;
#endif
#ifdef VX_CFG_EXT_DTCU_CLUSTER_ENABLE
  Dtcu::Ptr                   dtcu_;
#endif
#ifdef VX_CFG_EXT_TEX_ENABLE
  TexCore::Ptr                tex_core_;
  Cache::Ptr                  tcache_;
  TexBusArbiter::Ptr          tex_bus_arb_;
#endif
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
#ifdef VX_CFG_EXT_RTU_ENABLE
  RtuCore::Ptr                rtu_core_;
  Cache::Ptr                  rtcache_;
  RtuBusArbiter::Ptr          rtu_bus_arb_;
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

#ifdef VX_CFG_EXT_DXA_ENABLE
DxaCore::Ptr& Cluster::dxa_core() {
  return impl_->dxa_core();
}
#endif

#ifdef VX_CFG_EXT_DTCU_CLUSTER_ENABLE
Dtcu::Ptr& Cluster::dtcu() {
  return impl_->dtcu();
}
#endif

#ifdef VX_CFG_EXT_RASTER_ENABLE
RasterCore::Ptr& Cluster::raster_core() {
  return impl_->raster_core();
}
#endif

#ifdef VX_CFG_EXT_RTU_ENABLE
RtuCore::Ptr& Cluster::rtu_core() {
  return impl_->rtu_core();
}
#endif
