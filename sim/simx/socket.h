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

#pragma once

#include <simobject.h>
#include "cache.h"
// A placement variant without the master switch would advertise the engine in MISA
// while decode.cpp has no DTCU opcode at all and aborts on it. Caught here rather
// than at run time; socket.h is on every simx translation unit's path.
#if (defined(VX_CFG_EXT_DTCU_SOCKET_ENABLE) || defined(VX_CFG_EXT_DTCU_CLUSTER_ENABLE)) \
    && !defined(VX_CFG_EXT_DTCU_ENABLE)
#error "VX_CFG_EXT_DTCU_{SOCKET,CLUSTER}_ENABLE requires VX_CFG_EXT_DTCU_ENABLE; \
for one variant alone pass the umbrella plus the other variant's _DISABLE"
#endif
#if defined(VX_CFG_EXT_DTCU_ENABLE) && !defined(VX_CFG_EXT_DTCU_SOCKET_ENABLE) \
    && !defined(VX_CFG_EXT_DTCU_CLUSTER_ENABLE)
#error "VX_CFG_EXT_DTCU_ENABLE with both placement variants disabled: no engine would exist"
#endif
#ifdef VX_CFG_EXT_DTCU_SOCKET_ENABLE
#include "dtcu.h"
#endif

namespace vortex {

class Cluster;
class Core;

class Socket : public SimObject<Socket> {
public:
  struct PerfStats {
    Cache::PerfStats icache;
    Cache::PerfStats dcache;
#ifdef VX_CFG_EXT_DTCU_SOCKET_ENABLE
    // Socket-owned DTCU engine. Read under VX_DCR_MPM_CLASS_DTCU_SOCKET; the same
    // value on every core in this socket. Dtcu::PerfStats has no default member
    // initializers, so this field and its assignment in perf_stats() must stay under
    // the SAME macro -- divergent guards leave it indeterminate, not zero.
    Dtcu::PerfStats dtcu;
#endif
  };

  std::vector<SimChannel<MemReq>> mem_req_out;
  std::vector<SimChannel<MemRsp>> mem_rsp_in;

#ifdef VX_CFG_EXT_DTCU_SOCKET_ENABLE
  // DTCU_socket operand/descriptor READ port, headed for the cluster's L2.
  // Deliberately not another element of mem_req_out: that vector carries L1 MISS
  // traffic and its width is assumed by three separate index computations
  // (socket.cpp's l1_arb loop, cluster.cpp's socket row, constants.h's
  // VX_CFG_L2_NUM_REQS), so an extra element would end up unbound and dead rather
  // than erroring.
  SimChannel<MemReq> dtcu_mem_req_out;
  SimChannel<MemRsp> dtcu_mem_rsp_in;
#endif

  Socket(const SimContext& ctx,
         const char* name,
         uint32_t socket_id,
         Cluster* cluster);

  ~Socket();

  uint32_t id() const { return socket_id_; }

  Cluster* cluster() const { return cluster_; }

  bool running() const;

  int get_exitcode() const;

  void global_barrier_arrive(uint32_t bar_id, uint32_t count, uint32_t core_id);

  void global_barrier_resume(uint32_t bar_id, uint32_t core_id);

  PerfStats perf_stats() const;

  int dcr_write(uint32_t addr, uint32_t value);

  int dcr_read(uint32_t addr, uint32_t tag, uint32_t* value);

  std::shared_ptr<Core>& core(uint32_t idx);

#ifdef VX_CFG_EXT_DTCU_SOCKET_ENABLE
  // Socket-owned disaggregated tensor core. sfu_unit.cpp reaches it as
  // core_->socket()->dtcu() for DtcuType::START_SOCKET.
  Dtcu::Ptr& dtcu();
#endif

  // Forwarded cache flush (write-back eviction walk). The walk is a no-op
  // on write-through caches (`Cache::flush_begin` early-exits); forwarding
  // ensures a future write-back config exercises the same code path.
  void dcache_flush_begin();
  bool dcache_flush_done() const;
  void icache_flush_begin();
  bool icache_flush_done() const;

protected:
  void on_reset();
  void on_tick();

private:
  uint32_t socket_id_;
  Cluster* cluster_;

  class Impl;
  Impl* impl_;

  friend class SimObject<Socket>;
};

} // namespace vortex
