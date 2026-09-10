// Copyright © 2019-2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <VX_config.h>

#ifdef VX_CFG_VM_ENABLE

#include <cstdint>
#include <memory>
#include <queue>
#include <vector>
#include "../types.h"
#include "mmu_tlb.h"
#include "tlb_types.h"

namespace vortex {

// Shared per-cluster TLB. Serves miss traffic from every L1 TLB in the
// cluster through a pipelined lookup (one per tick, latency-deep pipe),
// a set-associative main array plus a small fully-associative megapage
// side array, and a VPN-dedup MSHR: concurrent misses on one VPN from
// any clients share a single walk and every attached requester is
// answered by its fill.
class L2Tlb : public SimObject<L2Tlb> {
public:
  using Ptr = std::shared_ptr<L2Tlb>;

  struct PerfStats {
    uint64_t reads = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t mshr_dedups = 0;
  };

  // Client side (one pair per L1 TLB instance, indexed by client_id).
  std::vector<SimChannel<TlbReq>> ReqIn;
  std::vector<SimChannel<TlbRsp>> RspOut;

  // Walker side.
  SimChannel<TlbReq> PtwReqOut;
  SimChannel<TlbRsp> PtwRspIn;

  L2Tlb(const SimContext& ctx, const char* name, uint32_t num_clients);
  ~L2Tlb();


  // True while any lookup, miss, or response is still in flight —
  // folded into the cluster's running() so completion cannot outrun
  // a pending walk.
  // True while any lookup, response, or walk is outstanding. Only the
  // channel traffic is visible to the platform, so this covers the rest.
  bool busy() const;

  // Distinct L1 miss stations one shared entry can serve.
  static constexpr size_t MAX_REQUESTERS_PER_ENTRY = 4;

  const PerfStats& perf_stats() const { return perf_; }

protected:
  void on_reset();
  void on_tick();

private:
  struct Entry {
    bool     valid = false;
    bool     mru = false;
    uint64_t vpn = 0;
    uint64_t ppn = 0;
    uint8_t  flags = 0;
  };

  struct MshrEntry {
    bool      valid = false;
    bool      issued = false;
    uint64_t  vpn = 0;
    TlbAccess access = TlbAccess::Read;  // allocating requester's intent,
    bool      amo = false;               // checked at walk time
    std::vector<std::pair<uint32_t, uint32_t>> requesters;  // (client, slot)
  };

  bool lookup(uint64_t vpn, uint64_t* ppn, uint8_t* flags, uint8_t* level);
  void fill(uint64_t vpn, uint64_t ppn, uint8_t flags, uint8_t level);

  uint32_t num_clients_;
  uint32_t num_sets_;
  uint32_t rr_client_ = 0;

  std::vector<std::vector<Entry>> sets_;   // [set][way], 4KB entries
  Tlb                             mega_;   // megapage side array
  std::vector<MshrEntry>          mshr_;
  // Lookup pipe: entries carry their ready-cycle so a lookup completes
  // VX_CFG_L2_TLB_LATENCY ticks after acceptance, one per tick.
  std::queue<std::pair<uint64_t, TlbReq>> pipe_;
  std::queue<TlbRsp>              rsp_q_;  // fills fan out one per tick

  PerfStats perf_;

  friend class SimObject<L2Tlb>;
};

} // namespace vortex

#endif // VX_CFG_VM_ENABLE
