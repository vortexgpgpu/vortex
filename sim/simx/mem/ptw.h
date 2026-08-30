// Copyright © 2019-2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#include <VX_types.h>
#include <VX_config.h>

#ifdef VX_CFG_VM_ENABLE

#include <cstdint>
#include <memory>
#include <vector>
#include <mem.h>
#include "../types.h"
#include "tlb_types.h"

namespace vortex {

// Shared per-cluster page-table walker complex: VX_CFG_PTW_NUM_WALKERS
// independent level-counted walkers behind the cluster TLB's MSHR, plus
// a small direct-mapped walk cache of non-leaf lookups. PTE fetches go
// through the cluster cache on a dedicated client channel; responses
// demux by walker index carried in the request tag. Faults (invalid or
// permission-failing PTEs) latch first-fault info and terminate the
// device with an error.
class Ptw : public SimObject<Ptw> {
public:
  using Ptr = std::shared_ptr<Ptw>;

  struct PerfStats {
    uint64_t walks = 0;
    uint64_t walk_latency = 0;
    uint64_t wc_hits = 0;
  };

  struct FaultInfo {
    bool     valid = false;
    uint64_t va = 0;
    uint8_t  access = 0;
    bool     amo = false;
  };

  // Miss/fill link to the cluster TLB.
  SimChannel<TlbReq> ReqIn;
  SimChannel<TlbRsp> RspOut;

  // PTE-fetch port into the cluster cache.
  SimChannel<MemReq> MemReqOut;
  SimChannel<MemRsp> MemRspIn;

  Ptw(const SimContext& ctx, const char* name);
  ~Ptw();

  void set_satp(uint64_t satp);

  // Clears the walk cache.
  void flush_walk_cache();

  // Drops the latched fault report once the host has read it.
  void clear_fault();

  bool busy() const;

  const PerfStats& perf_stats() const { return perf_; }
  const FaultInfo& fault_info() const { return fault_; }

protected:
  void on_reset();
  void on_tick();

private:
  enum WalkerState { W_IDLE, W_REQ, W_WAIT, W_FILL };

  struct Walker {
    WalkerState state = W_IDLE;
    uint64_t vpn = 0;
    TlbAccess access = TlbAccess::Read;
    bool     amo = false;
    uint32_t mshr_slot = 0;
    uint8_t  level = 0;
    uint64_t cur_ppn = 0;
    uint64_t pte_addr = 0;
    uint64_t final_ppn = 0;
    uint8_t  flags = 0;
    uint8_t  leaf_level = 0;
    bool     fault = false;
    uint64_t start_cycle = 0;
  };

  struct WcEntry {
    bool     valid = false;
    uint64_t tag = 0;
    uint64_t table_ppn = 0;
  };

  void on_pte_response(Walker& w, const MemRsp& rsp);
  void raise_fault(const Walker& w);

  std::unique_ptr<SATP_t> satp_;
  std::vector<Walker>     walkers_;
  std::vector<WcEntry>    walk_cache_;
  FaultInfo               fault_;
  PerfStats               perf_;

  friend class SimObject<Ptw>;
};

} // namespace vortex

#endif // VX_CFG_VM_ENABLE
