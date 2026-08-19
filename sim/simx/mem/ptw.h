// Copyright © 2019-2025
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
#include <simobject.h>
#include "../types.h"

namespace vortex {

// TLB miss request / fill response between the per-core MMUs and the shared
// page-table walker. Mirrors hw/rtl/mem/VX_ptw_bus_if.sv; `tag` is owned by
// the requesting MMU (its TLB bank index).
struct PtwReq {
  uint64_t vpn = 0;
  uint64_t root_ppn = 0;
  uint32_t tag = 0;
};

struct PtwRsp {
  uint64_t ppn = 0;        // 4 KB-resolved
  uint8_t  level = 0;
  uint8_t  flags = 0;
  bool     fault = false;
  uint32_t tag = 0;
};

// Device-level shared page-table walker: one instance per processor, fed by
// every core MMU. Up to VX_CFG_PTW_NUM_WALKERS walks proceed concurrently;
// PTE fetches go out on a dedicated L3 port. Non-leaf entries of the upper
// levels are cached in direct-mapped page-walk caches so a walk can start
// one (Sv32) or two (Sv39) levels below the root.
// Mirrors hw/rtl/mem/VX_mmu_ptw.sv + VX_mmu_pwc.sv.
class Ptw : public SimObject<Ptw> {
public:
  using Ptr = std::shared_ptr<Ptw>;

  struct PerfStats {
    uint64_t walks = 0;
    uint64_t latency = 0;      // sum of per-walk latencies
    uint64_t pwc1_hits = 0;
    uint64_t pwc1_misses = 0;
    uint64_t pwc2_hits = 0;
    uint64_t pwc2_misses = 0;
  };

  // One request/response pair per client MMU.
  std::vector<SimChannel<PtwReq>> ReqIn;
  std::vector<SimChannel<PtwRsp>> RspOut;

  // Dedicated L3 core port.
  SimChannel<MemReq> MemReqOut;
  SimChannel<MemRsp> MemRspIn;

  Ptw(const SimContext& ctx, const char* name, uint32_t num_clients);
  ~Ptw();

  // Drop the page-walk caches (page tables are about to change).
  void flush();

  const PerfStats& perf_stats() const { return perf_; }

protected:
  void on_reset();
  void on_tick();

private:
  struct Slot {
    enum State { IDLE, MEM_REQ, MEM_RSP, DONE };
    State    state = IDLE;
    uint64_t vpn = 0;
    uint64_t root_ppn = 0;
    uint64_t cur_ppn = 0;    // table being walked, then the leaf PPN
    uint8_t  level = 0;
    uint8_t  flags = 0;
    bool     fault = false;
    uint32_t client = 0;
    uint32_t tag = 0;
    uint64_t start_cycle = 0;
  };

  struct PwcEntry {
    bool     valid = false;
    uint64_t key = 0;
    uint64_t ppn = 0;
  };

  bool pwc_lookup(const std::vector<PwcEntry>& pwc, uint64_t key, uint64_t* ppn) const;
  void pwc_fill(std::vector<PwcEntry>& pwc, uint64_t key, uint64_t ppn);
  uint64_t vpn_slice(uint64_t vpn, uint32_t level) const;

  uint32_t num_clients_;
  std::vector<Slot> slots_;
  std::vector<PwcEntry> pwc1_;
  std::vector<PwcEntry> pwc2_;
  uint32_t client_rr_ = 0;
  uint32_t mem_rr_ = 0;
  uint32_t done_rr_ = 0;
  PerfStats perf_;

  friend class SimObject<Ptw>;
};

} // namespace vortex

#endif // VX_CFG_VM_ENABLE
