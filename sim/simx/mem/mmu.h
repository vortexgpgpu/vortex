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
#include <optional>
#include <queue>
#include <vector>
#include <mem.h>
#include "../types.h"
#include "mmu_tlb.h"
#include "tlb_types.h"

namespace vortex {

// Per-core L1 TLB stage. Sits on the per-core dcache (or icache) request
// path between the upstream (mem_unit/coalescer/fetch) and the downstream
// cache port. Hits translate in place; misses park in a small VPN-dedup
// miss station and are resolved by the shared cluster TLB/PTW over the
// tlb_miss_out/tlb_fill_in channels, so other requests keep flowing
// (hit-under-miss). Parked requests replay in arrival order per port.
//
// Architecture (per port p):
//   ReqIn[p]  --> [bypass | TLB lookup | miss station] --> ReqOut[p]
//   RspOut[p] <-- (passthrough)                        <-- RspIn[p]
class Mmu : public SimObject<Mmu> {
public:
  using Ptr = std::shared_ptr<Mmu>;

  // Upstream side (LSU/coalescer/fetch).
  std::vector<SimChannel<MemReq>> ReqIn;
  std::vector<SimChannel<MemRsp>> RspOut;

  // Downstream side (cache cluster).
  std::vector<SimChannel<MemReq>> ReqOut;
  std::vector<SimChannel<MemRsp>> RspIn;

  // Miss/fill link to the shared cluster TLB.
  SimChannel<TlbReq> TlbMissOut;
  SimChannel<TlbRsp> TlbFillIn;

  Mmu(const SimContext& ctx,
      const char* name,
      uint32_t num_ports,
      uint32_t tlb_size,
      uint32_t client_id,
      bool exec_side,
      bool rsp_reg = false,
      bool write_enable = true,
      bool park_reg = false,
      bool req_skid = false);

  ~Mmu();

  // SATP CSR write — invoked from CsrUnit on `csrw satp`. Flushes the
  // TLB on change (sfence.vma semantics).
  void set_satp(uint64_t satp);

  // True when no request is parked, no miss is outstanding, and no fault
  // report is queued. Parked work holds no channel packet, so quiescence
  // is not visible to the platform without it.
  bool empty() const;

  // Perf counter accessors.
  uint64_t tlb_reads()    const { return tlb_.reads(); }
  uint64_t tlb_hits()     const { return tlb_.hits(); }
  uint64_t tlb_misses()   const { return tlb_.misses(); }
  uint64_t tlb_evictions()const { return tlb_.evictions(); }

protected:
  void on_reset();
  void on_tick();

private:
  struct MshrEntry {
    bool     valid = false;
    bool     issued = false;
    uint64_t vpn = 0;
    // Freed after a good fill: a registered park whose lane probe saw this
    // entry re-arms it from the kept translation instead of walking again.
    bool     retired = false;
    // Re-armed from its kept translation: drains, as if filled, from this cycle.
    bool     rearmed = false;
    uint64_t rearm_cycle = 0;
    uint64_t r_ppn = 0;
    uint32_t r_flags = 0;
    uint32_t r_level = 0;
    std::vector<std::pair<uint32_t, MemReq>> parked;  // (port, request)
  };

  bool needs_translation(const MemReq& req) const;
  bool port_parked(uint32_t port) const;
  bool needs_kill_rsp(const MemReq& req) const;
  int  mshr_find(uint64_t vpn) const;
  int  mshr_alloc(uint64_t vpn);
  int  retired_find(uint64_t vpn) const;
  void replay_or_kill(uint32_t port, const MemReq& req, uint64_t ppn,
                      uint32_t flags, uint32_t level);
  void stage_drain();
  void retire(int id, uint64_t ppn, uint32_t flags, uint32_t level);
  TlbAccess access_of(const MemReq& req) const;
  void kill_request(uint32_t port, const MemReq& req);
  void report_fault(const MemReq& req);

  uint32_t                 num_ports_;
  uint32_t                 client_id_;
  bool                     exec_side_;
  // Each response leaves a cycle late through its own stage.
  bool                     rsp_reg_;
  // Clients may write: a flush token is ordered behind the port's parks.
  bool                     write_enable_;
  // A registered park: a lane's miss (new or joining) is first held in a
  // one-entry stage and reaches the miss station the next cycle at the
  // earliest; the stage takes a request only while empty, and a lane whose
  // page is staged holds meanwhile.
  bool                     park_reg_;
  struct Staged {
    uint32_t port;
    MemReq   req;
    uint64_t vpn;
    uint64_t cycle;
  };
  std::optional<Staged>    stage_;
  std::unique_ptr<SATP_t>  satp_;
  Tlb                      tlb_;

  std::vector<MshrEntry>          mshr_;
  std::vector<std::queue<MemReq>> replay_;   // per-port, drains before new input
  std::vector<std::queue<MemRsp>> fault_rsp_;// per-port, kills faulted accesses
  // Per-port single slot answering a permission fault on a hit; a lane whose
  // slot is still full stalls its next such fault.
  std::vector<std::optional<MemRsp>> pf_slot_;
  std::queue<TlbReq>              reports_;  // permission faults awaiting the walker

  // Requests of one VPN that a single miss-station entry can absorb.
  static constexpr size_t MAX_PARKED_PER_ENTRY = 2;

  // A port holds its client's two-entry output buffer; a lane that takes its
  // requests through a skid register adds that entry, and its client then
  // sees the occupancy as of the previous cycle.
  static constexpr uint32_t REQ_IN_SIZE = 2;

  friend class SimObject<Mmu>;
};

} // namespace vortex

#endif // VX_CFG_VM_ENABLE
