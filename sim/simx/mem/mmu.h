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
#include <memory>
#include <simobject.h>
#include <mem.h>
#include "../types.h"
#include "mmu_tlb.h"

namespace vortex {

// Per-core MMU SimObject. Sits on the per-core dcache (or icache) request
// path between the upstream (mem_unit/coalescer/lsu_dcache_adapter) and the
// downstream cache port.
//
// PTW PTE fetches go through this object's own downstream MemReq channel —
// i.e. through the cache hierarchy — exactly like a regular load.
//
// Architecture (per port p):
//   ReqIn[p]  --> [bypass | TLB lookup | PTW miss queue] --> ReqOut[p]
//   RspOut[p] <-- [filter PTW responses, forward rest]    <-- RspIn[p]
//
// PTW responses are distinguished from regular responses by a marker
// bit on the tag (`PTW_TAG_MARKER`) — the upstream caller never sees
// these because the MMU consumes them and walks the table.
class Mmu : public SimObject<Mmu> {
public:
  using Ptr = std::shared_ptr<Mmu>;

  // PTW marker bit on MemReq/MemRsp tag — distinguishes PTW PTE
  // fetches from regular upstream traffic on the shared dcache port.
  // It must sit above every real LSU requestor tag (those are small —
  // bounded by the coalescer/LSUQ depth) yet survive the dcache's
  // non-cacheable bypass packing, which multiplexes the requestor port
  // into the low bits with `tag = (tag << log2_num_inputs) | req_id` in
  // a uint32_t MemReq::tag. Bit 31 overflows that shift (the marker is
  // lost, so the PTE response is misrouted upstream and the PTW FSM
  // hangs); bit 24 leaves 16M of real-tag headroom below it and 7 bits
  // of shift headroom above it.
  static constexpr uint32_t PTW_TAG_MARKER = 1u << 24;

  // Upstream side (LSU/coalescer/fetch).
  std::vector<SimChannel<MemReq>> ReqIn;
  std::vector<SimChannel<MemRsp>> RspOut;

  // Downstream side (cache cluster).
  std::vector<SimChannel<MemReq>> ReqOut;
  std::vector<SimChannel<MemRsp>> RspIn;

  Mmu(const SimContext& ctx,
      const char* name,
      uint32_t num_ports);

  ~Mmu();

  // SATP CSR write — invoked from CsrUnit on `csrw satp`. Flushes the
  // TLB on change (sfence.vma semantics).
  void set_satp(uint64_t satp);

  // Perf counter accessors.
  uint64_t tlb_reads()    const { return tlb_.reads(); }
  uint64_t tlb_hits()     const { return tlb_.hits(); }
  uint64_t tlb_misses()   const { return tlb_.misses(); }
  uint64_t tlb_evictions()const { return tlb_.evictions(); }
  uint64_t ptw_walks()    const { return walks_; }
  uint64_t ptw_latency()  const { return walk_latency_; }

protected:
  void on_reset();
  void on_tick();

private:
  bool needs_translation(uint64_t addr) const;

  // PTE address for a given level given the current SATP and walk VA.
  uint64_t pte_addr(uint64_t base_ppn, uint64_t vpn_idx) const {
    return (base_ppn * VX_VM_PT_SIZE) + (vpn_idx * VX_VM_PTE_SIZE);
  }

  void start_ptw(uint64_t va, ACCESS_TYPE type, MemReq orig, uint32_t port);
  void on_ptw_response(const MemRsp& rsp);
  void drive_ptw();

  uint32_t                 num_ports_;
  std::unique_ptr<SATP_t>  satp_;
  Tlb                      tlb_;

  // PTW FSM. One walk in flight at a time.
  // The walk is VX_VM_PT_LEVEL-deep — Sv32 (2 levels) and Sv39 (3) — and
  // is driven by a level counter rather than per-level states.
  enum PtwState {
    PTW_IDLE,
    PTW_REQ,       // need to emit the current level's PTE fetch
    PTW_WAIT,      // waiting for the current level's PTE response
    PTW_FILL       // ready to fill TLB and replay
  };

  PtwState   ptw_state_     = PTW_IDLE;
  uint64_t   ptw_vaddr_     = 0;
  ACCESS_TYPE ptw_type_     = ACCESS_TYPE::LOAD;
  uint64_t   ptw_pte_addr_  = 0;     // address of the most recent PTE fetch
  uint64_t   ptw_cur_ppn_   = 0;     // base PPN of the page table at ptw_level_
  uint8_t    ptw_level_     = 0;     // current walk level: VX_VM_PT_LEVEL-1 .. 0
  uint64_t   ptw_final_ppn_ = 0;
  uint8_t    ptw_flags_     = 0;
  uint8_t    ptw_leaf_level_= 0;     // leaf level: 0 = 4KB, 1 = mega, 2 = giga
  MemReq     ptw_orig_req_;
  uint32_t   ptw_orig_port_ = 0;

  // Perf
  uint64_t walks_         = 0;
  uint64_t walk_latency_  = 0;
  uint64_t walk_start_cyc_= 0;

  friend class SimObject<Mmu>;
};

} // namespace vortex

#endif // VX_CFG_VM_ENABLE
