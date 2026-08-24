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
#include "ptw.h"

namespace vortex {

// Per-core MMU SimObject. Sits on the per-core dcache (or icache) request
// path between the upstream (mem_unit/coalescer/lsu_dcache_adapter) and the
// downstream cache port. The banked TLB translates inline on a hit; a miss
// parks the access on its bank and sends a walk request to the shared
// device-level Ptw, identified by the bank index. One outstanding walk per
// bank; hits in other banks proceed meanwhile.
// Mirrors hw/rtl/mem/VX_mmu.sv + VX_mmu_tlb.sv + VX_mmu_tlb_bank.sv.
class Mmu : public SimObject<Mmu> {
public:
  using Ptr = std::shared_ptr<Mmu>;

  // Upstream side (LSU/coalescer/fetch).
  std::vector<SimChannel<MemReq>> ReqIn;
  std::vector<SimChannel<MemRsp>> RspOut;

  // Downstream side (cache cluster).
  std::vector<SimChannel<MemReq>> ReqOut;
  std::vector<SimChannel<MemRsp>> RspIn;

  // Shared page-table walker.
  SimChannel<PtwReq> PtwReqOut;
  SimChannel<PtwRsp> PtwRspIn;

  Mmu(const SimContext& ctx,
      const char* name,
      uint32_t num_ports,
      uint32_t num_banks = 1);

  ~Mmu();

  // SATP CSR write — invoked from CsrUnit on `csrw satp`. Flushes the
  // TLB when the value changes (sfence.vma semantics).
  void set_satp(uint64_t satp);

  // Invalidate the TLB (DCR cache-flush path).
  void flush();

  // Perf counter accessors.
  uint64_t tlb_reads()    const { return tlb_.reads(); }
  uint64_t tlb_hits()     const { return tlb_.hits(); }
  uint64_t tlb_misses()   const { return tlb_.misses(); }
  uint64_t tlb_evictions()const { return tlb_.evictions(); }

protected:
  void on_reset();
  void on_tick();

private:
  // Round-trip cost of a translated access through the RTL lookup pipeline:
  // per-lane elastic buffer -> lane/bank crossbar -> banked CAM lookup ->
  // gather crossbar, plus the response-side elastic buffer (charged here on
  // the request instead of the response path). Calibrated against rtlsim.
  static constexpr uint64_t TRANSLATE_LATENCY = 5;

  bool needs_translation(uint64_t addr) const;

  // One parked access per TLB bank while its walk is in flight.
  struct BankMiss {
    enum State { IDLE, WALK_REQ, WALK_WAIT, REPLAY };
    State    state = IDLE;
    MemReq   req;
    uint32_t port = 0;
    PtwRsp   rsp;
  };

  uint32_t                num_ports_;
  uint32_t                num_banks_;
  std::unique_ptr<SATP_t> satp_;
  Tlb                     tlb_;
  std::vector<BankMiss>   banks_;
  uint32_t                miss_rr_ = 0;
  uint32_t                port_rr_ = 0;

  friend class SimObject<Mmu>;
};

} // namespace vortex

#endif // VX_CFG_VM_ENABLE
