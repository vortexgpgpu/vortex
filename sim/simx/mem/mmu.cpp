// Copyright © 2019-2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <VX_config.h>

#ifdef VM_ENABLE

#include "mmu.h"
#include "../debug.h"

namespace vortex {

Mmu::Mmu(const SimContext& ctx,
        const char* name,
        uint32_t num_ports)
  : SimObject<Mmu>(ctx, name)
  , ReqIn (num_ports, this)
  , RspOut(num_ports, this)
  , ReqOut(num_ports, this)
  , RspIn (num_ports, this)
  , num_ports_(num_ports)
  , tlb_(TLB_SIZE)
{}

Mmu::~Mmu() = default;

void Mmu::on_reset() {
  ptw_state_ = PTW_IDLE;
  // SATP is set externally via set_satp(); don't clear on simulator reset.
}

void Mmu::set_satp(uint64_t satp) {
  satp_ = std::make_unique<SATP_t>(satp);
  tlb_.flush();  // sfence.vma
}

bool Mmu::needs_translation(uint64_t addr) const {
  (void)addr;
  // The runtime installs identity PTEs at boot for every PA-addressed
  // region (IO MMIO, kernel image, page table, stacks), so any access
  // post-SATP-set walks the page table — there is no longer a need to
  // address-range bypass. The only path that skips translation is one
  // issued before SATP is programmed (BARE mode); this covers the
  // few instruction fetches between reset and the kernel's csrw satp.
  if (!satp_ || satp_->get_mode() == BARE) return false;
  return true;
}

void Mmu::start_ptw(uint64_t va, ACCESS_TYPE type, MemReq orig, uint32_t port) {
  ptw_state_     = PTW_L1_REQ;
  ptw_vaddr_     = va;
  ptw_type_      = type;
  ptw_orig_req_  = orig;
  ptw_orig_port_ = port;
  walk_start_cyc_= SimPlatform::instance().cycles();
  ++walks_;
}

void Mmu::on_ptw_response(const MemRsp& rsp) {
  // Extract the PTE word from the cache-line payload using the recorded
  // PTE address (low bits give the offset within the line).
  uint32_t off_words = (uint32_t)((ptw_pte_addr_ & (MEM_BLOCK_SIZE - 1)) / PTE_SIZE);
  uint64_t pte_bytes = 0;
  if (rsp.data) {
    auto* words = reinterpret_cast<const uint32_t*>(rsp.data->data());
    pte_bytes = words[off_words];
  }
  PTE_t pte(pte_bytes);

  // Validity check (mirrors VX_mmu_ptw + sim/common/mem.cpp::page_table_walk).
  bool invalid = (pte.v == 0) | ((pte.r == 0) & (pte.w == 1));
  bool is_leaf = (pte.r != 0) | (pte.w != 0) | (pte.x != 0);

  if (invalid) {
    // Page fault — for now, abort the simulator with a clear message.
    // TODO: route a page-fault exception back to the LSU.
    std::cerr << "MMU: page fault on PTE at 0x" << std::hex << ptw_pte_addr_
              << " (vaddr 0x" << ptw_vaddr_ << ")" << std::dec << std::endl;
    std::abort();
  }

  switch (ptw_state_) {
  case PTW_L1_WAIT:
    if (is_leaf) {
      // Megapage at L1 — the PTE's PPN encodes the megapage base; the
      // low VPN[0] + page-offset bits come from the original VA.
      ptw_final_ppn_   = pte.ppn;
      ptw_flags_       = pte.flags;
      ptw_leaf_level_  = 1;
      ptw_state_       = PTW_FILL;
    } else {
      ptw_l1_ppn_ = pte.ppn;
      ptw_state_  = PTW_L0_REQ;
    }
    break;
  case PTW_L0_WAIT:
    ptw_final_ppn_  = pte.ppn;
    ptw_flags_      = pte.flags;
    ptw_leaf_level_ = 0;
    ptw_state_      = PTW_FILL;
    break;
  default:
    // Unexpected — ignore.
    break;
  }
}

void Mmu::drive_ptw() {
  // Bits of VA per page-table level. Derived from PT geometry:
  // PT_SIZE / PTE_SIZE = entries per table = 2^VPN_BITS_PER_LEVEL.
  const uint32_t VPN_BITS = log2ceil(PT_SIZE / PTE_SIZE);
  const uint64_t VPN_MASK = (1ULL << VPN_BITS) - 1;
  switch (ptw_state_) {
  case PTW_L1_REQ: {
    uint64_t vpn1 = (ptw_vaddr_ >> (MEM_PAGE_LOG2_SIZE + VPN_BITS)) & VPN_MASK;
    ptw_pte_addr_ = pte_addr(satp_->get_base_ppn(), vpn1);
    MemReq req(ptw_pte_addr_, /*write*/false, AddrType::Global,
               PTW_TAG_MARKER, /*cid*/0, /*uuid*/0);
    if (ReqOut.at(0).try_send(req)) {
      DT(4, this->name() << " ptw L1-req: addr=0x" << std::hex << ptw_pte_addr_ << std::dec);
      ptw_state_ = PTW_L1_WAIT;
    }
    break;
  }
  case PTW_L0_REQ: {
    uint64_t vpn0 = (ptw_vaddr_ >> MEM_PAGE_LOG2_SIZE) & VPN_MASK;
    ptw_pte_addr_ = pte_addr(ptw_l1_ppn_, vpn0);
    MemReq req(ptw_pte_addr_, /*write*/false, AddrType::Global,
               PTW_TAG_MARKER, /*cid*/0, /*uuid*/0);
    if (ReqOut.at(0).try_send(req)) {
      DT(4, this->name() << " ptw L0-req: addr=0x" << std::hex << ptw_pte_addr_ << std::dec);
      ptw_state_ = PTW_L0_WAIT;
    }
    break;
  }
  case PTW_FILL: {
    // Compose the PA. For leaf at L0 the offset is just pgoff; for a
    // leaf at L1 (SV32 megapage), the offset additionally includes
    // VPN[0] from the original VA. PT_SIZE/PTE_SIZE bits per level.
    uint32_t off_bits = MEM_PAGE_LOG2_SIZE +
                        ptw_leaf_level_ * log2ceil(PT_SIZE / PTE_SIZE);
    uint64_t off_mask = (1ULL << off_bits) - 1;
    uint64_t pa_base = (ptw_final_ppn_ << MEM_PAGE_LOG2_SIZE) & ~off_mask;
    uint64_t pa = pa_base | (ptw_vaddr_ & off_mask);
    // Cache the per-4KB sub-page mapping in the TLB. A megapage walk
    // therefore only services the specific 4 KB that triggered the
    // miss; subsequent VAs in the same megapage will re-walk (correct,
    // just less optimal — fine for the rare system regions we identity-map).
    uint64_t vpn = ptw_vaddr_ >> MEM_PAGE_LOG2_SIZE;
    uint64_t ppn_4kb = pa >> MEM_PAGE_LOG2_SIZE;
    tlb_.fill(vpn, ppn_4kb, ptw_flags_);
    MemReq translated = ptw_orig_req_;
    translated.addr = pa;
    if (ReqOut.at(ptw_orig_port_).try_send(translated)) {
      walk_latency_ += (SimPlatform::instance().cycles() - walk_start_cyc_);
      ptw_state_ = PTW_IDLE;
    }
    break;
  }
  default:
    break;
  }
}

void Mmu::on_tick() {
  // 1) Drain responses. PTW responses are claimed by the FSM; everything
  // else flows back upstream unchanged.
  for (uint32_t p = 0; p < num_ports_; ++p) {
    if (RspIn.at(p).empty()) continue;
    const MemRsp& rsp = RspIn.at(p).peek();
    if (rsp.tag & PTW_TAG_MARKER) {
      // PTW response. Only PTW state cares about it.
      if (ptw_state_ == PTW_L1_WAIT || ptw_state_ == PTW_L0_WAIT) {
        on_ptw_response(rsp);
      }
      RspIn.at(p).pop();
    } else {
      if (RspOut.at(p).full()) continue;
      RspOut.at(p).send(rsp, 1);
      RspIn.at(p).pop();
    }
  }

  // 2) Run PTW FSM — emit pending PTE fetches / fill.
  if (ptw_state_ != PTW_IDLE && ptw_state_ != PTW_L1_WAIT && ptw_state_ != PTW_L0_WAIT) {
    drive_ptw();
  }

  // 3) Forward incoming requests. Bypass for non-translated addresses;
  // TLB-hit translates inline; TLB-miss kicks PTW (if free).
  for (uint32_t p = 0; p < num_ports_; ++p) {
    if (ReqIn.at(p).empty()) continue;
    const MemReq& req = ReqIn.at(p).peek();

    if (!needs_translation(req.addr)) {
      if (ReqOut.at(p).try_send(req)) {
        ReqIn.at(p).pop();
      }
      continue;
    }

    uint64_t vpn = req.addr >> MEM_PAGE_LOG2_SIZE;
    auto [hit, ppn] = tlb_.lookup(vpn);
    if (hit) {
      MemReq translated = req;
      translated.addr = (ppn << MEM_PAGE_LOG2_SIZE) |
                        (req.addr & ((1ULL << MEM_PAGE_LOG2_SIZE) - 1));
      if (ReqOut.at(p).try_send(translated)) {
        ReqIn.at(p).pop();
      }
    } else {
      // TLB miss — kick PTW if it's idle. Otherwise this request waits
      // (stays at the head of ReqIn[p]) until the in-flight walk completes.
      if (ptw_state_ == PTW_IDLE) {
        // ACCESS_TYPE inferred from req.write (no FETCH on dcache port).
        ACCESS_TYPE type = req.write ? ACCESS_TYPE::STORE : ACCESS_TYPE::LOAD;
        start_ptw(req.addr, type, req, p);
        ReqIn.at(p).pop();
      }
    }
  }
}

} // namespace vortex

#endif // VM_ENABLE
