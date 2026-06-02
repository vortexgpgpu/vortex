// Copyright © 2019-2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <VX_types.h>
#include <VX_config.h>

#ifdef VX_CFG_VM_ENABLE

#include "mmu.h"
#include "../debug.h"

#include <cstring>

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
  , tlb_(VX_CFG_TLB_SIZE)
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
  // Walk from the root table down: level VX_VM_PT_LEVEL-1 .. 0
  // (Sv32: L1->L0; Sv39: L2->L1->L0). The root table is at the SATP PPN.
  ptw_state_     = PTW_REQ;
  ptw_level_     = VX_VM_PT_LEVEL - 1;
  ptw_cur_ppn_   = satp_->get_base_ppn();
  ptw_vaddr_     = va;
  ptw_type_      = type;
  ptw_orig_req_  = orig;
  ptw_orig_port_ = port;
  walk_start_cyc_= SimPlatform::instance().cycles();
  ++walks_;
}

void Mmu::on_ptw_response(const MemRsp& rsp) {
  // Extract the PTE from the cache-line payload at the recorded PTE
  // address (low bits give the byte offset within the line). A PTE is
  // VX_VM_PTE_SIZE bytes — 4 for Sv32, 8 for Sv39.
  uint64_t pte_bytes = 0;
  if (rsp.data) {
    uint32_t byte_off = (uint32_t)(ptw_pte_addr_ & (VX_CFG_MEM_BLOCK_SIZE - 1));
    std::memcpy(&pte_bytes,
                reinterpret_cast<const uint8_t*>(rsp.data->data()) + byte_off,
                VX_VM_PTE_SIZE);
  }
  PTE_t pte(pte_bytes);

  // Validity check per RISC-V privileged spec (Sv32/Sv39).
  bool invalid = (pte.v == 0) | ((pte.r == 0) & (pte.w == 1));
  if (invalid) {
    // Page fault — for now, abort the simulator with a clear message.
    // TODO: route a page-fault exception back to the LSU.
    std::cerr << "MMU: page fault on PTE at 0x" << std::hex << ptw_pte_addr_
              << " (vaddr 0x" << ptw_vaddr_ << ")" << std::dec << std::endl;
    std::abort();
  }

  // A PTE with any of R/W/X set is a leaf; R=W=X=0 is a pointer to the
  // next-level table. A leaf found at level L is a (super)page — L0 =
  // 4 KB, L1 = megapage, L2 = gigapage; PTW_FILL composes the PA from
  // ptw_leaf_level_, so the level just needs to be recorded here.
  bool is_leaf = (pte.r != 0) | (pte.w != 0) | (pte.x != 0);
  if (is_leaf) {
    ptw_final_ppn_  = pte.ppn;
    ptw_flags_      = pte.flags;
    ptw_leaf_level_ = ptw_level_;
    ptw_state_      = PTW_FILL;
    return;
  }
  // Interior node — descend to the next level. A non-leaf at level 0
  // means the walk ran out of levels with no leaf: a page fault.
  if (ptw_level_ == 0) {
    std::cerr << "MMU: page fault — no leaf PTE for vaddr 0x"
              << std::hex << ptw_vaddr_ << std::dec << std::endl;
    std::abort();
  }
  ptw_cur_ppn_ = pte.ppn;
  --ptw_level_;
  ptw_state_   = PTW_REQ;
}

void Mmu::drive_ptw() {
  // Bits of VA per page-table level. Derived from PT geometry:
  // VX_VM_PT_SIZE / VX_VM_PTE_SIZE = entries per table = 2^VPN_BITS_PER_LEVEL.
  const uint32_t VPN_BITS = log2ceil(VX_VM_PT_SIZE / VX_VM_PTE_SIZE);
  const uint64_t VPN_MASK = (1ULL << VPN_BITS) - 1;
  switch (ptw_state_) {
  case PTW_REQ: {
    // Index the page table at the current level by this level's VPN
    // slice. The root level uses the SATP base PPN (set in start_ptw);
    // deeper levels use the interior PTE's PPN recorded by the response.
    uint32_t shift = VX_VM_PAGE_LOG2_SIZE + ptw_level_ * VPN_BITS;
    uint64_t vpn   = (ptw_vaddr_ >> shift) & VPN_MASK;
    ptw_pte_addr_  = pte_addr(ptw_cur_ppn_, vpn);
    MemReq req(MemOp::LD, ptw_pte_addr_, /*data*/nullptr, /*byteen*/0,
               PTW_TAG_MARKER, /*hart_id*/0, /*uuid*/0);
    if (ReqOut.at(0).try_send(req)) {
      DT(4, this->name() << " ptw L" << (uint32_t)ptw_level_
                         << "-req: addr=0x" << std::hex << ptw_pte_addr_ << std::dec);
      ptw_state_ = PTW_WAIT;
    }
    break;
  }
  case PTW_FILL: {
    // Compose the PA. For a leaf at level L the low 12 + L*VPN_BITS VA
    // bits are the offset within the (super)page — L0 = 4 KB (pgoff
    // only), L1 = megapage, L2 = gigapage — and come from the VA, not
    // the leaf PPN.
    uint32_t off_bits = VX_VM_PAGE_LOG2_SIZE +
                        ptw_leaf_level_ * log2ceil(VX_VM_PT_SIZE / VX_VM_PTE_SIZE);
    uint64_t off_mask = (1ULL << off_bits) - 1;
    uint64_t pa_base = (ptw_final_ppn_ << VX_VM_PAGE_LOG2_SIZE) & ~off_mask;
    uint64_t pa = pa_base | (ptw_vaddr_ & off_mask);
    // Cache the per-4KB sub-page mapping in the TLB. A megapage walk
    // therefore only services the specific 4 KB that triggered the
    // miss; subsequent VAs in the same megapage will re-walk (correct,
    // just less optimal — fine for the rare system regions we identity-map).
    uint64_t vpn = ptw_vaddr_ >> VX_VM_PAGE_LOG2_SIZE;
    uint64_t ppn_4kb = pa >> VX_VM_PAGE_LOG2_SIZE;
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
      if (ptw_state_ == PTW_WAIT) {
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
  if (ptw_state_ != PTW_IDLE && ptw_state_ != PTW_WAIT) {
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

    uint64_t vpn = req.addr >> VX_VM_PAGE_LOG2_SIZE;
    auto [hit, ppn] = tlb_.lookup(vpn);
    if (hit) {
      MemReq translated = req;
      translated.addr = (ppn << VX_VM_PAGE_LOG2_SIZE) |
                        (req.addr & ((1ULL << VX_VM_PAGE_LOG2_SIZE) - 1));
      if (ReqOut.at(p).try_send(translated)) {
        ReqIn.at(p).pop();
      }
    } else {
      // TLB miss — kick PTW if it's idle. Otherwise this request waits
      // (stays at the head of ReqIn[p]) until the in-flight walk completes.
      if (ptw_state_ == PTW_IDLE) {
        // ACCESS_TYPE inferred from req op (no FETCH on dcache port).
        ACCESS_TYPE type = req.is_write() ? ACCESS_TYPE::STORE : ACCESS_TYPE::LOAD;
        start_ptw(req.addr, type, req, p);
        ReqIn.at(p).pop();
      }
    }
  }
}

} // namespace vortex

#endif // VX_CFG_VM_ENABLE
