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
#include <iostream>

namespace vortex {

Mmu::Mmu(const SimContext& ctx,
        const char* name,
        uint32_t num_ports,
        uint32_t num_banks)
  : SimObject<Mmu>(ctx, name)
  , ReqIn (num_ports, this)
  , RspOut(num_ports, this)
  , ReqOut(num_ports, this)
  , RspIn (num_ports, this)
  , PtwReqOut(this)
  , PtwRspIn(this)
  , num_ports_(num_ports)
  , num_banks_(num_banks)
  , tlb_(VX_CFG_TLB_SIZE, num_banks)
  , banks_(num_banks)
{}

Mmu::~Mmu() = default;

void Mmu::on_reset() {
  for (auto& b : banks_) b.state = BankMiss::IDLE;
}

void Mmu::set_satp(uint64_t satp) {
  if (satp_ && satp_->get_satp() == satp)
    return;
  satp_ = std::make_unique<SATP_t>(satp);
  tlb_.flush();  // sfence.vma
}

void Mmu::flush() {
  tlb_.flush();
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

void Mmu::on_tick() {
  // 1) responses flow back upstream unchanged (the walker has its own port).
  for (uint32_t p = 0; p < num_ports_; ++p) {
    if (RspIn.at(p).empty()) continue;
    if (RspOut.at(p).full()) continue;
    RspOut.at(p).send(RspIn.at(p).peek(), 1);
    RspIn.at(p).pop();
  }

  // 2) walker fills: install the translation and let the bank replay.
  if (!PtwRspIn.empty()) {
    auto rsp = PtwRspIn.peek();
    PtwRspIn.pop();
    auto& bank = banks_.at(rsp.tag);
    __assert(bank.state == BankMiss::WALK_WAIT, "unexpected walker fill");
    if (!rsp.fault) {
      tlb_.fill(bank.req.addr >> VX_VM_PAGE_LOG2_SIZE, rsp.ppn, rsp.level, rsp.flags);
    }
    DT(3, this->name() << " tlb-fill: ppn=0x" << std::hex << rsp.ppn << std::dec << ", level=" << (int)rsp.level << ", fault=" << rsp.fault << ", bank=" << rsp.tag);
    bank.rsp = rsp;
    bank.state = BankMiss::REPLAY;
  }

  // 3) replay parked accesses (translated, or as-is on a fault). The parked
  // packet stays at the head of its ReqIn port until it leaves here, so the
  // walk-in-flight access remains visible to SimChannel in-flight accounting
  // (the processor's idle/flush decision) and later packets on that lane
  // cannot overtake it — same as the RTL's per-lane elastic buffer.
  for (auto& bank : banks_) {
    if (bank.state != BankMiss::REPLAY)
      continue;
    MemReq translated = ReqIn.at(bank.port).peek();
    if (!bank.rsp.fault) {
      translated.addr = (bank.rsp.ppn << VX_VM_PAGE_LOG2_SIZE)
                      | (translated.addr & ((1ULL << VX_VM_PAGE_LOG2_SIZE) - 1));
    }
    if (!ReqOut.at(bank.port).try_send(translated, TRANSLATE_LATENCY))
      continue;
    ReqIn.at(bank.port).pop();
    bank.state = BankMiss::IDLE;
  }

  // 4) send one pending walk request (round-robin over banks).
  if (!PtwReqOut.full()) {
    for (uint32_t i = 0; i < num_banks_; ++i) {
      uint32_t b = (miss_rr_ + i) % num_banks_;
      auto& bank = banks_.at(b);
      if (bank.state != BankMiss::WALK_REQ)
        continue;
      PtwReq req;
      req.vpn = bank.req.addr >> VX_VM_PAGE_LOG2_SIZE;
      req.root_ppn = satp_->get_base_ppn();
      req.tag = b;
      PtwReqOut.send(req, 1);
      bank.state = BankMiss::WALK_WAIT;
      miss_rr_ = b + 1;
      break;
    }
  }

  // 5) forward incoming requests. Bypass for non-translated addresses;
  // TLB-hit translates and forwards after the lookup-pipeline latency;
  // TLB-miss parks the access on its bank. Each bank is a single-ported CAM
  // fed through a crossbar in the RTL, so it accepts at most one lookup per
  // cycle; ports contend round-robin (mirrors VX_mmu_tlb's "R" arbiters).
  uint64_t bank_taken = 0;
  for (uint32_t i = 0; i < num_ports_; ++i) {
    uint32_t p = (port_rr_ + i) % num_ports_;
    if (ReqIn.at(p).empty()) continue;
    const MemReq& req = ReqIn.at(p).peek();

    if (!this->needs_translation(req.addr)) {
      if (ReqOut.at(p).try_send(req)) {
        ReqIn.at(p).pop();
      }
      continue;
    }

    uint64_t vpn = req.addr >> VX_VM_PAGE_LOG2_SIZE;
    uint32_t b = tlb_.bank_of(vpn);
    auto& bank = banks_.at(b);
    if (bank.state != BankMiss::IDLE) {
      // the bank is busy walking; this request waits at the head of ReqIn[p]
      continue;
    }
    if (bank_taken & (1ULL << b))
      continue;  // bank's lookup port already used this cycle
    if (ReqOut.at(p).full())
      continue;  // don't burn the bank slot (or the perf counters) on a stall
    bank_taken |= 1ULL << b;
    auto res = tlb_.lookup(vpn);
    if (res.hit) {
      MemReq translated = req;
      translated.addr = (res.ppn << VX_VM_PAGE_LOG2_SIZE)
                      | (req.addr & ((1ULL << VX_VM_PAGE_LOG2_SIZE) - 1));
      ReqOut.at(p).send(translated, TRANSLATE_LATENCY);
      ReqIn.at(p).pop();
    } else {
      DT(3, this->name() << " tlb-miss: addr=0x" << std::hex << req.addr << std::dec << ", bank=" << b << ", port=" << p);
      bank.req = req;
      bank.port = p;
      bank.state = BankMiss::WALK_REQ;
    }
  }
  ++port_rr_;
}

} // namespace vortex

#endif // VX_CFG_VM_ENABLE
