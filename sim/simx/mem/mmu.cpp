// Copyright © 2019-2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <VX_config.h>

#ifdef VX_CFG_VM_ENABLE

#include "mmu.h"
#include "debug.h"

using namespace vortex;

Mmu::Mmu(const SimContext& ctx,
         const char* name,
         uint32_t num_ports,
         uint32_t tlb_size,
         uint32_t client_id,
         bool exec_side)
  : SimObject(ctx, name)
  , ReqIn(num_ports, this)
  , RspOut(num_ports, this)
  , ReqOut(num_ports, this)
  , RspIn(num_ports, this)
  , TlbMissOut(this)
  , TlbFillIn(this)
  , num_ports_(num_ports)
  , client_id_(client_id)
  , exec_side_(exec_side)
  , tlb_(tlb_size)
  , mshr_(VX_CFG_L1_TLB_MSHR_SIZE)
  , replay_(num_ports)
  , fault_rsp_(num_ports)
{}

Mmu::~Mmu() {}

void Mmu::on_reset() {
  for (auto& e : mshr_) {
    e = MshrEntry();
  }
  for (auto& q : replay_) {
    q = {};
  }
  for (auto& q : fault_rsp_) {
    q = {};
  }
  reports_ = {};
  // Physical pages move under a virtual address between launches, so
  // cached translations cannot outlive one — the shared levels clear here
  // too and would otherwise disagree with this one.
  tlb_.flush();
}

void Mmu::set_satp(uint64_t satp) {
  // Every CTA and every spawned warp re-writes the same satp; only an
  // actual address-space change invalidates cached translations.
  if (satp_ && satp_->get_satp() == satp) {
    return;
  }
  satp_ = std::make_unique<SATP_t>(satp);
  tlb_.flush();  // sfence.vma
}

bool Mmu::empty() const {
  for (auto& e : mshr_) {
    if (e.valid) {
      return false;
    }
  }
  for (auto& q : replay_) {
    if (!q.empty()) {
      return false;
    }
  }
  for (auto& q : fault_rsp_) {
    if (!q.empty()) {
      return false;
    }
  }
  return reports_.empty();
}

bool Mmu::needs_translation(const MemReq& req) const {
  // The runtime installs identity PTEs at boot for every PA-addressed
  // region (kernel image, page table, stacks), so a plain access
  // post-SATP-set walks the page table — no address-range bypass is
  // needed. BARE mode (SATP unprogrammed) skips translation, covering
  // the few instruction fetches between reset and the kernel's csrw
  // satp. IO-flagged requests skip it too: the IO/OM apertures carry
  // device registers or encoded coordinates, not virtual addresses.
  if (!satp_ || satp_->get_mode() == BARE) {
    return false;
  }
  if (req.flags.io) {
    return false;
  }
  return true;
}

int Mmu::mshr_find(uint64_t vpn) const {
  for (size_t i = 0; i < mshr_.size(); ++i) {
    if (mshr_[i].valid && mshr_[i].vpn == vpn) {
      return (int)i;
    }
  }
  return -1;
}

int Mmu::mshr_alloc(uint64_t vpn) {
  for (size_t i = 0; i < mshr_.size(); ++i) {
    if (!mshr_[i].valid) {
      mshr_[i].valid = true;
      mshr_[i].issued = false;
      mshr_[i].vpn = vpn;
      mshr_[i].parked.clear();
      return (int)i;
    }
  }
  return -1;
}

TlbAccess Mmu::access_of(const MemReq& req) const {
  if (exec_side_) {
    return TlbAccess::Exec;
  }
  return req.is_write() ? TlbAccess::Write : TlbAccess::Read;
}

void Mmu::kill_request(uint32_t port, const MemReq& req) {
  // A killed access never reaches memory. The data side still owes the
  // pipeline whatever response the caches would have produced, or the warp
  // waits on it forever; the fetch side is left dangling on purpose,
  // because a fabricated instruction word would be decoded as real.
  if (exec_side_) {
    return;
  }
  // Same rule the caches apply: only a plain store retires without one.
  if (req.op != MemOp::ST || req.flags.strsp) {
    auto data = std::make_shared<mem_block_t>();
    data->fill(0);
    fault_rsp_.at(port).push(MemRsp(req.tag, req.hart_id, req.uuid, data));
  }
}

void Mmu::report_fault(const MemReq& req) {
  TlbReq report;
  report.vpn = req.addr >> VX_VM_PAGE_LOG2_SIZE;
  report.access = access_of(req);
  report.amo = memop_is_amo(req.op);
  report.client_id = client_id_;
  report.report_only = true;
  reports_.push(report);
}

void Mmu::on_tick() {
  // 1) Forward downstream responses upstream unchanged. Kill responses go
  // first: they belong to accesses that will never reach the cache, and the
  // warps waiting on them cannot drain until they land.
  for (uint32_t p = 0; p < num_ports_; ++p) {
    if (RspOut.at(p).full()) {
      continue;
    }
    if (!fault_rsp_.at(p).empty()) {
      RspOut.at(p).send(fault_rsp_.at(p).front(), 0);
      fault_rsp_.at(p).pop();
      continue;
    }
    if (RspIn.at(p).empty()) {
      continue;
    }
    // Responses pass straight through with no added latency: the translation
    // stage's cost is charged on the request path, not the reply path.
    RspOut.at(p).send(RspIn.at(p).peek(), 0);
    RspIn.at(p).pop();
  }

  // 2) Consume fills: install the translation, then move parked
  // requests to their per-port replay queues in arrival order. A fault
  // installs nothing and kills its parked accesses instead: memory is
  // never touched, but every access that owes the pipeline a response
  // still gets one, so the warp can drain and the launch can be torn
  // down. The fault itself is reported out of band.
  if (!TlbFillIn.empty()) {
    const TlbRsp& rsp = TlbFillIn.peek();
    int id = (int)rsp.slot;
    // Dropping a fill would leave its parked requests waiting forever.
    __assert(id >= 0 && id < (int)mshr_.size() && mshr_[id].valid,
             "TLB fill does not match an outstanding miss");
    uint64_t page_mask = (uint64_t(1) << VX_VM_PAGE_LOG2_SIZE) - 1;
    uint32_t shift = rsp.level * TLB_VPN_LEVEL_BITS;
    uint64_t low_mask = (uint64_t(1) << shift) - 1;
    if (!rsp.fault) {
      tlb_.fill(mshr_[id].vpn, rsp.ppn, rsp.flags, rsp.level);
    }
    for (auto& [port, req] : mshr_[id].parked) {
      // Requests of differing intent chain onto one entry, so the walk's
      // own check covers only the request that allocated it. Each parked
      // request is re-checked against the flags the walk brought back.
      if (rsp.fault) {
        kill_request(port, req);
      } else if (!tlb_perm_ok(rsp.flags, access_of(req), memop_is_amo(req.op))) {
        kill_request(port, req);
        report_fault(req);
        DT(3, this->name() << " perm-fault: vpn=0x" << std::hex
                           << (req.addr >> VX_VM_PAGE_LOG2_SIZE) << std::dec);
      } else {
        MemReq translated = req;
        uint64_t vpn = req.addr >> VX_VM_PAGE_LOG2_SIZE;
        uint64_t ppn = (rsp.ppn & ~low_mask) | (vpn & low_mask);
        translated.addr = (ppn << VX_VM_PAGE_LOG2_SIZE) | (req.addr & page_mask);
        replay_.at(port).push(translated);
      }
    }
    mshr_[id] = MshrEntry();
    TlbFillIn.pop();
  }

  // 3) Issue pending miss requests (one per tick over the shared link).
  for (size_t i = 0; i < mshr_.size(); ++i) {
    auto& e = mshr_[i];
    if (!e.valid || e.issued) {
      continue;
    }
    if (TlbMissOut.full()) {
      break;
    }
    const MemReq& head = e.parked.front().second;
    TlbReq miss;
    miss.vpn = e.vpn;
    miss.access = access_of(head);
    miss.amo = memop_is_amo(head.op);
    miss.client_id = client_id_;
    miss.slot = (uint32_t)i;
    TlbMissOut.send(miss, 1);
    DT(4, this->name() << " tlb-miss: " << miss);
    e.issued = true;
    break;
  }

  // 3b) Drain fault reports on the same link. These carry no slot and
  // expect no fill, so they cannot be folded into the miss station.
  if (!reports_.empty() && !TlbMissOut.full()) {
    TlbMissOut.send(reports_.front(), 1);
    reports_.pop();
  }

  // 4) Forward requests. This is a hit-under-miss stage: a hit issues
  // while older requests are still parked, so only same-address order is
  // guaranteed — those share a VPN, hence one entry and its arrival-order
  // parked list. Replays drain ahead of new input on the same port.
  // The entry array is banked with one lookup port per bank: the lowest
  // port wins a bank each cycle and later ports on the same bank hold
  // (the RTL's bank_conflict).
  std::vector<bool> bank_taken(tlb_.num_banks(), false);
  for (uint32_t p = 0; p < num_ports_; ++p) {
    if (!replay_.at(p).empty()) {
      if (ReqOut.at(p).try_send(replay_.at(p).front())) {
        replay_.at(p).pop();
      }
      continue;
    }

    if (ReqIn.at(p).empty()) {
      continue;
    }
    const MemReq& req = ReqIn.at(p).peek();

    if (!needs_translation(req)) {
      if (ReqOut.at(p).try_send(req)) {
        ReqIn.at(p).pop();
      }
      continue;
    }

    uint64_t vpn = req.addr >> VX_VM_PAGE_LOG2_SIZE;
    uint32_t bank = tlb_.bank_of(vpn);
    if (bank_taken.at(bank)) {
      continue;
    }
    bank_taken.at(bank) = true;
    auto res = tlb_.lookup(vpn);
    if (res.hit) {
      // A cached translation still has to satisfy the access: the entry
      // was installed for whichever intent first missed on this page.
      if (!tlb_perm_ok(res.flags, access_of(req), memop_is_amo(req.op))) {
        DT(3, this->name() << " perm-fault: vpn=0x" << std::hex << vpn << std::dec);
        kill_request(p, req);
        report_fault(req);
        ReqIn.at(p).pop();
        continue;
      }
      MemReq translated = req;
      translated.addr = (res.ppn << VX_VM_PAGE_LOG2_SIZE) |
                        (req.addr & ((1ULL << VX_VM_PAGE_LOG2_SIZE) - 1));
      if (ReqOut.at(p).try_send(translated)) {
        ReqIn.at(p).pop();
      }
    } else {
      // Park the miss and keep the port flowing (hit-under-miss).
      // Same-VPN misses chain on one entry, preserving arrival order;
      // a full entry or a full station stalls this port only.
      int id = mshr_find(vpn);
      if (id < 0) {
        id = mshr_alloc(vpn);
      }
      if (id >= 0 && mshr_[id].parked.size() < 2) {
        mshr_[id].parked.emplace_back(p, req);
        ReqIn.at(p).pop();
      }
    }
  }
}

#endif // VX_CFG_VM_ENABLE
