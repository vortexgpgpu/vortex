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
         bool exec_side,
         bool rsp_reg,
         bool write_enable,
         bool park_reg,
         bool req_skid)
  : SimObject(ctx, name)
  , ReqIn(num_ports, SimChannel<MemReq>(this, REQ_IN_SIZE + (req_skid ? 1 : 0)))
  , RspOut(num_ports, this)
  , ReqOut(num_ports, this)
  , RspIn(num_ports, this)
  , TlbMissOut(this)
  , TlbFillIn(this)
  , num_ports_(num_ports)
  , client_id_(client_id)
  , exec_side_(exec_side)
  , rsp_reg_(rsp_reg)
  , write_enable_(write_enable)
  , park_reg_(park_reg)
  , tlb_(tlb_size)
  , mshr_(VX_CFG_L1_TLB_MSHR_SIZE)
  , replay_(num_ports)
  , fault_rsp_(num_ports)
  , pf_slot_(num_ports)
{
  if (req_skid) {
    for (auto& in : ReqIn) {
      in.set_registered_ready(true);
    }
  }
}

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
  for (auto& s : pf_slot_) {
    s.reset();
  }
  stage_.reset();
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
  for (auto& e : mshr_) {
    e.retired = false;
  }
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
  for (auto& s : pf_slot_) {
    if (s) {
      return false;
    }
  }
  if (stage_) {
    return false;
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
  // device registers or encoded coordinates, not virtual addresses. So do
  // flush tokens, which carry no address.
  if (!satp_ || satp_->get_mode() == BARE) {
    return false;
  }
  if (req.flags.io || req.flags.flush) {
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
      mshr_[i].retired = false;
      mshr_[i].rearmed = false;
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
  if (needs_kill_rsp(req)) {
    auto data = std::make_shared<mem_block_t>();
    data->fill(0);
    fault_rsp_.at(port).push(MemRsp(req.tag, req.hart_id, req.uuid, data));
  }
}

bool Mmu::needs_kill_rsp(const MemReq& req) const {
  // Same rule the caches apply: only a plain store retires without one.
  return !exec_side_ && (req.op != MemOp::ST || req.flags.strsp);
}

bool Mmu::port_parked(uint32_t port) const {
  if (stage_ && stage_->port == port) {
    return true;
  }
  for (auto& e : mshr_) {
    if (!e.valid) {
      continue;
    }
    for (auto& pr : e.parked) {
      if (pr.first == port) {
        return true;
      }
    }
  }
  return false;
}

void Mmu::retire(int id, uint64_t ppn, uint32_t flags, uint32_t level) {
  // A freed entry keeps its translation only behind a registered park, the
  // one requester that can find its entry freed after probing it.
  uint64_t vpn = mshr_[id].vpn;
  mshr_[id] = MshrEntry();
  if (park_reg_) {
    mshr_[id].vpn = vpn;
    mshr_[id].retired = true;
    mshr_[id].r_ppn = ppn;
    mshr_[id].r_flags = flags;
    mshr_[id].r_level = level;
  }
}

int Mmu::retired_find(uint64_t vpn) const {
  for (size_t i = 0; i < mshr_.size(); ++i) {
    if (!mshr_[i].valid && mshr_[i].retired && mshr_[i].vpn == vpn) {
      return (int)i;
    }
  }
  return -1;
}

void Mmu::replay_or_kill(uint32_t port, const MemReq& req, uint64_t ppn,
                         uint32_t flags, uint32_t level) {
  // Requests of differing intent chain onto one entry, so the walk's own
  // check covers only the request that allocated it. Each parked request is
  // re-checked against the flags the walk brought back.
  if (!tlb_perm_ok(flags, access_of(req), memop_is_amo(req.op))) {
    kill_request(port, req);
    report_fault(req);
    DT(3, this->name() << " perm-fault: vpn=0x" << std::hex
                       << (req.addr >> VX_VM_PAGE_LOG2_SIZE) << std::dec);
    return;
  }
  uint64_t page_mask = (uint64_t(1) << VX_VM_PAGE_LOG2_SIZE) - 1;
  uint64_t low_mask = (uint64_t(1) << (level * TLB_VPN_LEVEL_BITS)) - 1;
  uint64_t vpn = req.addr >> VX_VM_PAGE_LOG2_SIZE;
  MemReq translated = req;
  translated.addr = ((((ppn & ~low_mask) | (vpn & low_mask))) << VX_VM_PAGE_LOG2_SIZE)
                  | (req.addr & page_mask);
  replay_.at(port).push(translated);
}

void Mmu::stage_drain() {
  // The staged park allocates (or joins) a cycle after it was taken, against
  // its own VPN; an entry that filled and freed meanwhile re-arms from its
  // kept translation.
  if (!stage_ || SimPlatform::instance().cycles() < stage_->cycle) {
    return;
  }
  auto& st = *stage_;
  int id = mshr_find(st.vpn);
  if (id < 0) {
    int r = retired_find(st.vpn);
    if (r >= 0) {
      // Re-armed: the entry is held again and drains as a filled one does,
      // two cycles on, absorbing same-page joins meanwhile.
      auto& e = mshr_[r];
      e.valid = true;
      e.issued = true;
      e.rearmed = true;
      e.rearm_cycle = SimPlatform::instance().cycles() + 2;
      e.parked.clear();
      e.parked.emplace_back(st.port, st.req);
      stage_.reset();
      return;
    }
    id = mshr_alloc(st.vpn);
  }
  if (id >= 0 && mshr_[id].parked.size() < MAX_PARKED_PER_ENTRY) {
    mshr_[id].parked.emplace_back(st.port, st.req);
    stage_.reset();
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
  uint64_t rsp_delay = rsp_reg_ ? 1 : 0;
  for (uint32_t p = 0; p < num_ports_; ++p) {
    if (RspOut.at(p).full()) {
      continue;
    }
    if (!fault_rsp_.at(p).empty()) {
      RspOut.at(p).send(fault_rsp_.at(p).front(), rsp_delay);
      fault_rsp_.at(p).pop();
      continue;
    }
    if (pf_slot_.at(p)) {
      RspOut.at(p).send(*pf_slot_.at(p), rsp_delay);
      pf_slot_.at(p).reset();
      continue;
    }
    if (RspIn.at(p).empty()) {
      continue;
    }
    // Responses pass straight through, through the port's response stage
    // when registered: the translation is charged on the request path.
    RspOut.at(p).send(RspIn.at(p).peek(), rsp_delay);
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
    if (!rsp.fault) {
      tlb_.fill(mshr_[id].vpn, rsp.ppn, rsp.flags, rsp.level);
    }
    for (auto& [port, req] : mshr_[id].parked) {
      if (rsp.fault) {
        kill_request(port, req);
      } else {
        replay_or_kill(port, req, rsp.ppn, rsp.flags, rsp.level);
      }
    }
    if (rsp.fault) {
      mshr_[id] = MshrEntry();
    } else {
      retire(id, rsp.ppn, rsp.flags, rsp.level);
    }
    TlbFillIn.pop();
  }

  // 2b) Drain re-armed entries once their fill-equivalent cycle comes.
  for (size_t i = 0; i < mshr_.size(); ++i) {
    auto& e = mshr_[i];
    if (!e.valid || !e.rearmed || SimPlatform::instance().cycles() < e.rearm_cycle) {
      continue;
    }
    for (auto& [port, req] : e.parked) {
      replay_or_kill(port, req, e.r_ppn, e.r_flags, e.r_level);
    }
    retire((int)i, e.r_ppn, e.r_flags, e.r_level);
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
    TlbMissOut.send(miss, 1 + INPUT_STAGE_LATENCY);
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
  // A new request spends one cycle in the RTL's input stage (the TLB tag
  // compare) before its lookup: hits and bypasses leave a cycle later than
  // replays, and a miss reaches the walker a cycle later.
  // With a registered park, the stage's state at the start of the cycle
  // decides both whether it takes a park and which page holds.
  bool stage_free = !stage_.has_value();
  std::optional<uint64_t> hold_vpn;
  if (stage_) {
    hold_vpn = stage_->vpn;
  }
  if (park_reg_) {
    stage_drain();
  }
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
    MemReq req = ReqIn.at(p).peek();

    if (!needs_translation(req)) {
      // A flush token reaches the cache behind its port's parked requests
      // (writing data-side clients only: an overtaken read is still correct).
      if (req.flags.flush && write_enable_ && !exec_side_ && port_parked(p)) {
        continue;
      }
      if (ReqOut.at(p).try_send(req, 1 + INPUT_STAGE_LATENCY)) {
        ReqIn.at(p).pop();
      }
      continue;
    }

    uint64_t vpn = req.addr >> VX_VM_PAGE_LOG2_SIZE;
    if (hold_vpn && *hold_vpn == vpn) {
      continue;  // its page is in the park stage
    }
    auto res = tlb_.lookup(vpn);
    // A page with a live miss-station entry (a re-armed one still holds its
    // requests) parks behind it, as the RTL's station probe does, so a hit
    // cannot overtake an older same-page request.
    if (res.hit && !(park_reg_ && mshr_find(vpn) >= 0)) {
      // A cached translation still has to satisfy the access: the entry
      // was installed for whichever intent first missed on this page.
      if (!tlb_perm_ok(res.flags, access_of(req), memop_is_amo(req.op))) {
        bool needs_rsp = needs_kill_rsp(req);
        if (needs_rsp && pf_slot_.at(p)) {
          continue;  // the port's fault slot is still owed out
        }
        DT(3, this->name() << " perm-fault: vpn=0x" << std::hex << vpn << std::dec);
        if (needs_rsp) {
          auto data = std::make_shared<mem_block_t>();
          data->fill(0);
          pf_slot_.at(p) = MemRsp(req.tag, req.hart_id, req.uuid, data);
        }
        report_fault(req);
        ReqIn.at(p).pop();
        continue;
      }
      MemReq translated = req;
      translated.addr = (res.ppn << VX_VM_PAGE_LOG2_SIZE) |
                        (req.addr & ((1ULL << VX_VM_PAGE_LOG2_SIZE) - 1));
      if (ReqOut.at(p).try_send(translated, 1 + INPUT_STAGE_LATENCY)) {
        ReqIn.at(p).pop();
      }
    } else {
      // Park the miss and keep the port flowing (hit-under-miss).
      // Same-VPN misses chain on one entry, preserving arrival order;
      // a full entry or a full station stalls this port only.
      if (park_reg_) {
        if (stage_free) {
          stage_ = Staged{p, req, vpn, SimPlatform::instance().cycles() + 1};
          stage_free = false;
          ReqIn.at(p).pop();
        }
        continue;
      }
      int id = mshr_find(vpn);
      if (id < 0) {
        id = mshr_alloc(vpn);
      }
      if (id >= 0 && mshr_[id].parked.size() < MAX_PARKED_PER_ENTRY) {
        mshr_[id].parked.emplace_back(p, req);
        ReqIn.at(p).pop();
      }
    }
  }
}

#endif // VX_CFG_VM_ENABLE
