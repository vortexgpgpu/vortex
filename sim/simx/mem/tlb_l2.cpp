// Copyright © 2019-2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <VX_config.h>

#ifdef VX_CFG_VM_ENABLE

#include "tlb_l2.h"
#include "debug.h"

using namespace vortex;

L2Tlb::L2Tlb(const SimContext& ctx, const char* name, uint32_t num_clients)
  : SimObject(ctx, name)
  , ReqIn(num_clients, this)
  , RspOut(num_clients, this)
  , PtwReqOut(this)
  , PtwRspIn(this)
  , num_clients_(num_clients)
  , num_sets_(VX_CFG_L2_TLB_SIZE / VX_CFG_L2_TLB_NUM_WAYS)
  , sets_(num_sets_, std::vector<Entry>(VX_CFG_L2_TLB_NUM_WAYS))
  , mega_(VX_CFG_L2_TLB_MEGA_SIZE)
  , mshr_(VX_CFG_L2_TLB_MSHR_SIZE)
{}

L2Tlb::~L2Tlb() {}

void L2Tlb::on_reset() {
  for (auto& set : sets_) {
    for (auto& e : set) {
      e = Entry();
    }
  }
  mega_.flush();
  for (auto& e : mshr_) {
    e = MshrEntry();
  }
  pipe_ = {};
  rsp_q_ = {};
  rr_client_ = 0;
}

bool L2Tlb::busy() const {
  if (!pipe_.empty() || !rsp_q_.empty()) {
    return true;
  }
  for (auto& e : mshr_) {
    if (e.valid) {
      return true;
    }
  }
  return false;
}

bool L2Tlb::lookup(uint64_t vpn, uint64_t* ppn, uint8_t* flags, uint8_t* level) {
  // Megapage side array first (one entry covers the whole range).
  {
    auto res = mega_.lookup(vpn);
    if (res.hit) {
      *ppn = res.ppn;
      *flags = res.flags;
      *level = res.level;
      return true;
    }
  }
  auto& set = sets_.at(vpn % num_sets_);
  for (auto& e : set) {
    if (e.valid && e.vpn == vpn) {
      e.mru = true;
      *ppn = e.ppn;
      *flags = e.flags;
      *level = 0;
      return true;
    }
  }
  return false;
}

void L2Tlb::fill(uint64_t vpn, uint64_t ppn, uint8_t flags, uint8_t level) {
  if (level != 0) {
    mega_.fill(vpn, ppn, flags, level);
    return;
  }
  auto& set = sets_.at(vpn % num_sets_);
  int victim = -1;
  for (size_t w = 0; w < set.size(); ++w) {
    if (!set[w].valid) {
      victim = (int)w;
      break;
    }
  }
  if (victim < 0) {
    for (size_t w = 0; w < set.size(); ++w) {
      if (!set[w].mru) {
        victim = (int)w;
        break;
      }
    }
  }
  if (victim < 0) {
    for (auto& e : set) {
      e.mru = false;
    }
    victim = 0;
  }
  if (set[victim].valid) {
    ++perf_.evictions;
  }
  set[victim].valid = true;
  set[victim].mru = true;
  set[victim].vpn = vpn;
  set[victim].ppn = ppn;
  set[victim].flags = flags;
}

void L2Tlb::on_tick() {
  uint64_t cycle = SimPlatform::instance().cycles();

  // 1) Consume walker fills: install (faults install nothing) and queue
  // a response for every requester attached to the MSHR entry.
  if (!PtwRspIn.empty()) {
    const TlbRsp& rsp = PtwRspIn.peek();
    uint32_t id = rsp.slot;
    // Dropping a fill would strand every L1 miss station attached to it.
    __assert(id < mshr_.size() && mshr_[id].valid,
             "walker fill does not match an outstanding entry");
    if (!rsp.fault) {
      fill(mshr_[id].vpn, rsp.ppn, rsp.flags, rsp.level);
    }
    for (auto& [client, slot] : mshr_[id].requesters) {
      TlbRsp out = rsp;
      out.client_id = client;
      out.slot = slot;
      rsp_q_.push(out);
    }
    mshr_[id] = MshrEntry();
    PtwRspIn.pop();
  }

  // 2) Fan responses out, one per tick.
  if (!rsp_q_.empty()) {
    const TlbRsp& out = rsp_q_.front();
    if (!RspOut.at(out.client_id).full()) {
      RspOut.at(out.client_id).send(out, 1);
      rsp_q_.pop();
    }
  }

  // 3) Retire the lookup-pipe head once its latency has elapsed. A miss
  // needs an MSHR slot (or a same-VPN entry to attach to); if neither
  // is available the head stalls.
  if (!pipe_.empty() && pipe_.front().first <= cycle) {
    const TlbReq& req = pipe_.front().second;
    if (req.report_only) {
      // Pure fault report: no lookup, no MSHR, no response — forward it
      // to the walker, which owns the latch the host reads.
      if (!PtwReqOut.full()) {
        PtwReqOut.send(req, 1);
        pipe_.pop();
      }
      return;
    }
    uint64_t ppn;
    uint8_t flags, level;
    if (lookup(req.vpn, &ppn, &flags, &level)) {
      ++perf_.hits;
      TlbRsp out;
      out.ppn = ppn;
      out.flags = flags;
      out.level = level;
      out.client_id = req.client_id;
      out.slot = req.slot;
      rsp_q_.push(out);
      pipe_.pop();
    } else {
      ++perf_.misses;
      int id = -1;
      for (size_t i = 0; i < mshr_.size(); ++i) {
        if (mshr_[i].valid && mshr_[i].vpn == req.vpn) {
          id = (int)i;
          ++perf_.mshr_dedups;
          break;
        }
      }
      if (id < 0) {
        for (size_t i = 0; i < mshr_.size(); ++i) {
          if (!mshr_[i].valid) {
            id = (int)i;
            mshr_[i].valid = true;
            mshr_[i].issued = false;
            mshr_[i].vpn = req.vpn;
            mshr_[i].access = req.access;
            mshr_[i].amo = req.amo;
            mshr_[i].requesters.clear();
            break;
          }
        }
      }
      if (id >= 0 && mshr_[id].requesters.size() < MAX_REQUESTERS_PER_ENTRY) {
        mshr_[id].requesters.emplace_back(req.client_id, req.slot);
        pipe_.pop();
      }
    }
  }

  // 4) Issue one pending walk request.
  for (size_t i = 0; i < mshr_.size(); ++i) {
    auto& e = mshr_[i];
    if (!e.valid || e.issued) {
      continue;
    }
    if (PtwReqOut.full()) {
      break;
    }
    TlbReq walk;
    walk.vpn = e.vpn;
    walk.access = e.access;
    walk.amo = e.amo;
    walk.client_id = 0;
    walk.slot = (uint32_t)i;
    PtwReqOut.send(walk, 1);
    DT(4, this->name() << " walk-req: " << walk);
    e.issued = true;
    break;
  }

  // 5) Accept one new lookup (round-robin over clients).
  for (uint32_t n = 0; n < num_clients_; ++n) {
    uint32_t c = (rr_client_ + n) % num_clients_;
    if (ReqIn.at(c).empty()) {
      continue;
    }
    ++perf_.reads;
    pipe_.emplace(cycle + VX_CFG_L2_TLB_LATENCY, ReqIn.at(c).peek());
    ReqIn.at(c).pop();
    rr_client_ = c + 1;
    break;
  }
}

#endif // VX_CFG_VM_ENABLE
