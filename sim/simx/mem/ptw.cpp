// Copyright © 2019-2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <VX_config.h>

#ifdef VX_CFG_VM_ENABLE

#include "ptw.h"
#include "debug.h"
#include <cstring>
#include <util.h>

using namespace vortex;

Ptw::Ptw(const SimContext& ctx, const char* name)
  : SimObject(ctx, name)
  , ReqIn(this)
  , RspOut(this)
  , MemReqOut(this)
  , MemRspIn(this)
  , walkers_(VX_CFG_PTW_NUM_WALKERS)
  , walk_cache_(VX_CFG_PTW_WALK_CACHE_SIZE)
{}

Ptw::~Ptw() {}

void Ptw::on_reset() {
  for (auto& w : walkers_) {
    w = Walker();
  }
  for (auto& e : walk_cache_) {
    e = WcEntry();
  }
  // fault_ deliberately survives: a launch is reset per dispatch, and a
  // fault raised by one launch of a batch must still be readable after the
  // next one starts. It clears when the host reads the report.
}

void Ptw::set_satp(uint64_t satp) {
  satp_ = std::make_unique<SATP_t>(satp);
  this->flush_walk_cache();
}

void Ptw::clear_fault() {
  fault_ = FaultInfo();
}

void Ptw::flush_walk_cache() {
  for (auto& e : walk_cache_) {
    e.valid = false;
  }
}

bool Ptw::busy() const {
  for (auto& w : walkers_) {
    if (w.state != W_IDLE) {
      return true;
    }
  }
  return false;
}

void Ptw::raise_fault(const Walker& w) {
  if (!fault_.valid) {
    fault_.valid = true;
    fault_.va = w.vpn << VX_VM_PAGE_LOG2_SIZE;
    fault_.access = (uint8_t)w.access;
    fault_.amo = w.amo;
  }
}

void Ptw::on_pte_response(Walker& w, const MemRsp& rsp) {
  uint64_t pte_bytes = 0;
  if (rsp.data) {
    uint32_t byte_off = (uint32_t)(w.pte_addr & (VX_CFG_MEM_BLOCK_SIZE - 1));
    std::memcpy(&pte_bytes,
                reinterpret_cast<const uint8_t*>(rsp.data->data()) + byte_off,
                VX_VM_PTE_SIZE);
  }
  PTE_t pte(pte_bytes);

  // Validity per the RISC-V privileged spec (Sv32/Sv39).
  bool invalid = (pte.v == 0) | ((pte.r == 0) & (pte.w == 1));
  if (invalid) {
    w.fault = true;
    w.state = W_FILL;
    return;
  }

  bool is_leaf = (pte.r != 0) | (pte.w != 0) | (pte.x != 0);
  if (is_leaf) {
    // Superpage PPN low bits must be zero (misaligned superpage).
    uint32_t shift = w.level * TLB_VPN_LEVEL_BITS;
    uint64_t low_mask = (uint64_t(1) << shift) - 1;
    bool misaligned = (pte.ppn & low_mask) != 0;
    // Only structural faults are raised here. Permissions belong to the
    // requester: a walk is shared by requests of differing intent, so a
    // check here would judge them all by whichever one allocated it.
    if (misaligned) {
      w.fault = true;
      w.state = W_FILL;
      return;
    }
    w.final_ppn = pte.ppn;
    w.flags = pte.flags;
    w.leaf_level = w.level;
    w.state = W_FILL;
    return;
  }

  // Interior node: a non-leaf at level 0 exhausts the walk.
  if (w.level == 0) {
    w.fault = true;
    w.state = W_FILL;
    return;
  }
  w.cur_ppn = pte.ppn;
  --w.level;
  if (w.level == 0) {
    // Cache the last-level table so spatially-adjacent walks skip the
    // non-leaf fetches.
    uint64_t tag = w.vpn >> TLB_VPN_LEVEL_BITS;
    auto& e = walk_cache_.at(tag % walk_cache_.size());
    e.valid = true;
    e.tag = tag;
    e.table_ppn = w.cur_ppn;
  }
  w.state = W_REQ;
}

void Ptw::on_tick() {
  // 1) PTE responses demux to their walker by request tag.
  if (!MemRspIn.empty()) {
    const MemRsp& rsp = MemRspIn.peek();
    uint32_t wid = rsp.tag;
    // A PTE response that matches no waiting walker cannot be recovered:
    // the walker it belonged to would wait forever and hang the launch.
    __assert(wid < walkers_.size() && walkers_[wid].state == W_WAIT,
             "PTE response does not match a waiting walker");
    on_pte_response(walkers_[wid], rsp);
    MemRspIn.pop();
  }

  // 2) Drive each walker.
  for (uint32_t i = 0; i < walkers_.size(); ++i) {
    auto& w = walkers_[i];
    switch (w.state) {
    case W_REQ: {
      uint32_t shift = VX_VM_PAGE_LOG2_SIZE + w.level * TLB_VPN_LEVEL_BITS;
      uint64_t idx = (w.vpn << VX_VM_PAGE_LOG2_SIZE) >> shift;
      idx &= (uint64_t(1) << TLB_VPN_LEVEL_BITS) - 1;
      w.pte_addr = (w.cur_ppn * VX_VM_PT_SIZE) + (idx * VX_VM_PTE_SIZE);
      MemReq req(MemOp::LD, w.pte_addr, nullptr, 0, i, 0, 0);
      if (MemReqOut.try_send(req)) {
        DT(4, this->name() << " pte-req: level=" << (int)w.level
                           << ", addr=0x" << std::hex << w.pte_addr << std::dec);
        w.state = W_WAIT;
      }
      break;
    }
    case W_FILL: {
      if (RspOut.full()) {
        break;
      }
      TlbRsp rsp;
      rsp.slot = w.mshr_slot;
      if (w.fault) {
        // The first fault latches the report the host reads back; the
        // response carries no translation, so the requester kills the
        // access rather than installing one.
        raise_fault(w);
        rsp.fault = true;
        RspOut.send(rsp, 1);
        perf_.walk_latency += (SimPlatform::instance().cycles() - w.start_cycle);
        w.state = W_IDLE;
        break;
      }
      uint32_t shift = w.leaf_level * TLB_VPN_LEVEL_BITS;
      uint64_t low_mask = (uint64_t(1) << shift) - 1;
      rsp.ppn = (w.final_ppn & ~low_mask) | (w.vpn & low_mask);
      rsp.flags = w.flags;
      rsp.level = w.leaf_level;
      RspOut.send(rsp, 1);
      DT(4, this->name() << " walk-fill: " << rsp);
      perf_.walk_latency += (SimPlatform::instance().cycles() - w.start_cycle);
      w.state = W_IDLE;
      break;
    }
    default:
      break;
    }
  }

  // 3) A report carries a fault caught against a cached translation: latch
  // it and retire the request, no walk and no response.
  if (!ReqIn.empty() && ReqIn.peek().report_only) {
    const TlbReq& req = ReqIn.peek();
    Walker w;
    w.vpn = req.vpn;
    w.access = req.access;
    w.amo = req.amo;
    raise_fault(w);
    ReqIn.pop();
    return;
  }

  // 4) Dispatch a new walk to a free walker.
  if (!ReqIn.empty()) {
    // Without a root there is nothing to walk and the request would sit
    // here forever, stalling the miss station that owns it.
    __assert(satp_ != nullptr, "walk requested before the page-table root was set");
    for (auto& w : walkers_) {
      if (w.state != W_IDLE) {
        continue;
      }
      const TlbReq& req = ReqIn.peek();
      w.vpn = req.vpn;
      w.access = req.access;
      w.amo = req.amo;
      w.mshr_slot = req.slot;
      w.fault = false;
      w.level = VX_VM_PT_LEVEL - 1;
      w.cur_ppn = satp_->get_base_ppn();
      w.start_cycle = SimPlatform::instance().cycles();
      ++perf_.walks;
      // Walk-cache probe: a hit skips straight to the last-level table.
      uint64_t tag = req.vpn >> TLB_VPN_LEVEL_BITS;
      auto& e = walk_cache_.at(tag % walk_cache_.size());
      if (e.valid && e.tag == tag && VX_VM_PT_LEVEL > 1) {
        w.level = 0;
        w.cur_ppn = e.table_ppn;
        ++perf_.wc_hits;
      }
      w.state = W_REQ;
      ReqIn.pop();
      break;
    }
  }
}

#endif // VX_CFG_VM_ENABLE
