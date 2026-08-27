// Copyright © 2019-2025
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

#include <VX_types.h>
#include <VX_config.h>

#ifdef VX_CFG_VM_ENABLE

#include "ptw.h"
#include "../debug.h"
#include <mem.h>
#include <util.h>
#include <cstring>
#include <iostream>

namespace vortex {

static constexpr uint32_t VPN_BITS_PER_LEVEL = log2ceil(VX_VM_PT_SIZE / VX_VM_PTE_SIZE);
static constexpr uint32_t TOP_LEVEL = VX_VM_PT_LEVEL - 1;

Ptw::Ptw(const SimContext& ctx, const char* name, uint32_t num_clients)
  : SimObject<Ptw>(ctx, name)
  , ReqIn (num_clients, this)
  , RspOut(num_clients, this)
  , MemReqOut(this)
  , MemRspIn(this)
  , num_clients_(num_clients)
  , slots_(VX_CFG_PTW_NUM_WALKERS)
  , pwc1_(VX_CFG_PTW_WALK_CACHE_SIZE)
  , pwc2_(VX_CFG_PTW_WALK_CACHE_SIZE)
{}

Ptw::~Ptw() = default;

void Ptw::on_reset() {
  for (auto& s : slots_) s = Slot{};
  perf_ = PerfStats();
  this->flush();
}

void Ptw::flush() {
  for (auto& e : pwc1_) e.valid = false;
  for (auto& e : pwc2_) e.valid = false;
  // A walk already in flight read PTEs that may predate the page-table
  // update this flush publishes; its upper-level results must not land in
  // the caches just emptied (the requesting TLB bank discards its result).
  for (auto& s : slots_) {
    if (s.state != Slot::IDLE) s.stale = true;
  }
}

uint64_t Ptw::vpn_slice(uint64_t vpn, uint32_t level) const {
  return (vpn >> (level * VPN_BITS_PER_LEVEL)) & ((1ULL << VPN_BITS_PER_LEVEL) - 1);
}

bool Ptw::pwc_lookup(const std::vector<PwcEntry>& pwc, uint64_t key, uint64_t* ppn) const {
  auto& e = pwc.at(key & (pwc.size() - 1));
  if (e.valid && e.key == key) {
    *ppn = e.ppn;
    return true;
  }
  return false;
}

void Ptw::pwc_fill(std::vector<PwcEntry>& pwc, uint64_t key, uint64_t ppn) {
  pwc.at(key & (pwc.size() - 1)) = PwcEntry{true, key, ppn};
}

void Ptw::on_tick() {
  // 1) memory responses: decode the PTE addressed by the owning slot
  if (!MemRspIn.empty()) {
    auto& rsp = MemRspIn.peek();
    auto& slot = slots_.at(rsp.tag);
    __assert(slot.state == Slot::MEM_RSP, "unexpected walker response");

    uint64_t pte_addr = (slot.cur_ppn * VX_VM_PT_SIZE)
                      + (this->vpn_slice(slot.vpn, slot.level) * VX_VM_PTE_SIZE);
    uint64_t pte_bytes = 0;
    if (rsp.data) {
      uint32_t byte_off = (uint32_t)(pte_addr & (VX_CFG_MEM_BLOCK_SIZE - 1));
      std::memcpy(&pte_bytes,
                  reinterpret_cast<const uint8_t*>(rsp.data->data()) + byte_off,
                  VX_VM_PTE_SIZE);
    }
    PTE_t pte(pte_bytes);
    DT(3, this->name() << " pte-rsp: slot=" << rsp.tag << ", pte=0x" << std::hex << pte_bytes << std::dec);

    bool invalid = (pte.v == 0) || ((pte.r == 0) && (pte.w == 1));
    bool is_leaf = (pte.r != 0) || (pte.w != 0) || (pte.x != 0);
    uint64_t sp_mask = (1ULL << (slot.level * VPN_BITS_PER_LEVEL)) - 1;
    bool misaligned = is_leaf && ((pte.ppn & sp_mask) != 0);
    bool fault = invalid || misaligned || (!is_leaf && slot.level == 0);

    if (fault) {
      // Match the RTL: report the fault to the TLB, which replays the
      // access untranslated. Keep the simulation loud about it.
      std::cerr << "PTW: page fault on PTE at 0x" << std::hex << pte_addr
                << " (vpn 0x" << slot.vpn << ")" << std::dec << std::endl;
      std::abort();
    }

    if (is_leaf) {
      // Resolve the (super)page leaf to the faulting 4 KB frame.
      slot.cur_ppn = (pte.ppn & ~sp_mask) | (slot.vpn & sp_mask);
      slot.flags = pte.flags;
      slot.fault = false;
      slot.state = Slot::DONE;
    } else {
      if (slot.stale) {
        // no PWC fill
      } else if (slot.level == TOP_LEVEL) {
        this->pwc_fill(pwc1_, (slot.root_ppn << VPN_BITS_PER_LEVEL)
                            | this->vpn_slice(slot.vpn, TOP_LEVEL), pte.ppn);
      } else if (VX_VM_PT_LEVEL == 3 && slot.level == 1) {
        this->pwc_fill(pwc2_, (slot.cur_ppn << VPN_BITS_PER_LEVEL)
                            | this->vpn_slice(slot.vpn, 1), pte.ppn);
      }
      slot.cur_ppn = pte.ppn;
      --slot.level;
      slot.state = Slot::MEM_REQ;
    }
    MemRspIn.pop();
  }

  // 2) issue one PTE fetch (round-robin over slots wanting memory)
  if (!MemReqOut.full()) {
    for (uint32_t i = 0; i < slots_.size(); ++i) {
      uint32_t s = (mem_rr_ + i) % slots_.size();
      auto& slot = slots_.at(s);
      if (slot.state != Slot::MEM_REQ)
        continue;
      uint64_t pte_addr = (slot.cur_ppn * VX_VM_PT_SIZE)
                        + (this->vpn_slice(slot.vpn, slot.level) * VX_VM_PTE_SIZE);
      MemReq req(MemOp::LD, pte_addr, nullptr, 0, s, 0, 0);
      DT(3, this->name() << " pte-fetch: addr=0x" << std::hex << pte_addr << std::dec << ", slot=" << s << ", level=" << (int)slot.level);
      MemReqOut.send(req, 1);
      slot.state = Slot::MEM_RSP;
      mem_rr_ = s + 1;
      break;
    }
  }

  // 3) hand one finished walk back (round-robin over done slots)
  for (uint32_t i = 0; i < slots_.size(); ++i) {
    uint32_t s = (done_rr_ + i) % slots_.size();
    auto& slot = slots_.at(s);
    if (slot.state != Slot::DONE)
      continue;
    if (RspOut.at(slot.client).full())
      continue;
    // slot.level stopped at the leaf level (0 = 4 KB page).
    PtwRsp rsp{slot.cur_ppn, slot.level, slot.flags, slot.fault, slot.tag};
    DT(3, this->name() << " walk-done: ppn=0x" << std::hex << slot.cur_ppn << std::dec << ", client=" << slot.client << ", slot=" << s);
    RspOut.at(slot.client).send(rsp, 1);
    perf_.latency += (SimPlatform::instance().cycles() - slot.start_cycle);
    slot.state = Slot::IDLE;
    done_rr_ = s + 1;
    break;
  }

  // 4) accept one new walk (round-robin over clients) into a free slot
  int free_slot = -1;
  for (uint32_t s = 0; s < slots_.size(); ++s) {
    if (slots_.at(s).state == Slot::IDLE) { free_slot = (int)s; break; }
  }
  if (free_slot >= 0) {
    for (uint32_t i = 0; i < num_clients_; ++i) {
      uint32_t c = (client_rr_ + i) % num_clients_;
      if (ReqIn.at(c).empty())
        continue;
      auto req = ReqIn.at(c).peek();
      ReqIn.at(c).pop();
      auto& slot = slots_.at(free_slot);
      slot = Slot{};
      slot.vpn = req.vpn;
      slot.root_ppn = req.root_ppn;
      slot.stale = false;
      slot.client = c;
      slot.tag = req.tag;
      slot.start_cycle = SimPlatform::instance().cycles();
      slot.state = Slot::MEM_REQ;

      // walk-cache lookups pick the starting level
      uint64_t ppn1 = 0, ppn2 = 0;
      bool hit1 = this->pwc_lookup(pwc1_, (req.root_ppn << VPN_BITS_PER_LEVEL)
                                        | this->vpn_slice(req.vpn, TOP_LEVEL), &ppn1);
      bool hit2 = hit1 && (VX_VM_PT_LEVEL == 3)
               && this->pwc_lookup(pwc2_, (ppn1 << VPN_BITS_PER_LEVEL)
                                        | this->vpn_slice(req.vpn, 1), &ppn2);
      if (hit2) {
        // hit2 implies Sv39 (TOP_LEVEL = 2); the guard keeps Sv32 folds sane
        slot.level = (uint8_t)((TOP_LEVEL >= 2) ? (TOP_LEVEL - 2) : 0);
        slot.cur_ppn = ppn2;
      } else if (hit1) {
        slot.level = TOP_LEVEL - 1;
        slot.cur_ppn = ppn1;
      } else {
        slot.level = TOP_LEVEL;
        slot.cur_ppn = req.root_ppn;
      }

      DT(3, this->name() << " walk-start: vpn=0x" << std::hex << req.vpn << std::dec << ", client=" << c << ", slot=" << free_slot << ", level=" << (int)slot.level);
      ++perf_.walks;
      if (hit1) {
        ++perf_.pwc1_hits;
        if (VX_VM_PT_LEVEL == 3) {
          if (hit2) ++perf_.pwc2_hits; else ++perf_.pwc2_misses;
        }
      } else {
        ++perf_.pwc1_misses;
      }
      client_rr_ = c + 1;
      break;
    }
  }
}

} // namespace vortex

#endif // VX_CFG_VM_ENABLE
