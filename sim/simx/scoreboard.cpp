// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "scoreboard.h"

using namespace vortex;

Scoreboard::Scoreboard(const SimContext& ctx, const char* name)
  : SimObject<Scoreboard>(ctx, name)
  , in_use_regs_(VX_CFG_NUM_WARPS) {
  for (auto& in_use_reg : in_use_regs_) {
    in_use_reg.resize((int)RegType::Count);
  }
  this->on_reset();
}

Scoreboard::~Scoreboard() {
}

void Scoreboard::on_reset() {
  for (auto& in_use_reg : in_use_regs_) {
    for (auto& mask : in_use_reg) {
      mask.reset();
    }
  }
  owners_.clear();
  commit_counts_.clear();
  pending_reserve_.clear();
}

bool Scoreboard::in_use(instr_trace_t* trace) const {
  if (trace->wb) {
    assert(trace->dst_reg.type != RegType::None);
    if (in_use_regs_.at(trace->wid).at((int)trace->dst_reg.type).test(trace->dst_reg.idx)) {
      return true;
    }
  }
  for (uint32_t i = 0; i < trace->src_regs.size(); ++i) {
    if (trace->src_regs[i].type != RegType::None) {
      if (in_use_regs_.at(trace->wid).at((int)trace->src_regs[i].type).test(trace->src_regs[i].idx)) {
        return true;
      }
    }
  }
  return false;
}

std::vector<Scoreboard::reg_use_t> Scoreboard::get_uses(instr_trace_t* trace) const {
  std::vector<reg_use_t> out;
  if (trace->wb) {
    assert(trace->dst_reg.type != RegType::None);
    if (in_use_regs_.at(trace->wid).at((int)trace->dst_reg.type).test(trace->dst_reg.idx)) {
      uint32_t reg_id = get_reg_id(trace->dst_reg, trace->wid);
      auto owner = owners_.at(reg_id);
      out.push_back({trace->dst_reg.type, trace->dst_reg.idx, owner->fu_type, owner->op_type, owner->uuid});
    }
  }
  for (uint32_t i = 0; i < trace->src_regs.size(); ++i) {
    if (trace->src_regs[i].type != RegType::None) {
      if (in_use_regs_.at(trace->wid).at((int)trace->src_regs[i].type).test(trace->src_regs[i].idx)) {
        uint32_t reg_id = get_reg_id(trace->src_regs[i], trace->wid);
        auto owner = owners_.at(reg_id);
        out.push_back({trace->src_regs[i].type, trace->src_regs[i].idx, owner->fu_type, owner->op_type, owner->uuid});
      }
    }
  }
  return out;
}

void Scoreboard::reserve(instr_trace_t* trace) {
  uint32_t reg_id = get_reg_id(trace->dst_reg, trace->wid);
  assert(trace->wb);
  in_use_regs_.at(trace->wid).at((int)trace->dst_reg.type).set(trace->dst_reg.idx);
  assert(owners_.count(reg_id) == 0);
  owners_[reg_id] = trace;
}

void Scoreboard::release(instr_trace_t* trace) {
  uint32_t reg_id = get_reg_id(trace->dst_reg, trace->wid);
  assert(trace->wb);
  assert(in_use_regs_.at(trace->wid).at((int)trace->dst_reg.type).test(trace->dst_reg.idx));
  assert(owners_.count(reg_id) != 0);
  owners_.erase(reg_id);
  commit_counts_.erase(reg_id);
  // RTU callback-trap handoff: if a parked WAIT's reservation was deferred
  // onto this register (because this op owned it at mret-restore time),
  // re-install it now instead of clearing the busy bit — the resumed
  // kernel's vx_rt_get_after still needs to stall on it until TERMINAL.
  auto pend = pending_reserve_.find(reg_id);
  if (pend != pending_reserve_.end()) {
    owners_[reg_id] = pend->second;          // busy bit stays set
    pending_reserve_.erase(pend);
    return;
  }
  in_use_regs_.at(trace->wid).at((int)trace->dst_reg.type).reset(trace->dst_reg.idx);
}

std::vector<instr_trace_t*> Scoreboard::snapshot_warp(uint32_t wid) {
  std::vector<instr_trace_t*> out;
  for (auto it = owners_.begin(); it != owners_.end(); ) {
    instr_trace_t* tr = it->second;
    if (tr->wid == wid) {
      out.push_back(tr);
      in_use_regs_.at(wid).at((int)tr->dst_reg.type).reset(tr->dst_reg.idx);
      commit_counts_.erase(it->first);
      it = owners_.erase(it);
    } else {
      ++it;
    }
  }
  return out;
}

void Scoreboard::restore_warp(const std::vector<instr_trace_t*>& snapshot) {
  for (auto* tr : snapshot) {
    uint32_t reg_id = get_reg_id(tr->dst_reg, tr->wid);
    if (owners_.count(reg_id) == 0) {
      in_use_regs_.at(tr->wid).at((int)tr->dst_reg.type).set(tr->dst_reg.idx);
      owners_[reg_id] = tr;
    } else {
      // A dispatcher writeback currently owns this reg (e.g. the epilogue
      // `lw <sts_reg>`). Hand the reservation back when it releases.
      pending_reserve_[reg_id] = tr;
    }
  }
}

bool Scoreboard::commit_packet(instr_trace_t* trace) {
  uint32_t reg_id = get_reg_id(trace->dst_reg, trace->wid);
  auto& n = commit_counts_[reg_id];
  ++n;
  if (n >= trace->num_pkts) {
    // All packets committed; release() will erase the counter entry.
    return true;
  }
  return false;
}
