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
#include <algorithm>

using namespace vortex;

Scoreboard::Scoreboard(const SimContext& ctx, const char* name)
  : SimObject<Scoreboard>(ctx, name)
  , in_use_regs_(VX_CFG_NUM_WARPS)
  , in_use_fcsr_(VX_CFG_NUM_WARPS, 0) {
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
  std::fill(in_use_fcsr_.begin(), in_use_fcsr_.end(), 0);
  owners_.clear();
  commit_counts_.clear();
  pending_releases_.clear();
}

void Scoreboard::on_tick() {
  uint64_t now = SimPlatform::instance().cycles();
  while (!pending_releases_.empty() && pending_releases_.front().due <= now) {
    auto& r = pending_releases_.front();
    if (r.fcsr != 0) {
      in_use_fcsr_.at(r.wid) &= ~r.fcsr;
    } else {
      owners_.erase(get_reg_id(r.reg, r.wid));
      in_use_regs_.at(r.wid).at((int)r.reg.type).reset(r.reg.idx);
    }
    pending_releases_.pop_front();
  }
  if (pending_releases_.empty()) {
    this->tick_sleep();
  }
}

bool Scoreboard::in_use(instr_trace_t* trace) const {
  auto& instr = *trace->instr_ptr;
  if (in_use_fcsr_.at(trace->wid) & (instr.fcsr_reads() | instr.fcsr_writes())) {
    return true;
  }
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
  commit_counts_.erase(reg_id);
  pending_releases_.push_back({trace->wid, trace->dst_reg, 0,
                               SimPlatform::instance().cycles() + kReleaseDelay});
  this->tick_wake();
}

void Scoreboard::reserve_fcsr(instr_trace_t* trace) {
  uint8_t fields = trace->instr_ptr->fcsr_writes();
  assert((in_use_fcsr_.at(trace->wid) & fields) == 0);
  in_use_fcsr_.at(trace->wid) |= fields;
}

void Scoreboard::release_fcsr(instr_trace_t* trace) {
  uint8_t fields = trace->instr_ptr->fcsr_writes();
  assert((in_use_fcsr_.at(trace->wid) & fields) == fields);
  pending_releases_.push_back({trace->wid, {RegType::None, 0}, fields,
                               SimPlatform::instance().cycles() + kReleaseDelay});
  this->tick_wake();
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
