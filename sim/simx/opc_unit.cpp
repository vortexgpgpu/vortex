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

#include "opc_unit.h"
#include "core.h"
#include <iostream>
#include <iomanip>
#include <simobject.h>

using namespace vortex;

// wid → local slot index inside this OpcUnit.
inline static constexpr uint32_t wid_to_opc_slot(uint32_t wid) {
  return (wid / ISSUE_WIDTH) / NUM_OPCS;
}

void OpcUnit::warp_regs_t::reset() {
  for (auto& bank : ireg_file) {
    for (auto& v : bank) v = 0;
  }
  for (auto& bank : freg_file) {
    for (auto& v : bank) v = 0;
  }
}

OpcUnit::OpcUnit(const SimContext &ctx, const char* name,
                 uint32_t num_warp_slots, uint32_t num_threads)
  : SimObject<OpcUnit>(ctx, name)
  , Input(this)
  , Output(this)
  , num_threads_(num_threads) {
  regs_.reserve(num_warp_slots);
  for (uint32_t i = 0; i < num_warp_slots; ++i) {
    regs_.emplace_back(num_threads);
  }
  this->on_reset();
}

OpcUnit::~OpcUnit() {}

void OpcUnit::on_reset() {
  total_stalls_ = 0;
  cur_trace_ = nullptr;
  release_cycle_ = 0;
  for (auto& w : regs_) {
    w.reset();
  }
}

static uint32_t compute_bank_conflicts(const instr_trace_t* trace) {
  uint32_t stalls = 0;
  for (uint32_t i = 0; i < NUM_SRC_REGS; ++i) {
    for (uint32_t j = i + 1; j < NUM_SRC_REGS; ++j) {
      if ((trace->src_regs[i].type == RegType::None)
       || (trace->src_regs[j].type == RegType::None))
        continue;
      if ((trace->src_regs[i].type == RegType::Integer && trace->src_regs[i].id() == 0)
       || (trace->src_regs[j].type == RegType::Integer && trace->src_regs[j].id() == 0))
        continue; // skip x0
      uint32_t bank_i = trace->src_regs[i].idx % NUM_GPR_BANKS;
      uint32_t bank_j = trace->src_regs[j].idx % NUM_GPR_BANKS;
      if (bank_i == bank_j)
        ++stalls;
    }
  }
  return stalls;
}

void OpcUnit::on_tick() {
  auto cur_cycle = SimPlatform::instance().cycles();

  // forward held uop once its collection phase has elapsed
  if (cur_trace_ != nullptr && cur_cycle >= release_cycle_) {
    if (!Output.try_send(cur_trace_, 1))
      return;
    DT(3, this->name() << "-pipeline operands: " << *cur_trace_);
    cur_trace_ = nullptr;
  }

  // accept next uop into the holding slot
  if (cur_trace_ == nullptr && !Input.empty()) {
    auto trace = Input.peek();
    uint32_t stalls = compute_bank_conflicts(trace);
    total_stalls_ += stalls;
    cur_trace_ = trace;
    release_cycle_ = cur_cycle + 1 + stalls;
    Input.pop();
  }
}

void OpcUnit::read_src(std::vector<reg_data_t>& out,
                       uint32_t wid,
                       uint32_t src_index,
                       const RegOpd& reg) const {
  __unused(src_index);
  const auto& slot = regs_[wid_to_opc_slot(wid)];
  switch (reg.type) {
  case RegType::Integer: {
    const auto& src = slot.ireg_file[reg.idx];
    for (uint32_t t = 0; t < num_threads_; ++t) {
      out[t].u = src[t];
    }
  } break;
  case RegType::Float: {
    const auto& src = slot.freg_file[reg.idx];
    for (uint32_t t = 0; t < num_threads_; ++t) {
      out[t].u64 = src[t];
    }
  } break;
  case RegType::None:
    break;
  default:
    std::abort();
  }
}

void OpcUnit::writeback(instr_trace_t* trace, uint32_t wid) {
  if (trace->dst_data.empty())
    return;
  uint32_t warp_slot = wid_to_opc_slot(wid);
  auto& rdest = trace->dst_reg;
  auto num_threads = num_threads_;
  switch (rdest.type) {
  case RegType::None:
    break;
  case RegType::Integer: {
    auto& bank = regs_.at(warp_slot).ireg_file.at(rdest.idx);
    DPH(2, "Dest Reg: " << rdest << "={");
    for (uint32_t t = 0; t < num_threads; ++t) {
      if (t) DPN(2, ", ");
      if (!trace->tmask.test(t)) {
        DPN(2, "-");
        continue;
      }
      bank.at(t) = trace->dst_data[t].i;
      DPN(2, "0x" << std::hex << trace->dst_data[t].u << std::dec);
    }
    DPN(2, "} (#" << std::dec << trace->uuid << ")" << std::endl);
  } break;
  case RegType::Float: {
    auto& bank = regs_.at(warp_slot).freg_file.at(rdest.idx);
    DPH(2, "Dest Reg: " << rdest << "={");
    for (uint32_t t = 0; t < num_threads; ++t) {
      if (t) DPN(2, ", ");
      if (!trace->tmask.test(t)) {
        DPN(2, "-");
        continue;
      }
      bank.at(t) = trace->dst_data[t].u64;
      if ((trace->dst_data[t].u64 >> 32) == 0xffffffff) {
        DPN(2, "0x" << std::hex << trace->dst_data[t].u32 << std::dec);
      } else {
        DPN(2, "0x" << std::hex << trace->dst_data[t].u64 << std::dec);
      }
    }
    DPN(2, "} (#" << std::dec << trace->uuid << ")" << std::endl);
  } break;
  default:
    std::cout << "Unrecognized register write back type: " << rdest.type << std::endl;
    std::abort();
    break;
  }
}
