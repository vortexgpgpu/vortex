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

#include "operands.h"
#include <util.h>
#include "core.h"
#include "debug.h"
#include "scheduler.h"

using namespace vortex;

// wid → which OpcUnit within the issue lane owns this warp.
inline static constexpr uint32_t wid_to_opc_idx(uint32_t wid) {
  return (wid / ISSUE_WIDTH) % NUM_OPCS;
}

// wid → slot inside that OpcUnit. Inverse of OpcUnit's
// (wis / NUM_OPCS) packing — see opc_unit.h for the lane/wis/opc/slot
// breakdown.
inline static constexpr uint32_t wid_to_slot(uint32_t wid) {
  return (wid / ISSUE_WIDTH) / NUM_OPCS;
}

namespace {

// Emit a "DEBUG SrcN Reg: <reg>={values...} (#uuid)" line for trace_csv.py.
inline void log_src_operand(uint32_t src_index,
                            const RegOpd& reg,
                            const std::vector<reg_data_t>& values,
                            const ThreadMask& tmask,
                            uint64_t uuid) {
  // All five params are referenced only inside DPH/DPN, which expand to
  // no-ops under NDEBUG. Suppress unused-warnings in release builds.
  __unused(src_index);
  __unused(reg);
  __unused(values);
  __unused(tmask);
  __unused(uuid);
  switch (reg.type) {
  case RegType::None:
    break;
  case RegType::Integer: {
    DPH(2, "Src" << src_index << " Reg: " << reg << "={");
    for (uint32_t t = 0; t < tmask.size(); ++t) {
      if (t) DPN(2, ", ");
      if (!tmask.test(t)) {
        DPN(2, "-");
        continue;
      }
      DPN(2, "0x" << std::hex << values[t].u << std::dec);
    }
    DPN(2, "} (#" << std::dec << uuid << ")" << std::endl);
  } break;
  case RegType::Float: {
    DPH(2, "Src" << src_index << " Reg: " << reg << "={");
    for (uint32_t t = 0; t < tmask.size(); ++t) {
      if (t) DPN(2, ", ");
      if (!tmask.test(t)) {
        DPN(2, "-");
        continue;
      }
      // NaN-boxed single-precision value lives in the low 32b; print just
      // those bits so the trace doesn't show the box pattern as noise.
      if ((values[t].u64 >> 32) == 0xffffffff) {
        DPN(2, "0x" << std::hex << values[t].u32 << std::dec);
      } else {
        DPN(2, "0x" << std::hex << values[t].u64 << std::dec);
      }
    }
    DPN(2, "} (#" << std::dec << uuid << ")" << std::endl);
  } break;
  default:
    std::abort();
  }
}

} // namespace

Operands::Operands(const SimContext &ctx, const char* name, Core* core)
    : SimObject<Operands>(ctx, name)
    , Input(this)
    , Output(this)
    , core_(core)
    , opc_units_(NUM_OPCS) {
  static_assert(NUM_OPCS <= PER_ISSUE_WARPS, "invalid NUM_OPCS value");
  char sname[100];

  // Per-OPC warp slot count: each OPC owns the warps where
  //   (wid % ISSUE_WIDTH == lane) && ((wid/IW) % NUM_OPCS == opc_idx)
  // Slot count = ceil(NUM_WARPS / (ISSUE_WIDTH * NUM_OPCS)).
  uint32_t num_warps   = NUM_WARPS;
  uint32_t num_threads = NUM_THREADS;
  uint32_t per_opc_warps = (num_warps + (ISSUE_WIDTH * NUM_OPCS) - 1) / (ISSUE_WIDTH * NUM_OPCS);

  for (uint32_t i = 0; i < NUM_OPCS; i++) {
    snprintf(sname, 100, "%s-opc%d", name, i);
    opc_units_.at(i) = SimPlatform::instance().create_object<OpcUnit>(sname, per_opc_warps, num_threads);
  }

  if (NUM_OPCS >= 2) {
    snprintf(sname, 100, "%s-rsp_arb", name);
    rsp_arb_ = TraceArbiter::Create(sname, ArbiterType::RoundRobin, NUM_OPCS, 1);
    for (uint32_t i = 0; i < NUM_OPCS; ++i) {
      opc_units_.at(i)->Output.bind(&rsp_arb_->Inputs.at(i));
    }
    rsp_arb_->Outputs.at(0).bind(&this->Output);
  } else {
    // pass-thru
    this->Input.bind(&opc_units_.at(0)->Input);
    opc_units_.at(0)->Output.bind(&this->Output);
  }
}

Operands::~Operands() {
  //--
}

void Operands::on_reset() {
  //--
}

void Operands::on_tick() {
  if (NUM_OPCS < 2)
    return; // pass-thru

  // process requests
  if (Input.empty())
    return;
  auto trace = this->Input.peek();
  if (opc_units_.at(wid_to_opc_idx(trace->wid))->Input.try_send(trace)) {
    Input.pop();
  }
}

uint32_t Operands::total_stalls() const {
  uint32_t total = 0;
  for (const auto& opc_unit : opc_units_) {
    total += opc_unit->total_stalls();
  }
  return total;
}

Word& Operands::dtm_ireg(uint32_t wid, uint32_t reg) {
  // DTM debug single-hart access: lane is selected by Core (we are inside
  // one issue-lane already), so the wid we receive is whole-warp; we look
  // up its OpcUnit + slot here.
  return opc_units_.at(wid_to_opc_idx(wid))->dtm_ireg(wid_to_slot(wid), reg);
}

int Operands::get_exit_code() const {
  // x3 of warp 0, thread 0 (RISC-V _exit convention). Warp 0 routes to
  // opc_units_[0] slot 0, so read directly via Operands' friend access.
  return static_cast<int>(opc_units_.at(0)->regs_.at(0).ireg_file.at(3).at(0));
}

void Operands::fetch_operands(instr_trace_t* trace) {
  assert(trace != nullptr);
  auto* opc = opc_units_.at(wid_to_opc_idx(trace->wid)).get();

  // Operand snapshot uses the warp's current tmask. trace->src_data is
  // pre-sized in instr_trace_t's constructor, so read_src is a tight
  // copy with no per-call allocation. None entries keep their default
  // values — downstream units gate per-source consumption on
  // trace->src_regs[i].type so default values are never read.
  auto& tmask = core_->scheduler().warp(trace->wid).tmask;
  for (uint32_t i = 0; i < NUM_SRC_REGS; ++i) {
    if (trace->src_regs[i].type == RegType::None)
      continue;
    opc->read_src(trace->src_data[i], trace->wid, i, trace->src_regs[i]);
    // Per-instruction trace line consumed by ci/trace_csv.py to match
    // each Src to its uuid.
    log_src_operand(i, trace->src_regs[i], trace->src_data[i], tmask, trace->uuid);
  }
}

void Operands::writeback(instr_trace_t* trace) {
  assert(trace != nullptr);
  auto* opc = opc_units_.at(wid_to_opc_idx(trace->wid)).get();
  opc->writeback(trace, trace->wid);
}
