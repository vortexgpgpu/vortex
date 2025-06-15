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

#include "vopc_unit.h"
#include "core.h"

using namespace vortex;

VOpcUnit::VOpcUnit(const SimContext &ctx, Core* core)
  : SimObject<VOpcUnit>(ctx, "vopc-unit")
  , Input(this, 1)
  , Output(this)
  , gpr_req_ports(this)
  , gpr_rsp_ports(this)
  , vgpr_req_ports(this)
  , vgpr_rsp_ports(this)
  , core_(core) {
  this->reset();
}

VOpcUnit::~VOpcUnit() {}

void VOpcUnit::reset() {
  pending_s_rsps_ = 0;
  pending_v_rsps_ = 0;
  vl_counter_ = 0;
  vlmul_counter_ = 0;
  red_counter_ = 0;
  wb_counter_ = 0;
  instr_pending_ = false;
  is_reduction_ = false;
  lsu_flush_ = false;
  total_stalls_ = 0;
}

void VOpcUnit::tick() {
  // process incoming instructions
  if (Input.empty())
    return;
  auto trace = Input.front();

  if (!instr_pending_) {
    // calculate operands to fetch
    std::bitset<NUM_SRC_REGS> sopd_to_fetch;
    assert(pending_s_rsps_ == 0);
    assert(pending_v_rsps_ == 0);

    // capture SIMD counters
    if (trace->fu_type == FUType::VPU) {
      auto trace_data = std::dynamic_pointer_cast<VecUnit::ExeTraceData>(trace->data);
      active_PC_ = trace->PC;
      if (trace_data->vpu_op == VpuOpType::VSET) {
        vl_counter_ = trace_data->vl;
        vlmul_counter_ = trace_data->vlmul;
      } else {
        vl_counter_ = 1;
        vlmul_counter_ = 1;
      }
      is_reduction_ = (trace_data->vpu_op >= VpuOpType::ARITH_R);
      if (is_reduction_) {
        red_counter_ = (vlmul_counter_ * vl_counter_) - 1;
        wb_counter_ = (red_counter_ > 1) ? (red_counter_ - 1) : 0;
      }
    } else {
      assert(trace->fu_type == FUType::LSU);
      auto trace_data = std::dynamic_pointer_cast<VecUnit::MemTraceData>(trace->data);
      vs2_opd_ = (trace->src_regs[1].type != RegType::None) ? 1 : -0;
      vl_counter_ = trace_data->vl;
      vlmul_counter_ = trace_data->vnf;
    }

    assert(vlmul_counter_ != 0);
    if (vl_counter_ == 0) {
      // Convert to Nop
      trace->fu_type = FUType::ALU;
      trace->op_type = AluType::ADD;
      this->Output.push(trace);
      Input.pop();
      return;
    }

    DT(4, "*** VOPC begin: vl=" << vl_counter_ << ", vlmul=" << vlmul_counter_ << ", " << *trace);

    // gather operands to fetch
    for (uint32_t i = 0; i < NUM_SRC_REGS; i++) {
      if (trace->src_regs[i].id() == 0)
        continue; // skip x0 or empty
      // skip duplicates
      bool is_dup = false;
      for (uint32_t j = 0; j < i; j++) {
        if (trace->src_regs[i].id() == trace->src_regs[j].id()) {
          is_dup = true;
          break;
        }
      }
      if (!is_dup) {
        if (trace->src_regs[i].type == RegType::Vector) {
          vopd_to_fetch_.set(i);
        } else {
          sopd_to_fetch.set(i);
        }
      }
    }

    // send GPR requests (we do this once)
    for (uint32_t i = 0; i < NUM_SRC_REGS; i++) {
      if (sopd_to_fetch.test(i)) {
        GprReq gpr_req;
        gpr_req.rid = trace->src_regs[i].id();
        gpr_req.wid = trace->wid;
        gpr_req.opd = i;
        gpr_req_ports.push(gpr_req);
        ++pending_s_rsps_;
      }
    }

    // send VGPR requests (we do this once)
    for (uint32_t i = 0; i < NUM_SRC_REGS; i++) {
      if (vopd_to_fetch_.test(i)) {
        GprReq gpr_req;
        gpr_req.rid = trace->src_regs[i].id();
        gpr_req.wid = trace->wid;
        gpr_req.opd = i;
        vgpr_req_ports.push(gpr_req);
        ++pending_v_rsps_;
      }
    }

    // mark current instruction as pending
    instr_pending_ = true;
  }

  // process incoming GPR responses
  if (!gpr_rsp_ports.empty()) {
    assert(pending_s_rsps_ != 0);
    --pending_s_rsps_;
    auto rsp = gpr_rsp_ports.front();
    __unused(rsp);
    gpr_rsp_ports.pop();
  }

  // process incoming VGPR responses
  if (!vgpr_rsp_ports.empty()) {
    assert(pending_v_rsps_ != 0);
    --pending_v_rsps_;
    auto rsp = vgpr_rsp_ports.front();
    __unused(rsp);
    vgpr_rsp_ports.pop();
  }

  // process outgoing instructions
  if (0 == pending_s_rsps_ && pending_v_rsps_ == 0) {
    auto trace = Input.front();
    bool done = false;
  #ifdef FUSED_VPU
    done = this->fused_schedule(trace);
  #else
    done = this->schedule(trace);
  #endif
    if (done) {
      // release instruction
      Input.pop();
      // reset states
      instr_pending_ = false;
      is_reduction_ = false;
      lsu_flush_ = false;
      red_counter_ = 0;
      wb_counter_ = 0;
    }
  }
}

bool VOpcUnit::schedule(instr_trace_t* trace) {
  // we need to run the instruction again for vlmul
  assert(vlmul_counter_ > 0);
  --vlmul_counter_;
  if (vlmul_counter_ != 0) {
    // fetch the vector operands again (skip vs2 operand for LD/ST)
    for (uint32_t i = 0; i < NUM_SRC_REGS; i++) {
      if (vopd_to_fetch_.test(i) && vs2_opd_ != i) {
        GprReq gpr_req;
        gpr_req.rid = trace->src_regs[i].id();
        gpr_req.wid = trace->wid;
        gpr_req.opd = i;
        vgpr_req_ports.push(gpr_req);
        ++pending_v_rsps_;
      }
    }
    // issue a cloned instruction trace
    auto trace_alloc = core_->trace_pool().allocate(1);
    auto new_trace = new (trace_alloc) instr_trace_t(*trace);
    new_trace->wb = false; // disable scoreboard release
    this->lsu_flush(new_trace);
    DT(4, "*** VOPC next group: vlmul=" << vlmul_counter_ << ", " << *new_trace);
    this->Output.push(new_trace);
    return false;
  }
  // we are done with all iterations, issue the original instruction
  this->lsu_flush(trace);
  DT(4, "*** VOPC done: " << *trace);
  this->Output.push(trace);
  return true;
}

bool VOpcUnit::fused_schedule(instr_trace_t* trace) {
  // reduction instructions are serialized via writeback
  if (is_reduction_) {
    if (red_counter_ == 0) {
      // wait on writeback
      if (wb_counter_ != 0)
        return false;
      // we are done with all iterations, issue the original instruction
      this->decode(trace);
      DT(4, "*** VOPC done: " << *trace);
      this->Output.push(trace);
      return true;
    } else {
      --red_counter_;
    }
  }

  // we need to run the instruction again for vlmul
  assert(vlmul_counter_ > 0);
  --vlmul_counter_;
  if (vlmul_counter_ != 0) {
    // fetch the vector operands again (skip vs2 operand for LD/ST)
    for (uint32_t i = 0; i < NUM_SRC_REGS; i++) {
      if (vopd_to_fetch_.test(i) && vs2_opd_ != i) {
        GprReq gpr_req;
        gpr_req.rid = trace->src_regs[i].id();
        gpr_req.wid = trace->wid;
        gpr_req.opd = i;
        vgpr_req_ports.push(gpr_req);
        ++pending_v_rsps_;
      }
    }

    if (is_reduction_ && red_counter_ == 0)
      return false; // we will issue the last trace next

    // issue a cloned instruction trace
    auto trace_alloc = core_->trace_pool().allocate(1);
    auto new_trace = new (trace_alloc) instr_trace_t(*trace);
    new_trace->wb = false; // disable scoreboard release
    this->decode(new_trace);
    DT(4, "*** VOPC next group: vlmul=" << vlmul_counter_ << ", " << *new_trace);
    this->Output.push(new_trace);
    return false;
  }

  // we need to run the instruction again for each lane
  assert(vl_counter_ > 0);
  --vl_counter_;
  if (vl_counter_ != 0) {
    // fetch the vector operands again (skip vs2 operand for LD/ST)
    for (uint32_t i = 0; i < NUM_SRC_REGS; i++) {
      if (vopd_to_fetch_.test(i)) {
        GprReq gpr_req;
        gpr_req.rid = trace->src_regs[i].id();
        gpr_req.wid = trace->wid;
        gpr_req.opd = i;
        vgpr_req_ports.push(gpr_req);
        ++pending_v_rsps_;
      }
    }
    // reset group counter
    if (trace->fu_type == FUType::VPU) {
      auto trace_data = std::dynamic_pointer_cast<VecUnit::ExeTraceData>(trace->data);
      vlmul_counter_ = trace_data->vlmul;
    } else {
      assert(trace->fu_type == FUType::LSU);
      auto trace_data = std::dynamic_pointer_cast<VecUnit::MemTraceData>(trace->data);
      vlmul_counter_ = trace_data->vnf;
    }

    if (is_reduction_ && red_counter_ == 0)
      return false; // we will issue the last trace next

    // issue a cloned instruction trace
    auto trace_alloc = core_->trace_pool().allocate(1);
    auto new_trace = new (trace_alloc) instr_trace_t(*trace);
    new_trace->wb = false; // disable scoreboard release
    this->decode(new_trace);
    DT(4, "*** VOPC next lane: vl=" << vl_counter_ << ", vlmul=" << vlmul_counter_ << ", " << *new_trace);
    this->Output.push(new_trace);
    return false;
  }

  // we are done with all iterations, issue the original instruction
  this->decode(trace);
  DT(4, "*** VOPC done: " << *trace);
  this->Output.push(trace);
  return true;
}

void VOpcUnit::decode(instr_trace_t* trace) {
  // translate to scalar pipeline
  switch (trace->fu_type) {
  case FUType::LSU:
    // no conversion
    break;
  case FUType::VPU: {
    // decode VPU instructions
    auto trace_data = std::dynamic_pointer_cast<VecUnit::ExeTraceData>(trace->data);
    auto vpu_op = trace_data->vpu_op;
    switch (vpu_op) {
    case VpuOpType::VSET:
      // no convertion
      break;
    case VpuOpType::ARITH:
    case VpuOpType::ARITH_R:
      trace->fu_type = FUType::ALU;
      trace->op_type = AluType::ADD;
      break;
    case VpuOpType::IMUL:
      trace->fu_type = FUType::ALU;
      trace->op_type = MdvType::MUL;
      break;
    case VpuOpType::IDIV:
      trace->fu_type = FUType::ALU;
      trace->op_type = MdvType::DIV;
      break;
    case VpuOpType::FMA:
    case VpuOpType::FMA_R:
      trace->fu_type = FUType::FPU;
      trace->op_type = FpuType::FADD;
      break;
    case VpuOpType::FDIV:
      trace->fu_type = FUType::FPU;
      trace->op_type = FpuType::FDIV;
      break;
    case VpuOpType::FSQRT:
      trace->fu_type = FUType::FPU;
      trace->op_type = FpuType::FSQRT;
      break;
    case VpuOpType::FCVT:
      trace->fu_type = FUType::FPU;
      trace->op_type = FpuType::F2I;
      break;
    case VpuOpType::FNCP:
    case VpuOpType::FNCP_R:
      trace->fu_type = FUType::FPU;
      trace->op_type = FpuType::FCMP;
      break;
    default:
      assert(false);
    }
  } break;
  default:
    assert(false);
  }

  this->lsu_flush(trace);
}

void VOpcUnit::writeback(instr_trace_t* trace) {
  // only notify writeback for the currently active reduction instructions
  if (instr_pending_ && wb_counter_ > 0 && trace->PC == active_PC_) {
    --wb_counter_;
  }
}

void VOpcUnit::lsu_flush(instr_trace_t* trace) {
  if (trace->fu_type != FUType::LSU)
    return;
  if (lsu_flush_) {
    trace->data = nullptr;
    return;
  }
  lsu_flush_ = true;
}