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
  , Input(this)
  , Output(this)
  , core_(core) {
  this->reset();
}

VOpcUnit::~VOpcUnit() {}

void VOpcUnit::reset() {
  total_stalls_ = 0;
}

void VOpcUnit::tick() {
  // process incoming instructions
  if (Input.empty())
    return;
  auto trace = Input.front();

  uint32_t scalar_stalls = 0;
  uint32_t vector_stalls = 0;

  // calculate bank conflict stalls
  for (uint32_t i = 0; i < NUM_SRC_REGS; ++i) {
    for (uint32_t j = i + 1; j < NUM_SRC_REGS; ++j) {
      if ((trace->src_regs[i].type == RegType::None)
       || (trace->src_regs[j].type == RegType::None))
        continue;
      if ((trace->src_regs[i].type == RegType::Integer && trace->src_regs[i].id() == 0)
       || (trace->src_regs[j].type == RegType::Integer && trace->src_regs[j].id() == 0))
        continue; // skip x0
      // bank conflict
      uint32_t bank_i = trace->src_regs[i].idx % NUM_GPR_BANKS;
      uint32_t bank_j = trace->src_regs[j].idx % NUM_GPR_BANKS;
      if (bank_i == bank_j) {
        if (trace->src_regs[i].type == RegType::Vector
         && trace->src_regs[j].type == RegType::Vector) {
          ++scalar_stalls;
        } else
        if ((trace->src_regs[i].type != RegType::Vector
          && trace->src_regs[j].type != RegType::Vector)) {
          ++vector_stalls;
        }
      }
    }
  }

  auto stalls  = std::max(scalar_stalls, vector_stalls);

  total_stalls_ += stalls;

  if (trace->fu_type == FUType::VPU) {
    // translate VPU instructions
    this->translate(trace);
  }

  this->Output.push(trace, 2 + stalls);

  DT(3, "pipeline-operands: " << *trace);

  Input.pop();
}

void VOpcUnit::translate(instr_trace_t* trace) {
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
}

void VOpcUnit::writeback(instr_trace_t* /*trace*/) {
  //--
}