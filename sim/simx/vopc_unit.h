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

#pragma once

#include "instr_trace.h"
#include "gpr_unit.h"

namespace vortex {

class Core;

class VOpcUnit : public SimObject<VOpcUnit> {
public:
  SimPort<instr_trace_t*> Input;
  SimPort<instr_trace_t*> Output;

  SimPort<GprReq> gpr_req_ports;
  SimPort<GprRsp> gpr_rsp_ports;

  SimPort<GprReq> vgpr_req_ports;
  SimPort<GprRsp> vgpr_rsp_ports;

  VOpcUnit(const SimContext &ctx, Core* core);

  virtual ~VOpcUnit();

  virtual void reset();

  virtual void tick();

  void writeback(instr_trace_t* trace);

  uint32_t total_stalls() const {
    return total_stalls_;
  }

private:

  bool schedule(instr_trace_t* trace);

  bool fused_schedule(instr_trace_t* trace);

  void decode(instr_trace_t* trace);

  void lsu_flush(instr_trace_t* trace);

  Core* core_;
  std::bitset<NUM_SRC_REGS> vopd_to_fetch_ = 0;
  uint32_t pending_s_rsps_ = 0;
  uint32_t pending_v_rsps_ = 0;
  uint32_t vl_counter_ = 0;
  uint32_t vlmul_counter_ = 0;
  uint32_t red_counter_ = 0;
  uint32_t wb_counter_ = 0;
  uint32_t vs2_opd_ = -1;
  Word     active_PC_;
  bool     instr_pending_ = false;
  bool     is_reduction_ = false;
  bool     lsu_flush_ = false;
  uint32_t total_stalls_ = 0;
};

} // namespace vortex
