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

class OpcUnit : public SimObject<OpcUnit> {
public:
  SimPort<instr_trace_t *> Input;
  SimPort<instr_trace_t *> Output;

  SimPort<GprReq> gpr_req_ports;
  SimPort<GprRsp> gpr_rsp_ports;

  OpcUnit(const SimContext &ctx);
  virtual ~OpcUnit();

  virtual void reset();

  virtual void tick();

  void writeback(instr_trace_t* trace);

  uint32_t total_stalls() const {
    return total_stalls_;
  }

private:
  uint32_t pending_rsps_ = 0;
  bool     instr_pending_ = false;
  uint32_t total_stalls_ = 0;
};

} // namespace vortex
