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

#include "func_unit.h"

namespace vortex {

// CSR functional unit (FUType::CSR) — RTL counterpart is VX_csr_unit.
// Owns CSR semantics: get/set CSR, FPU rounding mode lookup, fflags update.
// Per-warp fcsr / cta_csrs / mscratch live on the Scheduler and are reached
// via core_->scheduler().warp(wid).
class CsrUnit : public FuncUnit<NUM_SFU_BLOCKS> {
public:
  using Ptr = std::shared_ptr<CsrUnit>;

  CsrUnit(const SimContext& ctx, const char* name, Core* core);

  // CSR access surface used by FpuUnit (fcsr lookup / fflags update).
  Word get_csr(uint32_t addr, uint32_t wid, uint32_t tid);
  void set_csr(uint32_t addr, Word value, uint32_t wid, uint32_t tid);
  uint32_t get_fpu_rm(uint32_t funct3, uint32_t wid, uint32_t tid);
  void update_fcrs(uint32_t fflags, uint32_t wid, uint32_t tid);

protected:
  void on_tick() override;

private:
  // Per-unit functional execution (CSRRW / CSRRS / CSRRC). Called only
  // from this unit's tick().
  void execute(instr_trace_t* trace);

  uint32_t latency_of(const instr_trace_t* trace) const;
};

}
