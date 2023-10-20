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

#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <util.h>

#include "instr.h"
#include "core.h"

using namespace vortex;

Warp::Warp(Core *core, uint32_t warp_id)
    : warp_id_(warp_id)
    , arch_(core->arch())
    , core_(core)
    , ireg_file_(core->arch().num_threads(), std::vector<Word>(core->arch().num_regs()))
    , freg_file_(core->arch().num_threads(), std::vector<uint64_t>(core->arch().num_regs()))
    , vreg_file_(core->arch().num_threads(), std::vector<Byte>(core->arch().vsize()))
{
  this->reset();
}

void Warp::reset() {
  PC_ = core_->dcrs().base_dcrs.read(VX_DCR_BASE_STARTUP_ADDR0);
#if (XLEN == 64)
  PC_ = (uint64_t(core_->dcrs().base_dcrs.read(VX_DCR_BASE_STARTUP_ADDR1)) << 32) | PC_;
#endif
  tmask_.reset();  
  issued_instrs_ = 0;
  for (uint32_t i = 0, n = arch_.num_threads(); i < n; ++i) {
    for (auto& reg : ireg_file_.at(i)) {
      reg = 0;
    }
    for (auto& reg : freg_file_.at(i)) {
      reg = 0;
    }
    for (auto& reg : vreg_file_.at(i)) {
      reg = 0;
    }
  }
  uui_gen_.reset();
}

pipeline_trace_t* Warp::eval() {
  assert(tmask_.any());

#ifndef NDEBUG
  uint32_t instr_uuid = uui_gen_.get_uuid(PC_);
  uint32_t g_wid = core_->id() * arch_.num_warps() + warp_id_;
  uint32_t instr_id  = instr_uuid & 0xffff;
  uint32_t instr_ref = instr_uuid >> 16;
  uint64_t uuid = (uint64_t(instr_ref) << 32) | (g_wid << 16) | instr_id;
#else
  uint64_t uuid = 0;
#endif
  
  DPH(1, "Fetch: cid=" << core_->id() << ", wid=" << warp_id_ << ", tmask=");
  for (uint32_t i = 0, n = arch_.num_threads(); i < n; ++i)
    DPN(1, tmask_.test(i));
  DPN(1, ", PC=0x" << std::hex << PC_ << " (#" << std::dec << uuid << ")" << std::endl);

  // Fetch
  uint32_t instr_code = 0;
  core_->icache_read(&instr_code, PC_, sizeof(uint32_t));

  // Decode
  auto instr = core_->decoder_.decode(instr_code);
  if (!instr) {
    std::cout << std::hex << "Error: invalid instruction 0x" << instr_code << ", at PC=0x" << PC_ << " (#" << std::dec << uuid << ")" << std::endl;
    std::abort();
  }  

  DP(1, "Instr 0x" << std::hex << instr_code << ": " << *instr);

  // Create trace
  auto trace = new pipeline_trace_t(uuid, arch_);
  trace->cid   = core_->id();
  trace->wid   = warp_id_;
  trace->PC    = PC_;
  trace->tmask = tmask_;
  trace->rdest = instr->getRDest();
  trace->rdest_type = instr->getRDType();
    
  // Execute
  this->execute(*instr, trace);

  DP(5, "Register state:");
  for (uint32_t i = 0; i < arch_.num_regs(); ++i) {
    DPN(5, "  %r" << std::setfill('0') << std::setw(2) << std::dec << i << ':');
    // Integer register file
    for (uint32_t j = 0; j < arch_.num_threads(); ++j) {
      DPN(5, ' ' << std::setfill('0') << std::setw(XLEN/4) << std::hex << ireg_file_.at(j).at(i) << std::setfill(' ') << ' ');
    }
    DPN(5, '|');
    // Floating point register file
    for (uint32_t j = 0; j < arch_.num_threads(); ++j) {
      DPN(5, ' ' << std::setfill('0') << std::setw(16) << std::hex << freg_file_.at(j).at(i) << std::setfill(' ') << ' ');
    }
    DPN(5, std::endl);
  }  

  return trace;
}