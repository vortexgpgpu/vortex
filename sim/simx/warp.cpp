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
  PC_ = core_->dcrs().base_dcrs.read(DCR_BASE_STARTUP_ADDR0);
  #if (XLEN == 64)
    PC_ = (uint64_t(core_->dcrs().base_dcrs.read(DCR_BASE_STARTUP_ADDR1)) << 32) | PC_;
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
}

pipeline_trace_t* Warp::eval() {
  assert(tmask_.any());

  uint64_t uuid = (issued_instrs_++ * arch_.num_clusters() * arch_.num_cores() + core_->id()) * arch_.num_warps() + warp_id_;
  
  DPH(1, "Fetch: cid=" << core_->id() << ", wid=" << warp_id_ << ", tmask=");
  for (uint32_t i = 0, n = arch_.num_threads(); i < n; ++i)
    DPN(1, tmask_.test(i));
  DPN(1, ", PC=0x" << std::hex << PC_ << " (#" << std::dec << uuid << ")" << std::endl);

  /* Fetch and decode. */    

  uint32_t instr_code = 0;
  core_->icache_read(&instr_code, PC_, sizeof(uint32_t));
  auto instr = core_->decoder_.decode(instr_code);
  if (!instr) {
    std::cout << std::hex << "Error: invalid instruction 0x" << instr_code << ", at PC=" << PC_ << " (#" << std::dec << uuid << ")" << std::endl;
    std::abort();
  }  

  DP(1, "Instr 0x" << std::hex << instr_code << ": " << *instr);

  // Create trace
  auto trace = new pipeline_trace_t(uuid);
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