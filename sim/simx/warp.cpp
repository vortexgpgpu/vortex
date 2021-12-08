#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <util.h>

#include "instr.h"
#include "core.h"

using namespace vortex;

Warp::Warp(Core *core, Word id)
    : id_(id)
    , core_(core)
    , ireg_file_(core->arch().num_threads(), std::vector<Word>(core->arch().num_regs()))
    , freg_file_(core->arch().num_threads(), std::vector<Word>(core->arch().num_regs()))
    , vreg_file_(core->arch().num_threads(), std::vector<Byte>(core->arch().vsize()))
{
  this->clear();
}

void Warp::clear() {
  active_ = false;
  PC_ = STARTUP_ADDR;
  tmask_.reset();  
  for (int i = 0, n = core_->arch().num_threads(); i < n; ++i) {
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

void Warp::eval(pipeline_trace_t *trace) {
  assert(tmask_.any());

  DPH(2, "Fetch: coreid=" << core_->id() << ", wid=" << id_ << ", tmask=");
  for (int i = 0, n = core_->arch().num_threads(); i < n; ++i)
    DPN(2, tmask_.test(n-i-1));
  DPN(2, ", PC=0x" << std::hex << PC_ << " (#" << std::dec << trace->uuid << ")" << std::endl);

  /* Fetch and decode. */    

  Word instr_code = core_->icache_read(PC_, sizeof(Word));
  auto instr = core_->decoder().decode(instr_code);
  if (!instr) {
    std::cout << std::hex << "Error: invalid instruction 0x" << instr_code << ", at PC=" << PC_ << std::endl;
    std::abort();
  }  

  DP(2, "Instr 0x" << std::hex << instr_code << ": " << *instr);

  // Update trace
  trace->cid   = core_->id();
  trace->wid   = id_;
  trace->PC    = PC_;
  trace->tmask = tmask_;
  trace->rdest = instr->getRDest();
  trace->rdest_type = instr->getRDType();
    
  // Execute
  this->execute(*instr, trace);

  DP(4, "Register state:");
  for (int i = 0; i < core_->arch().num_regs(); ++i) {
    DPN(4, "  %r" << std::setfill('0') << std::setw(2) << std::dec << i << ':');
    for (int j = 0; j < core_->arch().num_threads(); ++j) {
      DPN(4, ' ' << std::setfill('0') << std::setw(8) << std::hex << ireg_file_.at(j).at(i) << std::setfill(' ') << ' ');
    }
    DPN(4, std::endl);
  }  
}