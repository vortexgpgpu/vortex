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
    , core_(core) {
  iRegFile_.resize(core_->arch().num_threads(), std::vector<Word>(core_->arch().num_regs(), 0));
  fRegFile_.resize(core_->arch().num_threads(), std::vector<Word>(core_->arch().num_regs(), 0));
  vRegFile_.resize(core_->arch().num_regs(), std::vector<Byte>(core_->arch().vsize(), 0));
  this->clear();
}

void Warp::clear() {
  PC_ = STARTUP_ADDR;
  tmask_.reset();
  active_ = false;
}

void Warp::step(Pipeline *pipeline) {
  assert(tmask_.any());

  DPH(2, "Step: wid=" << id_ << ", PC=0x" << std::hex << PC_ << ", tmask=");
  for (int i = 0, n = core_->arch().num_threads(); i < n; ++i)
    DPN(2, tmask_[n-i-1]);
  DPN(2, "\n");

  /* Fetch and decode. */    

  Word fetched = core_->icache_fetch(PC_);
  auto instr = core_->decoder().decode(fetched, PC_);

  // Update pipeline
  pipeline->valid = true;
  pipeline->PC = PC_;
  pipeline->rdest = instr->getRDest();
  pipeline->rdest_type = instr->getRDType();
  pipeline->used_iregs.reset();
  pipeline->used_fregs.reset();
  pipeline->used_vregs.reset();

  switch (pipeline->rdest_type) {
  case 1:
    pipeline->used_iregs[pipeline->rdest] = 1;
    break;
  case 2:
    pipeline->used_fregs[pipeline->rdest] = 1;
    break;
  case 3:
    pipeline->used_vregs[pipeline->rdest] = 1;
    break;
  default:
    break;
  }

  for (int i = 0; i < instr->getNRSrc(); ++i) {
    int type = instr->getRSType(i);
    int reg = instr->getRSrc(i);
    switch (type) {
    case 1:
      pipeline->used_iregs[reg] = 1;
      break;
    case 2:
      pipeline->used_fregs[reg] = 1;
      break;
    case 3:
      pipeline->used_vregs[reg] = 1;
      break;
    default:
      break;
    }
  }
  
  // Execute
  this->execute(*instr, pipeline);

  D(4, "Register state:");
  for (int i = 0; i < core_->arch().num_regs(); ++i) {
    DPN(4, "  %r" << std::setfill('0') << std::setw(2) << std::dec << i << ':');
    for (int j = 0; j < core_->arch().num_threads(); ++j) {
      DPN(4, ' ' << std::setfill('0') << std::setw(8) << std::hex << iRegFile_[j][i] << std::setfill(' ') << ' ');
    }
    DPN(4, std::endl);
  }  
}