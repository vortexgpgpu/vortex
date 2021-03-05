#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>

#include "util.h"
#include "instr.h"
#include "core.h"

using namespace vortex;

Warp::Warp(Core *core, Word id)
    : id_(id)
    , active_(false)
    , core_(core)
    , PC_(0x80000000)
    , steps_(0)
    , insts_(0)
    , loads_(0)
    , stores_(0) {

  tmask_.reset();

  iRegFile_.resize(core_->arch().num_threads(), std::vector<Word>(core_->arch().num_regs(), 0));
  fRegFile_.resize(core_->arch().num_threads(), std::vector<Word>(core_->arch().num_regs(), 0));
  vRegFile_.resize(core_->arch().num_regs(), std::vector<Byte>(core_->arch().vsize(), 0));  
}

void Warp::step(trace_inst_t *trace_inst) {
  assert(tmask_.any());

  Size fetchPos(0);
  Size decPos;
  Size wordSize(core_->arch().wsize());
  std::vector<Byte> fetchBuffer(wordSize);

  ++steps_;

  D(3, "current PC=0x" << std::hex << PC_);

  // std::cout << "PC: " << std::hex << PC << "\n";
  trace_inst->PC = PC_;

  /* Fetch and decode. */
  if (wordSize < sizeof(PC_))
    PC_ &= ((1ll << (wordSize * 8)) - 1);
    
  unsigned fetchSize = 4;
  fetchBuffer.resize(fetchSize);
  Word fetched = core_->icache_fetch(PC_ + fetchPos, 0);
  writeWord(fetchBuffer, fetchPos, fetchSize, fetched);

  decPos = 0;
  std::shared_ptr<Instr> instr = core_->decoder().decode(fetchBuffer, decPos, trace_inst);

  // Update PC
  PC_ += decPos;

  // Execute
  this->execute(*instr, trace_inst);

  // At Debug Level 3, print debug info after each instruction.
  D(4, "Register state:");
  for (int i = 0; i < core_->arch().num_regs(); ++i) {
    DPN(4, "  %r" << std::setfill('0') << std::setw(2) << std::dec << i << ':');
    for (int j = 0; j < core_->arch().num_threads(); ++j) {
      DPN(4, ' ' << std::setfill('0') << std::setw(8) << std::hex << iRegFile_[j][i] << std::setfill(' ') << ' ');
    }
    DPN(4, std::endl);
  }

  DPH(3, "Thread mask:");
  for (int i = 0; i < core_->arch().num_threads(); ++i)
    DPN(3, " " << tmask_[i]);
  DPN(3, "\n");
}

void Warp::printStats() const {
  std::cout << "Steps : " << steps_ << std::endl
            << "Insts : " << insts_ << std::endl
            << "Loads : " << loads_ << std::endl
            << "Stores: " << stores_ << std::endl;
}