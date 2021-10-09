#include <iostream>
#include "pipeline.h"

using namespace vortex;

namespace vortex {
std::ostream &operator<<(std::ostream &os, const Pipeline& pipeline) {
  os << pipeline.name_ << ": valid=" << pipeline.valid << std::endl;
  os << pipeline.name_ << ": stalled=" << pipeline.stalled << std::endl;
  os << pipeline.name_ << ": stall_warp=" << pipeline.stall_warp << std::endl;      
  os << pipeline.name_ << ": wid=" << pipeline.wid << std::endl;
  os << pipeline.name_ << ": PC=" << std::hex << pipeline.PC << std::endl;
  os << pipeline.name_ << ": used_iregs=" << pipeline.used_iregs << std::endl;
  os << pipeline.name_ << ": used_fregs=" << pipeline.used_fregs << std::endl;
  os << pipeline.name_ << ": used_vregs=" << pipeline.used_vregs << std::endl;
  return os;
}
}

Pipeline::Pipeline(const char* name) 
: name_(name) {
  this->clear();
}

void Pipeline::clear() {
  valid = false;
  stalled = false;
  stall_warp = false;
  wid = 0;
  PC = 0;
  used_iregs.reset();
  used_fregs.reset();
  used_vregs.reset();
}

bool Pipeline::enter(Pipeline *drain) {
  if (drain) {
    if (drain->stalled) {
      this->stalled = true;
      return false;
    }
    drain->valid = false;
  }
  this->stalled = false;
  if (!this->valid)
    return false;
  return true;
}

void Pipeline::next(Pipeline *drain) {
  if (drain) {
    drain->valid = this->valid;
    drain->stalled = this->stalled;
    drain->stall_warp = this->stall_warp;
    drain->wid = this->wid;
    drain->PC = this->PC;
    drain->rdest = this->rdest;
    drain->rdest_type = this->rdest_type;
    drain->used_iregs = this->used_iregs;
    drain->used_fregs = this->used_fregs;
    drain->used_vregs = this->used_vregs;
  }
}