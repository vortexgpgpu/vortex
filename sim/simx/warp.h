#ifndef __WARP_H
#define __WARP_H

#include <vector>
#include <stack>
#include "types.h"

namespace vortex {

class Arch;
class Core;
class Instr;
class pipeline_trace_t;

struct DomStackEntry {
  DomStackEntry(const ThreadMask &tmask, Word PC) 
    : tmask(tmask)
    , PC(PC)
    , fallThrough(false)
    , unanimous(false) 
  {}

  DomStackEntry(const ThreadMask &tmask)
      : tmask(tmask)
      , PC(0)
      , fallThrough(true)
      , unanimous(false) 
  {}

  ThreadMask tmask;
  Word PC;
  bool fallThrough;
  bool unanimous;
};

struct vtype {
  uint32_t vill;
  uint32_t vediv;
  uint32_t vsew;
  uint32_t vlmul;
};

class Warp {
public:
  Warp(Core *core, uint32_t warp_id);

  void clear();
  
  bool active() const {
    return active_;
  }

  void suspend() {
    active_ = false;
  }

  void activate() {
    active_ = true;
  }

  std::size_t getActiveThreads() const {
    if (active_)
      return tmask_.count();
    return 0;
  }

  uint32_t id() const {
    return warp_id_;
  }

  Word getPC() const {
    return PC_;
  }

  void setPC(Word PC) {
    PC_ = PC;
  }

  void setTmask(size_t index, bool value) {
    tmask_.set(index, value);
    active_ = tmask_.any();
  }

  uint32_t getTmask() const {
    if (active_)
      return tmask_.to_ulong();
    return 0;
  }

  uint32_t getIRegValue(uint32_t reg) const {
    return ireg_file_.at(0).at(reg);
  }

  uint64_t incr_instrs() {
    return issued_instrs_++;
  }

  pipeline_trace_t* eval();

private:

  void execute(const Instr &instr, pipeline_trace_t *trace);
  
  uint32_t warp_id_;
  const Arch& arch_;
  Core *core_;
  bool active_;
  uint64_t issued_instrs_;
  
  Word PC_;
  ThreadMask tmask_;  
  
  std::vector<std::vector<Word>>     ireg_file_;
  std::vector<std::vector<uint64_t>> freg_file_;
  std::vector<std::vector<Byte>>     vreg_file_;
  std::stack<DomStackEntry>          ipdom_stack_;

  struct vtype vtype_;
  uint32_t vl_;
};

}

#endif