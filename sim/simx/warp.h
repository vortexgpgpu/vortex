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
    , fallthrough(false)
  {}

  DomStackEntry(const ThreadMask &tmask) 
    : tmask(tmask)
    , fallthrough(true)
  {}

  ThreadMask tmask;
  Word PC;
  bool fallthrough;
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

  void reset();

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
  }

  uint64_t getTmask() const {
    return tmask_.to_ulong();
  }

  Word getIRegValue(uint32_t reg) const {
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