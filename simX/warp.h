#ifndef __WARP_H
#define __WARP_H

#include <vector>
#include <stack>
#include "types.h"

namespace vortex {

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
  int vill;
  int vediv;
  int vsew;
  int vlmul;
};

class Core;
class Instr;
class trace_inst_t;

class Warp {
public:
  Warp(Core *core, Word id = 0);
  
  bool active() const {
    return active_;
  }

  void activate() {
    active_ = true;
  }

  std::size_t getActiveThreads() const {
    if (active_)
      return tmask_.count();
    return 0;
  }

  void printStats() const;

  Core *core() {
    return core_;
  }

  Word id() const {
    return id_;
  }

  Word getPC() const {
    return PC_;
  }

  void setPC(Word PC) {
    PC_ = PC;
  }

  void setTmask(size_t index, bool value) {
    tmask_[index] = value;
    active_ = tmask_.any();
  }

  void step(trace_inst_t *);

private:

  void execute(Instr &instr, trace_inst_t *);
  
  Word id_;
  bool active_;
  Core *core_;
  
  Word PC_;
  ThreadMask tmask_;  
  
  std::vector<std::vector<Word>> iRegFile_;
  std::vector<std::vector<Word>> fRegFile_;
  std::vector<std::vector<Byte>> vRegFile_;
  std::stack<DomStackEntry> domStack_;

  struct vtype vtype_;
  int vl_;
  
  unsigned long steps_;
  unsigned long insts_;
  unsigned long loads_;
  unsigned long stores_;
};

}

#endif