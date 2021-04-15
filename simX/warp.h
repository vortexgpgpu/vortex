#ifndef __WARP_H
#define __WARP_H

#include <vector>
#include <stack>
#include "types.h"

namespace vortex {

class Core;
class Instr;
class Pipeline;
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
class Warp {
public:
  Warp(Core *core, Word id);

  void clear();
  
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

  Word getIRegValue(int reg) const {
    return iRegFile_[0][reg];
  }

  void step(Pipeline *);

private:

  uint32_t aes32esi(int mix_columns, int byte_select, uint32_t word);
  uint32_t aes32dsi(int inv_mix_columns, int byte_select, uint32_t word);
  uint8_t xtime(uint8_t byte);
  uint8_t s_box_replace(uint8_t byte);
  uint8_t inv_s_box_replace(uint8_t byte);
  uint32_t rotr(int n, uint32_t x);
  uint32_t Sigma0(uint32_t x);
  uint32_t Sigma1(uint32_t x);
  uint32_t sigma0(uint32_t x);
  uint32_t sigma1(uint32_t x);

  void execute(const Instr &instr, Pipeline *);
  
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
};

}

#endif