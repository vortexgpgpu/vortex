#ifndef __WARP_H
#define __WARP_H

#include <vector>
#include <stack>
#include "types.h"

namespace vortex {

template <typename T>
class Reg {
public:
  Reg()
      : value_(0), cpuId_(0), regNum_(0) {}
  Reg(Word c, Word n)
      : value_(0), cpuId_(c), regNum_(n) {}
  Reg(Word c, Word n, T v)
      : value_(v), cpuId_(c), regNum_(n) {}

  const T &value() const {
    return value_;
  }

  Reg &operator=(T r) {
    if (regNum_) {
      value_ = r;
      doWrite();
    }
    return *this;
  }

  operator T() const {
    doRead();
    return value_;
  }

  void trunc(Size s) {
    Word mask((~0ull >> (sizeof(Word) - s) * 8));
    value_ &= mask;
  }

private:
  T value_;
  Word cpuId_, regNum_;

  void doWrite() const {}
  void doRead() const {}
};

///////////////////////////////////////////////////////////////////////////////

struct DomStackEntry {
  DomStackEntry(
      unsigned p, 
      const std::vector<std::vector<Reg<Word>>> &m,
      std::vector<bool> &tm, 
      Word pc
    ) : pc(pc)
      , fallThrough(false)
      , uni(false) {
    for (unsigned i = 0; i < m.size(); ++i) {
      tmask.push_back(!bool(m[i][p]) && tm[i]);
    }
  }

  DomStackEntry(const std::vector<bool> &tmask)
      : tmask(tmask), fallThrough(true), uni(false) {}

  std::vector<bool> tmask;
  Word pc;
  bool fallThrough;
  bool uni;
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

  void step(trace_inst_t *);
  
  bool running() const {
    return (activeThreads_ != 0);
  }

  void printStats() const;

  Core *core() {
    return core_;
  }

  Word id() const {
    return id_;
  }

  Word get_pc() const {
    return pc_;
  }

  void set_pc(Word pc) {
    pc_ = pc;
  }

  void setActiveThreads(Size activeThreads) {
    activeThreads_ = activeThreads;
  }

  Size getActiveThreads() const {
    return activeThreads_;
  }

  void setSpawned(bool spawned) {
    spawned_ = spawned;
  }

  void setTmask(size_t index, bool value) {
    tmask_[index] = value;
  }

private:

  void execute(Instr &instr, trace_inst_t *);

  struct MemAccess {
    MemAccess(bool w, Word a)
        : wr(w), addr(a) {}
    bool wr;
    Word addr;
  };
  
  std::vector<MemAccess> memAccesses_;

  Word id_;
  Core *core_;
  Word pc_;
  Word shadowPc_;  
  Size activeThreads_;
  Size shadowActiveThreads_;
  std::vector<std::vector<Reg<Word>>> iRegFile_;
  std::vector<std::vector<Reg<Word>>> fRegFile_;
  std::vector<Reg<uint32_t>> csrs_;

  std::vector<bool> tmask_;
  std::vector<bool> shadowTmask_;
  std::stack<DomStackEntry> domStack_;

  std::vector<Word> shadowIReg_;
  std::vector<Word> shadowFReg_;

  struct vtype vtype_; // both of them are XLEN WIDE
  int vl_;             // both of them are XLEN WIDE
  Word VLEN_;          // total vector length

  std::vector<std::vector<Reg<char *>>> vregFile_; // 32 vector registers

  bool spawned_;

  unsigned long steps_;
  unsigned long insts_;
  unsigned long loads_;
  unsigned long stores_;
};

}

#endif