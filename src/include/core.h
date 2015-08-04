/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#ifndef __CORE_H
#define __CORE_H

#include <string>
#include <vector>
#include <stack>
#include <map>
#include <set>

#include "types.h"
#include "archdef.h"
#include "enc.h"
#include "mem.h"
#include "debug.h"

namespace Harp {
#ifdef EMU_INSTRUMENTATION
  void reg_doWrite(Word cpuId, Word regNum);
  void reg_doRead(Word cpuId, Word regNum);
#endif

  template <typename T> class Reg {
  public:
    Reg(): cpuId(0), regNum(0), val(0) {}
    Reg(Word c, Word n): cpuId(c), regNum(n), val(0) {}

    Reg &operator=(T r) { val = r; doWrite(); return *this; }

    operator T() const { doRead(); return val; }

    void trunc(Size s) {
      Word mask((~0ull >> (sizeof(Word)-s)*8));
      val &= mask;
    }

  private:
    Word cpuId, regNum;
    T val;

#ifdef EMU_INSTRUMENTATION
    /* Access size here is 8, representing the register size of 64-bit cores. */
    void doWrite() const { reg_doWrite(cpuId, regNum); }
    void doRead() const { reg_doRead(cpuId, regNum); }
#else
    void doWrite() const {}
    void doRead() const {}
#endif
  };

  // Entry in the IPDOM Stack
  struct DomStackEntry {
    DomStackEntry(
      unsigned p, const std::vector<std::vector<Reg<bool> > >& m,
      std::vector<bool> &tm, Word pc
    ): pc(pc), fallThrough(false)
    {
      for (unsigned i = 0; i < m.size(); ++i)
        tmask.push_back(!bool(m[i][p]) && tm[i]);
    }

    DomStackEntry(const std::vector<bool> &tmask):
      tmask(tmask), fallThrough(true) {}

    bool fallThrough;
    std::vector<bool> tmask;
    Word pc;
  };

  class Warp;

  class Core {
  public:
    Core(const ArchDef &a, Decoder &d, MemoryUnit &mem, Word id=0);

    bool interrupt(Word r0);
    bool running() const;
    void step();

    const ArchDef &a;
    Decoder &iDec;
    MemoryUnit &mem;

    Word interruptEntry;

    std::vector<Warp> w;
    std::map<Word, std::set<Warp *> > b; // Barriers
  };

  class Warp {
  public:
    Warp(Core *c, Word id=0);

    void step();
    bool interrupt(Word r0);
    bool running() const { return activeThreads; }
#ifdef EMU_INSTRUMENTATION
    bool getSupervisorMode() const { return supervisorMode; }
#endif

//  private:
    Core *core;

    Word pc, shadowPc, id;
    Size activeThreads, shadowActiveThreads;
    std::vector<std::vector<Reg<Word> > > reg;
    std::vector<std::vector<Reg<bool> > > pred;

    std::vector<bool> tmask, shadowTmask;
    std::stack<DomStackEntry> domStack;

    std::vector<Word> shadowReg;
    std::vector<bool> shadowPReg;

    bool interruptEnable, shadowInterruptEnable, supervisorMode, 
         shadowSupervisorMode, spawned;

    friend class Instruction;
  };
};

#endif
