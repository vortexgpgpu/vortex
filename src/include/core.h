/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#ifndef __CORE_H
#define __CORE_H

#include <string>
#include <vector>

#include "types.h"
#include "archdef.h"
#include "enc.h"
#include "mem.h"

namespace Harp {
  template <typename T> class Reg {
  public:
    Reg(): cpuId(0), regNum(0) {}
    Reg(Word c, Word n): cpuId(c), regNum(n) {}

    Reg &operator=(T r) { val = r; doWrite(); return *this; }
    operator T() { doRead(); return val; }

  private:
    Word cpuId, regNum;
    T val;

#ifdef EMU_INSTRUMENTATION
#error TODO: instrument Harp::Reg.
#else
    void doWrite() {}
    void doRead() {}
#endif
  };

  class Core {
  public:
    Core(const ArchDef &a, Decoder &d, MemoryUnit &mem, Word id=0);
    void step();
    bool interrupt(Word r0);
    bool running() const { return activeThreads; }

  private:
    const ArchDef &a;
    Decoder &iDec;
    MemoryUnit &mem;

    Word pc, interruptEntry, shadowPc, id;
    Size activeThreads, shadowActiveThreads;
    std::vector<std::vector<Reg<Word> > > reg;
    std::vector<std::vector<Reg<bool> > > pred;

    std::vector<Word> shadowReg;
    std::vector<bool> shadowPReg;

    bool interruptEnable, shadowInterruptEnable, supervisorMode, 
         shadowSupervisorMode;

    friend class Instruction;
  };
};

#endif
