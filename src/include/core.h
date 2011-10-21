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
#ifdef EMU_INSTRUMENTATION
  void reg_doWrite(Word cpuId, Word regNum);
  void reg_doRead(Word cpuId, Word regNum);
#endif

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
    /* Access size here is 8, representing the register size of 64-bit cores. */
    void doWrite() { reg_doWrite(cpuId, regNum); }
    void doRead() { reg_doRead(cpuId, regNum); }
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
#ifdef EMU_INSTRUMENTATION
    bool getSupervisorMode() const { return supervisorMode; }
#endif

  private:
    const ArchDef a;
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
