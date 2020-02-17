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

#include "Vcache_simX.h"
#include "verilated.h"

// #ifdef VCD_OUTPUT
#include <verilated_vcd_c.h>
// #endif


#include "trace.h"

namespace Harp {

#ifdef EMU_INSTRUMENTATION
  void reg_doWrite(Word cpuId, Word regNum);
  void reg_doRead(Word cpuId, Word regNum);
#endif

  template <typename T> class Reg {
  public:
    Reg(): val(0), cpuId(0), regNum(0) {}
    Reg(Word c, Word n): val(0), cpuId(c), regNum(n) {}
    Reg(Word c, Word n, T v): val(v), cpuId(c), regNum(n) {}


    Reg &operator=(T r) { if (regNum) {val = r; doWrite();} return *this; }

    operator T() const { doRead(); return val; }

    void trunc(Size s) {
      Word mask((~0ull >> (sizeof(Word)-s)*8));
      val &= mask;
    }

    T val;
  private:
    Word cpuId, regNum;
//    T val;

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
      unsigned p, const std::vector<std::vector<Reg<Word> > >& m,
      std::vector<bool> &tm, Word pc
    ): pc(pc), fallThrough(false), uni(false)
    {
      std::cout << "DomStackEntry TMASK: ";
      for (unsigned i = 0; i < m.size(); ++i)
      {
        std::cout << " " << (!bool(m[i][p]) && tm[i]);
        tmask.push_back(!bool(m[i][p]) && tm[i]);
      }
      std::cout << "\n";
    }

    DomStackEntry(const std::vector<bool> &tmask):
      tmask(tmask), fallThrough(true), uni(false) {}

    std::vector<bool> tmask;   
    Word pc;
    bool fallThrough;
    bool uni;    
  };

  struct vtype
  {
    int vill;
    int vediv;
    int vsew;
    int vlmul;
  };

  class Warp;

  class Core {
  public:
    Core(const ArchDef &a, Decoder &d, MemoryUnit &mem, Word id=0);

    Vcache_simX   * cache_simulator;
    VerilatedVcdC * m_trace;

    bool renameTable[32][32];
    bool vecRenameTable[32];
    bool stallWarp[32];
    bool foundSchedule;

    trace_inst_t inst_in_fetch;
    trace_inst_t inst_in_decode;
    trace_inst_t inst_in_scheduler;
    trace_inst_t inst_in_exe;
    trace_inst_t inst_in_lsu;
    trace_inst_t inst_in_wb;

    bool release_warp;
    int release_warp_num;

    bool interrupt(Word r0);
    bool running() const;

    void getCacheDelays(trace_inst_t *);
    void warpScheduler();
    void fetch();
    void decode();
    void scheduler();
    void execute_unit();
    void load_store();
    void writeback();

    void step();

    void printStats() const;
    
    const ArchDef &a;
    Decoder &iDec;
    MemoryUnit &mem;

    Word interruptEntry;

    unsigned long steps;
    unsigned long num_cycles;
    unsigned long num_instructions;
    std::vector<Warp> w;
    std::map<Word, std::set<Warp *> > b; // Barriers
    int schedule_w;
  };

  class Warp {
  public:
    Warp(Core *c, Word id=0);

    void step(trace_inst_t *);
    bool interrupt(Word r0);
    bool running() const { return activeThreads; }
#ifdef EMU_INSTRUMENTATION
    bool getSupervisorMode() const { return supervisorMode; }
#endif

    void printStats() const;

    struct MemAccess {
      MemAccess(bool w, Word a): wr(w), addr(a) {}
      bool wr;
      Word addr;
    };
    std::vector<MemAccess> memAccesses;
    
//  private:
    Core *core;

    Word pc, shadowPc, id;
    Size activeThreads, shadowActiveThreads;
    std::vector<std::vector<Reg<Word> > > reg;
    std::vector<std::vector<Reg<bool> > > pred;
    std::vector<Reg<uint16_t> > csr;

    std::vector<bool> tmask, shadowTmask;
    std::stack<DomStackEntry> domStack;

    std::vector<Word> shadowReg;
    std::vector<bool> shadowPReg;

    //Vector CSR
    struct vtype vtype; //both of them are XLEN WIDE
    int vl;    //both of them are XLEN WIDE
    Word VLEN; //Total vector length

    std::vector<std::vector<Reg<char*>>> vreg; // 32 vector registers

    bool interruptEnable, shadowInterruptEnable;
    bool supervisorMode, shadowSupervisorMode;
    bool spawned;

    unsigned long steps, insts, loads, stores;
    
    friend class Instruction;
  };
}

#endif
