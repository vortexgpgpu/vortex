/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/

#include <iostream>

#include "include/types.h"
#include "include/util.h"
#include "include/archdef.h"
#include "include/mem.h"
#include "include/enc.h"
#include "include/core.h"
#include "include/debug.h"

#ifdef EMU_INSTRUMENTATION
#include "include/qsim-harp.h"
#endif

using namespace Harp;
using namespace std;

#ifdef EMU_INSTRUMENTATION
void Harp::reg_doRead(Word cpuId, Word regNum) {
  Harp::OSDomain::osDomain->do_reg(cpuId, regNum, 8, true);
}

void Harp::reg_doWrite(Word cpuId, Word regNum) {
  Harp::OSDomain::osDomain->do_reg(cpuId, regNum, 8, false);
}
#endif

Core::Core(const ArchDef &a, Decoder &d, MemoryUnit &mem, Word id) : 
  a(a), iDec(d), mem(mem), pc(0), interruptEnable(false), supervisorMode(true),
  activeThreads(1), reg(0), pred(0), shadowReg(a.getNRegs()),
  shadowPReg(a.getNPRegs()), interruptEntry(0), id(id)
{
  /* Build the register file. */
  Word regNum(0);
  for (Word j = 0; j < a.getNThds(); ++j) {
    reg.push_back(vector<Reg<Word> >(0));
    for (Word i = 0; i < a.getNRegs(); ++i) {
      reg[j].push_back(Reg<Word>(id, regNum++));
    }

    pred.push_back(vector<Reg<bool> >(0));
    for (Word i = 0; i < a.getNPRegs(); ++i) {
      pred[j].push_back(Reg<bool>(id, regNum++));
    }

    tmask.push_back(true);
    shadowTmask.push_back(true);
  }

  /* Set initial register contents. */
  reg[0][0] = (a.getNThds()<<(a.getWordSize()*8 / 2)) | id;
}

void Core::step() {
  Size fetchPos(0), decPos, wordSize(a.getWordSize());
  vector<Byte> fetchBuffer(wordSize);

  if (activeThreads == 0) return;

  D(3, "in step pc=0x" << hex << pc);

  /* Fetch and decode. */
  if (wordSize < sizeof(pc)) pc &= ((1ll<<(wordSize*8))-1);
  Instruction *inst;
  bool fetchMore;
  do {
    /* Todo: speed this up for the byte encoder? */
    try {
      fetchMore = false;
      unsigned fetchSize(wordSize - (pc+fetchPos)%wordSize);
      fetchBuffer.resize(fetchPos + fetchSize);
      Word fetched = mem.fetch(pc + fetchPos, supervisorMode);
      writeWord(fetchBuffer, fetchPos, fetchSize, fetched);
      decPos = 0;
      inst = iDec.decode(fetchBuffer, decPos);
    } catch (OutOfBytes o) {
      D(3, "Caught OutOfBytes. Fetching more.");
      fetchMore = true;
    } catch (MemoryUnit::PageFault pf) {
      fetchPos = 0;
      fetchMore = true;
      interrupt(pf.notFound?1:2);
      reg[0][1] = pf.faultAddr;
    }
  } while (fetchMore);
  D(3, "Fetched at 0x" << hex << pc);
  D(3, "0x" << hex << pc << ": " << *inst);

#ifdef EMU_INSTRUMENTATION
  { Addr pcPhys(mem.virtToPhys(pc));
    Harp::OSDomain::osDomain->
      do_inst(0, pc, pcPhys, decPos, mem.getPtr(pcPhys, decPos), 
              (enum inst_type)inst->instTable[inst->getOpcode()].iType);
  }
#endif

  // Update pc
  pc += decPos;

  // Execute
  try {
    inst->executeOn(*this);
  } catch (MemoryUnit::PageFault pf) {
    pc -= decPos; /* Reset to beginning of faulting address. */
    interrupt(pf.notFound?1:2);
    reg[0][1] = pf.faultAddr;
  } catch (DivergentBranchException e) {
    pc -= decPos;
    interrupt(4);
  } catch (DomainException e) {
    interrupt(5);
  }
 
  // At Debug Level 3, print debug info after each instruction.
  #ifdef USE_DEBUG
  if (USE_DEBUG >= 3) {
    D(3, "Register state:");
    for (unsigned i = 0; i < reg[0].size(); ++i) {
      D_RAW("  %r" << dec << i << ':');
      for (unsigned j = 0; j < reg.size(); ++j) 
        D_RAW(' ' << hex << reg[j][i] << ' ');
      D_RAW('(' << shadowReg[i] << ')' << endl);
    }
    D(3, "Predicate state:");
    D_RAW("  ");
    for (unsigned j = 0; j < pred.size(); ++j) {
      for (unsigned i = 0; i < pred[j].size(); ++i) D_RAW(pred[0][i]);
      D_RAW(endl);
    }
    D_RAW(" (");
    for (unsigned i = 0; i < shadowPReg.size(); ++i) D_RAW(shadowPReg[i]);
    D_RAW(')' << endl);

    D(3, "Thread mask:");
    D_RAW("  ");
    for (unsigned i = 0; i < tmask.size(); ++i) D_RAW(tmask[i] << ' ');
    D_RAW(endl);
  }
  #endif

  // Clean up.
  delete inst;
}

bool Core::interrupt(Word r0) {
  if (!interruptEnable) return false;

#ifdef EMU_INSTRUMENTATION
  Harp::OSDomain::osDomain->do_int(0, r0);
#endif

  shadowActiveThreads = activeThreads;
  shadowTmask = tmask;
  shadowInterruptEnable = interruptEnable; /* For traps. */
  shadowSupervisorMode = supervisorMode;
  
  for (Word i = 0; i < reg[0].size(); ++i) shadowReg[i] = reg[0][i];
  for (Word i = 0; i < pred[0].size(); ++i) shadowPReg[i] = pred[0][i];
  for (Word i = 0; i < reg.size(); ++i) tmask[i] = 1;

  shadowPc = pc;
  activeThreads = 1;
  interruptEnable = false;
  supervisorMode = true;
  reg[0][0] = r0;
  pc = interruptEntry;

  return true;
}
