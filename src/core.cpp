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
  activeThreads(1), reg(0), pred(0), shadowReg(a.getNRegs()), shadowPReg(a.getNPRegs()),
  interruptEntry(0), id(id)
{
  /* Build the register file. */
  Word regNum(0);
  for (Word j = 0; j < a.getNThds(); ++j) {
    reg.push_back(vector<Reg<Word> >(0));
    for (Word i = 0; i < a.getNRegs(); ++i)
      reg[j].push_back(Reg<Word>(id, regNum++));

    pred.push_back(vector<Reg<bool> >(0));
    for (Word i = 0; i < a.getNPRegs(); ++i)
      pred[j].push_back(Reg<bool>(id, regNum++));
  }

  /* Set initial register contents. */
  reg[0][0] = (a.getNThds()<<(a.getWordSize()*8 / 2)) | id;
}

void Core::step() {
  Size fetchPos(0), decPos, wordSize(a.getWordSize());
  vector<Byte> fetchBuffer(wordSize);

  if (activeThreads == 0) return;

  /* Fetch and decode. */
  if (wordSize < sizeof(pc)) pc &= ((1ll<<(wordSize*8))-1);
  Instruction *inst;
  bool fetchMore;
  do {
    /* Todo: speed this up for the byte encoder? */
    try {
      fetchMore = false;
      fetchBuffer.resize(fetchPos + wordSize);
      Word fetched = mem.fetch(pc + fetchPos, supervisorMode);
      writeWord(fetchBuffer, fetchPos, wordSize, fetched);
      decPos = 0;
      inst = iDec.decode(fetchBuffer, decPos);
    } catch (OutOfBytes o) {
      //cout << "Caught OutOfBytes. Fetching more.\n";
      fetchMore = true;
    } catch (MemoryUnit::PageFault pf) {
      fetchPos = 0;
      fetchMore = true;
      reg[0][1] = pf.faultAddr;
      interrupt(pf.notFound?1:2);
    }
  } while (fetchMore);
  //cout << "0x" << hex << pc << ": " << *inst << '\n';

#ifdef EMU_INSTRUMENTATION
  { Addr pcPhys(mem.virtToPhys(pc));
    Harp::OSDomain::osDomain->
      do_inst(0, pc, pcPhys, decPos, mem.getPtr(pcPhys, decPos), 
              (enum inst_type)inst->instTable[inst->getOpcode()].iType);
  }
#endif

  /* Update pc */
  pc += decPos;

  /* Execute */
  try {
    inst->executeOn(*this);
  } catch (MemoryUnit::PageFault pf) {
    pc -= decPos; /* Reset to beginning of faulting address. */
    reg[0][1] = pf.faultAddr;
    interrupt(pf.notFound?1:2);
  } catch (DivergentBranchException e) {
    pc -= decPos;
    interrupt(4);
  } catch (DomainException e) {
    interrupt(5);
  }

  /* Clean up. */
  delete inst;
}

bool Core::interrupt(Word r0) {
  if (!interruptEnable) return false;

#ifdef EMU_INSTRUMENTATION
  Harp::OSDomain::osDomain->do_int(0, r0);
#endif

  shadowActiveThreads = activeThreads;
  shadowInterruptEnable = interruptEnable; /* For traps. */
  shadowSupervisorMode = supervisorMode;
  
  for (Word i = 0; i < reg[0].size(); ++i) shadowReg[i] = reg[0][i];
  for (Word i = 0; i < pred[0].size(); ++i) shadowPReg[i] = pred[0][i];

  shadowPc = pc;
  activeThreads = 1;
  interruptEnable = false;
  supervisorMode = true;
  reg[0][0] = r0;
  pc = interruptEntry;

  return true;
}
