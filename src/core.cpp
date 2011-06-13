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

using namespace Harp;
using namespace std;

Core::Core(const ArchDef &a, Decoder &d, MemoryUnit &mem) : 
  a(a), iDec(d), mem(mem), pc(0), interruptEnable(false), supervisorMode(true),
  activeThreads(1),
  reg(a.getNThds(), vector<Word>(a.getNRegs())),
  pred(a.getNPRegs(), vector<bool>(a.getNPRegs())),
  shadowReg(), shadowPReg(), interruptEntry(0)
{ }

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

  //cout << "Interrupt: " << r0 << '\n';

  shadowActiveThreads = activeThreads;
  shadowInterruptEnable = interruptEnable; /* For traps. */
  shadowSupervisorMode = supervisorMode;
  shadowReg = reg[0];
  shadowPReg = pred[0];
  shadowPc = pc;
  activeThreads = 1;
  interruptEnable = false;
  supervisorMode = true;
  reg[0][0] = r0;
  pc = interruptEntry;

  return true;
}
