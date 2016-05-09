/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#include <iostream>
#include <stdlib.h>

#include "include/instruction.h"
#include "include/obj.h"
#include "include/core.h"
#include "include/harpfloat.h"
#include "include/debug.h"

#ifdef EMU_INSTRUMENTATION
#include "include/qsim-harp.h"
#endif

using namespace Harp;
using namespace std;

/* It is important that this stays consistent with the Harp::Instruction::Opcode
   enum. */

Instruction::InstTableEntry Instruction::instTable[] = {
  //str        cflow  relad  allsrc priv   argcl        itype
  {"nop",      false, false, false, false, AC_NONE,     ITYPE_NULL    },
  {"di",       false, false, false, true,  AC_NONE,     ITYPE_NULL    },
  {"ei",       false, false, false, true,  AC_NONE,     ITYPE_NULL    },
  {"tlbadd",   false, false, true,  true,  AC_3REGSRC,  ITYPE_NULL    },
  {"tlbflush", false, false, false, true,  AC_NONE,     ITYPE_NULL    },
  {"neg",      false, false, false, false, AC_2REG,     ITYPE_INTBASIC},
  {"not",      false, false, false, false, AC_2REG,     ITYPE_INTBASIC},
  {"and",      false, false, false, false, AC_3REG,     ITYPE_INTBASIC},
  {"or",       false, false, false, false, AC_3REG,     ITYPE_INTBASIC},
  {"xor",      false, false, false, false, AC_3REG,     ITYPE_INTBASIC},
  {"add",      false, false, false, false, AC_3REG,     ITYPE_INTBASIC},
  {"sub",      false, false, false, false, AC_3REG,     ITYPE_INTBASIC},
  {"mul",      false, false, false, false, AC_3REG,     ITYPE_INTMUL  },
  {"div",      false, false, false, false, AC_3REG,     ITYPE_INTDIV  },
  {"mod",      false, false, false, false, AC_3REG,     ITYPE_INTDIV  },
  {"shl",      false, false, false, false, AC_3REG,     ITYPE_INTBASIC},
  {"shr",      false, false, false, false, AC_3REG,     ITYPE_INTBASIC},
  {"andi",     false, false, false, false, AC_3IMM,     ITYPE_INTBASIC},
  {"ori",      false, false, false, false, AC_3IMM,     ITYPE_INTBASIC},
  {"xori",     false, false, false, false, AC_3IMM,     ITYPE_INTBASIC},
  {"addi",     false, false, false, false, AC_3IMM,     ITYPE_INTBASIC},
  {"subi",     false, false, false, false, AC_3IMM,     ITYPE_INTBASIC},
  {"muli",     false, false, false, false, AC_3IMM,     ITYPE_INTMUL  },
  {"divi",     false, false, false, false, AC_3IMM,     ITYPE_INTDIV  },
  {"modi",     false, false, false, false, AC_3IMM,     ITYPE_INTDIV  },
  {"shli",     false, false, false, false, AC_3IMM,     ITYPE_INTBASIC},
  {"shri",     false, false, false, false, AC_3IMM,     ITYPE_INTBASIC},
  {"jali",     true,  true,  false, false, AC_2IMM,     ITYPE_CALL    },
  {"jalr",     true,  false, false, false, AC_2REG,     ITYPE_CALL    },
  {"jmpi",     true,  true,  true,  false, AC_1IMM,     ITYPE_BR      },
  {"jmpr",     true,  false, true,  false, AC_1REG,     ITYPE_RET     },
  {"clone",    true,  false, false, false, AC_1REG,     ITYPE_NULL    },
  {"jalis",    true,  true,  false, false, AC_3IMM,     ITYPE_CALL    },
  {"jalrs",    true,  false, false, false, AC_3REG,     ITYPE_CALL    },
  {"jmprt",    true,  false, true,  false, AC_1REG,     ITYPE_RET     },
  {"ld",       false, false, false, false, AC_3IMM,     ITYPE_NULL    },
  {"st",       false, false, true,  false, AC_3IMMSRC,  ITYPE_NULL    },
  {"ldi",      false, false, false, false, AC_2IMM,     ITYPE_NULL    },
  {"rtop",     false, false, false, false, AC_PREG_REG, ITYPE_NULL    },
  {"andp",     false, false, false, false, AC_3PREG,    ITYPE_INTBASIC},
  {"orp",      false, false, false, false, AC_3PREG,    ITYPE_INTBASIC},
  {"xorp",     false, false, false, false, AC_3PREG,    ITYPE_INTBASIC},
  {"notp",     false, false, false, false, AC_2PREG,    ITYPE_INTBASIC},
  {"isneg",    false, false, false, false, AC_PREG_REG, ITYPE_INTBASIC},
  {"iszero",   false, false, false, false, AC_PREG_REG, ITYPE_INTBASIC},
  {"halt",     false, false, false, false, AC_NONE,     ITYPE_NULL    },
  {"trap",     true,  false, false, false, AC_NONE,     ITYPE_TRAP    },
  {"jmpru",    false, false, false, true,  AC_1REG,     ITYPE_RET     },
  {"skep",     false, false, false, true,  AC_1REG,     ITYPE_NULL    },
  {"reti",     true,  false, false, true,  AC_NONE,     ITYPE_RET     },
  {"tlbrm",    false, false, false, true,  AC_1REG,     ITYPE_NULL    },
  {"itof",     false, false, false, false, AC_2REG,     ITYPE_FPBASIC },
  {"ftoi",     false, false, false, false, AC_2REG,     ITYPE_FPBASIC },
  {"fadd",     false, false, false, false, AC_3REG,     ITYPE_FPBASIC },
  {"fsub",     false, false, false, false, AC_3REG,     ITYPE_FPBASIC },
  {"fmul",     false, false, false, false, AC_3REG,     ITYPE_FPMUL   },
  {"fdiv",     false, false, false, false, AC_3REG,     ITYPE_FPDIV   },
  {"fneg",     false, false, false, false, AC_2REG,     ITYPE_FPBASIC },
  {"wspawn",   false, false, true,  false, AC_3REG,     ITYPE_NULL    },
  {"split",    false, false, true,  false, AC_NONE,     ITYPE_NULL    },
  {"join",     false, false, true,  false, AC_NONE,     ITYPE_NULL    },
  {"bar",      false, false, true,  false, AC_2REGSRC,  ITYPE_NULL    },
  {NULL,false,false,false,false,AC_NONE,ITYPE_NULL}/////// End of table.
};

ostream &Harp::operator<<(ostream& os, Instruction &inst) {
  os << dec;

  if (inst.predicated) {
    os << "@p" << dec << inst.pred << " ? ";
  }

  os << Instruction::instTable[inst.op].opString << ' ';
  if (inst.rdestPresent) os << "%r" << dec << inst.rdest << ' ';
  if (inst.pdestPresent) os << "@p" << inst.pdest << ' ';
  for (int i = 0; i < inst.nRsrc; i++) {
    os << "%r" << dec << inst.rsrc[i] << ' ';
  }
  for (int i = 0; i < inst.nPsrc; i++) {
    os << "@p" << dec << inst.psrc[i] << ' ';
  }
  if (inst.immsrcPresent) {
    if (inst.refLiteral) os << inst.refLiteral->name;
    else os << "#0x" << hex << inst.immsrc;
  }

  os << ';';
  return os;
}

bool checkUnanimous(unsigned p, const std::vector<std::vector<Reg<bool> > >& m,
  const std::vector<bool> &tm) {
  bool same;
  unsigned i;
  for (i = 0; i < m.size(); ++i) {
    if (tm[i]) {
      same = m[i][p];
      break;
    }
  }
  if (i == m.size())
    throw DivergentBranchException();
  for (; i < m.size(); ++i) {
    if (tm[i]) {
      if (same != (bool(m[i][p]))) {
        return false;
      }
    } 
  }
  return true;
}

void Instruction::executeOn(Warp &c) {
  D(3, "Begin instruction execute.");

  /* If I try to execute a privileged instruction in user mode, throw an
     exception 3. */
  if (instTable[op].privileged && !c.supervisorMode) {
    c.interrupt(3);
    return;
  }

  /* Also throw exceptions on non-masked divergent branches. */
  if (instTable[op].controlFlow) {
    Size t, count, active;
    for (t = 0, count = 0, active = 0; t < c.activeThreads; ++t) {
      if ((!predicated || c.pred[t][pred]) && c.tmask[t]) ++count;
      if (c.tmask[t]) ++active;
    }

    if (count != 0 && count != active)
      throw DivergentBranchException();
  }

  Size nextActiveThreads = c.activeThreads;
  Size wordSz = c.core->a.getWordSize();
  Word nextPc = c.pc;

  c.memAccesses.clear();
  
  // If we have a load, overwriting a register's contents, we have to make sure
  // ahead of time it will not fault. Otherwise we may perform an indirect load
  // by mistake.
  if (op == LD && rdest == rsrc[0]) {
    for (Size t = 0; t < c.activeThreads; t++) {
      if ((!predicated || c.pred[t][pred]) && c.tmask[t]) {
        Word memAddr = c.reg[t][rsrc[0]] + immsrc;
        c.core->mem.read(memAddr, c.supervisorMode);
      }
    }
  }

  bool sjOnce(true), // Has not yet split or joined once.
       pcSet(false); // PC has already been set
  for (Size t = 0; t < c.activeThreads; t++) {
    vector<Reg<Word> > &reg(c.reg[t]);
    vector<Reg<bool> > &pReg(c.pred[t]);
    stack<DomStackEntry> &domStack(c.domStack);

    // If this thread is masked out, don't execute the instruction, unless it's
    // a split or join.
    if (((predicated && !pReg[pred]) || !c.tmask[t]) &&
          op != SPLIT && op != JOIN) continue;

    ++c.insts;
    
    Word memAddr;  
    switch (op) {
      case NOP: break;
      case DI: c.interruptEnable = false;
               break;
      case EI: c.interruptEnable = true;
               break;
      case TLBADD: c.core->mem.tlbAdd(reg[rsrc[0]], reg[rsrc[1]], reg[rsrc[2]]);
                   break;
      case TLBFLUSH: c.core->mem.tlbFlush();
                     break;
      case ADD: reg[rdest] = reg[rsrc[0]] + reg[rsrc[1]];
                reg[rdest].trunc(wordSz);
                break;
      case SUB: reg[rdest] = reg[rsrc[0]] - reg[rsrc[1]];
                reg[rdest].trunc(wordSz);
                break;
      case MUL: reg[rdest] = reg[rsrc[0]] * reg[rsrc[1]];
                reg[rdest].trunc(wordSz);
                break;
      case DIV: if (reg[rsrc[1]] == 0) throw DomainException();
                reg[rdest] = reg[rsrc[0]] / reg[rsrc[1]];
                break;
      case SHL: reg[rdest] = reg[rsrc[0]] << reg[rsrc[1]];
                reg[rdest].trunc(wordSz);
                break;
      case SHR: reg[rdest] = Word_s(reg[rsrc[0]]) >> reg[rsrc[1]];
                reg[rdest].trunc(wordSz);
                break;
      case MOD: if (reg[rsrc[1]] == 0) throw DomainException();
                reg[rdest] = reg[rsrc[0]] % reg[rsrc[1]];
                break;
      case AND: reg[rdest] = reg[rsrc[0]] & reg[rsrc[1]];
                break;
      case OR:  reg[rdest] = reg[rsrc[0]] | reg[rsrc[1]];
                break;
      case XOR: reg[rdest] = reg[rsrc[0]] ^ reg[rsrc[1]];
                break;
      case NEG: reg[rdest] = -(Word_s)reg[rsrc[0]];
                reg[rdest].trunc(wordSz);
                break;
      case NOT: reg[rdest] = ~(Word_s)reg[rsrc[0]];
                reg[rdest].trunc(wordSz);
                break;
      case ADDI: reg[rdest] = reg[rsrc[0]] + immsrc;
                 reg[rdest].trunc(wordSz);
                 break;
      case SUBI: reg[rdest] = reg[rsrc[0]] - immsrc;
                 reg[rdest].trunc(wordSz);
                 break;
      case MULI: reg[rdest] = reg[rsrc[0]] * immsrc;
                 reg[rdest].trunc(wordSz);
                 break;
      case DIVI: if (immsrc == 0) throw DomainException();
                 reg[rdest] = reg[rsrc[0]] / immsrc;
                 break;
      case MODI: if (immsrc == 0) throw DomainException();
                 reg[rdest] = reg[rsrc[0]] % immsrc;
                 break;
      case SHRI: reg[rdest] = Word_s(reg[rsrc[0]]) >> immsrc;
                 break;
      case SHLI: reg[rdest] = reg[rsrc[0]] << immsrc;
                 reg[rdest].trunc(wordSz);
                 break;
      case ANDI: reg[rdest] = reg[rsrc[0]] & immsrc;
                 break;
      case ORI:  reg[rdest] = reg[rsrc[0]] | immsrc;
                 break;
      case XORI: reg[rdest] = reg[rsrc[0]] ^ immsrc;
                 break;
      case JMPI: if (!pcSet) nextPc = c.pc + immsrc;
                 pcSet = true;
                 break;
      case JALI: reg[rdest] = c.pc;
                 if (!pcSet) nextPc = c.pc + immsrc;
                 pcSet = true;
                 break;
      case JALR: reg[rdest] = c.pc;
                 if (!pcSet) nextPc = reg[rsrc[0]];
                 pcSet = true;
                 break;
      case JMPR: if (!pcSet) nextPc = reg[rsrc[0]];
                 pcSet = true;
                 break;
      case CLONE: c.reg[reg[rsrc[0]]] = reg;
                  break;
      case JALIS: nextActiveThreads = reg[rsrc[0]];
                  reg[rdest] = c.pc;
                  if (!pcSet) nextPc = c.pc + immsrc;
                  pcSet = true;
                  break;
      case JALRS: nextActiveThreads = reg[rsrc[0]];
                  reg[rdest] = c.pc;
                  if (!pcSet) nextPc = reg[rsrc[1]];
                  pcSet = true;
                  break;
      case JMPRT: nextActiveThreads = 1;
                  if (!pcSet) nextPc = reg[rsrc[0]];
                  pcSet = true;
                  break;
      case LD: ++c.loads;
	       memAddr = reg[rsrc[0]] + immsrc;
#ifdef EMU_INSTRUMENTATION
               Harp::OSDomain::osDomain->
                 do_mem(0, memAddr, c.core->mem.virtToPhys(memAddr), 8, true);
#endif
               reg[rdest] = c.core->mem.read(memAddr, c.supervisorMode);
               c.memAccesses.push_back(Warp::MemAccess(false, memAddr));
               break;
      case ST: ++c.stores;
	       memAddr = reg[rsrc[1]] + immsrc;
               c.core->mem.write(memAddr, reg[rsrc[0]], c.supervisorMode);
               c.memAccesses.push_back(Warp::MemAccess(true, memAddr));
#ifdef EMU_INSTRUMENTATION
               Harp::OSDomain::osDomain->
                 do_mem(0, memAddr, c.core->mem.virtToPhys(memAddr), 8, true);
#endif
               break;
      case LDI: reg[rdest] = immsrc;
                reg[rdest].trunc(wordSz);
                break;
      case RTOP: pReg[pdest] = reg[rsrc[0]];
                 break;
      case ISZERO: pReg[pdest] = !reg[rsrc[0]];
                   break;
      case NOTP: pReg[pdest] = !(pReg[psrc[0]]);
                 break;
      case ANDP: pReg[pdest] = pReg[psrc[0]] & pReg[psrc[1]];
                 break;
      case ORP: pReg[pdest] = pReg[psrc[0]] | pReg[psrc[1]];
                break;
      case XORP: pReg[pdest] = pReg[psrc[0]] != pReg[psrc[1]];
                 break;
      case ISNEG: pReg[pdest] = (1ll<<(wordSz*8 - 1))&reg[rsrc[0]];
                  break;
      case HALT: c.activeThreads = 0;
                 nextActiveThreads = 0;
                 break;
      case TRAP: c.interrupt(0);
                 break;
      case JMPRU: c.supervisorMode = false;
                  if (!pcSet) nextPc = reg[rsrc[0]];
                  pcSet = true;
                  break;
      case SKEP: c.core->interruptEntry = reg[rsrc[0]];
                 break;
      case RETI: if (t == 0) {
                   c.tmask = c.shadowTmask;
                   nextActiveThreads = c.shadowActiveThreads;
                   c.interruptEnable = c.shadowInterruptEnable;
                   c.supervisorMode = c.shadowSupervisorMode;
                   for (unsigned i = 0; i < reg.size(); ++i)
                     reg[i] = c.shadowReg[i];
                   for (unsigned i = 0; i < pReg.size(); ++i)
                     pReg[i] = c.shadowPReg[i];
                   if (!pcSet) { nextPc = c.shadowPc; pcSet = true; }
                 }
                 break;
      case ITOF: reg[rdest] = Float(double(Word_s(reg[rsrc[0]])), wordSz);
                 break; 
      case FTOI: reg[rdest] = Word_s(double(Float(reg[rsrc[0]], wordSz)));
                 reg[rdest].trunc(wordSz);
                 break;
      case FNEG: reg[rdest] = Float(-double(Float(reg[rsrc[0]],wordSz)),wordSz);
                 break;
      case FADD: reg[rdest] = Float(double(Float(reg[rsrc[0]], wordSz)) +
                                    double(Float(reg[rsrc[1]], wordSz)),wordSz);
                 break;
      case FSUB: reg[rdest] = Float(double(Float(reg[rsrc[0]], wordSz)) -
                                    double(Float(reg[rsrc[1]], wordSz)),wordSz);
                 break;
      case FMUL: reg[rdest] = Float(double(Float(reg[rsrc[0]], wordSz)) *
                                    double(Float(reg[rsrc[1]], wordSz)),wordSz);
                 break;
      case FDIV: reg[rdest] = Float(double(Float(reg[rsrc[0]], wordSz)) /
                                    double(Float(reg[rsrc[1]], wordSz)),wordSz);
                 break;
      case SPLIT: if (sjOnce) {
		   sjOnce = false;
                    if (checkUnanimous(pred, c.pred, c.tmask)) {
                      DomStackEntry e(c.tmask);
                      e.uni = true;
                      c.domStack.push(e);
                      break;
                    }
                   DomStackEntry e(pred, c.pred, c.tmask, c.pc);
                   c.domStack.push(c.tmask);
                   c.domStack.push(e);
                   for (unsigned i = 0; i < e.tmask.size(); ++i)
                     c.tmask[i] = !e.tmask[i] && c.tmask[i];
                 }
                 break;
      case JOIN: if (sjOnce) {
		   sjOnce = false;
                   if (!c.domStack.empty() && c.domStack.top().uni) {
                     D(2, "Uni branch at join");
  		               c.tmask = c.domStack.top().tmask;
                     c.domStack.pop();
                     break;
                   }
                   if (!c.domStack.top().fallThrough) {
                     if (!pcSet) nextPc = c.domStack.top().pc;
                     pcSet = true;
                   }
		   c.tmask = c.domStack.top().tmask;
                   c.domStack.pop();
                 }
	         break;
      case WSPAWN: if (sjOnce) {
                     sjOnce = false;
                     D(0, "Spawning a new warp.");
                     for (unsigned i = 0; i < c.core->w.size(); ++i) {
                       Warp &newWarp(c.core->w[i]);
                       if (newWarp.spawned == false) {
                         newWarp.pc = reg[rsrc[0]];
                         newWarp.reg[0][rdest] = reg[rsrc[1]];
                         newWarp.activeThreads = 1;
		         newWarp.supervisorMode = false;
                         newWarp.spawned = true;
                         break;
                       }
                     }
                   }
                   break;
      case BAR: if (sjOnce) {
                  sjOnce = false;
                  Word id(reg[rsrc[0]]), n(reg[rsrc[1]]);
                  set<Warp*> &b(c.core->b[id]);

                  // Add current warp to the barrier and halt.
                  b.insert(&c);
                  c.shadowActiveThreads = c.activeThreads;
                  nextActiveThreads = 0;

                  D(2, "Barrier " << id << ' ' << b.size() << " of " << n);

                  // If the barrier's full, reactivate warps waiting at it
                  if (b.size() == n) {
                    set<Warp*>::iterator it;
                    for (it = b.begin(); it != b.end(); ++it)
                      (*it)->activeThreads = (*it)->shadowActiveThreads;
                    c.core->b.erase(id);
                    nextActiveThreads = c.shadowActiveThreads;
                  }
		}
                break;

      default:
        cout << "ERROR: Unsupported instruction: " << *this << "\n";
        exit(1);
    }
  }

  D(3, "End instruction execute.");

  c.activeThreads = nextActiveThreads;

  // This way, if pc was set by a side effect (such as interrupt), it will
  // retain its new value.
  if (pcSet) c.pc = nextPc;
  
  if (nextActiveThreads > c.reg.size()) {
    cerr << "Error: attempt to spawn " << nextActiveThreads << " threads. "
         << c.reg.size() << " available.\n";
    abort();
  }
}
