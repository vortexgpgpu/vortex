/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#include <iostream>
#include <stdlib.h>

#include "include/instruction.h"
#include "include/obj.h"
#include "include/core.h"
#include "include/harpfloat.h"

using namespace Harp;
using namespace std;

/* It is important that this stays consistent with the Harp::Instruction::Opcode
   enum. */

Instruction::InstTableEntry Instruction::instTable[] = {
  //str        cflow  relad  allsrc priv   argcl
  {"nop",      false, false, false, false, AC_NONE    },
  {"di",       false, false, false, true,  AC_NONE    },
  {"ei",       false, false, false, true,  AC_NONE    },
  {"tlbadd",   false, false, true,  true,  AC_3REGSRC },
  {"tlbflush", false, false, false, true,  AC_NONE    },
  {"neg",      false, false, false, false, AC_2REG    },
  {"not",      false, false, false, false, AC_2REG    },
  {"and",      false, false, false, false, AC_3REG    },
  {"or",       false, false, false, false, AC_3REG    },
  {"xor",      false, false, false, false, AC_3REG    },
  {"add",      false, false, false, false, AC_3REG    },
  {"sub",      false, false, false, false, AC_3REG    },
  {"mul",      false, false, false, false, AC_3REG    },
  {"div",      false, false, false, false, AC_3REG    },
  {"mod",      false, false, false, false, AC_3REG    },
  {"shl",      false, false, false, false, AC_3REG    },
  {"shr",      false, false, false, false, AC_3REG    },
  {"andi",     false, false, false, false, AC_3IMM    },
  {"ori",      false, false, false, false, AC_3IMM    },
  {"xori",     false, false, false, false, AC_3IMM    },
  {"addi",     false, false, false, false, AC_3IMM    },
  {"subi",     false, false, false, false, AC_3IMM    },
  {"muli",     false, false, false, false, AC_3IMM    },
  {"divi",     false, false, false, false, AC_3IMM    },
  {"modi",     false, false, false, false, AC_3IMM    },
  {"shli",     false, false, false, false, AC_3IMM    },
  {"shri",     false, false, false, false, AC_3IMM    },
  {"jali",     true,  true,  false, false, AC_2IMM    },
  {"jalr",     true,  false, false, false, AC_2REG    },
  {"jmpi",     true,  true,  true,  false, AC_1IMM    },
  {"jmpr",     true,  false, true,  false, AC_1REG    },
  {"clone",    true,  false, false, false, AC_1REG    },
  {"jalis",    true,  true,  false, false, AC_3IMM    },
  {"jalrs",    true,  false, false, false, AC_3REG    },
  {"jmprt",    true,  false, true,  false, AC_1REG    },
  {"ld",       false, false, false, false, AC_3IMM    },
  {"st",       false, false,  true,  false, AC_3IMMSRC },
  {"ldi",      false, false, false, false, AC_2IMM    },
  {"rtop",     false, false, false, false, AC_PREG_REG},
  {"andp",     false, false, false, false, AC_3PREG   },
  {"orp",      false, false, false, false, AC_3PREG   },
  {"xorp",     false, false, false, false, AC_3PREG   },
  {"notp",     false, false, false, false, AC_3PREG   },
  {"isneg",    false, false, false, false, AC_PREG_REG},
  {"iszero",   false, false, false, false, AC_PREG_REG},
  {"halt",     false, false, false, true,  AC_NONE    },
  {"trap",     true,  false, false, false, AC_NONE    },
  {"jmpru",    false, false, false, true,  AC_1REG    },
  {"skep",     false, false, false, true,  AC_1REG    },
  {"reti",     true,  false, false, true,  AC_NONE    },
  {"tlbrm",    false, false, false, true,  AC_1REG    },
  {"itof",     false, false, false, false, AC_2REG    },
  {"ftoi",     false, false, false, false, AC_2REG    },
  {"fadd",     false, false, false, false, AC_3REG    },
  {"fsub",     false, false, false, false, AC_3REG    },
  {"fmul",     false, false, false, false, AC_3REG    },
  {"fdiv",     false, false, false, false, AC_3REG    },
  {"fneg",     false, false, false, false, AC_2REG    },
  {NULL,false,false,false,false,AC_NONE}/////////////// End of table.
};

ostream &Harp::operator<<(ostream& os, Instruction &inst) {
  if (inst.predicated) {
    os << "@p" << inst.pred << " ? ";
  }

  os << Instruction::instTable[inst.op].opString << ' ';
  if (inst.rdestPresent) os << "%r" << inst.rdest << ' ';
  if (inst.pdestPresent) os << "@p" << inst.pdest << ' ';
  for (int i = 0; i < inst.nRsrc; i++) {
    os << "%r" << inst.rsrc[i] << ' ';
  }
  for (int i = 0; i < inst.nPsrc; i++) {
    os << "@p" << inst.psrc[i] << ' ';
  }
  if (inst.immsrcPresent) {
    if (inst.refLiteral) os << inst.refLiteral->name;
    else os << "#0x" << hex << inst.immsrc;
  }

  os << ';';
  return os;
}

void Instruction::executeOn(Core &c) {
  /* If I try to execute a privileged instruction in user mode, throw an
     exception 3. */
  if (instTable[op].privileged && !c.supervisorMode) {
    c.interrupt(3);
    return;
  }

  if (predicated && instTable[op].controlFlow) {
    bool p0 = c.pred[0][pred];
    for (Size t = 1; t < c.activeThreads; t++) {
      if (c.pred[t][pred] != p0) throw DivergentBranchException();
    }
  }

  Size nextActiveThreads = c.activeThreads;
  Size wordSz = c.a.getWordSize();

  for (Size t = 0; t < c.activeThreads; t++) {
    vector<Word> &reg(c.reg[t]);
    vector<bool> &pReg(c.pred[t]);

    if (predicated && !pReg[pred]) continue;

    Word memAddr;  
    switch (op) {
      case NOP: break;
      case DI: c.interruptEnable = false;
               break;
      case EI: c.interruptEnable = true;
               break;
      case TLBADD: c.mem.tlbAdd(reg[rsrc[0]], reg[rsrc[1]], reg[rsrc[2]]);
                   break;
      case TLBFLUSH: c.mem.tlbFlush();
                     break;
      case ADD: reg[rdest] = reg[rsrc[0]] + reg[rsrc[1]];
                break;
      case SUB: reg[rdest] = reg[rsrc[0]] - reg[rsrc[1]];
                break;
      case MUL: reg[rdest] = reg[rsrc[0]] + reg[rsrc[1]];
                break;
      case DIV: if (reg[rsrc[1]] == 0) throw DomainException();
                reg[rdest] = reg[rsrc[0]] / reg[rsrc[1]];
                break;
      case SHL: reg[rdest] = reg[rsrc[0]] << reg[rsrc[1]];
                break;
      case MOD: if (reg[rsrc[1]] == 0) throw DomainException();
                reg[rdest] = reg[rsrc[0]] % reg[rsrc[1]];
                break;
      case AND: reg[rdest] = reg[rsrc[0]] & reg[rsrc[1]];
                break;
      case NEG: reg[rdest] = -(Word_s)reg[rsrc[0]];
                break;
      case ADDI: reg[rdest] = reg[rsrc[0]] + immsrc;
                 break;
      case SUBI: reg[rdest] = reg[rsrc[0]] - immsrc;
                 break;
      case MULI: reg[rdest] = reg[rsrc[0]] * immsrc;
                 break;
      case DIVI: if (immsrc == 0) throw DomainException();
                 reg[rdest] = reg[rsrc[0]] / immsrc;
                 break;
      case MODI: if (immsrc == 0) throw DomainException();
                 reg[rdest] = reg[rsrc[0]] % immsrc;
                 break;
      case SHRI: reg[rdest] = reg[rsrc[0]] >> immsrc;
                 break;
      case SHLI: reg[rdest] = reg[rsrc[0]] << immsrc;
                 break;
      case ANDI: reg[rdest] = reg[rsrc[0]] & immsrc;
                 break;
      case JMPI: c.pc += immsrc;
                 break;
      case JALI: reg[rdest] = c.pc;
                 c.pc += immsrc;
                 break;
      case JMPR: c.pc = reg[rsrc[0]];
                 break;
      case CLONE: c.reg[reg[rsrc[0]]] = reg;
                  break;
      case JALIS: nextActiveThreads = reg[rsrc[0]];
                  reg[rdest] = c.pc;
                  c.pc += immsrc;
                  break;
      case JMPRT: nextActiveThreads = 1;
                  c.pc = reg[rsrc[0]];
                  break;
      case LD: memAddr = reg[rsrc[0]] + immsrc;
               reg[rdest] = c.mem.read(memAddr, c.supervisorMode);
               break;
      case ST: memAddr = reg[rsrc[1]] + immsrc;
               c.mem.write(memAddr, reg[rsrc[0]], c.supervisorMode);
               break;
      case LDI: reg[rdest] = immsrc;
                break;
      case RTOP: pReg[pdest] = reg[rsrc[0]];
                 break;
      case NOTP: pReg[pdest] = !(pReg[psrc[0]]);
                 break;
      case ISNEG: pReg[pdest] = (1ll<<(wordSz*8 - 1))&reg[rsrc[0]];
                  break;
      case HALT: c.activeThreads = 0;
                 nextActiveThreads = 0;
                 break;
      case TRAP: c.interrupt(0);
                 break;
      case JMPRU: c.supervisorMode = false;
                  c.pc = reg[rsrc[0]];
                  break;
      case SKEP: c.interruptEntry = reg[rsrc[0]];
                 break;
      case RETI: if (t == 0) {
                   nextActiveThreads = c.shadowActiveThreads;
                   c.interruptEnable = c.shadowInterruptEnable;
                   c.supervisorMode = c.shadowSupervisorMode;
                   reg = c.shadowReg;
                   pReg = c.shadowPReg;
                   c.pc = c.shadowPc;
                 }
                 break;
      case ITOF: reg[rdest] = Float(double(reg[rsrc[0]]), wordSz);
                 break; 
      case FTOI: reg[rdest] = Word_s(double(Float(reg[rsrc[0]], wordSz)));
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
      default:
        cout << "ERROR: Unsupported instruction: " << *this << "\n";
        exit(1);
    }

    if (instTable[op].controlFlow) break;
  }

  c.activeThreads = nextActiveThreads;
}
