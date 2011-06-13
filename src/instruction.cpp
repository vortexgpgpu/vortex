/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#include <iostream>
#include <stdlib.h>

#include "include/instruction.h"
#include "include/obj.h"
#include "include/core.h"

using namespace Harp;
using namespace std;

/* It is important that this stays consistent with the Harp::Instruction::Opcode
   enum. */
const char *Instruction::opStrings[] = {
  "nop", "di", "ei", "tlbadd", "tlbflush", "neg", "not", "and", "or", "xor",
  "add", "sub", "mul", "div", "mod", "shl", "shr","andi", 
  "ori", "xori", "addi", "subi", "muli", "divi", "modi", "shli", "shri", 
  "jali", "jalr", "jmpi", "jmpr", "clone", "jalis", "jalrs",
  "jmprt", "ld", "st", "ldi", "rtop", "andp", "orp", "xorp", "notp", 
  "isneg", "iszero", "halt", "trap", "jmpru", "skep", "reti", "tlbrm", 0
};

const bool Instruction::isControlFlow[] = {
  false, false, false, false, false, false, false, false, false, false,
  false, false, false, false, false, false, false, false,
  false, false, false, false, false, false, false, false, false,
  true,  true,  true,  true,  true,  true,  true,
  true,  false, false, false, false, false, false, false, false,
  false, false, false, false, true,  false, true,  false
};

const bool Instruction::relAddress[] = {
  false, false, false, false, false, false, false, false, false, false,
  false, false, false, false, false, false, false, false,
  false, false, false, false, false, false, false, false, false,
  true,  false, true,  false, false, true,  false,
  false, false, false, false, false, false, false, false, false,
  false, false, false, false, false, false, false, false
};

const bool Instruction::allSrcArgs[] = {
  false, false, false, true,  false, false, false, false, false, false,
  false, false, false, false, false, false, false, false,
  false, false, false, false, false, false, false, false, false,
  false, false, true,  true,  false, false, false,
  true,  false, true,  false, false, false, false, false, false,
  false, false, false, false, true,  true,  false, false
};

const bool Instruction::privileged[] = {
  false, true,  true,  true,  true,  false, false, false, false, false,
  false, false, false, false, false, false, false, false,
  false, false, false, false, false, false, false, false, false,
  false, false, false, false, false, false, false,
  false, false, false, false, false, false, false, false, false,
  false, false, true,  false, true,  true,  true,  true
};

const Instruction::ArgClass Instruction::argClasses[] = {
  AC_NONE, AC_NONE, AC_NONE, AC_3REGSRC, AC_NONE, AC_2REG, AC_2REG, AC_3REG,
  AC_3REG, AC_3REG,
  AC_3REG, AC_3REG, AC_3REG, AC_3REG, AC_3REG, AC_3REG, AC_3REG, AC_3IMM,
  AC_3IMM, AC_3IMM, AC_3IMM, AC_3IMM, AC_3IMM, AC_3IMM, AC_3IMM, AC_3IMM,
  AC_3IMM,
  AC_2IMM, AC_2REG, AC_1IMM, AC_1REG, AC_1REG, AC_3IMM, AC_3REG,
  AC_1REG, AC_3IMM, AC_3IMMSRC, AC_2IMM, AC_PREG_REG, AC_3PREG, AC_3PREG, 
  AC_3PREG, AC_2PREG,
  AC_PREG_REG, AC_PREG_REG, AC_NONE, AC_NONE, AC_1REG, AC_1REG, AC_NONE,
  AC_1REG
};

ostream &Harp::operator<<(ostream& os, Instruction &inst) {
  if (inst.predicated) {
    os << "@p" << inst.pred << " ? ";
  }

  os << Instruction::opStrings[inst.op] << ' ';
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
  if (privileged[op] && !c.supervisorMode) {
    c.interrupt(3);
    return;
  }

  if (predicated && isControlFlow[op]) {
    bool p0 = c.pred[0][pred];
    for (Size t = 1; t < c.activeThreads; t++) {
      if (c.pred[t][pred] != p0) throw DivergentBranchException();
    }
  }

  Size nextActiveThreads = c.activeThreads;

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
      case ISNEG: pReg[pdest] = (1ll<<(c.a.getWordSize()*8 - 1))&reg[rsrc[0]];
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
      default:
        cout << "ERROR: Unsupported instruction: " << *this << "\n";
        exit(1);
    }

    if (isControlFlow[op]) break;
  }

  c.activeThreads = nextActiveThreads;
}
