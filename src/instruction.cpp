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

ostream &Harp::operator<<(ostream& os, Instruction &inst) {
  os << dec;

  // if (inst.predicated) {
  //   os << "@p" << dec << inst.pred << " ? ";
  // }

  // os << inst.instTable[inst.op].opString << ' ';
  // if (inst.rdestPresent) os << "%r" << dec << inst.rdest << ' ';
  // if (inst.pdestPresent) os << "@p" << inst.pdest << ' ';
  // for (int i = 0; i < inst.nRsrc; i++) {
  //   os << "%r" << dec << inst.rsrc[i] << ' ';
  // }
  // for (int i = 0; i < inst.nPsrc; i++) {
  //   os << "@p" << dec << inst.psrc[i] << ' ';
  // }
  // if (inst.immsrcPresent) {
  //   if (inst.refLiteral) os << inst.refLiteral->name;
  //   else os << "#0x" << hex << inst.immsrc;
  // }

  os << instTable[inst.op].opString;

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

Word signExt(Word w, Size bit, Word mask) {
  if (w>>(bit-1)) w |= ~mask;
  return w;
}

void Instruction::executeOn(Warp &c) {
  D(3, "Begin instruction execute.");

  /* If I try to execute a privileged instruction in user mode, throw an
     exception 3. */
  if (instTable[op].privileged && !c.supervisorMode) {
    std::cout << "INTERRUPT SUPERVISOR\n";
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
  if (op == L_INST && rdest == rsrc[0]) {
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
    // if (((predicated && !pReg[pred]) || !c.tmask[t]) &&
    //       op != SPLIT && op != JOIN) continue;

    ++c.insts;

    Word memAddr;
    Word shift_by;
    Word shamt;
    Word temp;
    Word data_read;
    int op1, op2;
    switch (op) {

      case NOP: break;
      case R_INST:
        switch (func3)
        {
          case 0:
            if (func7)
            {
                reg[rdest] = reg[rsrc[0]] - reg[rsrc[1]];
                reg[rdest].trunc(wordSz);
            }
            else
            {
                reg[rdest] = reg[rsrc[0]] + reg[rsrc[1]];
                reg[rdest].trunc(wordSz);
            }
            break;
          case 1:
                reg[rdest] = reg[rsrc[0]] << reg[rsrc[1]];
                reg[rdest].trunc(wordSz);
            break;
          case 2:
            if ( int(reg[rsrc[0]]) <  int(reg[rsrc[1]]))
            {
              reg[rdest] = 1;
            }
            else
            {
              reg[rdest] = 0;
            }
            break;
          case 3:
            if ( Word_u(reg[rsrc[0]]) <  Word_u(reg[rsrc[1]]))
            {
              reg[rdest] = 1;
            }
            else
            {
              reg[rdest] = 0;
            }
            break;
          case 4:
            reg[rdest] = reg[rsrc[0]] ^ reg[rsrc[1]];
            break;
          case 5:
            if (func7)
            {
                reg[rdest] = int(reg[rsrc[0]]) >> int(reg[rsrc[1]]);
                reg[rdest].trunc(wordSz);
            }
            else
            {
                reg[rdest] = Word_u(reg[rsrc[0]]) >> Word_u(reg[rsrc[1]]);
                reg[rdest].trunc(wordSz);
            }
            break;
          case 6:
            reg[rdest] = reg[rsrc[0]] | reg[rsrc[1]];
            break;
          case 7:
            reg[rdest] = reg[rsrc[0]] & reg[rsrc[1]];
            break;
          default:
            cout << "ERROR: UNSUPPORTED R INST\n";
            exit(1);
        }
        break;

      case L_INST:
           
           memAddr   = ((reg[rsrc[0]] + immsrc) & 0xFFFFFFFC);
           shift_by  = ((reg[rsrc[0]] + immsrc) & 0x00000003) * 8;
           data_read = c.core->mem.read(memAddr, c.supervisorMode);
           // std::cout <<std::hex<< "EXECUTE: " << reg[rsrc[0]] << " + " << immsrc << " = " << memAddr <<  " -> data_read: " << data_read << "\n";
#ifdef EMU_INSTRUMENTATION
           Harp::OSDomain::osDomain->
             do_mem(0, memAddr, c.core->mem.virtToPhys(memAddr), 8, true);
#endif
        switch (func3)
        {

          case 0:
            // LB
            reg[rdest] = signExt((data_read >> shift_by) & 0xFF, 8, 0xFF);
            break;
          case 1:
            // LH
            // std::cout << "shifting by: " << shift_by << "  final data: " << ((data_read >> shift_by) & 0xFFFF, 16, 0xFFFF) << "\n";
            reg[rdest] = signExt((data_read >> shift_by) & 0xFFFF, 16, 0xFFFF);
            break;
          case 2:
            reg[rdest] = int(data_read & 0xFFFFFFFF);
            break;
          case 4:
            // LBU
            reg[rdest] = unsigned((data_read >> shift_by) & 0xFF);
            break;
          case 5:
            reg[rdest] = unsigned((data_read >> shift_by) & 0xFFFF);
            break;
          default:
            cout << "ERROR: UNSUPPORTED L INST\n";
            exit(1);
          c.memAccesses.push_back(Warp::MemAccess(false, memAddr));
        }
        break;
      case I_INST:
        switch (func3)
        {

          case 0:
            // ADDI
            reg[rdest] = reg[rsrc[0]] + immsrc;
            reg[rdest].trunc(wordSz);
            break;
          case 2:
            // SLTI
            if ( int(reg[rsrc[0]]) <  int(immsrc))
            {
              reg[rdest] = 1;
            }
            else
            {
              reg[rdest] = 0;
            }
            break;
          case 3:
            // SLTIU
            op1 = (unsigned) reg[rsrc[0]];
            if ( unsigned(reg[rsrc[0]]) <  unsigned(immsrc))
            {
              reg[rdest] = 1;
            }
            else
            {
              reg[rdest] = 0;
            }
            break;
          case 4:
            // XORI
            reg[rdest] = reg[rsrc[0]] ^ immsrc;
            break;
          case 6:
            // ORI;
            reg[rdest] = reg[rsrc[0]] | immsrc;
            break;
          case 7:
            // ANDI
            reg[rdest] = reg[rsrc[0]] & immsrc;
            break;
          case 1:
            // SLLI
            reg[rdest] = reg[rsrc[0]] << immsrc;
            reg[rdest].trunc(wordSz);
            break;
          case 5:
            if (!func7)
            {
              // SRAI
                op1 = reg[rsrc[0]];
                op2 = immsrc;
                reg[rdest] = op1 >> op2;
                reg[rdest].trunc(wordSz);
            }
            else
            {
              // SRLI
                reg[rdest] = Word_u(reg[rsrc[0]]) >> Word_u(immsrc);
                reg[rdest].trunc(wordSz);
            }
            break;
          default:
            cout << "ERROR: UNSUPPORTED L INST\n";
            exit(1);
        }
        break;
      case S_INST:
        ++c.stores;
        memAddr = reg[rsrc[0]] + immsrc;
        // std::cout << "STORE MEM ADDRESS: " << std::hex << reg[rsrc[0]] << " + " << immsrc << "\n";
        switch (func3)
        {
          case 0:
            c.core->mem.write(memAddr, reg[rsrc[1]] & 0x000000FF, c.supervisorMode, 1);
            break;
          case 1:
            // std::cout << std::hex << "INST: about to write: " << reg[rsrc[1]] << " to " << memAddr << "\n"; 
            c.core->mem.write(memAddr, reg[rsrc[1]], c.supervisorMode, 2);
            break;
          case 2:
            c.core->mem.write(memAddr, reg[rsrc[1]], c.supervisorMode, 4);
            break;
          default:
            cout << "ERROR: UNSUPPORTED S INST\n";
            exit(1);
        }
        c.memAccesses.push_back(Warp::MemAccess(true, memAddr));
#ifdef EMU_INSTRUMENTATION
       Harp::OSDomain::osDomain->
       do_mem(0, memAddr, c.core->mem.virtToPhys(memAddr), 8, true);
#endif
        break;
      case B_INST:
        switch (func3)
        {
          case 0:
            // BEQ
            if (int(reg[rsrc[0]]) == int(reg[rsrc[1]]))
            {
              if (!pcSet) nextPc = (c.pc - 4) + immsrc;
              pcSet = true;
            }
            break;
          case 1:
            // BNE
            if (int(reg[rsrc[0]]) != int(reg[rsrc[1]]))
            {
              if (!pcSet) nextPc = (c.pc - 4) + immsrc;
              pcSet = true;
            }
            break;
          case 4:
            // BLT
            if (int(reg[rsrc[0]]) < int(reg[rsrc[1]]))
            {
              if (!pcSet) nextPc = (c.pc - 4) + immsrc;
              pcSet = true;
            }
            break;
          case 5:
            // BGE
            if (int(reg[rsrc[0]]) >= int(reg[rsrc[1]]))
            {
              if (!pcSet) nextPc = (c.pc - 4) + immsrc;
              pcSet = true;
            }
            break;
          case 6:
            // BLTU
            if (Word_u(reg[rsrc[0]]) < Word_u(reg[rsrc[1]]))
            {
              if (!pcSet) nextPc = (c.pc - 4) + immsrc;
              pcSet = true;
            }
            break;
          case 7:
            // BGEU
            if (Word_u(reg[rsrc[0]]) >= Word_u(reg[rsrc[1]]))
            {
              if (!pcSet) nextPc = (c.pc - 4) + immsrc;
              pcSet = true;
            }
            break;
        }
        break;
      case LUI_INST:
        reg[rdest] = (immsrc << 12) & 0xfffff000;
        break;
      case AUIPC_INST:
        reg[rdest] = ((immsrc << 12) & 0xfffff000) + (c.pc - 4);
        break;
      case JAL_INST:
        if (!pcSet) nextPc = (c.pc - 4) + immsrc;
        if (rdest != 0)
        {
          reg[rdest] = c.pc;
        }
        pcSet = true;
        break;
      case JALR_INST:
        if (!pcSet) nextPc = reg[rsrc[0]] + immsrc;

        if (rdest != 0)
        {
          reg[rdest] = c.pc;
        }
        pcSet = true;
        break;
      case SYS_INST:
        temp = reg[rsrc[0]];
        switch (func3)
        {
          case 1:
            if (rdest != 0)
            {
              reg[rdest] = c.csr[immsrc & 0x00000FFF];
            }
              c.csr[immsrc & 0x00000FFF] = temp;
            
            break;
          case 2:
            if (rdest != 0)
            {
              reg[rdest] = c.csr[immsrc & 0x00000FFF];
            }
              c.csr[immsrc & 0x00000FFF] = temp |  c.csr[immsrc & 0x00000FFF];
            
            break;
          case 3:
            if (rdest != 0)
            {              
              reg[rdest]                 = c.csr[immsrc & 0x00000FFF];
            }
              c.csr[immsrc & 0x00000FFF] = temp &  (~c.csr[immsrc & 0x00000FFF]);
            
            break;
          case 5:
            if (rdest != 0)
            {
              reg[rdest] = c.csr[immsrc & 0x00000FFF];
            }
              c.csr[immsrc & 0x00000FFF] = rsrc[0];
            
            break;
          case 6:
            if (rdest != 0)
            {              
              reg[rdest]                 = c.csr[immsrc & 0x00000FFF];
            }
              c.csr[immsrc & 0x00000FFF] = rsrc[0] |  c.csr[immsrc & 0x00000FFF];
            
            break;
          case 7:
            if (rdest != 0)
            {              
              reg[rdest] = c.csr[immsrc & 0x00000FFF];
            }
              c.csr[immsrc & 0x00000FFF] = rsrc[0] &  (~c.csr[immsrc & 0x00000FFF]);
            
            break;
          case 0:
          if (immsrc < 2)
          {
            std::cout << "INTERRUPT ECALL/EBREAK\n";
            nextActiveThreads = 0;
            c.interrupt(0);
          }
            break;
          default:
            break;
        }
        break;
      case TRAP:
        std::cout << "INTERRUPT TRAP\n";
        nextActiveThreads = 0;
        c.interrupt(0);
        break;
      case FENCE:
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
