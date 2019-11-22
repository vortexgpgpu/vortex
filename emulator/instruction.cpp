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

bool checkUnanimous(unsigned p, const std::vector<std::vector<Reg<Word> > >& m,
  const std::vector<bool> &tm) {
  bool same;
  unsigned i;
  for (i = 0; i < m.size(); ++i) {
    if (tm[i]) {
      same = m[i][p];
      break;
    }
  }
  if (i == m.size()) throw DivergentBranchException();

  //std::cout << "same: " << same << "  with -> ";
  for (; i < m.size(); ++i) {
    if (tm[i]) {
      //std::cout << " " << (bool(m[i][p]));
      if (same != (bool(m[i][p]))) {
        //std::cout << " FALSE\n";
        return false;
      }
    } 
  }
  //std::cout << " TRUE\n";
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

  // /* Also throw exceptions on non-masked divergent branches. */
  // if (instTable[op].controlFlow) {
  //   Size t, count, active;
  //   for (t = 0, count = 0, active = 0; t < c.activeThreads; ++t) {
  //     if ((!predicated || c.pred[t][pred]) && c.tmask[t]) ++count;
  //     if (c.tmask[t]) ++active;
  //   }

  //   if (count != 0 && count != active)
  //     throw DivergentBranchException();
  // }

  Size nextActiveThreads = c.activeThreads;
  Size wordSz = c.core->a.getWordSize();
  Word nextPc = c.pc;

  c.memAccesses.clear();
  
  // If we have a load, overwriting a register's contents, we have to make sure
  // ahead of time it will not fault. Otherwise we may perform an indirect load
  // by mistake.
  // if (op == L_INST && rdest == rsrc[0]) {
  //   for (Size t = 0; t < c.activeThreads; t++) {
  //     if ((!predicated || c.pred[t][pred]) && c.tmask[t]) {
  //       Word memAddr = c.reg[t][rsrc[0]] + immsrc;
  //       c.core->mem.read(memAddr, c.supervisorMode);
  //     }
  //   }
  // }

  bool sjOnce(true), // Has not yet split or joined once.
       pcSet(false); // PC has already been set
  for (Size t = 0; t < c.activeThreads; t++) {
    vector<Reg<Word> > &reg(c.reg[t]);
    vector<Reg<bool> > &pReg(c.pred[t]);
    stack<DomStackEntry> &domStack(c.domStack);

    //std::cout << std::hex << "opcode: " << op << "  func3: " << func3 << "\n";
    //if (op == GPGPU) //std::cout << "OPCODE MATCHED GPGPU\n";

    // If this thread is masked out, don't execute the instruction, unless it's
    // a split or join.
    // if (((predicated && !pReg[pred]) || !c.tmask[t]) &&
    //       op != SPLIT && op != JOIN) continue;

    bool split = (op == GPGPU) && (func3 == 2);
    bool join  = (op == GPGPU) && (func3 == 3);


    // predicated = (op == GPGPU) && ((func3 == 7) || (func3 == 2));


    // bool is_branch = (op == B_INST);
    // bool is_jump   = (op == JAL_INST) || (op == JALR_INST); 

    bool is_gpgpu   = (op == GPGPU);

    bool is_tmc     = is_gpgpu && (func3 == 0); 
    bool is_wspawn  = is_gpgpu && (func3 == 1); 
    bool is_barrier = is_gpgpu && (func3 == 4); 
    bool is_split   = is_gpgpu && (func3 == 2); 
    bool is_join    = is_gpgpu && (func3 == 3);

    bool gpgpu_zero = (is_tmc || is_barrier || is_wspawn) && (t != 0);

    bool not_active = !c.tmask[t];

    if (not_active || gpgpu_zero)
    {
      continue;
    }

    // printf("Predicated: %d, split: %d, join: %d\n",predicated, split, join );
    // printf("%d && ((%d) || (%d))\n",(op == GPGPU), (func3 == 7), (func3 == 2) );

    // cout << "before " << op << " = " << GPGPU << "\n";
    // if (((predicated && !reg[pred]) || !c.tmask[t]) && !split && !join)
    // {
    //   // cout << "about to continue\n";
    //   continue;
    // }
    // cout << "after\n";

    ++c.insts;

    Word memAddr;
    Word shift_by;
    Word shamt;
    Word temp;
    Word data_read;
    int op1, op2;
    bool m_exten;
    // std::cout << "op = " << op << "\n";
    // std::cout << "R_INST: " << R_INST << "\n";
    int num_to_wspawn;
    switch (op) {

      case NOP:
        //std::cout << "NOP_INST\n";
        break;
      case R_INST:
        // std::cout << "R_INST\n";
        m_exten = func7 & 0x1;

        if (m_exten)
        {
          // std::cout << "FOUND A MUL/DIV\n";

          switch (func3)
          {
            case 0:
              // MUL
              // cout << "MUL\n";
              reg[rdest] = ((int) reg[rsrc[0]]) * ((int) reg[rsrc[1]]);
              break;
            case 1:
              // MULH
              {
                int64_t first  = (int64_t) reg[rsrc[0]];
                if (reg[rsrc[0]] & 0x80000000)
                {
                  first = first | 0xFFFFFFFF00000000;
                }
                int64_t second = (int64_t) reg[rsrc[1]];
                if (reg[rsrc[1]] & 0x80000000)
                {
                  second = second | 0xFFFFFFFF00000000;
                }
                // cout << "mulh: " << std::dec << first << " * " << second;
                uint64_t result = first * second;
                reg[rdest] = ( result >> 32) & 0xFFFFFFFF;
                // cout << " = " << result << "   or  " <<  reg[rdest] << "\n";
              }
              break;
            case 2:
              // MULHSU
              {
                int64_t first  = (int64_t) reg[rsrc[0]];
                if (reg[rsrc[0]] & 0x80000000)
                {
                  first = first | 0xFFFFFFFF00000000;
                }
                int64_t second = (int64_t) reg[rsrc[1]];
                reg[rdest] = (( first * second ) >> 32) & 0xFFFFFFFF;
              }
              break;
            case 3:
              // MULHU
              {
                uint64_t first  = (uint64_t) reg[rsrc[0]];
                uint64_t second = (uint64_t) reg[rsrc[1]];
                // cout << "MULHU\n";
                reg[rdest] = (( first * second) >> 32) & 0xFFFFFFFF;
              }
                break;
            case 4:
              // DIV
              if (reg[rsrc[1]] == 0) 
              {
                reg[rdest] = -1;
                break;
              }
              // cout << "dividing: " << dec << ((int) reg[rsrc[0]]) << " / " << ((int) reg[rsrc[1]]);
              reg[rdest] = ( (int) reg[rsrc[0]]) / ( (int) reg[rsrc[1]]);
              // cout << " = " << ((int) reg[rdest]) << "\n";
              break;
            case 5:
              // DIVU
              if (reg[rsrc[1]] == 0) 
              {
                reg[rdest] = -1;
                break;
              }
              reg[rdest] = ((uint32_t) reg[rsrc[0]]) / ((uint32_t) reg[rsrc[1]]);
              break;
            case 6:
              // REM
              if (reg[rsrc[1]] == 0) 
              {
                reg[rdest] = reg[rsrc[0]];
                break;
              }
              reg[rdest] = ((int) reg[rsrc[0]]) % ((int) reg[rsrc[1]]);
              break;
            case 7:
              // REMU
              if (reg[rsrc[1]] == 0) 
              {
                reg[rdest] = reg[rsrc[0]];
                break;
              }
              reg[rdest] = ((uint32_t) reg[rsrc[0]]) % ((uint32_t) reg[rsrc[1]]);
              break;
            default:
              cout << "unsupported MUL/DIV instr\n";
              exit(1);
          }
        }
        else
        {
          // std::cout << "NORMAL R-TYPE\n";
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
        }
        break;
      case L_INST:
           //std::cout << "L_INST\n";
           memAddr   = ((reg[rsrc[0]] + immsrc) & 0xFFFFFFFC);
           shift_by  = ((reg[rsrc[0]] + immsrc) & 0x00000003) * 8;
           data_read = c.core->mem.read(memAddr, c.supervisorMode);
           // std::cout << std::hex << "EXECUTE: " << reg[rsrc[0]] << " + " << immsrc << " = " << memAddr <<  " -> data_read: " << data_read << "\n";

           D(3, "LOAD WORD: " << hex << "0x" << memAddr << " = " << data_read << dec << '\n');

        switch (func3)
        {

          case 0:
            // LB
            reg[rdest] = signExt((data_read >> shift_by) & 0xFF, 8, 0xFF);
            break;
          case 1:
            // LH
            // //std::cout << "shifting by: " << shift_by << "  final data: " << ((data_read >> shift_by) & 0xFFFF, 16, 0xFFFF) << "\n";
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
        //std::cout << "I_INST\n";
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
            if ((func7 == 0))
            {
              // SRLI
                // //std::cout << "WTF\n";
                bool isNeg  = ((0x80000000 & reg[rsrc[0]])) > 0;
                Word result = Word_u(reg[rsrc[0]]) >> Word_u(immsrc);
                // if (isNeg)
                // {
                //   Word mask = 0x80000000;
                //   for (int i = 32; i < Word_u(immsrc); i++)
                //   {
                //     result |= mask;
                //     mask = mask >> 1;
                //   }
                // }

                reg[rdest] = result;
      
                reg[rdest].trunc(wordSz);
            }
            else
            {
                // SRAI
                // //std::cout << "WOHOOOOO\n";
                op1 = reg[rsrc[0]];
                op2 = immsrc;
                reg[rdest] = op1 >> op2;
                reg[rdest].trunc(wordSz);
            }
            break;
          default:
            cout << "ERROR: UNSUPPORTED L INST\n";
            exit(1);
        }
        break;
      case S_INST:
        //std::cout << "S_INST\n";
        ++c.stores;
        memAddr = reg[rsrc[0]] + immsrc;
           D(3, "STORE WORD: " << hex << "0x" << memAddr << " = " << reg[rsrc[1]] << dec << '\n');
        // std::cout << "STORE MEM ADDRESS: " << std::hex << reg[rsrc[0]] << " + " << immsrc << "\n";
        // //std::cout << "FUNC3: " << func3 << "\n";
        if ((memAddr == 0x00010000) && (t == 0))
        {
          unsigned num = reg[rsrc[1]];
          fprintf(stderr, "%c", (char) reg[rsrc[1]]);
          break;
        }
        switch (func3)
        {
          case 0:
            // //std::cout << "SB\n";
            c.core->mem.write(memAddr, reg[rsrc[1]] & 0x000000FF, c.supervisorMode, 1);
            break;
          case 1:
            // //std::cout << "SH\n";
            c.core->mem.write(memAddr, reg[rsrc[1]], c.supervisorMode, 2);
            break;
          case 2:
            // //std::cout << std::hex << "SW: about to write: " << reg[rsrc[1]] << " to " << memAddr << "\n"; 
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
        //std::cout << "B_INST\n";
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
        //std::cout << "LUI_INST\n";
        reg[rdest] = (immsrc << 12) & 0xfffff000;
        break;
      case AUIPC_INST:
        //std::cout << "AUIPC_INST\n";
        reg[rdest] = ((immsrc << 12) & 0xfffff000) + (c.pc - 4);
        break;
      case JAL_INST:
        std::cout << "JAL_INST\n";
        if (!pcSet) nextPc = (c.pc - 4) + immsrc;

        if (!pcSet)
        {
          std::cout << "JAL... immsrc: " << hex << immsrc << "\n";
          std::cout << "JAL... pc base: " << hex << (c.pc - 4) << "\n";
          std::cout << "JAL... SETTING PC: " << nextPc << "\n";
        }
        if (rdest != 0)
        {
          reg[rdest] = c.pc;
        }
        pcSet = true;
        break;
      case JALR_INST:
        std::cout << "JALR_INST\n";
        if (!pcSet) nextPc = reg[rsrc[0]] + immsrc;
        if (!pcSet) {/*std::cout << "JALR... SETTING PC: " << nextPc << "\n";*/ }
        if (rdest != 0)
        {
          reg[rdest] = c.pc;
        }
        pcSet = true;
        break;
      case SYS_INST:
        //std::cout << "SYS_INST\n";
        temp = reg[rsrc[0]];
        if (immsrc == 0x20) // ThreadID
        {
          reg[rdest] = t;
          D(2, "CSR Reading tid " << hex << immsrc << dec << " and returning " << reg[rdest]);
        } else if (immsrc == 0x21) // WarpID
        {
          reg[rdest] = c.id;
          D(2, "CSR Reading wid " << hex << immsrc << dec << " and returning " << reg[rdest]);
        }
        // switch (func3)
        // {
        //   case 1:
        //     // printf("Case 1\n");
        //     if (rdest != 0)
        //     {
        //       reg[rdest] = c.csr[immsrc & 0x00000FFF];
        //     }
        //     c.csr[immsrc & 0x00000FFF] = temp;
            
        //     break;
        //   case 2:
        //     // printf("Case 2\n");
        //     if (rdest != 0)
        //     {
        //       // printf("Reading from CSR: %d = %d\n", (immsrc & 0x00000FFF),  c.csr[immsrc & 0x00000FFF]);
        //       reg[rdest] = c.csr[immsrc & 0x00000FFF];
        //     }
        //     // printf("Writing to CSR --> %d = %d\n", immsrc,  (temp |  c.csr[immsrc & 0x00000FFF]));
        //     c.csr[immsrc & 0x00000FFF] = temp |  c.csr[immsrc & 0x00000FFF];
            
        //     break;
        //   case 3:
        //     // printf("Case 3\n");
        //     if (rdest != 0)
        //     {              
        //       reg[rdest]                 = c.csr[immsrc & 0x00000FFF];
        //     }
        //       c.csr[immsrc & 0x00000FFF] = temp &  (~c.csr[immsrc & 0x00000FFF]);
            
        //     break;
        //   case 5:
        //     // printf("Case 5\n");
        //     if (rdest != 0)
        //     {
        //       reg[rdest] = c.csr[immsrc & 0x00000FFF];
        //     }
        //       c.csr[immsrc & 0x00000FFF] = rsrc[0];
            
        //     break;
        //   case 6:
        //     // printf("Case 6\n");
        //     if (rdest != 0)
        //     {              
        //       reg[rdest]                 = c.csr[immsrc & 0x00000FFF];
        //     }
        //       c.csr[immsrc & 0x00000FFF] = rsrc[0] |  c.csr[immsrc & 0x00000FFF];
            
        //     break;
        //   case 7:
        //     // printf("Case 7\n");
        //     if (rdest != 0)
        //     {              
        //       reg[rdest] = c.csr[immsrc & 0x00000FFF];
        //     }
        //       c.csr[immsrc & 0x00000FFF] = rsrc[0] &  (~c.csr[immsrc & 0x00000FFF]);
            
        //     break;
        //   case 0:
        //   if (immsrc < 2)
        //   {
        //     //std::cout << "INTERRUPT ECALL/EBREAK\n";
        //     nextActiveThreads = 0;
        //     c.spawned = false;
        //     // c.interrupt(0);
        //   }
        //     break;
        //   default:
        //     break;
        // }
        break;
      case TRAP:
        //std::cout << "INTERRUPT TRAP\n";
        nextActiveThreads = 0;
        c.interrupt(0);
        break;
      case FENCE:
        //std::cout << "FENCE_INST\n";
        break;
      case PJ_INST:
        // pred jump reg
        //std::cout << "pred jump... src: " << rsrc[0] << std::hex << " val: " << reg[rsrc[0]] << " dest: " <<  reg[rsrc[1]] << "\n";
        if (reg[rsrc[0]])
        {
          if (!pcSet) nextPc = reg[rsrc[1]];
          pcSet = true;
        }
        break;
      case GPGPU:
        //std::cout << "GPGPU\n";
        switch(func3)
        {
          case 1:
            // WSPAWN
            std::cout << "WSPAWN\n";
            if (sjOnce)
            {
              sjOnce = false;
              // //std::cout << "SIZE: " << c.core->w.size() << "\n";
              num_to_wspawn = reg[rsrc[0]];

              D(0, "Spawning " << num_to_wspawn << " new warps at PC: " << hex << reg[rsrc[1]]);
              for (unsigned i = 1; i < num_to_wspawn; ++i)
              {
                // std::cout << "SPAWNING WARP\n";
                Warp &newWarp(c.core->w[i]);
                // //std::cout << "STARTING\n";
                // if (newWarp.spawned == false)
                {
                  // //std::cout << "ABOUT TO START\n";
                  newWarp.pc     = reg[rsrc[1]];
                  // newWarp.reg[0] = reg;
                  // newWarp.csr = c.csr;
                  for (int kk = 0; kk < newWarp.tmask.size(); kk++)
                  {
                    if (kk == 0)
                    {
                      newWarp.tmask[kk] = true;
                    }
                    else
                    {
                      newWarp.tmask[kk] = false;
                    }
                  }
                  newWarp.activeThreads = 1;
                  newWarp.supervisorMode = false;
                  newWarp.spawned = true;
                }
              }
              break;
            }
            break;
          case 2:
          {
            // SPLIT
            //std::cout << "SPLIT\n";
            if (sjOnce)
            {
              sjOnce = false;
              if (checkUnanimous(pred, c.reg, c.tmask)) {
                std::cout << "Unanimous pred: " << pred << "  val: " << reg[pred] << "\n";
                DomStackEntry e(c.tmask);
                e.uni = true;
                c.domStack.push(e);
                break;
              }
              cout << "Split: Original TM: ";
              for (auto y : c.tmask) cout << y << " ";
              cout << "\n";

              DomStackEntry e(pred, c.reg, c.tmask, c.pc);
              c.domStack.push(c.tmask);
              c.domStack.push(e);
              for (unsigned i = 0; i < e.tmask.size(); ++i)
              {
                c.tmask[i] = !e.tmask[i] && c.tmask[i];
              }


              cout << "Split: New TM\n";
              for (auto y : c.tmask) cout << y << " ";
              cout << "\n";
              cout << "Split: Pushed TM PC: " << hex << e.pc << dec << "\n";
              for (auto y : e.tmask) cout << y << " ";
              cout << "\n";
            }
            break;
          }
          case 3:
            // JOIN
            //std::cout << "JOIN\n";
            D(3, "JOIN INSTRUCTION");
            if (sjOnce)
            {
              sjOnce = false;
              if (!c.domStack.empty() && c.domStack.top().uni) {
                D(2, "Uni branch at join");
                printf("NEW DOMESTACK: \n");
                c.tmask = c.domStack.top().tmask;
                c.domStack.pop();
                break;
              }
              if (!c.domStack.top().fallThrough) {
                if (!pcSet) {
                  nextPc = c.domStack.top().pc;
                  cout << "join: NOT FALLTHROUGH PC: " << hex << nextPc << dec << '\n';
                }
                  pcSet = true;
              }

              cout << "Join: Old TM: ";
              for (auto y : c.tmask) cout << y << " ";
              cout << "\n";
              c.tmask = c.domStack.top().tmask;

              cout << "Join: New TM: " << '\n';
              for (auto y : c.tmask) cout << y << " ";
              cout << "\n";

              c.domStack.pop();
            }
            break;
          case 4:
            // is_barrier
            break;
          case 0:
            // TMC
            //std::cout << "JALRS\n";

            nextActiveThreads = reg[rsrc[0]];
            {
              for (int ff = 0; ff < c.tmask.size(); ff++)
              {
                if (ff < nextActiveThreads)
                {
                  c.tmask[ff] = true;
                }
                else
                {
                  c.tmask[ff] = false;
                }
              }
            }
            if (nextActiveThreads == 0)
            {
              c.spawned = false;
            }
            // reg[rdest] = c.pc;
            // if (!pcSet) nextPc = reg[rsrc[0]];
            // pcSet = true;
            // //std::cout << "ACTIVE_THREDS: " << rsrc[1] << " val: " << reg[rsrc[1]] << "\n";
            // //std::cout << "nextPC: " << rsrc[0] << " val: " << std::hex << reg[rsrc[0]] << "\n";
            break;
          default:
            cout << "ERROR: UNSUPPORTED GPGPU INSTRUCTION " << *this << "\n";
        }
        break;
      default:
        cout << "pc: " << hex << (c.pc) << "\n";
        cout << "aERROR: Unsupported instruction: " << *this << "\n" << flush;
        exit(1);
    }
  }

  D(3, "End instruction execute.");

  c.activeThreads = nextActiveThreads;

  // if (nextActiveThreads != 0)
  // {
  //   for (int i = 7; i >= c.activeThreads; i--)
  //   {
  //     c.tmask[i] = c.tmask[i] && false;
  //   }
  // }



  // //std::cout << "new thread mask: ";
  // for (int i = 0; i < c.tmask.size(); ++i) //std::cout << " " << c.tmask[i];
  // //std::cout << "\n";

  // This way, if pc was set by a side effect (such as interrupt), it will
  // retain its new value.
  if (pcSet)
  {
    c.pc = nextPc;
    cout << "Next PC: " << hex << nextPc << dec << "\n";
  }
  
  if (nextActiveThreads > c.reg.size()) {
    cerr << "Error: attempt to spawn " << nextActiveThreads << " threads. "
         << c.reg.size() << " available.\n";
    abort();
  }
}
