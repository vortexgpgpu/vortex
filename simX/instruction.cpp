/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#include <iostream>
#include <stdlib.h>
#include <math.h>

#include "include/instruction.h"
#include "include/obj.h"
#include "include/core.h"
#include "include/harpfloat.h"
#include "include/debug.h"

#ifdef EMU_INSTRUMENTATION
#include "include/qsim-harp.h"
#endif
#include <fcntl.h>
 #include <sys/types.h>
#include <sys/stat.h>

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

void upload(unsigned * addr, char * src, int size, Warp & c)
{

  // cerr << "WRITING FINAL: " << *src << " size: " << size << "\n";

  unsigned current_addr = *addr;

  c.core->mem.write(current_addr, size, c.supervisorMode, 4);
  current_addr += 4;


  for (int i = 0; i < size; i++)
  {
    unsigned value = src[i] & 0x000000FF;
    // cerr << "UPLOAD: (" << hex << current_addr << dec << ") = " << hex << ( value) << dec << "\n";
    c.core->mem.write(current_addr, value, c.supervisorMode, 1);
    current_addr += 1;
  }

  current_addr += (current_addr % 4);

  *addr = current_addr;
}

void download(unsigned * addr, char * drain, Warp & c)
{
  unsigned current_addr = *addr;

  int size;

  size = c.core->mem.read(current_addr, c.supervisorMode);
  current_addr += 4;


  for (int i = 0; i < size; i++)
  {
    unsigned read_word = c.core->mem.read(current_addr, c.supervisorMode);
    char     read_byte = (char) (read_word & 0x000000FF);
    drain[i] = read_byte;
    current_addr += 1;
  }

  current_addr += (current_addr % 4);

  *addr = current_addr;
}

void downloadAlloc(unsigned * addr, char ** drain_ptr, int & size, Warp & c)
{
  unsigned current_addr = *addr;

  size = c.core->mem.read(current_addr, c.supervisorMode);
  current_addr += 4;

  (*drain_ptr) = (char *) malloc(size);

  char * drain = *drain_ptr;

  for (int i = 0; i < size; i++)
  {
    unsigned read_word = c.core->mem.read(current_addr, c.supervisorMode);
    char     read_byte = (char) (read_word & 0x000000FF);
    drain[i] = read_byte;
    current_addr += 1;
  }

  *addr = current_addr;
}

#define CLOSE  1
#define ISATTY 2
#define LSEEK  3
#define READ   4
#define WRITE  5
#define FSTAT  6
#define OPEN   7

void trap_to_simulator(Warp & c)
{
    unsigned read_buffer  = 0x71000000;
    unsigned write_buffer = 0x72000000;

    // cerr << "RAW READ BUFFER:\n";
    // for (int i = 0; i < 10; i++)
    // {
    //     unsigned new_addr = read_buffer + (4*i);
    //     unsigned data_read = c.core->mem.read(new_addr, c.supervisorMode); 
    //     cerr << hex << new_addr << ": " << data_read << "\n";
    // }

    for (int j = 0; j < 1024; j+=1)
    {
      c.core->mem.write((write_buffer+j), 0, c.supervisorMode, 1);
    }

    int command;
    download(&read_buffer, (char *) &command, c);

    // cerr << "Command: " << hex << command << dec << '\n';

    switch (command)
    {
        case(CLOSE):
        {
            cerr << "trap_to_simulator: CLOSE not supported yet\n";
        }
        break;
        case(ISATTY):
        {

            cerr << "trap_to_simulator: ISATTY not supported yet\n";
        }
        break;
        case (LSEEK):
        {

            // cerr << "trap_to_simulator: LSEEK not supported yet\n";
            int fd;
            int offset;
            int whence;

            download(&read_buffer, (char *) &fd     , c);
            download(&read_buffer, (char *) &offset , c);
            download(&read_buffer, (char *) &whence , c);


            int retval = lseek(fd, offset, whence);

            upload(&write_buffer, (char *) &retval, sizeof(int), c);

        }
        break;
        case (READ):
        {

            // cerr << "trap_to_simulator: READ not supported yet\n";
            int file;
            unsigned ptr;
            int len;

            download(&read_buffer, (char *) &file    , c);
            download(&read_buffer, (char *) &ptr     , c);
            download(&read_buffer, (char *) &len     , c);

            char * buff = (char *) malloc(len);

            int ret = read(file, buff, len);

            for (int i = 0; i < len; i++)
            {
              c.core->mem.write(ptr, buff[i], c.supervisorMode, 1);
              ptr++;
            }
            // c.core->mem.write(ptr, 0, c.supervisorMode, 1);
            free(buff);

        }
        break;
        case (WRITE):
        {
            int file;
            download(&read_buffer, (char *) &file, c);

            file = (file == 1) ? 2 : file;

            int size;
            char * buf;
            downloadAlloc(&read_buffer, &buf, size, c);

            int e = write(file, buf, size);
            free(buf);
        }
        break;
        case (FSTAT):
        {
            cerr << "trap_to_simulator: FSTAT not supported yet\n";
            int file;
            download(&read_buffer, (char *) &file, c);

            struct stat st;
            fstat(file, &st);

            fprintf(stderr, "------------------------\n");
            fprintf(stderr, "Size of struct: %ld\n", sizeof(struct stat));
            fprintf(stderr, "st_mode: %x\n", st.st_mode);
            fprintf(stderr, "st_dev: %ld\n", st.st_dev);
            fprintf(stderr, "st_ino: %ld\n", st.st_ino);
            fprintf(stderr, "st_uid: %x\n", st.st_uid);
            fprintf(stderr, "st_gid: %x\n", st.st_gid);
            fprintf(stderr, "st_rdev: %ld\n", st.st_rdev);
            fprintf(stderr, "st_size: %ld\n", st.st_size);
            fprintf(stderr, "st_blksize: %ld\n", st.st_blksize);
            fprintf(stderr, "st_blocks: %ld\n", st.st_blocks);
            fprintf(stderr, "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n");

            upload(&write_buffer, (char *) &st.st_mode    , sizeof(st.st_mode), c);
            upload(&write_buffer, (char *) &st.st_dev     , sizeof(st.st_dev), c);
            // upload(&write_buffer, (char *) &st.st_uid     , sizeof(st.st_uid), c);
            // upload(&write_buffer, (char *) &st.st_gid     , sizeof(st.st_gid), c);
            // upload(&write_buffer, (char *) &st.st_size    , sizeof(st.st_size), c);
            // upload(&write_buffer, (char *) &st.st_blksize , sizeof(st.st_blksize), c);
            // upload(&write_buffer, (char *) &st.st_blocks  , sizeof(st.st_blocks), c);

            // upload(&write_buffer, (char *) &st, sizeof(struct stat), c);

            cerr << "RAW Write BUFFER:\n";
            unsigned original_write_buffer = 0x72000000;
            for (int i = 0; i < 10; i++)
            {
                unsigned new_addr = original_write_buffer + (4*i);
                unsigned data_read = c.core->mem.read(new_addr, c.supervisorMode); 
                cerr << hex << new_addr << ": " << data_read << "\n";
            }
        }
        break;
        case (OPEN):
        {
          // cerr << "$$$$$$$$$$$$$$$$$$$$$$$$$ OPEN FROM simX\n";
          unsigned name_ptr;
          unsigned flags;
          unsigned mode;

          download(&read_buffer, (char *) &name_ptr, c);
          download(&read_buffer, (char *) &flags   , c);
          download(&read_buffer, (char *) &mode    , c);

          char buffer[255];
          unsigned read_word;
          char     read_byte;

          int curr_ind = 0;

          read_word = c.core->mem.read(name_ptr, c.supervisorMode);
          read_byte = (char) (read_word & 0x000000FF);
          while (read_byte != 0)
          {
            buffer[curr_ind] = read_byte;

            name_ptr++;
            curr_ind++;
            read_word = c.core->mem.read(name_ptr, c.supervisorMode);
            read_byte = (char) (read_word & 0x000000FF);
          }
          buffer[curr_ind] = 0; 


          int fd = open(buffer, flags, mode);

          // fprintf(stderr, "Name: --%s-- and fd: %d\n", buffer, fd);

          upload(&write_buffer, (char *) &fd, sizeof(int), c);


        }
        break;
        default:
        {

            cerr << "trap_to_simulator: DEFAULT not supported yet\n";
        }
        break;
    }

}

void Instruction::executeOn(Warp &c, trace_inst_t * trace_inst) {
  /* If I try to execute a privileged instruction in user mode, throw an
     exception 3. */
  if (instTable[op].privileged && !c.supervisorMode) {
    D(3, "INTERRUPT SUPERVISOR\n");
    c.interrupt(3);
    return;
  }

  bool is_vec = false;

  Size nextActiveThreads = c.activeThreads;
  Size wordSz = c.core->a.getWordSize();
  Word nextPc = c.pc;
  Word VLMAX;

  c.memAccesses.clear();
  

  unsigned real_pc = c.pc - 4;
  if ((real_pc) == (0x70000000))
  {
    trap_to_simulator(c);
  }

  bool sjOnce(true), // Has not yet split or joined once.
       pcSet(false); // PC has already been set
  for (Size t = 0; t < c.activeThreads; t++) {
    vector<Reg<Word> > &reg(c.reg[t]);
    vector<Reg<bool> > &pReg(c.pred[t]);
    stack<DomStackEntry> &domStack(c.domStack);


    bool split = (op == GPGPU) && (func3 == 2);
    bool join  = (op == GPGPU) && (func3 == 3);


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
              D(3, "MUL: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
              reg[rdest] = ((int) reg[rsrc[0]]) * ((int) reg[rsrc[1]]);
              break;
            case 1:
              // MULH
              D(3, "MULH: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
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
              D(3, "MULHSU: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
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
              D(3, "MULHU: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
              {
                uint64_t first  = (uint64_t) reg[rsrc[0]];
                uint64_t second = (uint64_t) reg[rsrc[1]];
                // cout << "MULHU\n";
                reg[rdest] = (( first * second) >> 32) & 0xFFFFFFFF;
              }
                break;
            case 4:
              // DIV
              D(3, "DIV: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
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
              D(3, "DIVU: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
              if (reg[rsrc[1]] == 0) 
              {
                reg[rdest] = -1;
                break;
              }
              reg[rdest] = ((uint32_t) reg[rsrc[0]]) / ((uint32_t) reg[rsrc[1]]);
              break;
            case 6:
              // REM
              D(3, "REM: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
              if (reg[rsrc[1]] == 0) 
              {
                reg[rdest] = reg[rsrc[0]];
                break;
              }
              reg[rdest] = ((int) reg[rsrc[0]]) % ((int) reg[rsrc[1]]);
              break;
            case 7:
              // REMU
              D(3, "REMU: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
              if (reg[rsrc[1]] == 0) 
              {
                reg[rdest] = reg[rsrc[0]];
                break;
              }
              reg[rdest] = ((uint32_t) reg[rsrc[0]]) % ((uint32_t) reg[rsrc[1]]);
              break;
            default:
              cout << "unsupported MUL/DIV instr\n";
              std::abort();
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
                  D(3, "SUBI: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
                  reg[rdest] = reg[rsrc[0]] - reg[rsrc[1]];
                  reg[rdest].trunc(wordSz);
              }
              else
              {
                  D(3, "ADDI: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
                  reg[rdest] = reg[rsrc[0]] + reg[rsrc[1]];
                  reg[rdest].trunc(wordSz);
              }
              break;
            case 1:
              D(3, "SLLI: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
              reg[rdest] = reg[rsrc[0]] << reg[rsrc[1]];
              reg[rdest].trunc(wordSz);
              break;
            case 2:
              D(3, "SLTI: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
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
              D(3, "SLTU: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
              if (Word_u(reg[rsrc[0]]) <  Word_u(reg[rsrc[1]]))
              {
                reg[rdest] = 1;
              }
              else
              {
                reg[rdest] = 0;
              }
              break;
            case 4:
              D(3, "XORI: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
              reg[rdest] = reg[rsrc[0]] ^ reg[rsrc[1]];
              break;
            case 5:
              if (func7)
              {  
                  D(3, "SRLI: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
                  reg[rdest] = int(reg[rsrc[0]]) >> int(reg[rsrc[1]]);
                  reg[rdest].trunc(wordSz);
              }
              else
              {
                  D(3, "SRLU: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
                  reg[rdest] = Word_u(reg[rsrc[0]]) >> Word_u(reg[rsrc[1]]);
                  reg[rdest].trunc(wordSz);
              }
              break;
            case 6:
              D(3, "ORI: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
              reg[rdest] = reg[rsrc[0]] | reg[rsrc[1]];
              break;
            case 7:
              D(3, "ANDI: r" << rdest << " <- r" << rsrc[0] << ", r" << rsrc[1]);
              reg[rdest] = reg[rsrc[0]] & reg[rsrc[1]];
              break;
            default:
              cout << "ERROR: UNSUPPORTED R INST\n";
              std::abort();
          }
        }
        break;
      case L_INST:
        memAddr   = ((reg[rsrc[0]] + immsrc) & 0xFFFFFFFC);
        shift_by  = ((reg[rsrc[0]] + immsrc) & 0x00000003) * 8;
        data_read = c.core->mem.read(memAddr, c.supervisorMode);
        trace_inst->is_lw = true;
        trace_inst->mem_addresses[t] = memAddr;
        switch (func3) {
          case 0:
            // LBI
            D(3, "LBI: r" << rdest << " <- r" << rsrc[0] << ", imm=" << (int)immsrc);
            reg[rdest] = signExt((data_read >> shift_by) & 0xFF, 8, 0xFF);
            break;
          case 1:
            // LWI
             D(3, "LWI: r" << rdest << " <- r" << rsrc[0] << ", imm=" << (int)immsrc);
            reg[rdest] = signExt((data_read >> shift_by) & 0xFFFF, 16, 0xFFFF);
            break;
          case 2:
            // LDI
            D(3, "LDI: r" << rdest << " <- r" << rsrc[0] << ", imm=" << (int)immsrc); 
            reg[rdest] = int(data_read & 0xFFFFFFFF);
            break;
          case 4:
            // LBU
            D(3, "LBU: r" << rdest << " <- r" << rsrc[0] << ", imm=" << (int)immsrc);
            reg[rdest] = unsigned((data_read >> shift_by) & 0xFF);
            break;
          case 5:
            // LWU
            D(3, "LWU: r" << rdest << " <- r" << rsrc[0] << ", imm=" << (int)immsrc);
            reg[rdest] = unsigned((data_read >> shift_by) & 0xFFFF);
            break;
          default:
            cout << "ERROR: UNSUPPORTED L INST\n";
            std::abort();
          c.memAccesses.push_back(Warp::MemAccess(false, memAddr));
        }
        D(3, "LOAD MEM ADDRESS: " << std::hex << memAddr);
        D(3, "LOAD MEM DATA: " << std::hex << data_read);
        break;
      case I_INST:
        //std::cout << "I_INST\n";
        switch (func3)
        {
          case 0:
            // ADDI
            D(3, "ADDI: r" << rdest << " <- r" << rsrc[0] << ", imm=" << immsrc);
            reg[rdest] = reg[rsrc[0]] + immsrc;
            reg[rdest].trunc(wordSz);
            break;
          case 2:
            // SLTI
            D(3, "SLTI: r" << rdest << " <- r" << rsrc[0] << ", imm=" << immsrc);
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
            D(3, "SLTIU: r" << rdest << " <- r" << rsrc[0] << ", imm=" << immsrc);
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
            D(3, "XORI: r" << rdest << " <- r" << rsrc[0] << ", imm=0x" << hex << immsrc);
            reg[rdest] = reg[rsrc[0]] ^ immsrc;
            break;
          case 6:
            // ORI
            D(3, "ORI: r" << rdest << " <- r" << rsrc[0] << ", imm=0x" << hex << immsrc);
            reg[rdest] = reg[rsrc[0]] | immsrc;
            break;
          case 7:
            // ANDI
            D(3, "ANDI: r" << rdest << " <- r" << rsrc[0] << ", imm=0x" << hex << immsrc);
            reg[rdest] = reg[rsrc[0]] & immsrc;
            break;
          case 1:
            // SLLI
            D(3, "SLLI: r" << rdest << " <- r" << rsrc[0] << ", imm=0x" << hex << immsrc);
            reg[rdest] = reg[rsrc[0]] << immsrc;
            reg[rdest].trunc(wordSz);
            break;
          case 5:
            if ((func7 == 0))
            {
              // SRLI
              D(3, "SRLI: r" << rdest << " <- r" << rsrc[0] << ", imm=" << immsrc);
              bool isNeg  = ((0x80000000 & reg[rsrc[0]])) > 0;
              Word result = Word_u(reg[rsrc[0]]) >> Word_u(immsrc);
              reg[rdest] = result;      
              reg[rdest].trunc(wordSz);
            }
            else
            {
              // SRAI
              D(3, "SRAI: r" << rdest << " <- r" << rsrc[0] << ", imm=" << immsrc);
              op1 = reg[rsrc[0]];
              op2 = immsrc;
              reg[rdest] = op1 >> op2;
              reg[rdest].trunc(wordSz);
            }
            break;
          default:
            cout << "ERROR: UNSUPPORTED L INST\n";
            std::abort();
        }
        break;
      case S_INST:
        ++c.stores;
        memAddr = reg[rsrc[0]] + immsrc;                
        trace_inst->is_sw = true;
        trace_inst->mem_addresses[t] = memAddr;
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
            // SB
            D(3, "SB: r" << rsrc[1] << " <- r" << rsrc[0] << ", imm=" << (int)immsrc);
            c.core->mem.write(memAddr, reg[rsrc[1]] & 0x000000FF, c.supervisorMode, 1);
            break;
          case 1:
            // SH
            D(3, "SH: r" << rsrc[1] << " <- r" << rsrc[0] << ", imm=" << (int)immsrc);
            c.core->mem.write(memAddr, reg[rsrc[1]], c.supervisorMode, 2);
            break;
          case 2:
            // SD
            D(3, "SD: r" << rsrc[1] << " <- r" << rsrc[0] << ", imm=" << (int)immsrc);
            c.core->mem.write(memAddr, reg[rsrc[1]], c.supervisorMode, 4);
            break;
          default:
            cout << "ERROR: UNSUPPORTED S INST\n";
            std::abort();
        }
        D(3, "STORE MEM ADDRESS: " << std::hex << memAddr);
        c.memAccesses.push_back(Warp::MemAccess(true, memAddr));
#ifdef EMU_INSTRUMENTATION
       Harp::OSDomain::osDomain->
       do_mem(0, memAddr, c.core->mem.virtToPhys(memAddr), 8, true);
#endif
        break;
      case B_INST:
        trace_inst->stall_warp = true;        
        switch (func3)
        {
          case 0:
            // BEQ
            D(3,"BEQ: r" << rsrc[0] << ", r" << rsrc[1] << ", imm=" << (int)immsrc);
            if (int(reg[rsrc[0]]) == int(reg[rsrc[1]]))
            {
              if (!pcSet) nextPc = (c.pc - 4) + immsrc;
              pcSet = true;
            }
            break;
          case 1:
            // BNE
            D(3,"BNE: r" << rsrc[0] << ", r" << rsrc[1] << ", imm=" << (int)immsrc);
            if (int(reg[rsrc[0]]) != int(reg[rsrc[1]]))
            {
              if (!pcSet) nextPc = (c.pc - 4) + immsrc;
              pcSet = true;
            }
            break;
          case 4:
            // BLT
            D(3,"BLT: r" << rsrc[0] << ", r" << rsrc[1] << ", imm=" << (int)immsrc);
            if (int(reg[rsrc[0]]) < int(reg[rsrc[1]]))
            {
              if (!pcSet) nextPc = (c.pc - 4) + immsrc;
              pcSet = true;
            }
            break;
          case 5:
            // BGE
            D(3,"BGE: r" << rsrc[0] << ", r" << rsrc[1] << ", imm=" << (int)immsrc);
            if (int(reg[rsrc[0]]) >= int(reg[rsrc[1]]))
            {
              if (!pcSet) nextPc = (c.pc - 4) + immsrc;
              pcSet = true;
            }
            break;
          case 6:
            // BLTU
            D(3,"BLTU: r" << rsrc[0] << ", r" << rsrc[1] << ", imm=" << (int)immsrc);
            if (Word_u(reg[rsrc[0]]) < Word_u(reg[rsrc[1]]))
            {
              if (!pcSet) nextPc = (c.pc - 4) + immsrc;
              pcSet = true;
            }
            break;
          case 7:
            // BGEU
            D(3,"BGEU: r" << rsrc[0] << ", r" << rsrc[1] << ", imm=" << (int)immsrc);
            if (Word_u(reg[rsrc[0]]) >= Word_u(reg[rsrc[1]]))
            {
              if (!pcSet) nextPc = (c.pc - 4) + immsrc;
              pcSet = true;
            }
            break;
        }
        break;
      case LUI_INST:
        D(3, "LUI: r" << rdest << " <- imm=0x" << hex << immsrc);
        reg[rdest] = (immsrc << 12) & 0xfffff000;
        break;
      case AUIPC_INST:
        D(3, "AUIPC: r" << rdest << " <- imm=0x" << hex << immsrc);
        reg[rdest] = ((immsrc << 12) & 0xfffff000) + (c.pc - 4);
        break;
      case JAL_INST:
        D(3, "JAL: r" << rdest << " <- imm=" << (int)immsrc);
        trace_inst->stall_warp = true;
        if (!pcSet) nextPc = (c.pc - 4) + immsrc;
        if (!pcSet) {/*std::cout << "JAL... SETTING PC: " << nextPc << "\n"; */}
        if (rdest != 0) {
          reg[rdest] = c.pc;
        }
        pcSet = true;
        break;
      case JALR_INST:
        D(3, "JALR: r" << rdest << " <- r" << rsrc[0] << ", imm=" << (int)immsrc);
        trace_inst->stall_warp = true;
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

        if (!c.core->a.is_cpu_mode()) {
          //
          // GPGPU CSR extension
          //
          if (immsrc == 0x20) // ThreadID
          {
            reg[rdest] = t;
            D(2, "CSR Reading tid " << hex << immsrc << dec << " and returning " << reg[rdest]);
          } 
          else if (immsrc == 0x21) // WarpID
          {
            reg[rdest] = c.id;
            D(2, "CSR Reading wid " << hex << immsrc << dec << " and returning " << reg[rdest]);
          } 
          else if (immsrc == 0x25)
          {
            reg[rdest] = c.core->num_instructions;
          }  
          else if (immsrc == 0x26)
          {
            reg[rdest] = c.core->num_cycles;
          }
        } else {
          switch (func3)
          {
            case 1:
              // printf("Case 1\n");
              if (rdest != 0)
              {
                reg[rdest] = c.csr[immsrc & 0x00000FFF];
              }
              c.csr[immsrc & 0x00000FFF] = temp;
              
              break;
            case 2:
              // printf("Case 2\n");
              if (rdest != 0)
              {
                // printf("Reading from CSR: %d = %d\n", (immsrc & 0x00000FFF),  c.csr[immsrc & 0x00000FFF]);
                reg[rdest] = c.csr[immsrc & 0x00000FFF];
              }
              // printf("Writing to CSR --> %d = %d\n", immsrc,  (temp |  c.csr[immsrc & 0x00000FFF]));
              c.csr[immsrc & 0x00000FFF] = temp |  c.csr[immsrc & 0x00000FFF];
              
              break;
            case 3:
              // printf("Case 3\n");
              if (rdest != 0)
              {              
                reg[rdest]                 = c.csr[immsrc & 0x00000FFF];
              }
                c.csr[immsrc & 0x00000FFF] = temp &  (~c.csr[immsrc & 0x00000FFF]);
              
              break;
            case 5:
              // printf("Case 5\n");
              if (rdest != 0)
              {
                reg[rdest] = c.csr[immsrc & 0x00000FFF];
              }
                c.csr[immsrc & 0x00000FFF] = rsrc[0];
              
              break;
            case 6:
              // printf("Case 6\n");
              if (rdest != 0)
              {              
                reg[rdest]                 = c.csr[immsrc & 0x00000FFF];
              }
                c.csr[immsrc & 0x00000FFF] = rsrc[0] |  c.csr[immsrc & 0x00000FFF];
              
              break;
            case 7:
              // printf("Case 7\n");
              if (rdest != 0)
              {              
                reg[rdest] = c.csr[immsrc & 0x00000FFF];
              }
                c.csr[immsrc & 0x00000FFF] = rsrc[0] &  (~c.csr[immsrc & 0x00000FFF]);
              
              break;
            case 0:
            if (immsrc < 2)
            {
              //std::cout << "INTERRUPT ECALL/EBREAK\n";
              nextActiveThreads = 0;
              c.spawned = false;
              // c.interrupt(0);
            }
              break;
            default:
              break;
          }
        }
        break;
      case TRAP:
        D(3, "TRAP");
        nextActiveThreads = 0;
        c.interrupt(0);
        break;
      case FENCE:
        D(3, "FENCE");
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
            D(3, "WSPAWN");
            trace_inst->wspawn = true;
            if (sjOnce)
            {
              sjOnce = false;
              // //std::cout << "SIZE: " << c.core->w.size() << "\n";
              num_to_wspawn = std::min<unsigned>(reg[rsrc[0]], c.core->a.getNWarps());

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
            D(3, "SPLIT");
            trace_inst->stall_warp = true;
            if (sjOnce)
            {
              sjOnce = false;
              if (checkUnanimous(pred, c.reg, c.tmask)) {
                D(3, "Unanimous pred: " << pred << "  val: " << reg[pred] << "\n");
                DomStackEntry e(c.tmask);
                e.uni = true;
                c.domStack.push(e);
                break;
              }
              D(3, "Split: Original TM: ");
              for (auto y : c.tmask) D(3, y << " ");

              DomStackEntry e(pred, c.reg, c.tmask, c.pc);
              c.domStack.push(c.tmask);
              c.domStack.push(e);
              for (unsigned i = 0; i < e.tmask.size(); ++i)
              {
                c.tmask[i] = !e.tmask[i] && c.tmask[i];
              }


              D(3, "Split: New TM");
              for (auto y : c.tmask) D(3, y << " ");
              D(3, "Split: Pushed TM PC: " << hex << e.pc << dec << "\n");
              for (auto y : e.tmask) D(3, y << " ");
            }
            break;
          }
          case 3:
            // JOIN
            D(3, "JOIN");
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
                  D(3, "join: NOT FALLTHROUGH PC: " << hex << nextPc << dec);
                }
                  pcSet = true;
              }

              D(3, "Join: Old TM: ");
              for (auto y : c.tmask) D(3, y << " ");
              cout << "\n";
              c.tmask = c.domStack.top().tmask;

              D(3, "Join: New TM: ");
              for (auto y : c.tmask) D(3, y << " ");

              c.domStack.pop();
            }
            break;
          case 4:
            trace_inst->stall_warp = true;
            // is_barrier
            break;
          case 0:
            // TMC
            D(3, "TMC");
            trace_inst->stall_warp = true;
            nextActiveThreads = std::min<unsigned>(reg[rsrc[0]], c.core->a.getNThds());
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
      case VSET_ARITH:
        D(3,"VSET_ARITH");
        is_vec = true;
        switch(func3) {
          case 0: // vector-vector
          trace_inst->vs1 = rsrc[0];
          trace_inst->vs2 = rsrc[1];
          trace_inst->vd  = rdest;
          switch(func6)
          {
            case 0:
            {
              is_vec = true;
              D(3, "Addition " << rsrc[0] << " " << rsrc[1] << " Dest:" << rdest);
              vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
              vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
              vector<Reg<char *>> & vd  = c.vreg[rdest];
              vector<Reg<char *>> & mask  = c.vreg[0];

              if (c.vtype.vsew == 8)
              {
                for (uint8_t i = 0; i < c.vl; i++)
                {
                  uint8_t *mask_ptr = (uint8_t*) mask[i].val;
                  uint8_t value = (*mask_ptr & 0x1);
                  if(vmask || (!vmask && value)){
                    uint8_t * first_ptr  = (uint8_t *) vr1[i].val; 
                    uint8_t * second_ptr = (uint8_t *) vr2[i].val;
                    uint8_t result = *first_ptr + *second_ptr;
                    D(3, "Adding " << *first_ptr << " + " << *second_ptr << " = " << result);

                    uint8_t * result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = result;
                  }

                }

              } else if (c.vtype.vsew == 16)
              {

                for (uint16_t i = 0; i < c.vl; i++)
                {
                  uint16_t *mask_ptr = (uint16_t*) mask[i].val;
                  uint16_t value = (*mask_ptr & 0x1);
                  if(vmask || (!vmask && value)){
                    uint16_t * first_ptr  = (uint16_t *) vr1[i].val; 
                    uint16_t * second_ptr = (uint16_t *) vr2[i].val;
                    uint16_t result = *first_ptr + *second_ptr;
                    D(3, "Adding " << *first_ptr << " + " << *second_ptr << " = " << result);

                    uint16_t * result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = result;
                  }

                }
              } else if (c.vtype.vsew == 32)
              {
                D(3, "Doing 32 bit vector addition");
                for (Word i = 0; i < c.vl; i++)
                {
                  int *mask_ptr = (int*) mask[i].val;
                  int value = (*mask_ptr & 0x1);
                  if(vmask || (!vmask && value)){
                    int * first_ptr  = (int *) vr1[i].val; 
                    int * second_ptr = (int *) vr2[i].val;
                    int result = *first_ptr + *second_ptr;
                    D(3, "Adding " << *first_ptr << " + " << *second_ptr << " = " << result);

                    int * result_ptr = (int *) vd[i].val;
                    *result_ptr = result;
                  }

                }
              }

              D(3, "Vector Register state after addition:" << flush);
              for(int i=0; i < c.vreg.size(); i++)
              {
                for(int j=0; j< c.vreg[0].size(); j++)
                {
                  if (c.vtype.vsew == 8)
                  {
                    uint8_t * ptr_val = (uint8_t *) c.vreg[i][j].val;
                    D(3, "reg[" << i << "][" << j << "] = " << *ptr_val);     
                  } else if (c.vtype.vsew == 16)
                  {
                    uint16_t * ptr_val = (uint16_t *) c.vreg[i][j].val;
                    D(3, "reg[" << i << "][" << j << "] = " << *ptr_val);     
                  } else if (c.vtype.vsew == 32)
                  {
                    uint32_t * ptr_val = (uint32_t *) c.vreg[i][j].val;
                    D(3, "reg[" << i << "][" << j << "] = " << *ptr_val);     
                  }
                }
              }

              D(3, "After vector register state after addition" << flush);
            }
            break;
            case 24: //vmseq
            {
              vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
              vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
              vector<Reg<char *>> & vd  = c.vreg[rdest];
              if(c.vtype.vsew == 8){
                for(uint8_t i = 0; i < c.vl; i++){
                  uint8_t *first_ptr = (uint8_t *)vr1[i].val;
                  uint8_t *second_ptr = (uint8_t *)vr2[i].val;
                  uint8_t result = (*first_ptr == *second_ptr) ? 1 : 0;
                  D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  uint8_t * result_ptr = (uint8_t *) vd[i].val;
                  *result_ptr = result;
                }

              } else if(c.vtype.vsew == 16) {
                for(uint16_t i = 0; i < c.vl; i++){
                  uint16_t *first_ptr = (uint16_t *)vr1[i].val;
                  uint16_t *second_ptr = (uint16_t *)vr2[i].val;
                  uint16_t result = (*first_ptr == *second_ptr) ? 1 : 0;
                  D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  uint16_t * result_ptr = (uint16_t *) vd[i].val;
                  *result_ptr = result;
                }

              } else if(c.vtype.vsew == 32) {
                for(uint32_t i = 0; i < c.vl; i++){
                  uint32_t *first_ptr = (uint32_t *)vr1[i].val;
                  uint32_t *second_ptr = (uint32_t *)vr2[i].val;
                  uint32_t result = (*first_ptr == *second_ptr) ? 1 : 0;
                  D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  uint32_t * result_ptr = (uint32_t *) vd[i].val;
                  *result_ptr = result;
                }
              }

            }
            break;
            case 25: //vmsne
            {
              vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
              vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
              vector<Reg<char *>> & vd  = c.vreg[rdest];
              if(c.vtype.vsew == 8){
                for(uint8_t i = 0; i < c.vl; i++){
                  uint8_t *first_ptr = (uint8_t *)vr1[i].val;
                  uint8_t *second_ptr = (uint8_t *)vr2[i].val;
                  uint8_t result = (*first_ptr != *second_ptr) ? 1 : 0;
                  D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  uint8_t * result_ptr = (uint8_t *) vd[i].val;
                  *result_ptr = result;
                }

              } else if(c.vtype.vsew == 16) {
                for(uint16_t i = 0; i < c.vl; i++){
                  uint16_t *first_ptr = (uint16_t *)vr1[i].val;
                  uint16_t *second_ptr = (uint16_t *)vr2[i].val;
                  uint16_t result = (*first_ptr != *second_ptr) ? 1 : 0;
                  D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  uint16_t * result_ptr = (uint16_t *) vd[i].val;
                  *result_ptr = result;
                }

              } else if(c.vtype.vsew == 32) {
                for(uint32_t i = 0; i < c.vl; i++){
                  uint32_t *first_ptr = (uint32_t *)vr1[i].val;
                  uint32_t *second_ptr = (uint32_t *)vr2[i].val;
                  uint32_t result = (*first_ptr != *second_ptr) ? 1 : 0;
                  D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  uint32_t * result_ptr = (uint32_t *) vd[i].val;
                  *result_ptr = result;
                }
              }

            }
            break;
            case 26: //vmsltu
            {
              vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
              vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
              vector<Reg<char *>> & vd  = c.vreg[rdest];
              if(c.vtype.vsew == 8){
                for(uint8_t i = 0; i < c.vl; i++){
                  uint8_t *first_ptr = (uint8_t *)vr1[i].val;
                  uint8_t *second_ptr = (uint8_t *)vr2[i].val;
                  uint8_t result = (*first_ptr < *second_ptr) ? 1 : 0;
                  D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  uint8_t * result_ptr = (uint8_t *) vd[i].val;
                  *result_ptr = result;
                }

              } else if(c.vtype.vsew == 16) {
                for(uint16_t i = 0; i < c.vl; i++){
                  uint16_t *first_ptr = (uint16_t *)vr1[i].val;
                  uint16_t *second_ptr = (uint16_t *)vr2[i].val;
                  uint16_t result = (*first_ptr < *second_ptr) ? 1 : 0;
                  D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  uint16_t * result_ptr = (uint16_t *) vd[i].val;
                  *result_ptr = result;
                }

              } else if(c.vtype.vsew == 32) {
                for(uint32_t i = 0; i < c.vl; i++){
                  uint32_t *first_ptr = (uint32_t *)vr1[i].val;
                  uint32_t *second_ptr = (uint32_t *)vr2[i].val;
                  uint32_t result = (*first_ptr < *second_ptr) ? 1 : 0;
                  D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  uint32_t * result_ptr = (uint32_t *) vd[i].val;
                  *result_ptr = result;
                }
              }

            }
            break;
            case 27: //vmslt
            {
              vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
              vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
              vector<Reg<char *>> & vd  = c.vreg[rdest];
              if(c.vtype.vsew == 8){
                for(int8_t i = 0; i < c.vl; i++){
                  int8_t *first_ptr = (int8_t *)vr1[i].val;
                  int8_t *second_ptr = (int8_t *)vr2[i].val;
                  int8_t result = (*first_ptr < *second_ptr) ? 1 : 0;
                  D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  int8_t * result_ptr = (int8_t *) vd[i].val;
                  *result_ptr = result;
                }

              } else if(c.vtype.vsew == 16) {
                for(int16_t i = 0; i < c.vl; i++){
                  int16_t *first_ptr = (int16_t *)vr1[i].val;
                  int16_t *second_ptr = (int16_t *)vr2[i].val;
                  int16_t result = (*first_ptr < *second_ptr) ? 1 : 0;
                  D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  int16_t * result_ptr = (int16_t *) vd[i].val;
                  *result_ptr = result;
                }

              } else if(c.vtype.vsew == 32) {
                for(int32_t i = 0; i < c.vl; i++){
                  int32_t *first_ptr = (int32_t *)vr1[i].val;
                  int32_t *second_ptr = (int32_t *)vr2[i].val;
                  int32_t result = (*first_ptr < *second_ptr) ? 1 : 0;
                  D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  int32_t * result_ptr = (int32_t *) vd[i].val;
                  *result_ptr = result;
                }
              }
            }
            break;
            case 28: //vmsleu
            {
              vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
              vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
              vector<Reg<char *>> & vd  = c.vreg[rdest];
              if(c.vtype.vsew == 8){
                for(uint8_t i = 0; i < c.vl; i++){
                  uint8_t *first_ptr = (uint8_t *)vr1[i].val;
                  uint8_t *second_ptr = (uint8_t *)vr2[i].val;
                  uint8_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
                  D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  uint8_t * result_ptr = (uint8_t *) vd[i].val;
                  *result_ptr = result;
                }

              } else if(c.vtype.vsew == 16) {
                for(uint16_t i = 0; i < c.vl; i++){
                  uint16_t *first_ptr = (uint16_t *)vr1[i].val;
                  uint16_t *second_ptr = (uint16_t *)vr2[i].val;
                  uint16_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
                  D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  uint16_t * result_ptr = (uint16_t *) vd[i].val;
                  *result_ptr = result;
                }

              } else if(c.vtype.vsew == 32) {
                for(uint32_t i = 0; i < c.vl; i++){
                  uint32_t *first_ptr = (uint32_t *)vr1[i].val;
                  uint32_t *second_ptr = (uint32_t *)vr2[i].val;
                  uint32_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
                  D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  uint32_t * result_ptr = (uint32_t *) vd[i].val;
                  *result_ptr = result;
                }
              }
            }
            break;
            case 29: //vmsle
            {
              vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
              vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
              vector<Reg<char *>> & vd  = c.vreg[rdest];
              if(c.vtype.vsew == 8){
                for(int8_t i = 0; i < c.vl; i++){
                  int8_t *first_ptr = (int8_t *)vr1[i].val;
                  int8_t *second_ptr = (int8_t *)vr2[i].val;
                  int8_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
                  D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  int8_t * result_ptr = (int8_t *) vd[i].val;
                  *result_ptr = result;
                }

              } else if(c.vtype.vsew == 16) {
                for(int16_t i = 0; i < c.vl; i++){
                  int16_t *first_ptr = (int16_t *)vr1[i].val;
                  int16_t *second_ptr = (int16_t *)vr2[i].val;
                  int16_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
                  D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  int16_t * result_ptr = (int16_t *) vd[i].val;
                  *result_ptr = result;
                }

              } else if(c.vtype.vsew == 32) {
                for(int32_t i = 0; i < c.vl; i++){
                  int32_t *first_ptr = (int32_t *)vr1[i].val;
                  int32_t *second_ptr = (int32_t *)vr2[i].val;
                  int32_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
                  D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  int32_t * result_ptr = (int32_t *) vd[i].val;
                  *result_ptr = result;
                }
              }
            }
            break;
            case 30: //vmsgtu
            {
              vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
              vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
              vector<Reg<char *>> & vd  = c.vreg[rdest];
              if(c.vtype.vsew == 8){
                for(uint8_t i = 0; i < c.vl; i++){
                  uint8_t *first_ptr = (uint8_t *)vr1[i].val;
                  uint8_t *second_ptr = (uint8_t *)vr2[i].val;
                  uint8_t result = (*first_ptr > *second_ptr) ? 1 : 0;
                  D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  uint8_t * result_ptr = (uint8_t *) vd[i].val;
                  *result_ptr = result;
                }

              } else if(c.vtype.vsew == 16) {
                for(uint16_t i = 0; i < c.vl; i++){
                  uint16_t *first_ptr = (uint16_t *)vr1[i].val;
                  uint16_t *second_ptr = (uint16_t *)vr2[i].val;
                  uint16_t result = (*first_ptr > *second_ptr) ? 1 : 0;
                  D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  uint16_t * result_ptr = (uint16_t *) vd[i].val;
                  *result_ptr = result;
                }

              } else if(c.vtype.vsew == 32) {
                for(uint32_t i = 0; i < c.vl; i++){
                  uint32_t *first_ptr = (uint32_t *)vr1[i].val;
                  uint32_t *second_ptr = (uint32_t *)vr2[i].val;
                  uint32_t result = (*first_ptr > *second_ptr) ? 1 : 0;
                  D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  uint32_t * result_ptr = (uint32_t *) vd[i].val;
                  *result_ptr = result;
                }
              }
            }
            break;
            case 31: //vmsgt
            {
              vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
              vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
              vector<Reg<char *>> & vd  = c.vreg[rdest];
              if(c.vtype.vsew == 8){
                for(int8_t i = 0; i < c.vl; i++){
                  int8_t *first_ptr = (int8_t *)vr1[i].val;
                  int8_t *second_ptr = (int8_t *)vr2[i].val;
                  int8_t result = (*first_ptr > *second_ptr) ? 1 : 0;
                  D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  int8_t * result_ptr = (int8_t *) vd[i].val;
                  *result_ptr = result;
                }

              } else if(c.vtype.vsew == 16) {
                for(int16_t i = 0; i < c.vl; i++){
                  int16_t *first_ptr = (int16_t *)vr1[i].val;
                  int16_t *second_ptr = (int16_t *)vr2[i].val;
                  int16_t result = (*first_ptr > *second_ptr) ? 1 : 0;
                  D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  int16_t * result_ptr = (int16_t *) vd[i].val;
                  *result_ptr = result;
                }

              } else if(c.vtype.vsew == 32) {
                for(int32_t i = 0; i < c.vl; i++){
                  int32_t *first_ptr = (int32_t *)vr1[i].val;
                  int32_t *second_ptr = (int32_t *)vr2[i].val;
                  int32_t result = (*first_ptr > *second_ptr) ? 1 : 0;
                  D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                  int32_t * result_ptr = (int32_t *) vd[i].val;
                  *result_ptr = result;
                }
              }
            }
            break;
          }
          break;
          case 2:
          {
            trace_inst->vs1 = rsrc[0];
            trace_inst->vs2 = rsrc[1];
            trace_inst->vd  = rdest;
            Word VLMAX = (c.vtype.vlmul * c.VLEN)/c.vtype.vsew;

            switch(func6){
              case 24: //vmandnot
              {
                D(3, "vmandnot");
                vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
                vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
                vector<Reg<char *>> & vd  = c.vreg[rdest];
                if(c.vtype.vsew == 8){
                  for(uint8_t i = 0; i < c.vl; i++){
                    uint8_t *first_ptr = (uint8_t *)vr1[i].val;
                    uint8_t *second_ptr = (uint8_t *)vr2[i].val;
                    uint8_t first_value = (*first_ptr & 0x1);
                    uint8_t second_value = (*second_ptr & 0x1);
                    uint8_t result = (first_value & !second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    uint8_t * result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint8_t i = c.vl; i < VLMAX; i++){
                    uint8_t *result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = 0;
                  }


                } else if(c.vtype.vsew == 16) {
                  for(uint16_t i = 0; i < c.vl; i++){
                    uint16_t *first_ptr = (uint16_t *)vr1[i].val;
                    uint16_t *second_ptr = (uint16_t *)vr2[i].val;
                    uint16_t first_value = (*first_ptr & 0x1);
                    uint16_t second_value = (*second_ptr & 0x1);
                    uint16_t result = (first_value & !second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    uint16_t * result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint16_t i = c.vl; i < VLMAX; i++){
                    uint16_t *result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = 0;
                  }


                } else if(c.vtype.vsew == 32) {
                  for(uint32_t i = 0; i < c.vl; i++){
                    uint32_t *first_ptr = (uint32_t *)vr1[i].val;
                    uint32_t *second_ptr = (uint32_t *)vr2[i].val;
                    uint32_t first_value = (*first_ptr & 0x1);
                    uint32_t second_value = (*second_ptr & 0x1);
                    uint32_t result = (first_value & !second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    uint32_t * result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(Word i = c.vl; i < VLMAX; i++){
                    uint32_t *result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = 0;
                  }

                }
              }              
              break;
              case 25: //vmand
              {
                D(3, "vmand");
                vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
                vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
                vector<Reg<char *>> & vd  = c.vreg[rdest];
                if(c.vtype.vsew == 8){
                  for(uint8_t i = 0; i < c.vl; i++){
                    uint8_t *first_ptr = (uint8_t *)vr1[i].val;
                    uint8_t *second_ptr = (uint8_t *)vr2[i].val;
                    uint8_t first_value = (*first_ptr & 0x1);
                    uint8_t second_value = (*second_ptr & 0x1);
                    uint8_t result = (first_value & second_value);
                    D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    uint8_t * result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint8_t i = c.vl; i < VLMAX; i++){
                    uint8_t *result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = 0;
                  }

                } else if(c.vtype.vsew == 16) {
                  for(uint16_t i = 0; i < c.vl; i++){
                    uint16_t *first_ptr = (uint16_t *)vr1[i].val;
                    uint16_t *second_ptr = (uint16_t *)vr2[i].val;
                    uint16_t first_value = (*first_ptr & 0x1);
                    uint16_t second_value = (*second_ptr & 0x1);
                    uint16_t result = (first_value & second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    uint16_t * result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = result;
                  }

                  for(uint16_t i = c.vl; i < VLMAX; i++){
                    uint16_t *result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = 0;
                  }

                } else if(c.vtype.vsew == 32) {
                  for(uint32_t i = 0; i < c.vl; i++){
                    uint32_t *first_ptr = (uint32_t *)vr1[i].val;
                    uint32_t *second_ptr = (uint32_t *)vr2[i].val;
                    uint32_t first_value = (*first_ptr & 0x1);
                    uint32_t second_value = (*second_ptr & 0x1);
                    uint32_t result = (first_value & second_value);
                    D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    uint32_t * result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = result;
                  }

                  for(Word i = c.vl; i < VLMAX; i++){
                    uint32_t *result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = 0;
                  }
                }
              }              
              break;
              case 26: //vmor
              {
                D(3, "vmor");
                vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
                vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
                vector<Reg<char *>> & vd  = c.vreg[rdest];
                if(c.vtype.vsew == 8){
                  for(uint8_t i = 0; i < c.vl; i++){
                    uint8_t *first_ptr = (uint8_t *)vr1[i].val;
                    uint8_t *second_ptr = (uint8_t *)vr2[i].val;
                    uint8_t first_value = (*first_ptr & 0x1);
                    uint8_t second_value = (*second_ptr & 0x1);
                    uint8_t result = (first_value | second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    uint8_t * result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint8_t i = c.vl; i < VLMAX; i++){
                    uint8_t *result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = 0;
                  }

                } else if(c.vtype.vsew == 16) {
                  uint16_t *result_ptr;
                  for(uint16_t i = 0; i < c.vl; i++){
                    uint16_t *first_ptr = (uint16_t *)vr1[i].val;
                    uint16_t *second_ptr = (uint16_t *)vr2[i].val;
                    uint16_t first_value = (*first_ptr & 0x1);
                    uint16_t second_value = (*second_ptr & 0x1);
                    uint16_t result = (first_value | second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint16_t i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = 0;
                  }
                } else if(c.vtype.vsew == 32) {
                  uint32_t *result_ptr;
                  for(uint32_t i = 0; i < c.vl; i++){
                    uint32_t *first_ptr = (uint32_t *)vr1[i].val;
                    uint32_t *second_ptr = (uint32_t *)vr2[i].val;
                    uint32_t first_value = (*first_ptr & 0x1);
                    uint32_t second_value = (*second_ptr & 0x1);
                    uint32_t result = (first_value | second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  D(3, "VLMAX: " << VLMAX);
                  for(Word i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = 0;
                  }
                }
              }              
              break;
              case 27: //vmxor
              {
                D(3, "vmxor");
                vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
                vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
                vector<Reg<char *>> & vd  = c.vreg[rdest];
                if(c.vtype.vsew == 8){
                  uint8_t *result_ptr;
                  for(uint8_t i = 0; i < c.vl; i++){
                    uint8_t *first_ptr = (uint8_t *)vr1[i].val;
                    uint8_t *second_ptr = (uint8_t *)vr2[i].val;
                    uint8_t first_value = (*first_ptr & 0x1);
                    uint8_t second_value = (*second_ptr & 0x1);
                    uint8_t result = (first_value ^ second_value);
                    D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);
                    result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint8_t i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = 0;
                  }
                } else if(c.vtype.vsew == 16) {
                  uint16_t *result_ptr;
                  for(uint16_t i = 0; i < c.vl; i++){
                    uint16_t *first_ptr = (uint16_t *)vr1[i].val;
                    uint16_t *second_ptr = (uint16_t *)vr2[i].val;
                    uint16_t first_value = (*first_ptr & 0x1);
                    uint16_t second_value = (*second_ptr & 0x1);
                    uint16_t result = (first_value ^ second_value);
                    D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint16_t i = c.vl; i < VLMAX; i++){
                    uint16_t *result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = 0;
                  }

                } else if(c.vtype.vsew == 32) {
                  uint32_t *result_ptr;
                  
                  for(uint32_t i = 0; i < c.vl; i++){
                    uint32_t *first_ptr = (uint32_t *)vr1[i].val;
                    uint32_t *second_ptr = (uint32_t *)vr2[i].val;
                    uint32_t first_value = (*first_ptr & 0x1);
                    uint32_t second_value = (*second_ptr & 0x1);
                    uint32_t result = (first_value ^ second_value);
                    D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(Word i = c.vl; i < VLMAX; i++){
                    uint32_t *result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = 0;
                  }
                }
              }              
              break;
              case 28: //vmornot
              {
                D(3, "vmornot");
                vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
                vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
                vector<Reg<char *>> & vd  = c.vreg[rdest];
                if(c.vtype.vsew == 8){
                  for(uint8_t i = 0; i < c.vl; i++){
                    uint8_t *first_ptr = (uint8_t *)vr1[i].val;
                    uint8_t *second_ptr = (uint8_t *)vr2[i].val;
                    uint8_t first_value = (*first_ptr & 0x1);
                    uint8_t second_value = (*second_ptr & 0x1);
                    uint8_t result = (first_value | !second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    uint8_t * result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint8_t i = c.vl; i < VLMAX; i++){
                    uint8_t *result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = 0;
                  }
                } else if(c.vtype.vsew == 16) {
                  for(uint16_t i = 0; i < c.vl; i++){
                    uint16_t *first_ptr = (uint16_t *)vr1[i].val;
                    uint16_t *second_ptr = (uint16_t *)vr2[i].val;
                    uint16_t first_value = (*first_ptr & 0x1);
                    uint16_t second_value = (*second_ptr & 0x1);
                    uint16_t result = (first_value | !second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    uint16_t * result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint16_t i = c.vl; i < VLMAX; i++){
                    uint16_t *result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = 0;
                  }

                } else if(c.vtype.vsew == 32) {
                  for(uint32_t i = 0; i < c.vl; i++){
                    uint32_t *first_ptr = (uint32_t *)vr1[i].val;
                    uint32_t *second_ptr = (uint32_t *)vr2[i].val;
                    uint32_t first_value = (*first_ptr & 0x1);
                    uint32_t second_value = (*second_ptr & 0x1);
                    uint32_t result = (first_value | !second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    uint32_t * result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(Word i = c.vl; i < VLMAX; i++){
                    uint32_t *result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = 0;
                  }
                }
              }              
              break;
              case 29: //vmnand
              {
                D(3, "vmnand");
                vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
                vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
                vector<Reg<char *>> & vd  = c.vreg[rdest];
                if(c.vtype.vsew == 8){
                  for(uint8_t i = 0; i < c.vl; i++){
                    uint8_t *first_ptr = (uint8_t *)vr1[i].val;
                    uint8_t *second_ptr = (uint8_t *)vr2[i].val;
                    uint8_t first_value = (*first_ptr & 0x1);
                    uint8_t second_value = (*second_ptr & 0x1);
                    uint8_t result = !(first_value & second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    uint8_t * result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint8_t i = c.vl; i < VLMAX; i++){
                    uint8_t *result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = 0;
                  }

                } else if(c.vtype.vsew == 16) {
                  for(uint16_t i = 0; i < c.vl; i++){
                    uint16_t *first_ptr = (uint16_t *)vr1[i].val;
                    uint16_t *second_ptr = (uint16_t *)vr2[i].val;
                    uint16_t first_value = (*first_ptr & 0x1);
                    uint16_t second_value = (*second_ptr & 0x1);
                    uint16_t result = !(first_value & second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    uint16_t * result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = result;
                  }

                  for(uint16_t i = c.vl; i < VLMAX; i++){
                    uint16_t *result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = 0;
                  }

                } else if(c.vtype.vsew == 32) {
                  for(uint32_t i = 0; i < c.vl; i++){
                    uint32_t *first_ptr = (uint32_t *)vr1[i].val;
                    uint32_t *second_ptr = (uint32_t *)vr2[i].val;
                    uint32_t first_value = (*first_ptr & 0x1);
                    uint32_t second_value = (*second_ptr & 0x1);
                    uint32_t result = !(first_value & second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    uint32_t * result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = result;
                  }

                  for(Word i = c.vl; i < VLMAX; i++){
                    uint32_t *result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = 0;
                  }

                }
              }   
              break;
              case 30: //vmnor
              {
                D(3, "vmnor");
                vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
                vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
                vector<Reg<char *>> & vd  = c.vreg[rdest];
                if(c.vtype.vsew == 8){
                  uint8_t *result_ptr;

                  for(uint8_t i = 0; i < c.vl; i++){
                    uint8_t *first_ptr = (uint8_t *)vr1[i].val;
                    uint8_t *second_ptr = (uint8_t *)vr2[i].val;
                    uint8_t first_value = (*first_ptr & 0x1);
                    uint8_t second_value = (*second_ptr & 0x1);
                    uint8_t result = !(first_value | second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint8_t i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = 0;
                  }
                } else if(c.vtype.vsew == 16) {
                  for(uint16_t i = 0; i < c.vl; i++){
                    uint16_t *first_ptr = (uint16_t *)vr1[i].val;
                    uint16_t *second_ptr = (uint16_t *)vr2[i].val;
                    uint16_t first_value = (*first_ptr & 0x1);
                    uint16_t second_value = (*second_ptr & 0x1);
                    uint16_t result = !(first_value | second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    uint16_t * result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint16_t i = c.vl; i < VLMAX; i++){
                    uint16_t *result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = 0;
                  }

                } else if(c.vtype.vsew == 32) {

                  for(uint32_t i = 0; i < c.vl; i++){
                    uint32_t *first_ptr = (uint32_t *)vr1[i].val;
                    uint32_t *second_ptr = (uint32_t *)vr2[i].val;
                    uint32_t first_value = (*first_ptr & 0x1);
                    uint32_t second_value = (*second_ptr & 0x1);
                    uint32_t result = !(first_value | second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    uint32_t * result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(Word i = c.vl; i < VLMAX; i++){
                    uint32_t *result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = 0;
                  }

                }
              }              
              break; 
              case 31: //vmxnor
              {
                D(3, "vmxnor");
                uint8_t *result_ptr;

                vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
                vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
                vector<Reg<char *>> & vd  = c.vreg[rdest];
                if(c.vtype.vsew == 8){
                  for(uint8_t i = 0; i < c.vl; i++){
                    uint8_t *first_ptr = (uint8_t *)vr1[i].val;
                    uint8_t *second_ptr = (uint8_t *)vr2[i].val;
                    uint8_t first_value = (*first_ptr & 0x1);
                    uint8_t second_value = (*second_ptr & 0x1);
                    uint8_t result = !(first_value ^ second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint8_t i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = 0;
                  }
                } 
                else if(c.vtype.vsew == 16) {
                  uint16_t *result_ptr;
                  for(uint16_t i = 0; i < c.vl; i++){
                    uint16_t *first_ptr = (uint16_t *)vr1[i].val;
                    uint16_t *second_ptr = (uint16_t *)vr2[i].val;
                    uint16_t first_value = (*first_ptr & 0x1);
                    uint16_t second_value = (*second_ptr & 0x1);
                    uint16_t result = !(first_value ^ second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint16_t i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = 0;
                  }

                } else if(c.vtype.vsew == 32) {
                  uint32_t *result_ptr;

                  for(uint32_t i = 0; i < c.vl; i++){
                    uint32_t *first_ptr = (uint32_t *)vr1[i].val;
                    uint32_t *second_ptr = (uint32_t *)vr2[i].val;
                    uint32_t first_value = (*first_ptr & 0x1);
                    uint32_t second_value = (*second_ptr & 0x1);
                    uint32_t result = !(first_value ^ second_value);
                    D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(Word i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = 0;
                  }

                }
              }              
              break; 
              case 37: //vmul
              {
                D(3, "vmul");
                uint8_t *result_ptr;

                vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
                vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
                vector<Reg<char *>> & vd  = c.vreg[rdest];
                if(c.vtype.vsew == 8){
                  for(uint8_t i = 0; i < c.vl; i++){
                    uint8_t *first_ptr = (uint8_t *)vr1[i].val;
                    uint8_t *second_ptr = (uint8_t *)vr2[i].val;
                    uint8_t result =  (*first_ptr * *second_ptr);
                    D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint8_t i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = 0;
                  }
                } 
                else if(c.vtype.vsew == 16) {
                  uint16_t *result_ptr;
                  for(uint16_t i = 0; i < c.vl; i++){
                    uint16_t *first_ptr = (uint16_t *)vr1[i].val;
                    uint16_t *second_ptr = (uint16_t *)vr2[i].val;
                    uint16_t result = (*first_ptr * *second_ptr);
                    D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint16_t i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = 0;
                  }

                } else if(c.vtype.vsew == 32) {
                  uint32_t *result_ptr;

                  for(uint32_t i = 0; i < c.vl; i++){
                    uint32_t *first_ptr = (uint32_t *)vr1[i].val;
                    uint32_t *second_ptr = (uint32_t *)vr2[i].val;
                    uint32_t result = (*first_ptr * *second_ptr);
                    D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(Word i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = 0;
                  }
                }
              }
              break;
              case 45: //vmacc   
              {
                D(3, "vmacc");
                uint8_t *result_ptr;

                vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
                vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
                vector<Reg<char *>> & vd  = c.vreg[rdest];
                if(c.vtype.vsew == 8){
                  for(uint8_t i = 0; i < c.vl; i++){
                    uint8_t *first_ptr = (uint8_t *)vr1[i].val;
                    uint8_t *second_ptr = (uint8_t *)vr2[i].val;
                    uint8_t result =  (*first_ptr * *second_ptr);
                    D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr += result;
                  }
                  for(uint8_t i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = 0;
                  }
                } 
                else if(c.vtype.vsew == 16) {
                  uint16_t *result_ptr;
                  for(uint16_t i = 0; i < c.vl; i++){
                    uint16_t *first_ptr = (uint16_t *)vr1[i].val;
                    uint16_t *second_ptr = (uint16_t *)vr2[i].val;
                    uint16_t result = (*first_ptr * *second_ptr);
                    D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr += result;
                  }
                  for(uint16_t i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = 0;
                  }

                } else if(c.vtype.vsew == 32) {
                  uint32_t *result_ptr;

                  for(uint32_t i = 0; i < c.vl; i++){
                    uint32_t *first_ptr = (uint32_t *)vr1[i].val;
                    uint32_t *second_ptr = (uint32_t *)vr2[i].val;
                    uint32_t result = (*first_ptr * *second_ptr);
                    D(3,"Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr += result;
                  }
                  for(Word i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = 0;
                  }
                }
              }
              break;        
            }
          }
          break;
          case 6: 
          {
            switch(func6)
            {
              case 0:
              {
                D(3, "vmadd.vx");
                uint8_t *result_ptr;

                //vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
                vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
                vector<Reg<char *>> & vd  = c.vreg[rdest];
                if(c.vtype.vsew == 8){
                  for(uint8_t i = 0; i < c.vl; i++){
                    //uint8_t *first_ptr = (uint8_t *)vr1[i].val;
                    uint8_t *second_ptr = (uint8_t *)vr2[i].val;
                    uint8_t result =  (reg[rsrc[0]] + *second_ptr);
                    D(3,"Comparing " << reg[rsrc[0]] << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint8_t i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = 0;
                  }
                } 
                else if(c.vtype.vsew == 16) {
                  uint16_t *result_ptr;
                  for(uint16_t i = 0; i < c.vl; i++){
                    //uint16_t *first_ptr = (uint16_t *)vr1[i].val;
                    uint16_t *second_ptr = (uint16_t *)vr2[i].val;
                    uint16_t result = (reg[rsrc[0]] + *second_ptr);
                    D(3,"Comparing " << reg[rsrc[0]] << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint16_t i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = 0;
                  }

                } else if(c.vtype.vsew == 32) {
                  uint32_t *result_ptr;

                  for(uint32_t i = 0; i < c.vl; i++){
                    //uint32_t *first_ptr = (uint32_t *)vr1[i].val;
                    uint32_t *second_ptr = (uint32_t *)vr2[i].val;
                    uint32_t result = (reg[rsrc[0]] + *second_ptr);
                    D(3,"Comparing " << reg[rsrc[0]] << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(Word i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = 0;
                  }
                }                
              }
              break;
              case 37: //vmul.vx
              {
                D(3, "vmul.vx");
                uint8_t *result_ptr;

                //vector<Reg<char *>> & vr1 = c.vreg[rsrc[0]];
                vector<Reg<char *>> & vr2 = c.vreg[rsrc[1]];
                vector<Reg<char *>> & vd  = c.vreg[rdest];
                if(c.vtype.vsew == 8){
                  for(uint8_t i = 0; i < c.vl; i++){
                    //uint8_t *first_ptr = (uint8_t *)vr1[i].val;
                    uint8_t *second_ptr = (uint8_t *)vr2[i].val;
                    uint8_t result =  (reg[rsrc[0]] * *second_ptr);
                    D(3,"Comparing " << reg[rsrc[0]] << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint8_t i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint8_t *) vd[i].val;
                    *result_ptr = 0;
                  }
                } 
                else if(c.vtype.vsew == 16) {
                  uint16_t *result_ptr;
                  for(uint16_t i = 0; i < c.vl; i++){
                    //uint16_t *first_ptr = (uint16_t *)vr1[i].val;
                    uint16_t *second_ptr = (uint16_t *)vr2[i].val;
                    uint16_t result = (reg[rsrc[0]] * *second_ptr);
                    D(3,"Comparing " << reg[rsrc[0]] << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(uint16_t i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint16_t *) vd[i].val;
                    *result_ptr = 0;
                  }

                } else if(c.vtype.vsew == 32) {
                  uint32_t *result_ptr;

                  for(uint32_t i = 0; i < c.vl; i++){
                    //uint32_t *first_ptr = (uint32_t *)vr1[i].val;
                    uint32_t *second_ptr = (uint32_t *)vr2[i].val;
                    uint32_t result = (reg[rsrc[0]] * *second_ptr);
                    D(3,"Comparing " << reg[rsrc[0]] << " + " << *second_ptr << " = " << result);

                    result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = result;
                  }
                  for(Word i = c.vl; i < VLMAX; i++){
                    result_ptr = (uint32_t *) vd[i].val;
                    *result_ptr = 0;
                  }
                }                
              }
              break;
            }
          }
          break;
          case 7:
          {
            is_vec = true;
            c.vtype.vill = 0; //TODO
            c.vtype.vediv = vediv;
            c.vtype.vsew = vsew;
            c.vtype.vlmul = vlmul;

            Word VLMAX = (vlmul * c.VLEN)/vsew;
            D(3, "lmul:" << vlmul << " sew:" << vsew << " ediv: " << vediv << "rsrc" << reg[rsrc[0]] << "VLMAX" << VLMAX);

            if(reg[rsrc[0]] <= VLMAX){
              c.vl = reg[rsrc[0]];
            }
            else if(reg[rsrc[0]] < 2*VLMAX) {
              c.vl = (int)ceil((reg[rsrc[0]]*1.0)/2.0);
              D(3, "Length:" << c.vl << ceil(reg[rsrc[0]]/2));
            }
            else if(reg[rsrc[0]] >= (2*VLMAX)) {
              c.vl = VLMAX;
            }
            reg[rdest] = c.vl;
            D(3, "VL:" << reg[rdest]);

            Word regNum(0);

            c.vreg.clear();
            for (int j = 0; j < 32; j++)
            {
              c.vreg.push_back(vector<Reg<char*>>());
              for (int i = 0; i < (c.VLEN/vsew); ++i)
              {
                int * elem_ptr = (int *) malloc(vsew/8);
                for (int f = 0; f < (vsew/32); f++) elem_ptr[f] =  0;
                c.vreg[j].push_back(Reg<char*>(c.id, regNum++, (char *)  elem_ptr));
              }
            }
          }
          break;
          default:
          {
            cout << "default???\n" << flush;

          }
        }
      break;
      case VL:
      {
        is_vec = true;
        D(3, "Executing vector load");
        VLMAX = (c.vtype.vlmul * c.VLEN)/c.vtype.vsew;
        D(3, "lmul: " << c.vtype.vlmul << " VLEN:" << c.VLEN << "sew: " << c.vtype.vsew);
        D(3, "src: " << rsrc[0] << " " << reg[rsrc[0]]);
        D(3, "dest" << rdest);
        D(3, "width" << vlsWidth);
        vector<Reg<char *>> & vd  = c.vreg[rdest];

        switch(vlsWidth)
        {
            case 6: //load word and unit strided (not checking for unit stride)
            {
                for(Word i = 0; i < c.vl; i++) {
                  memAddr   = ((reg[rsrc[0]]) & 0xFFFFFFFC) + (i*c.vtype.vsew/8);  
                  data_read = c.core->mem.read(memAddr, c.supervisorMode);
                  D(3, "Mem addr: " << std::hex << memAddr << " Data read " << data_read);
                  int * result_ptr = (int *) vd[i].val;
                  *result_ptr = data_read;

                  trace_inst->is_lw = true;  
                  trace_inst->mem_addresses[i] = memAddr;
              }
              /*for(Word i = c.vl; i < VLMAX; i++){
                int * result_ptr = (int *) vd[i].val;
                *result_ptr = 0;
              }*/

              D(3, "Vector Register state ----:");
                // for(int i=0; i < 32; i++)
                // {
                //   for(int j=0; j< c.vl; j++)
                //   {
                //     cout << "starting iter" << endl;
                //       if (c.vtype.vsew == 8)
                //       {
                //         uint8_t * ptr_val = (uint8_t *) c.vreg[i][j].val;
                //         std::cout << "reg[" << i << "][" << j << "] = " << *ptr_val << std::endl;     
                //       } else if (c.vtype.vsew == 16)
                //       {
                //         uint16_t * ptr_val = (uint16_t *) c.vreg[i][j].val;
                //         std::cout << "reg[" << i << "][" << j << "] = " << *ptr_val << std::endl;     
                //       } else if (c.vtype.vsew == 32)
                //       {
                //         uint32_t * ptr_val = (uint32_t *) c.vreg[i][j].val;
                //         std::cout << "reg[" << i << "][" << j << "] = " << *ptr_val << std::endl;     
                //       }

                //       cout << "Finished iter" << endl;
                //   }
                // }

                // cout << "Finished loop" << endl;
            }
            // cout << "aaaaaaaaaaaaaaaaaaaaaa" << endl;
            break;
            default:
            {
              cout << "Serious default??\n" << flush;
            }
            break;
        }
        break;
      }
      break;
      case VS:
        is_vec = true;
        VLMAX = (c.vtype.vlmul * c.VLEN)/c.vtype.vsew;
        for(Word i = 0; i < c.vl; i++)
        {
          // cout << "iter" << endl;
          ++c.stores;
          memAddr = reg[rsrc[0]] + (i*c.vtype.vsew/8);
          // std::cout << "STORE MEM ADDRESS *** : " << std::hex << memAddr << "\n";

          
          trace_inst->is_sw = true;
          trace_inst->mem_addresses[i] = memAddr;

          switch (vlsWidth)
          {
            case 6: //store word and unit strided (not checking for unit stride)
            {
              uint32_t * ptr_val = (uint32_t *) c.vreg[vs3][i].val;
              D(3, "value: " << flush << (*ptr_val) << flush);
              c.core->mem.write(memAddr, *ptr_val, c.supervisorMode, 4);
              D(3, "store: " << memAddr << " value:" << *ptr_val << flush);
            }
            break;
            default:
              cout << "ERROR: UNSUPPORTED S INST\n" << flush;
              std::abort();
          }
          // cout << "Loop finished" << endl;
          // c.memAccesses.push_back(Warp::MemAccess(true, memAddr));
        }

        // cout << "After for loop" << endl;
      break;
      default:
        D(3, "pc: " << hex << (c.pc-4));
        D(3, "aERROR: Unsupported instruction: " << *this);
        std::abort();
    }

    // break;
    // cout << "outside case" << endl << flush;

  }

  // std::cout << "finished instruction" << endl << flush;

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
    D(3,"Next PC: " << hex << nextPc << dec);
  }
  
  if (nextActiveThreads > c.reg.size()) {
    cerr << "Error: attempt to spawn " << nextActiveThreads << " threads. "
         << c.reg.size() << " available.\n";
    abort();
  }
}
