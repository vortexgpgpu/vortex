#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <bitset>
#include <climits>
#include <cfenv>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "util.h"
#include "warp.h"
#include "instr.h"
#include "core.h"

using namespace vortex;

struct DivergentBranchException {};

static bool checkUnanimous(unsigned p, 
                           const std::vector<std::vector<Reg<Word>>> &m,
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

// Convert 32-bit integer register file to IEEE-754 floating point number.
float intregToFloat(uint32_t input) {
  // 31th bit
  bool sign = input & 0x80000000;
  // Exponent: 23th ~ 30th bits -> 8 bits in total
  int32_t exp  = ((input & 0x7F800000)>>23);
  // printf("exp = %u\n", exp);
  // 0th ~ 22th bits -> 23 bits fraction
  uint32_t frac = input & 0x007FFFFF;
  // Frac_value= 1 + sum{i = 1}{23}{b_{23-i}*2^{-i}}
  double frac_value;
  if (exp == 0) {  // subnormal
    if (frac == 0) // zero
      if (sign) return -0.0;
      else return 0.0;
    frac_value = 0.0;
  } else
    frac_value = 1.0;

  for (int i = 0; i < 23; i++) {
    int bi = frac & 0x1;
    frac_value += static_cast<double>(bi * pow(2.0, i-23));
    frac = (frac >> 1);
  }
  
  return (float)((static_cast<double>(pow(-1.0, sign))) * (static_cast<double>(pow(2.0, exp - 127.0)))* frac_value);
}

// Convert a floating point number to IEEE-754 32-bit representation, 
// so that it could be stored in a 32-bit integer register file
// Reference: https://www.wikihow.com/Convert-a-Number-from-Decimal-to-IEEE-754-Floating-Point-Representation
 //            https://www.technical-recipes.com/2012/converting-between-binary-and-decimal-representations-of-ieee-754-floating-point-numbers-in-c/
uint32_t floatToBin(float in_value) {
  union  {
       float input;   // assumes sizeof(float) == sizeof(int)
       int   output;
  }    data;

  data.input = in_value;

  std::bitset<sizeof(float) * CHAR_BIT>   bits(data.output);
  std::string mystring = bits.to_string<char, std::char_traits<char>, std::allocator<char> >();
  // Convert binary to uint32_t
  Word result = stoul(mystring, nullptr, 2);
  return result;
}

// print out floating point exception after execution
void show_fe_exceptions(void) {
    printf("exceptions raised:");
    if(fetestexcept(FE_DIVBYZERO)) printf(" FE_DIVBYZERO");
    if(fetestexcept(FE_INEXACT))   printf(" FE_INEXACT");
    if(fetestexcept(FE_INVALID))   printf(" FE_INVALID");
    if(fetestexcept(FE_OVERFLOW))  printf(" FE_OVERFLOW");
    if(fetestexcept(FE_UNDERFLOW)) printf(" FE_UNDERFLOW");
    feclearexcept(FE_ALL_EXCEPT);
    printf("\n");
}

// https://en.wikipedia.org/wiki/Single-precision_floating-point_format
// check floating-point number in binary format is NaN
uint8_t fpBinIsNan(uint32_t din) {
  bool fsign = din & 0x80000000;
  uint32_t expo = (din>>23) & 0x000000FF;
  uint32_t fraction = din & 0x007FFFFF;
  uint32_t bit_22 = din & 0x00400000;

  if ((expo==0xFF) && (fraction!=0)) 
    // if (!fsign && (fraction == 0x00400000)) 
    if(!fsign && (bit_22))
      return 1; // quiet NaN, return 1
    else 
      return 2; // signaling NaN, return 2
  return 0;
}

// check floating-point number in binary format is zero
uint8_t fpBinIsZero(uint32_t din) {
  bool fsign = din & 0x80000000;
  uint32_t expo = (din>>23) & 0x000000FF;
  uint32_t fraction = din & 0x007FFFFF;

  if ((expo==0) && (fraction==0))
    if (fsign)
      return 1; // negative 0
    else
      return 2; // positive 0
  return 0;  // not zero
}

// check floating-point number in binary format is infinity
uint8_t fpBinIsInf(uint32_t din) {
  bool fsign = din & 0x80000000;
  uint32_t expo = (din>>23) & 0x000000FF;
  uint32_t fraction = din & 0x007FFFFF;

  if ((expo==0xFF) && (fraction==0))
    if (fsign)
      return 1; // negative infinity
    else
      return 2; // positive infinity
  return 0;  // not infinity
}

void Warp::execute(Instr &instr, trace_inst_t *trace_inst) {
  Size nextActiveThreads = activeThreads_;
  Size wordSz = core_->arch().getWordSize();
  Word nextPc = pc_;

  memAccesses_.clear();

  bool sjOnce(true);  // Has not yet split or joined once.
  bool pcSet(false);  // PC has already been set
  
  Word func3 = instr.getFunc3();
  Word func6 = instr.getFunc6();
  Word func7 = instr.getFunc7();

  Opcode opcode = instr.getOpcode();
  RegNum rdest  = instr.getRDest();
  RegNum rsrc0  = instr.getRSrc(0);
  RegNum rsrc1  = instr.getRSrc(1);
  RegNum rsrc2  = instr.getRSrc(2);
  Word immsrc   = instr.getImm();
  bool vmask    = instr.getVmask();

  for (Size t = 0; t < activeThreads_; t++) {
    std::vector<Reg<Word>> &iregs = iRegFile_[t];
    std::vector<Reg<Word>> &fregs = fRegFile_[t];

    bool is_gpgpu = (opcode == GPGPU);
    bool is_tmc = is_gpgpu && (func3 == 0);
    bool is_wspawn = is_gpgpu && (func3 == 1);
    bool is_barrier = is_gpgpu && (func3 == 4);

    bool not_active = !tmask_[t];
    bool gpgpu_zero = (is_tmc || is_barrier || is_wspawn) && (t != 0);    

    if (not_active || gpgpu_zero)
      continue;

    ++insts_;

    switch (opcode) {
    case NOP:
      //std::cout << "NOP_INST\n";
      break;
    case R_INST: {
      // std::cout << "R_INST\n";
      Word m_exten = func7 & 0x1;
      if (m_exten) {
        // std::cout << "FOUND A MUL/DIV\n";

        switch (func3) {
        case 0:
          // MUL
          D(3, "MUL: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          iregs[rdest] = ((int)iregs[rsrc0]) * ((int)iregs[rsrc1]);
          break;
        case 1:
          // MULH
          D(3, "MULH: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          {
            int64_t first = (int64_t)iregs[rsrc0];
            if (iregs[rsrc0] & 0x80000000) {
              first = first | 0xFFFFFFFF00000000;
            }
            int64_t second = (int64_t)iregs[rsrc1];
            if (iregs[rsrc1] & 0x80000000) {
              second = second | 0xFFFFFFFF00000000;
            }
            // cout << "mulh: " << std::dec << first << " * " << second;
            uint64_t result = first * second;
            iregs[rdest] = (result >> 32) & 0xFFFFFFFF;
            // cout << " = " << result << "   or  " <<  iregs[rdest] << "\n";
          }
          break;
        case 2:
          // MULHSU
          D(3, "MULHSU: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          {
            int64_t first = (int64_t)iregs[rsrc0];
            if (iregs[rsrc0] & 0x80000000) {
              first = first | 0xFFFFFFFF00000000;
            }
            int64_t second = (int64_t)iregs[rsrc1];
            iregs[rdest] = ((first * second) >> 32) & 0xFFFFFFFF;
          }
          break;
        case 3:
          // MULHU
          D(3, "MULHU: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          {
            uint64_t first = (uint64_t)iregs[rsrc0];
            uint64_t second = (uint64_t)iregs[rsrc1];
            // cout << "MULHU\n";
            iregs[rdest] = ((first * second) >> 32) & 0xFFFFFFFF;
          }
          break;
        case 4:
          // DIV
          D(3, "DIV: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          if (iregs[rsrc1] == 0) {
            iregs[rdest] = -1;
            break;
          }
          // cout << "dividing: " << std::dec << ((int) iregs[rsrc0]) << " / " << ((int) iregs[rsrc1]);
          iregs[rdest] = ((int)iregs[rsrc0]) / ((int)iregs[rsrc1]);
          // cout << " = " << ((int) iregs[rdest]) << "\n";
          break;
        case 5:
          // DIVU
          D(3, "DIVU: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          if (iregs[rsrc1] == 0) {
            iregs[rdest] = -1;
            break;
          }
          iregs[rdest] = ((uint32_t)iregs[rsrc0]) / ((uint32_t)iregs[rsrc1]);
          break;
        case 6:
          // REM
          D(3, "REM: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          if (iregs[rsrc1] == 0) {
            iregs[rdest] = iregs[rsrc0];
            break;
          }
          iregs[rdest] = ((int)iregs[rsrc0]) % ((int)iregs[rsrc1]);
          break;
        case 7:
          // REMU
          D(3, "REMU: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          if (iregs[rsrc1] == 0) {
            iregs[rdest] = iregs[rsrc0];
            break;
          }
          iregs[rdest] = ((uint32_t)iregs[rsrc0]) % ((uint32_t)iregs[rsrc1]);
          break;
        default:
          std::cout << "unsupported MUL/DIV instr\n";
          std::abort();
        }
      } else {
        // std::cout << "NORMAL R-TYPE\n";
        switch (func3) {
        case 0:
          if (func7) {
            D(3, "SUBI: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
            iregs[rdest] = iregs[rsrc0] - iregs[rsrc1];
            iregs[rdest].trunc(wordSz);
          } else {
            D(3, "ADDI: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
            iregs[rdest] = iregs[rsrc0] + iregs[rsrc1];
            iregs[rdest].trunc(wordSz);
          }
          break;
        case 1:
          D(3, "SLLI: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          iregs[rdest] = iregs[rsrc0] << iregs[rsrc1];
          iregs[rdest].trunc(wordSz);
          break;
        case 2:
          D(3, "SLTI: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          if (int(iregs[rsrc0]) < int(iregs[rsrc1])) {
            iregs[rdest] = 1;
          } else {
            iregs[rdest] = 0;
          }
          break;
        case 3:
          D(3, "SLTU: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          if (Word_u(iregs[rsrc0]) < Word_u(iregs[rsrc1])) {
            iregs[rdest] = 1;
          } else {
            iregs[rdest] = 0;
          }
          break;
        case 4:
          D(3, "XORI: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          iregs[rdest] = iregs[rsrc0] ^ iregs[rsrc1];
          break;
        case 5:
          if (func7) {
            D(3, "SRLI: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
            iregs[rdest] = int(iregs[rsrc0]) >> int(iregs[rsrc1]);
            iregs[rdest].trunc(wordSz);
          } else {
            D(3, "SRLU: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
            iregs[rdest] = Word_u(iregs[rsrc0]) >> Word_u(iregs[rsrc1]);
            iregs[rdest].trunc(wordSz);
          }
          break;
        case 6:
          D(3, "ORI: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          iregs[rdest] = iregs[rsrc0] | iregs[rsrc1];
          break;
        case 7:
          D(3, "ANDI: r" << rdest << " <- r" << rsrc0 << ", r" << rsrc1);
          iregs[rdest] = iregs[rsrc0] & iregs[rsrc1];
          break;
        default:
          std::cout << "ERROR: UNSUPPORTED R INST\n";
          std::abort();
        }
      }
    } break;
    case L_INST: {
      Word memAddr = ((iregs[rsrc0] + immsrc) & 0xFFFFFFFC);
      Word shift_by = ((iregs[rsrc0] + immsrc) & 0x00000003) * 8;
      Word data_read = core_->mem().read(memAddr, 0);
      trace_inst->is_lw = true;
      trace_inst->mem_addresses[t] = memAddr;
      switch (func3) {
      case 0:
        // LBI
        D(3, "LBI: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        iregs[rdest] = signExt((data_read >> shift_by) & 0xFF, 8, 0xFF);
        break;
      case 1:
        // LWI
        D(3, "LWI: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        iregs[rdest] = signExt((data_read >> shift_by) & 0xFFFF, 16, 0xFFFF);
        break;
      case 2:
        // LDI
        D(3, "LDI: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        iregs[rdest] = int(data_read & 0xFFFFFFFF);
        break;
      case 4:
        // LBU
        D(3, "LBU: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        iregs[rdest] = unsigned((data_read >> shift_by) & 0xFF);
        break;
      case 5:
        // LWU
        D(3, "LWU: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        iregs[rdest] = unsigned((data_read >> shift_by) & 0xFFFF);
        break;
      default:
        std::cout << "ERROR: UNSUPPORTED L INST\n";
        std::abort();        
      }
      D(3, "LOAD MEM ADDRESS: " << std::hex << memAddr);
      D(3, "LOAD MEM DATA: " << std::hex << data_read);
      memAccesses_.push_back(Warp::MemAccess(false, memAddr));
    } break;
    case I_INST:
      //std::cout << "I_INST\n";
      switch (func3) {
      case 0:
        // ADDI
        D(3, "ADDI: r" << rdest << " <- r" << rsrc0 << ", imm=" << immsrc);
        iregs[rdest] = iregs[rsrc0] + immsrc;
        iregs[rdest].trunc(wordSz);
        break;
      case 2:
        // SLTI
        D(3, "SLTI: r" << rdest << " <- r" << rsrc0 << ", imm=" << immsrc);
        if (int(iregs[rsrc0]) < int(immsrc)) {
          iregs[rdest] = 1;
        } else {
          iregs[rdest] = 0;
        }
        break;
      case 3: {
        // SLTIU
        D(3, "SLTIU: r" << rdest << " <- r" << rsrc0 << ", imm=" << immsrc);
        if (unsigned(iregs[rsrc0]) < unsigned(immsrc)) {
          iregs[rdest] = 1;
        } else {
          iregs[rdest] = 0;
        }
      } break;
      case 4:
        // XORI
        D(3, "XORI: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        iregs[rdest] = iregs[rsrc0] ^ immsrc;
        break;
      case 6:
        // ORI
        D(3, "ORI: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        iregs[rdest] = iregs[rsrc0] | immsrc;
        break;
      case 7:
        // ANDI
        D(3, "ANDI: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        iregs[rdest] = iregs[rsrc0] & immsrc;
        break;
      case 1:
        // SLLI
        D(3, "SLLI: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        iregs[rdest] = iregs[rsrc0] << immsrc;
        iregs[rdest].trunc(wordSz);
        break;
      case 5:
        if ((func7 == 0)) {
          // SRLI
          D(3, "SRLI: r" << rdest << " <- r" << rsrc0 << ", imm=" << immsrc);
          Word result = Word_u(iregs[rsrc0]) >> Word_u(immsrc);
          iregs[rdest] = result;
          iregs[rdest].trunc(wordSz);
        } else {
          // SRAI
          D(3, "SRAI: r" << rdest << " <- r" << rsrc0 << ", imm=" << immsrc);
          Word op1 = iregs[rsrc0];
          Word op2 = immsrc;
          iregs[rdest] = op1 >> op2;
          iregs[rdest].trunc(wordSz);
        }
        break;
      default:
        std::cout << "ERROR: UNSUPPORTED L INST\n";
        std::abort();
      }
      break;
    case S_INST: {
      ++stores_;
      Word memAddr = iregs[rsrc0] + immsrc;
      trace_inst->is_sw = true;
      trace_inst->mem_addresses[t] = memAddr;
      // //std::cout << "FUNC3: " << func3 << "\n";
      if ((memAddr == 0x00010000) && (t == 0)) {
        Word num = iregs[rsrc1];
        fprintf(stderr, "%c", (char)num);
        break;
      }
      switch (func3) {
      case 0:
        // SB
        D(3, "SB: r" << rsrc1 << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        core_->mem().write(memAddr, iregs[rsrc1] & 0x000000FF, 0, 1);
        break;
      case 1:
        // SH
        D(3, "SH: r" << rsrc1 << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        core_->mem().write(memAddr, iregs[rsrc1], 0, 2);
        break;
      case 2:
        // SD
        D(3, "SD: r" << rsrc1 << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
        core_->mem().write(memAddr, iregs[rsrc1], 0, 4);
        break;
      default:
        std::cout << "ERROR: UNSUPPORTED S INST\n";
        std::abort();
      }
      D(3, "STORE MEM ADDRESS: " << std::hex << memAddr);
      memAccesses_.push_back(Warp::MemAccess(true, memAddr));
    } break;
    case B_INST:
      trace_inst->stall_warp = true;
      switch (func3) {
      case 0:
        // BEQ
        D(3, "BEQ: r" << rsrc0 << ", r" << rsrc1 << ", imm=0x" << std::hex << immsrc);
        if (int(iregs[rsrc0]) == int(iregs[rsrc1])) {
          if (!pcSet)
            nextPc = (pc_ - 4) + immsrc;
          pcSet = true;
        }
        break;
      case 1:
        // BNE
        D(3, "BNE: r" << rsrc0 << ", r" << rsrc1 << ", imm=0x" << std::hex << immsrc);
        if (int(iregs[rsrc0]) != int(iregs[rsrc1])) {
          if (!pcSet)
            nextPc = (pc_ - 4) + immsrc;
          pcSet = true;
        }
        break;
      case 4:
        // BLT
        D(3, "BLT: r" << rsrc0 << ", r" << rsrc1 << ", imm=0x" << std::hex << immsrc);
        if (int(iregs[rsrc0]) < int(iregs[rsrc1])) {
          if (!pcSet)
            nextPc = (pc_ - 4) + immsrc;
          pcSet = true;
        }
        break;
      case 5:
        // BGE
        D(3, "BGE: r" << rsrc0 << ", r" << rsrc1 << ", imm=0x" << std::hex << immsrc);
        if (int(iregs[rsrc0]) >= int(iregs[rsrc1])) {
          if (!pcSet)
            nextPc = (pc_ - 4) + immsrc;
          pcSet = true;
        }
        break;
      case 6:
        // BLTU
        D(3, "BLTU: r" << rsrc0 << ", r" << rsrc1 << ", imm=0x" << std::hex << immsrc);
        if (Word_u(iregs[rsrc0]) < Word_u(iregs[rsrc1])) {
          if (!pcSet)
            nextPc = (pc_ - 4) + immsrc;
          pcSet = true;
        }
        break;
      case 7:
        // BGEU
        D(3, "BGEU: r" << rsrc0 << ", r" << rsrc1 << ", imm=0x" << std::hex << immsrc);
        if (Word_u(iregs[rsrc0]) >= Word_u(iregs[rsrc1])) {
          if (!pcSet)
            nextPc = (pc_ - 4) + immsrc;
          pcSet = true;
        }
        break;
      }
      break;
    case LUI_INST:
      D(3, "LUI: r" << rdest << " <- imm=0x" << std::hex << immsrc);
      iregs[rdest] = (immsrc << 12) & 0xfffff000;
      break;
    case AUIPC_INST:
      D(3, "AUIPC: r" << rdest << " <- imm=0x" << std::hex << immsrc);
      iregs[rdest] = ((immsrc << 12) & 0xfffff000) + (pc_ - 4);
      break;
    case JAL_INST:
      D(3, "JAL: r" << rdest << " <- imm=0x" << std::hex << immsrc);
      trace_inst->stall_warp = true;
      if (!pcSet) {
        nextPc = (pc_ - 4) + immsrc;
        //std::cout << "JAL... SETTING PC: " << nextPc << "\n";      
      }
      if (rdest != 0) {
        iregs[rdest] = pc_;
      }
      pcSet = true;
      break;
    case JALR_INST:
      D(3, "JALR: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
      trace_inst->stall_warp = true;
      if (!pcSet) {
        nextPc = iregs[rsrc0] + immsrc;
        //std::cout << "JALR... SETTING PC: " << nextPc << "\n";
      }
      if (rdest != 0) {
        iregs[rdest] = pc_;
      }
      pcSet = true;
      break;
    case SYS_INST: {
      D(3, "SYS_INST: r" << rdest << " <- r" << rsrc0 << ", imm=0x" << std::hex << immsrc);
      Word rs1 = iregs[rsrc0];
      Word csr_addr = immsrc & 0x00000FFF;
      // GPGPU CSR extension
      if (csr_addr == CSR_WTID) {
        // Warp threadID
        iregs[rdest] = t;
      } else if (csr_addr == CSR_LTID) {
        // Core threadID
        iregs[rdest] = t + 
                     id_ * core_->arch().getNumThreads();
      } else if (csr_addr == CSR_GTID) {
        // Processor threadID
        iregs[rdest] = t + 
                     id_ * core_->arch().getNumThreads() + 
                     core_->arch().getNumThreads() * core_->arch().getNumWarps() * core_->id();
      } else if (csr_addr == CSR_LWID) {
        // Core warpID
        iregs[rdest] = id_;
      } else if (csr_addr == CSR_GWID) {
        // Processor warpID        
        iregs[rdest] = id_ + core_->arch().getNumWarps() * core_->id();
      } else if (csr_addr == CSR_GCID) {
        // Processor coreID
        iregs[rdest] = core_->id();
      } else if (csr_addr == CSR_NT) {
        // Number of threads per warp
        iregs[rdest] = core_->arch().getNumThreads();
      } else if (csr_addr == CSR_NW) {
        // Number of warps per core
        iregs[rdest] = core_->arch().getNumWarps();
      } else if (csr_addr == CSR_NC) {
        // Number of cores
        iregs[rdest] = core_->arch().getNumCores();
      } else if (csr_addr == CSR_INSTRET) {
        // NumInsts
        iregs[rdest] = (Word)core_->num_instructions();
      } else if (csr_addr == CSR_INSTRET_H) {
        // NumInsts
        iregs[rdest] = (Word)(core_->num_instructions() >> 32);
      } else if (csr_addr == CSR_CYCLE) {
        // NumCycles
        iregs[rdest] = (Word)core_->num_steps();
      } else if (csr_addr == CSR_CYCLE_H) {
        // NumCycles
        iregs[rdest] = (Word)(core_->num_steps() >> 32);
      } else {
        switch (func3) {
        case 0:
          if (csr_addr < 2) {
            // ECALL/EBREAK
            nextActiveThreads = 0;
            spawned_ = false;
          }
          break;
        case 1:
          // CSRRW
          if (rdest != 0) {
            iregs[rdest] = csrs_[csr_addr];
          }
          csrs_[csr_addr] = rs1;
          break;
        case 2:
          // CSRRS
          if (rdest != 0) {
            iregs[rdest] = csrs_[csr_addr];
          }
          csrs_[csr_addr] = rs1 | csrs_[csr_addr];
          break;
        case 3:
          // CSRRC
          if (rdest != 0) {
            iregs[rdest] = csrs_[csr_addr];
          }
          csrs_[csr_addr] = rs1 & (~csrs_[csr_addr]);
          break;
        case 5:
          // CSRRWI
          if (rdest != 0) {
            iregs[rdest] = csrs_[csr_addr];
          }
          csrs_[csr_addr] = rsrc0;
          break;
        case 6:
          // CSRRSI
          if (rdest != 0) {
            iregs[rdest] = csrs_[csr_addr];
          }
          csrs_[csr_addr] = rsrc0 | csrs_[csr_addr];
          break;
        case 7:
          // CSRRCI
          if (rdest != 0) {
            iregs[rdest] = csrs_[csr_addr];
          }
          csrs_[csr_addr] = rsrc0 & (~csrs_[csr_addr]);
          break;
        default:
          break;
        }
      }
    } break;
    case FENCE:
      D(3, "FENCE");
      break;
    case PJ_INST:
      D(3, "PJ_INST: r" << rsrc0 << ", r" << rsrc1);
      if (iregs[rsrc0]) {
        if (!pcSet)
          nextPc = iregs[rsrc1];
        pcSet = true;
      }
      break;
    case GPGPU:
      switch (func3) {
      case 1:
        // WSPAWN
        D(3, "WSPAWN: r" << rsrc0 << ", r" << rsrc1);
        trace_inst->wspawn = true;
        if (sjOnce) {
          sjOnce = false;
          unsigned num_to_wspawn = std::min<unsigned>(iregs[rsrc0], core_->arch().getNumWarps());
          D(0, "Spawning " << num_to_wspawn << " new warps at PC: " << std::hex << iregs[rsrc1]);
          for (unsigned i = 1; i < num_to_wspawn; ++i) {
            Warp &newWarp(core_->warp(i));
            {
              newWarp.set_pc(iregs[rsrc1]);
              for (size_t kk = 0; kk < tmask_.size(); kk++) {
                if (kk == 0) {
                  newWarp.setTmask(kk, true);
                } else {
                  newWarp.setTmask(kk, false);
                }
              }
              newWarp.setActiveThreads(1);
              newWarp.setSpawned(true);
            }
          }
          break;
        }
        break;
      case 2: {
        // SPLIT
        D(3, "SPLIT: r" << rsrc0);
        trace_inst->stall_warp = true;
        if (sjOnce) {
          sjOnce = false;
          if (checkUnanimous(rsrc0, iRegFile_, tmask_)) {
            D(3, "Unanimous pred: " << rsrc0 << "  val: " << iregs[rsrc0] << "\n");
            DomStackEntry e(tmask_);
            e.uni = true;
            domStack_.push(e);
            break;
          }
          D(3, "Split: Original TM: ");
          DX( for (auto y : tmask_) D(3, y << " "); )

          DomStackEntry e(rsrc0, iRegFile_, tmask_, pc_);
          domStack_.push(tmask_);
          domStack_.push(e);
          for (unsigned i = 0; i < e.tmask.size(); ++i) {
            tmask_[i] = !e.tmask[i] && tmask_[i];
          }

          D(3, "Split: New TM");
          DX( for (auto y : tmask_) D(3, y << " "); )
          D(3, "Split: Pushed TM PC: " << std::hex << e.pc << std::dec << "\n");
          DX( for (auto y : e.tmask) D(3, y << " "); )
        }
        break;
      }
      case 3:
        // JOIN
        D(3, "JOIN");
        if (sjOnce) {
          sjOnce = false;
          if (!domStack_.empty() && domStack_.top().uni) {
            D(2, "Uni branch at join");
            printf("NEW DOMESTACK: \n");
            tmask_ = domStack_.top().tmask;
            domStack_.pop();
            break;
          }
          if (!domStack_.top().fallThrough) {
            if (!pcSet) {
              nextPc = domStack_.top().pc;
              D(3, "join: NOT FALLTHROUGH PC: " << std::hex << nextPc << std::dec);
            }
            pcSet = true;
          }

          D(3, "Join: Old TM: ");
          DX( for (auto y : tmask_) D(3, y << " "); )
          std::cout << "\n";
          tmask_ = domStack_.top().tmask;

          D(3, "Join: New TM: ");
          DX( for (auto y : tmask_) D(3, y << " "); )

          domStack_.pop();
        }
        break;
      case 4:
        trace_inst->stall_warp = true;
        // is_barrier
        break;
      case 0:
        // TMC
        D(3, "TMC: r" << rsrc0);
        trace_inst->stall_warp = true;
        nextActiveThreads = std::min<unsigned>(iregs[rsrc0], core_->arch().getNumThreads());
        {
          for (size_t ff = 0; ff < tmask_.size(); ff++) {
            if (ff < nextActiveThreads) {
              tmask_[ff] = true;
            } else {
              tmask_[ff] = false;
            }
          }
        }
        if (nextActiveThreads == 0) {
          spawned_ = false;
        }
        break;
      default:
        std::cout << "ERROR: UNSUPPORTED GPGPU INSTRUCTION " << instr << "\n";
      }
      break;
    case VSET_ARITH: {
      D(3, "VSET_ARITH");
      int VLMAX = (instr.getVlmul() * VLEN_) / instr.getVsew();
      switch (func3) {
      case 0: // vector-vector
        trace_inst->vs1 = rsrc0;
        trace_inst->vs2 = rsrc1;
        trace_inst->vd  = rdest;
        switch (func6) {
        case 0: {
          D(3, "Addition " << rsrc0 << " " << rsrc1 << " Dest:" << rdest);
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          std::vector<Reg<char *>> &mask = vregFile_[0];

          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *mask_ptr = (uint8_t *)mask[i].value();
              uint8_t value = (*mask_ptr & 0x1);
              if (vmask || (!vmask && value)) {
                uint8_t *first_ptr = (uint8_t *)vr1[i].value();
                uint8_t *second_ptr = (uint8_t *)vr2[i].value();
                uint8_t result = *first_ptr + *second_ptr;
                D(3, "Adding " << *first_ptr << " + " << *second_ptr << " = " << result);

                uint8_t *result_ptr = (uint8_t *)vd[i].value();
                *result_ptr = result;
              }
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *mask_ptr = (uint16_t *)mask[i].value();
              uint16_t value = (*mask_ptr & 0x1);
              if (vmask || (!vmask && value)) {
                uint16_t *first_ptr = (uint16_t *)vr1[i].value();
                uint16_t *second_ptr = (uint16_t *)vr2[i].value();
                uint16_t result = *first_ptr + *second_ptr;
                D(3, "Adding " << *first_ptr << " + " << *second_ptr << " = " << result);

                uint16_t *result_ptr = (uint16_t *)vd[i].value();
                *result_ptr = result;
              }
            }
          } else if (vtype_.vsew == 32) {
            D(3, "Doing 32 bit vector addition");
            for (int i = 0; i < vl_; i++) {
              int *mask_ptr = (int *)mask[i].value();
              int value = (*mask_ptr & 0x1);
              if (vmask || (!vmask && value)) {
                int *first_ptr = (int *)vr1[i].value();
                int *second_ptr = (int *)vr2[i].value();
                int result = *first_ptr + *second_ptr;
                D(3, "Adding " << *first_ptr << " + " << *second_ptr << " = " << result);

                int *result_ptr = (int *)vd[i].value();
                *result_ptr = result;
              }
            }
          }
          
          DX( 
            D(3, "Vector Register state after addition:");
            for (size_t i = 0; i < vregFile_.size(); i++) {
              for (size_t j = 0; j < vregFile_[0].size(); j++) {
                if (vtype_.vsew == 8) {
                  uint8_t *ptr_val = (uint8_t *)vregFile_[i][j].value();
                  D(3, "reg[" << i << "][" << j << "] = " << *ptr_val);
                } else if (vtype_.vsew == 16) {
                  uint16_t *ptr_val = (uint16_t *)vregFile_[i][j].value();
                  D(3, "reg[" << i << "][" << j << "] = " << *ptr_val);
                } else if (vtype_.vsew == 32) {
                  uint32_t *ptr_val = (uint32_t *)vregFile_[i][j].value();
                  D(3, "reg[" << i << "][" << j << "] = " << *ptr_val);
                }
              }
            }
            D(3, "After vector register state after addition");
          )    
                
        } break;
        case 24: //vmseq
        {
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (*first_ptr == *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (*first_ptr == *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (*first_ptr == *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
          }

        } break;
        case 25: //vmsne
        {
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (*first_ptr != *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (*first_ptr != *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (*first_ptr != *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
          }

        } break;
        case 26: //vmsltu
        {
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (*first_ptr < *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (*first_ptr < *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (*first_ptr < *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
          }

        } break;
        case 27: //vmslt
        {
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              int8_t *first_ptr = (int8_t *)vr1[i].value();
              int8_t *second_ptr = (int8_t *)vr2[i].value();
              int8_t result = (*first_ptr < *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int8_t *result_ptr = (int8_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t *first_ptr = (int16_t *)vr1[i].value();
              int16_t *second_ptr = (int16_t *)vr2[i].value();
              int16_t result = (*first_ptr < *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int16_t *result_ptr = (int16_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t *first_ptr = (int32_t *)vr1[i].value();
              int32_t *second_ptr = (int32_t *)vr2[i].value();
              int32_t result = (*first_ptr < *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int32_t *result_ptr = (int32_t *)vd[i].value();
              *result_ptr = result;
            }
          }
        } break;
        case 28: //vmsleu
        {
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
          }
        } break;
        case 29: //vmsle
        {
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              int8_t *first_ptr = (int8_t *)vr1[i].value();
              int8_t *second_ptr = (int8_t *)vr2[i].value();
              int8_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int8_t *result_ptr = (int8_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t *first_ptr = (int16_t *)vr1[i].value();
              int16_t *second_ptr = (int16_t *)vr2[i].value();
              int16_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int16_t *result_ptr = (int16_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t *first_ptr = (int32_t *)vr1[i].value();
              int32_t *second_ptr = (int32_t *)vr2[i].value();
              int32_t result = (*first_ptr <= *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int32_t *result_ptr = (int32_t *)vd[i].value();
              *result_ptr = result;
            }
          }
        } break;
        case 30: //vmsgtu
        {
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (*first_ptr > *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (*first_ptr > *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (*first_ptr > *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
          }
        } break;
        case 31: //vmsgt
        {
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              int8_t *first_ptr = (int8_t *)vr1[i].value();
              int8_t *second_ptr = (int8_t *)vr2[i].value();
              int8_t result = (*first_ptr > *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int8_t *result_ptr = (int8_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t *first_ptr = (int16_t *)vr1[i].value();
              int16_t *second_ptr = (int16_t *)vr2[i].value();
              int16_t result = (*first_ptr > *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int16_t *result_ptr = (int16_t *)vd[i].value();
              *result_ptr = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t *first_ptr = (int32_t *)vr1[i].value();
              int32_t *second_ptr = (int32_t *)vr2[i].value();
              int32_t result = (*first_ptr > *second_ptr) ? 1 : 0;
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              int32_t *result_ptr = (int32_t *)vd[i].value();
              *result_ptr = result;
            }
          }
        } break;
        }
        break;
      case 2: {
        trace_inst->vs1 = rsrc0;
        trace_inst->vs2 = rsrc1;
        trace_inst->vd = rdest;

        switch (func6) {
        case 24: //vmandnot
        {
          D(3, "vmandnot");
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t first_value = (*first_ptr & 0x1);
              uint8_t second_value = (*second_ptr & 0x1);
              uint8_t result = (first_value & !second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t first_value = (*first_ptr & 0x1);
              uint16_t second_value = (*second_ptr & 0x1);
              uint16_t result = (first_value & !second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t first_value = (*first_ptr & 0x1);
              uint32_t second_value = (*second_ptr & 0x1);
              uint32_t result = (first_value & !second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 25: //vmand
        {
          D(3, "vmand");
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t first_value = (*first_ptr & 0x1);
              uint8_t second_value = (*second_ptr & 0x1);
              uint8_t result = (first_value & second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t first_value = (*first_ptr & 0x1);
              uint16_t second_value = (*second_ptr & 0x1);
              uint16_t result = (first_value & second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }

            for (int i = vl_; i < VLMAX; i++) {
              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t first_value = (*first_ptr & 0x1);
              uint32_t second_value = (*second_ptr & 0x1);
              uint32_t result = (first_value & second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }

            for (int i = vl_; i < VLMAX; i++) {
              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 26: //vmor
        {
          D(3, "vmor");
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t first_value = (*first_ptr & 0x1);
              uint8_t second_value = (*second_ptr & 0x1);
              uint8_t result = (first_value | second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 16) {
            uint16_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t first_value = (*first_ptr & 0x1);
              uint16_t second_value = (*second_ptr & 0x1);
              uint16_t result = (first_value | second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 32) {
            uint32_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t first_value = (*first_ptr & 0x1);
              uint32_t second_value = (*second_ptr & 0x1);
              uint32_t result = (first_value | second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            D(3, "VLMAX: " << VLMAX);
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 27: //vmxor
        {
          D(3, "vmxor");
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            uint8_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t first_value = (*first_ptr & 0x1);
              uint8_t second_value = (*second_ptr & 0x1);
              uint8_t result = (first_value ^ second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);
              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 16) {
            uint16_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t first_value = (*first_ptr & 0x1);
              uint16_t second_value = (*second_ptr & 0x1);
              uint16_t result = (first_value ^ second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            uint32_t *result_ptr;

            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t first_value = (*first_ptr & 0x1);
              uint32_t second_value = (*second_ptr & 0x1);
              uint32_t result = (first_value ^ second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 28: //vmornot
        {
          D(3, "vmornot");
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t first_value = (*first_ptr & 0x1);
              uint8_t second_value = (*second_ptr & 0x1);
              uint8_t result = (first_value | !second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t first_value = (*first_ptr & 0x1);
              uint16_t second_value = (*second_ptr & 0x1);
              uint16_t result = (first_value | !second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t first_value = (*first_ptr & 0x1);
              uint32_t second_value = (*second_ptr & 0x1);
              uint32_t result = (first_value | !second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 29: //vmnand
        {
          D(3, "vmnand");
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t first_value = (*first_ptr & 0x1);
              uint8_t second_value = (*second_ptr & 0x1);
              uint8_t result = !(first_value & second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint8_t *result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t first_value = (*first_ptr & 0x1);
              uint16_t second_value = (*second_ptr & 0x1);
              uint16_t result = !(first_value & second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }

            for (int i = vl_; i < VLMAX; i++) {
              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t first_value = (*first_ptr & 0x1);
              uint32_t second_value = (*second_ptr & 0x1);
              uint32_t result = !(first_value & second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }

            for (int i = vl_; i < VLMAX; i++) {
              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 30: //vmnor
        {
          D(3, "vmnor");
          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            uint8_t *result_ptr;

            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t first_value = (*first_ptr & 0x1);
              uint8_t second_value = (*second_ptr & 0x1);
              uint8_t result = !(first_value | second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t first_value = (*first_ptr & 0x1);
              uint16_t second_value = (*second_ptr & 0x1);
              uint16_t result = !(first_value | second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint16_t *result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {

            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t first_value = (*first_ptr & 0x1);
              uint32_t second_value = (*second_ptr & 0x1);
              uint32_t result = !(first_value | second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              uint32_t *result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 31: //vmxnor
        {
          D(3, "vmxnor");
          uint8_t *result_ptr;

          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t first_value = (*first_ptr & 0x1);
              uint8_t second_value = (*second_ptr & 0x1);
              uint8_t result = !(first_value ^ second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 16) {
            uint16_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t first_value = (*first_ptr & 0x1);
              uint16_t second_value = (*second_ptr & 0x1);
              uint16_t result = !(first_value ^ second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            uint32_t *result_ptr;

            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t first_value = (*first_ptr & 0x1);
              uint32_t second_value = (*second_ptr & 0x1);
              uint32_t result = !(first_value ^ second_value);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 37: //vmul
        {
          D(3, "vmul");
          uint8_t *result_ptr;

          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (*first_ptr * *second_ptr);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 16) {
            uint16_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (*first_ptr * *second_ptr);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            uint32_t *result_ptr;

            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (*first_ptr * *second_ptr);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 45: //vmacc
        {
          D(3, "vmacc");
          uint8_t *result_ptr;

          std::vector<Reg<char *>> &vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (*first_ptr * *second_ptr);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr += result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 16) {
            uint16_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (*first_ptr * *second_ptr);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr += result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            uint32_t *result_ptr;

            for (int i = 0; i < vl_; i++) {
              uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (*first_ptr * *second_ptr);
              D(3, "Comparing " << *first_ptr << " + " << *second_ptr << " = " << result);

              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr += result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        }
      } break;
      case 6: {
        switch (func6) {
        case 0: {
          D(3, "vmadd.vx");
          uint8_t *result_ptr;

          //vector<Reg<char *>> & vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              //uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (iregs[rsrc0] + *second_ptr);
              D(3, "Comparing " << iregs[rsrc0] << " + " << *second_ptr << " = " << result);

              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 16) {
            uint16_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              //uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (iregs[rsrc0] + *second_ptr);
              D(3, "Comparing " << iregs[rsrc0] << " + " << *second_ptr << " = " << result);

              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            uint32_t *result_ptr;

            for (int i = 0; i < vl_; i++) {
              //uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (iregs[rsrc0] + *second_ptr);
              D(3, "Comparing " << iregs[rsrc0] << " + " << *second_ptr << " = " << result);

              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        case 37: //vmul.vx
        {
          D(3, "vmul.vx");
          uint8_t *result_ptr;

          //vector<Reg<char *>> & vr1 = vregFile_[rsrc0];
          std::vector<Reg<char *>> &vr2 = vregFile_[rsrc1];
          std::vector<Reg<char *>> &vd = vregFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              //uint8_t *first_ptr = (uint8_t *)vr1[i].value();
              uint8_t *second_ptr = (uint8_t *)vr2[i].value();
              uint8_t result = (iregs[rsrc0] * *second_ptr);
              D(3, "Comparing " << iregs[rsrc0] << " + " << *second_ptr << " = " << result);

              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint8_t *)vd[i].value();
              *result_ptr = 0;
            }
          } else if (vtype_.vsew == 16) {
            uint16_t *result_ptr;
            for (int i = 0; i < vl_; i++) {
              //uint16_t *first_ptr = (uint16_t *)vr1[i].value();
              uint16_t *second_ptr = (uint16_t *)vr2[i].value();
              uint16_t result = (iregs[rsrc0] * *second_ptr);
              D(3, "Comparing " << iregs[rsrc0] << " + " << *second_ptr << " = " << result);

              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint16_t *)vd[i].value();
              *result_ptr = 0;
            }

          } else if (vtype_.vsew == 32) {
            uint32_t *result_ptr;

            for (int i = 0; i < vl_; i++) {
              //uint32_t *first_ptr = (uint32_t *)vr1[i].value();
              uint32_t *second_ptr = (uint32_t *)vr2[i].value();
              uint32_t result = (iregs[rsrc0] * *second_ptr);
              D(3, "Comparing " << iregs[rsrc0] << " + " << *second_ptr << " = " << result);

              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              result_ptr = (uint32_t *)vd[i].value();
              *result_ptr = 0;
            }
          }
        } break;
        }
      } break;
      case 7: {
        vtype_.vill = 0; //TODO
        vtype_.vediv = instr.getVediv();
        vtype_.vsew  = instr.getVsew();
        vtype_.vlmul = instr.getVlmul();

        D(3, "lmul:" << vtype_.vlmul << " sew:" << vtype_.vsew  << " ediv: " << vtype_.vediv << "rsrc_" << iregs[rsrc0] << "VLMAX" << VLMAX);

        int s0 = iregs[rsrc0];

        if (s0 <= VLMAX) {
          vl_ = s0;
        } else if (s0 < (2 * VLMAX)) {
          vl_ = (int)ceil((s0 * 1.0) / 2.0);
          D(3, "Length:" << vl_ << ceil(s0 / 2));
        } else if (s0 >= (2 * VLMAX)) {
          vl_ = VLMAX;
        }
        
        iregs[rdest] = vl_;
        D(3, "VL:" << iregs[rdest]);

        Word regNum(0);

        vregFile_.clear();
        for (int j = 0; j < 32; j++) {
          vregFile_.push_back(std::vector<Reg<char *>>());
          for (Word i = 0; i < (VLEN_ / instr.getVsew()); ++i) {
            int *elem_ptr = (int *)malloc(instr.getVsew() / 8);
            for (Word f = 0; f < (instr.getVsew() / 32); f++)
              elem_ptr[f] = 0;
            vregFile_[j].push_back(Reg<char *>(id_, regNum++, (char *)elem_ptr));
          }
        }
      } break;
      default: {
        std::cout << "default???\n" << std::flush;
      }
      }
    } break;
    case (FL | VL):
      if ( func3==0x2 ) {
        //std::cout << "FL_INST\n"; 
        // rs1 is integer is register!
        Word memAddr = ((iregs[rsrc0] + immsrc) & 0xFFFFFFFC); // alignment
        D(9,"something weird happen!");
        Word data_read = core_->mem().read(memAddr, 0);
        D(3, "Memaddr");
        D_RAW(' ' << setw(8) << hex << memAddr << endl);
        trace_inst->is_lw = true;
        trace_inst->mem_addresses[t] = memAddr;
        // //std::cout <<std::hex<< "EXECUTE: " << fregs[rsrc0] << " + " << immsrc << " = " << memAddr <<  " -> data_read: " << data_read << "\n";
        switch (func3) {
        case 2: // FLW
          fregs[rdest] = data_read & 0xFFFFFFFF;
          D(3, "fpReg[rd]");
          D_RAW(' ' << setw(8) << hex << fregs[rdest] << endl);              
          break;
        default:
          std::cout << "ERROR: UNSUPPORTED FL INST\n";
          exit(1);
        }
        D(3, "LOAD MEM ADDRESS: " << std::hex << memAddr);
        D(3, "LOAD MEM DATA: " << std::hex << data_read);
        memAccesses_.push_back(Warp::MemAccess(false, memAddr));  
      } else {        
        D(3, "Executing vector load");      
        D(3, "lmul: " << vtype_.vlmul << " VLEN:" << VLEN_ << "sew: " << vtype_.vsew);
        D(3, "src: " << rsrc0 << " " << iregs[rsrc0]);
        D(3, "dest" << rdest);
        D(3, "width" << instr.getVlsWidth());

        std::vector<Reg<char *>> &vd = vregFile_[rdest];

        switch (instr.getVlsWidth()) {
        case 6: { //load word and unit strided (not checking for unit stride)
          for (int i = 0; i < vl_; i++) {
            Word memAddr = ((iregs[rsrc0]) & 0xFFFFFFFC) + (i * vtype_.vsew / 8);
            Word data_read = core_->mem().read(memAddr, 0);
            D(3, "Mem addr: " << std::hex << memAddr << " Data read " << data_read);
            int *result_ptr = (int *)vd[i].value();
            *result_ptr = data_read;

            trace_inst->is_lw = true;
            trace_inst->mem_addresses[i] = memAddr;
          }
          D(3, "Vector Register state ----:");
          // cout << "Finished loop" << std::endl;
        } break;
        default:
          std::cout << "Serious default??\n" << std::flush;
        }
        break;
      } 
      break;
    case (FS | VS):
      if ((func3 == 0x1) || (func3 == 0x2) 
       || (func3 == 0x3) || (func3 == 0x4)) {
        //std::cout << "FS_INST\n";
        ++stores_;
        // base is integer register!
        Word memAddr = iregs[rsrc0] + immsrc;
        D(3, "STORE MEM ADDRESS: " << std::hex << fregs[rsrc0] << " + " << immsrc << "\n");
        D(3, "STORE MEM ADDRESS: " << std::hex << memAddr);
        trace_inst->is_sw = true;
        trace_inst->mem_addresses[t] = memAddr;
        // //std::cout << "FUNC3: " << func3 << "\n";
        if ((memAddr == 0x00010000) && (t == 0)) { // ** Is this protected mem space?
          unsigned num = fregs[rsrc1];
          fprintf(stderr, "%c", (char) fregs[rsrc1]);
          break;
        }
        switch (func3) {
        case 1:
          std::cout << "ERROR: UNSUPPORTED FS INST\n";
          std::cout << "FSH\n";
          exit(1);
          // c.core->mem.write(memAddr, fregs[rsrc1], c.supervisorMode, 2);
          break;
        case 2:
          // //std::cout << std::hex << "FSW: about to write: " << fregs[rsrc1] << " to " << memAddr << "\n"; 
          core_->mem().write(memAddr, fregs[rsrc1], 0, 4);
          break;
        case 3:
          std::cout << "ERROR: UNSUPPORTED FS INST\n";
          std::cout << std::hex << "FSD (*not implemented*): about to write: " << fregs[rsrc1] << " to " << memAddr << "\n"; 
          exit(1);
          // c.core->mem.write(memAddr, reg[rsrc1], c.supervisorMode, 8);
          break;  
        case 4:
          std::cout << "ERROR: UNSUPPORTED FS INST\n";
          std::cout << std::hex << "FSQ (*not implemented*): about to write: " << fregs[rsrc1] << " to " << memAddr << "\n"; 
          exit(1);
          // c.core->mem.write(memAddr, reg[rsrc1], c.supervisorMode, 16);
          break;                           
        default:
          std::cout << "ERROR: UNSUPPORTED FS INST\n";
          exit(1);
        }
        D(3, "STORE MEM ADDRESS: " << std::hex << memAddr);
        memAccesses_.push_back(Warp::MemAccess(true, memAddr));
      } else {
        for (int i = 0; i < vl_; i++) {
          // cout << "iter" << std::endl;
          ++stores_;
          Word memAddr = iregs[rsrc0] + (i * vtype_.vsew / 8);
          // std::cout << "STORE MEM ADDRESS *** : " << std::hex << memAddr << "\n";

          trace_inst->is_sw = true;
          trace_inst->mem_addresses[i] = memAddr;

          switch (instr.getVlsWidth()) {
          case 6: //store word and unit strided (not checking for unit stride)
          {
            uint32_t *ptr_val = (uint32_t *)vregFile_[instr.getVs3()][i].value();
            D(3, "value: " << std::flush << (*ptr_val) << std::flush);
            core_->mem().write(memAddr, *ptr_val, 0, 4);
            D(3, "store: " << memAddr << " value:" << *ptr_val << std::flush);
          } break;
          default:
            std::cout << "ERROR: UNSUPPORTED S INST\n" << std::flush;
            std::abort();
          }
          // cout << "Loop finished" << std::endl;
        }
      }
      break;    
    case FCI: // floating point computational instruction
      switch (func7) {
        case 0x00: //FADD
        case 0x04: //FSUB
        case 0x08: //FMUL
        case 0x0c: //FDIV
        case 0x2c: //FSQRT
        {
          if (fpBinIsNan(fregs[rsrc0]) || fpBinIsNan(fregs[rsrc1])) { // if one of op is NaN
            D(3, "one or two rsrc is NaN!");
            // one of them is not quiet NaN, them set FCSR
            if ((fpBinIsNan(fregs[rsrc0])==2) | (fpBinIsNan(fregs[rsrc1])==2)) {
              csrs_[0x003] = csrs_[0x003] | 0x10; // set NV bit                
              csrs_[0x001] = csrs_[0x001] | 0x10; // set NV bit
            }
            if (fpBinIsNan(fregs[rsrc0]) && fpBinIsNan(fregs[rsrc1])) 
              fregs[rdest] = 0x7fc00000;  // canonical(quiet) NaN 
            else if (fpBinIsNan(fregs[rsrc0]))
              fregs[rdest] = fregs[rsrc1];
            else
              fregs[rdest] = fregs[rsrc0];
          } else {
            float fpsrc_0 = intregToFloat(fregs[rsrc0]);
            float fpsrc_1 = intregToFloat(fregs[rsrc1]);
            float fpOut;    
                      
            feclearexcept(FE_ALL_EXCEPT);

            if (func7 == 0x00)    // FADD 
              fpOut = fpsrc_0 + fpsrc_1;
            else if (func7==0x04) // FSUB
              fpOut = fpsrc_0 - fpsrc_1;
            else if (func7==0x08) // FMUL
              fpOut = fpsrc_0 * fpsrc_1;
            else if (func7==0x0c) // FDIV
              fpOut = fpsrc_0 / fpsrc_1;
            else if (func7==0x2c) // FSQRT
              fpOut = sqrt(fpsrc_0);  
            else {
              printf("#[ERROR]: bad thing happened in fadd/fsub/fmul...\n");                            
              exit(1);
            }
            //show_fe_exceptions(); // once shown, it will clear corresponding bits, just for debug
            
            // fcsr defined in riscv
            if (fetestexcept(FE_INEXACT)) {
              csrs_[0x003] = csrs_[0x003] | 0x1; // set NX bit
              csrs_[0x001] = csrs_[0x001] | 0x1; // set NX bit
            }
            
            if (fetestexcept(FE_UNDERFLOW)) {
              csrs_[0x003] = csrs_[0x003] | 0x2; // set UF bit
              csrs_[0x001] = csrs_[0x001] | 0x2; // set UF bit
            }

            if (fetestexcept(FE_OVERFLOW)) {
              csrs_[0x003] = csrs_[0x003] | 0x4; // set OF bit
              csrs_[0x001] = csrs_[0x001] | 0x4; // set OF bit
            }

            if (fetestexcept(FE_DIVBYZERO)) {
              csrs_[0x003] = csrs_[0x003] | 0x8; // set DZ bit
              csrs_[0x001] = csrs_[0x001] | 0x8; // set DZ bit
            } 

            if (fetestexcept(FE_INVALID)) {
              csrs_[0x003] = csrs_[0x003] | 0x10; // set NX bit
              csrs_[0x001] = csrs_[0x001] | 0x10; // set NX bit
            }

            D(3, "fpOut: " << fpOut);
            if (fpBinIsNan(floatToBin(fpOut)) == 0) {
              fregs[rdest] = floatToBin(fpOut);
            } else  {              
              // According to risc-v spec p.64 section 11.3
              // If the result is NaN, it is the canonical NaN
              fregs[rdest] = 0x7fc00000;
            }            
          }
        } break;

        // FSGNJ.S, FSGNJN.S FSGJNX.S
        case 0x10: {
          bool       fsign1 = fregs[rsrc0] & 0x80000000;
          uint32_t   fdata1 = fregs[rsrc0] & 0x7FFFFFFF;
          bool       fsign2 = fregs[rsrc1] & 0x80000000;
          uint32_t   fdata2 = fregs[rsrc1] & 0x7FFFFFFF;
          
          D(3, "fdata1 " << hex << fdata1 << endl);       
          D(3, "fsign2 " << hex << fsign2 << endl);   

          switch(func3) {            
          case 0: // FSGNJ.S
            fregs[rdest] = (fsign2 << 31) | fdata1;
            break;          
          case 1: // FSGNJN.S
            fsign2 = !fsign2;
            fregs[rdest] = (fsign2 << 31) | fdata1;
            break;          
          case 2: { // FSGJNX.S
            bool sign = fsign1 ^ fsign2;
            fregs[rdest] = (sign << 31) | fdata1;
            } break;
          }
        } break;

        // FMIN.S, FMAX.S
        case 0x14: {
          if (fpBinIsNan(fregs[rsrc0]) || fpBinIsNan(fregs[rsrc1])) { // if one of src is NaN
            // one of them is not quiet NaN, them set FCSR
            if ((fpBinIsNan(fregs[rsrc0])==2) | (fpBinIsNan(fregs[rsrc1])==2)) {
              csrs_[0x003] = csrs_[0x003] | 0x10; // set NV bit
              csrs_[0x001] = csrs_[0x001] | 0x10; // set NV bit
            }
            if (fpBinIsNan(fregs[rsrc0]) && fpBinIsNan(fregs[rsrc1])) 
              fregs[rdest] = 0x7fc00000;  // canonical(quiet) NaN 
            else if (fpBinIsNan(fregs[rsrc0]))
              fregs[rdest] = fregs[rsrc1];
            else
              fregs[rdest] = fregs[rsrc0];
          } else {
            uint8_t sr0IsZero = fpBinIsZero(fregs[rsrc0]);
            uint8_t sr1IsZero = fpBinIsZero(fregs[rsrc1]);

            if (sr0IsZero && sr1IsZero && (sr0IsZero != sr1IsZero)) { // both are zero and not equal
              // handle corner case that compare +0 and -0              
              if (func3) {
                // FMAX.S
                fregs[rdest] = (sr1IsZero==2)? fregs[rsrc1] : fregs[rsrc0];
              } else {
                // FMIM.S
                fregs[rdest] = (sr1IsZero==2)? fregs[rsrc0] : fregs[rsrc1];
              }
            } else {
              float rs1 = intregToFloat(fregs[rsrc0]);
              float rs2 = intregToFloat(fregs[rsrc1]);              
              if (func3) {
                // FMAX.S
                float fmax = std::max(rs1, rs2); 
                fregs[rdest] = floatToBin(fmax);
              } else {
                // FMIN.S
                float fmin = std::min(rs1, rs2);
                fregs[rdest] = floatToBin(fmin);
              }
            }
          }
        } break;
        
        // FCVT.W.S FCVT.WU.S
        case 0x60: {
          // TODO: Need to clip result if rounded result is not representable in the destination format
          // typedef uint32_t Word_u;
          // typedef int32_t  Word_s;
          // FCVT.W.S
          // Convert floating point to 32-bit signed integer
          float fpSrc = intregToFloat(fregs[rsrc0]);
          Word result = 0x00000000;
          bool outOfRange = false;
          // FCVT.W.S
          if (rsrc1 == 0) { // Not sure if need to change to floating point reg
            if (fpSrc > pow(2.0, 31) - 1 || fpBinIsNan(fregs[rsrc0]) || fpBinIsInf(fregs[rsrc0]) == 2) {
              feclearexcept(FE_ALL_EXCEPT);              
              outOfRange = true;
              // result = 2^31 - 1
              result = 0x7FFFFFFF;
            } else if (fpSrc < -1*pow(2.0, 31) || fpBinIsInf(fregs[rsrc0]) == 1) {
              feclearexcept(FE_ALL_EXCEPT);              
              outOfRange = true;
              // result = -1*2^31
              result = 0x80000000;
            } else {
              feclearexcept(FE_ALL_EXCEPT);              
              result = (int32_t) fpSrc;
            }
          } else {
            // FCVT.WU.S
            // Convert floating point to 32-bit unsigned integer
            if (fpSrc > pow(2.0, 32) - 1 || fpBinIsNan(fregs[rsrc0]) || fpBinIsInf(fregs[rsrc0]) == 2) {
              feclearexcept(FE_ALL_EXCEPT);              
              outOfRange = true;
              // result = 2^32 - 1
              result = 0xFFFFFFFF;
            } else if (fpSrc <= -1.0 || fpBinIsInf(fregs[rsrc0]) == 1) {
              feclearexcept(FE_ALL_EXCEPT);              
              outOfRange = true;
              // result = 0
              result = 0x00000000;
            } else {
              feclearexcept(FE_ALL_EXCEPT);              
              result = (uint32_t) fpSrc;
            }
          }

          //show_fe_exceptions(); // once shown, it will clear corresponding bits, just for debug
          
          // fcsr defined in riscv
          if (fetestexcept(FE_INEXACT)) {
            csrs_[0x003] = csrs_[0x003] | 0x1; // set NX bit
            csrs_[0x001] = csrs_[0x001] | 0x1; // set NX bit
          }
          
          if (fetestexcept(FE_UNDERFLOW)) {
            csrs_[0x003] = csrs_[0x003] | 0x2; // set UF bit
            csrs_[0x001] = csrs_[0x001] | 0x2; // set UF bit
          }

          if (fetestexcept(FE_OVERFLOW)) {
            csrs_[0x003] = csrs_[0x003] | 0x4; // set OF bit
            csrs_[0x001] = csrs_[0x001] | 0x4; // set OF bit
          }

          if (fetestexcept(FE_DIVBYZERO)) {
            csrs_[0x003] = csrs_[0x003] | 0x8; // set DZ bit
            csrs_[0x001] = csrs_[0x001] | 0x8; // set DZ bit
          } 

          if (fetestexcept(FE_INVALID) || outOfRange) {
            csrs_[0x003] = csrs_[0x003] | 0x10; // set NV bit
            csrs_[0x001] = csrs_[0x001] | 0x10; // set NV bit
          }

          iregs[rdest] = result;
        } break;
        
        // FMV.X.W FCLASS.S
        case 0x70: {
          // FCLASS.S
          if (func3) {
            // Examine the value in fpReg rs1 and write to integer rd
            // a 10-bit mask to indicate the class of the fp number
            iregs[rdest] = 0; // clear all bits

            bool fsign = fregs[rsrc0] & 0x80000000;
            uint32_t expo = (fregs[rsrc0]>>23) & 0x000000FF;
            uint32_t fraction = fregs[rsrc0] & 0x007FFFFF;

            if ((expo==0) && (fraction==0))
              iregs[rdest] = fsign? (1<<3) : (1<<4); // +/- 0
            else if ((expo==0) && (fraction!=0))
              iregs[rdest] = fsign? (1<<2) : (1<<5); // +/- subnormal
            else if ((expo==0xFF) && (fraction==0))
              iregs[rdest] = fsign? (1<<0) : (1<<7); // +/- infinity
            else if ((expo==0xFF) && (fraction!=0)) 
              if (!fsign && (fraction == 0x00400000)) 
                iregs[rdest] = (1<<9);               // quiet NaN
              else 
                iregs[rdest] = (1<<8);               // signaling NaN
            else
              iregs[rdest] = fsign? (1<<1) : (1<<6); // +/- normal
          } else {          
            // FMV.X.W
            // Move bit values from floating-point register rs1 to integer register rd
            // Since we are using integer register to represent floating point register, 
            // just simply assign here.
            iregs[rdest] = fregs[rsrc0];
          } 
        } break;
        
        // FEQ.S FLT.S FLE.S
        // rdest is integer register
        case 0x50: {
          // TODO: FLT.S and FLE.S perform IEEE 754-2009, signaling comparisons, set
          // TODO: the invalid operation exception flag if either input is NaN
          if (fpBinIsNan(fregs[rsrc0]) || fpBinIsNan(fregs[rsrc1])) {
            // FLE.S or FLT.S
            if (func3 == 0 || func3 == 1) {
              // If either input is NaN, set NV bit
              csrs_[0x003] = csrs_[0x003] | 0x10; // set NV bit
              csrs_[0x001] = csrs_[0x001] | 0x10; // set NV bit
            } else { // FEQ.S
              // Only set NV bit if it is signaling NaN
              if (fpBinIsNan(fregs[rsrc0]) == 2 || fpBinIsNan(fregs[rsrc1]) == 2) {
                // If either input is NaN, set NV bit
                csrs_[0x003] = csrs_[0x003] | 0x10; // set NV bit
                csrs_[0x001] = csrs_[0x001] | 0x10; // set NV bit
              }
            }
            // The result is 0 if either operand is NaN
            iregs[rdest] = 0;
          } else {
            switch(func3) {              
              case 0: {
                // FLE.S
                if (intregToFloat(fregs[rsrc0]) <= intregToFloat(fregs[rsrc1])) {
                  iregs[rdest] = 1;
                } else {
                  iregs[rdest] = 0;
                }
              } break;
              
              case 1: {
                // FLT.S
                if (intregToFloat(fregs[rsrc0]) < intregToFloat(fregs[rsrc1])) {
                  iregs[rdest] = 1;
                } else {
                  iregs[rdest] = 0;
                }
              }
              break;
              
              case 2: {
                // FEQ.S
                if (intregToFloat(fregs[rsrc0]) == intregToFloat(fregs[rsrc1])) {
                  iregs[rdest] = 1;
                } else {
                  iregs[rdest] = 0;
                }
              }
              break;
            }
          }
        } break;
        
        case 0x68: {
          // Cast integer to floating point
          float data = iregs[rsrc0];          
          if (rsrc1) {
            // FCVT.S.WU
            // Convert 32-bit unsigned integer to floating point
            fregs[rdest] = floatToBin(data);
          } else {
            // FCVT.S.W
            // Convert 32-bit signed integer to floating point
            // iregs[rsrc0] is actually a unsigned number
            data = (int) iregs[rsrc0];
            D(3, "data" << data);
            fregs[rdest] = floatToBin(data);
          }
        } break;

        case 0x78: {
          // FMV.W.X
          // Move bit values from integer register rs1 to floating register rd
          // Since we are using integer register to represent floating point register, 
          // just simply assign here.
          fregs[rdest] = iregs[rsrc0];
        }
        break;
      }
    break;

    case FMADD:      
    case FMSUB:      
    case FMNMADD:
    case FMNMSUB: {
      // multiplicands are infinity and zero, them set FCSR
      if (fpBinIsZero(fregs[rsrc0])|| fpBinIsZero(fregs[rsrc1])|| fpBinIsInf(fregs[rsrc0]) || fpBinIsInf(fregs[rsrc1])) {
        csrs_[0x003] = csrs_[0x003] | 0x10; // set NV bit
        csrs_[0x001] = csrs_[0x001] | 0x10; // set NV bit
      }

      if (fpBinIsNan(fregs[rsrc0]) || fpBinIsNan(fregs[rsrc1]) || fpBinIsNan(fregs[rsrc2])) { // if one of op is NaN
        // if addend is not quiet NaN, them set FCSR
        if ((fpBinIsNan(fregs[rsrc0])==2) | (fpBinIsNan(fregs[rsrc1])==2) | (fpBinIsNan(fregs[rsrc1])==2)) {
          csrs_[0x003] = csrs_[0x003] | 0x10; // set NV bit
          csrs_[0x001] = csrs_[0x001] | 0x10; // set NV bit          
        }

        fregs[rdest] = 0x7fc00000;  // canonical(quiet) NaN 
      } else {
        float rs1 = intregToFloat(fregs[rsrc0]);
        float rs2 = intregToFloat(fregs[rsrc1]);
        float rs3 = intregToFloat(fregs[rsrc2]);
        float fpOut(0.0);
        feclearexcept(FE_ALL_EXCEPT);          
        switch (opcode) {
          case FMADD:    
            // rd = (rs1*rs2)+rs3
            fpOut = (rs1 * rs2) + rs3;  break;
          case FMSUB:      
            // rd = (rs1*rs2)-rs3
            fpOut = (rs1 * rs2) - rs3; break;
          case FMNMADD:
            // rd = -(rs1*rs2)+rs3
            fpOut = -1*(rs1 * rs2) - rs3; break;        
          case FMNMSUB: 
            // rd = -(rs1*rs2)-rs3
            fpOut = -1*(rs1 * rs2) + rs3; break;
          default:
            printf("#[ERROR] FMADD/FMSUB... wrong\n");
            std::abort();
            break;                 
        }  

        // fcsr defined in riscv
        if (fetestexcept(FE_INEXACT)) {
          csrs_[0x003] = csrs_[0x003] | 0x1; // set NX bit
          csrs_[0x001] = csrs_[0x001] | 0x1; // set NX bit
        }
        
        if (fetestexcept(FE_UNDERFLOW)) {
          csrs_[0x003] = csrs_[0x003] | 0x2; // set UF bit
          csrs_[0x001] = csrs_[0x001] | 0x2; // set UF bit
        }

        if (fetestexcept(FE_OVERFLOW)) {
          csrs_[0x003] = csrs_[0x003] | 0x4; // set OF bit
          csrs_[0x001] = csrs_[0x001] | 0x4; // set OF bit
        }

        if (fetestexcept(FE_DIVBYZERO)) {
          csrs_[0x003] = csrs_[0x003] | 0x8; // set DZ bit
          csrs_[0x001] = csrs_[0x001] | 0x8; // set DZ bit
        } 

        if (fetestexcept(FE_INVALID)) {
          csrs_[0x003] = csrs_[0x003] | 0x10; // set NV bit
          csrs_[0x001] = csrs_[0x001] | 0x10; // set NV bit
        }

        fregs[rdest] = floatToBin(fpOut);
      }
    }
    break;
    default:
      D(3, "pc: " << std::hex << (pc_ - 4));
      D(3, "ERROR: Unsupported instruction: " << instr);
      std::abort();
    }
  }

  activeThreads_ = nextActiveThreads;

  // This way, if pc was set by a side effect (such as interrupt), it will
  // retain its new value.
  if (pcSet) {
    pc_ = nextPc;
    D(3, "Next PC: " << std::hex << nextPc << std::dec);
  }

  if (nextActiveThreads > iRegFile_.size()) {
    std::cerr << "Error: attempt to spawn " << nextActiveThreads << " threads. "
              << iRegFile_.size() << " available.\n";
    abort();
  }
}
