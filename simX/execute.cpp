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
#include <assert.h>
#include "util.h"
#include "warp.h"
#include "instr.h"
#include "core.h"

using namespace vortex;

struct DivergentBranchException {};

static bool checkUnanimous(unsigned p, 
                           const std::vector<std::vector<Word>> &m,
                           const ThreadMask &tm) {
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
    if (frac == 0) {
      // zero
      if (sign) 
        return -0.0;
      else 
        return 0.0;
    }
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

  if ((expo==0xFF) && (fraction!=0)) {
    // if (!fsign && (fraction == 0x00400000)) 
    if (!fsign && (bit_22))
      return 1; // quiet NaN, return 1
    else 
      return 2; // signaling NaN, return 2
  }
  return 0;
}

// check floating-point number in binary format is zero
uint8_t fpBinIsZero(uint32_t din) {
  bool fsign = din & 0x80000000;
  uint32_t expo = (din>>23) & 0x000000FF;
  uint32_t fraction = din & 0x007FFFFF;

  if ((expo==0) && (fraction==0)) {
    if (fsign)
      return 1; // negative 0
    else
      return 2; // positive 0
  }
  return 0;  // not zero
}

// check floating-point number in binary format is infinity
uint8_t fpBinIsInf(uint32_t din) {
  bool fsign = din & 0x80000000;
  uint32_t expo = (din>>23) & 0x000000FF;
  uint32_t fraction = din & 0x007FFFFF;

  if ((expo==0xFF) && (fraction==0)) {
    if (fsign)
      return 1; // negative infinity
    else
      return 2; // positive infinity
  }
  return 0;  // not infinity
}

void Warp::execute(Instr &instr, trace_inst_t *trace_inst) {
  assert(tmask_.any());

  Word nextPC = PC_;
  bool updatePC = false;
  bool runOnce = false;
  
  Word func3 = instr.getFunc3();
  Word func6 = instr.getFunc6();
  Word func7 = instr.getFunc7();

  auto opcode = instr.getOpcode();
  int rdest  = instr.getRDest();
  int rsrc0  = instr.getRSrc(0);
  int rsrc1  = instr.getRSrc(1);
  int rsrc2  = instr.getRSrc(2);
  Word immsrc= instr.getImm();
  bool vmask = instr.getVmask();

  int num_threads = core_->arch().num_threads();
  for (int t = 0; t < num_threads; t++) {
    if (!tmask_.test(t) || runOnce)
      continue;
    
    auto &iregs = iRegFile_.at(t);
    auto &fregs = fRegFile_.at(t);

    ++insts_;

    switch (opcode) {
    case NOP:
      //std::cout << "NOP_INST\n";
      break;
    case R_INST: {
      // std::cout << "R_INST\n";
      Word is_mul_ext = func7 & 0x1;
      if (is_mul_ext) {
        // std::cout << "FOUND A MUL/DIV\n";
        switch (func3) {
        case 0:
          // MUL
          D(3, "MUL: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
          if (rdest) iregs[rdest] = ((int)iregs[rsrc0]) * ((int)iregs[rsrc1]);
          break;
        case 1:
          // MULH
          D(3, "MULH: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
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
            if (rdest) iregs[rdest] = (result >> 32) & 0xFFFFFFFF;
            // cout << " = " << result << "   or  " <<  iregs[rdest] << "\n";
          }
          break;
        case 2:
          // MULHSU
          D(3, "MULHSU: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
          {
            int64_t first = (int64_t)iregs[rsrc0];
            if (iregs[rsrc0] & 0x80000000) {
              first = first | 0xFFFFFFFF00000000;
            }
            int64_t second = (int64_t)iregs[rsrc1];
            if (rdest) iregs[rdest] = ((first * second) >> 32) & 0xFFFFFFFF;
          }
          break;
        case 3:
          // MULHU
          D(3, "MULHU: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
          {
            uint64_t first = (uint64_t)iregs[rsrc0];
            uint64_t second = (uint64_t)iregs[rsrc1];
            // cout << "MULHU\n";
            if (rdest) iregs[rdest] = ((first * second) >> 32) & 0xFFFFFFFF;
          }
          break;
        case 4:
          // DIV
          D(3, "DIV: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
          if (iregs[rsrc1] == 0) {
            if (rdest) iregs[rdest] = -1;
            break;
          }
          // cout << "dividing: " << std::dec << ((int) iregs[rsrc0]) << " / " << ((int) iregs[rsrc1]);
          if (rdest) iregs[rdest] = ((int)iregs[rsrc0]) / ((int)iregs[rsrc1]);
          // cout << " = " << ((int) iregs[rdest]) << "\n";
          break;
        case 5:
          // DIVU
          D(3, "DIVU: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
          if (iregs[rsrc1] == 0) {
            if (rdest) iregs[rdest] = -1;
            break;
          }
          if (rdest) iregs[rdest] = ((uint32_t)iregs[rsrc0]) / ((uint32_t)iregs[rsrc1]);
          break;
        case 6:
          // REM
          D(3, "REM: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
          if (iregs[rsrc1] == 0) {
            if (rdest) iregs[rdest] = iregs[rsrc0];
            break;
          }
          if (rdest) iregs[rdest] = ((int)iregs[rsrc0]) % ((int)iregs[rsrc1]);
          break;
        case 7:
          // REMU
          D(3, "REMU: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
          if (iregs[rsrc1] == 0) {
            if (rdest) iregs[rdest] = iregs[rsrc0];
            break;
          }
          if (rdest) iregs[rdest] = ((uint32_t)iregs[rsrc0]) % ((uint32_t)iregs[rsrc1]);
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
            D(3, "SUBI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
            if (rdest) iregs[rdest] = iregs[rsrc0] - iregs[rsrc1];
          } else {
            D(3, "ADDI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
            if (rdest) iregs[rdest] = iregs[rsrc0] + iregs[rsrc1];
          }
          break;
        case 1:
          D(3, "SLLI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
          if (rdest) iregs[rdest] = iregs[rsrc0] << iregs[rsrc1];
          break;
        case 2:
          D(3, "SLTI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
          if (int(iregs[rsrc0]) < int(iregs[rsrc1])) {
            if (rdest) iregs[rdest] = 1;
          } else {
            if (rdest) iregs[rdest] = 0;
          }
          break;
        case 3:
          D(3, "SLTU: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
          if (Word(iregs[rsrc0]) < Word(iregs[rsrc1])) {
            if (rdest) iregs[rdest] = 1;
          } else {
            if (rdest) iregs[rdest] = 0;
          }
          break;
        case 4:
          D(3, "XORI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
          if (rdest) iregs[rdest] = iregs[rsrc0] ^ iregs[rsrc1];
          break;
        case 5:
          if (func7) {
            D(3, "SRLI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
            if (rdest) iregs[rdest] = int(iregs[rsrc0]) >> int(iregs[rsrc1]);
          } else {
            D(3, "SRLU: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
            if (rdest) iregs[rdest] = Word(iregs[rsrc0]) >> Word(iregs[rsrc1]);
          }
          break;
        case 6:
          D(3, "ORI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
          if (rdest) iregs[rdest] = iregs[rsrc0] | iregs[rsrc1];
          break;
        case 7:
          D(3, "AND: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
          if (rdest) iregs[rdest] = iregs[rsrc0] & iregs[rsrc1];
          break;
        default:
          std::cout << "ERROR: UNSUPPORTED R INST\n";
          std::abort();
        }
      }
    } break;
    case I_INST:
      //std::cout << "I_INST\n";
      switch (func3) {
      case 0:
        // ADDI
        D(3, "ADDI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=" << immsrc);
        if (rdest) iregs[rdest] = iregs[rsrc0] + immsrc;
        break;
      case 2:
        // SLTI
        D(3, "SLTI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=" << immsrc);
        if (int(iregs[rsrc0]) < int(immsrc)) {
          if (rdest) iregs[rdest] = 1;
        } else {
          if (rdest) iregs[rdest] = 0;
        }
        break;
      case 3: {
        // SLTIU
        D(3, "SLTIU: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=" << immsrc);
        if (unsigned(iregs[rsrc0]) < unsigned(immsrc)) {
          if (rdest) iregs[rdest] = 1;
        } else {
          if (rdest) iregs[rdest] = 0;
        }
      } break;
      case 4:
        // XORI
        D(3, "XORI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=0x" << std::hex << immsrc);
        if (rdest) iregs[rdest] = iregs[rsrc0] ^ immsrc;
        break;
      case 6:
        // ORI
        D(3, "ORI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=0x" << std::hex << immsrc);
        if (rdest) iregs[rdest] = iregs[rsrc0] | immsrc;
        break;
      case 7:
        // ANDI
        D(3, "ANDI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=0x" << std::hex << immsrc);
        if (rdest) iregs[rdest] = iregs[rsrc0] & immsrc;
        break;
      case 1:
        // SLLI
        D(3, "SLLI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=0x" << std::hex << immsrc);
        if (rdest) iregs[rdest] = iregs[rsrc0] << immsrc;
        break;
      case 5:
        if ((func7 == 0)) {
          // SRLI
          D(3, "SRLI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=" << immsrc);
          Word result = Word(iregs[rsrc0]) >> Word(immsrc);
          if (rdest) iregs[rdest] = result;
        } else {
          // SRAI
          D(3, "SRAI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=" << immsrc);
          Word op1 = iregs[rsrc0];
          Word op2 = immsrc;
          if (rdest) iregs[rdest] = op1 >> op2;
        }
        break;
      default:
        std::cout << "ERROR: UNSUPPORTED L INST\n";
        std::abort();
      }
      break;
    case L_INST: {
      ++loads_;
      Word memAddr = ((iregs[rsrc0] + immsrc) & 0xFFFFFFFC);
      Word shift_by = ((iregs[rsrc0] + immsrc) & 0x00000003) * 8;
      Word data_read = core_->dcache_read(memAddr, 0);
      trace_inst->is_lw = true;
      trace_inst->mem_addresses[t] = memAddr;
      switch (func3) {
      case 0:
        // LBI
        D(3, "LBI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=0x" << std::hex << immsrc);
        if (rdest) iregs[rdest] = signExt((data_read >> shift_by) & 0xFF, 8, 0xFF);
        break;
      case 1:
        // LWI
        D(3, "LHI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=0x" << std::hex << immsrc);
        if (rdest) iregs[rdest] = signExt((data_read >> shift_by) & 0xFFFF, 16, 0xFFFF);
        break;
      case 2:
        // LDI
        D(3, "LWI: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=0x" << std::hex << immsrc);
        if (rdest) iregs[rdest] = int(data_read & 0xFFFFFFFF);
        break;
      case 4:
        // LBU
        D(3, "LBU: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=0x" << std::hex << immsrc);
        if (rdest) iregs[rdest] = unsigned((data_read >> shift_by) & 0xFF);
        break;
      case 5:
        // LWU
        D(3, "LHU: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=0x" << std::hex << immsrc);
        if (rdest) iregs[rdest] = unsigned((data_read >> shift_by) & 0xFFFF);
        break;
      default:
        std::cout << "ERROR: UNSUPPORTED L INST\n";
        std::abort();        
      }
      D(3, "LOAD MEM ADDRESS: " << std::hex << memAddr);
    } break;
    case S_INST: {
      ++stores_;
      Word memAddr = iregs[rsrc0] + immsrc;
      trace_inst->is_sw = true;
      trace_inst->mem_addresses[t] = memAddr;
      switch (func3) {
      case 0:
        // SB
        D(3, "SB: r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1] << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=0x" << std::hex << immsrc);
        core_->dcache_write(memAddr, iregs[rsrc1] & 0x000000FF, 0, 1);
        break;
      case 1:
        // SH
        D(3, "SH: r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1] << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=0x" << std::hex << immsrc);
        core_->dcache_write(memAddr, iregs[rsrc1], 0, 2);
        break;
      case 2:
        // SW
        D(3, "SW: r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1] << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=0x" << std::hex << immsrc);
        core_->dcache_write(memAddr, iregs[rsrc1], 0, 4);
        break;
      default:
        std::cout << "ERROR: UNSUPPORTED S INST\n";
        std::abort();
      }
      D(3, "STORE MEM ADDRESS: " << std::hex << memAddr);
    } break;
    case B_INST:
      trace_inst->stall_warp = true;
      switch (func3) {
      case 0:
        // BEQ
        D(3, "BEQ: r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1] << ", imm=0x" << std::hex << immsrc);
        if (int(iregs[rsrc0]) == int(iregs[rsrc1])) {
          if (!updatePC)
            nextPC = (PC_ - 4) + immsrc;
          updatePC = true;
        }
        break;
      case 1:
        // BNE
        D(3, "BNE: r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1] << ", imm=0x" << std::hex << immsrc);
        if (int(iregs[rsrc0]) != int(iregs[rsrc1])) {
          if (!updatePC)
            nextPC = (PC_ - 4) + immsrc;
          updatePC = true;
        }
        break;
      case 4:
        // BLT
        D(3, "BLT: r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1] << ", imm=0x" << std::hex << immsrc);
        if (int(iregs[rsrc0]) < int(iregs[rsrc1])) {
          if (!updatePC)
            nextPC = (PC_ - 4) + immsrc;
          updatePC = true;
        }
        break;
      case 5:
        // BGE
        D(3, "BGE: r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1] << ", imm=0x" << std::hex << immsrc);
        if (int(iregs[rsrc0]) >= int(iregs[rsrc1])) {
          if (!updatePC)
            nextPC = (PC_ - 4) + immsrc;
          updatePC = true;
        }
        break;
      case 6:
        // BLTU
        D(3, "BLTU: r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1] << ", imm=0x" << std::hex << immsrc);
        if (Word(iregs[rsrc0]) < Word(iregs[rsrc1])) {
          if (!updatePC)
            nextPC = (PC_ - 4) + immsrc;
          updatePC = true;
        }
        break;
      case 7:
        // BGEU
        D(3, "BGEU: r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1] << ", imm=0x" << std::hex << immsrc);
        if (Word(iregs[rsrc0]) >= Word(iregs[rsrc1])) {
          if (!updatePC)
            nextPC = (PC_ - 4) + immsrc;
          updatePC = true;
        }
        break;
      }
      break;
    case LUI_INST:
      D(3, "LUI: r" << std::dec << rdest << " <- imm=0x" << std::hex << immsrc);
      if (rdest) iregs[rdest] = (immsrc << 12) & 0xfffff000;
      break;
    case AUIPC_INST:
      D(3, "AUIPC: r" << std::dec << rdest << " <- imm=0x" << std::hex << immsrc);
      if (rdest) iregs[rdest] = ((immsrc << 12) & 0xfffff000) + (PC_ - 4);
      break;
    case JAL_INST:
      D(3, "JAL: r" << std::dec << rdest << " <- imm=0x" << std::hex << immsrc);
      trace_inst->stall_warp = true;
      if (!updatePC) {
        nextPC = (PC_ - 4) + immsrc;
        //std::cout << "JAL... SETTING PC: " << nextPC << "\n";      
      }
      if (rdest) iregs[rdest] = PC_;
      updatePC = true;
      break;
    case JALR_INST:
      D(3, "JALR: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=0x" << std::hex << immsrc);
      trace_inst->stall_warp = true;
      if (!updatePC) {
        nextPC = iregs[rsrc0] + immsrc;
        //std::cout << "JALR... SETTING PC: " << nextPC << "\n";
      }
      if (rdest) iregs[rdest] = PC_;
      updatePC = true;
      break;
    case SYS_INST: {
      D(3, "SYS_INST: r" << std::dec << rdest << " <- r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", imm=0x" << std::hex << immsrc);
      Word rs1 = iregs[rsrc0];
      Word csr_addr = immsrc & 0x00000FFF;
      switch (func3) {
      case 0:
        if (csr_addr < 2) {
          // ECALL/EBREAK
          tmask_.reset();
          active_ = tmask_.any();
        }
        break;
      case 1:
        // CSRRW
        if (rdest) iregs[rdest] = core_->get_csr(csr_addr, t, id_);
        core_->set_csr(csr_addr, rs1);
        break;
      case 2:
        // CSRRS
        if (rdest) iregs[rdest] = core_->get_csr(csr_addr, t, id_);
        core_->set_csr(csr_addr, rs1 | core_->get_csr(csr_addr, t, id_));
        break;
      case 3:
        // CSRRC
        if (rdest) iregs[rdest] = core_->get_csr(csr_addr, t, id_);
        core_->set_csr(csr_addr, rs1 & ~core_->get_csr(csr_addr, t, id_));
        break;
      case 5:
        // CSRRWI
        if (rdest) iregs[rdest] = core_->get_csr(csr_addr, t, id_);
        core_->set_csr(csr_addr, rsrc0);
        break;
      case 6:
        // CSRRSI
        if (rdest) iregs[rdest] = core_->get_csr(csr_addr, t, id_);
        core_->set_csr(csr_addr, rsrc0 | core_->get_csr(csr_addr, t, id_));
        break;
      case 7:
        // CSRRCI
        if (rdest) iregs[rdest] = core_->get_csr(csr_addr, t, id_);
        core_->set_csr(csr_addr, rsrc0 & ~core_->get_csr(csr_addr, t, id_));
        break;
      default:
        break;
      }
    } break;
    case FENCE:
      D(3, "FENCE");
      break;
    case PJ_INST:
      D(3, "PJ_INST: r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
      if (iregs[rsrc0]) {
        if (!updatePC)
          nextPC = iregs[rsrc1];
        updatePC = true;
      }
      break;
    case GPGPU:
      switch (func3) {
      case 0: {
        // TMC
        D(3, "TMC: r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0]);
        trace_inst->stall_warp = true;
        int active_threads = std::min<int>(iregs[rsrc0], core_->arch().num_threads());          
        tmask_.reset();
        for (int i = 0; i < active_threads; ++i) {
          tmask_[i] = true;
        }
        active_ = tmask_.any();
        runOnce = true;
      } break;
      case 1: {
        // WSPAWN
        D(3, "WSPAWN: r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
        trace_inst->wspawn = true;        
        int active_warps = std::min<int>(iregs[rsrc0], core_->arch().num_warps());
        D(0, "Spawning " << (active_warps-1) << " warps at PC: " << std::hex << iregs[rsrc1]);

        for (int i = 1; i < active_warps; ++i) {
          Warp &newWarp = core_->warp(i);
          newWarp.setPC(iregs[rsrc1]);
          newWarp.setTmask(0, true);
        }
        runOnce = true;
      } break;
      case 2: {
        // SPLIT
        D(3, "SPLIT: r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0]);
        trace_inst->stall_warp = true;
        if (checkUnanimous(rsrc0, iRegFile_, tmask_)) {
          D(3, "Unanimous pred: " << rsrc0 << "  val: " << iregs[rsrc0] << "\n");
          DomStackEntry e(tmask_);
          e.unanimous = true;
          domStack_.push(e);
          break;
        }

        D(3, "Split: Original TM: ");
        DX( for (int i = 0; i < core_->arch().num_threads(); ++i) D(3, tmask_[i] << " "); )

        ThreadMask tmask;
        for (int i = 0; i < core_->arch().num_threads(); ++i) {
          tmask[i] = tmask_[i] && !iRegFile_[i][rsrc0];
        }

        DomStackEntry e(tmask, PC_);
        domStack_.push(tmask_);
        domStack_.push(e);
        for (unsigned i = 0; i < e.tmask.size(); ++i) {
          tmask_[i] = !e.tmask[i] && tmask_[i];
        }
        active_ = tmask_.any();

        D(3, "Split: New TM");
        DX( for (int i = 0; i < core_->arch().num_threads(); ++i) D(3, tmask_[i] << " "); )
        D(3, "Split: Pushed TM PC: " << std::hex << e.PC << std::dec << "\n");
        DX( for (int i = 0; i < core_->arch().num_threads(); ++i) D(3, e.tmask[i] << " "); )

        runOnce = true;
      } break;
      case 3: {
        // JOIN
        D(3, "JOIN");
        if (!domStack_.empty() && domStack_.top().unanimous) {
          D(2, "Uni branch at join");
          printf("NEW DOMESTACK: \n");
          tmask_ = domStack_.top().tmask;
          active_ = tmask_.any();
          domStack_.pop();
          break;
        }

        if (!domStack_.top().fallThrough) {
          if (!updatePC) {
            nextPC = domStack_.top().PC;
            D(3, "join: NOT FALLTHROUGH PC: " << std::hex << nextPC << std::dec);
          }
          updatePC = true;
        }

        D(3, "Join: Old TM: ");
        DX( for (int i = 0; i < core_->arch().num_threads(); ++i) D(3, tmask_[i] << " "); )
        std::cout << "\n";
        tmask_ = domStack_.top().tmask;
        active_ = tmask_.any();

        D(3, "Join: New TM: ");
        DX( for (int i = 0; i < core_->arch().num_threads(); ++i) D(3, tmask_[i] << " "); )

        domStack_.pop();
        runOnce = true;
      } break;
      case 4: {
        // BAR
        D(3, "BAR: r" << std::dec << rsrc0 << "=0x" << std::hex << iregs[rsrc0] << ", r" << std::dec << rsrc1 << "=0x" << std::hex << iregs[rsrc1]);
        active_ = false;
        core_->barrier(iregs[rsrc0], iregs[rsrc1], id_);
        trace_inst->stall_warp = true; 
        runOnce = true;       
      } break;
      default:
        std::cout << "ERROR: UNSUPPORTED GPGPU INSTRUCTION " << instr << "\n";
      }
      break;
    case VSET_ARITH: {
      D(3, "VSET_ARITH");
      int VLEN = core_->arch().vsize() * 8;
      int VLMAX = (instr.getVlmul() * VLEN) / instr.getVsew();
      switch (func3) {
      case 0: // vector-vector
        switch (func6) {
        case 0: {
          D(4, "Addition " << rsrc0 << " " << rsrc1 << " Dest:" << rdest);
          auto& vr1 = vRegFile_[rsrc0];
          auto& vr2 = vRegFile_[rsrc1];
          auto& vd = vRegFile_[rdest];
          auto& mask = vRegFile_[0];

          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t emask = *(uint8_t *)(mask.data() + i);
              uint8_t value = emask & 0x1;
              if (vmask || (!vmask && value)) {
                uint8_t first = *(uint8_t *)(vr1.data() + i);
                uint8_t second = *(uint8_t *)(vr2.data() + i);
                uint8_t result = first + second;
                D(4, "Adding " << first << " + " << second << " = " << result);
                *(uint8_t *)(vd.data() + i) = result;
              }
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t emask = *(uint16_t *)(mask.data() + i);
              uint16_t value = emask & 0x1;
              if (vmask || (!vmask && value)) {
                uint16_t first = *(uint16_t *)(vr1.data() + i);
                uint16_t second = *(uint16_t *)(vr2.data() + i);
                uint16_t result = first + second;
                D(4, "Adding " << first << " + " << second << " = " << result);
                *(uint16_t *)(vd.data() + i) = result;
              }
            }
          } else if (vtype_.vsew == 32) {
            D(4, "Doing 32 bit vector addition");
            for (int i = 0; i < vl_; i++) {
              uint32_t emask = *(uint32_t *)(mask.data() + i);
              uint32_t value = emask & 0x1;
              if (vmask || (!vmask && value)) {
                uint32_t first = *(uint32_t *)(vr1.data() + i);
                uint32_t second = *(uint32_t *)(vr2.data() + i);
                uint32_t result = first + second;
                D(4, "Adding " << first << " + " << second << " = " << result);
                *(uint32_t *)(vd.data() + i) = result;
              }
            }
          }                
        } break;
        case 24: //vmseq
        {
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first == second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first == second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first == second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
          }

        } break;
        case 25: //vmsne
        {
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first != second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first != second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first != second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
          }

        } break;
        case 26: //vmsltu
        {
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first < second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first < second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first < second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
          }

        } break;
        case 27: //vmslt
        {
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              int8_t first = *(int8_t *)(vr1.data() + i);
              int8_t second = *(int8_t *)(vr2.data() + i);
              int8_t result = (first < second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t first = *(int16_t *)(vr1.data() + i);
              int16_t second = *(int16_t *)(vr2.data() + i);
              int16_t result = (first < second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(int16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t first = *(int32_t *)(vr1.data() + i);
              int32_t second = *(int32_t *)(vr2.data() + i);
              int32_t result = (first < second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(int32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 28: //vmsleu
        {
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first <= second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first <= second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first <= second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 29: //vmsle
        {
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              int8_t first = *(int8_t *)(vr1.data() + i);
              int8_t second = *(int8_t *)(vr2.data() + i);
              int8_t result = (first <= second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t first = *(int16_t *)(vr1.data() + i);
              int16_t second = *(int16_t *)(vr2.data() + i);
              int16_t result = (first <= second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(int16_t *)(vd.data() + i) = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t first = *(int32_t *)(vr1.data() + i);
              int32_t second = *(int32_t *)(vr2.data() + i);
              int32_t result = (first <= second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(int32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 30: //vmsgtu
        {
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first > second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first > second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first > second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 31: //vmsgt
        {
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              int8_t first = *(int8_t *)(vr1.data() + i);
              int8_t second = *(int8_t *)(vr2.data() + i);
              int8_t result = (first > second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t first = *(int16_t *)(vr1.data() + i);
              int16_t second = *(int16_t *)(vr2.data() + i);
              int16_t result = (first > second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(int16_t *)(vd.data() + i) = result;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t first = *(int32_t *)(vr1.data() + i);
              int32_t second = *(int32_t *)(vr2.data() + i);
              int32_t result = (first > second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(int32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        }
        break;
      case 2: {
        switch (func6) {
        case 24: //vmandnot
        {
          D(3, "vmandnot");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value = (first & 0x1);
              uint8_t second_value = (second & 0x1);
              uint8_t result = (first_value & !second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }            
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value = (first & 0x1);
              uint16_t second_value = (second & 0x1);
              uint16_t result = (first_value & !second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value = (first & 0x1);
              uint32_t second_value = (second & 0x1);
              uint32_t result = (first_value & !second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 25: //vmand
        {
          D(3, "vmand");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value = (first & 0x1);
              uint8_t second_value = (second & 0x1);
              uint8_t result = (first_value & second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value = (first & 0x1);
              uint16_t second_value = (second & 0x1);
              uint16_t result = (first_value & second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value = (first & 0x1);
              uint32_t second_value = (second & 0x1);
              uint32_t result = (first_value & second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 26: //vmor
        {
          D(3, "vmor");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value = (first & 0x1);
              uint8_t second_value = (second & 0x1);
              uint8_t result = (first_value | second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value = (first & 0x1);
              uint16_t second_value = (second & 0x1);
              uint16_t result = (first_value | second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value = (first & 0x1);
              uint32_t second_value = (second & 0x1);
              uint32_t result = (first_value | second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 27: //vmxor
        {
          D(3, "vmxor");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value = (first & 0x1);
              uint8_t second_value = (second & 0x1);
              uint8_t result = (first_value ^ second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value = (first & 0x1);
              uint16_t second_value = (second & 0x1);
              uint16_t result = (first_value ^ second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value = (first & 0x1);
              uint32_t second_value = (second & 0x1);
              uint32_t result = (first_value ^ second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 28: //vmornot
        {
          D(3, "vmornot");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value = (first & 0x1);
              uint8_t second_value = (second & 0x1);
              uint8_t result = (first_value | !second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value = (first & 0x1);
              uint16_t second_value = (second & 0x1);
              uint16_t result = (first_value | !second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value = (first & 0x1);
              uint32_t second_value = (second & 0x1);
              uint32_t result = (first_value | !second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 29: //vmnand
        {
          D(3, "vmnand");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value = (first & 0x1);
              uint8_t second_value = (second & 0x1);
              uint8_t result = !(first_value & second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }

          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value = (first & 0x1);
              uint16_t second_value = (second & 0x1);
              uint16_t result = !(first_value & second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value = (first & 0x1);
              uint32_t second_value = (second & 0x1);
              uint32_t result = !(first_value & second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 30: //vmnor
        {
          D(3, "vmnor");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value = (first & 0x1);
              uint8_t second_value = (second & 0x1);
              uint8_t result = !(first_value | second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value = (first & 0x1);
              uint16_t second_value = (second & 0x1);
              uint16_t result = !(first_value | second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }

          } else if (vtype_.vsew == 32) {

            for (int i = 0; i < vl_; i++) {
              uint32_t first = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value = (first & 0x1);
              uint32_t second_value = (second & 0x1);
              uint32_t result = !(first_value | second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 31: //vmxnor
        {
          D(3, "vmxnor");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value = (first & 0x1);
              uint8_t second_value = (second & 0x1);
              uint8_t result = !(first_value ^ second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value = (first & 0x1);
              uint16_t second_value = (second & 0x1);
              uint16_t result = !(first_value ^ second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value = (first & 0x1);
              uint32_t second_value = (second & 0x1);
              uint32_t result = !(first_value ^ second_value);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 37: //vmul
        {
          D(3, "vmul");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first * second);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first * second);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first * second);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 45: //vmacc
        {
          D(3, "vmacc");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first * second);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) += result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first * second);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) += result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first * second);
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) += result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        }
      } break;
      case 6: {
        switch (func6) {
        case 0: {
          D(3, "vmadd.vx");
          //vector<Word> & vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              //uint8_t first = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (iregs[rsrc0] + second);
              D(4, "Comparing " << iregs[rsrc0] << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              //uint16_t first = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (iregs[rsrc0] + second);
              D(4, "Comparing " << iregs[rsrc0] << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              //uint32_t first = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (iregs[rsrc0] + second);
              D(4, "Comparing " << iregs[rsrc0] << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 37: //vmul.vx
        {
          D(3, "vmul.vx");
          //vector<Word> & vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              //uint8_t first = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (iregs[rsrc0] * second);
              D(4, "Comparing " << iregs[rsrc0] << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              //uint16_t first = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (iregs[rsrc0] * second);
              D(4, "Comparing " << iregs[rsrc0] << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }

          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              //uint32_t first = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (iregs[rsrc0] * second);
              D(4, "Comparing " << iregs[rsrc0] << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
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
        } else if (s0 >= (2 * VLMAX)) {
          vl_ = VLMAX;
        }        
        if (rdest) iregs[rdest] = vl_;
      } break;
      default: {
        std::abort();
      }
      }
    } break;
    case (FL | VL):
      ++loads_;
      if ( func3==0x2 ) {
        //std::cout << "FL_INST\n"; 
        // rs1 is integer is register!
        Word memAddr = ((iregs[rsrc0] + immsrc) & 0xFFFFFFFC); // alignment
        D(9,"something weird happen!");
        Word data_read = core_->dcache_read(memAddr, 0);
        D(3, "Memaddr");
        DPN(3, ' ' << std::setw(8) << std::hex << memAddr << std::endl);
        trace_inst->is_lw = true;
        trace_inst->mem_addresses[t] = memAddr;
        // //std::cout <<std::hex<< "EXECUTE: " << fregs[rsrc0] << " + " << immsrc << " = " << memAddr <<  " -> data_read: " << data_read << "\n";
        switch (func3) {
        case 2: // FLW
          fregs[rdest] = data_read & 0xFFFFFFFF;
          D(4, "fpReg[rd] " << std::setw(8) << std::hex << fregs[rdest] << std::endl);
          break;
        default:
          std::cout << "ERROR: UNSUPPORTED FL INST\n";
          exit(1);
        }
        D(3, "LOAD MEM ADDRESS: " << std::hex << memAddr);
      } else {  
        D(3, "Executing vector load");      
        D(4, "lmul: " << vtype_.vlmul << " VLEN:" << (core_->arch().vsize() * 8) << "sew: " << vtype_.vsew);
        D(4, "src: " << rsrc0 << " " << iregs[rsrc0]);
        D(4, "dest" << rdest);
        D(4, "width" << instr.getVlsWidth());

        auto &vd = vRegFile_[rdest];

        switch (instr.getVlsWidth()) {
        case 6: { //load word and unit strided (not checking for unit stride)
          for (int i = 0; i < vl_; i++) {
            Word memAddr = ((iregs[rsrc0]) & 0xFFFFFFFC) + (i * vtype_.vsew / 8);
            Word data_read = core_->dcache_read(memAddr, 0);
            D(4, "Mem addr: " << std::hex << memAddr << " Data read " << data_read);
            int *result_ptr = (int *)(vd.data() + i);
            *result_ptr = data_read;

            trace_inst->is_lw = true;
            trace_inst->mem_addresses[i] = memAddr;
            
            D(3, "STORE MEM ADDRESS: " << std::hex << memAddr);
          }
          // cout << "Finished loop" << std::endl;
        } break;
        default:
          std::abort();
        }
        break;
      } 
      break;
    case (FS | VS):
      ++stores_;
      if ((func3 == 0x1) || (func3 == 0x2) 
       || (func3 == 0x3) || (func3 == 0x4)) {
        //std::cout << "FS_INST\n";
        // base is integer register!
        Word memAddr = iregs[rsrc0] + immsrc;
        trace_inst->is_sw = true;
        trace_inst->mem_addresses[t] = memAddr;

        switch (func3) {
        case 1:
          std::cout << "ERROR: UNSUPPORTED FS INST\n";
          std::cout << "FSH\n";
          exit(1);
          // c.core->mem.write(memAddr, fregs[rsrc1], c.supervisorMode, 2);
          break;
        case 2:
          // //std::cout << std::hex << "FSW: about to write: " << fregs[rsrc1] << " to " << memAddr << "\n"; 
          core_->dcache_write(memAddr, fregs[rsrc1], 0, 4);
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
      } else {
        for (int i = 0; i < vl_; i++) {
          // cout << "iter" << std::endl;
          Word memAddr = iregs[rsrc0] + (i * vtype_.vsew / 8);
          // std::cout << "STORE MEM ADDRESS *** : " << std::hex << memAddr << "\n";

          trace_inst->is_sw = true;
          trace_inst->mem_addresses[i] = memAddr;

          switch (instr.getVlsWidth()) {
          case 6: //store word and unit strided (not checking for unit stride)
          {
            uint32_t value = *(uint32_t *)(vRegFile_[instr.getVs3()].data() + i);
            core_->dcache_write(memAddr, value, 0, 4);
            D(4, "store: " << memAddr << " value:" << value);
          } break;
          default:
            std::abort();
          }          
          D(3, "STORE MEM ADDRESS: " << std::hex << memAddr);
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
              core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x10); // set NV bit                
              core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x10); // set NV bit
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
              core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x1); // set NX bit
              core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x1); // set NX bit
            }
            
            if (fetestexcept(FE_UNDERFLOW)) {
              core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x2); // set UF bit
              core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x2); // set UF bit
            }

            if (fetestexcept(FE_OVERFLOW)) {
              core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x4); // set OF bit
              core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x4); // set OF bit
            }

            if (fetestexcept(FE_DIVBYZERO)) {
              core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x8); // set DZ bit
              core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x8); // set DZ bit
            } 

            if (fetestexcept(FE_INVALID)) {
              core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x10); // set NX bit
              core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x10); // set NX bit
            }

            D(4, "fpOut: " << fpOut);
            if (fpBinIsNan(floatToBin(fpOut)) == 0) {
              fregs[rdest] = floatToBin(fpOut);
            } else  {              
              // According to risc-v spec p.64 section 11.3
              // If the result is NaN, it is the canonical NaN
              fregs[rdest] = 0x7fc00000;
            }          
          }
        } break;

        // FSGNJ.S, FSGNJN.S FSGNJX.S
        case 0x10: {
          bool     fsign1 = fregs[rsrc0] & 0x80000000;
          uint32_t fdata1 = fregs[rsrc0] & 0x7FFFFFFF;
          bool     fsign2 = fregs[rsrc1] & 0x80000000;

          switch (func3) {            
          case 0: // FSGNJ.S
            fregs[rdest] = (fsign2 << 31) | fdata1;
            break;          
          case 1: // FSGNJN.S
            fsign2 = !fsign2;
            fregs[rdest] = (fsign2 << 31) | fdata1;
            break;          
          case 2: { // FSGNJX.S
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
              core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x10); // set NV bit
              core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x10); // set NV bit
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
                fregs[rdest] = (sr1IsZero==2) ? fregs[rsrc1] : fregs[rsrc0];
              } else {
                // FMIM.S
                fregs[rdest] = (sr1IsZero==2) ? fregs[rsrc0] : fregs[rsrc1];
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

          //show_fe_exceptions();
          
          // fcsr defined in riscv
          if (fetestexcept(FE_INEXACT)) {
            core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x1); // set NX bit
            core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x1); // set NX bit
          }
          
          if (fetestexcept(FE_UNDERFLOW)) {
            core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x2); // set UF bit
            core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x2); // set UF bit
          }

          if (fetestexcept(FE_OVERFLOW)) {
            core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x4); // set OF bit
            core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x4); // set OF bit
          }

          if (fetestexcept(FE_DIVBYZERO)) {
            core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x8); // set DZ bit
            core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x8); // set DZ bit
          } 

          if (fetestexcept(FE_INVALID) || outOfRange) {
            core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x10); // set NV bit
            core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x10); // set NV bit
          }

          if (rdest) iregs[rdest] = result;
        } break;
        
        // FMV.X.W FCLASS.S
        case 0x70: {
          // FCLASS.S
          if (func3) {
            // Examine the value in fpReg rs1 and write to integer rd
            // a 10-bit mask to indicate the class of the fp number
            if (rdest) iregs[rdest] = 0; // clear all bits

            bool fsign = fregs[rsrc0] & 0x80000000;
            uint32_t expo = (fregs[rsrc0]>>23) & 0x000000FF;
            uint32_t fraction = fregs[rsrc0] & 0x007FFFFF;

            if ((expo==0) && (fraction==0)) {
             if (rdest) iregs[rdest] = fsign? (1<<3) : (1<<4); // +/- 0
            } else if ((expo==0) && (fraction!=0)) {
              if (rdest) iregs[rdest] = fsign? (1<<2) : (1<<5); // +/- subnormal
            } else if ((expo==0xFF) && (fraction==0)) {
              if (rdest) iregs[rdest] = fsign? (1<<0) : (1<<7); // +/- infinity
            } else if ((expo==0xFF) && (fraction!=0)) { 
              if (!fsign && (fraction == 0x00400000)) {
                if (rdest) iregs[rdest] = (1<<9);               // quiet NaN
              } else { 
                if (rdest) iregs[rdest] = (1<<8);               // signaling NaN
              }
            } else {
              if (rdest) iregs[rdest] = fsign? (1<<1) : (1<<6); // +/- normal
            }
          } else {          
            // FMV.X.W
            // Move bit values from floating-point register rs1 to integer register rd
            // Since we are using integer register to represent floating point register, 
            // just simply assign here.
            if (rdest) iregs[rdest] = fregs[rsrc0];
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
              core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x10); // set NV bit
              core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x10); // set NV bit
            } else { // FEQ.S
              // Only set NV bit if it is signaling NaN
              if (fpBinIsNan(fregs[rsrc0]) == 2 || fpBinIsNan(fregs[rsrc1]) == 2) {
                // If either input is NaN, set NV bit
                core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x10); // set NV bit
                core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x10); // set NV bit
              }
            }
            // The result is 0 if either operand is NaN
            if (rdest) iregs[rdest] = 0;
          } else {
            switch(func3) {              
              case 0: {
                // FLE.S
                if (intregToFloat(fregs[rsrc0]) <= intregToFloat(fregs[rsrc1])) {
                  if (rdest) iregs[rdest] = 1;
                } else {
                  if (rdest) iregs[rdest] = 0;
                }
              } break;
              
              case 1: {
                // FLT.S
                if (intregToFloat(fregs[rsrc0]) < intregToFloat(fregs[rsrc1])) {
                  if (rdest) iregs[rdest] = 1;
                } else {
                  if (rdest) iregs[rdest] = 0;
                }
              }
              break;
              
              case 2: {
                // FEQ.S
                if (intregToFloat(fregs[rsrc0]) == intregToFloat(fregs[rsrc1])) {
                  if (rdest) iregs[rdest] = 1;
                } else {
                  if (rdest) iregs[rdest] = 0;
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
            data = (int)iregs[rsrc0];
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
        core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x10); // set NV bit
        core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x10); // set NV bit
      }

      if (fpBinIsNan(fregs[rsrc0]) || fpBinIsNan(fregs[rsrc1]) || fpBinIsNan(fregs[rsrc2])) { 
        // if one of op is NaN
        // if addend is not quiet NaN, them set FCSR
        if ((fpBinIsNan(fregs[rsrc0])==2) | (fpBinIsNan(fregs[rsrc1])==2) | (fpBinIsNan(fregs[rsrc1])==2)) {
          core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x10); // set NV bit
          core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x10); // set NV bit          
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
          core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x1); // set NX bit
          core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x1); // set NX bit
        }
        
        if (fetestexcept(FE_UNDERFLOW)) {
          core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x2); // set UF bit
          core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x2); // set UF bit
        }

        if (fetestexcept(FE_OVERFLOW)) {
          core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x4); // set OF bit
          core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x4); // set OF bit
        }

        if (fetestexcept(FE_DIVBYZERO)) {
          core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x8); // set DZ bit
          core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x8); // set DZ bit
        } 

        if (fetestexcept(FE_INVALID)) {
          core_->set_csr(0x003, core_->get_csr(0x003, t, id_) | 0x10); // set NV bit
          core_->set_csr(0x001, core_->get_csr(0x001, t, id_) | 0x10); // set NV bit
        }

        fregs[rdest] = floatToBin(fpOut);
      }
    }
    break;
    default:
      D(3, "PC: " << std::hex << (PC_ - 4));
      D(3, "ERROR: Unsupported instruction: " << instr);
      std::abort();
    }

    if (instr.hasRDest()) {
      if (instr.is_FpDest()) {
        D(3, "r" << std::dec << rdest << "=0x" << std::hex << std::hex << fregs[rdest]);
      } else {
        D(3, "r" << std::dec << rdest << "=0x" << std::hex << std::hex << iregs[rdest]);
      }
    }
  }

  if (updatePC) {
    PC_ = nextPC;
    D(3, "Next PC: " << std::hex << nextPC << std::dec);
  }
}
