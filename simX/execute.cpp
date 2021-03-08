#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <bitset>
#include <climits>
#include <sys/types.h>
#include <sys/stat.h>
#include <cfenv>
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
  size_t i;
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

void Warp::execute(const Instr &instr, Pipeline *pipeline) {
  assert(tmask_.any());

  Word nextPC = PC_ + core_->arch().wsize();
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
  Word vmask = instr.getVmask();

  int num_threads = core_->arch().num_threads();
  for (int t = 0; t < num_threads; t++) {
    if (!tmask_.test(t) || runOnce)
      continue;
    
    auto &iregs = iRegFile_.at(t);
    auto &fregs = fRegFile_.at(t);

    Word rsdata[3];
    Word iresult;
    Word fresult;

    int num_rsrcs = instr.getNRSrc();
    if (num_rsrcs) {    
      DPH(3, "[" << std::dec << t << "] Src Registers: ");
      for (int i = 0; i < num_rsrcs; ++i) {    
        int rst = instr.getRSType(i);
        int rs = instr.getRSrc(i);        
        if (i) DPN(3, ", ");
        switch (rst) {
        case 1: 
          rsdata[i] = iregs[rs];
          DPN(3, "r" << std::dec << rs << "=0x" << std::hex << rsdata[i]); 
          break;
        case 2: 
          rsdata[i] = fregs[rs];
          DPN(3, "fr" << std::dec << rs << "=0x" << std::hex << rsdata[i]); 
          break;
        default: break;
        }
      }
      DPN(3, std::endl);
    }
  
    switch (opcode) {
    case NOP:
      break;
    case LUI_INST:
      iresult = (immsrc << 12) & 0xfffff000;
      break;
    case AUIPC_INST:
      iresult = ((immsrc << 12) & 0xfffff000) + PC_;
      break;
    case R_INST: {
      if (func7 & 0x1) {
        switch (func3) {
        case 0:
          // MUL
          iresult = ((WordI)iregs[rsrc0]) * ((WordI)iregs[rsrc1]);
          break;
        case 1: {
          // MULH
          int64_t first = (int64_t)iregs[rsrc0];
          if (iregs[rsrc0] & 0x80000000) {
            first = first | 0xFFFFFFFF00000000;
          }
          int64_t second = (int64_t)iregs[rsrc1];
          if (iregs[rsrc1] & 0x80000000) {
            second = second | 0xFFFFFFFF00000000;
          }
          uint64_t result = first * second;
          iresult = (result >> 32) & 0xFFFFFFFF;
        } break;
        case 2: {
          // MULHSU          
          int64_t first = (int64_t)iregs[rsrc0];
          if (iregs[rsrc0] & 0x80000000) {
            first = first | 0xFFFFFFFF00000000;
          }
          int64_t second = (int64_t)iregs[rsrc1];
          iresult = ((first * second) >> 32) & 0xFFFFFFFF;
        } break;
        case 3: {
          // MULHU
          uint64_t first = (uint64_t)iregs[rsrc0];
          uint64_t second = (uint64_t)iregs[rsrc1];
          iresult = ((first * second) >> 32) & 0xFFFFFFFF;
        } break;
        case 4: {
          // DIV
          WordI dividen = iregs[rsrc0];
          WordI divisor = iregs[rsrc1];
          if (divisor == 0) {
            iresult = -1;
          } else if (dividen == WordI(0x80000000) && divisor == WordI(0xffffffff)) {
            iresult = dividen;
          } else {
            iresult = dividen / divisor;
          }
        } break;
        case 5: {
          // DIVU
          Word dividen = iregs[rsrc0];
          Word divisor = iregs[rsrc1];
          if (divisor == 0) {
            iresult = -1;
          } else {
            iresult = dividen / divisor;
          }
        } break;
        case 6: {
          // REM
          WordI dividen = iregs[rsrc0];
          WordI divisor = iregs[rsrc1];
          if (iregs[rsrc1] == 0) {
            iresult = dividen;
          } else if (dividen == WordI(0x80000000) && divisor == WordI(0xffffffff)) {
            iresult = 0;
          } else {
            iresult = dividen % divisor;
          }
        } break;
        case 7: {
          // REMU
          Word dividen = iregs[rsrc0];
          Word divisor = iregs[rsrc1];
          if (iregs[rsrc1] == 0) {
            iresult = dividen;
          } else {
            iresult = dividen % divisor;
          }
        } break;
        default:
          std::cout << "unsupported MUL/DIV instr\n";
          std::abort();
        }
      } else {
        switch (func3) {
        case 0:
          if (func7) {
            iresult = iregs[rsrc0] - iregs[rsrc1];
          } else {
            iresult = iregs[rsrc0] + iregs[rsrc1];
          }
          break;
        case 1:
          iresult = iregs[rsrc0] << iregs[rsrc1];
          break;
        case 2:
          if (WordI(iregs[rsrc0]) < WordI(iregs[rsrc1])) {
            iresult = 1;
          } else {
            iresult = 0;
          }
          break;
        case 3:
          if (Word(iregs[rsrc0]) < Word(iregs[rsrc1])) {
            iresult = 1;
          } else {
            iresult = 0;
          }
          break;
        case 4:
          iresult = iregs[rsrc0] ^ iregs[rsrc1];
          break;
        case 5:
          if (func7) {
            iresult = WordI(iregs[rsrc0]) >> WordI(iregs[rsrc1]);
          } else {
            iresult = Word(iregs[rsrc0]) >> Word(iregs[rsrc1]);
          }
          break;
        case 6:
          iresult = iregs[rsrc0] | iregs[rsrc1];
          break;
        case 7:
          iresult = iregs[rsrc0] & iregs[rsrc1];
          break;
        default:
          std::abort();
        }
      }
    } break;
    case I_INST:
      switch (func3) {
      case 0:
        // ADDI
        iresult = iregs[rsrc0] + immsrc;
        break;
      case 1:
        // SLLI
        iresult = iregs[rsrc0] << immsrc;
        break;
      case 2:
        // SLTI
        if (WordI(iregs[rsrc0]) < WordI(immsrc)) {
          iresult = 1;
        } else {
          iresult = 0;
        }
        break;
      case 3: {
        // SLTIU
        if (Word(iregs[rsrc0]) < Word(immsrc)) {
          iresult = 1;
        } else {
          iresult = 0;
        }
      } break;
      case 4:
        // XORI
        iresult = iregs[rsrc0] ^ immsrc;
        break;
      case 5:
        if (func7) {
          // SRAI
          Word result = WordI(iregs[rsrc0]) >> immsrc;
          iresult = result;
        } else {
          // SRLI
          Word result = Word(iregs[rsrc0]) >> immsrc;
          iresult = result;
        }
        break;
      case 6:
        // ORI
        iresult = iregs[rsrc0] | immsrc;
        break;
      case 7:
        // ANDI
        iresult = iregs[rsrc0] & immsrc;
        break;
      default:
        std::abort();
      }
      break;
    case B_INST:
      switch (func3) {
      case 0:
        // BEQ
        if (iregs[rsrc0] == iregs[rsrc1]) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 1:
        // BNE
        if (iregs[rsrc0] != iregs[rsrc1]) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 4:
        // BLT
        if (WordI(iregs[rsrc0]) < WordI(iregs[rsrc1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 5:
        // BGE
        if (WordI(iregs[rsrc0]) >= WordI(iregs[rsrc1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 6:
        // BLTU
        if (Word(iregs[rsrc0]) < Word(iregs[rsrc1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 7:
        // BGEU
        if (Word(iregs[rsrc0]) >= Word(iregs[rsrc1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      }
      pipeline->stall_warp = true;
      runOnce = true;
      break;
    case JAL_INST:
      iresult = nextPC;
      nextPC = PC_ + immsrc;  
      pipeline->stall_warp = true;
      runOnce = true;
      break;
    case JALR_INST:
      iresult = nextPC;
      nextPC = iregs[rsrc0] + immsrc;
      pipeline->stall_warp = true;
      runOnce = true;
      break;
    case L_INST: {
      Word memAddr   = ((iregs[rsrc0] + immsrc) & 0xFFFFFFFC); // Address word alignment
      Word shift_by  = ((iregs[rsrc0] + immsrc) & 0x00000003) * 8;
      Word data_read = core_->dcache_read(memAddr, 0);
      D(3, "LOAD MEM ADDRESS: " << std::hex << memAddr << ", DATA=0x" << data_read);
      switch (func3) {
      case 0:
        // LBI
        iresult = signExt((data_read >> shift_by) & 0xFF, 8, 0xFF);
        break;
      case 1:
        // LHI
        iresult = signExt((data_read >> shift_by) & 0xFFFF, 16, 0xFFFF);
        break;
      case 2:
        // LW
        iresult = data_read;
        break;
      case 4:
        // LBU
        iresult = Word((data_read >> shift_by) & 0xFF);
        break;
      case 5:
        // LHU
        iresult = Word((data_read >> shift_by) & 0xFFFF);
        break;
      default:
        std::abort();        
      }
    } break;
    case S_INST: {
      Word memAddr = iregs[rsrc0] + immsrc;
      switch (func3) {
      case 0:
        // SB
        core_->dcache_write(memAddr, iregs[rsrc1] & 0x000000FF, 0, 1);
        break;
      case 1:
        // SH
        core_->dcache_write(memAddr, iregs[rsrc1], 0, 2);
        break;
      case 2:
        // SW
        core_->dcache_write(memAddr, iregs[rsrc1], 0, 4);
        break;
      default:
        std::abort();
      }
      D(3, "STORE MEM ADDRESS: " << std::hex << memAddr);
    } break;
    case SYS_INST: {
      Word csr_addr = immsrc & 0x00000FFF;
      Word csr_value = core_->get_csr(csr_addr, t, id_);
      switch (func3) {
      case 0:
        if (csr_addr < 2) {
          // ECALL/EBREAK
          tmask_.reset();
          active_ = tmask_.any();
          pipeline->stall_warp = true; 
        }
        break;
      case 1:
        // CSRRW
        iresult = csr_value;
        core_->set_csr(csr_addr, iregs[rsrc0], t, id_);
        break;
      case 2:
        // CSRRS
        iresult = csr_value;
        core_->set_csr(csr_addr, csr_value | iregs[rsrc0], t, id_);
        break;
      case 3:
        // CSRRC
        iresult = csr_value;
        core_->set_csr(csr_addr, csr_value & ~iregs[rsrc0], t, id_);
        break;
      case 5:
        // CSRRWI
        iresult = csr_value;
        core_->set_csr(csr_addr, rsrc0, t, id_);
        break;
      case 6:
        // CSRRSI
        iresult = csr_value;
        core_->set_csr(csr_addr, csr_value | rsrc0, t, id_);
        break;
      case 7:
        // CSRRCI
        iresult = csr_value;
        core_->set_csr(csr_addr, csr_value & ~rsrc0, t, id_);
        break;
      default:
        break;
      }
    } break;
    case FENCE:
      D(3, "FENCE");
      pipeline->stall_warp = true; 
      runOnce = true;
      break;
    case (FL | VL):
      if (func3 == 0x2) {
        Word memAddr = iregs[rsrc0] + immsrc;
        Word data_read = core_->dcache_read(memAddr, 0);        
        D(3, "LOAD MEM ADDRESS: " << std::hex << memAddr << ", DATA=0x" << data_read);
        fresult = data_read;
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
            
            D(3, "STORE MEM ADDRESS: " << std::hex << memAddr);
          }
        } break;
        default:
          std::abort();
        }
        break;
      } 
      break;
    case (FS | VS):
      if (func3 == 0x2) {
        Word memAddr = iregs[rsrc0] + immsrc;
        core_->dcache_write(memAddr, fregs[rsrc1], 0, 4);
        D(3, "STORE MEM ADDRESS: " << std::hex << memAddr);
      } else {
        for (int i = 0; i < vl_; i++) {
          Word memAddr = iregs[rsrc0] + (i * vtype_.vsew / 8);
          switch (instr.getVlsWidth()) {
          case 6: {
            //store word and unit strided (not checking for unit stride)          
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
          if (fpBinIsNan(fregs[rsrc0]) || fpBinIsNan(fregs[rsrc1])) { 
            // if one of op is NaN, one of them is not quiet NaN, them set FCSR
            if ((fpBinIsNan(fregs[rsrc0])==2) | (fpBinIsNan(fregs[rsrc1])==2)) {
              core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x10, t, id_); // set NV bit                
              core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x10, t, id_); // set NV bit
            }
            if (fpBinIsNan(fregs[rsrc0]) && fpBinIsNan(fregs[rsrc1])) 
              fresult = 0x7fc00000;  // canonical(quiet) NaN 
            else if (fpBinIsNan(fregs[rsrc0]))
              fresult = fregs[rsrc1];
            else
              fresult = fregs[rsrc0];
          } else {
            float fpsrc_0 = intregToFloat(fregs[rsrc0]);
            float fpsrc_1 = intregToFloat(fregs[rsrc1]);
            float fpDest;                          
            
            feclearexcept(FE_ALL_EXCEPT);

            if (func7 == 0x00)    // FADD 
              fpDest = fpsrc_0 + fpsrc_1;
            else if (func7==0x04) // FSUB
              fpDest = fpsrc_0 - fpsrc_1;
            else if (func7==0x08) // FMUL
              fpDest = fpsrc_0 * fpsrc_1;
            else if (func7==0x0c) // FDIV
              fpDest = fpsrc_0 / fpsrc_1;
            else if (func7==0x2c) // FSQRT
              fpDest = sqrt(fpsrc_0);  
            else {
              std::abort();
            }            
            // fcsr defined in riscv
            if (fetestexcept(FE_INEXACT)) {
              core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x1, t, id_); // set NX bit
              core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x1, t, id_); // set NX bit
            }
            
            if (fetestexcept(FE_UNDERFLOW)) {
              core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x2, t, id_); // set UF bit
              core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x2, t, id_); // set UF bit
            }

            if (fetestexcept(FE_OVERFLOW)) {
              core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x4, t, id_); // set OF bit
              core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x4, t, id_); // set OF bit
            }

            if (fetestexcept(FE_DIVBYZERO)) {
              core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x8, t, id_); // set DZ bit
              core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x8, t, id_); // set DZ bit
            } 

            if (fetestexcept(FE_INVALID)) {
              core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x10, t, id_); // set NX bit
              core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x10, t, id_); // set NX bit
            }

            D(4, "fpDest: " << fpDest);
            if (fpBinIsNan(floatToBin(fpDest)) == 0) {
              fresult = floatToBin(fpDest);
            } else  {              
              // According to risc-v spec p.64 section 11.3
              // If the result is NaN, it is the canonical NaN
              fresult = 0x7fc00000;
            }          
          }
        } break;

        // FSGNJ.S, FSGNJN.S, FSGNJX.S
        case 0x10: {
          bool     fsign1 = fregs[rsrc0] & 0x80000000;
          uint32_t fdata1 = fregs[rsrc0] & 0x7FFFFFFF;
          bool     fsign2 = fregs[rsrc1] & 0x80000000;
          switch (func3) {            
          case 0: // FSGNJ.S
            fresult = (fsign2 << 31) | fdata1;
            break;          
          case 1: // FSGNJN.S
            fsign2 = !fsign2;
            fresult = (fsign2 << 31) | fdata1;
            break;          
          case 2: { // FSGNJX.S
            bool sign = fsign1 ^ fsign2;
            fresult = (sign << 31) | fdata1;
            } break;
          }
        } break;

        // FMIN.S, FMAX.S
        case 0x14: {
          if (fpBinIsNan(fregs[rsrc0]) || fpBinIsNan(fregs[rsrc1])) { // if one of src is NaN
            // one of them is not quiet NaN, them set FCSR
            if ((fpBinIsNan(fregs[rsrc0])==2) | (fpBinIsNan(fregs[rsrc1])==2)) {
              core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x10, t, id_); // set NV bit
              core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x10, t, id_); // set NV bit
            }
            if (fpBinIsNan(fregs[rsrc0]) && fpBinIsNan(fregs[rsrc1])) 
              fresult = 0x7fc00000;  // canonical(quiet) NaN 
            else if (fpBinIsNan(fregs[rsrc0]))
              fresult = fregs[rsrc1];
            else
              fresult = fregs[rsrc0];
          } else {
            uint8_t sr0IsZero = fpBinIsZero(fregs[rsrc0]);
            uint8_t sr1IsZero = fpBinIsZero(fregs[rsrc1]);

            if (sr0IsZero && sr1IsZero && (sr0IsZero != sr1IsZero)) { 
              // both are zero and not equal
              // handle corner case that compare +0 and -0              
              if (func3) {
                // FMAX.S
                fresult = (sr1IsZero==2) ? fregs[rsrc1] : fregs[rsrc0];
              } else {
                // FMIM.S
                fresult = (sr1IsZero==2) ? fregs[rsrc0] : fregs[rsrc1];
              }
            } else {
              float rs1 = intregToFloat(fregs[rsrc0]);
              float rs2 = intregToFloat(fregs[rsrc1]);              
              if (func3) {
                // FMAX.S
                float fmax = std::max(rs1, rs2); 
                fresult = floatToBin(fmax);
              } else {
                // FMIN.S
                float fmin = std::min(rs1, rs2);
                fresult = floatToBin(fmin);
              }
            }
          }
        } break;
        
        // FCVT.W.S FCVT.WU.S
        case 0x60: {          
          float fpSrc = intregToFloat(fregs[rsrc0]);
          Word result;
          bool outOfRange = false;          
          if (rsrc1 == 0) { 
            // FCVT.W.S
            // Convert floating point to 32-bit signed integer
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
          
          if (fetestexcept(FE_INEXACT)) {
            core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x1, t, id_); // set NX bit
            core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x1, t, id_); // set NX bit
          }
          
          if (fetestexcept(FE_UNDERFLOW)) {
            core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x2, t, id_); // set UF bit
            core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x2, t, id_); // set UF bit
          }

          if (fetestexcept(FE_OVERFLOW)) {
            core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x4, t, id_); // set OF bit
            core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x4, t, id_); // set OF bit
          }

          if (fetestexcept(FE_DIVBYZERO)) {
            core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x8, t, id_); // set DZ bit
            core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x8, t, id_); // set DZ bit
          } 

          if (fetestexcept(FE_INVALID) || outOfRange) {
            core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x10, t, id_); // set NV bit
            core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x10, t, id_); // set NV bit
          }

          iresult = result;
        } break;
        
        // FMV.X.W FCLASS.S
        case 0x70: {
          // FCLASS.S
          if (func3) {
            // Examine the value in fpReg rs1 and write to integer rd
            // a 10-bit mask to indicate the class of the fp number
            iresult = 0; // clear all bits

            bool fsign = fregs[rsrc0] & 0x80000000;
            uint32_t expo = (fregs[rsrc0]>>23) & 0x000000FF;
            uint32_t fraction = fregs[rsrc0] & 0x007FFFFF;

            if ((expo==0) && (fraction==0)) {
             iresult = fsign ? (1<<3) : (1<<4); // +/- 0
            } else if ((expo==0) && (fraction!=0)) {
              iresult = fsign ? (1<<2) : (1<<5); // +/- subnormal
            } else if ((expo==0xFF) && (fraction==0)) {
              iresult = fsign ? (1<<0) : (1<<7); // +/- infinity
            } else if ((expo==0xFF) && (fraction!=0)) { 
              if (!fsign && (fraction == 0x00400000)) {
                iresult = (1<<9);               // quiet NaN
              } else { 
                iresult = (1<<8);               // signaling NaN
              }
            } else {
              iresult = fsign ? (1<<1) : (1<<6); // +/- normal
            }
          } else {          
            // FMV.X.W
            // Move bit values from floating-point register rs1 to integer register rd
            // Since we are using integer register to represent floating point register, 
            // just simply assign here.
            iresult = fregs[rsrc0];
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
              core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x10, t, id_); // set NV bit
              core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x10, t, id_); // set NV bit
            } else { // FEQ.S
              // Only set NV bit if it is signaling NaN
              if (fpBinIsNan(fregs[rsrc0]) == 2 || fpBinIsNan(fregs[rsrc1]) == 2) {
                // If either input is NaN, set NV bit
                core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x10, t, id_); // set NV bit
                core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x10, t, id_); // set NV bit
              }
            }
            // The result is 0 if either operand is NaN
            iresult = 0;
          } else {
            switch(func3) {              
            case 0: {
              // FLE.S
              iresult = (intregToFloat(fregs[rsrc0]) <= intregToFloat(fregs[rsrc1]));
            } break;              
            case 1: {
              // FLT.S
              iresult = (intregToFloat(fregs[rsrc0]) < intregToFloat(fregs[rsrc1]));
            } break;              
            case 2: {
              // FEQ.S
              iresult = (intregToFloat(fregs[rsrc0]) == intregToFloat(fregs[rsrc1]));
            } break;
            default:
              std::abort();
            }
          }
        } break;
        
        case 0x68:
          // Cast integer to floating point              
          if (rsrc1) {
            // FCVT.S.WU: convert 32-bit unsigned integer to floating point
            float data = iregs[rsrc0];      
            fresult = floatToBin(data);
          } else {
            // FCVT.S.W: convert 32-bit signed integer to floating point
            // iregs[rsrc0] is actually a unsigned number
            float data = (WordI)iregs[rsrc0];
            fresult = floatToBin(data);
          }
        break;

        case 0x78: {
          // FMV.W.X
          // Move bit values from integer register rs1 to floating register rd
          // Since we are using integer register to represent floating point register, 
          // just simply assign here.
          fresult = iregs[rsrc0];
        }
        break;
      }
    break;

    case FMADD:      
    case FMSUB:      
    case FMNMADD:
    case FMNMSUB: {
      // multiplicands are infinity and zero, them set FCSR
      if (fpBinIsZero(fregs[rsrc0]) || fpBinIsZero(fregs[rsrc1]) || fpBinIsInf(fregs[rsrc0]) || fpBinIsInf(fregs[rsrc1])) {
        core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x10, t, id_); // set NV bit
        core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x10, t, id_); // set NV bit
      }
      if (fpBinIsNan(fregs[rsrc0]) || fpBinIsNan(fregs[rsrc1]) || fpBinIsNan(fregs[rsrc2])) { 
        // if one of op is NaN, if addend is not quiet NaN, them set FCSR
        if ((fpBinIsNan(fregs[rsrc0])==2) | (fpBinIsNan(fregs[rsrc1])==2) | (fpBinIsNan(fregs[rsrc1])==2)) {
          core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x10, t, id_); // set NV bit
          core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x10, t, id_); // set NV bit          
        }
        fresult = 0x7fc00000;  // canonical(quiet) NaN 
      } else {
        float rs1 = intregToFloat(fregs[rsrc0]);
        float rs2 = intregToFloat(fregs[rsrc1]);
        float rs3 = intregToFloat(fregs[rsrc2]);
        float fpDest(0.0);
        feclearexcept(FE_ALL_EXCEPT);          
        switch (opcode) {
          case FMADD:    
            // rd = (rs1*rs2)+rs3
            fpDest = (rs1 * rs2) + rs3;  break;
          case FMSUB:      
            // rd = (rs1*rs2)-rs3
            fpDest = (rs1 * rs2) - rs3; break;
          case FMNMADD:
            // rd = -(rs1*rs2)+rs3
            fpDest = -1*(rs1 * rs2) - rs3; break;        
          case FMNMSUB: 
            // rd = -(rs1*rs2)-rs3
            fpDest = -1*(rs1 * rs2) + rs3; break;
          default:
            std::abort();
            break;                 
        }  

        // fcsr defined in riscv
        if (fetestexcept(FE_INEXACT)) {
          core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x1, t, id_); // set NX bit
          core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x1, t, id_); // set NX bit
        }
        
        if (fetestexcept(FE_UNDERFLOW)) {
          core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x2, t, id_); // set UF bit
          core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x2, t, id_); // set UF bit
        }

        if (fetestexcept(FE_OVERFLOW)) {
          core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x4, t, id_); // set OF bit
          core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x4, t, id_); // set OF bit
        }

        if (fetestexcept(FE_DIVBYZERO)) {
          core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x8, t, id_); // set DZ bit
          core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x8, t, id_); // set DZ bit
        } 

        if (fetestexcept(FE_INVALID)) {
          core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x10, t, id_); // set NV bit
          core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x10, t, id_); // set NV bit
        }

        fresult = floatToBin(fpDest);
      }
    }
    break;
    case GPGPU:
      switch (func3) {
      case 0: {
        // TMC
        int active_threads = std::min<int>(iregs[rsrc0], num_threads);          
        tmask_.reset();
        for (int i = 0; i < active_threads; ++i) {
          tmask_[i] = true;
        }
        active_ = tmask_.any();
        pipeline->stall_warp = true;
        runOnce = true;
      } break;
      case 1: {
        // WSPAWN
        int active_warps = std::min<int>(iregs[rsrc0], core_->arch().num_warps());
        D(0, "Spawning " << (active_warps-1) << " warps at PC: " << std::hex << iregs[rsrc1]);
        for (int i = 1; i < active_warps; ++i) {
          Warp &newWarp = core_->warp(i);
          newWarp.setPC(iregs[rsrc1]);
          newWarp.setTmask(0, true);
        }
        pipeline->stall_warp = true;
        runOnce = true;
      } break;
      case 2: {
        // SPLIT    
        if (checkUnanimous(rsrc0, iRegFile_, tmask_)) {
          D(3, "Unanimous pred: " << rsrc0 << "  val: " << iregs[rsrc0] << "\n");
          DomStackEntry e(tmask_);
          e.unanimous = true;
          domStack_.push(e);
        } else {
          D(3, "Split: Original TM: ");
          DX( for (int i = 0; i < num_threads; ++i) D(3, tmask_[i] << " "); )

          ThreadMask tmask;
          for (int i = 0; i < num_threads; ++i) {
            tmask[i] = tmask_[i] && !iRegFile_[i][rsrc0];
          }

          DomStackEntry e(tmask, nextPC);
          domStack_.push(tmask_);
          domStack_.push(e);
          for (size_t i = 0; i < e.tmask.size(); ++i) {
            tmask_[i] = !e.tmask[i] && tmask_[i];
          }
          active_ = tmask_.any();

          D(3, "Split: New TM");
          DX( for (int i = 0; i < num_threads; ++i) D(3, tmask_[i] << " "); )

          D(3, "Split: Pushed TM PC: " << std::hex << e.PC << std::dec << "\n");
          DX( for (int i = 0; i < num_threads; ++i) D(3, e.tmask[i] << " "); )
        }
        pipeline->stall_warp = true;
        runOnce = true;
      } break;
      case 3: {
        // JOIN
        D(3, "JOIN");
        if (!domStack_.empty() && domStack_.top().unanimous) {
          D(2, "Uninimous branch at join");
          tmask_ = domStack_.top().tmask;
          active_ = tmask_.any();
          domStack_.pop();
        } else {
          if (!domStack_.top().fallThrough) {
            nextPC = domStack_.top().PC;
            D(3, "join: NOT FALLTHROUGH PC: " << std::hex << nextPC << std::dec);
          }

          D(3, "Join: Old TM: ");
          DX( for (int i = 0; i < num_threads; ++i) D(3, tmask_[i] << " "); )
          std::cout << "\n";
          tmask_ = domStack_.top().tmask;
          active_ = tmask_.any();

          D(3, "Join: New TM: ");
          DX( for (int i = 0; i < num_threads; ++i) D(3, tmask_[i] << " "); )

          domStack_.pop();
        }
        pipeline->stall_warp = true;
        runOnce = true;
      } break;
      case 4: {
        // BAR
        active_ = false;
        core_->barrier(iregs[rsrc0], iregs[rsrc1], id_);
        pipeline->stall_warp = true; 
        runOnce = true;       
      } break;
      default:
        std::abort();
      }
      break;
    case VSET: {
      D(3, "VSET");
      int VLEN = core_->arch().vsize() * 8;
      int VLMAX = (instr.getVlmul() * VLEN) / instr.getVsew();
      switch (func3) {
      case 0: // vector-vector
        switch (func6) {
        case 0: {
          auto& vr1 = vRegFile_[rsrc0];
          auto& vr2 = vRegFile_[rsrc1];
          auto& vd = vRegFile_[rdest];
          auto& mask = vRegFile_[0];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t emask = *(uint8_t *)(mask.data() + i);
              uint8_t value = emask & 0x1;
              if (vmask || (!vmask && value)) {
                uint8_t first  = *(uint8_t *)(vr1.data() + i);
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
                uint16_t first  = *(uint16_t *)(vr1.data() + i);
                uint16_t second = *(uint16_t *)(vr2.data() + i);
                uint16_t result = first + second;
                D(4, "Adding " << first << " + " << second << " = " << result);
                *(uint16_t *)(vd.data() + i) = result;
              }
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t emask = *(uint32_t *)(mask.data() + i);
              uint32_t value = emask & 0x1;
              if (vmask || (!vmask && value)) {
                uint32_t first  = *(uint32_t *)(vr1.data() + i);
                uint32_t second = *(uint32_t *)(vr2.data() + i);
                uint32_t result = first + second;
                D(4, "Adding " << first << " + " << second << " = " << result);
                *(uint32_t *)(vd.data() + i) = result;
              }
            }
          }                
        } break;
        case 24: {
          //vmseq
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first == second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first == second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first == second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 25: { 
          //vmsne
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first != second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first != second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first != second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 26: {
          //vmsltu
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first < second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first < second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first < second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 27: {
          //vmslt
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              int8_t first  = *(int8_t *)(vr1.data() + i);
              int8_t second = *(int8_t *)(vr2.data() + i);
              int8_t result = (first < second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t first  = *(int16_t *)(vr1.data() + i);
              int16_t second = *(int16_t *)(vr2.data() + i);
              int16_t result = (first < second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(int16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t first  = *(int32_t *)(vr1.data() + i);
              int32_t second = *(int32_t *)(vr2.data() + i);
              int32_t result = (first < second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(int32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 28: {
          //vmsleu
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first <= second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first <= second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first <= second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 29: {
          //vmsle
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              int8_t first  = *(int8_t *)(vr1.data() + i);
              int8_t second = *(int8_t *)(vr2.data() + i);
              int8_t result = (first <= second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t first  = *(int16_t *)(vr1.data() + i);
              int16_t second = *(int16_t *)(vr2.data() + i);
              int16_t result = (first <= second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(int16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t first  = *(int32_t *)(vr1.data() + i);
              int32_t second = *(int32_t *)(vr2.data() + i);
              int32_t result = (first <= second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(int32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 30: {
          //vmsgtu
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first > second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first > second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first > second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
          }
        } break;
        case 31: {
          //vmsgt
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              int8_t first  = *(int8_t *)(vr1.data() + i);
              int8_t second = *(int8_t *)(vr2.data() + i);
              int8_t result = (first > second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t first  = *(int16_t *)(vr1.data() + i);
              int16_t second = *(int16_t *)(vr2.data() + i);
              int16_t result = (first > second) ? 1 : 0;
              D(4, "Comparing " << first << " + " << second << " = " << result);
              *(int16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t first  = *(int32_t *)(vr1.data() + i);
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
        case 24: { 
          // vmandnot
          D(3, "vmandnot");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value  = (first & 0x1);
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
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value  = (first & 0x1);
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
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value  = (first & 0x1);
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
        case 25: {
          // vmand
          D(3, "vmand");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value  = (first & 0x1);
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
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value  = (first & 0x1);
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
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value  = (first & 0x1);
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
        case 26: {
          // vmor
          D(3, "vmor");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value  = (first & 0x1);
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
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value  = (first & 0x1);
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
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value  = (first & 0x1);
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
        case 27: { 
          //vmxor
          D(3, "vmxor");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value  = (first & 0x1);
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
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value  = (first & 0x1);
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
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value  = (first & 0x1);
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
        case 28: {
          //vmornot
          D(3, "vmornot");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value  = (first & 0x1);
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
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value  = (first & 0x1);
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
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value  = (first & 0x1);
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
        case 29: {
          //vmnand
          D(3, "vmnand");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value  = (first & 0x1);
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
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value  = (first & 0x1);
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
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value  = (first & 0x1);
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
        case 30: {
          //vmnor
          D(3, "vmnor");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value  = (first & 0x1);
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
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value  = (first & 0x1);
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
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value  = (first & 0x1);
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
        case 31: {
          //vmxnor
          D(3, "vmxnor");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t first_value  = (first & 0x1);
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
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t first_value  = (first & 0x1);
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
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t first_value  = (first & 0x1);
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
        case 37: {
          //vmul
          D(3, "vmul");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
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
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
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
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
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
        case 45: {
          // vmacc
          D(3, "vmacc");
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
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
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
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
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
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
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
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
        case 37: {
          // vmul.vx
          D(3, "vmul.vx");
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
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
        vtype_.vill = 0;
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
        iresult = vl_;
      } break;
      default:
        std::abort();
      }
    } break;    
    default:
      std::abort();
    }

    int rdt = instr.getRDType();
    switch (rdt) {
    case 1:      
      if (rdest) {
        D(3, "[" << std::dec << t << "] Dest Register: r" << rdest << "=0x" << std::hex << std::hex << iresult);
        iregs[rdest] = iresult;
      }
      break;
    case 2:
      D(3, "[" << std::dec << t << "] Dest Register: fr" << rdest << "=0x" << std::hex << std::hex << fresult);
      fregs[rdest] = fresult;
      break;
    default:
      break;
    }
  }

  PC_ += core_->arch().wsize();
  if (PC_ != nextPC) {
    D(3, "Next PC: " << std::hex << nextPC << std::dec);
    PC_ = nextPC;
  }
}
