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

static void update_fcrs(Core* core, int tid, int wid, bool outOfRange = false) {
  if (fetestexcept(FE_INEXACT)) {
    core->set_csr(CSR_FCSR, core->get_csr(CSR_FCSR, tid, wid) | 0x1, tid, wid); // set NX bit
    core->set_csr(CSR_FFLAGS, core->get_csr(CSR_FFLAGS, tid, wid) | 0x1, tid, wid); // set NX bit
  }
  
  if (fetestexcept(FE_UNDERFLOW)) {
    core->set_csr(CSR_FCSR, core->get_csr(CSR_FCSR, tid, wid) | 0x2, tid, wid); // set UF bit
    core->set_csr(CSR_FFLAGS, core->get_csr(CSR_FFLAGS, tid, wid) | 0x2, tid, wid); // set UF bit
  }

  if (fetestexcept(FE_OVERFLOW)) {
    core->set_csr(CSR_FCSR, core->get_csr(CSR_FCSR, tid, wid) | 0x4, tid, wid); // set OF bit
    core->set_csr(CSR_FFLAGS, core->get_csr(CSR_FFLAGS, tid, wid) | 0x4, tid, wid); // set OF bit
  }

  if (fetestexcept(FE_DIVBYZERO)) {
    core->set_csr(CSR_FCSR, core->get_csr(CSR_FCSR, tid, wid) | 0x8, tid, wid); // set DZ bit
    core->set_csr(CSR_FFLAGS, core->get_csr(CSR_FFLAGS, tid, wid) | 0x8, tid, wid); // set DZ bit
  } 

  if (fetestexcept(FE_INVALID) || outOfRange) {
    core->set_csr(CSR_FCSR, core->get_csr(CSR_FCSR, tid, wid) | 0x10, tid, wid); // set NV bit
    core->set_csr(CSR_FFLAGS, core->get_csr(CSR_FFLAGS, tid, wid) | 0x10, tid, wid); // set NV bit
  }
}

uint8_t Warp::xtime(uint8_t byte) {
  static const uint8_t xor_with[] = {0x00, 0x1b};
  return ((byte << 1) & 0xff) ^ xor_with[byte >> 7];
}

uint32_t Warp::aes32esi(int mix_columns, int byte_select, uint32_t word) {
  if (byte_select < 0 || byte_select > 3) {
    std::cout << "byte_select " << byte_select << " out of range\n";
    std::abort();
  }

  uint8_t victim = (word >> (byte_select << 3)) & 0xff;
  uint8_t val = s_box_replace(victim);
  if (mix_columns) {
    // x * val = {02}.val
    uint8_t xval = xtime(val);
    // important observation: {03}.b = ({01} ^ {02}).b = b ^ {02}.b
    // {03}.val = (x^2 + x) * val
    uint8_t x21val = xval ^ val;
    switch (byte_select) {
      case 0: return (x21val << 24) | (   val << 16) | (   val << 8) | xval;
      case 1: return (   val << 24) | (   val << 16) | (  xval << 8) | x21val;
      case 2: return (   val << 24) | (  xval << 16) | (x21val << 8) | val;
      case 3: return (  xval << 24) | (x21val << 16) | (   val << 8) | val;
    }
  } else {
    return val << (byte_select << 3);
  }
}

uint32_t Warp::aes32dsi(int inv_mix_columns, int byte_select, uint32_t word) {
  if (byte_select < 0 || byte_select > 3) {
    std::cout << "byte_select " << byte_select << " out of range\n";
    std::abort();
  }

  uint8_t victim = (word >> (byte_select << 3)) & 0xff;
  uint8_t val = inv_s_box_replace(victim);
  if (inv_mix_columns) {
    uint8_t xval = xtime(val); // x.val = {02}.val
    uint8_t x2val = xtime(xval); // x^2.val = {04}.val
    uint8_t x3val = xtime(x2val); // x^3.val = {08}.val

    // {0e}.b = ({02} ^ {04} ^ {08}).b = {02}.b ^ {04}.b ^ {08}.b
    uint8_t x321val = xval ^ x2val ^ x3val;
    // {0b}.b = ({01} ^ {02} ^ {08}).b = b ^ {02}.b ^ {08}.b
    uint8_t x310val = val ^ xval ^ x3val;
    // {0d}.b = ({01} ^ {04} ^ {08}).b = b ^ {04}.b ^ {08}.b
    uint8_t x320val = val ^ x2val ^ x3val;
    // {09}.b = ({01} ^ {08}).b        = b ^ {08}.b
    uint8_t x30val = val ^ x3val;

    switch (byte_select) {
      case 0: return (x310val << 24) | (x320val << 16) | ( x30val << 8) | x321val;
      case 1: return (x320val << 24) | ( x30val << 16) | (x321val << 8) | x310val;
      case 2: return ( x30val << 24) | (x321val << 16) | (x310val << 8) | x320val;
      case 3: return (x321val << 24) | (x310val << 16) | (x320val << 8) | x30val;
    }
  } else {
    return val << (byte_select << 3);
  }
}

uint32_t Warp::rotr(int n, uint32_t x) {
  return (x >> n) | (x << (32 - n));
}

uint32_t Warp::Sigma0(uint32_t x) {
  return rotr(2, x) ^ rotr(13, x) ^ rotr(22, x);
}

uint32_t Warp::Sigma1(uint32_t x) {
  return rotr(6, x) ^ rotr(11, x) ^ rotr(25, x);
}

uint32_t Warp::sigma0(uint32_t x) {
  return rotr(7, x) ^ rotr(18, x) ^ (x >> 3);
}

uint32_t Warp::sigma1(uint32_t x) {
  return rotr(17, x) ^ rotr(19, x) ^ (x >> 10);
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
  Word immsrc= instr.getImm();
  Word vmask = instr.getVmask();

  int num_threads = core_->arch().num_threads();
  for (int t = 0; t < num_threads; t++) {
    if (!tmask_.test(t) || runOnce)
      continue;
    
    auto &iregs = iRegFile_.at(t);
    auto &fregs = fRegFile_.at(t);

    Word rsdata[3];
    Word rddata;

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
      rddata = (immsrc << 12) & 0xfffff000;
      break;
    case AUIPC_INST:
      rddata = ((immsrc << 12) & 0xfffff000) + PC_;
      break;
    case R_INST: {
      if (func7 & 0x1) {
        switch (func3) {
        case 0:
          if (func7 == 1) {
            // MUL
            rddata = ((WordI)rsdata[0]) * ((WordI)rsdata[1]);
          } else if (func7 & 0x19 == 0x19) {
            switch ((func7 >> 1) & 0x3) {
              // AES32ESI
              case 0:
                rddata = rsdata[0] ^ aes32esi(0, func7 >> 5, rsdata[1]);
                break;
              // AES32ESMI
              case 1:
                rddata = rsdata[0] ^ aes32esi(1, func7 >> 5, rsdata[1]);
                break;
              // AES32DSI
              case 2:
                rddata = rsdata[0] ^ aes32dsi(0, func7 >> 5, rsdata[1]);
                break;
              // AES32DSMI
              case 3:
                rddata = rsdata[0] ^ aes32dsi(1, func7 >> 5, rsdata[1]);
                break;
            }
          } else {
            std::cout << "unsupported MUL instr\n";
            std::abort();
          }
          break;
        case 1: {
          // MULH
          int64_t first = (int64_t)rsdata[0];
          if (rsdata[0] & 0x80000000) {
            first = first | 0xFFFFFFFF00000000;
          }
          int64_t second = (int64_t)rsdata[1];
          if (rsdata[1] & 0x80000000) {
            second = second | 0xFFFFFFFF00000000;
          }
          uint64_t result = first * second;
          rddata = (result >> 32) & 0xFFFFFFFF;
        } break;
        case 2: {
          // MULHSU          
          int64_t first = (int64_t)rsdata[0];
          if (rsdata[0] & 0x80000000) {
            first = first | 0xFFFFFFFF00000000;
          }
          int64_t second = (int64_t)rsdata[1];
          rddata = ((first * second) >> 32) & 0xFFFFFFFF;
        } break;
        case 3: {
          // MULHU
          uint64_t first = (uint64_t)rsdata[0];
          uint64_t second = (uint64_t)rsdata[1];
          rddata = ((first * second) >> 32) & 0xFFFFFFFF;
        } break;
        case 4: {
          // DIV
          WordI dividen = rsdata[0];
          WordI divisor = rsdata[1];
          if (divisor == 0) {
            rddata = -1;
          } else if (dividen == WordI(0x80000000) && divisor == WordI(0xffffffff)) {
            rddata = dividen;
          } else {
            rddata = dividen / divisor;
          }
        } break;
        case 5: {
          // DIVU
          Word dividen = rsdata[0];
          Word divisor = rsdata[1];
          if (divisor == 0) {
            rddata = -1;
          } else {
            rddata = dividen / divisor;
          }
        } break;
        case 6: {
          // REM
          WordI dividen = rsdata[0];
          WordI divisor = rsdata[1];
          if (rsdata[1] == 0) {
            rddata = dividen;
          } else if (dividen == WordI(0x80000000) && divisor == WordI(0xffffffff)) {
            rddata = 0;
          } else {
            rddata = dividen % divisor;
          }
        } break;
        case 7: {
          // REMU
          Word dividen = rsdata[0];
          Word divisor = rsdata[1];
          if (rsdata[1] == 0) {
            rddata = dividen;
          } else {
            rddata = dividen % divisor;
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
            rddata = rsdata[0] - rsdata[1];
          } else {
            rddata = rsdata[0] + rsdata[1];
          }
          break;
        case 1:
          if (func7) {
            // ROL
            rddata = (rsdata[0] << rsdata[1]) | (rsdata[0] >> (32 - rsdata[1]));
          } else {
            // SLL
            rddata = rsdata[0] << rsdata[1];
          }
          break;
        case 2:
          rddata = (WordI(rsdata[0]) < WordI(rsdata[1]));
          break;
        case 3:
          rddata = (Word(rsdata[0]) < Word(rsdata[1]));
          break;
        case 4:
          rddata = rsdata[0] ^ rsdata[1];
          break;
        case 5:
          if (func7 == 0x30) {
            // ROR
            rddata = (rsdata[0] >> rsdata[1]) | (rsdata[0] << (32 - rsdata[1]));
          } else if (func7) {
            // SRA
            rddata = WordI(rsdata[0]) >> WordI(rsdata[1]);
          } else {
            // SRL
            rddata = Word(rsdata[0]) >> Word(rsdata[1]);
          }
          break;
        case 6:
          rddata = rsdata[0] | rsdata[1];
          break;
        case 7:
          rddata = rsdata[0] & rsdata[1];
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
        rddata = rsdata[0] + immsrc;
        break;
      case 1:
        if (!(immsrc >> 5)) {
          // SLLI
          rddata = rsdata[0] << immsrc;
        } else {
          switch (immsrc & 0x3) {
            // SHA256SUM0
            case 0:
            rddata = Sigma0(rsdata[0]);
            break;
            // SHA256SUM1
            case 1:
            rddata = Sigma1(rsdata[0]);
            break;
            // SHA256SIG0
            case 2:
            rddata = sigma0(rsdata[0]);
            break;
            // SHA256SIG1
            case 3:
            rddata = sigma1(rsdata[0]);
            break;
          }
        }
        break;
      case 2:
        // SLTI
        rddata = (WordI(rsdata[0]) < WordI(immsrc));
        break;
      case 3: {
        // SLTIU
        rddata = (Word(rsdata[0]) < Word(immsrc));
      } break;
      case 4:
        // XORI
        rddata = rsdata[0] ^ immsrc;
        break;
      case 5:
        if (func7 == 0x30) {
          // RORI
          if (immsrc < 0) {
            rddata = (rsdata[0] << -immsrc) | (rsdata[0] >> (32 + immsrc));
          } else {
            rddata = (rsdata[0] >> immsrc) | (rsdata[0] << (32 - immsrc));
          }
        } else if (func7) {
          // SRAI
          Word result = WordI(rsdata[0]) >> immsrc;
          rddata = result;
        } else {
          // SRLI
          Word result = Word(rsdata[0]) >> immsrc;
          rddata = result;
        }
        break;
      case 6:
        // ORI
        rddata = rsdata[0] | immsrc;
        break;
      case 7:
        // ANDI
        rddata = rsdata[0] & immsrc;
        break;
      default:
        std::abort();
      }
      break;
    case B_INST:
      switch (func3) {
      case 0:
        // BEQ
        if (rsdata[0] == rsdata[1]) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 1:
        // BNE
        if (rsdata[0] != rsdata[1]) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 4:
        // BLT
        if (WordI(rsdata[0]) < WordI(rsdata[1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 5:
        // BGE
        if (WordI(rsdata[0]) >= WordI(rsdata[1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 6:
        // BLTU
        if (Word(rsdata[0]) < Word(rsdata[1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 7:
        // BGEU
        if (Word(rsdata[0]) >= Word(rsdata[1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      }
      pipeline->stall_warp = true;
      runOnce = true;
      break;
    case JAL_INST:
      rddata = nextPC;
      nextPC = PC_ + immsrc;  
      pipeline->stall_warp = true;
      runOnce = true;
      break;
    case JALR_INST:
      rddata = nextPC;
      nextPC = rsdata[0] + immsrc;
      pipeline->stall_warp = true;
      runOnce = true;
      break;
    case L_INST: {
      Word memAddr   = ((rsdata[0] + immsrc) & 0xFFFFFFFC); // word aligned
      Word shift_by  = ((rsdata[0] + immsrc) & 0x00000003) * 8;
      Word data_read = core_->dcache_read(memAddr, 4);
      D(3, "LOAD MEM: ADDRESS=0x" << std::hex << memAddr << ", DATA=0x" << data_read);
      switch (func3) {
      case 0:
        // LBI
        rddata = signExt((data_read >> shift_by) & 0xFF, 8, 0xFF);
        break;
      case 1:
        // LHI
        rddata = signExt((data_read >> shift_by) & 0xFFFF, 16, 0xFFFF);
        break;
      case 2:
        // LW
        rddata = data_read;
        break;
      case 4:
        // LBU
        rddata = Word((data_read >> shift_by) & 0xFF);
        break;
      case 5:
        // LHU
        rddata = Word((data_read >> shift_by) & 0xFFFF);
        break;
      default:
        std::abort();        
      }
    } break;
    case S_INST: {
      Word memAddr = rsdata[0] + immsrc;
      D(3, "STORE MEM: ADDRESS=0x" << std::hex << memAddr);
      switch (func3) {
      case 0:
        // SB
        core_->dcache_write(memAddr, rsdata[1] & 0x000000FF, 1);
        break;
      case 1:
        // SH
        core_->dcache_write(memAddr, rsdata[1], 2);
        break;
      case 2:
        // SW
        core_->dcache_write(memAddr, rsdata[1], 4);
        break;
      default:
        std::abort();
      }
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
        rddata = csr_value;
        core_->set_csr(csr_addr, rsdata[0], t, id_);
        break;
      case 2:
        // CSRRS
        rddata = csr_value;
        core_->set_csr(csr_addr, csr_value | rsdata[0], t, id_);
        break;
      case 3:
        // CSRRC
        rddata = csr_value;
        core_->set_csr(csr_addr, csr_value & ~rsdata[0], t, id_);
        break;
      case 5:
        // CSRRWI
        rddata = csr_value;
        core_->set_csr(csr_addr, rsrc0, t, id_);
        break;
      case 6:
        // CSRRSI
        rddata = csr_value;
        core_->set_csr(csr_addr, csr_value | rsrc0, t, id_);
        break;
      case 7:
        // CSRRCI
        rddata = csr_value;
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
        Word memAddr = rsdata[0] + immsrc;
        Word data_read = core_->dcache_read(memAddr, 4);        
        D(3, "LOAD MEM: ADDRESS=0x" << std::hex << memAddr << ", DATA=0x" << data_read);
        rddata = data_read;
      } else {  
        D(3, "Executing vector load");      
        D(4, "lmul: " << vtype_.vlmul << " VLEN:" << (core_->arch().vsize() * 8) << "sew: " << vtype_.vsew);
        D(4, "src: " << rsrc0 << " " << rsdata[0]);
        D(4, "dest" << rdest);
        D(4, "width" << instr.getVlsWidth());

        auto &vd = vRegFile_[rdest];

        switch (instr.getVlsWidth()) {
        case 6: { 
          //load word and unit strided (not checking for unit stride)
          for (int i = 0; i < vl_; i++) {
            Word memAddr = ((rsdata[0]) & 0xFFFFFFFC) + (i * vtype_.vsew / 8);
            D(3, "STORE MEM: ADDRESS=0x" << std::hex << memAddr);
            Word data_read = core_->dcache_read(memAddr, 4);
            D(4, "Mem addr: " << std::hex << memAddr << " Data read " << data_read);
            int *result_ptr = (int *)(vd.data() + i);
            *result_ptr = data_read;            
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
        Word memAddr = rsdata[0] + immsrc;
        core_->dcache_write(memAddr, rsdata[1], 4);
        D(3, "STORE MEM: ADDRESS=0x" << std::hex << memAddr);
      } else {
        for (int i = 0; i < vl_; i++) {
          Word memAddr = rsdata[0] + (i * vtype_.vsew / 8);
          D(3, "STORE MEM: ADDRESS=0x" << std::hex << memAddr);
          switch (instr.getVlsWidth()) {
          case 6: {
            //store word and unit strided (not checking for unit stride)          
            uint32_t value = *(uint32_t *)(vRegFile_[instr.getVs3()].data() + i);
            core_->dcache_write(memAddr, value, 4);
            D(4, "store: " << memAddr << " value:" << value);
          } break;
          default:
            std::abort();
          }          
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
          if (fpBinIsNan(rsdata[0]) || fpBinIsNan(rsdata[1])) { 
            // if one of op is NaN, one of them is not quiet NaN, them set FCSR
            if ((fpBinIsNan(rsdata[0])==2) | (fpBinIsNan(rsdata[1])==2)) {
              core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x10, t, id_); // set NV bit                
              core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x10, t, id_); // set NV bit
            }
            if (fpBinIsNan(rsdata[0]) && fpBinIsNan(rsdata[1])) 
              rddata = 0x7fc00000;  // canonical(quiet) NaN 
            else if (fpBinIsNan(rsdata[0]))
              rddata = rsdata[1];
            else
              rddata = rsdata[0];
          } else {
            float fpsrc_0 = intregToFloat(rsdata[0]);
            float fpsrc_1 = intregToFloat(rsdata[1]);
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
            
            // update fcsrs
            update_fcrs(core_, t, id_);

            D(4, "fpDest: " << fpDest);
            if (fpBinIsNan(floatToBin(fpDest)) == 0) {
              rddata = floatToBin(fpDest);
            } else  {              
              // According to risc-v spec p.64 section 11.3
              // If the result is NaN, it is the canonical NaN
              rddata = 0x7fc00000;
            }          
          }
        } break;

        // FSGNJ.S, FSGNJN.S, FSGNJX.S
        case 0x10: {
          bool     fsign1 = rsdata[0] & 0x80000000;
          uint32_t fdata1 = rsdata[0] & 0x7FFFFFFF;
          bool     fsign2 = rsdata[1] & 0x80000000;
          switch (func3) {            
          case 0: // FSGNJ.S
            rddata = (fsign2 << 31) | fdata1;
            break;          
          case 1: // FSGNJN.S
            fsign2 = !fsign2;
            rddata = (fsign2 << 31) | fdata1;
            break;          
          case 2: { // FSGNJX.S
            bool sign = fsign1 ^ fsign2;
            rddata = (sign << 31) | fdata1;
            } break;
          }
        } break;

        // FMIN.S, FMAX.S
        case 0x14: {
          if (fpBinIsNan(rsdata[0]) || fpBinIsNan(rsdata[1])) { // if one of src is NaN
            // one of them is not quiet NaN, them set FCSR
            if ((fpBinIsNan(rsdata[0])==2) | (fpBinIsNan(rsdata[1])==2)) {
              core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x10, t, id_); // set NV bit
              core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x10, t, id_); // set NV bit
            }
            if (fpBinIsNan(rsdata[0]) && fpBinIsNan(rsdata[1])) 
              rddata = 0x7fc00000;  // canonical(quiet) NaN 
            else if (fpBinIsNan(rsdata[0]))
              rddata = rsdata[1];
            else
              rddata = rsdata[0];
          } else {
            uint8_t sr0IsZero = fpBinIsZero(rsdata[0]);
            uint8_t sr1IsZero = fpBinIsZero(rsdata[1]);

            if (sr0IsZero && sr1IsZero && (sr0IsZero != sr1IsZero)) { 
              // both are zero and not equal
              // handle corner case that compare +0 and -0              
              if (func3) {
                // FMAX.S
                rddata = (sr1IsZero==2) ? rsdata[1] : rsdata[0];
              } else {
                // FMIM.S
                rddata = (sr1IsZero==2) ? rsdata[0] : rsdata[1];
              }
            } else {
              float rs1 = intregToFloat(rsdata[0]);
              float rs2 = intregToFloat(rsdata[1]);              
              if (func3) {
                // FMAX.S
                float fmax = std::max(rs1, rs2); 
                rddata = floatToBin(fmax);
              } else {
                // FMIN.S
                float fmin = std::min(rs1, rs2);
                rddata = floatToBin(fmin);
              }
            }
          }
        } break;
        
        // FCVT.W.S FCVT.WU.S
        case 0x60: {          
          float fpSrc = intregToFloat(rsdata[0]);
          Word result;
          bool outOfRange = false;          
          if (rsrc1 == 0) { 
            // FCVT.W.S
            // Convert floating point to 32-bit signed integer
            if (fpSrc > pow(2.0, 31) - 1 || fpBinIsNan(rsdata[0]) || fpBinIsInf(rsdata[0]) == 2) {
              feclearexcept(FE_ALL_EXCEPT);              
              outOfRange = true;
              // result = 2^31 - 1
              result = 0x7FFFFFFF;
            } else if (fpSrc < -1*pow(2.0, 31) || fpBinIsInf(rsdata[0]) == 1) {
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
            if (fpSrc > pow(2.0, 32) - 1 || fpBinIsNan(rsdata[0]) || fpBinIsInf(rsdata[0]) == 2) {
              feclearexcept(FE_ALL_EXCEPT);              
              outOfRange = true;
              // result = 2^32 - 1
              result = 0xFFFFFFFF;
            } else if (fpSrc <= -1.0 || fpBinIsInf(rsdata[0]) == 1) {
              feclearexcept(FE_ALL_EXCEPT);              
              outOfRange = true;
              // result = 0
              result = 0x00000000;
            } else {
              feclearexcept(FE_ALL_EXCEPT);              
              result = (uint32_t) fpSrc;
            }
          }

          // update fcsrs
          update_fcrs(core_, t, id_, outOfRange);

          rddata = result;
        } break;
        
        // FMV.X.W FCLASS.S
        case 0x70: {
          // FCLASS.S
          if (func3) {
            // Examine the value in fpReg rs1 and write to integer rd
            // a 10-bit mask to indicate the class of the fp number
            rddata = 0; // clear all bits

            bool fsign = rsdata[0] & 0x80000000;
            uint32_t expo = (rsdata[0]>>23) & 0x000000FF;
            uint32_t fraction = rsdata[0] & 0x007FFFFF;

            if ((expo==0) && (fraction==0)) {
             rddata = fsign ? (1<<3) : (1<<4); // +/- 0
            } else if ((expo==0) && (fraction!=0)) {
              rddata = fsign ? (1<<2) : (1<<5); // +/- subnormal
            } else if ((expo==0xFF) && (fraction==0)) {
              rddata = fsign ? (1<<0) : (1<<7); // +/- infinity
            } else if ((expo==0xFF) && (fraction!=0)) { 
              if (!fsign && (fraction == 0x00400000)) {
                rddata = (1<<9);               // quiet NaN
              } else { 
                rddata = (1<<8);               // signaling NaN
              }
            } else {
              rddata = fsign ? (1<<1) : (1<<6); // +/- normal
            }
          } else {          
            // FMV.X.W
            // Move bit values from floating-point register rs1 to integer register rd
            // Since we are using integer register to represent floating point register, 
            // just simply assign here.
            rddata = rsdata[0];
          } 
        } break;
        
        // FEQ.S FLT.S FLE.S
        // rdest is integer register
        case 0x50: {
          // TODO: FLT.S and FLE.S perform IEEE 754-2009, signaling comparisons, set
          // TODO: the invalid operation exception flag if either input is NaN
          if (fpBinIsNan(rsdata[0]) || fpBinIsNan(rsdata[1])) {
            // FLE.S or FLT.S
            if (func3 == 0 || func3 == 1) {
              // If either input is NaN, set NV bit
              core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x10, t, id_); // set NV bit
              core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x10, t, id_); // set NV bit
            } else { // FEQ.S
              // Only set NV bit if it is signaling NaN
              if (fpBinIsNan(rsdata[0]) == 2 || fpBinIsNan(rsdata[1]) == 2) {
                // If either input is NaN, set NV bit
                core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x10, t, id_); // set NV bit
                core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x10, t, id_); // set NV bit
              }
            }
            // The result is 0 if either operand is NaN
            rddata = 0;
          } else {
            switch(func3) {              
            case 0: {
              // FLE.S
              rddata = (intregToFloat(rsdata[0]) <= intregToFloat(rsdata[1]));
            } break;              
            case 1: {
              // FLT.S
              rddata = (intregToFloat(rsdata[0]) < intregToFloat(rsdata[1]));
            } break;              
            case 2: {
              // FEQ.S
              rddata = (intregToFloat(rsdata[0]) == intregToFloat(rsdata[1]));
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
            float data = rsdata[0];      
            rddata = floatToBin(data);
          } else {
            // FCVT.S.W: convert 32-bit signed integer to floating point
            // rsdata[0] is actually a unsigned number
            float data = (WordI)rsdata[0];
            rddata = floatToBin(data);
          }
        break;

        case 0x78: {
          // FMV.W.X
          // Move bit values from integer register rs1 to floating register rd
          // Since we are using integer register to represent floating point register, 
          // just simply assign here.
          rddata = rsdata[0];
        }
        break;
      }
    break;

    case FMADD:      
    case FMSUB:      
    case FMNMADD:
    case FMNMSUB: {
      // multiplicands are infinity and zero, them set FCSR
      if (fpBinIsZero(rsdata[0]) || fpBinIsZero(rsdata[1]) || fpBinIsInf(rsdata[0]) || fpBinIsInf(rsdata[1])) {
        core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x10, t, id_); // set NV bit
        core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x10, t, id_); // set NV bit
      }
      if (fpBinIsNan(rsdata[0]) || fpBinIsNan(rsdata[1]) || fpBinIsNan(rsdata[2])) { 
        // if one of op is NaN, if addend is not quiet NaN, them set FCSR
        if ((fpBinIsNan(rsdata[0])==2) | (fpBinIsNan(rsdata[1])==2) | (fpBinIsNan(rsdata[1])==2)) {
          core_->set_csr(CSR_FCSR, core_->get_csr(CSR_FCSR, t, id_) | 0x10, t, id_); // set NV bit
          core_->set_csr(CSR_FFLAGS, core_->get_csr(CSR_FFLAGS, t, id_) | 0x10, t, id_); // set NV bit          
        }
        rddata = 0x7fc00000;  // canonical(quiet) NaN 
      } else {
        float rs1 = intregToFloat(rsdata[0]);
        float rs2 = intregToFloat(rsdata[1]);
        float rs3 = intregToFloat(rsdata[2]);
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

        // update fcsrs
        update_fcrs(core_, t, id_);

        rddata = floatToBin(fpDest);
      }
    }
    break;
    case GPGPU:
      switch (func3) {
      case 0: {
        // TMC
        int active_threads = std::min<int>(rsdata[0], num_threads);          
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
        int active_warps = std::min<int>(rsdata[0], core_->arch().num_warps());
        D(0, "Spawning " << (active_warps-1) << " warps at PC: " << std::hex << rsdata[1]);
        for (int i = 1; i < active_warps; ++i) {
          Warp &newWarp = core_->warp(i);
          newWarp.setPC(rsdata[1]);
          newWarp.setTmask(0, true);
        }
        pipeline->stall_warp = true;
        runOnce = true;
      } break;
      case 2: {
        // SPLIT    
        if (checkUnanimous(rsrc0, iRegFile_, tmask_)) {
          D(3, "Unanimous pred: " << rsrc0 << "  val: " << rsdata[0] << "\n");
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
        core_->barrier(rsdata[0], rsdata[1], id_);
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
              uint8_t result = (rsdata[0] + second);
              D(4, "Comparing " << rsdata[0] << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (rsdata[0] + second);
              D(4, "Comparing " << rsdata[0] << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (rsdata[0] + second);
              D(4, "Comparing " << rsdata[0] << " + " << second << " = " << result);
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
              uint8_t result = (rsdata[0] * second);
              D(4, "Comparing " << rsdata[0] << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (rsdata[0] * second);
              D(4, "Comparing " << rsdata[0] << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (rsdata[0] * second);
              D(4, "Comparing " << rsdata[0] << " + " << second << " = " << result);
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

        D(3, "lmul:" << vtype_.vlmul << " sew:" << vtype_.vsew  << " ediv: " << vtype_.vediv << "rsrc_" << rsdata[0] << "VLMAX" << VLMAX);

        int s0 = rsdata[0];
        if (s0 <= VLMAX) {
          vl_ = s0;
        } else if (s0 < (2 * VLMAX)) {
          vl_ = (int)ceil((s0 * 1.0) / 2.0);
        } else if (s0 >= (2 * VLMAX)) {
          vl_ = VLMAX;
        }        
        rddata = vl_;
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
        D(3, "[" << std::dec << t << "] Dest Register: r" << rdest << "=0x" << std::hex << std::hex << rddata);
        iregs[rdest] = rddata;
      }
      break;
    case 2:
      D(3, "[" << std::dec << t << "] Dest Register: fr" << rdest << "=0x" << std::hex << std::hex << rddata);
      fregs[rdest] = rddata;
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

uint8_t Warp::s_box_replace(uint8_t byte) {
    static const uint8_t s_box[256] = {
        [0x00] = 0x63, [0x01] = 0x7c, [0x02] = 0x77, [0x03] = 0x7b,
        [0x04] = 0xf2, [0x05] = 0x6b, [0x06] = 0x6f, [0x07] = 0xc5,
        [0x08] = 0x30, [0x09] = 0x01, [0x0a] = 0x67, [0x0b] = 0x2b,
        [0x0c] = 0xfe, [0x0d] = 0xd7, [0x0e] = 0xab, [0x0f] = 0x76,
        [0x10] = 0xca, [0x11] = 0x82, [0x12] = 0xc9, [0x13] = 0x7d,
        [0x14] = 0xfa, [0x15] = 0x59, [0x16] = 0x47, [0x17] = 0xf0,
        [0x18] = 0xad, [0x19] = 0xd4, [0x1a] = 0xa2, [0x1b] = 0xaf,
        [0x1c] = 0x9c, [0x1d] = 0xa4, [0x1e] = 0x72, [0x1f] = 0xc0,
        [0x20] = 0xb7, [0x21] = 0xfd, [0x22] = 0x93, [0x23] = 0x26,
        [0x24] = 0x36, [0x25] = 0x3f, [0x26] = 0xf7, [0x27] = 0xcc,
        [0x28] = 0x34, [0x29] = 0xa5, [0x2a] = 0xe5, [0x2b] = 0xf1,
        [0x2c] = 0x71, [0x2d] = 0xd8, [0x2e] = 0x31, [0x2f] = 0x15,
        [0x30] = 0x04, [0x31] = 0xc7, [0x32] = 0x23, [0x33] = 0xc3,
        [0x34] = 0x18, [0x35] = 0x96, [0x36] = 0x05, [0x37] = 0x9a,
        [0x38] = 0x07, [0x39] = 0x12, [0x3a] = 0x80, [0x3b] = 0xe2,
        [0x3c] = 0xeb, [0x3d] = 0x27, [0x3e] = 0xb2, [0x3f] = 0x75,
        [0x40] = 0x09, [0x41] = 0x83, [0x42] = 0x2c, [0x43] = 0x1a,
        [0x44] = 0x1b, [0x45] = 0x6e, [0x46] = 0x5a, [0x47] = 0xa0,
        [0x48] = 0x52, [0x49] = 0x3b, [0x4a] = 0xd6, [0x4b] = 0xb3,
        [0x4c] = 0x29, [0x4d] = 0xe3, [0x4e] = 0x2f, [0x4f] = 0x84,
        [0x50] = 0x53, [0x51] = 0xd1, [0x52] = 0x00, [0x53] = 0xed,
        [0x54] = 0x20, [0x55] = 0xfc, [0x56] = 0xb1, [0x57] = 0x5b,
        [0x58] = 0x6a, [0x59] = 0xcb, [0x5a] = 0xbe, [0x5b] = 0x39,
        [0x5c] = 0x4a, [0x5d] = 0x4c, [0x5e] = 0x58, [0x5f] = 0xcf,
        [0x60] = 0xd0, [0x61] = 0xef, [0x62] = 0xaa, [0x63] = 0xfb,
        [0x64] = 0x43, [0x65] = 0x4d, [0x66] = 0x33, [0x67] = 0x85,
        [0x68] = 0x45, [0x69] = 0xf9, [0x6a] = 0x02, [0x6b] = 0x7f,
        [0x6c] = 0x50, [0x6d] = 0x3c, [0x6e] = 0x9f, [0x6f] = 0xa8,
        [0x70] = 0x51, [0x71] = 0xa3, [0x72] = 0x40, [0x73] = 0x8f,
        [0x74] = 0x92, [0x75] = 0x9d, [0x76] = 0x38, [0x77] = 0xf5,
        [0x78] = 0xbc, [0x79] = 0xb6, [0x7a] = 0xda, [0x7b] = 0x21,
        [0x7c] = 0x10, [0x7d] = 0xff, [0x7e] = 0xf3, [0x7f] = 0xd2,
        [0x80] = 0xcd, [0x81] = 0x0c, [0x82] = 0x13, [0x83] = 0xec,
        [0x84] = 0x5f, [0x85] = 0x97, [0x86] = 0x44, [0x87] = 0x17,
        [0x88] = 0xc4, [0x89] = 0xa7, [0x8a] = 0x7e, [0x8b] = 0x3d,
        [0x8c] = 0x64, [0x8d] = 0x5d, [0x8e] = 0x19, [0x8f] = 0x73,
        [0x90] = 0x60, [0x91] = 0x81, [0x92] = 0x4f, [0x93] = 0xdc,
        [0x94] = 0x22, [0x95] = 0x2a, [0x96] = 0x90, [0x97] = 0x88,
        [0x98] = 0x46, [0x99] = 0xee, [0x9a] = 0xb8, [0x9b] = 0x14,
        [0x9c] = 0xde, [0x9d] = 0x5e, [0x9e] = 0x0b, [0x9f] = 0xdb,
        [0xa0] = 0xe0, [0xa1] = 0x32, [0xa2] = 0x3a, [0xa3] = 0x0a,
        [0xa4] = 0x49, [0xa5] = 0x06, [0xa6] = 0x24, [0xa7] = 0x5c,
        [0xa8] = 0xc2, [0xa9] = 0xd3, [0xaa] = 0xac, [0xab] = 0x62,
        [0xac] = 0x91, [0xad] = 0x95, [0xae] = 0xe4, [0xaf] = 0x79,
        [0xb0] = 0xe7, [0xb1] = 0xc8, [0xb2] = 0x37, [0xb3] = 0x6d,
        [0xb4] = 0x8d, [0xb5] = 0xd5, [0xb6] = 0x4e, [0xb7] = 0xa9,
        [0xb8] = 0x6c, [0xb9] = 0x56, [0xba] = 0xf4, [0xbb] = 0xea,
        [0xbc] = 0x65, [0xbd] = 0x7a, [0xbe] = 0xae, [0xbf] = 0x08,
        [0xc0] = 0xba, [0xc1] = 0x78, [0xc2] = 0x25, [0xc3] = 0x2e,
        [0xc4] = 0x1c, [0xc5] = 0xa6, [0xc6] = 0xb4, [0xc7] = 0xc6,
        [0xc8] = 0xe8, [0xc9] = 0xdd, [0xca] = 0x74, [0xcb] = 0x1f,
        [0xcc] = 0x4b, [0xcd] = 0xbd, [0xce] = 0x8b, [0xcf] = 0x8a,
        [0xd0] = 0x70, [0xd1] = 0x3e, [0xd2] = 0xb5, [0xd3] = 0x66,
        [0xd4] = 0x48, [0xd5] = 0x03, [0xd6] = 0xf6, [0xd7] = 0x0e,
        [0xd8] = 0x61, [0xd9] = 0x35, [0xda] = 0x57, [0xdb] = 0xb9,
        [0xdc] = 0x86, [0xdd] = 0xc1, [0xde] = 0x1d, [0xdf] = 0x9e,
        [0xe0] = 0xe1, [0xe1] = 0xf8, [0xe2] = 0x98, [0xe3] = 0x11,
        [0xe4] = 0x69, [0xe5] = 0xd9, [0xe6] = 0x8e, [0xe7] = 0x94,
        [0xe8] = 0x9b, [0xe9] = 0x1e, [0xea] = 0x87, [0xeb] = 0xe9,
        [0xec] = 0xce, [0xed] = 0x55, [0xee] = 0x28, [0xef] = 0xdf,
        [0xf0] = 0x8c, [0xf1] = 0xa1, [0xf2] = 0x89, [0xf3] = 0x0d,
        [0xf4] = 0xbf, [0xf5] = 0xe6, [0xf6] = 0x42, [0xf7] = 0x68,
        [0xf8] = 0x41, [0xf9] = 0x99, [0xfa] = 0x2d, [0xfb] = 0x0f,
        [0xfc] = 0xb0, [0xfd] = 0x54, [0xfe] = 0xbb, [0xff] = 0x16
    };
    return s_box[byte];
}

uint8_t Warp::inv_s_box_replace(uint8_t byte) {
    static const uint8_t inv_s_box[256] = {
        [0x00] = 0x52, [0x01] = 0x09, [0x02] = 0x6a, [0x03] = 0xd5,
        [0x04] = 0x30, [0x05] = 0x36, [0x06] = 0xa5, [0x07] = 0x38,
        [0x08] = 0xbf, [0x09] = 0x40, [0x0a] = 0xa3, [0x0b] = 0x9e,
        [0x0c] = 0x81, [0x0d] = 0xf3, [0x0e] = 0xd7, [0x0f] = 0xfb,
        [0x10] = 0x7c, [0x11] = 0xe3, [0x12] = 0x39, [0x13] = 0x82,
        [0x14] = 0x9b, [0x15] = 0x2f, [0x16] = 0xff, [0x17] = 0x87,
        [0x18] = 0x34, [0x19] = 0x8e, [0x1a] = 0x43, [0x1b] = 0x44,
        [0x1c] = 0xc4, [0x1d] = 0xde, [0x1e] = 0xe9, [0x1f] = 0xcb,
        [0x20] = 0x54, [0x21] = 0x7b, [0x22] = 0x94, [0x23] = 0x32,
        [0x24] = 0xa6, [0x25] = 0xc2, [0x26] = 0x23, [0x27] = 0x3d,
        [0x28] = 0xee, [0x29] = 0x4c, [0x2a] = 0x95, [0x2b] = 0x0b,
        [0x2c] = 0x42, [0x2d] = 0xfa, [0x2e] = 0xc3, [0x2f] = 0x4e,
        [0x30] = 0x08, [0x31] = 0x2e, [0x32] = 0xa1, [0x33] = 0x66,
        [0x34] = 0x28, [0x35] = 0xd9, [0x36] = 0x24, [0x37] = 0xb2,
        [0x38] = 0x76, [0x39] = 0x5b, [0x3a] = 0xa2, [0x3b] = 0x49,
        [0x3c] = 0x6d, [0x3d] = 0x8b, [0x3e] = 0xd1, [0x3f] = 0x25,
        [0x40] = 0x72, [0x41] = 0xf8, [0x42] = 0xf6, [0x43] = 0x64,
        [0x44] = 0x86, [0x45] = 0x68, [0x46] = 0x98, [0x47] = 0x16,
        [0x48] = 0xd4, [0x49] = 0xa4, [0x4a] = 0x5c, [0x4b] = 0xcc,
        [0x4c] = 0x5d, [0x4d] = 0x65, [0x4e] = 0xb6, [0x4f] = 0x92,
        [0x50] = 0x6c, [0x51] = 0x70, [0x52] = 0x48, [0x53] = 0x50,
        [0x54] = 0xfd, [0x55] = 0xed, [0x56] = 0xb9, [0x57] = 0xda,
        [0x58] = 0x5e, [0x59] = 0x15, [0x5a] = 0x46, [0x5b] = 0x57,
        [0x5c] = 0xa7, [0x5d] = 0x8d, [0x5e] = 0x9d, [0x5f] = 0x84,
        [0x60] = 0x90, [0x61] = 0xd8, [0x62] = 0xab, [0x63] = 0x00,
        [0x64] = 0x8c, [0x65] = 0xbc, [0x66] = 0xd3, [0x67] = 0x0a,
        [0x68] = 0xf7, [0x69] = 0xe4, [0x6a] = 0x58, [0x6b] = 0x05,
        [0x6c] = 0xb8, [0x6d] = 0xb3, [0x6e] = 0x45, [0x6f] = 0x06,
        [0x70] = 0xd0, [0x71] = 0x2c, [0x72] = 0x1e, [0x73] = 0x8f,
        [0x74] = 0xca, [0x75] = 0x3f, [0x76] = 0x0f, [0x77] = 0x02,
        [0x78] = 0xc1, [0x79] = 0xaf, [0x7a] = 0xbd, [0x7b] = 0x03,
        [0x7c] = 0x01, [0x7d] = 0x13, [0x7e] = 0x8a, [0x7f] = 0x6b,
        [0x80] = 0x3a, [0x81] = 0x91, [0x82] = 0x11, [0x83] = 0x41,
        [0x84] = 0x4f, [0x85] = 0x67, [0x86] = 0xdc, [0x87] = 0xea,
        [0x88] = 0x97, [0x89] = 0xf2, [0x8a] = 0xcf, [0x8b] = 0xce,
        [0x8c] = 0xf0, [0x8d] = 0xb4, [0x8e] = 0xe6, [0x8f] = 0x73,
        [0x90] = 0x96, [0x91] = 0xac, [0x92] = 0x74, [0x93] = 0x22,
        [0x94] = 0xe7, [0x95] = 0xad, [0x96] = 0x35, [0x97] = 0x85,
        [0x98] = 0xe2, [0x99] = 0xf9, [0x9a] = 0x37, [0x9b] = 0xe8,
        [0x9c] = 0x1c, [0x9d] = 0x75, [0x9e] = 0xdf, [0x9f] = 0x6e,
        [0xa0] = 0x47, [0xa1] = 0xf1, [0xa2] = 0x1a, [0xa3] = 0x71,
        [0xa4] = 0x1d, [0xa5] = 0x29, [0xa6] = 0xc5, [0xa7] = 0x89,
        [0xa8] = 0x6f, [0xa9] = 0xb7, [0xaa] = 0x62, [0xab] = 0x0e,
        [0xac] = 0xaa, [0xad] = 0x18, [0xae] = 0xbe, [0xaf] = 0x1b,
        [0xb0] = 0xfc, [0xb1] = 0x56, [0xb2] = 0x3e, [0xb3] = 0x4b,
        [0xb4] = 0xc6, [0xb5] = 0xd2, [0xb6] = 0x79, [0xb7] = 0x20,
        [0xb8] = 0x9a, [0xb9] = 0xdb, [0xba] = 0xc0, [0xbb] = 0xfe,
        [0xbc] = 0x78, [0xbd] = 0xcd, [0xbe] = 0x5a, [0xbf] = 0xf4,
        [0xc0] = 0x1f, [0xc1] = 0xdd, [0xc2] = 0xa8, [0xc3] = 0x33,
        [0xc4] = 0x88, [0xc5] = 0x07, [0xc6] = 0xc7, [0xc7] = 0x31,
        [0xc8] = 0xb1, [0xc9] = 0x12, [0xca] = 0x10, [0xcb] = 0x59,
        [0xcc] = 0x27, [0xcd] = 0x80, [0xce] = 0xec, [0xcf] = 0x5f,
        [0xd0] = 0x60, [0xd1] = 0x51, [0xd2] = 0x7f, [0xd3] = 0xa9,
        [0xd4] = 0x19, [0xd5] = 0xb5, [0xd6] = 0x4a, [0xd7] = 0x0d,
        [0xd8] = 0x2d, [0xd9] = 0xe5, [0xda] = 0x7a, [0xdb] = 0x9f,
        [0xdc] = 0x93, [0xdd] = 0xc9, [0xde] = 0x9c, [0xdf] = 0xef,
        [0xe0] = 0xa0, [0xe1] = 0xe0, [0xe2] = 0x3b, [0xe3] = 0x4d,
        [0xe4] = 0xae, [0xe5] = 0x2a, [0xe6] = 0xf5, [0xe7] = 0xb0,
        [0xe8] = 0xc8, [0xe9] = 0xeb, [0xea] = 0xbb, [0xeb] = 0x3c,
        [0xec] = 0x83, [0xed] = 0x53, [0xee] = 0x99, [0xef] = 0x61,
        [0xf0] = 0x17, [0xf1] = 0x2b, [0xf2] = 0x04, [0xf3] = 0x7e,
        [0xf4] = 0xba, [0xf5] = 0x77, [0xf6] = 0xd6, [0xf7] = 0x26,
        [0xf8] = 0xe1, [0xf9] = 0x69, [0xfa] = 0x14, [0xfb] = 0x63,
        [0xfc] = 0x55, [0xfd] = 0x21, [0xfe] = 0x0c, [0xff] = 0x7d,
    };
    return inv_s_box[byte];
}
