#include <iostream>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <bitset>
#include <climits>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <util.h>
#include <rvfloats.h>
#include "warp.h"
#include "instr.h"
#include "core.h"

using namespace vortex;

static bool HasDivergentThreads(const ThreadMask &thread_mask,                                
                                const std::vector<std::vector<DoubleWord>> &reg_file,
                                unsigned reg) {
  bool cond;
  size_t thread_idx = 0;
  size_t num_threads = reg_file.size();
  for (; thread_idx < num_threads; ++thread_idx) {
    if (thread_mask[thread_idx]) {
      cond = bool(reg_file[thread_idx][reg]);
      break;
    }
  }  
  assert(thread_idx != num_threads);  
  for (; thread_idx < num_threads; ++thread_idx) {
    if (thread_mask[thread_idx]) {
      if (cond != (bool(reg_file[thread_idx][reg]))) {
        return true;
      }
    }
  }
  return false;
}

inline uint32_t get_fpu_rm(uint32_t func3, Core* core, uint32_t tid, uint32_t wid) {
  return (func3 == 0x7) ? core->get_csr(CSR_FRM, tid, wid) : func3;
}

inline void update_fcrs(uint32_t fflags, Core* core, uint32_t tid, uint32_t wid) {
  if (fflags) {
    core->set_csr(CSR_FCSR, core->get_csr(CSR_FCSR, tid, wid) | fflags, tid, wid);
    core->set_csr(CSR_FFLAGS, core->get_csr(CSR_FFLAGS, tid, wid) | fflags, tid, wid);
  }
}

void Warp::execute(const Instr &instr, Pipeline *pipeline) {
  assert(tmask_.any());

  // simx64
  DoubleWord nextPC = PC_ + 4;
  bool runOnce = false;
  
  Word func3 = instr.getFunc3();
  Word func6 = instr.getFunc6();
  Word func7 = instr.getFunc7();

  auto opcode = instr.getOpcode();
  int rdest  = instr.getRDest();
  int rsrc0  = instr.getRSrc(0);
  int rsrc1  = instr.getRSrc(1);
  DoubleWord immsrc= instr.getImm();
  DoubleWord vmask = instr.getVmask();

  int num_threads = core_->arch().num_threads();
  for (int t = 0; t < num_threads; t++) {
    if (!tmask_.test(t) || runOnce)
      continue;
    
    auto &iregs = iRegFile_.at(t);
    auto &fregs = fRegFile_.at(t);

    DoubleWord rsdata[3];
    DoubleWord rddata;

    int num_rsrcs = instr.getNRSrc();
    if (num_rsrcs) {    
      DPH(2, "[" << std::dec << t << "] Src Regs: ");
      for (int i = 0; i < num_rsrcs; ++i) {    
        int rst = instr.getRSType(i);
        int rs = instr.getRSrc(i);        
        if (i) DPN(2, ", ");
        switch (rst) {
        case 1: 
          rsdata[i] = iregs[rs];
          DPN(2, "r" << std::dec << rs << "=0x" << std::hex << rsdata[i]); 
          break;
        case 2: 
          rsdata[i] = fregs[rs];
          DPN(2, "fr" << std::dec << rs << "=0x" << std::hex << rsdata[i]); 
          break;
        default: break;
        }
      }
      DPN(2, std::endl);
    }

    bool rd_write = false;
  
    switch (opcode) {
    case NOP:
      break;
    case LUI_INST:
      rddata = (immsrc << 12) & 0xfffffffffffff000;
      rd_write = true;
      break;
    case AUIPC_INST:
      // simx64
      rddata = ((immsrc << 12) & 0xfffffffffffff000) + PC_;
      rd_write = true;
      break;
    case R_INST: {
      if (func7 & 0x1) {
        switch (func3) {
        case 0:
          // RV32M: MUL
          rddata = ((DoubleWordI)rsdata[0]) * ((DoubleWordI)rsdata[1]);
          break;
        case 1: {
          // RV32M: MULH
          __int128_t first = signExt128((__int128_t)rsdata[0], 64, 0xFFFFFFFFFFFFFFFF);
          __int128_t second = signExt128((__int128_t)rsdata[1], 64, 0xFFFFFFFFFFFFFFFF);
          __uint128_t result = first * second;
          rddata = (result >> 64) & 0xFFFFFFFFFFFFFFFF; 
        } break;
        case 2: {
          // RV32M: MULHSU
          __int128_t first = signExt128((__int128_t)rsdata[0], 64, 0xFFFFFFFFFFFFFFFF);
          __int128_t second = (__int128_t)rsdata[1];
          __uint128_t result = first * second;
          rddata = (result >> 64) & 0xFFFFFFFFFFFFFFFF;           
        } break;
        case 3: {
          // RV32M: MULHU
          __uint128_t first = (__uint128_t)rsdata[0];
          __uint128_t second = (__uint128_t)rsdata[1];
          rddata = ((first * second) >> 64) & 0xFFFFFFFFFFFFFFFF;
        } break;
        case 4: {
          // RV32M: DIV
          DoubleWordI dividen = rsdata[0];
          DoubleWordI divisor = rsdata[1];
          if (divisor == 0) {
            rddata = -1;
          } else if (dividen == DoubleWordI(0x8000000000000000) && divisor == DoubleWordI(0xFFFFFFFFFFFFFFFF)) {
            rddata = dividen;
          } else {
            rddata = dividen / divisor;
          }
        } break;
        case 5: {
          // RV32M: DIVU
          DoubleWord dividen = rsdata[0];
          DoubleWord divisor = rsdata[1];
          if (divisor == 0) {
            rddata = -1;
          } else {
            rddata = dividen / divisor;
          }
        } break;
        case 6: {
          // RV32M: REM
          DoubleWordI dividen = rsdata[0];
          DoubleWordI divisor = rsdata[1];
          if (divisor == 0) {
            rddata = dividen;
          } else if (dividen == DoubleWordI(0x8000000000000000) && divisor == DoubleWordI(0xFFFFFFFFFFFFFFFF)) {
            rddata = 0;
          } else {
            rddata = dividen % divisor;
          }
        } break;
        case 7: {
          // RV32M: REMU
          DoubleWord dividen = rsdata[0];
          DoubleWord divisor = rsdata[1];
          if (divisor == 0) {
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
            // RV32I: SUB
            rddata = rsdata[0] - rsdata[1];
          } else {
            // RV32I: ADD
            rddata = rsdata[0] + rsdata[1];       
          }
          break;
        case 1:
          // RV32I: SLL
          rddata = rsdata[0] << rsdata[1];
          break;
        case 2:
          // RV32I: SLT (signed)
          rddata = (DoubleWordI(rsdata[0]) < DoubleWordI(rsdata[1]));
          break;
        case 3:
          // RV32I: SLTU (unsigned)
          rddata = (DoubleWord(rsdata[0]) < DoubleWord(rsdata[1]));
          break;
        case 4:
          // RV32I: XOR
          rddata = rsdata[0] ^ rsdata[1];
          break;
        case 5:
          if (func7) {
            // RV32I: SRA
            rddata = DoubleWordI(rsdata[0]) >> DoubleWordI(rsdata[1]);
          } else {
            // RV32I: SRL
            rddata = DoubleWord(rsdata[0]) >> DoubleWord(rsdata[1]);
          }
          break;
        case 6:
          // RV32I: OR
          rddata = rsdata[0] | rsdata[1];
          break;
        case 7:
          // RV32I: AND
          rddata = rsdata[0] & rsdata[1];
          break;
        default:
          std::abort();
        }
      }
      rd_write = true;
    } break;
    case I_INST:
      switch (func3) {
      case 0:
        // RV32I: ADDI
        rddata = rsdata[0] + immsrc;
        break;
      case 1:
        // RV64I: SLLI
        rddata = rsdata[0] << immsrc;
        break;
      case 2:
        // RV32I: SLTI
        rddata = (DoubleWordI(rsdata[0]) < DoubleWordI(immsrc));
        break;
      case 3: {
        // RV32I: SLTIU
        rddata = (DoubleWord(rsdata[0]) < DoubleWord(immsrc));
      } break;
      case 4:
        // RV32I: XORI
        rddata = rsdata[0] ^ immsrc;
        break;
      case 5:
        if (func7) {
          // RV64I: SRAI
          // rs1 shifted by lower 6 bits of immsrc
          DoubleWord result = DoubleWordI(rsdata[0]) >> immsrc;
          rddata = result;
        } else {
          // RV64I: SRLI
          // rs1 shifted by lower 6 bits of immsrc
          DoubleWord result = DoubleWord(rsdata[0]) >> immsrc;
          rddata = result;
        }
        break;
      case 6:
        // RV32I: ORI
        rddata = rsdata[0] | immsrc;
        break;
      case 7:
        // RV32I: ANDI
        rddata = rsdata[0] & immsrc;
        break;
      default:
        std::abort();
      }
      rd_write = true;
      break;
    case B_INST:
      switch (func3) {
      case 0:
        // RV32I: BEQ
        if (rsdata[0] == rsdata[1]) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 1:
        // RV32I: BNE
        if (rsdata[0] != rsdata[1]) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 4:
        // RV32I: BLT
        if (DoubleWordI(rsdata[0]) < DoubleWordI(rsdata[1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 5:
        // RV32I: BGE
        if (DoubleWordI(rsdata[0]) >= DoubleWordI(rsdata[1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 6:
        // RV32I: BLTU
        if (DoubleWord(rsdata[0]) < DoubleWord(rsdata[1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 7:
        // RV32I: BGEU
        if (DoubleWord(rsdata[0]) >= DoubleWord(rsdata[1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      }
      pipeline->stall_warp = true;
      runOnce = true;
      break;
    // RV32I: JAL
    case JAL_INST:
      rddata = nextPC;
      nextPC = PC_ + immsrc;  
      pipeline->stall_warp = true;
      runOnce = true;
      rd_write = true;
      break;
    // RV32I: JALR
    case JALR_INST:
      rddata = nextPC;
      nextPC = DoubleWord(rsdata[0]) + DoubleWord(immsrc);
      pipeline->stall_warp = true;
      runOnce = true;
      rd_write = true;
      break;
    case L_INST: {
      DoubleWord memAddr   = ((rsdata[0] + immsrc) & 0xFFFFFFFC); // DoubleWord aligned
      DoubleWord shift_by  = ((rsdata[0] + immsrc) & 0x00000003) * 8;
      DoubleWord data_read = core_->dcache_read(memAddr, 8);
      D(3, "LOAD MEM: ADDRESS=0x" << std::hex << memAddr << ", DATA=0x" << data_read);
      switch (func3) {
      case 0:
        // RV32I: LBI
        rddata = signExt((data_read >> shift_by) & 0xFF, 8, 0xFF);
        break;
      case 1:
        // RV32I: LHI
        rddata = signExt((data_read >> shift_by) & 0xFFFF, 16, 0xFFFF);
        break;
      case 2:
        // RV32I: LW
        rddata = signExt((data_read >> shift_by) & 0xFFFFFFFF, 32, 0xFFFFFFFF);
        break;
      case 3:
        // RV64I: LD
        rddata = DoubleWord(data_read);
        break;
      case 4:
        // RV32I: LBU
        rddata = DoubleWord((data_read >> shift_by) & 0xFF);
        break;
      case 5:
        // RV32I: LHU
        rddata = DoubleWord((data_read >> shift_by) & 0xFFFF);
        break;
      case 6:
        // RV64I: LWU
        rddata = DoubleWord((data_read >> shift_by) & 0xFFFFFFFF);
        break;
      default:
        std::abort();        
      }
      rd_write = true;
    } break;
    case S_INST: {
      DoubleWord memAddr = rsdata[0] + immsrc;
      D(3, "STORE MEM: ADDRESS=0x" << std::hex << memAddr);
      switch (func3) {
      case 0:
        // RV32I: SB
        core_->dcache_write(memAddr, rsdata[1] & 0x000000FF, 1);
        break;
      case 1:
        // RV32I: SH
        core_->dcache_write(memAddr, rsdata[1] & 0x0000FFFF, 2);
        break;
      case 2:
        // RV32I: SW
        core_->dcache_write(memAddr, rsdata[1] & 0xFFFFFFFF, 4);
        break;
      case 3:
        // RV64I: SD
        core_ ->dcache_write(memAddr, rsdata[1], 8);
        break;
      default:
        std::abort();
      }
    } break;
    // simx64
    case R_INST_64: {
      if (func7 & 0x1){
        switch (func3) {
          case 0: 
            // RV64M: MULW
            rddata = signExt((WordI)rsdata[0] * (WordI)rsdata[1], 32, 0xFFFFFFFF);
            break;
          case 4: {
            // RV64M: DIVW
            int32_t dividen = (WordI) rsdata[0];
            int32_t divisor = (WordI) rsdata[1];
            if (divisor == 0){
              rddata = -1;
            } else if (dividen == WordI(0x80000000) && divisor == WordI(0xFFFFFFFF)) {
              rddata = signExt(dividen, 32, 0xFFFFFFFF);
            } else {
              rddata = signExt(dividen / divisor, 32, 0xFFFFFFFF);
            }
          } break;     
          case 5: {
            // RV64M: DIVUW
            uint32_t dividen = (Word) rsdata[0];
            uint32_t divisor = (Word) rsdata[1];
            if (divisor == 0){
              rddata = -1;
            } else {
              rddata = signExt(dividen / divisor, 32, 0xFFFFFFFF);
            }
          } break;
          case 6: {
            // RV64M: REMW
            int32_t dividen = (WordI) rsdata[0];
            int32_t divisor = (WordI) rsdata[1];
            if (divisor == 0){
              rddata = signExt(dividen, 32, 0xFFFFFFFF);
            } else if (dividen == WordI(0x80000000) && divisor == WordI(0xFFFFFFFF)) {
              rddata = 0;
            } else {
              rddata = signExt(dividen % divisor, 32, 0xFFFFFFFF);
            }
          } break; 
          case 7: {
            // RV64M: REMUW
            uint32_t dividen = (Word) rsdata[0];
            uint32_t divisor = (Word) rsdata[1];
            if (divisor == 0){
              rddata = signExt(dividen, 32, 0xFFFFFFFF);
            } else {
              rddata = signExt(dividen % divisor, 32, 0xFFFFFFFF);
            }
          } break; 
          default:
            std::abort();
        }
      } else {
        switch (func3) {
        case 0: 
          if (func7){
            // RV64I: SUBW
            rddata = signExt((Word)rsdata[0] - (Word)rsdata[1], 32, 0xFFFFFFFF);
          }
          else{
            // RV64I: ADDW
            rddata = signExt((Word)rsdata[0] + (Word)rsdata[1], 32, 0xFFFFFFFF);
          }    
          break;
        case 1: 
          // RV64I: SLLW
          rddata = signExt((Word)rsdata[0] << (Word)rsdata[1], 32, 0xFFFFFFFF);
          break;
        case 5:
          if (func7) {
            // RV64I: SRAW
            rddata = signExt((WordI)rsdata[0] >> (WordI)rsdata[1], 32, 0xFFFFFFFF);
          } else {
            // RV64I: SRLW
            rddata = signExt((Word)rsdata[0] >> (Word)rsdata[1], 32, 0xFFFFFFFF);
          }
          break;
        default:
          std::abort();
        }
      }
      rd_write = true;
    } break;
      
    // simx64
    case I_INST_64: {
      switch (func3) {
        case 0:
          // RV64I: ADDIW
          rddata = signExt((Word)rsdata[0] + (Word)immsrc, 32, 0xFFFFFFFF);
          break;
        case 1: 
          // RV64I: SLLIW
          rddata = signExt((Word)rsdata[0] << (Word)immsrc, 32, 0xFFFFFFFF);
          break;
        case 5:
          if (func7) {
            // RV64I: SRAIW
            DoubleWord result = signExt((WordI)rsdata[0] >> (WordI)immsrc, 32, 0xFFFFFFFF);
            rddata = result;
          } else {
            // RV64I: SRLIW
            DoubleWord result = signExt((Word)rsdata[0] >> (Word)immsrc, 32, 0xFFFFFFFF);
            rddata = result;
          }
          break;
        default:
          std::abort();
      }
      rd_write = true;
    } break;
    case SYS_INST: {
      DoubleWord csr_addr = immsrc & 0x00000FFF;
      DoubleWord csr_value = core_->get_csr(csr_addr, t, id_);
      switch (func3) {
      case 0:
        if (csr_addr < 2) {
          // ECALL/EBREAK
          core_->trigger_ebreak();
        }
        break;
      case 1:
        // RV32I: CSRRW
        rddata = csr_value;
        core_->set_csr(csr_addr, rsdata[0], t, id_);
        rd_write = true;
        break;
      case 2:
        // RV32I: CSRRS
        rddata = csr_value;
        core_->set_csr(csr_addr, csr_value | rsdata[0], t, id_);
        rd_write = true;
        break;
      case 3:
        // RV32I: CSRRC
        rddata = csr_value;
        core_->set_csr(csr_addr, csr_value & ~rsdata[0], t, id_);
        rd_write = true;
        break;
      case 5:
        // RV32I: CSRRWI
        rddata = csr_value;
        core_->set_csr(csr_addr, rsrc0, t, id_);
        rd_write = true;
        break;
      case 6:
        // RV32I: CSRRSI
        rddata = csr_value;
        core_->set_csr(csr_addr, csr_value | rsrc0, t, id_);
        rd_write = true;
        break;
      case 7:
        // RV32I: CSRRCI
        rddata = csr_value;
        core_->set_csr(csr_addr, csr_value & ~rsrc0, t, id_);
        rd_write = true;
        break;
      default:
        break;
      }
    } break;
    // RV32I: FENCE
    case FENCE:
      pipeline->stall_warp = true; 
      runOnce = true;
      break;
    case (FL | VL):
      if (func3 == 0x2) {
        // RV32F: FLW
        DoubleWord memAddr = rsdata[0] + immsrc;
        DoubleWord data_read = core_->dcache_read(memAddr, 4);        
        D(3, "LOAD MEM: ADDRESS=0x" << std::hex << memAddr << ", DATA=0x" << data_read);
        // simx64
        rddata = data_read | 0xFFFFFFFF00000000;
      } else {  
        D(3, "Executing vector load");      
        D(3, "lmul: " << vtype_.vlmul << " VLEN:" << (core_->arch().vsize() * 8) << "sew: " << vtype_.vsew);
        D(3, "src: " << rsrc0 << " " << rsdata[0]);
        D(3, "dest" << rdest);
        D(3, "width" << instr.getVlsWidth());

        auto &vd = vRegFile_[rdest];

        switch (instr.getVlsWidth()) {
        case 6: { 
          //load DoubleWord and unit strided (not checking for unit stride)
          for (int i = 0; i < vl_; i++) {
            DoubleWord memAddr = ((rsdata[0]) & 0xFFFFFFFC) + (i * vtype_.vsew / 8);
            D(3, "STORE MEM: ADDRESS=0x" << std::hex << memAddr);
            DoubleWord data_read = core_->dcache_read(memAddr, 4);
            D(3, "Mem addr: " << std::hex << memAddr << " Data read " << data_read);
            int *result_ptr = (int *)(vd.data() + i);
            *result_ptr = data_read;            
          }
        } break;
        default:
          std::abort();
        }
        break;
      } 
      rd_write = true;
      break;
    case (FS | VS):
      if (func3 == 0x2) {
        DoubleWord memAddr = rsdata[0] + immsrc;
        core_->dcache_write(memAddr, rsdata[1], 4);
        D(3, "STORE MEM: ADDRESS=0x" << std::hex << memAddr);
      } else {
        for (int i = 0; i < vl_; i++) {
          DoubleWord memAddr = rsdata[0] + (i * vtype_.vsew / 8);
          D(3, "STORE MEM: ADDRESS=0x" << std::hex << memAddr);
          switch (instr.getVlsWidth()) {
          case 6: {
            //store DoubleWord and unit strided (not checking for unit stride)          
            uint32_t value = *(uint32_t *)(vRegFile_[instr.getVs3()].data() + i);
            core_->dcache_write(memAddr, value, 4);
            D(3, "store: " << memAddr << " value:" << value);
          } break;
          default:
            std::abort();
          }          
        }
      }
      break;    
    case FCI: { 
      uint32_t frm = get_fpu_rm(func3, core_, t, id_);
      uint32_t fflags = 0;
      switch (func7) {
      case 0x00: // RV32F: FADD
        rddata = rv_fadd(rsdata[0], rsdata[1], frm, &fflags);
        break;
      case 0x04: // RV32F: FSUB
        rddata = rv_fsub(rsdata[0], rsdata[1], frm, &fflags);
        break;
      case 0x08: // RV32F: FMUL
        rddata = rv_fmul(rsdata[0], rsdata[1], frm, &fflags);
        break;
      case 0x0c: // RV32F: FDIV
        rddata = rv_fdiv(rsdata[0], rsdata[1], frm, &fflags);
        break;
      case 0x2c: // RV32F: FSQRT
        rddata = rv_fsqrt(rsdata[0], frm, &fflags);
        break;        
      case 0x10:
        switch (func3) {            
        case 0: // RV32F: FSGNJ.S
          rddata = rv_fsgnj(rsdata[0], rsdata[1]);
          break;          
        case 1: // RV32F: FSGNJN.S
          rddata = rv_fsgnjn(rsdata[0], rsdata[1]);
          break;          
        case 2: // RV32F: FSGNJX.S
          rddata = rv_fsgnjx(rsdata[0], rsdata[1]);
          break;
        }
        break;
      case 0x14:                
        if (func3) {
          // RV32F: FMAX.S
          rddata = rv_fmax(rsdata[0], rsdata[1], &fflags);
        } else {
          // RV32F: FMIN.S
          rddata = rv_fmin(rsdata[0], rsdata[1], &fflags);
        }
        break;
      case 0x60:
        switch(rsrc1) {
          case 0: 
            // RV32F: FCVT.W.S
            rddata = signExt(rv_ftoi(rsdata[0], frm, &fflags), 32, 0xFFFFFFFF);
            break;
          case 1:
            // RV32F: FCVT.WU.S
            rddata = signExt(rv_ftou(rsdata[0], frm, &fflags), 32, 0xFFFFFFFF);
            break;
          case 2:
            // RV64F: FCVT.L.S
            rddata = rv_ftol(rsdata[0], frm, &fflags);
            break;
          case 3:
            // RV64F: FCVT.LU.S
            rddata = rv_ftolu(rsdata[0], frm, &fflags);
            break;
        }
        break;
      case 0x70:      
        if (func3) {
          // RV32F: FCLASS.S
          rddata = rv_fclss(rsdata[0]);
        } else {          
          // RV32F: FMV.X.W
          rddata = signExt((Word)rsdata[0], 32, 0xFFFFFFFF);
        } 
        break;
      case 0x50:          
        switch(func3) {              
        case 0:
          // RV32F: FLE.S
          rddata = rv_fle(rsdata[0], rsdata[1], &fflags);    
          break;              
        case 1:
          // RV32F: FLT.S
          rddata = rv_flt(rsdata[0], rsdata[1], &fflags);
          break;              
        case 2:
          // RV32F: FEQ.S
          rddata = rv_feq(rsdata[0], rsdata[1], &fflags);
          break;
        } break;        
      case 0x68:
        switch(rsrc1) {
          case 0: 
            // RV32F: FCVT.S.W
            rddata = rv_itof(rsdata[0], frm, &fflags);
            break;
          case 1:
            // RV32F: FCVT.S.WU
            rddata = rv_utof(rsdata[0], frm, &fflags);
            break;
          case 2:
            // RV64F: FCVT.S.L
            rddata = rv_ltof(rsdata[0], frm, &fflags);
            break;
          case 3:
            // RV64F: FCVT.S.LU
            rddata = rv_lutof(rsdata[0], frm, &fflags);
            break;
        }
        break;
      case 0x78:
        // RV32F: FMV.W.X
        rddata = rsdata[0];
        break;
      }
      update_fcrs(fflags, core_, t, id_);
      rd_write = true;
    } break;
    case FMADD:      
    case FMSUB:      
    case FMNMADD:
    case FMNMSUB: {
      int frm = get_fpu_rm(func3, core_, t, id_);
      // simx64
      Word fflags = 0;
      switch (opcode) {
      case FMADD:
        // RV32F: FMADD
        rddata = rv_fmadd(rsdata[0], rsdata[1], rsdata[2], frm, &fflags);
        break;
      case FMSUB:
        // RV32F: FMSUB
        rddata = rv_fmsub(rsdata[0], rsdata[1], rsdata[2], frm, &fflags);
        break;
      case FMNMADD:
        // RV32F: FNMADD
        rddata = rv_fnmadd(rsdata[0], rsdata[1], rsdata[2], frm, &fflags);
        break;  
      case FMNMSUB:
        // RV32F: FNMSUB
        rddata = rv_fnmsub(rsdata[0], rsdata[1], rsdata[2], frm, &fflags);
        break;
      default:
        break;
      }              
      update_fcrs(fflags, core_, t, id_);
      rd_write = true;
    } break;
    case GPGPU:
      switch (func3) {
      case 0: {
        // TMC        
        if (rsrc1) {
          // predicate mode
          ThreadMask pred;
          for (int i = 0; i < num_threads; ++i) {
            pred[i] = tmask_[i] ? (iRegFile_[i][rsrc0] != 0) : 0;
          }
          if (pred.any()) {
            tmask_ &= pred;
          }
        } else {
          tmask_.reset();
          for (int i = 0; i < num_threads; ++i) {
            tmask_[i] = rsdata[0] & (1 << i);
          }
        }
        D(3, "*** TMC " << tmask_);
        active_ = tmask_.any();
        pipeline->stall_warp = true;
        runOnce = true;
      } break;
      case 1: {
        // WSPAWN
        int active_warps = std::min<int>(rsdata[0], core_->arch().num_warps());
        D(3, "*** Spawning " << (active_warps-1) << " warps at PC: " << std::hex << rsdata[1]);
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
        if (HasDivergentThreads(tmask_, iRegFile_, rsrc0)) {          
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

          DPH(3, "*** Split: New TM=");
          for (int i = 0; i < num_threads; ++i) DPN(3, tmask_[num_threads-i-1]);
          DPN(3, ", Pushed TM=");
          for (int i = 0; i < num_threads; ++i) DPN(3, e.tmask[num_threads-i-1]);
          DPN(3, ", PC=0x" << std::hex << e.PC << "\n");
        } else {
          D(3, "*** Unanimous pred");
          DomStackEntry e(tmask_);
          e.unanimous = true;
          domStack_.push(e);
        }
        pipeline->stall_warp = true;
        runOnce = true;
      } break;
      case 3: {
        // JOIN
        if (!domStack_.empty() && domStack_.top().unanimous) {
          D(3, "*** Uninimous branch at join");
          tmask_ = domStack_.top().tmask;
          active_ = tmask_.any();
          domStack_.pop();
        } else {
          if (!domStack_.top().fallThrough) {
            nextPC = domStack_.top().PC;
            D(3, "*** Join: next PC: " << std::hex << nextPC << std::dec);
          }

          tmask_ = domStack_.top().tmask;
          active_ = tmask_.any();

          DPH(3, "*** Join: New TM=");
          for (int i = 0; i < num_threads; ++i) DPN(3, tmask_[num_threads-i-1]);
          DPN(3, "\n");

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
      case 6: {
        // PREFETCH
        int addr = rsdata[0];
        printf("*** PREFETCHED %d ***\n", addr);
      } break;
      default:
        std::abort();
      }
      break;
    case VSET: {
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
                D(3, "Adding " << first << " + " << second << " = " << result);
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
                D(3, "Adding " << first << " + " << second << " = " << result);
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
                D(3, "Adding " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first == second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first == second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first != second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first != second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first < second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first < second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t first  = *(int16_t *)(vr1.data() + i);
              int16_t second = *(int16_t *)(vr2.data() + i);
              int16_t result = (first < second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(int16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t first  = *(int32_t *)(vr1.data() + i);
              int32_t second = *(int32_t *)(vr2.data() + i);
              int32_t result = (first < second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first <= second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first <= second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t first  = *(int16_t *)(vr1.data() + i);
              int16_t second = *(int16_t *)(vr2.data() + i);
              int16_t result = (first <= second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(int16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t first  = *(int32_t *)(vr1.data() + i);
              int32_t second = *(int32_t *)(vr2.data() + i);
              int32_t result = (first <= second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t first  = *(uint16_t *)(vr1.data() + i);
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (first > second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t first  = *(uint32_t *)(vr1.data() + i);
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (first > second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              int16_t first  = *(int16_t *)(vr1.data() + i);
              int16_t second = *(int16_t *)(vr2.data() + i);
              int16_t result = (first > second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(int16_t *)(vd.data() + i) = result;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              int32_t first  = *(int32_t *)(vr1.data() + i);
              int32_t second = *(int32_t *)(vr2.data() + i);
              int32_t result = (first > second) ? 1 : 0;
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 25: {
          // vmand
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 26: {
          // vmor
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 27: { 
          //vmxor
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 28: {
          //vmornot
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 29: {
          //vmnand
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 30: {
          //vmnor
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 31: {
          //vmxnor
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 37: {
          //vmul
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first * second);
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 45: {
          // vmacc
          auto &vr1 = vRegFile_[rsrc0];
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (first * second);
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
              D(3, "Comparing " << first << " + " << second << " = " << result);
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
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (rsdata[0] + second);
              D(3, "Comparing " << rsdata[0] << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (rsdata[0] + second);
              D(3, "Comparing " << rsdata[0] << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (rsdata[0] + second);
              D(3, "Comparing " << rsdata[0] << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint32_t *)(vd.data() + i) = 0;
            }
          }
        } break;
        case 37: {
          // vmul.vx
          auto &vr2 = vRegFile_[rsrc1];
          auto &vd = vRegFile_[rdest];
          if (vtype_.vsew == 8) {
            for (int i = 0; i < vl_; i++) {
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = (rsdata[0] * second);
              D(3, "Comparing " << rsdata[0] << " + " << second << " = " << result);
              *(uint8_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint8_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 16) {
            for (int i = 0; i < vl_; i++) {
              uint16_t second = *(uint16_t *)(vr2.data() + i);
              uint16_t result = (rsdata[0] * second);
              D(3, "Comparing " << rsdata[0] << " + " << second << " = " << result);
              *(uint16_t *)(vd.data() + i) = result;
            }
            for (int i = vl_; i < VLMAX; i++) {
              *(uint16_t *)(vd.data() + i) = 0;
            }
          } else if (vtype_.vsew == 32) {
            for (int i = 0; i < vl_; i++) {
              uint32_t second = *(uint32_t *)(vr2.data() + i);
              uint32_t result = (rsdata[0] * second);
              D(3, "Comparing " << rsdata[0] << " + " << second << " = " << result);
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

    if (rd_write) {
      int rdt = instr.getRDType();
      switch (rdt) {
      case 1:      
        if (rdest) {
          D(2, "[" << std::dec << t << "] Dest Regs: r" << rdest << "=0x" << std::hex << std::hex << rddata);
          iregs[rdest] = rddata;
        }
        break;
      case 2:
        D(2, "[" << std::dec << t << "] Dest Regs: fr" << rdest << "=0x" << std::hex << std::hex << rddata);
        fregs[rdest] = rddata;
        break;
      default:
        break;
      }
    }
  }

  // simx64
  PC_ += 4;
  if (PC_ != nextPC) {
    D(3, "*** Next PC: " << std::hex << nextPC << std::dec);
    PC_ = nextPC;
  }
}
