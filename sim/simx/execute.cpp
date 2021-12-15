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
                                const std::vector<std::vector<DWord>> &reg_file,
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

void Warp::execute(const Instr &instr, pipeline_trace_t *trace) {
  assert(tmask_.any());

  DWord nextPC = PC_ + core_->arch().wsize();

  Word func2  = instr.getFunc2();
  Word func3  = instr.getFunc3();
  Word func6  = instr.getFunc6();
  Word func7  = instr.getFunc7();

  auto opcode = instr.getOpcode();
  int rdest   = instr.getRDest();
  int rsrc0   = instr.getRSrc(0);
  int rsrc1   = instr.getRSrc(1);
  int rsrc2   = instr.getRSrc(2);
  DWord immsrc = instr.getImm();
  Word vmask  = instr.getVmask();

  int num_threads = core_->arch().num_threads();

  std::vector<DWord[3]> rsdata(num_threads);
  std::vector<DWord> rddata(num_threads);
  
  int num_rsrcs = instr.getNRSrc();
  if (num_rsrcs) {              
    for (int i = 0; i < num_rsrcs; ++i) {    
      DPH(2, "Src Reg [" << std::dec << i << "]: ");
      auto type = instr.getRSType(i);
      int reg = instr.getRSrc(i);        
      switch (type) {
      case RegType::Integer: 
        DPN(2, "r" << std::dec << reg << "={");
        for (int t = 0; t < num_threads; ++t) {
          if (t) DPN(2, ", ");
          if (!tmask_.test(t)) {
            DPN(2, "-");
            continue;            
          }
          rsdata[t][i] = ireg_file_.at(t)[reg];          
          DPN(2, std::hex << rsdata[t][i]); 
        }
        DPN(2, "}" << std::endl);
        break;
      case RegType::Float: 
        DPN(2, "fr" << std::dec << reg << "={");
        for (int t = 0; t < num_threads; ++t) {
          if (t) DPN(2, ", ");
          if (!tmask_.test(t)) {
            DPN(2, "-");
            continue;            
          }
          rsdata[t][i] = freg_file_.at(t)[reg];
          DPN(2, std::hex << rsdata[t][i]); 
        }
        DPN(2, "}" << std::endl);
        break;
      default: 
        std::abort();
        break;
      }      
    }
  }

  bool rd_write = false;
  
  switch (opcode) {
  case NOP:
    break;
  case LUI_INST:
    trace->exe_type = ExeType::ALU;
    trace->alu.type = AluType::ARITH;
    for (int t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      rddata[t] = (immsrc << 12) & 0xfffffffffffff000;
    }    
    rd_write = true;
    break;
  case AUIPC_INST:
    trace->exe_type = ExeType::ALU;
    trace->alu.type = AluType::ARITH;
    for (int t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      rddata[t] = ((immsrc << 12) & 0xfffffffffffff000) + PC_;
    }    
    rd_write = true;
    break;
  case R_INST:
    trace->exe_type = ExeType::ALU;    
    trace->alu.type = AluType::ARITH;
    trace->used_iregs.set(rsrc0);
    trace->used_iregs.set(rsrc1);
    for (int t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      if (func7 & 0x1) {
        switch (func3) {
        case 0:
          // RV32M: MUL
          rddata[t] = ((DWordI)rsdata[t][0]) * ((DWordI)rsdata[t][1]);
          trace->alu.type = AluType::IMUL;
          break;
        case 1: {
          // RV32M: MULH
          __int128_t first = sext128((__int128_t)rsdata[t][0], 64);
          __int128_t second = sext128((__int128_t)rsdata[t][1], 64);
          rddata[t] = ((first * second) >> 64) & 0xFFFFFFFFFFFFFFFF;
          trace->alu.type = AluType::IMUL;
        } break;
        case 2: {
          // RV32M: MULHSU       
          __int128_t first = sext128((__int128_t)rsdata[t][0], 64);
          __int128_t second = (__int128_t)rsdata[t][1];
          rddata[t] = ((first * second) >> 64) & 0xFFFFFFFFFFFFFFFF;
          trace->alu.type = AluType::IMUL;
        } break;
        case 3: {
          // RV32M: MULHU
          __uint128_t first = (__int128_t)rsdata[t][0];
          __uint128_t second = (__int128_t)rsdata[t][1];
          rddata[t] = ((first * second) >> 64) & 0xFFFFFFFFFFFFFFFF;
          trace->alu.type = AluType::IMUL;
        } break;
        case 4: {
          // RV32M: DIV
          DWordI dividen = rsdata[t][0];
          DWordI divisor = rsdata[t][1];
          if (divisor == 0) {
            rddata[t] = -1;
          } else if (dividen == DWordI(0x8000000000000000) && divisor == DWordI(0xffffffffffffffff)) {
            rddata[t] = dividen;
          } else {
            rddata[t] = dividen / divisor;
          }
          trace->alu.type = AluType::IDIV;
        } break;
        case 5: {
          // RV32M: DIVU
          DWord dividen = rsdata[t][0];
          DWord divisor = rsdata[t][1];
          if (divisor == 0) {
            rddata[t] = -1;
          } else {
            rddata[t] = dividen / divisor;
          }
          trace->alu.type = AluType::IDIV;
        } break;
        case 6: {
          // RV32M: REM
          DWordI dividen = rsdata[t][0];
          DWordI divisor = rsdata[t][1];
          if (rsdata[t][1] == 0) {
            rddata[t] = dividen;
          } else if (dividen == DWordI(0x8000000000000000) && divisor == DWordI(0xffffffffffffffff)) {
            rddata[t] = 0;
          } else {
            rddata[t] = dividen % divisor;
          }
          trace->alu.type = AluType::IDIV;
        } break;
        case 7: {
          // RV32M: REMU
          DWord dividen = rsdata[t][0];
          DWord divisor = rsdata[t][1];
          if (rsdata[t][1] == 0) {
            rddata[t] = dividen;
          } else {
            rddata[t] = dividen % divisor;
          }
          trace->alu.type = AluType::IDIV;
        } break;
        default:
          std::abort();
        }
      } else {
        switch (func3) {
        case 0:
          if (func7) {
            // RV32I: SUB
            rddata[t] = rsdata[t][0] - rsdata[t][1];
          } else {
            // RV32I: ADD
            rddata[t] = rsdata[t][0] + rsdata[t][1];
          }
          break;
        case 1:
          // RV32I: SHL
          rddata[t] = rsdata[t][0] << rsdata[t][1];
          break;
        case 2:
          // RV32I: LT
          rddata[t] = (DWordI(rsdata[t][0]) < DWordI(rsdata[t][1]));
          break;
        case 3:
          // RV32I: LTU
          rddata[t] = (DWord(rsdata[t][0]) < DWord(rsdata[t][1]));
          break;
        case 4:
          // RV32I: XOR
          rddata[t] = rsdata[t][0] ^ rsdata[t][1];
          break;
        case 5:
          if (func7) {
            // RV32I: SRA
            rddata[t] = DWordI(rsdata[t][0]) >> DWordI(rsdata[t][1]);
          } else {
            // RV32I: SHR
            rddata[t] = DWord(rsdata[t][0]) >> DWord(rsdata[t][1]);
          }
          break;
        case 6:
          // RV32I: OR
          rddata[t] = rsdata[t][0] | rsdata[t][1];
          break;
        case 7:
          // RV32I: AND
          rddata[t] = rsdata[t][0] & rsdata[t][1];
          break;
        default:
          std::abort();
        }
      }
    }    
    rd_write = true;
    break;
  case I_INST:
    trace->exe_type = ExeType::ALU;    
    trace->alu.type = AluType::ARITH;    
    trace->used_iregs.set(rsrc0);
    for (int t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      switch (func3) {
      case 0:
        // RV32I: ADDI
        rddata[t] = rsdata[t][0] + immsrc;
        break;
      case 1:
        // RV32I: SLLI
        rddata[t] = rsdata[t][0] << immsrc;
        break;
      case 2:
        // RV32I: SLTI
        rddata[t] = (DWordI(rsdata[t][0]) < DWordI(immsrc));
        break;
      case 3: {
        // RV32I: SLTIU
        rddata[t] = rsdata[t][0] < immsrc;
      } break;
      case 4:
        // RV32I: XORI
        rddata[t] = rsdata[t][0] ^ immsrc;
        break;
      case 5:
        if (func7) {
          // RV32I: SRAI
          DWord result = DWordI(rsdata[t][0]) >> immsrc;
          rddata[t] = result;
        } else {
          // RV32I: SRLI
          DWord result = rsdata[t][0] >> immsrc;
          rddata[t] = result;
        }
        break;
      case 6:
        // RV32I: ORI
        rddata[t] = rsdata[t][0] | immsrc;
        break;
      case 7:
        // RV32I: ANDI
        rddata[t] = rsdata[t][0] & immsrc;
        break;
      }
    }
    rd_write = true;
    break;
  case R_INST_64:
    trace->exe_type = ExeType::ALU;    
    trace->alu.type = AluType::ARITH;
    trace->used_iregs.set(rsrc0);
    trace->used_iregs.set(rsrc1);
    for (int t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      if (func7 & 0x1){
        switch (func3) {
          case 0: 
            // RV64M: MULW
            rddata[t] = sext64((WordI)rsdata[t][0] * (WordI)rsdata[t][1], 32);
            break;
          case 4: {
            // RV64M: DIVW
            int32_t dividen = (WordI) rsdata[t][0];
            int32_t divisor = (WordI) rsdata[t][1];
            if (divisor == 0){
              rddata[t] = -1;
            } else if (dividen == WordI(0x80000000) && divisor == WordI(0xFFFFFFFF)) {
              rddata[t] = sext64(dividen, 32);
            } else {
              rddata[t] = sext64(dividen / divisor, 32);
            }
          } break;     
          case 5: {
            // RV64M: DIVUW
            uint32_t dividen = (Word) rsdata[t][0];
            uint32_t divisor = (Word) rsdata[t][1];
            if (divisor == 0){
              rddata[t] = -1;
            } else {
              rddata[t] = sext64(dividen / divisor, 32);
            }
          } break;
          case 6: {
            // RV64M: REMW
            int32_t dividen = (WordI) rsdata[t][0];
            int32_t divisor = (WordI) rsdata[t][1];
            if (divisor == 0){
              rddata[t] = sext64(dividen, 32);
            } else if (dividen == WordI(0x80000000) && divisor == WordI(0xFFFFFFFF)) {
              rddata[t] = 0;
            } else {
              rddata[t] = sext64(dividen % divisor, 32);
            }
          } break; 
          case 7: {
            // RV64M: REMUW
            uint32_t dividen = (Word) rsdata[t][0];
            uint32_t divisor = (Word) rsdata[t][1];
            if (divisor == 0){
              rddata[t] = sext64(dividen, 32);
            } else {
              rddata[t] = sext64(dividen % divisor, 32);
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
            rddata[t] = sext64((Word)rsdata[t][0] - (Word)rsdata[t][1], 32);
          }
          else{
            // RV64I: ADDW
            rddata[t] = sext64((Word)rsdata[t][0] + (Word)rsdata[t][1], 32);
          }    
          break;
        case 1: 
          // RV64I: SLLW
          rddata[t] = sext64((Word)rsdata[t][0] << (Word)rsdata[t][1], 32);
          break;
        case 5:
          if (func7) {
            // RV64I: SRAW
            rddata[t] = sext64((WordI)rsdata[t][0] >> (WordI)rsdata[t][1], 32);
          } else {
            // RV64I: SRLW
            rddata[t] = sext64((Word)rsdata[t][0] >> (Word)rsdata[t][1], 32);
          }
          break;
        default:
          std::abort();
        }
      }
    }
    rd_write = true;
    break; 
  case I_INST_64:
    trace->exe_type = ExeType::ALU;    
    trace->alu.type = AluType::ARITH;    
    trace->used_iregs.set(rsrc0);
    for (int t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      switch (func3) {
        case 0: {
          // RV64I: ADDIW
          rddata[t] = sext64((Word)rsdata[t][0] + (Word)immsrc, 32);
          break;
        }
        case 1: 
          // RV64I: SLLIW
          rddata[t] = sext64((Word)rsdata[t][0] << (Word)immsrc, 32);
          break;
        case 5:
          if (func7) {
            // RV64I: SRAIW
            DWord result = sext64((WordI)rsdata[t][0] >> (WordI)immsrc, 32);
            rddata[t] = result;
          } else {
            // RV64I: SRLIW
            DWord result = sext64((Word)rsdata[t][0] >> (Word)immsrc, 32);
            rddata[t] = result;
          }
          break;
        default:
          std::abort();
      }
    }
    rd_write = true;
    break;
  case B_INST:    
    trace->exe_type = ExeType::ALU;    
    trace->alu.type = AluType::BRANCH;    
    trace->used_iregs.set(rsrc0);
    trace->used_iregs.set(rsrc1);
    for (int t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      switch (func3) {
      case 0:
        // RV32I: BEQ
        if (rsdata[t][0] == rsdata[t][1]) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 1:
        // RV32I: BNE
        if (rsdata[t][0] != rsdata[t][1]) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 4:
        // RV32I: BLT
        if (DWordI(rsdata[t][0]) < DWordI(rsdata[t][1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 5:
        // RV32I: BGE
        if (DWordI(rsdata[t][0]) >= DWordI(rsdata[t][1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 6:
        // RV32I: BLTU
        if (DWord(rsdata[t][0]) < DWord(rsdata[t][1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      case 7:
        // RV32I: BGEU
        if (DWord(rsdata[t][0]) >= DWord(rsdata[t][1])) {
          nextPC = PC_ + immsrc;
        }
        break;
      default:
        std::abort();
      }
      break; // runonce
    }
    trace->fetch_stall = true;
    break;
  case JAL_INST:    
    trace->exe_type = ExeType::ALU;    
    trace->alu.type = AluType::BRANCH;
    for (int t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      rddata[t] = nextPC;
      nextPC = PC_ + immsrc;  
      trace->fetch_stall = true;
      break; // runonce
    }
    rd_write = true;
    break;
  case JALR_INST:
    trace->exe_type = ExeType::ALU;    
    trace->alu.type = AluType::BRANCH;
    trace->used_iregs.set(rsrc0);
    for (int t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      rddata[t] = nextPC;
      nextPC = rsdata[t][0] + immsrc;
      trace->fetch_stall = true;
      break; // runOnce
    }
    rd_write = true;
    break;
  case L_INST:
  case FL:
    trace->exe_type = ExeType::LSU;    
    trace->lsu.type = LsuType::LOAD;
    trace->used_iregs.set(rsrc0);
    if (opcode == L_INST 
    || (opcode == FL && func3 == 2)) {
      for (int t = 0; t < num_threads; ++t) {
        if (!tmask_.test(t))
          continue;
        DWord mem_addr  = ((rsdata[t][0] + immsrc) & 0xFFFFFFFFFFFFFFF8); // double word aligned
        DWord shift_by  = ((rsdata[t][0] + immsrc) & 0x00000007) * 8;
        DWord data_read = core_->dcache_read(mem_addr, 8);
        trace->mem_addrs.at(t).push_back({mem_addr, 8});
        DP(4, "LOAD MEM: ADDRESS=0x" << std::hex << mem_addr << ", DATA=0x" << data_read);
        switch (func3) {
        case 0:
          // RV32I: LBI
          rddata[t] = sext64((data_read >> shift_by) & 0xFF, 8);
          break;
        case 1:
          // RV32I: LHI
          rddata[t] = sext64((data_read >> shift_by) & 0xFFFF, 16);
          break;
        case 2:
          // RV32I: LW
          rddata[t] = sext64((data_read >> shift_by) & 0xFFFFFFFF, 32);
          break;
        case 3:
          // RV64I: LD
          rddata[t] = data_read;
          break;
        case 4:
          // RV32I: LBU
          rddata[t] = DWord((data_read >> shift_by) & 0xFF);
          break;
        case 5:
          // RV32I: LHU
          rddata[t] = DWord((data_read >> shift_by) & 0xFFFF);
          break; 
        case 6:
          // RV64I: LWU
          rddata[t] = DWord((data_read >> shift_by) & 0xFFFFFFFF);
          break;
        default:
          std::abort();      
        }
      }
    } else {
      DP(4, "Executing vector load");      
      DP(4, "lmul: " << vtype_.vlmul << " VLEN:" << (core_->arch().vsize() * 8) << "sew: " << vtype_.vsew);
      DP(4, "dest: v" << rdest);
      DP(4, "width" << instr.getVlsWidth());
      auto &vd = vreg_file_.at(rdest);
      switch (instr.getVlsWidth()) {
      case 6: { 
        // load word and unit strided (not checking for unit stride)
        for (int i = 0; i < vl_; i++) {
          Word mem_addr = ((rsdata[i][0]) & 0xFFFFFFFC) + (i * vtype_.vsew / 8);
          DP(4, "LOAD MEM: ADDRESS=0x" << std::hex << mem_addr);
          Word data_read = core_->dcache_read(mem_addr, 4);
          DP(4, "Mem addr: " << std::hex << mem_addr << " Data read " << data_read);
          int *result_ptr = (int *)(vd.data() + i);
          *result_ptr = data_read;            
        }
      } break;
      default:
        std::abort();
      }
    }
    rd_write = true;
    break;
  case S_INST:   
  case FS:  
    trace->exe_type = ExeType::LSU;    
    trace->lsu.type = LsuType::STORE;
    trace->used_iregs.set(rsrc0);
    trace->used_iregs.set(rsrc1);
    if (opcode == S_INST 
    || (opcode == FS && func3 == 2)) {
      for (int t = 0; t < num_threads; ++t) {
        if (!tmask_.test(t))
          continue;
        DWord mem_addr = rsdata[t][0] + immsrc;
        trace->mem_addrs.at(t).push_back({mem_addr, (1u << func3)});
        DP(4, "STORE MEM: ADDRESS=0x" << std::hex << mem_addr);
        switch (func3) {
        case 0:
          // RV32I: SB
          core_->dcache_write(mem_addr, rsdata[t][1] & 0x000000FF, 1);
          break;
        case 1:
          // RV32I: SH
          core_->dcache_write(mem_addr, rsdata[t][1] & 0x0000FFFF, 2);
          break;
        case 2:
          // RV32I: SW
          core_->dcache_write(mem_addr, rsdata[t][1] & 0xFFFFFFFF, 4);
          break;
        case 3:
          // RV64I: SD
          core_->dcache_write(mem_addr, rsdata[t][1], 8);
          break;
        default:
          std::abort();
        }
      }
    } else {
      for (int i = 0; i < vl_; i++) {
        Word mem_addr = rsdata[i][0] + (i * vtype_.vsew / 8);
        DP(4, "STORE MEM: ADDRESS=0x" << std::hex << mem_addr);
        switch (instr.getVlsWidth()) {
        case 6: {
          // store word and unit strided (not checking for unit stride)          
          uint32_t value = *(uint32_t *)(vreg_file_.at(instr.getVs3()).data() + i);
          core_->dcache_write(mem_addr, value, 4);
          DP(4, "store: " << mem_addr << " value:" << value);
        } break;
        default:
          std::abort();
        }          
      }
    }
    break;
  case SYS_INST:
    for (int t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      Word csr_addr = immsrc;
      Word csr_value;
      if (func3 == 0) {
        trace->exe_type = ExeType::ALU;
        trace->alu.type = AluType::SYSCALL;
        trace->fetch_stall = true;
        switch (csr_addr) {
        case 0: // RV32I: ECALL
          core_->trigger_ecall();
          break;
        case 1: // RV32I: EBREAK
          core_->trigger_ebreak();
          break;
        case 0x002: // URET
        case 0x102: // SRET
        case 0x302: // MRET
          break;
        default:
          std::abort();
        }                
      } else {
        trace->exe_type = ExeType::CSR;
        csr_value = core_->get_csr(csr_addr, t, id_);
        switch (func3) {
        case 1:
          // RV32I: CSRRW
          rddata[t] = csr_value;
          core_->set_csr(csr_addr, rsdata[t][0], t, id_);      
          trace->used_iregs.set(rsrc0);
          rd_write = true;
          break;
        case 2:
          // RV32I: CSRRS
          rddata[t] = csr_value;
          core_->set_csr(csr_addr, csr_value | rsdata[t][0], t, id_);
          trace->used_iregs.set(rsrc0);
          rd_write = true;
          break;
        case 3:
          // RV32I: CSRRC
          rddata[t] = csr_value;
          core_->set_csr(csr_addr, csr_value & ~rsdata[t][0], t, id_);
          trace->used_iregs.set(rsrc0);
          rd_write = true;
          break;
        case 5:
          // RV32I: CSRRWI
          rddata[t] = csr_value;
          core_->set_csr(csr_addr, rsrc0, t, id_);      
          rd_write = true;
          break;
        case 6:
          // RV32I: CSRRSI;
          rddata[t] = csr_value;
          core_->set_csr(csr_addr, csr_value | rsrc0, t, id_);
          rd_write = true;
          break;
        case 7:
          // RV32I: CSRRCI
          rddata[t] = csr_value;
          core_->set_csr(csr_addr, csr_value & ~rsrc0, t, id_);
          rd_write = true;
          break;
        default:
          break;
        }
      }
    } 
    break;
  case FENCE:
    trace->exe_type = ExeType::LSU;    
    trace->lsu.type = LsuType::FENCE;
    break;   
  case FCI:        
    trace->exe_type = ExeType::FPU;     
    for (int t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue; 
      uint32_t frm = get_fpu_rm(func3, core_, t, id_);
      uint32_t fflags = 0;
      switch (func7) {
      case 0x00: // RV32F: FADD.S
        rddata[t] = rv_fadd_s(rsdata[t][0], rsdata[t][1], frm, &fflags);
        trace->fpu.type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      case 0x01: // RV32D: FADD.D
        rddata[t] = rv_fadd_d(rsdata[t][0], rsdata[t][1], frm, &fflags);
        trace->fpu.type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      case 0x04: // RV32F: FSUB.S
        rddata[t] = rv_fsub_s(rsdata[t][0], rsdata[t][1], frm, &fflags);
        trace->fpu.type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      case 0x05: // RV32D: FSUB.D
        rddata[t] = rv_fsub_d(rsdata[t][0], rsdata[t][1], frm, &fflags);
        trace->fpu.type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      case 0x08: // RV32F: FMUL.S
        rddata[t] = rv_fmul_s(rsdata[t][0], rsdata[t][1], frm, &fflags);
        trace->fpu.type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      case 0x09: // RV32F: FMUL.D
        rddata[t] = rv_fmul_d(rsdata[t][0], rsdata[t][1], frm, &fflags);
        trace->fpu.type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      case 0x0c: // RV32F: FDIV.S
        rddata[t] = rv_fdiv_s(rsdata[t][0], rsdata[t][1], frm, &fflags);
        trace->fpu.type = FpuType::FDIV;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      case 0x0d: // RV32F: FDIV.D
        rddata[t] = rv_fdiv_d(rsdata[t][0], rsdata[t][1], frm, &fflags);
        trace->fpu.type = FpuType::FDIV;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      case 0x2c: // RV32F: FSQRT.S
        rddata[t] = rv_fsqrt_s(rsdata[t][0], frm, &fflags);
        trace->fpu.type = FpuType::FSQRT;
        trace->used_fregs.set(rsrc0);
        break;
      case 0x2d: // RV32D: FSQRT.D
        rddata[t] = rv_fsqrt_d(rsdata[t][0], frm, &fflags);
        trace->fpu.type = FpuType::FSQRT;
        trace->used_fregs.set(rsrc0);
        break;         
      case 0x10:
        switch (func3) {            
        case 0: // RV32F: FSGNJ.S
          rddata[t] = rv_fsgnj_s(rsdata[t][0], rsdata[t][1]);
          break;          
        case 1: // RV32F: FSGNJN.S
          rddata[t] = rv_fsgnjn_s(rsdata[t][0], rsdata[t][1]);
          break;          
        case 2: // RV32F: FSGNJX.S
          rddata[t] = rv_fsgnjx_s(rsdata[t][0], rsdata[t][1]);
          break;
        }
      case 0x11:
        switch (func3) {            
        case 0: // RV32F: FSGNJ.D
          rddata[t] = rv_fsgnj_d(rsdata[t][0], rsdata[t][1]);
          break;          
        case 1: // RV32F: FSGNJN.D
          rddata[t] = rv_fsgnjn_d(rsdata[t][0], rsdata[t][1]);
          break;          
        case 2: // RV32F: FSGNJX.D
          rddata[t] = rv_fsgnjx_d(rsdata[t][0], rsdata[t][1]);
          break;
        }
        trace->fpu.type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      case 0x14:              
        if (func3) {
          // RV32F: FMAX.S
          rddata[t] = rv_fmax_s(rsdata[t][0], rsdata[t][1], &fflags);
        } else {
          // RV32F: FMIN.S
          rddata[t] = rv_fmin_s(rsdata[t][0], rsdata[t][1], &fflags);
        }
        trace->fpu.type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);        
        break;
      case 0x15:              
        if (func3) {
          // RV32D: FMAX.D
          rddata[t] = rv_fmax_d(rsdata[t][0], rsdata[t][1], &fflags);
        } else {
          // RV32D: FMIN.D
          rddata[t] = rv_fmin_d(rsdata[t][0], rsdata[t][1], &fflags);
        }
        trace->fpu.type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);        
        break;
      case 0x60:
        switch(rsrc1) {
          case 0: 
            // RV32F: FCVT.W.S
            rddata[t] = sext64(rv_ftoi_s(rsdata[t][0], frm, &fflags), 32);
            break;
          case 1:
            // RV32F: FCVT.WU.S
            rddata[t] = sext64(rv_ftou_s(rsdata[t][0], frm, &fflags), 32);
            break;
          case 2:
            // RV64F: FCVT.L.S
            rddata[t] = rv_ftol_s(rsdata[t][0], frm, &fflags);
            break;
          case 3:
            // RV64F: FCVT.LU.S
            rddata[t] = rv_ftolu_s(rsdata[t][0], frm, &fflags);
            break;
        }
        trace->fpu.type = FpuType::FCVT;
        trace->used_fregs.set(rsrc0);
        break;
      case 0x61:
        switch(rsrc1) {
          case 0: 
            // RV32F: FCVT.W.D
            rddata[t] = sext64(rv_ftoi_d(rsdata[t][0], frm, &fflags), 32);
            break;
          case 1:
            // RV32F: FCVT.WU.D
            rddata[t] = sext64(rv_ftou_d(rsdata[t][0], frm, &fflags), 32);
            break;
          case 2:
            // RV64F: FCVT.L.D
            rddata[t] = rv_ftol_d(rsdata[t][0], frm, &fflags);
            break;
          case 3:
            // RV64F: FCVT.LU.D
            rddata[t] = rv_ftolu_d(rsdata[t][0], frm, &fflags);
            break;
        }
        trace->fpu.type = FpuType::FCVT;
        trace->used_fregs.set(rsrc0);
        break;
      case 0x70:      
        if (func3) {
          // RV32F: FCLASS.S
          rddata[t] = rv_fclss_s(rsdata[t][0]);
        } else {          
          // RV32F: FMV.X.W
          rddata[t] = rsdata[t][0];
        }        
        trace->fpu.type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        break;
      case 0x71:      
        if (func3) {
          // RV32D: FCLASS.D
          rddata[t] = rv_fclss_d(rsdata[t][0]);
        } else {          
          // RV64D: FMV.X.D
          rddata[t] = rsdata[t][0];
        }        
        trace->fpu.type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        break;
      case 0x50:             
        switch(func3) {              
        case 0:
          // RV32F: FLE.S
          rddata[t] = rv_fle_s(rsdata[t][0], rsdata[t][1], &fflags);    
          break;              
        case 1:
          // RV32F: FLT.S
          rddata[t] = rv_flt_s(rsdata[t][0], rsdata[t][1], &fflags);
          break;              
        case 2:
          // RV32F: FEQ.S
          rddata[t] = rv_feq_s(rsdata[t][0], rsdata[t][1], &fflags);
          break;
        } 
        trace->fpu.type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;  
      case 0x51:             
        switch(func3) {              
        case 0:
          // RV32D: FLE.D
          rddata[t] = rv_fle_d(rsdata[t][0], rsdata[t][1], &fflags);    
          break;              
        case 1:
          // RV32D: FLT.D
          rddata[t] = rv_flt_d(rsdata[t][0], rsdata[t][1], &fflags);
          break;              
        case 2:
          // RV32D: FEQ.D
          rddata[t] = rv_feq_d(rsdata[t][0], rsdata[t][1], &fflags);
          break;
        } 
        trace->fpu.type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;        
      case 0x68:
        switch(rsrc1) {
          case 0: 
            // RV32F: FCVT.S.W
            rddata[t] = rv_itof_s(rsdata[t][0], frm, &fflags);
            break;
          case 1:
            // RV32F: FCVT.S.WU
            rddata[t] = rv_utof_s(rsdata[t][0], frm, &fflags);
            break;
          case 2:
            // RV64F: FCVT.S.L
            rddata[t] = rv_ltof_s(rsdata[t][0], frm, &fflags);
            break;
          case 3:
            // RV64F: FCVT.S.LU
            rddata[t] = rv_lutof_s(rsdata[t][0], frm, &fflags);
            break;
        }
        trace->fpu.type = FpuType::FCVT;
        trace->used_iregs.set(rsrc0);
        break;
      case 0x69:
        switch(rsrc1) {
          case 0: 
            // RV32D: FCVT.D.W
            rddata[t] = rv_itof_d(rsdata[t][0], frm, &fflags);
            break;
          case 1:
            // RV32F: FCVT.D.WU
            rddata[t] = rv_utof_d(rsdata[t][0], frm, &fflags);
            break;
          case 2:
            // RV64D: FCVT.D.L
            rddata[t] = rv_ltof_d(rsdata[t][0], frm, &fflags);
            break;
          case 3:
            // RV64D: FCVT.D.LU
            rddata[t] = rv_lutof_d(rsdata[t][0], frm, &fflags);
            break;
        }
        trace->fpu.type = FpuType::FCVT;
        trace->used_iregs.set(rsrc0);
        break;
      case 0x78:  // FMV.W.X
      case 0x79:  // FMV.D.X
        rddata[t] = rsdata[t][0];
        trace->fpu.type = FpuType::FNCP;
        trace->used_iregs.set(rsrc0);
        break;
      }
      update_fcrs(fflags, core_, t, id_);
    }
    rd_write = true;
    break;
  case FMADD:      
  case FMSUB:      
  case FMNMADD:
  case FMNMSUB: 
    trace->fpu.type = FpuType::FMA;
    trace->used_fregs.set(rsrc0);
    trace->used_fregs.set(rsrc1);
    trace->used_fregs.set(rsrc2);
    for (int t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      int frm = get_fpu_rm(func3, core_, t, id_);
      Word fflags = 0;
      switch (opcode) {
      case FMADD:
        if (func2)
          // RV32D: FMADD.D
          rddata[t] = rv_fmadd_d(rsdata[t][0], rsdata[t][1], rsdata[t][2], frm, &fflags);
        else
          // RV32F: FMADD.S
          rddata[t] = rv_fmadd_s(rsdata[t][0], rsdata[t][1], rsdata[t][2], frm, &fflags);
        break;
      case FMSUB:
        if (func2)
          // RV32D: FMSUB.D
          rddata[t] = rv_fmsub_d(rsdata[t][0], rsdata[t][1], rsdata[t][2], frm, &fflags);
        else 
          // RV32F: FMSUB.S
          rddata[t] = rv_fmsub_s(rsdata[t][0], rsdata[t][1], rsdata[t][2], frm, &fflags);
        break;
      case FMNMADD:
        if (func2)
          // RV32D: FNMADD.D
          rddata[t] = rv_fnmadd_d(rsdata[t][0], rsdata[t][1], rsdata[t][2], frm, &fflags);
        else
          // RV32F: FNMADD.S
          rddata[t] = rv_fnmadd_s(rsdata[t][0], rsdata[t][1], rsdata[t][2], frm, &fflags);
        break; 
      case FMNMSUB:
        if (func2)
          // RV32D: FNMSUB.D
          rddata[t] = rv_fnmsub_d(rsdata[t][0], rsdata[t][1], rsdata[t][2], frm, &fflags);
        else
          // RV32F: FNMSUB.S
          rddata[t] = rv_fnmsub_s(rsdata[t][0], rsdata[t][1], rsdata[t][2], frm, &fflags);
        break;
      default:
        break;
      }              
      update_fcrs(fflags, core_, t, id_);
    }
    rd_write = true;
    break;
  case GPGPU: {    
    int ts = 0;
    for (int t = 0; t < num_threads; ++t) {
      if (tmask_.test(t)) {
        ts = t;
        break;
      }
    }
    switch (func3) {
    case 0: {
      // TMC   
      trace->exe_type = ExeType::GPU;     
      trace->gpu.type = GpuType::TMC;
      trace->used_iregs.set(rsrc0);
      trace->fetch_stall = true;
      if (rsrc1) {
        // predicate mode
        ThreadMask pred;
        for (int i = 0; i < num_threads; ++i) {
          pred[i] = tmask_.test(i) ? (ireg_file_.at(i).at(rsrc0) != 0) : 0;
        }
        if (pred.any()) {
          tmask_ &= pred;
        }
      } else {
        tmask_.reset();
        for (int i = 0; i < num_threads; ++i) {
          tmask_.set(i, rsdata.at(ts)[0] & (1 << i));
        }
      }
      DPH(3, "*** New TMC: ");
      for (int i = 0; i < num_threads; ++i)
        DPN(3, tmask_.test(num_threads-i-1));
      DPN(3, std::endl);

      active_ = tmask_.any();
      trace->gpu.active_warps.reset();
      trace->gpu.active_warps.set(id_, active_);
    } break;
    case 1: {
      // WSPAWN
      trace->exe_type = ExeType::GPU;
      trace->gpu.type = GpuType::WSPAWN;
      trace->used_iregs.set(rsrc0);
      trace->used_iregs.set(rsrc1);
      trace->fetch_stall = true;
      trace->gpu.active_warps = core_->wspawn(rsdata.at(ts)[0], rsdata.at(ts)[1]);
    } break;
    case 2: {
      // SPLIT    
      trace->exe_type = ExeType::GPU;
      trace->gpu.type = GpuType::SPLIT;
      trace->used_iregs.set(rsrc0);
      trace->fetch_stall = true;
      if (HasDivergentThreads(tmask_, ireg_file_, rsrc0)) {          
        ThreadMask tmask;
        for (int i = 0; i < num_threads; ++i) {
          tmask[i] = tmask_.test(i) && !ireg_file_.at(i).at(rsrc0);
        }

        DomStackEntry e(tmask, nextPC);
        dom_stack_.push(tmask_);
        dom_stack_.push(e);
        for (size_t i = 0; i < e.tmask.size(); ++i) {
          tmask_.set(i, !e.tmask.test(i) && tmask_.test(i));
        }
        active_ = tmask_.any();

        DPH(3, "*** Split: New TM=");
        for (int i = 0; i < num_threads; ++i) DPN(3, tmask_.test(num_threads-i-1));
        DPN(3, ", Pushed TM=");
        for (int i = 0; i < num_threads; ++i) DPN(3, e.tmask.test(num_threads-i-1));
        DPN(3, ", PC=0x" << std::hex << e.PC << "\n");
      } else {
        DP(3, "*** Unanimous pred");
        DomStackEntry e(tmask_);
        e.unanimous = true;
        dom_stack_.push(e);
      }        
    } break;
    case 3: {
      // JOIN
      trace->exe_type = ExeType::GPU;
      trace->gpu.type = GpuType::JOIN;        
      trace->fetch_stall = true;        
      if (!dom_stack_.empty() && dom_stack_.top().unanimous) {
        DP(3, "*** Uninimous branch at join");
        tmask_ = dom_stack_.top().tmask;
        active_ = tmask_.any();
        dom_stack_.pop();
      } else {
        if (!dom_stack_.top().fallThrough) {
          nextPC = dom_stack_.top().PC;
          DP(3, "*** Join: next PC: " << std::hex << nextPC << std::dec);
        }

        tmask_ = dom_stack_.top().tmask;
        active_ = tmask_.any();

        DPH(3, "*** Join: New TM=");
        for (int i = 0; i < num_threads; ++i) DPN(3, tmask_.test(num_threads-i-1));
        DPN(3, "\n");

        dom_stack_.pop();
      }        
    } break;
    case 4: {
      // BAR
      trace->exe_type = ExeType::GPU; 
      trace->gpu.type = GpuType::BAR;
      trace->used_iregs.set(rsrc0);
      trace->used_iregs.set(rsrc1);
      trace->fetch_stall = true;
      trace->gpu.active_warps = core_->barrier(rsdata[ts][0], rsdata[ts][1], id_);
    } break;
    case 5: {
      // PREFETCH
      trace->exe_type = ExeType::LSU; 
      trace->lsu.type = LsuType::PREFETCH; 
      trace->used_iregs.set(rsrc0);
      for (int t = 0; t < num_threads; ++t) {
        if (!tmask_.test(t))
          continue;
        auto mem_addr = rsdata[t][0];
        trace->mem_addrs.at(t).push_back({mem_addr, 4});
      }
    } break;
    default:
      std::abort();
    }
  }  break;
  case GPU: {    
    switch (func3) {
    case 0: { // TEX
      trace->exe_type = ExeType::GPU; 
      trace->gpu.type = GpuType::TEX;
      trace->used_iregs.set(rsrc0);
      trace->used_iregs.set(rsrc1);
      trace->used_iregs.set(rsrc2);
      for (int t = 0; t < num_threads; ++t) {
        if (!tmask_.test(t))
          continue;        
        auto unit  = func2;
        auto u     = rsdata[t][0];
        auto v     = rsdata[t][1];
        auto lod   = rsdata[t][2];
        auto color = core_->tex_read(unit, u, v, lod, &trace->mem_addrs.at(t));
        rddata[t] = color;
      }
      rd_write = true;
    } break;
    case 1: 
      switch (func2) {
      case 0: { // CMOV
        trace->exe_type = ExeType::ALU;
        trace->alu.type = AluType::CMOV;
        trace->used_iregs.set(rsrc0);
        trace->used_iregs.set(rsrc1);
        trace->used_iregs.set(rsrc2);
        for (int t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t))
            continue;     
          rddata[t] = rsdata[t][0] ? rsdata[t][1] : rsdata[t][2];
        }
        rd_write = true;
      } break;
      default:
        std::abort();
      }
      break;
    default:
      std::abort();
    }
  } break;
  case VSET: {
    int VLEN = core_->arch().vsize() * 8;
    int VLMAX = (instr.getVlmul() * VLEN) / instr.getVsew();
    switch (func3) {
    case 0: // vector-vector
      switch (func6) {
      case 0: {
        auto& vr1 = vreg_file_.at(rsrc0);
        auto& vr2 = vreg_file_.at(rsrc1);
        auto& vd = vreg_file_.at(rdest);
        auto& mask = vreg_file_.at(0);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t emask = *(uint8_t *)(mask.data() + i);
            uint8_t value = emask & 0x1;
            if (vmask || (!vmask && value)) {
              uint8_t first  = *(uint8_t *)(vr1.data() + i);
              uint8_t second = *(uint8_t *)(vr2.data() + i);
              uint8_t result = first + second;
              DP(3, "Adding " << first << " + " << second << " = " << result);
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
              DP(3, "Adding " << first << " + " << second << " = " << result);
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
              DP(3, "Adding " << first << " + " << second << " = " << result);
              *(uint32_t *)(vd.data() + i) = result;
            }
          }
        }                
      } break;
      case 24: {
        // vmseq
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first == second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (int i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first == second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (int i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (first == second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
        }
      } break;
      case 25: { 
        // vmsne
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first != second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (int i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first != second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (int i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (first != second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
        }
      } break;
      case 26: {
        // vmsltu
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first < second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (int i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first < second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (int i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (first < second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
        }
      } break;
      case 27: {
        // vmslt
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            int8_t first  = *(int8_t *)(vr1.data() + i);
            int8_t second = *(int8_t *)(vr2.data() + i);
            int8_t result = (first < second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (int i = 0; i < vl_; i++) {
            int16_t first  = *(int16_t *)(vr1.data() + i);
            int16_t second = *(int16_t *)(vr2.data() + i);
            int16_t result = (first < second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(int16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (int i = 0; i < vl_; i++) {
            int32_t first  = *(int32_t *)(vr1.data() + i);
            int32_t second = *(int32_t *)(vr2.data() + i);
            int32_t result = (first < second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(int32_t *)(vd.data() + i) = result;
          }
        }
      } break;
      case 28: {
        // vmsleu
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first <= second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (int i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first <= second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (int i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (first <= second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
        }
      } break;
      case 29: {
        // vmsle
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            int8_t first  = *(int8_t *)(vr1.data() + i);
            int8_t second = *(int8_t *)(vr2.data() + i);
            int8_t result = (first <= second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (int i = 0; i < vl_; i++) {
            int16_t first  = *(int16_t *)(vr1.data() + i);
            int16_t second = *(int16_t *)(vr2.data() + i);
            int16_t result = (first <= second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(int16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (int i = 0; i < vl_; i++) {
            int32_t first  = *(int32_t *)(vr1.data() + i);
            int32_t second = *(int32_t *)(vr2.data() + i);
            int32_t result = (first <= second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(int32_t *)(vd.data() + i) = result;
          }
        }
      } break;
      case 30: {
        // vmsgtu
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first > second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (int i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first > second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (int i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (first > second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
        }
      } break;
      case 31: {
        // vmsgt
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            int8_t first  = *(int8_t *)(vr1.data() + i);
            int8_t second = *(int8_t *)(vr2.data() + i);
            int8_t result = (first > second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (int i = 0; i < vl_; i++) {
            int16_t first  = *(int16_t *)(vr1.data() + i);
            int16_t second = *(int16_t *)(vr2.data() + i);
            int16_t result = (first > second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(int16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (int i = 0; i < vl_; i++) {
            int32_t first  = *(int32_t *)(vr1.data() + i);
            int32_t second = *(int32_t *)(vr2.data() + i);
            int32_t result = (first > second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value & !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (int i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 25: {
        // vmand
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (int i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 26: {
        // vmor
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (int i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 27: { 
        // vmxor
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (int i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 28: {
        // vmornot
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value | !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (int i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 29: {
        // vmnand
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = !(first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (int i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 30: {
        // vmnor
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = !(first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (int i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 31: {
        // vmxnor
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = !(first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (int i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 37: {
        // vmul
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (int i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 45: {
        // vmacc
        auto &vr1 = vreg_file_.at(rsrc0);
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
            DP(3, "Comparing " << first << " + " << second << " = " << result);
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
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (rsdata[i][0] + second);
            DP(3, "Comparing " << rsdata[i][0] << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (int i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (int i = 0; i < vl_; i++) {
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (rsdata[i][0] + second);
            DP(3, "Comparing " << rsdata[i][0] << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (int i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (int i = 0; i < vl_; i++) {
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (rsdata[i][0] + second);
            DP(3, "Comparing " << rsdata[i][0] << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (int i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 37: {
        // vmul.vx
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (int i = 0; i < vl_; i++) {
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (rsdata[i][0] * second);
            DP(3, "Comparing " << rsdata[i][0] << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (int i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (int i = 0; i < vl_; i++) {
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (rsdata[i][0] * second);
            DP(3, "Comparing " << rsdata[i][0] << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (int i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (int i = 0; i < vl_; i++) {
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (rsdata[i][0] * second);
            DP(3, "Comparing " << rsdata[i][0] << " + " << second << " = " << result);
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

      DP(3, "lmul:" << vtype_.vlmul << " sew:" << vtype_.vsew  << " ediv: " << vtype_.vediv << "rsrc_" << rsdata[0][0] << "VLMAX" << VLMAX);

      int s0 = rsdata[0][0];
      if (s0 <= VLMAX) {
        vl_ = s0;
      } else if (s0 < (2 * VLMAX)) {
        vl_ = (int)ceil((s0 * 1.0) / 2.0);
      } else if (s0 >= (2 * VLMAX)) {
        vl_ = VLMAX;
      }        
      rddata[0] = vl_;
    } break;
    default:
      std::abort();
    }
  } break;    
  default:
    std::abort();
  }

  if (rd_write) {
    trace->wb = true;
    DPH(2, "Dest Reg: ");
    auto rdt = instr.getRDType();    
    switch (rdt) {
    case RegType::Integer:      
      if (rdest) {    
        DPN(2, "r" << std::dec << rdest << "={");    
        for (int t = 0; t < num_threads; ++t) {
          if (t) DPN(2, ", ");
          if (!tmask_.test(t)) {
            DPN(2, "-");
            continue;            
          }
          ireg_file_.at(t)[rdest] = rddata[t];
          DPN(2, "0x" << std::hex << rddata[t]);         
        }
        DPN(2, "}" << std::endl);
        trace->used_iregs[rdest] = 1;
      }
      break;
    case RegType::Float:
      DPN(2, "fr" << std::dec << rdest << "={");
      for (int t = 0; t < num_threads; ++t) {
        if (t) DPN(2, ", ");
        if (!tmask_.test(t)) {
          DPN(2, "-");
          continue;            
        }
        freg_file_.at(t)[rdest] = rddata[t];        
        DPN(2, "0x" << std::hex << rddata[t]);         
      }
      DPN(2, "}" << std::endl);
      trace->used_fregs[rdest] = 1;
      break;
    default:
      std::abort();
      break;
    }
  }

  PC_ += core_->arch().wsize();
  if (PC_ != nextPC) {
    DP(3, "*** Next PC: " << std::hex << nextPC << std::dec);
    PC_ = nextPC;
  }
}