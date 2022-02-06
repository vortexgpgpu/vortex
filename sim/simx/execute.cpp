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

union reg_data_t {
  Word     i;  
  FWord    f;
  uint64_t _;
};

static bool HasDivergentThreads(const ThreadMask &thread_mask,                                
                                const std::vector<std::vector<Word>> &reg_file,
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

static void update_fcrs(uint32_t fflags, Core* core, uint32_t tid, uint32_t wid) {
  if (fflags) {
    core->set_csr(CSR_FCSR, core->get_csr(CSR_FCSR, tid, wid) | fflags, tid, wid);
    core->set_csr(CSR_FFLAGS, core->get_csr(CSR_FFLAGS, tid, wid) | fflags, tid, wid);
  }
}

inline uint64_t nan_box(uint32_t value) {
  uint64_t mask = 0xffffffff00000000;
  return value | mask;
}

inline bool is_nan_boxed(uint64_t value) {
  return (uint32_t(value >> 32) == 0xffffffff);
}

static bool checkBoxedArgs(FWord* out, FWord a, FWord b, uint32_t* fflags) {  
  bool xa = is_nan_boxed(a);
  bool xb = is_nan_boxed(b);  
  if (xa && xb)
    return true;
  if (xa) {
    // a is NaN boxed but b isn't
    *out = nan_box((uint32_t)a);
  } else if (xb) {
    // b is NaN boxed but a isn't
    *out = nan_box(0xffc00000);
  } else {
    // Both a and b aren't NaN boxed
    *out = nan_box(0x7fc00000);
  }
  *fflags = 0;
  return false;
}

static bool checkBoxedArgs(FWord* out, FWord a, uint32_t* fflags) {  
  bool xa = is_nan_boxed(a);
  if (xa)
    return true;
  *out = nan_box(0x7fc00000);
  *fflags = 0;
  return false;
}

static bool checkBoxedCmpArgs(Word* out, FWord a, FWord b, uint32_t* fflags) {  
  bool xa = is_nan_boxed(a);
  bool xb = is_nan_boxed(b);  
  if (xa && xb)
    return true;
  *out = 0;
  *fflags = 0;
  return false;
}

void Warp::execute(const Instr &instr, pipeline_trace_t *trace) {
  assert(tmask_.any());

  auto nextPC = PC_ + core_->arch().wsize();

  auto func2  = instr.getFunc2();
  auto func3  = instr.getFunc3();
  auto func6  = instr.getFunc6();
  auto func7  = instr.getFunc7();

  auto opcode = instr.getOpcode();
  auto rdest  = instr.getRDest();
  auto rsrc0  = instr.getRSrc(0);
  auto rsrc1  = instr.getRSrc(1);
  auto rsrc2  = instr.getRSrc(2);
  auto immsrc = sext((Word)instr.getImm(), 32);
  auto vmask  = instr.getVmask();

  auto num_threads = core_->arch().num_threads();

  std::vector<reg_data_t[3]> rsdata(num_threads);
  std::vector<reg_data_t> rddata(num_threads);

  auto num_rsrcs = instr.getNRSrc();
  if (num_rsrcs) {              
    for (uint32_t i = 0; i < num_rsrcs; ++i) {    
      DPH(2, "Src Reg [" << std::dec << i << "]: ");
      auto type = instr.getRSType(i);
      auto reg = instr.getRSrc(i);        
      switch (type) {
      case RegType::Integer: 
        DPN(2, type << std::dec << reg << "={");
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (t) DPN(2, ", ");
          if (!tmask_.test(t)) {
            DPN(2, "-");
            continue;            
          }
          rsdata[t][i].i = ireg_file_.at(t)[reg];          
          DPN(2, std::hex << rsdata[t][i].i); 
        }
        DPN(2, "}" << std::endl);
        break;
      case RegType::Float: 
        DPN(2, type << std::dec << reg << "={");
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (t) DPN(2, ", ");
          if (!tmask_.test(t)) {
            DPN(2, "-");
            continue;            
          }
          rsdata[t][i].f = freg_file_.at(t)[reg];
          DPN(2, std::hex << rsdata[t][i].f); 
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
  // RV32I: LUI
  case LUI_INST: {
    trace->exe_type = ExeType::ALU;
    trace->alu.type = AluType::ARITH;
    for (uint32_t t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      rddata[t].i = immsrc << 12;
    }    
    rd_write = true;
    break;
  }
  // RV32I: AUIPC
  case AUIPC_INST: {
    trace->exe_type = ExeType::ALU;
    trace->alu.type = AluType::ARITH;
    for (uint32_t t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      rddata[t].i = (immsrc << 12) + PC_;
    }    
    rd_write = true;
    break;
  }
  case R_INST: {
    trace->exe_type = ExeType::ALU;    
    trace->alu.type = AluType::ARITH;
    trace->used_iregs.set(rsrc0);
    trace->used_iregs.set(rsrc1);
    for (uint32_t t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      if (func7 & 0x1) {
        switch (func3) {
        case 0: {
          // RV32M: MUL
          rddata[t].i = (WordI)rsdata[t][0].i * (WordI)rsdata[t][1].i;
          trace->alu.type = AluType::IMUL;
          break;
        }
        case 1: {
          // RV32M: MULH
          DWordI first = sext((DWord)rsdata[t][0].i, XLEN);
          DWordI second = sext((DWord)rsdata[t][1].i, XLEN);
          rddata[t].i = (first * second) >> XLEN;
          trace->alu.type = AluType::IMUL;
          break;
        }
        case 2: {
          // RV32M: MULHSU       
          DWordI first = sext((DWord)rsdata[t][0].i, XLEN);
          DWord second = rsdata[t][1].i;
          rddata[t].i = (first * second) >> XLEN;
          trace->alu.type = AluType::IMUL;
          break;
        } 
        case 3: {
          // RV32M: MULHU
          DWord first = rsdata[t][0].i;
          DWord second = rsdata[t][1].i;
          rddata[t].i = (first * second) >> XLEN;
          trace->alu.type = AluType::IMUL;
          break;
        } 
        case 4: {
          // RV32M: DIV
          WordI dividen = rsdata[t][0].i;
          WordI divisor = rsdata[t][1].i; 
          WordI largest_negative = WordI(1) << (XLEN-1);  
          if (divisor == 0) {
            rddata[t].i = -1;
          } else if (dividen == largest_negative && divisor == -1) {
            rddata[t].i = dividen;
          } else {
            rddata[t].i = dividen / divisor;
          }
          trace->alu.type = AluType::IDIV;
          break;
        } 
        case 5: {
          // RV32M: DIVU
          Word dividen = rsdata[t][0].i;
          Word divisor = rsdata[t][1].i;
          if (divisor == 0) {
            rddata[t].i = -1;
          } else {
            rddata[t].i = dividen / divisor;
          }
          trace->alu.type = AluType::IDIV;
          break;
        } 
        case 6: {
          // RV32M: REM
          WordI dividen = rsdata[t][0].i;
          WordI divisor = rsdata[t][1].i;
          WordI largest_negative = WordI(1) << (XLEN-1);
          if (rsdata[t][1].i == 0) {
            rddata[t].i = dividen;
          } else if (dividen == largest_negative && divisor == -1) {
            rddata[t].i = 0;
          } else {
            rddata[t].i = dividen % divisor;
          }
          trace->alu.type = AluType::IDIV;
          break;
        } 
        case 7: {
          // RV32M: REMU
          Word dividen = rsdata[t][0].i;
          Word divisor = rsdata[t][1].i;
          if (rsdata[t][1].i == 0) {
            rddata[t].i = dividen;
          } else {
            rddata[t].i = dividen % divisor;
          }
          trace->alu.type = AluType::IDIV;
          break;
        } 
        default:
          std::abort();
        }
      } else {
        switch (func3) {
        case 0: {
          if (func7) {
            // RV32I: SUB
            rddata[t].i = rsdata[t][0].i - rsdata[t][1].i;
          } else {
            // RV32I: ADD
            rddata[t].i = rsdata[t][0].i + rsdata[t][1].i;
          }
          break;
        }
        case 1: {
          // RV32I: SLL
          Word shamt_mask = (Word(1) << log2up(XLEN)) - 1;
          Word shamt = rsdata[t][1].i & shamt_mask;
          rddata[t].i = rsdata[t][0].i << shamt;
          break;
        }
        case 2: {
          // RV32I: SLT
          rddata[t].i = WordI(rsdata[t][0].i) < WordI(rsdata[t][1].i);
          break;
        }
        case 3: {
          // RV32I: SLTU
          rddata[t].i = Word(rsdata[t][0].i) < Word(rsdata[t][1].i);
          break;
        }
        case 4: {
          // RV32I: XOR
          rddata[t].i = rsdata[t][0].i ^ rsdata[t][1].i;
          break;
        }
        case 5: {
          Word shamt_mask = ((Word)1 << log2up(XLEN)) - 1;
          Word shamt = rsdata[t][1].i & shamt_mask;
          if (func7) {
            // RV32I: SRA
            rddata[t].i = WordI(rsdata[t][0].i) >> shamt;
          } else {
            // RV32I: SRL
            rddata[t].i = Word(rsdata[t][0].i) >> shamt;
          }
          break;
        }
        case 6: {
          // RV32I: OR
          rddata[t].i = rsdata[t][0].i | rsdata[t][1].i;
          break;
        }
        case 7: {
          // RV32I: AND
          rddata[t].i = rsdata[t][0].i & rsdata[t][1].i;
          break;
        }
        default:
          std::abort();
        }
      }
    }    
    rd_write = true;
    break;
  }
  case I_INST: {
    trace->exe_type = ExeType::ALU;    
    trace->alu.type = AluType::ARITH;    
    trace->used_iregs.set(rsrc0);
    for (uint32_t t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      switch (func3) {
      case 0: {
        // RV32I: ADDI
        rddata[t].i = rsdata[t][0].i + immsrc;
        break;
      }
      case 1: {
        // RV64I: SLLI
        rddata[t].i = rsdata[t][0].i << immsrc;
        break;
      }
      case 2: {
        // RV32I: SLTI
        rddata[t].i = WordI(rsdata[t][0].i) < WordI(immsrc);
        break;
      }
      case 3: {
        // RV32I: SLTIU
        rddata[t].i = rsdata[t][0].i < immsrc;
        break;
      } 
      case 4: {
        // RV32I: XORI
        rddata[t].i = rsdata[t][0].i ^ immsrc;
        break;
      }
      case 5: {
        if (func7) {
          // RV64I: SRAI
          Word result = WordI(rsdata[t][0].i) >> immsrc;
          rddata[t].i = result;
        } else {
          // RV64I: SRLI
          Word result = Word(rsdata[t][0].i) >> immsrc;
          rddata[t].i = result;
        }
        break;
      }
      case 6: {
        // RV32I: ORI
        rddata[t].i = rsdata[t][0].i | immsrc;
        break;
      }
      case 7: {
        // RV32I: ANDI
        rddata[t].i = rsdata[t][0].i & immsrc;
        break;
      }
      }
    }
    rd_write = true;
    break;
  }
  case R_INST_W: {
    trace->exe_type = ExeType::ALU;    
    trace->alu.type = AluType::ARITH;
    trace->used_iregs.set(rsrc0);
    trace->used_iregs.set(rsrc1);
    for (uint32_t t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      if (func7 & 0x1){
        switch (func3) {
          case 0: {
            // RV64M: MULW
            int32_t product = (int32_t)rsdata[t][0].i * (int32_t)rsdata[t][1].i;
            rddata[t].i = sext((uint64_t)product, 32);
            trace->alu.type = AluType::IMUL;
            break;
          }
          case 4: {
            // RV64M: DIVW
            int32_t dividen = (int32_t)rsdata[t][0].i;
            int32_t divisor = (int32_t)rsdata[t][1].i;
            int32_t quotient;
            int32_t largest_negative = 0x80000000;
            if (divisor == 0){
              quotient = -1;
            } else if (dividen == largest_negative && divisor == -1) {
              quotient = dividen;
            } else {
              quotient = dividen / divisor;
            }
            rddata[t].i = sext((uint64_t)quotient, 32);
            trace->alu.type = AluType::IDIV;
            break;
          }      
          case 5: {
            // RV64M: DIVUW
            uint32_t dividen = (uint32_t)rsdata[t][0].i;
            uint32_t divisor = (uint32_t)rsdata[t][1].i;
            uint32_t quotient;
            if (divisor == 0){
              quotient = -1;
            } else {
              quotient = dividen / divisor;
            }
            rddata[t].i = sext((uint64_t)quotient, 32);
            trace->alu.type = AluType::IDIV;
            break;
          } 
          case 6: {
            // RV64M: REMW
            int32_t dividen = (uint32_t)rsdata[t][0].i;
            int32_t divisor = (uint32_t)rsdata[t][1].i;
            int32_t remainder;
            int32_t largest_negative = 0x80000000;
            if (divisor == 0){
              remainder = dividen;
            } else if (dividen == largest_negative && divisor == -1) {
              remainder = 0;
            } else {
              remainder = dividen % divisor;
            }
            rddata[t].i = sext((uint64_t)remainder, 32);
            trace->alu.type = AluType::IDIV;
            break;
          }  
          case 7: {
            // RV64M: REMUW
            uint32_t dividen = (uint32_t)rsdata[t][0].i;
            uint32_t divisor = (uint32_t)rsdata[t][1].i;
            uint32_t remainder;
            if (divisor == 0){
              remainder = dividen;
            } else {
              remainder = dividen % divisor;
            }
            rddata[t].i = sext((uint64_t)remainder, 32);
            trace->alu.type = AluType::IDIV;
            break;
          }  
          default:
            std::abort();
        }
      } else {
        switch (func3) {
        case 0: {
          if (func7){
            // RV64I: SUBW
            uint32_t result = (uint32_t)rsdata[t][0].i - (uint32_t)rsdata[t][1].i;
            rddata[t].i = sext((uint64_t)result, 32);
          }
          else{
            // RV64I: ADDW
            uint32_t result = (uint32_t)rsdata[t][0].i + (uint32_t)rsdata[t][1].i;
            rddata[t].i = sext((uint64_t)result, 32);
          }    
          break;
        }
        case 1: {
          // RV64I: SLLW
          uint32_t shamt_mask = 0x1F;
          uint32_t shamt = rsdata[t][1].i & shamt_mask;
          uint32_t result = (uint32_t)rsdata[t][0].i << shamt;
          rddata[t].i = sext((uint64_t)result, 32);
          break;
        }
        case 5: {
          uint32_t shamt_mask = 0x1F;
          uint32_t shamt = rsdata[t][1].i & shamt_mask;
          uint32_t result;
          if (func7) {
            // RV64I: SRAW
            result = (int32_t)rsdata[t][0].i >> shamt;
          } else {
            // RV64I: SRLW
            result = (uint32_t)rsdata[t][0].i >> shamt;
          }
          rddata[t].i = sext((uint64_t)result, 32);
          break;
        }
        default:
          std::abort();
        }
      }
    }
    rd_write = true;
    break; 
  }
  case I_INST_W: {
    trace->exe_type = ExeType::ALU;    
    trace->alu.type = AluType::ARITH;    
    trace->used_iregs.set(rsrc0);
    for (uint32_t t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      switch (func3) {
        case 0: {
          // RV64I: ADDIW
          uint32_t result = (uint32_t)rsdata[t][0].i + (uint32_t)immsrc;
          rddata[t].i = sext((uint64_t)result, 32);
          break;
        }
        case 1: {
          // RV64I: SLLIW
          uint32_t shamt_mask = 0x1F;
          uint32_t shamt = immsrc & shamt_mask;
          uint32_t result = rsdata[t][0].i << shamt;
          rddata[t].i = sext((uint64_t)result, 32);
          break;
        }  
        case 5: {
          uint32_t shamt_mask = 0x1F;
          uint32_t shamt = immsrc & shamt_mask;
          uint32_t result;
          if (func7) {
            // RV64I: SRAIW
            result = (int32_t)rsdata[t][0].i >> shamt;
          } else {
            // RV64I: SRLIW
            result = (uint32_t)rsdata[t][0].i >> shamt;
          }
          rddata[t].i = sext((uint64_t)result, 32);
          break;
        }
        default:
          std::abort();
      }
    }
    rd_write = true;
    break;
  }
  case B_INST: {   
    trace->exe_type = ExeType::ALU;    
    trace->alu.type = AluType::BRANCH;    
    trace->used_iregs.set(rsrc0);
    trace->used_iregs.set(rsrc1);
    for (uint32_t t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      switch (func3) {
      case 0: {
        // RV32I: BEQ
        if (rsdata[t][0].i == rsdata[t][1].i) {
          nextPC = uint32_t(PC_ + immsrc);
        }
        break;
      }
      case 1: {
        // RV32I: BNE
        if (rsdata[t][0].i != rsdata[t][1].i) {
          nextPC = uint32_t(PC_ + immsrc);
        }
        break;
      }
      case 4: {
        // RV32I: BLT
        if (WordI(rsdata[t][0].i) < WordI(rsdata[t][1].i)) {
          nextPC = uint32_t(PC_ + immsrc);
        }
        break;
      }
      case 5: {
        // RV32I: BGE
        if (WordI(rsdata[t][0].i) >= WordI(rsdata[t][1].i)) {
          nextPC = uint32_t(PC_ + immsrc);
        }
        break;
      }
      case 6: {
        // RV32I: BLTU
        if (Word(rsdata[t][0].i) < Word(rsdata[t][1].i)) {
          nextPC = uint32_t(PC_ + immsrc);
        }
        break;
      }
      case 7: {
        // RV32I: BGEU
        if (Word(rsdata[t][0].i) >= Word(rsdata[t][1].i)) {
          nextPC = uint32_t(PC_ + immsrc);
        }
        break;
      }
      default:
        std::abort();
      }
      break; // runonce
    }
    trace->fetch_stall = true;
    break;
  }
  // RV32I: JAL
  case JAL_INST: {
    trace->exe_type = ExeType::ALU;    
    trace->alu.type = AluType::BRANCH;
    for (uint32_t t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      rddata[t].i = nextPC;
      nextPC = uint32_t(PC_ + immsrc);  
      trace->fetch_stall = true;
      break; // runonce
    }
    rd_write = true;
    break;
  }
  // RV32I: JALR
  case JALR_INST: {
    trace->exe_type = ExeType::ALU;    
    trace->alu.type = AluType::BRANCH;
    trace->used_iregs.set(rsrc0);
    for (uint32_t t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      rddata[t].i = nextPC;
      nextPC = uint32_t(rsdata[t][0].i + immsrc);
      trace->fetch_stall = true;
      break; // runOnce
    }
    rd_write = true;
    break;
  }
  case L_INST:
  case FL: {
    trace->exe_type = ExeType::LSU;    
    trace->lsu.type = LsuType::LOAD;
    trace->used_iregs.set(rsrc0);
    if ((opcode == L_INST )
     || (opcode == FL && func3 == 2)
     || (opcode == FL && func3 == 3)) {
      uint32_t mem_bytes = 1 << (func3 & 0x3);
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!tmask_.test(t))
          continue;
        uint64_t mem_addr = rsdata[t][0].i + immsrc;         
        uint64_t mem_data = 0;
        core_->dcache_read(&mem_data, mem_addr, mem_bytes);
        trace->mem_addrs.at(t).push_back({mem_addr, mem_bytes});        
        DP(4, "LOAD MEM: ADDRESS=0x" << std::hex << mem_addr << ", DATA=0x" << mem_data);
        switch (func3) {
        case 0:
          // RV32I: LB
          rddata[t].i = sext((Word)mem_data, 8);
          break;
        case 1:
          // RV32I: LH
          rddata[t].i = sext((Word)mem_data, 16);
          break;
        case 2:
          if (opcode == L_INST) {
            // RV32I: LW
            rddata[t].i = sext((Word)mem_data, 32);
          } else {
            // RV32F: FLW
            rddata[t].f = nan_box((uint32_t)mem_data);
          }
          break;
        case 3: // RV64I: LD
                // RV32D: FLD
        case 4: // RV32I: LBU
        case 5: // RV32I: LHU
        case 6: // RV64I: LWU
          rddata[t]._ = mem_data;
          break;
        default:
          std::abort();      
        }
      }
    } else {
      auto &vd = vreg_file_.at(rdest);
      switch (instr.getVlsWidth()) {
      case 6: {
        for (uint32_t i = 0; i < vl_; i++) {
          Word mem_addr = ((rsdata[i][0].i) & 0xFFFFFFFC) + (i * vtype_.vsew / 8);
          Word mem_data = 0;
          core_->dcache_read(&mem_data, mem_addr, 4);
          Word *result_ptr = (Word *)(vd.data() + i);
          *result_ptr = mem_data;
          DP(4, "LOAD MEM: ADDRESS=0x" << std::hex << mem_addr << ", DATA=0x" << mem_data);        
        }
        break;
      } 
      default:
        std::abort();
      }
    }
    rd_write = true;
    break;
  }
  case S_INST:   
  case FS: {
    trace->exe_type = ExeType::LSU;    
    trace->lsu.type = LsuType::STORE;
    trace->used_iregs.set(rsrc0);
    trace->used_iregs.set(rsrc1);
    if ((opcode == S_INST)
     || (opcode == FS && func3 == 2)
     || (opcode == FS && func3 == 3)) {
      uint32_t mem_bytes = 1 << (func3 & 0x3);
      uint64_t mask = ((uint64_t(1) << (8 * mem_bytes))-1);
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!tmask_.test(t))
          continue;
        uint64_t mem_addr = rsdata[t][0].i + immsrc;
        uint64_t mem_data = rsdata[t][1]._;
        if (mem_bytes < 8) {
          mem_data &= mask;
        }
        trace->mem_addrs.at(t).push_back({mem_addr, mem_bytes});        
        DP(4, "STORE MEM: ADDRESS=0x" << std::hex << mem_addr << ", DATA=0x" << mem_data);
        switch (func3) {
        case 0:
        case 1:
        case 2:
        case 3:
          core_->dcache_write(&mem_data, mem_addr, mem_bytes);  
          break;
        default:
          std::abort();
        }
      }
    } else {
      for (uint32_t i = 0; i < vl_; i++) {
        uint64_t mem_addr = rsdata[i][0].i + (i * vtype_.vsew / 8);        
        switch (instr.getVlsWidth()) {
        case 6: {
          // store word and unit strided (not checking for unit stride)          
          uint32_t mem_data = *(uint32_t *)(vreg_file_.at(instr.getVs3()).data() + i);
          core_->dcache_write(&mem_data, mem_addr, 4);
          DP(4, "STORE MEM: ADDRESS=0x" << std::hex << mem_addr << ", DATA=0x" << mem_data);
          break;
        } 
        default:
          std::abort();
        }          
      }
    }
    break;
  }
  case SYS_INST: {
    for (uint32_t t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      uint32_t csr_addr = immsrc;
      uint32_t csr_value;
      if (func3 == 0) {
        trace->exe_type = ExeType::ALU;
        trace->alu.type = AluType::SYSCALL;
        trace->fetch_stall = true;
        switch (csr_addr) {
        case 0: { // RV32I: ECALL
          core_->trigger_ecall();
          break;
        }
        case 1: { // RV32I: EBREAK
          core_->trigger_ebreak();
          break;
        }
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
        case 1: {
          // RV32I: CSRRW
          rddata[t].i = csr_value;
          core_->set_csr(csr_addr, rsdata[t][0].i, t, id_);      
          trace->used_iregs.set(rsrc0);
          rd_write = true;
          break;
        }
        case 2: {
          // RV32I: CSRRS
          rddata[t].i = csr_value;
          core_->set_csr(csr_addr, csr_value | rsdata[t][0].i, t, id_);
          trace->used_iregs.set(rsrc0);
          rd_write = true;
          break;
        }
        case 3: {
          // RV32I: CSRRC
          rddata[t].i = csr_value;
          core_->set_csr(csr_addr, csr_value & ~rsdata[t][0].i, t, id_);
          trace->used_iregs.set(rsrc0);
          rd_write = true;
          break;
        }
        case 5: {
          // RV32I: CSRRWI
          rddata[t].i = csr_value;
          core_->set_csr(csr_addr, rsrc0, t, id_);      
          rd_write = true;
          break;
        }
        case 6: {
          // RV32I: CSRRSI;
          rddata[t].i = csr_value;
          core_->set_csr(csr_addr, csr_value | rsrc0, t, id_);
          rd_write = true;
          break;
        }
        case 7: {
          // RV32I: CSRRCI
          rddata[t].i = csr_value;
          core_->set_csr(csr_addr, csr_value & ~rsrc0, t, id_);
          rd_write = true;
          break;
        }
        default:
          break;
        }
      }
    } 
    break;
  }
  // RV32I: FENCE
  case FENCE: {
    trace->exe_type = ExeType::LSU;    
    trace->lsu.type = LsuType::FENCE;
    break;
  }   
  case FCI: {       
    trace->exe_type = ExeType::FPU;     
    for (uint32_t t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue; 
      uint32_t frm = get_fpu_rm(func3, core_, t, id_);
      uint32_t fflags = 0;
      switch (func7) {
      case 0x00: { // RV32F: FADD.S
        if (checkBoxedArgs(&rddata[t].f, rsdata[t][0].f, rsdata[t][1].f, &fflags)) {
          rddata[t].f = nan_box(rv_fadd_s(rsdata[t][0].f, rsdata[t][1].f, frm, &fflags));
        }
        trace->fpu.type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x01: { // RV32D: FADD.D
        rddata[t].f = rv_fadd_d(rsdata[t][0].f, rsdata[t][1].f, frm, &fflags);
        trace->fpu.type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x04: { // RV32F: FSUB.S
        if (checkBoxedArgs(&rddata[t].f, rsdata[t][0].f, rsdata[t][1].f, &fflags)) {
          rddata[t].f = nan_box(rv_fsub_s(rsdata[t][0].f, rsdata[t][1].f, frm, &fflags));
        }
        trace->fpu.type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x05: { // RV32D: FSUB.D
        rddata[t].f = rv_fsub_d(rsdata[t][0].f, rsdata[t][1].f, frm, &fflags);
        trace->fpu.type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x08: { // RV32F: FMUL.S
        if (checkBoxedArgs(&rddata[t].f, rsdata[t][0].f, rsdata[t][1].f, &fflags)) {
          rddata[t].f = nan_box(rv_fmul_s(rsdata[t][0].f, rsdata[t][1].f, frm, &fflags));
        }
        trace->fpu.type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x09: { // RV32D: FMUL.D
        rddata[t].f = rv_fmul_d(rsdata[t][0].f, rsdata[t][1].f, frm, &fflags);
        trace->fpu.type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x0c: { // RV32F: FDIV.S
        if (checkBoxedArgs(&rddata[t].f, rsdata[t][0].f, rsdata[t][1].f, &fflags)) {
          rddata[t].f = nan_box(rv_fdiv_s(rsdata[t][0].f, rsdata[t][1].f, frm, &fflags));
        }
        trace->fpu.type = FpuType::FDIV;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x0d: { // RV32D: FDIV.D
        rddata[t].f = rv_fdiv_d(rsdata[t][0].f, rsdata[t][1].f, frm, &fflags);
        trace->fpu.type = FpuType::FDIV;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x2c: { // RV32F: FSQRT.S
        if (checkBoxedArgs(&rddata[t].f, rsdata[t][0].f, &fflags)) {
          rddata[t].f = nan_box(rv_fsqrt_s(rsdata[t][0].f, frm, &fflags));
        }
        trace->fpu.type = FpuType::FSQRT;
        trace->used_fregs.set(rsrc0);
        break;
      }
      case 0x2d: { // RV32D: FSQRT.D
        rddata[t].f = rv_fsqrt_d(rsdata[t][0].f, frm, &fflags);
        trace->fpu.type = FpuType::FSQRT;
        trace->used_fregs.set(rsrc0);
        break;  
      }       
      case 0x10: {
        if (checkBoxedArgs(&rddata[t].f, rsdata[t][0].f, rsdata[t][1].f, &fflags)) {
          switch (func3) {            
          case 0: // RV32F: FSGNJ.S
            rddata[t].f = nan_box(rv_fsgnj_s(rsdata[t][0].f, rsdata[t][1].f));
            break;          
          case 1: // RV32F: FSGNJN.S
            rddata[t].f = nan_box(rv_fsgnjn_s(rsdata[t][0].f, rsdata[t][1].f));
            break;          
          case 2: // RV32F: FSGNJX.S
            rddata[t].f = nan_box(rv_fsgnjx_s(rsdata[t][0].f, rsdata[t][1].f));
            break;
          }
        }
        trace->fpu.type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x11: {
        switch (func3) {            
        case 0: // RV32D: FSGNJ.D
          rddata[t].f = rv_fsgnj_d(rsdata[t][0].f, rsdata[t][1].f);
          break;          
        case 1: // RV32D: FSGNJN.D
          rddata[t].f = rv_fsgnjn_d(rsdata[t][0].f, rsdata[t][1].f);
          break;          
        case 2: // RV32D: FSGNJX.D
          rddata[t].f = rv_fsgnjx_d(rsdata[t][0].f, rsdata[t][1].f);
          break;
        }
        trace->fpu.type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x14: {   
        if (checkBoxedArgs(&rddata[t].f, rsdata[t][0].f, rsdata[t][1].f, &fflags)) {         
          if (func3) {
            // RV32F: FMAX.S
            rddata[t].f = nan_box(rv_fmax_s(rsdata[t][0].f, rsdata[t][1].f, &fflags));
          } else {
            // RV32F: FMIN.S
            rddata[t].f = nan_box(rv_fmin_s(rsdata[t][0].f, rsdata[t][1].f, &fflags));
          }
        }
        trace->fpu.type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);        
        break;
      }
      case 0x15: {            
        if (func3) {
          // RV32D: FMAX.D
          rddata[t].f = rv_fmax_d(rsdata[t][0].f, rsdata[t][1].f, &fflags);
        } else {
          // RV32D: FMIN.D
          rddata[t].f = rv_fmin_d(rsdata[t][0].f, rsdata[t][1].f, &fflags);
        }
        trace->fpu.type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);        
        break;
      }
      case 0x20: {
        // RV32D: FCVT.S.D
        rddata[t].f = nan_box(rv_dtof(rsdata[t][0].f));
        trace->fpu.type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);        
        break;
      }
      case 0x21: {
        // RV32D: FCVT.D.S
        rddata[t].f = rv_ftod(rsdata[t][0].f);
        trace->fpu.type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);        
        break;
      }
      case 0x60: {
        switch (rsrc1) {
          case 0: 
            // RV32F: FCVT.W.S
            rddata[t].i = sext((uint64_t)rv_ftoi_s(rsdata[t][0].f, frm, &fflags), 32);
            break;
          case 1:
            // RV32F: FCVT.WU.S
            rddata[t].i = sext((uint64_t)rv_ftou_s(rsdata[t][0].f, frm, &fflags), 32);
            break;
          case 2:
            // RV64F: FCVT.L.S
            rddata[t].i = rv_ftol_s(rsdata[t][0].f, frm, &fflags);
            break;
          case 3:
            // RV64F: FCVT.LU.S
            rddata[t].i = rv_ftolu_s(rsdata[t][0].f, frm, &fflags);
            break;
        }
      trace->fpu.type = FpuType::FCVT;
      trace->used_fregs.set(rsrc0);
      break;
    }
      case 0x61: {
        switch (rsrc1) {
          case 0: 
            // RV32D: FCVT.W.D
            rddata[t].i = sext((uint64_t)rv_ftoi_d(rsdata[t][0].f, frm, &fflags), 32);
            break;
          case 1:
            // RV32D: FCVT.WU.D
            rddata[t].i = sext((uint64_t)rv_ftou_d(rsdata[t][0].f, frm, &fflags), 32);
            break;
          case 2:
            // RV64D: FCVT.L.D
            rddata[t].i = rv_ftol_d(rsdata[t][0].f, frm, &fflags);
            break;
          case 3:
            // RV64D: FCVT.LU.D
            rddata[t].i = rv_ftolu_d(rsdata[t][0].f, frm, &fflags);
            break;
        }
        trace->fpu.type = FpuType::FCVT;
        trace->used_fregs.set(rsrc0);
        break;
      }
      case 0x70: {     
        if (func3) {
          // RV32F: FCLASS.S
          rddata[t].i = rv_fclss_s(rsdata[t][0].f);
        } else {          
          // RV32F: FMV.X.W
          uint32_t result = (uint32_t)rsdata[t][0].f;
          rddata[t].i = sext((uint64_t)result, 32);
        }        
        trace->fpu.type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        break;
      }
      case 0x71: {    
        if (func3) {
          // RV32D: FCLASS.D
          rddata[t].i = rv_fclss_d(rsdata[t][0].f);
        } else {          
          // RV64D: FMV.X.D
          rddata[t].i = rsdata[t][0].f;
        }        
        trace->fpu.type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        break;
      }
      case 0x50: {      
        if (checkBoxedCmpArgs(&rddata[t].i, rsdata[t][0].f, rsdata[t][1].f, &fflags)) {     
          switch (func3) {              
          case 0:
            // RV32F: FLE.S
            rddata[t].i = rv_fle_s(rsdata[t][0].f, rsdata[t][1].f, &fflags);    
            break;              
          case 1:
            // RV32F: FLT.S
            rddata[t].i = rv_flt_s(rsdata[t][0].f, rsdata[t][1].f, &fflags);
            break;              
          case 2:
            // RV32F: FEQ.S
            rddata[t].i = rv_feq_s(rsdata[t][0].f, rsdata[t][1].f, &fflags);
            break;
          }
        } 
        trace->fpu.type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break; 
      }
      case 0x51: {           
        switch (func3) {              
        case 0:
          // RV32D: FLE.D
          rddata[t].i = rv_fle_d(rsdata[t][0].f, rsdata[t][1].f, &fflags);    
          break;              
        case 1:
          // RV32D: FLT.D
          rddata[t].i = rv_flt_d(rsdata[t][0].f, rsdata[t][1].f, &fflags);
          break;              
        case 2:
          // RV32D: FEQ.D
          rddata[t].i = rv_feq_d(rsdata[t][0].f, rsdata[t][1].f, &fflags);
          break;
        } 
        trace->fpu.type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;  
      }      
      case 0x68: {
        switch (rsrc1) {
          case 0: 
            // RV32F: FCVT.S.W
            rddata[t].f = nan_box(rv_itof_s(rsdata[t][0].i, frm, &fflags));
            break;
          case 1:
            // RV32F: FCVT.S.WU
            rddata[t].f = nan_box(rv_utof_s(rsdata[t][0].i, frm, &fflags));
            break;
          case 2:
            // RV64F: FCVT.S.L
            rddata[t].f = nan_box(rv_ltof_s(rsdata[t][0].i, frm, &fflags));
            break;
          case 3:
            // RV64F: FCVT.S.LU
            rddata[t].f = nan_box(rv_lutof_s(rsdata[t][0].i, frm, &fflags));
            break;
        }
        trace->fpu.type = FpuType::FCVT;
        trace->used_iregs.set(rsrc0);
        break;
      }
      case 0x69: {
        switch (rsrc1) {
          case 0: 
            // RV32D: FCVT.D.W
            rddata[t].f = rv_itof_d(rsdata[t][0].i, frm, &fflags);
            break;
          case 1:
            // RV32D: FCVT.D.WU
            rddata[t].f = rv_utof_d(rsdata[t][0].i, frm, &fflags);
            break;
          case 2:
            // RV64D: FCVT.D.L
            rddata[t].f = rv_ltof_d(rsdata[t][0].i, frm, &fflags);
            break;
          case 3:
            // RV64D: FCVT.D.LU
            rddata[t].f = rv_lutof_d(rsdata[t][0].i, frm, &fflags);
            break;
        }
        trace->fpu.type = FpuType::FCVT;
        trace->used_iregs.set(rsrc0);
        break;
      }
      case 0x78: { // RV32F: FMV.W.X
        rddata[t].f = nan_box((uint32_t)rsdata[t][0].i);
        trace->fpu.type = FpuType::FNCP;
        trace->used_iregs.set(rsrc0);
        break;
      }
      case 0x79: { // RV64D: FMV.D.X
        rddata[t].f = rsdata[t][0].i;
        trace->fpu.type = FpuType::FNCP;
        trace->used_iregs.set(rsrc0);
        break;
      }
      }
      update_fcrs(fflags, core_, t, id_);
    }
    rd_write = true;
    break;
  }
  case FMADD:      
  case FMSUB:      
  case FMNMADD:
  case FMNMSUB: {
    trace->fpu.type = FpuType::FMA;
    trace->used_fregs.set(rsrc0);
    trace->used_fregs.set(rsrc1);
    trace->used_fregs.set(rsrc2);
    for (uint32_t t = 0; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      uint32_t frm = get_fpu_rm(func3, core_, t, id_);
      uint32_t fflags = 0;
      switch (opcode) {
      case FMADD:
        if (func2)
          // RV32D: FMADD.D
          rddata[t].f = rv_fmadd_d(rsdata[t][0].f, rsdata[t][1].f, rsdata[t][2].f, frm, &fflags);
        else
          // RV32F: FMADD.S
          rddata[t].f = nan_box(rv_fmadd_s(rsdata[t][0].f, rsdata[t][1].f, rsdata[t][2].f, frm, &fflags));
        break;
      case FMSUB:
        if (func2)
          // RV32D: FMSUB.D
          rddata[t].f = rv_fmsub_d(rsdata[t][0].f, rsdata[t][1].f, rsdata[t][2].f, frm, &fflags);
        else 
          // RV32F: FMSUB.S
          rddata[t].f = nan_box(rv_fmsub_s(rsdata[t][0].f, rsdata[t][1].f, rsdata[t][2].f, frm, &fflags));
        break;
      case FMNMADD:
        if (func2)
          // RV32D: FNMADD.D
          rddata[t].f = rv_fnmadd_d(rsdata[t][0].f, rsdata[t][1].f, rsdata[t][2].f, frm, &fflags);
        else
          // RV32F: FNMADD.S
          rddata[t].f = nan_box(rv_fnmadd_s(rsdata[t][0].f, rsdata[t][1].f, rsdata[t][2].f, frm, &fflags));
        break; 
      case FMNMSUB:
        if (func2)
          // RV32D: FNMSUB.D
          rddata[t].f = rv_fnmsub_d(rsdata[t][0].f, rsdata[t][1].f, rsdata[t][2].f, frm, &fflags);
        else
          // RV32F: FNMSUB.S
          rddata[t].f = nan_box(rv_fnmsub_s(rsdata[t][0].f, rsdata[t][1].f, rsdata[t][2].f, frm, &fflags));
        break;
      default:
        break;
      }              
      update_fcrs(fflags, core_, t, id_);
    }
    rd_write = true;
    break;
  }
  case GPGPU: {    
    uint32_t ts = 0;
    for (uint32_t t = 0; t < num_threads; ++t) {
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
        for (uint32_t t = 0; t < num_threads; ++t) {
          pred[t] = tmask_.test(t) ? (ireg_file_.at(t).at(rsrc0) != 0) : 0;
        }
        if (pred.any()) {
          tmask_ &= pred;
        }
      } else {
        tmask_.reset();
        for (uint32_t t = 0; t < num_threads; ++t) {
          tmask_.set(t, rsdata.at(ts)[0].i & (1 << t));
        }
      }
      DPH(3, "*** New TMC: ");
      for (uint32_t i = 0; i < num_threads; ++i)
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
      trace->gpu.active_warps = core_->wspawn(rsdata.at(ts)[0].i, rsdata.at(ts)[1].i);
    } break;
    case 2: {
      // SPLIT    
      trace->exe_type = ExeType::GPU;
      trace->gpu.type = GpuType::SPLIT;
      trace->used_iregs.set(rsrc0);
      trace->fetch_stall = true;
      if (HasDivergentThreads(tmask_, ireg_file_, rsrc0)) {          
        ThreadMask tmask;
        for (uint32_t t = 0; t < num_threads; ++t) {
          tmask[t] = tmask_.test(t) && !ireg_file_.at(t).at(rsrc0);
        }

        DomStackEntry e(tmask, nextPC);
        dom_stack_.push(tmask_);
        dom_stack_.push(e);
        for (uint32_t t = 0, n = e.tmask.size(); t < n; ++t) {
          tmask_.set(t, !e.tmask.test(t) && tmask_.test(t));
        }
        active_ = tmask_.any();

        DPH(3, "*** Split: New TM=");
        for (uint32_t t = 0; t < num_threads; ++t) DPN(3, tmask_.test(num_threads-t-1));
        DPN(3, ", Pushed TM=");
        for (uint32_t t = 0; t < num_threads; ++t) DPN(3, e.tmask.test(num_threads-t-1));
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
        for (uint32_t t = 0; t < num_threads; ++t) DPN(3, tmask_.test(num_threads-t-1));
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
      trace->gpu.active_warps = core_->barrier(rsdata[ts][0].i, rsdata[ts][1].i, id_);
    } break;
    case 5: {
      // PREFETCH
      trace->exe_type = ExeType::LSU; 
      trace->lsu.type = LsuType::PREFETCH; 
      trace->used_iregs.set(rsrc0);
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!tmask_.test(t))
          continue;
        auto mem_addr = rsdata[t][0].i;
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
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (!tmask_.test(t))
          continue;        
        auto unit  = func2;
        auto u     = rsdata[t][0].i;
        auto v     = rsdata[t][1].i;
        auto lod   = rsdata[t][2].i;
        auto color = core_->tex_read(unit, u, v, lod, &trace->mem_addrs.at(t));
        rddata[t].i = color;
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
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (!tmask_.test(t))
            continue;     
          rddata[t].i = rsdata[t][0].i ? rsdata[t][1].i : rsdata[t][2].i;
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
    uint32_t VLEN = core_->arch().vsize() * 8;
    uint32_t VLMAX = (instr.getVlmul() * VLEN) / instr.getVsew();
    switch (func3) {
    case 0: // vector-vector
      switch (func6) {
      case 0: {
        auto& vr1 = vreg_file_.at(rsrc0);
        auto& vr2 = vreg_file_.at(rsrc1);
        auto& vd = vreg_file_.at(rdest);
        auto& mask = vreg_file_.at(0);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first == second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first == second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first != second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first != second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first < second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first < second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            int8_t first  = *(int8_t *)(vr1.data() + i);
            int8_t second = *(int8_t *)(vr2.data() + i);
            int8_t result = (first < second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            int16_t first  = *(int16_t *)(vr1.data() + i);
            int16_t second = *(int16_t *)(vr2.data() + i);
            int16_t result = (first < second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(int16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first <= second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first <= second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            int8_t first  = *(int8_t *)(vr1.data() + i);
            int8_t second = *(int8_t *)(vr2.data() + i);
            int8_t result = (first <= second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            int16_t first  = *(int16_t *)(vr1.data() + i);
            int16_t second = *(int16_t *)(vr2.data() + i);
            int16_t result = (first <= second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(int16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first > second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first > second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            int8_t first  = *(int8_t *)(vr1.data() + i);
            int8_t second = *(int8_t *)(vr2.data() + i);
            int8_t result = (first > second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            int16_t first  = *(int16_t *)(vr1.data() + i);
            int16_t second = *(int16_t *)(vr2.data() + i);
            int16_t result = (first > second) ? 1 : 0;
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(int16_t *)(vd.data() + i) = result;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value & !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }            
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = (first_value & !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = (first_value & !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = (first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = (first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = (first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = (first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = (first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = (first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = (first_value | !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = (first_value | !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = (first_value | !second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = !(first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = !(first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = !(first_value & second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = !(first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = !(first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = !(first_value | second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t first_value  = (first & 0x1);
            uint8_t second_value = (second & 0x1);
            uint8_t result = !(first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t first_value  = (first & 0x1);
            uint16_t second_value = (second & 0x1);
            uint16_t result = !(first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t first_value  = (first & 0x1);
            uint32_t second_value = (second & 0x1);
            uint32_t result = !(first_value ^ second_value);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t first  = *(uint8_t *)(vr1.data() + i);
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) += result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t first  = *(uint16_t *)(vr1.data() + i);
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) += result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t first  = *(uint32_t *)(vr1.data() + i);
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (first * second);
            DP(3, "Comparing " << first << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) += result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
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
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (rsdata[i][0].i + second);
            DP(3, "Comparing " << rsdata[i][0].i << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (rsdata[i][0].i + second);
            DP(3, "Comparing " << rsdata[i][0].i << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (rsdata[i][0].i + second);
            DP(3, "Comparing " << rsdata[i][0].i << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      case 37: {
        // vmul.vx
        auto &vr2 = vreg_file_.at(rsrc1);
        auto &vd = vreg_file_.at(rdest);
        if (vtype_.vsew == 8) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint8_t second = *(uint8_t *)(vr2.data() + i);
            uint8_t result = (rsdata[i][0].i * second);
            DP(3, "Comparing " << rsdata[i][0].i << " + " << second << " = " << result);
            *(uint8_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint8_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 16) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint16_t second = *(uint16_t *)(vr2.data() + i);
            uint16_t result = (rsdata[i][0].i * second);
            DP(3, "Comparing " << rsdata[i][0].i << " + " << second << " = " << result);
            *(uint16_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint16_t *)(vd.data() + i) = 0;
          }
        } else if (vtype_.vsew == 32) {
          for (uint32_t i = 0; i < vl_; i++) {
            uint32_t second = *(uint32_t *)(vr2.data() + i);
            uint32_t result = (rsdata[i][0].i * second);
            DP(3, "Comparing " << rsdata[i][0].i << " + " << second << " = " << result);
            *(uint32_t *)(vd.data() + i) = result;
          }
          for (uint32_t i = vl_; i < VLMAX; i++) {
            *(uint32_t *)(vd.data() + i) = 0;
          }
        }
      } break;
      }
    } break;
    case 7: {
      vtype_.vill  = 0;
      vtype_.vediv = instr.getVediv();
      vtype_.vsew  = instr.getVsew();
      vtype_.vlmul = instr.getVlmul();

      DP(3, "lmul:" << vtype_.vlmul << " sew:" << vtype_.vsew  << " ediv: " << vtype_.vediv << "rsrc_" << rsdata[0][0].i << "VLMAX" << VLMAX);

      auto s0 = rsdata[0][0].i;
      if (s0 <= VLMAX) {
        vl_ = s0;
      } else if (s0 < (2 * VLMAX)) {
        vl_ = (uint32_t)ceil((s0 * 1.0) / 2.0);
      } else if (s0 >= (2 * VLMAX)) {
        vl_ = VLMAX;
      }        
      rddata[0].i = vl_;
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
    auto type = instr.getRDType();    
    switch (type) {
    case RegType::Integer:      
      if (rdest) {    
        DPN(2, type << std::dec << rdest << "={");    
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (t) DPN(2, ", ");
          if (!tmask_.test(t)) {
            DPN(2, "-");
            continue;            
          }
          ireg_file_.at(t)[rdest] = rddata[t].i;
          DPN(2, "0x" << std::hex << rddata[t].i);         
        }
        DPN(2, "}" << std::endl);
        trace->used_iregs[rdest] = 1;
      }
      break;
    case RegType::Float:
      DPN(2, type << std::dec << rdest << "={");
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (t) DPN(2, ", ");
        if (!tmask_.test(t)) {
          DPN(2, "-");
          continue;            
        }
        freg_file_.at(t)[rdest] = rddata[t].f;        
        DPN(2, "0x" << std::hex << rddata[t].f);         
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