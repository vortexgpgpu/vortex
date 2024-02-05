// Copyright © 2019-2023
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
  Word     u;
  WordI    i;
  WordF    f;
  float    f32;
  double   f64;
  uint32_t u32;
  uint64_t u64; 
  int32_t  i32;
  int64_t  i64;
};

inline uint32_t get_fpu_rm(uint32_t func3, Core* core, uint32_t tid, uint32_t wid) {
  return (func3 == 0x7) ? core->get_csr(VX_CSR_FRM, tid, wid) : func3;
}

inline void update_fcrs(uint32_t fflags, Core* core, uint32_t tid, uint32_t wid) {
  if (fflags) {
    core->set_csr(VX_CSR_FCSR, core->get_csr(VX_CSR_FCSR, tid, wid) | fflags, tid, wid);
    core->set_csr(VX_CSR_FFLAGS, core->get_csr(VX_CSR_FFLAGS, tid, wid) | fflags, tid, wid);
  }
}

inline uint64_t nan_box(uint32_t value) {
  uint64_t mask = 0xffffffff00000000;
  return value | mask;
}

inline bool is_nan_boxed(uint64_t value) {
  return (uint32_t(value >> 32) == 0xffffffff);
}

inline int64_t check_boxing(int64_t a) {  
  if (is_nan_boxed(a))
    return a;
  return nan_box(0x7fc00000); // NaN
}

void Warp::execute(const Instr &instr, pipeline_trace_t *trace) {
  assert(tmask_.any());

  auto next_pc = PC_ + 4;
  auto next_tmask = tmask_; 

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

  auto num_threads = arch_.num_threads();

  uint32_t thread_start = 0;
  for (; thread_start < num_threads; ++thread_start) {
      if (tmask_.test(thread_start))
        break;
  }

  std::vector<reg_data_t[3]> rsdata(num_threads);
  std::vector<reg_data_t> rddata(num_threads);

  auto num_rsrcs = instr.getNRSrc();
  if (num_rsrcs) {              
    for (uint32_t i = 0; i < num_rsrcs; ++i) {          
      auto type = instr.getRSType(i);
      auto reg = instr.getRSrc(i);        
      switch (type) {
      case RegType::Integer: 
        DPH(2, "Src" << std::dec << i << " Reg: " << type << std::dec << reg << "={");
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (t) DPN(2, ", ");
          if (!tmask_.test(t)) {
            DPN(2, "-");
            continue;            
          }
          rsdata[t][i].u = ireg_file_.at(t)[reg];          
          DPN(2, "0x" << std::hex << rsdata[t][i].i);
        }
        DPN(2, "}" << std::endl);
        break;
      case RegType::Float: 
        DPH(2, "Src" << std::dec << i << " Reg: " << type << std::dec << reg << "={");
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (t) DPN(2, ", ");
          if (!tmask_.test(t)) {
            DPN(2, "-");
            continue;            
          }
          rsdata[t][i].u64 = freg_file_.at(t)[reg];
          DPN(2, "0x" << std::hex << rsdata[t][i].f);
        }
        DPN(2, "}" << std::endl);
        break;
      case RegType::Vector:
        // TODO:
        break;
      case RegType::None:
        break;
      }      
    }
  }

  bool rd_write = false;
  
  switch (opcode) {  
  case LUI_INST: {
    // RV32I: LUI
    trace->exe_type = ExeType::ALU;
    trace->alu_type = AluType::ARITH;
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      rddata[t].i = immsrc << 12;
    }    
    rd_write = true;
    break;
  }  
  case AUIPC_INST: {
    // RV32I: AUIPC
    trace->exe_type = ExeType::ALU;
    trace->alu_type = AluType::ARITH;
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      rddata[t].i = (immsrc << 12) + PC_;
    }    
    rd_write = true;
    break;
  }
  case R_INST: {
    trace->exe_type = ExeType::ALU;    
    trace->alu_type = AluType::ARITH;
    trace->used_iregs.set(rsrc0);
    trace->used_iregs.set(rsrc1);
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      if (func7 & 0x1) {
        switch (func3) {
        case 0: {
          // RV32M: MUL
          rddata[t].i = rsdata[t][0].i * rsdata[t][1].i;
          trace->alu_type = AluType::IMUL;
          break;
        }
        case 1: {
          // RV32M: MULH
          auto first = static_cast<DWordI>(rsdata[t][0].i);
          auto second = static_cast<DWordI>(rsdata[t][1].i);
          rddata[t].i = (first * second) >> XLEN;
          trace->alu_type = AluType::IMUL;
          break;
        }
        case 2: {
          // RV32M: MULHSU       
          auto first = static_cast<DWordI>(rsdata[t][0].i);
          auto second = static_cast<DWord>(rsdata[t][1].u);
          rddata[t].i = (first * second) >> XLEN;
          trace->alu_type = AluType::IMUL;
          break;
        } 
        case 3: {
          // RV32M: MULHU
          auto first = static_cast<DWord>(rsdata[t][0].u);
          auto second = static_cast<DWord>(rsdata[t][1].u);
          rddata[t].i = (first * second) >> XLEN;
          trace->alu_type = AluType::IMUL;
          break;
        } 
        case 4: {
          // RV32M: DIV
          auto dividen = rsdata[t][0].i;
          auto divisor = rsdata[t][1].i; 
          auto largest_negative = WordI(1) << (XLEN-1);  
          if (divisor == 0) {
            rddata[t].i = -1;
          } else if (dividen == largest_negative && divisor == -1) {
            rddata[t].i = dividen;
          } else {
            rddata[t].i = dividen / divisor;
          }
          trace->alu_type = AluType::IDIV;
          break;
        } 
        case 5: {
          // RV32M: DIVU
          auto dividen = rsdata[t][0].u;
          auto divisor = rsdata[t][1].u;
          if (divisor == 0) {
            rddata[t].i = -1;
          } else {
            rddata[t].i = dividen / divisor;
          }
          trace->alu_type = AluType::IDIV;
          break;
        } 
        case 6: {
          // RV32M: REM
          auto dividen = rsdata[t][0].i;
          auto divisor = rsdata[t][1].i;
          auto largest_negative = WordI(1) << (XLEN-1);
          if (rsdata[t][1].i == 0) {
            rddata[t].i = dividen;
          } else if (dividen == largest_negative && divisor == -1) {
            rddata[t].i = 0;
          } else {
            rddata[t].i = dividen % divisor;
          }
          trace->alu_type = AluType::IDIV;
          break;
        } 
        case 7: {
          // RV32M: REMU
          auto dividen = rsdata[t][0].u;
          auto divisor = rsdata[t][1].u;
          if (rsdata[t][1].i == 0) {
            rddata[t].i = dividen;
          } else {
            rddata[t].i = dividen % divisor;
          }
          trace->alu_type = AluType::IDIV;
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
          rddata[t].i = rsdata[t][0].i < rsdata[t][1].i;
          break;
        }
        case 3: {
          // RV32I: SLTU
          rddata[t].i = rsdata[t][0].u < rsdata[t][1].u;
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
            rddata[t].i = rsdata[t][0].i >> shamt;
          } else {
            // RV32I: SRL
            rddata[t].i = rsdata[t][0].u >> shamt;
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
    trace->alu_type = AluType::ARITH;    
    trace->used_iregs.set(rsrc0);
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      switch (func3) {
      case 0: {
        // RV32I: ADDI
        rddata[t].i = rsdata[t][0].i + immsrc;
        break;
      }
      case 1: {
        // RV32I: SLLI
        rddata[t].i = rsdata[t][0].i << immsrc;
        break;
      }
      case 2: {
        // RV32I: SLTI
        rddata[t].i = rsdata[t][0].i < WordI(immsrc);
        break;
      }
      case 3: {
        // RV32I: SLTIU
        rddata[t].i = rsdata[t][0].u < immsrc;
        break;
      } 
      case 4: {
        // RV32I: XORI
        rddata[t].i = rsdata[t][0].i ^ immsrc;
        break;
      }
      case 5: {
        if (func7) {
          // RV32I: SRAI
          Word result = rsdata[t][0].i >> immsrc;
          rddata[t].i = result;
        } else {
          // RV32I: SRLI
          Word result = rsdata[t][0].u >> immsrc;
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
    trace->alu_type = AluType::ARITH;
    trace->used_iregs.set(rsrc0);
    trace->used_iregs.set(rsrc1);
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      if (func7 & 0x1) {
        switch (func3) {
          case 0: {
            // RV64M: MULW
            int32_t product = (int32_t)rsdata[t][0].i * (int32_t)rsdata[t][1].i;
            rddata[t].i = sext((uint64_t)product, 32);
            trace->alu_type = AluType::IMUL;
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
            trace->alu_type = AluType::IDIV;
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
            trace->alu_type = AluType::IDIV;
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
            trace->alu_type = AluType::IDIV;
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
            trace->alu_type = AluType::IDIV;
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
    trace->alu_type = AluType::ARITH;    
    trace->used_iregs.set(rsrc0);
    for (uint32_t t = thread_start; t < num_threads; ++t) {
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
    trace->alu_type = AluType::BRANCH;    
    trace->used_iregs.set(rsrc0);
    trace->used_iregs.set(rsrc1);
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      switch (func3) {
      case 0: {
        // RV32I: BEQ
        if (rsdata[t][0].i == rsdata[t][1].i) {
          next_pc = PC_ + immsrc;
        }
        break;
      }
      case 1: {
        // RV32I: BNE
        if (rsdata[t][0].i != rsdata[t][1].i) {
          next_pc = PC_ + immsrc;
        }
        break;
      }
      case 4: {
        // RV32I: BLT
        if (rsdata[t][0].i < rsdata[t][1].i) {
          next_pc = PC_ + immsrc;
        }
        break;
      }
      case 5: {
        // RV32I: BGE
        if (rsdata[t][0].i >= rsdata[t][1].i) {
          next_pc = PC_ + immsrc;
        }
        break;
      }
      case 6: {
        // RV32I: BLTU
        if (rsdata[t][0].u < rsdata[t][1].u) {
          next_pc = PC_ + immsrc;
        }
        break;
      }
      case 7: {
        // RV32I: BGEU
        if (rsdata[t][0].u >= rsdata[t][1].u) {
          next_pc = PC_ + immsrc;
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
  case JAL_INST: {
    // RV32I: JAL
    trace->exe_type = ExeType::ALU;    
    trace->alu_type = AluType::BRANCH;
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      rddata[t].i = next_pc;
    }
    next_pc = PC_ + immsrc;
    trace->fetch_stall = true;
    rd_write = true;
    break;
  }  
  case JALR_INST: {
    // RV32I: JALR
    trace->exe_type = ExeType::ALU;    
    trace->alu_type = AluType::BRANCH;
    trace->used_iregs.set(rsrc0);
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      rddata[t].i = next_pc;
    }
    next_pc = rsdata[thread_start][0].i + immsrc;
    trace->fetch_stall = true;
    rd_write = true;
    break;
  }
  case L_INST:
  case FL: {
    trace->exe_type = ExeType::LSU;    
    trace->lsu_type = LsuType::LOAD;
    trace->used_iregs.set(rsrc0);
    auto trace_data = std::make_shared<LsuTraceData>(num_threads);
    trace->data = trace_data;
    if ((opcode == L_INST )
     || (opcode == FL && func3 == 2)
     || (opcode == FL && func3 == 3)) {
      uint32_t data_bytes = 1 << (func3 & 0x3);
      uint32_t data_width = 8 * data_bytes;
      for (uint32_t t = thread_start; t < num_threads; ++t) {
        if (!tmask_.test(t))
          continue;
        uint64_t mem_addr = rsdata[t][0].i + immsrc;         
        uint64_t read_data = 0;
        core_->dcache_read(&read_data, mem_addr, data_bytes);
        trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
        switch (func3) {
        case 0: // RV32I: LB
        case 1: // RV32I: LH
          rddata[t].i = sext((Word)read_data, data_width);
          break;
        case 2:
          if (opcode == L_INST) {
            // RV32I: LW
            rddata[t].i = sext((Word)read_data, data_width);
          } else {
            // RV32F: FLW
            rddata[t].u64 = nan_box((uint32_t)read_data);
          }
          break;
        case 3: // RV64I: LD
                // RV32D: FLD
        case 4: // RV32I: LBU
        case 5: // RV32I: LHU
        case 6: // RV64I: LWU
          rddata[t].u64 = read_data;
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
    trace->lsu_type = LsuType::STORE;
    trace->used_iregs.set(rsrc0);
    trace->used_iregs.set(rsrc1);    
    auto trace_data = std::make_shared<LsuTraceData>(num_threads);
    trace->data = trace_data;
    if ((opcode == S_INST)
     || (opcode == FS && func3 == 2)
     || (opcode == FS && func3 == 3)) {
      uint32_t data_bytes = 1 << (func3 & 0x3);
      for (uint32_t t = thread_start; t < num_threads; ++t) {
        if (!tmask_.test(t))
          continue;
        uint64_t mem_addr = rsdata[t][0].i + immsrc;
        uint64_t write_data = rsdata[t][1].u64;
        trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
        switch (func3) {
        case 0:
        case 1:
        case 2:
        case 3:
          core_->dcache_write(&write_data, mem_addr, data_bytes);  
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
          break;
        } 
        default:
          std::abort();
        }          
      }
    }
    break;
  }
  case AMO: {
    trace->exe_type = ExeType::LSU;    
    trace->lsu_type = LsuType::LOAD;
    trace->used_iregs.set(rsrc0);
    trace->used_iregs.set(rsrc1);
    auto trace_data = std::make_shared<LsuTraceData>(num_threads);
    trace->data = trace_data;
    auto amo_type = func7 >> 2;
    uint32_t data_bytes = 1 << (func3 & 0x3);
    uint32_t data_width = 8 * data_bytes;
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      uint64_t mem_addr = rsdata[t][0].u;
      trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
      if (amo_type == 0x02) { // LR
        uint64_t read_data = 0;
        core_->dcache_read(&read_data, mem_addr, data_bytes);        
        core_->dcache_amo_reserve(mem_addr);
        rddata[t].i = sext((Word)read_data, data_width);
      } else 
      if (amo_type == 0x03) { // SC
        if (core_->dcache_amo_check(mem_addr)) {
          core_->dcache_write(&rsdata[t][1].u64, mem_addr, data_bytes);
          rddata[t].i = 0;
        } else {
          rddata[t].i = 1;
        }
      } else {      
        uint64_t read_data = 0;
        core_->dcache_read(&read_data, mem_addr, data_bytes);
        auto read_data_i = sext((WordI)read_data, data_width);        
        auto rs1_data_i  = sext((WordI)rsdata[t][1].u64, data_width);
        auto read_data_u = zext((Word)read_data, data_width);
        auto rs1_data_u  = zext((Word)rsdata[t][1].u64, data_width);
        uint64_t result;
        switch (amo_type) {
        case 0x00:  // AMOADD
          result = read_data_i + rs1_data_i;
          break;
        case 0x01:  // AMOSWAP
          result = rs1_data_u;
          break;
        case 0x04:  // AMOXOR
          result = read_data_u ^ rs1_data_u;
          break;
        case 0x08:  // AMOOR
          result = read_data_u | rs1_data_u;
          break;
        case 0x0c:  // AMOAND
          result = read_data_u & rs1_data_u;
          break;
        case 0x10:  // AMOMIN
          result = std::min(read_data_i, rs1_data_i);
          break;
        case 0x14:  // AMOMAX
          result = std::max(read_data_i, rs1_data_i);
          break;
        case 0x18:  // AMOMINU
          result = std::min(read_data_u, rs1_data_u);
          break;
        case 0x1c:  // AMOMAXU
          result = std::max(read_data_u, rs1_data_u);
          break;
        default:
          std::abort();
        }
        core_->dcache_write(&result, mem_addr, data_bytes);
        rddata[t].i = read_data_i;
      }
    }
    rd_write = true;
    break;
  }
  case SYS_INST: {
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      uint32_t csr_addr = immsrc;
      uint32_t csr_value;
      if (func3 == 0) {
        trace->exe_type = ExeType::ALU;
        trace->alu_type = AluType::SYSCALL;
        trace->fetch_stall = true;
        switch (csr_addr) {
        case 0: 
          // RV32I: ECALL
          core_->trigger_ecall();
          break;
        case 1: 
          // RV32I: EBREAK
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
        trace->exe_type = ExeType::SFU;        
        trace->fetch_stall = true;
        csr_value = core_->get_csr(csr_addr, t, warp_id_);
        switch (func3) {
        case 1: {
          // RV32I: CSRRW
          rddata[t].i = csr_value;
          core_->set_csr(csr_addr, rsdata[t][0].i, t, warp_id_);      
          trace->used_iregs.set(rsrc0);
          trace->sfu_type = SfuType::CSRRW;
          rd_write = true;
          break;
        }
        case 2: {
          // RV32I: CSRRS
          rddata[t].i = csr_value;
          if (rsdata[t][0].i != 0) {
            core_->set_csr(csr_addr, csr_value | rsdata[t][0].i, t, warp_id_);
          }
          trace->used_iregs.set(rsrc0);
          trace->sfu_type = SfuType::CSRRS;
          rd_write = true;
          break;
        }
        case 3: {
          // RV32I: CSRRC
          rddata[t].i = csr_value;
          if (rsdata[t][0].i != 0) {
            core_->set_csr(csr_addr, csr_value & ~rsdata[t][0].i, t, warp_id_);
          }
          trace->used_iregs.set(rsrc0);
          trace->sfu_type = SfuType::CSRRC;
          rd_write = true;
          break;
        }
        case 5: {
          // RV32I: CSRRWI
          rddata[t].i = csr_value;
          core_->set_csr(csr_addr, rsrc0, t, warp_id_);
          trace->sfu_type = SfuType::CSRRW;    
          rd_write = true;
          break;
        }
        case 6: {
          // RV32I: CSRRSI;
          rddata[t].i = csr_value;
          if (rsrc0 != 0) {
            core_->set_csr(csr_addr, csr_value | rsrc0, t, warp_id_);
          }
          trace->sfu_type = SfuType::CSRRS;
          rd_write = true;
          break;
        }
        case 7: {
          // RV32I: CSRRCI
          rddata[t].i = csr_value;
          if (rsrc0 != 0) {
            core_->set_csr(csr_addr, csr_value & ~rsrc0, t, warp_id_);
          }
          trace->sfu_type = SfuType::CSRRC;
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
  case FENCE: {
    // RV32I: FENCE
    trace->exe_type = ExeType::LSU;    
    trace->lsu_type = LsuType::FENCE;
    break;
  }
  case FCI: {     
    trace->exe_type = ExeType::FPU;     
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue; 
      uint32_t frm = get_fpu_rm(func3, core_, t, warp_id_);
      uint32_t fflags = 0;
      switch (func7) {
      case 0x00: { // RV32F: FADD.S
        rddata[t].u64 = nan_box(rv_fadd_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), frm, &fflags));
        trace->fpu_type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x01: { // RV32D: FADD.D
        rddata[t].u64 = rv_fadd_d(rsdata[t][0].u64, rsdata[t][1].u64, frm, &fflags);
        trace->fpu_type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x04: { // RV32F: FSUB.S
        rddata[t].u64 = nan_box(rv_fsub_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), frm, &fflags));
        trace->fpu_type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x05: { // RV32D: FSUB.D
        rddata[t].u64 = rv_fsub_d(rsdata[t][0].u64, rsdata[t][1].u64, frm, &fflags);
        trace->fpu_type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x08: { // RV32F: FMUL.S
        rddata[t].u64 = nan_box(rv_fmul_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), frm, &fflags));
        trace->fpu_type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x09: { // RV32D: FMUL.D
        rddata[t].u64 = rv_fmul_d(rsdata[t][0].u64, rsdata[t][1].u64, frm, &fflags);
        trace->fpu_type = FpuType::FMA;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x0c: { // RV32F: FDIV.S
        rddata[t].u64 = nan_box(rv_fdiv_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), frm, &fflags));
        trace->fpu_type = FpuType::FDIV;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x0d: { // RV32D: FDIV.D
        rddata[t].u64 = rv_fdiv_d(rsdata[t][0].u64, rsdata[t][1].u64, frm, &fflags);
        trace->fpu_type = FpuType::FDIV;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x10: {
        switch (func3) {            
        case 0: // RV32F: FSGNJ.S
          rddata[t].u64 = nan_box(rv_fsgnj_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64)));
          break;          
        case 1: // RV32F: FSGNJN.S
          rddata[t].u64 = nan_box(rv_fsgnjn_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64)));
          break;          
        case 2: // RV32F: FSGNJX.S
          rddata[t].u64 = nan_box(rv_fsgnjx_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64)));
          break;
        }
        trace->fpu_type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x11: {
        switch (func3) {            
        case 0: // RV32D: FSGNJ.D
          rddata[t].u64 = rv_fsgnj_d(rsdata[t][0].u64, rsdata[t][1].u64);
          break;          
        case 1: // RV32D: FSGNJN.D
          rddata[t].u64 = rv_fsgnjn_d(rsdata[t][0].u64, rsdata[t][1].u64);
          break;          
        case 2: // RV32D: FSGNJX.D
          rddata[t].u64 = rv_fsgnjx_d(rsdata[t][0].u64, rsdata[t][1].u64);
          break;
        }
        trace->fpu_type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;
      }
      case 0x14: {   
        if (func3) {
          // RV32F: FMAX.S
          rddata[t].u64 = nan_box(rv_fmax_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), &fflags));
        } else {
          // RV32F: FMIN.S
          rddata[t].u64 = nan_box(rv_fmin_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), &fflags));
        }
        trace->fpu_type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);        
        break;
      }
      case 0x15: {            
        if (func3) {
          // RV32D: FMAX.D
          rddata[t].u64 = rv_fmax_d(rsdata[t][0].u64, rsdata[t][1].u64, &fflags);
        } else {
          // RV32D: FMIN.D
          rddata[t].u64 = rv_fmin_d(rsdata[t][0].u64, rsdata[t][1].u64, &fflags);
        }
        trace->fpu_type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);        
        break;
      }
      case 0x20: {
        // RV32D: FCVT.S.D
        rddata[t].u64 = nan_box(rv_dtof(rsdata[t][0].u64));
        trace->fpu_type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);        
        break;
      }
      case 0x21: {
        // RV32D: FCVT.D.S
        rddata[t].u64 = rv_ftod(check_boxing(rsdata[t][0].u64));
        trace->fpu_type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);        
        break;
      }
      case 0x2c: { // RV32F: FSQRT.S
        rddata[t].u64 = nan_box(rv_fsqrt_s(check_boxing(rsdata[t][0].u64), frm, &fflags));
        trace->fpu_type = FpuType::FSQRT;
        trace->used_fregs.set(rsrc0);
        break;
      }
      case 0x2d: { // RV32D: FSQRT.D
        rddata[t].u64 = rv_fsqrt_d(rsdata[t][0].u64, frm, &fflags);
        trace->fpu_type = FpuType::FSQRT;
        trace->used_fregs.set(rsrc0);
        break;  
      }     
      case 0x50: {      
        switch (func3) {              
        case 0:
          // RV32F: FLE.S
          rddata[t].i = rv_fle_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), &fflags);    
          break;              
        case 1:
          // RV32F: FLT.S
          rddata[t].i = rv_flt_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), &fflags);
          break;              
        case 2:
          // RV32F: FEQ.S
          rddata[t].i = rv_feq_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), &fflags);
          break;
        } 
        trace->fpu_type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break; 
      }
      case 0x51: {           
        switch (func3) {              
        case 0:
          // RV32D: FLE.D
          rddata[t].i = rv_fle_d(rsdata[t][0].u64, rsdata[t][1].u64, &fflags);    
          break;              
        case 1:
          // RV32D: FLT.D
          rddata[t].i = rv_flt_d(rsdata[t][0].u64, rsdata[t][1].u64, &fflags);
          break;              
        case 2:
          // RV32D: FEQ.D
          rddata[t].i = rv_feq_d(rsdata[t][0].u64, rsdata[t][1].u64, &fflags);
          break;
        } 
        trace->fpu_type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        trace->used_fregs.set(rsrc1);
        break;  
      }    
      case 0x60: {
        switch (rsrc1) {
        case 0: 
          // RV32F: FCVT.W.S
          rddata[t].i = sext((uint64_t)rv_ftoi_s(check_boxing(rsdata[t][0].u64), frm, &fflags), 32);
          break;
        case 1:
          // RV32F: FCVT.WU.S
          rddata[t].i = sext((uint64_t)rv_ftou_s(check_boxing(rsdata[t][0].u64), frm, &fflags), 32);
          break;
        case 2:
          // RV64F: FCVT.L.S
          rddata[t].i = rv_ftol_s(check_boxing(rsdata[t][0].u64), frm, &fflags);
          break;
        case 3:
          // RV64F: FCVT.LU.S
          rddata[t].i = rv_ftolu_s(check_boxing(rsdata[t][0].u64), frm, &fflags);
          break;
        }
        trace->fpu_type = FpuType::FCVT;
        trace->used_fregs.set(rsrc0);
        break;
      }
      case 0x61: {
        switch (rsrc1) {
        case 0: 
          // RV32D: FCVT.W.D
          rddata[t].i = sext((uint64_t)rv_ftoi_d(rsdata[t][0].u64, frm, &fflags), 32);
          break;
        case 1:
          // RV32D: FCVT.WU.D
          rddata[t].i = sext((uint64_t)rv_ftou_d(rsdata[t][0].u64, frm, &fflags), 32);
          break;
        case 2:
          // RV64D: FCVT.L.D
          rddata[t].i = rv_ftol_d(rsdata[t][0].u64, frm, &fflags);
          break;
        case 3:
          // RV64D: FCVT.LU.D
          rddata[t].i = rv_ftolu_d(rsdata[t][0].u64, frm, &fflags);
          break;
        }
        trace->fpu_type = FpuType::FCVT;
        trace->used_fregs.set(rsrc0);
        break;
      }  
      case 0x68: {
        switch (rsrc1) {
        case 0: 
          // RV32F: FCVT.S.W
          rddata[t].u64 = nan_box(rv_itof_s(rsdata[t][0].i, frm, &fflags));
          break;
        case 1:
          // RV32F: FCVT.S.WU
          rddata[t].u64 = nan_box(rv_utof_s(rsdata[t][0].i, frm, &fflags));
          break;
        case 2:
          // RV64F: FCVT.S.L
          rddata[t].u64 = nan_box(rv_ltof_s(rsdata[t][0].i, frm, &fflags));
          break;
        case 3:
          // RV64F: FCVT.S.LU
          rddata[t].u64 = nan_box(rv_lutof_s(rsdata[t][0].i, frm, &fflags));
          break;
        }
        trace->fpu_type = FpuType::FCVT;
        trace->used_iregs.set(rsrc0);
        break;
      }
      case 0x69: {
        switch (rsrc1) {
        case 0: 
          // RV32D: FCVT.D.W
          rddata[t].u64 = rv_itof_d(rsdata[t][0].i, frm, &fflags);
          break;
        case 1:
          // RV32D: FCVT.D.WU
          rddata[t].u64 = rv_utof_d(rsdata[t][0].i, frm, &fflags);
          break;
        case 2:
          // RV64D: FCVT.D.L
          rddata[t].u64 = rv_ltof_d(rsdata[t][0].i, frm, &fflags);
          break;
        case 3:
          // RV64D: FCVT.D.LU
          rddata[t].u64 = rv_lutof_d(rsdata[t][0].i, frm, &fflags);
          break;
        }
        trace->fpu_type = FpuType::FCVT;
        trace->used_iregs.set(rsrc0);
        break;
      }
      case 0x70: {     
        if (func3) {
          // RV32F: FCLASS.S
          rddata[t].i = rv_fclss_s(check_boxing(rsdata[t][0].u64));
        } else {          
          // RV32F: FMV.X.S
          uint32_t result = (uint32_t)rsdata[t][0].u64;
          rddata[t].i = sext((uint64_t)result, 32);
        }        
        trace->fpu_type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        break;
      }
      case 0x71: {    
        if (func3) {
          // RV32D: FCLASS.D
          rddata[t].i = rv_fclss_d(rsdata[t][0].u64);
        } else {          
          // RV64D: FMV.X.D
          rddata[t].i = rsdata[t][0].u64;
        }        
        trace->fpu_type = FpuType::FNCP;
        trace->used_fregs.set(rsrc0);
        break;
      }
      case 0x78: { // RV32F: FMV.S.X
        rddata[t].u64 = nan_box((uint32_t)rsdata[t][0].i);
        trace->fpu_type = FpuType::FNCP;
        trace->used_iregs.set(rsrc0);
        break;
      }
      case 0x79: { // RV64D: FMV.D.X
        rddata[t].u64 = rsdata[t][0].i;
        trace->fpu_type = FpuType::FNCP;
        trace->used_iregs.set(rsrc0);
        break;
      }
      }
      update_fcrs(fflags, core_, t, warp_id_);
    }
    rd_write = true;
    break;
  }
  case FMADD:      
  case FMSUB:      
  case FMNMADD:
  case FMNMSUB: {
    trace->fpu_type = FpuType::FMA;
    trace->used_fregs.set(rsrc0);
    trace->used_fregs.set(rsrc1);
    trace->used_fregs.set(rsrc2);
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!tmask_.test(t))
        continue;
      uint32_t frm = get_fpu_rm(func3, core_, t, warp_id_);
      uint32_t fflags = 0;
      switch (opcode) {
      case FMADD:
        if (func2)
          // RV32D: FMADD.D
          rddata[t].u64 = rv_fmadd_d(rsdata[t][0].u64, rsdata[t][1].u64, rsdata[t][2].u64, frm, &fflags);
        else
          // RV32F: FMADD.S
          rddata[t].u64 = nan_box(rv_fmadd_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), check_boxing(rsdata[t][2].u64), frm, &fflags));
        break;
      case FMSUB:
        if (func2)
          // RV32D: FMSUB.D
          rddata[t].u64 = rv_fmsub_d(rsdata[t][0].u64, rsdata[t][1].u64, rsdata[t][2].u64, frm, &fflags);
        else 
          // RV32F: FMSUB.S
          rddata[t].u64 = nan_box(rv_fmsub_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), check_boxing(rsdata[t][2].u64), frm, &fflags));
        break;
      case FMNMADD:
        if (func2)
          // RV32D: FNMADD.D
          rddata[t].u64 = rv_fnmadd_d(rsdata[t][0].u64, rsdata[t][1].u64, rsdata[t][2].u64, frm, &fflags);
        else
          // RV32F: FNMADD.S
          rddata[t].u64 = nan_box(rv_fnmadd_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), check_boxing(rsdata[t][2].u64), frm, &fflags));
        break; 
      case FMNMSUB:
        if (func2)
          // RV32D: FNMSUB.D
          rddata[t].u64 = rv_fnmsub_d(rsdata[t][0].u64, rsdata[t][1].u64, rsdata[t][2].u64, frm, &fflags);
        else
          // RV32F: FNMSUB.S
          rddata[t].u64 = nan_box(rv_fnmsub_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), check_boxing(rsdata[t][2].u64), frm, &fflags));
        break;
      default:
        break;
      }              
      update_fcrs(fflags, core_, t, warp_id_);
    }
    rd_write = true;
    break;
  }
  case EXT1: {   
    switch (func7) {
    case 0: {
      switch (func3) {
      case 0: {
        // TMC  
        trace->exe_type = ExeType::SFU;     
        trace->sfu_type = SfuType::TMC;
        trace->used_iregs.set(rsrc0);
        trace->fetch_stall = true;
        next_tmask.reset();
        for (uint32_t t = 0; t < num_threads; ++t) {
          next_tmask.set(t, rsdata.at(thread_start)[0].i & (1 << t));
        }
      } break;
      case 1: {
        // WSPAWN
        trace->exe_type = ExeType::SFU;
        trace->sfu_type = SfuType::WSPAWN;
        trace->used_iregs.set(rsrc0);
        trace->used_iregs.set(rsrc1);
        trace->fetch_stall = true;
        core_->wspawn(rsdata.at(thread_start)[0].i, rsdata.at(thread_start)[1].i);
      } break;
      case 2: {
        // SPLIT
        trace->exe_type = ExeType::SFU;
        trace->sfu_type = SfuType::SPLIT;
        trace->used_iregs.set(rsrc0);
        trace->fetch_stall = true;

        ThreadMask then_tmask, else_tmask;
        for (uint32_t t = 0; t < num_threads; ++t) {
          auto cond = ireg_file_.at(t).at(rsrc0);
          then_tmask[t] = tmask_.test(t) && cond;
          else_tmask[t] = tmask_.test(t) && !cond;
        }

        bool is_divergent = then_tmask.any() && else_tmask.any();
        if (is_divergent) {             
          if (ipdom_stack_.size() == arch_.ipdom_size()) {
            std::cout << "IPDOM stack is full! size=" << std::dec << ipdom_stack_.size() << ", PC=0x" << std::hex << PC_ << " (#" << std::dec << trace->uuid << ")\n" << std::dec << std::flush;
            std::abort();
          }
          // set new thread mask
          next_tmask = then_tmask;
          // push reconvergence thread mask onto the stack
          ipdom_stack_.emplace(tmask_);
          // push else's thread mask onto the stack
          ipdom_stack_.emplace(else_tmask, next_pc);
        }
        // return divergent state
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          rddata[t].i = is_divergent;
        }
        rd_write = true;
      } break;
      case 3: {
        // JOIN
        trace->exe_type = ExeType::SFU;
        trace->sfu_type = SfuType::JOIN;
        trace->used_iregs.set(rsrc0);
        trace->fetch_stall = true;

        int is_divergent = ireg_file_.at(thread_start).at(rsrc0);
        if (is_divergent != 0) {
          if (ipdom_stack_.empty()) {
            std::cout << "IPDOM stack is empty!\n" << std::flush;
            std::abort();
          }
          next_tmask = ipdom_stack_.top().tmask;
          if (!ipdom_stack_.top().fallthrough) {
            next_pc = ipdom_stack_.top().PC;
          }          
          ipdom_stack_.pop();
        }    
      } break;
      case 4: {
        // BAR
        trace->exe_type = ExeType::SFU; 
        trace->sfu_type = SfuType::BAR;
        trace->used_iregs.set(rsrc0);
        trace->used_iregs.set(rsrc1);
        trace->fetch_stall = true;
        trace->data = std::make_shared<SFUTraceData>(rsdata[thread_start][0].i, rsdata[thread_start][1].i);
      } break;
      case 5: {
        // PRED  
        trace->exe_type = ExeType::SFU;     
        trace->sfu_type = SfuType::PRED;
        trace->used_iregs.set(rsrc0);
        trace->used_iregs.set(rsrc1);
        trace->fetch_stall = true;
        ThreadMask pred;
        for (uint32_t t = 0; t < num_threads; ++t) {
          pred[t] = tmask_.test(t) && (ireg_file_.at(t).at(rsrc0) & 0x1);
        }
        if (pred.any()) {
          next_tmask &= pred;
        } else {
          next_tmask = ireg_file_.at(thread_start).at(rsrc1);
        }      
      } break;
      default:
        std::abort();
      }
    } break;
    default:
      std::abort();
    }
  } break;
  case EXT2: {    
    switch (func3) {
    case 1:
      switch (func2) {
      case 0: { // CMOV
        trace->exe_type = ExeType::SFU;
        trace->sfu_type = SfuType::CMOV;
        trace->used_iregs.set(rsrc0);
        trace->used_iregs.set(rsrc1);
        trace->used_iregs.set(rsrc2);
        for (uint32_t t = thread_start; t < num_threads; ++t) {
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
    uint32_t VLEN = arch_.vsize() * 8;
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

      auto s0 = rsdata[0][0].u;
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
    auto type = instr.getRDType();    
    switch (type) {
    case RegType::Integer:      
      if (rdest) {   
        DPH(2, "Dest Reg: " << type << std::dec << rdest << "={");    
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
        assert(rdest != 0);
      } else {
        // disable writes to x0
        trace->wb = false;
      }
      break;
    case RegType::Float:
      DPH(2, "Dest Reg: " << type << std::dec << rdest << "={");
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (t) DPN(2, ", ");
        if (!tmask_.test(t)) {
          DPN(2, "-");
          continue;            
        }
        freg_file_.at(t)[rdest] = rddata[t].u64;        
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

  PC_ += 4;
  if (PC_ != next_pc) {
    DP(3, "*** Next PC=0x" << std::hex << next_pc << std::dec);
    PC_ = next_pc;
  }
  if (tmask_ != next_tmask) {    
    DPH(3, "*** New Tmask=");
    for (uint32_t i = 0; i < num_threads; ++i)
      DPN(3, next_tmask.test(i));
    DPN(3, std::endl);
    tmask_ = next_tmask;
    if (!next_tmask.any()) {
      core_->active_warps_.reset(warp_id_);
    }
  }
}