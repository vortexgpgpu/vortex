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
#include "emulator.h"
#include "instr.h"
#include "core.h"
#include "processor_impl.h"
#include "VX_types.h"

using namespace vortex;

inline uint64_t nan_box(uint32_t value) {
  return value | 0xffffffff00000000;
}

inline bool is_nan_boxed(uint64_t value) {
  return (uint32_t(value >> 32) == 0xffffffff);
}

inline int64_t check_boxing(int64_t a) {
  if (is_nan_boxed(a))
    return a;
  return nan_box(0x7fc00000); // NaN
}

void Emulator::execute(const Instr &instr, uint32_t wid, instr_trace_t *trace) {
  auto& warp = warps_.at(wid);
  assert(warp.tmask.any());

  // initialize instruction trace
  trace->cid   = core_->id();
  trace->wid   = wid;
  trace->PC    = warp.PC;
  trace->tmask = warp.tmask;
  trace->dst_reg = {instr.getRDType(), instr.getRDest()};

  auto next_pc = warp.PC + 4;
  auto next_tmask = warp.tmask;

  auto opcode = instr.getOpcode();
  auto func2  = instr.getFunc2();
  auto func3  = instr.getFunc3();
  auto func7  = instr.getFunc7();
  auto rdest  = instr.getRDest();
  auto rsrc0  = instr.getRSrc(0);
  auto rsrc1  = instr.getRSrc(1);
  auto rsrc2  = instr.getRSrc(2);
  auto immsrc = sext((Word)instr.getImm(), 32);

  auto num_threads = arch_.num_threads();

  uint32_t thread_start = 0;
  for (; thread_start < num_threads; ++thread_start) {
      if (warp.tmask.test(thread_start))
        break;
  }

  int32_t thread_last = num_threads - 1;
  for (; thread_last >= 0; --thread_last) {
      if (warp.tmask.test(thread_last))
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
        DPH(2, "Src" << i << " Reg: " << type << reg << "={");
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (t) DPN(2, ", ");
          if (!warp.tmask.test(t)) {
            DPN(2, "-");
            continue;
          }
          rsdata[t][i].u = warp.ireg_file.at(t)[reg];
          DPN(2, "0x" << std::hex << rsdata[t][i].i << std::dec);
        }
        DPN(2, "}" << std::endl);
        break;
      case RegType::Float:
        DPH(2, "Src" << i << " Reg: " << type << reg << "={");
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (t) DPN(2, ", ");
          if (!warp.tmask.test(t)) {
            DPN(2, "-");
            continue;
          }
          rsdata[t][i].u64 = warp.freg_file.at(t)[reg];
          DPN(2, "0x" << std::hex << rsdata[t][i].f << std::dec);
        }
        DPN(2, "}" << std::endl);
        break;
      case RegType::Vector:
        break;
      default:
        break;
      }
    }
  }

  bool rd_write = false;

  switch (opcode) {
  case Opcode::LUI: {
    // RV32I: LUI
    trace->fu_type = FUType::ALU;
    trace->alu_type = AluType::ARITH;
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!warp.tmask.test(t))
        continue;
      rddata[t].i = immsrc;
    }
    rd_write = true;
    break;
  }
  case Opcode::AUIPC: {
    // RV32I: AUIPC
    trace->fu_type = FUType::ALU;
    trace->alu_type = AluType::ARITH;
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!warp.tmask.test(t))
        continue;
      rddata[t].i = immsrc + warp.PC;
    }
    rd_write = true;
    break;
  }
  case Opcode::R: {
    trace->fu_type = FUType::ALU;
    trace->alu_type = AluType::ARITH;
    trace->src_regs[0] = {RegType::Integer, rsrc0};
    trace->src_regs[1] = {RegType::Integer, rsrc1};
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!warp.tmask.test(t))
        continue;
      if (func7 == 0x7) {
        auto value = rsdata[t][0].i;
        auto cond = rsdata[t][1].i;
        if (func3 == 0x5) {
          // CZERO.EQZ
          rddata[t].i = (cond == 0) ? 0 : value;
          trace->alu_type = AluType::ARITH;
        } else
        if (func3 == 0x7) {
          // CZERO.NEZ
          rddata[t].i = (cond != 0) ? 0 : value;
          trace->alu_type = AluType::ARITH;
        } else {
          std::abort();
        }
      } else
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
          if (func7 & 0x20) {
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
          if (func7 & 0x20) {
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
  case Opcode::I: {
    trace->fu_type = FUType::ALU;
    trace->alu_type = AluType::ARITH;
    trace->src_regs[0] = {RegType::Integer, rsrc0};
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!warp.tmask.test(t))
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
        if (func7 & 0x20) {
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
  case Opcode::R_W: {
    trace->fu_type = FUType::ALU;
    trace->alu_type = AluType::ARITH;
    trace->src_regs[0] = {RegType::Integer, rsrc0};
    trace->src_regs[1] = {RegType::Integer, rsrc1};
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!warp.tmask.test(t))
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
          if (func7 & 0x20){
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
          if (func7 & 0x20) {
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
  case Opcode::I_W: {
    trace->fu_type = FUType::ALU;
    trace->alu_type = AluType::ARITH;
    trace->src_regs[0] = {RegType::Integer, rsrc0};
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!warp.tmask.test(t))
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
          if (func7 & 0x20) {
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
  case Opcode::B: {
    trace->fu_type = FUType::ALU;
    trace->alu_type = AluType::BRANCH;
    trace->src_regs[0] = {RegType::Integer, rsrc0};
    trace->src_regs[1] = {RegType::Integer, rsrc1};
    bool all_taken = false;
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!warp.tmask.test(t))
        continue;
      bool curr_taken = false;
      switch (func3) {
      case 0: {
        // RV32I: BEQ
        if (rsdata[t][0].i == rsdata[t][1].i) {
          next_pc = warp.PC + immsrc;
          curr_taken = true;
        }
        break;
      }
      case 1: {
        // RV32I: BNE
        if (rsdata[t][0].i != rsdata[t][1].i) {
          next_pc = warp.PC + immsrc;
          curr_taken = true;
        }
        break;
      }
      case 4: {
        // RV32I: BLT
        if (rsdata[t][0].i < rsdata[t][1].i) {
          next_pc = warp.PC + immsrc;
          curr_taken = true;
        }
        break;
      }
      case 5: {
        // RV32I: BGE
        if (rsdata[t][0].i >= rsdata[t][1].i) {
          next_pc = warp.PC + immsrc;
          curr_taken = true;
        }
        break;
      }
      case 6: {
        // RV32I: BLTU
        if (rsdata[t][0].u < rsdata[t][1].u) {
          next_pc = warp.PC + immsrc;
          curr_taken = true;
        }
        break;
      }
      case 7: {
        // RV32I: BGEU
        if (rsdata[t][0].u >= rsdata[t][1].u) {
          next_pc = warp.PC + immsrc;
          curr_taken = true;
        }
        break;
      }
      default:
        std::abort();
      }
      if (t == thread_start) {
        all_taken = curr_taken;
      } else {
        if (all_taken != curr_taken) {
          std::cout << "divergent branch! PC=0x" << std::hex << warp.PC << std::dec << " (#" << trace->uuid << ")\n" << std::flush;
          std::abort();
        }
      }
    }
    trace->fetch_stall = true;
    break;
  }
  case Opcode::JAL: {
    // RV32I: JAL
    trace->fu_type = FUType::ALU;
    trace->alu_type = AluType::BRANCH;
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!warp.tmask.test(t))
        continue;
      rddata[t].i = next_pc;
    }
    next_pc = warp.PC + immsrc;
    trace->fetch_stall = true;
    rd_write = true;
    break;
  }
  case Opcode::JALR: {
    // RV32I: JALR
    trace->fu_type = FUType::ALU;
    trace->alu_type = AluType::BRANCH;
    trace->src_regs[0] = {RegType::Integer, rsrc0};
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!warp.tmask.test(t))
        continue;
      rddata[t].i = next_pc;
    }
    next_pc = rsdata[thread_last][0].i + immsrc;
    trace->fetch_stall = true;
    rd_write = true;
    break;
  }
  case Opcode::L:
  case Opcode::FL: {
    trace->fu_type = FUType::LSU;
    trace->lsu_type = LsuType::LOAD;
    trace->src_regs[0] = {RegType::Integer, rsrc0};
    auto trace_data = std::make_shared<LsuTraceData>(num_threads);
    trace->data = trace_data;
    if ((opcode == Opcode::L )
     || (opcode == Opcode::FL && func3 == 2)
     || (opcode == Opcode::FL && func3 == 3)) {
      uint32_t data_bytes = 1 << (func3 & 0x3);
      uint32_t data_width = 8 * data_bytes;
      for (uint32_t t = thread_start; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint64_t mem_addr = rsdata[t][0].i + immsrc;         
        uint64_t read_data = 0;
        this->dcache_read(&read_data, mem_addr, data_bytes);
        trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
        switch (func3) {
        case 0: // RV32I: LB
        case 1: // RV32I: LH
          rddata[t].i = sext((Word)read_data, data_width);
          break;
        case 2:
          if (opcode == Opcode::L) {
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
      rd_write = true;
    } else {
      loadVector(instr, wid, rsdata);
    }
    break;
  }
  case Opcode::S:
  case Opcode::FS: {
    trace->fu_type = FUType::LSU;
    trace->lsu_type = LsuType::STORE;
    auto data_type = (opcode == Opcode::FS) ? RegType::Float : RegType::Integer;
    trace->src_regs[0] = {RegType::Integer, rsrc0};
    trace->src_regs[1] = {data_type, rsrc1};
    auto trace_data = std::make_shared<LsuTraceData>(num_threads);
    trace->data = trace_data;
    if ((opcode == Opcode::S)
     || (opcode == Opcode::FS && func3 == 2)
     || (opcode == Opcode::FS && func3 == 3)) {
      uint32_t data_bytes = 1 << (func3 & 0x3);
      for (uint32_t t = thread_start; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        uint64_t mem_addr = rsdata[t][0].i + immsrc;
        uint64_t write_data = rsdata[t][1].u64;
        trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
        switch (func3) {
        case 0:
        case 1:
        case 2:
        case 3:
          this->dcache_write(&write_data, mem_addr, data_bytes);  
          break;
        default:
          std::abort();
        }
      }
    } else {
      storeVector(instr, wid, rsdata);
    }
    break;
  }
  case Opcode::AMO: {
    trace->fu_type = FUType::LSU;
    trace->lsu_type = LsuType::LOAD;
    trace->src_regs[0] = {RegType::Integer, rsrc0};
    trace->src_regs[1] = {RegType::Integer, rsrc1};
    auto trace_data = std::make_shared<LsuTraceData>(num_threads);
    trace->data = trace_data;
    auto amo_type = func7 >> 2;
    uint32_t data_bytes = 1 << (func3 & 0x3);
    uint32_t data_width = 8 * data_bytes;
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!warp.tmask.test(t))
        continue;
      uint64_t mem_addr = rsdata[t][0].u;
      trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
      if (amo_type == 0x02) { // LR
        uint64_t read_data = 0;
        this->dcache_read(&read_data, mem_addr, data_bytes);
        this->dcache_amo_reserve(mem_addr);
        rddata[t].i = sext((Word)read_data, data_width);
      } else
      if (amo_type == 0x03) { // SC
        if (this->dcache_amo_check(mem_addr)) {
          this->dcache_write(&rsdata[t][1].u64, mem_addr, data_bytes);
          rddata[t].i = 0;
        } else {
          rddata[t].i = 1;
        }
      } else {
        uint64_t read_data = 0;
        this->dcache_read(&read_data, mem_addr, data_bytes);
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
        this->dcache_write(&result, mem_addr, data_bytes);
        rddata[t].i = read_data_i;
      }
    }
    rd_write = true;
    break;
  }
  case Opcode::SYS: {
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!warp.tmask.test(t))
        continue;
      uint32_t csr_addr = immsrc;
      Word csr_value;
      if (func3 == 0) {
        trace->fu_type = FUType::ALU;
        trace->alu_type = AluType::SYSCALL;
        trace->fetch_stall = true;
        switch (csr_addr) {
        case 0x000: // RV32I: ECALL
          this->trigger_ecall(); // Re-added for riscv-vector test functionality
          break;
        case 0x001: // RV32I: EBREAK
          this->trigger_ebreak(); // Re-added for riscv-vector test functionality
          break;
        case 0x002: // RV32I: URET
        case 0x102: // RV32I: SRET
        case 0x302: // RV32I: MRET
          break;
        default:
          std::abort();
        }
      } else {
        trace->fu_type = FUType::SFU;
        // stall the fetch stage for FPU CSRs
        trace->fetch_stall = (csr_addr <= VX_CSR_FCSR);
        csr_value = this->get_csr(csr_addr, t, wid);
        switch (func3) {
        case 1: {
          // RV32I: CSRRW
          rddata[t].i = csr_value;
          this->set_csr(csr_addr, rsdata[t][0].i, t, wid);
          trace->src_regs[0] = {RegType::Integer, rsrc0};
          trace->sfu_type = SfuType::CSRRW;
          rd_write = true;
          break;
        }
        case 2: {
          // RV32I: CSRRS
          rddata[t].i = csr_value;
          if (rsdata[t][0].i != 0) {
            this->set_csr(csr_addr, csr_value | rsdata[t][0].i, t, wid);
          }
          trace->src_regs[0] = {RegType::Integer, rsrc0};
          trace->sfu_type = SfuType::CSRRS;
          rd_write = true;
          break;
        }
        case 3: {
          // RV32I: CSRRC
          rddata[t].i = csr_value;
          if (rsdata[t][0].i != 0) {
            this->set_csr(csr_addr, csr_value & ~rsdata[t][0].i, t, wid);
          }
          trace->src_regs[0] = {RegType::Integer, rsrc0};
          trace->sfu_type = SfuType::CSRRC;
          rd_write = true;
          break;
        }
        case 5: {
          // RV32I: CSRRWI
          rddata[t].i = csr_value;
          this->set_csr(csr_addr, rsrc0, t, wid);
          trace->sfu_type = SfuType::CSRRW;
          rd_write = true;
          break;
        }
        case 6: {
          // RV32I: CSRRSI;
          rddata[t].i = csr_value;
          if (rsrc0 != 0) {
            this->set_csr(csr_addr, csr_value | rsrc0, t, wid);
          }
          trace->sfu_type = SfuType::CSRRS;
          rd_write = true;
          break;
        }
        case 7: {
          // RV32I: CSRRCI
          rddata[t].i = csr_value;
          if (rsrc0 != 0) {
            this->set_csr(csr_addr, csr_value & ~rsrc0, t, wid);
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
  case Opcode::FENCE: {
    // RV32I: FENCE
    trace->fu_type = FUType::LSU;
    trace->lsu_type = LsuType::FENCE;
    break;
  }
  case Opcode::FCI: {
    trace->fu_type = FUType::FPU;
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!warp.tmask.test(t))
        continue;
      uint32_t frm = (func3 == 0x7) ? this->get_csr(VX_CSR_FRM, t, wid) : func3;
      uint32_t fflags = 0;
      switch (func7) {
      case 0x00: { // RV32F: FADD.S
        rddata[t].u64 = nan_box(rv_fadd_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), frm, &fflags));
        trace->fpu_type = FpuType::FMA;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x01: { // RV32D: FADD.D
        rddata[t].u64 = rv_fadd_d(rsdata[t][0].u64, rsdata[t][1].u64, frm, &fflags);
        trace->fpu_type = FpuType::FMA;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x04: { // RV32F: FSUB.S
        rddata[t].u64 = nan_box(rv_fsub_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), frm, &fflags));
        trace->fpu_type = FpuType::FMA;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x05: { // RV32D: FSUB.D
        rddata[t].u64 = rv_fsub_d(rsdata[t][0].u64, rsdata[t][1].u64, frm, &fflags);
        trace->fpu_type = FpuType::FMA;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x08: { // RV32F: FMUL.S
        rddata[t].u64 = nan_box(rv_fmul_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), frm, &fflags));
        trace->fpu_type = FpuType::FMA;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x09: { // RV32D: FMUL.D
        rddata[t].u64 = rv_fmul_d(rsdata[t][0].u64, rsdata[t][1].u64, frm, &fflags);
        trace->fpu_type = FpuType::FMA;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x0c: { // RV32F: FDIV.S
        rddata[t].u64 = nan_box(rv_fdiv_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), frm, &fflags));
        trace->fpu_type = FpuType::FDIV;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x0d: { // RV32D: FDIV.D
        rddata[t].u64 = rv_fdiv_d(rsdata[t][0].u64, rsdata[t][1].u64, frm, &fflags);
        trace->fpu_type = FpuType::FDIV;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
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
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
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
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
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
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
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
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x20: {
        // RV32D: FCVT.S.D
        rddata[t].u64 = nan_box(rv_dtof(rsdata[t][0].u64));
        trace->fpu_type = FpuType::FNCP;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        break;
      }
      case 0x21: {
        // RV32D: FCVT.D.S
        rddata[t].u64 = rv_ftod(check_boxing(rsdata[t][0].u64));
        trace->fpu_type = FpuType::FNCP;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        break;
      }
      case 0x2c: { // RV32F: FSQRT.S
        rddata[t].u64 = nan_box(rv_fsqrt_s(check_boxing(rsdata[t][0].u64), frm, &fflags));
        trace->fpu_type = FpuType::FSQRT;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        break;
      }
      case 0x2d: { // RV32D: FSQRT.D
        rddata[t].u64 = rv_fsqrt_d(rsdata[t][0].u64, frm, &fflags);
        trace->fpu_type = FpuType::FSQRT;
        trace->src_regs[0] = {RegType::Float, rsrc0};
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
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
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
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
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
        trace->src_regs[0] = {RegType::Float, rsrc0};
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
        trace->src_regs[0] = {RegType::Float, rsrc0};
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
        trace->src_regs[0] = {RegType::Integer, rsrc0};
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
        trace->src_regs[0] = {RegType::Integer, rsrc0};
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
        trace->src_regs[0] = {RegType::Float, rsrc0};
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
        trace->src_regs[0] = {RegType::Float, rsrc0};
        break;
      }
      case 0x78: { // RV32F: FMV.S.X
        rddata[t].u64 = nan_box((uint32_t)rsdata[t][0].i);
        trace->fpu_type = FpuType::FNCP;
        trace->src_regs[0] = {RegType::Integer, rsrc0};
        break;
      }
      case 0x79: { // RV64D: FMV.D.X
        rddata[t].u64 = rsdata[t][0].i;
        trace->fpu_type = FpuType::FNCP;
        trace->src_regs[0] = {RegType::Integer, rsrc0};
        break;
      }
      }
      if (fflags) {
        this->set_csr(VX_CSR_FCSR, this->get_csr(VX_CSR_FCSR, t, wid) | fflags, t, wid);
        this->set_csr(VX_CSR_FFLAGS, this->get_csr(VX_CSR_FFLAGS, t, wid) | fflags, t, wid);
      }
    }
    rd_write = true;
    break;
  }
  case Opcode::FMADD:
  case Opcode::FMSUB:
  case Opcode::FMNMADD:
  case Opcode::FMNMSUB: {
    trace->fpu_type = FpuType::FMA;
    trace->src_regs[0] = {RegType::Float, rsrc0};
    trace->src_regs[1] = {RegType::Float, rsrc1};
    trace->src_regs[2] = {RegType::Float, rsrc2};
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!warp.tmask.test(t))
        continue;
      uint32_t frm = this->get_fpu_rm(func3, t, wid);
      uint32_t fflags = 0;
      switch (opcode) {
      case Opcode::FMADD:
        if (func2)
          // RV32D: FMADD.D
          rddata[t].u64 = rv_fmadd_d(rsdata[t][0].u64, rsdata[t][1].u64, rsdata[t][2].u64, frm, &fflags);
        else
          // RV32F: FMADD.S
          rddata[t].u64 = nan_box(rv_fmadd_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), check_boxing(rsdata[t][2].u64), frm, &fflags));
        break;
      case Opcode::FMSUB:
        if (func2)
          // RV32D: FMSUB.D
          rddata[t].u64 = rv_fmsub_d(rsdata[t][0].u64, rsdata[t][1].u64, rsdata[t][2].u64, frm, &fflags);
        else
          // RV32F: FMSUB.S
          rddata[t].u64 = nan_box(rv_fmsub_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), check_boxing(rsdata[t][2].u64), frm, &fflags));
        break;
      case Opcode::FMNMADD:
        if (func2)
          // RV32D: FNMADD.D
          rddata[t].u64 = rv_fnmadd_d(rsdata[t][0].u64, rsdata[t][1].u64, rsdata[t][2].u64, frm, &fflags);
        else
          // RV32F: FNMADD.S
          rddata[t].u64 = nan_box(rv_fnmadd_s(check_boxing(rsdata[t][0].u64), check_boxing(rsdata[t][1].u64), check_boxing(rsdata[t][2].u64), frm, &fflags));
        break;
      case Opcode::FMNMSUB:
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
      if (fflags) {
        this->set_csr(VX_CSR_FCSR, this->get_csr(VX_CSR_FCSR, t, wid) | fflags, t, wid);
        this->set_csr(VX_CSR_FFLAGS, this->get_csr(VX_CSR_FFLAGS, t, wid) | fflags, t, wid);
      }
    }
    rd_write = true;
    break;
  }
  case Opcode::EXT1: {
    switch (func7) {
    case 0: {
      switch (func3) {
      case 0: {
        // TMC
        trace->fu_type = FUType::SFU;
        trace->sfu_type = SfuType::TMC;
        trace->src_regs[0] = {RegType::Integer, rsrc0};
        trace->fetch_stall = true;
        next_tmask.reset();
        for (uint32_t t = 0; t < num_threads; ++t) {
          next_tmask.set(t, rsdata.at(thread_last)[0].i & (1 << t));
        }
      } break;
      case 1: {
        // WSPAWN
        trace->fu_type = FUType::SFU;
        trace->sfu_type = SfuType::WSPAWN;
        trace->src_regs[0] = {RegType::Integer, rsrc0};
        trace->src_regs[1] = {RegType::Integer, rsrc1};
        trace->fetch_stall = true;
        trace->data = std::make_shared<SFUTraceData>(rsdata.at(thread_last)[0].i, rsdata.at(thread_last)[1].i);
      } break;
      case 2: {
        // SPLIT
        trace->fu_type = FUType::SFU;
        trace->sfu_type = SfuType::SPLIT;
        trace->src_regs[0] = {RegType::Integer, rsrc0};
        trace->fetch_stall = true;

        auto stack_size = warp.ipdom_stack.size();

        ThreadMask then_tmask, else_tmask;
        auto not_pred = (rsrc1 != 0);
        for (uint32_t t = 0; t < num_threads; ++t) {
          auto cond = (warp.ireg_file.at(t).at(rsrc0) & 0x1) ^ not_pred;
          then_tmask[t] = warp.tmask.test(t) && cond;
          else_tmask[t] = warp.tmask.test(t) && !cond;
        }

        bool is_divergent = then_tmask.any() && else_tmask.any();
        if (is_divergent) {
          if (stack_size == ipdom_size_) {
            std::cout << "IPDOM stack is full! size=" << stack_size << ", PC=0x" << std::hex << warp.PC << std::dec << " (#" << trace->uuid << ")\n" << std::flush;
            std::abort();
          }
          // set new thread mask to the larger set
          if (then_tmask.count() >= else_tmask.count()) {
            next_tmask = then_tmask;
          } else {
            next_tmask = else_tmask;
          }
          // push reconvergence and not-taken thread mask onto the stack
          auto ntaken_tmask = ~next_tmask & warp.tmask;
          warp.ipdom_stack.emplace(warp.tmask, ntaken_tmask, next_pc);
        }
        // return divergent state
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          rddata[t].i = stack_size;
        }
        rd_write = true;
      } break;
      case 3: {
        // JOIN
        trace->fu_type = FUType::SFU;
        trace->sfu_type = SfuType::JOIN;
        trace->src_regs[0] = {RegType::Integer, rsrc0};
        trace->fetch_stall = true;

        auto stack_ptr = warp.ireg_file.at(thread_last).at(rsrc0);
        if (stack_ptr != warp.ipdom_stack.size()) {
          if (warp.ipdom_stack.empty()) {
            std::cout << "IPDOM stack is empty!\n" << std::flush;
            std::abort();
          }
          if (warp.ipdom_stack.top().fallthrough) {
            next_tmask = warp.ipdom_stack.top().orig_tmask;
            warp.ipdom_stack.pop();
          } else {
            next_tmask = warp.ipdom_stack.top().else_tmask;
            next_pc = warp.ipdom_stack.top().PC;
            warp.ipdom_stack.top().fallthrough = true;
          }
        }
      } break;
      case 4: {
        // BAR
        trace->fu_type = FUType::SFU;
        trace->sfu_type = SfuType::BAR;
        trace->src_regs[0] = {RegType::Integer, rsrc0};
        trace->src_regs[1] = {RegType::Integer, rsrc1};
        trace->fetch_stall = true;
        trace->data = std::make_shared<SFUTraceData>(rsdata[thread_last][0].i, rsdata[thread_last][1].i);
      } break;
      case 5: {
        // PRED
        trace->fu_type = FUType::SFU;
        trace->sfu_type = SfuType::PRED;
        trace->src_regs[0] = {RegType::Integer, rsrc0};
        trace->src_regs[1] = {RegType::Integer, rsrc1};
        trace->fetch_stall = true;
        ThreadMask pred;
        auto not_pred = rdest & 0x1;
        for (uint32_t t = 0; t < num_threads; ++t) {
          auto cond = (warp.ireg_file.at(t).at(rsrc0) & 0x1) ^ not_pred;
          pred[t] = warp.tmask.test(t) && cond;
        }
        if (pred.any()) {
          next_tmask &= pred;
        } else {
          next_tmask = warp.ireg_file.at(thread_last).at(rsrc1);
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
  case Opcode::TCU:
  { //TODO - make it data-type flexible
    uint32_t mem_bytes = 1;
    DP(3, "mem_bytes=" << mem_bytes << std::endl);
    uint16_t tc_size = this->get_csr(VX_TC_SIZE, 0, wid);
    uint32_t TC_per_warp = this->get_csr(VX_TC_NUM, 0, wid);

    DP(3, "tc_size=" << tc_size << std::endl);
    DP(3, "TC_per_warp=" << TC_per_warp << std::endl);

    //Number of loads - dependant on the thread config
    uint32_t n_tiles = this->get_csr(VX_MAT_MUL_SIZE, 0, wid);  //CSR instruction before MLOAD will ensure that this csr has value
    int num_data_per_thread;
    int num_data_per_thread_st;
    uint32_t num_threads_actv;
    uint32_t num_threads_actv_st;
    uint32_t data_bytes_load;
    uint32_t data_bytes_store;
    uint32_t num_threads_per_tc = MAX (1, num_threads/TC_per_warp);

    //LOAD
    if(num_threads > tc_size*tc_size*n_tiles*TC_per_warp)
    {
      num_threads_actv = tc_size*tc_size*n_tiles*TC_per_warp;
      num_data_per_thread = 1;
    }
    else
    {
      num_threads_actv = num_threads;
      num_data_per_thread = (tc_size*tc_size*n_tiles)/num_threads_per_tc;
    }
    data_bytes_load = mem_bytes*num_data_per_thread;

    //STORE
    if(num_threads > tc_size*tc_size*TC_per_warp)
    {
      num_threads_actv_st = tc_size*tc_size*TC_per_warp;
      num_data_per_thread_st = 1;
    }
    else
    {
      num_threads_actv_st = num_threads;
      num_data_per_thread_st = (tc_size*tc_size)/num_threads_per_tc;
    }
    data_bytes_store = mem_bytes*num_data_per_thread_st;

    DP(3, "Num Tiles=" << n_tiles << std::endl);

    switch (func3) {
      case 0:
      { //Matrix Load

        DP (4, "TCU LOAD");
        trace->fu_type = FUType::LSU;
        trace->lsu_type = LsuType::TCU_LOAD;

        trace->src_regs[0] = {RegType::Integer, rsrc0};
        auto trace_data = std::make_shared<LsuTraceData>(num_threads);
        trace->data = trace_data;

        for (uint32_t t = thread_start; t < num_threads_actv; ++t)
        {
          if (!warp.tmask.test(t))
            continue;
          DP(3, "Thread ID" << t);

          uint32_t base_addr = rsdata[t][0].i ;
          trace_data->mem_addrs.at(t) = {base_addr, data_bytes_load};

          //Load A or B (depends on immsrc)
          int loop_offset = 0;
          DP(3, "n_tiles = " << n_tiles << "; num_data_per_thread = " << num_data_per_thread <<std::endl);
            for (int n=0; n<num_data_per_thread; n++)
            {
              Word* temp_ref = &(warp.ireg_file.at(t).at(rsrc0));
              this->dcache_read(temp_ref, (base_addr+(n*mem_bytes)+(loop_offset*mem_bytes)), mem_bytes);

              scratchpad[loop_offset + (immsrc*(n_tiles)*tc_size*tc_size) + (t*num_data_per_thread) + n] = *temp_ref;
              DP(3, "Scratchpad Index: " << loop_offset + (immsrc*(n_tiles)*tc_size*tc_size) + (t*num_data_per_thread) + n << ", Value: " << scratchpad[loop_offset + (immsrc*(n_tiles)*tc_size*tc_size) + (t*num_data_per_thread) + n]);
            }
        }
        rd_write = true;
      } break;
      case 1:
      {
        DP(4, "TCU STORE");
        trace->fu_type = FUType::LSU;
        trace->lsu_type = LsuType::TCU_STORE;

        auto trace_data = std::make_shared<LsuTraceData>(num_threads);
        trace->data = trace_data;

        for (uint32_t t = thread_start; t < num_threads_actv_st; ++t)
        {
          if (!warp.tmask.test(t))
            continue;

          DP(3, "Thread ID" << t);
          uint32_t base_addr = rsdata[t][0].i ;

          trace_data->mem_addrs.at(t) = {base_addr, data_bytes_store};

          //Store C
          for (int n=0; n<num_data_per_thread_st; n++)
          {
            Word* temp_ref = &(warp.ireg_file.at(t).at(rsrc0));
            *temp_ref = scratchpad[(n_tiles*tc_size*tc_size*2) + (t*num_data_per_thread_st) + n];

            this->dcache_write(temp_ref, base_addr+(n*mem_bytes), mem_bytes);
          }
        }
        //Clear the scratchpad
        for(long unsigned int i=0 ; i < scratchpad.size(); i++)
        {
          scratchpad[i] = 0;
        }
      }
      break;
      case 2:
      { //Matrix Multiply
        DP(4, "TCU MULTIPLY MAT");
        trace->fu_type = FUType::TCU;
        trace->tcu_type = TCUType::TCU_MUL;
        uint32_t threads_per_tc = MAX (1, num_threads/TC_per_warp);
        for (uint32_t t = thread_start; t < num_threads_actv; ++t)
        {
          if (!warp.tmask.test(t))
            continue;

          DP(3, "Thread ID" << t);
          //TC operation [only 1 thread in 1 warp needs to do this]
          if (t%threads_per_tc == 0)
          {
            /*
            // TODO : Fix needed for functional correctness
            // TODO : change to systolic array implementation
            uint32_t thread_offset = t*(tc_size*tc_size);

            int loop_offset = 0;
            int offset_b = n_tiles*n_tiles*n_tiles*tc_size*tc_size;
            uint32_t accu_offset = (n_tiles)*(n_tiles)*(n_tiles)*tc_size*tc_size*2;
            for(int tiles = 0 ; tiles < n_tiles ; tiles++)  //What's the HW implication of this?? A counter implementation?
            {
              for (int i = 0; i < tc_size; i++) { //ROW-1
                for (int j = 0; j < tc_size; j++) { //COL-2
                  int sum = 0;
                  for (int k = 0; k < tc_size; k++)
                  { //COL-1
                    sum = sum + scratchpad[loop_offset + thread_offset*n_tiles + i * tc_size + k] *scratchpad[loop_offset + thread_offset*n_tiles + offset_b + (k * tc_size + j)];
                  }
                  scratchpad[accu_offset + thread_offset +(i * tc_size + j)] += sum; //[i * col2 + j] = sum
                  DP(3, "Scratchpad Index: " << accu_offset + (i * tc_size + j) << " , Value=" << scratchpad[accu_offset + (i * tc_size + j)]);
                }
              }
              loop_offset += tc_size*tc_size; //Move to the next tiled matmul fragment
            }
            */
          }
        }

      }break;
      default:
        std::abort();
    }
  } break;
  case Opcode::VSET: {
    auto func6 = instr.getFunc6();
    if ((func3 == 0x7) || (func3 == 0x2 && func6 == 16) || (func3 == 0x1 && func6 == 16)) {
      rd_write = true;
    }
    executeVector(instr, wid, rsdata, rddata);
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
        DPH(2, "Dest Reg: " << type << rdest << "={");
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (t) DPN(2, ", ");
          if (!warp.tmask.test(t)) {
            DPN(2, "-");
            continue;
          }
          warp.ireg_file.at(t)[rdest] = rddata[t].i;
          DPN(2, "0x" << std::hex << rddata[t].i << std::dec);
        }
        DPN(2, "}" << std::endl);
        trace->dst_reg = {type, rdest};
        assert(rdest != 0);
      } else {
        // disable writes to x0
        trace->wb = false;
      }
      break;
    case RegType::Float:
      DPH(2, "Dest Reg: " << type << rdest << "={");
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (t) DPN(2, ", ");
        if (!warp.tmask.test(t)) {
          DPN(2, "-");
          continue;
        }
        warp.freg_file.at(t)[rdest] = rddata[t].u64;
        DPN(2, "0x" << std::hex << rddata[t].f << std::dec);
      }
      DPN(2, "}" << std::endl);
      trace->dst_reg = {type, rdest};
      break;
    default:
      std::cout << "Unrecognized register write back type: " << type << std::endl;
      std::abort();
      break;
    }
  }

  warp.PC += 4;

  if (warp.PC != next_pc) {
    DP(3, "*** Next PC=0x" << std::hex << next_pc << std::dec);
    warp.PC = next_pc;
  }

  if (warp.tmask != next_tmask) {
    DP(3, "*** New Tmask=" << ThreadMaskOS(next_tmask, num_threads));
    warp.tmask = next_tmask;
    if (!next_tmask.any()) {
      active_warps_.reset(wid);
    }
  }
}