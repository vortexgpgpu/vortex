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
#ifdef EXT_V_ENABLE
#include "processor_impl.h"
#endif
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

inline void read_register(std::vector<reg_data_t>& out, uint32_t src_index, const Instr &instr, const warp_t& warp) {
  auto type = instr.getRSType(src_index);
  auto reg = instr.getRSrc(src_index);
  switch (type) {
  case RegType::None:
    break;
  case RegType::Integer: {
    DPH(2, "Src" << src_index << " Reg: " << type << reg << "={");
    auto& reg_data = warp.ireg_file.at(reg);
    out.resize(reg_data.size());
    for (uint32_t t = 0; t < reg_data.size(); ++t) {
      if (t) DPN(2, ", ");
      if (!warp.tmask.test(t)) {
        DPN(2, "-");
        continue;
      }
      out[t].u = reg_data.at(t);
      DPN(2, "0x" << std::hex << out[t].i << std::dec);
    }
    DPN(2, "}" << std::endl);
  } break;
  case RegType::Float: {
    DPH(2, "Src" << src_index << " Reg: " << type << reg << "={");
    auto& reg_data = warp.freg_file.at(reg);
    out.resize(reg_data.size());
    for (uint32_t t = 0; t < reg_data.size(); ++t) {
      if (t) DPN(2, ", ");
      if (!warp.tmask.test(t)) {
        DPN(2, "-");
        continue;
      }
      out[t].u64 = reg_data.at(t);
      if ((out[t].u64 >> 32) == 0xffffffff) {
        DPN(2, "0x" << std::hex << out[t].u32 << std::dec);
      } else {
        DPN(2, "0x" << std::hex << out[t].u64 << std::dec);
      }
    }
    DPN(2, "}" << std::endl);
  } break;
#ifdef EXT_V_ENABLE
  case RegType::Vector:
  break;
#endif
  default:
    std::abort();
    break;
  }
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

  std::vector<reg_data_t> rd_data(num_threads);

  std::vector<reg_data_t> rs1_data;
  std::vector<reg_data_t> rs2_data;
  std::vector<reg_data_t> rs3_data;

  // reading source registers
  switch (instr.getNRSrc()) {
  case 0:
    // no source register
    break;
  case 3:
    read_register(rs3_data, 2, instr, warp);
    [[fallthrough]];
  case 2:
    read_register(rs2_data, 1, instr, warp);
    [[fallthrough]];
  case 1:
    read_register(rs1_data, 0, instr, warp);
    break;
  default:
    std::abort();
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
      rd_data[t].i = immsrc;
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
      rd_data[t].i = immsrc + warp.PC;
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
        auto value = rs1_data[t].i;
        auto cond = rs2_data[t].i;
        if (func3 == 0x5) {
          // CZERO.EQZ
          rd_data[t].i = (cond == 0) ? 0 : value;
          trace->alu_type = AluType::ARITH;
        } else
        if (func3 == 0x7) {
          // CZERO.NEZ
          rd_data[t].i = (cond != 0) ? 0 : value;
          trace->alu_type = AluType::ARITH;
        } else {
          std::abort();
        }
      } else
      if (func7 & 0x1) {
        switch (func3) {
        case 0: {
          // RV32M: MUL
          rd_data[t].i = rs1_data[t].i * rs2_data[t].i;
          trace->alu_type = AluType::IMUL;
          break;
        }
        case 1: {
          // RV32M: MULH
          auto first = static_cast<DWordI>(rs1_data[t].i);
          auto second = static_cast<DWordI>(rs2_data[t].i);
          rd_data[t].i = (first * second) >> XLEN;
          trace->alu_type = AluType::IMUL;
          break;
        }
        case 2: {
          // RV32M: MULHSU
          auto first = static_cast<DWordI>(rs1_data[t].i);
          auto second = static_cast<DWord>(rs2_data[t].u);
          rd_data[t].i = (first * second) >> XLEN;
          trace->alu_type = AluType::IMUL;
          break;
        }
        case 3: {
          // RV32M: MULHU
          auto first = static_cast<DWord>(rs1_data[t].u);
          auto second = static_cast<DWord>(rs2_data[t].u);
          rd_data[t].i = (first * second) >> XLEN;
          trace->alu_type = AluType::IMUL;
          break;
        }
        case 4: {
          // RV32M: DIV
          auto dividen = rs1_data[t].i;
          auto divisor = rs2_data[t].i;
          auto largest_negative = WordI(1) << (XLEN-1);
          if (divisor == 0) {
            rd_data[t].i = -1;
          } else if (dividen == largest_negative && divisor == -1) {
            rd_data[t].i = dividen;
          } else {
            rd_data[t].i = dividen / divisor;
          }
          trace->alu_type = AluType::IDIV;
          break;
        }
        case 5: {
          // RV32M: DIVU
          auto dividen = rs1_data[t].u;
          auto divisor = rs2_data[t].u;
          if (divisor == 0) {
            rd_data[t].i = -1;
          } else {
            rd_data[t].i = dividen / divisor;
          }
          trace->alu_type = AluType::IDIV;
          break;
        }
        case 6: {
          // RV32M: REM
          auto dividen = rs1_data[t].i;
          auto divisor = rs2_data[t].i;
          auto largest_negative = WordI(1) << (XLEN-1);
          if (rs2_data[t].i == 0) {
            rd_data[t].i = dividen;
          } else if (dividen == largest_negative && divisor == -1) {
            rd_data[t].i = 0;
          } else {
            rd_data[t].i = dividen % divisor;
          }
          trace->alu_type = AluType::IDIV;
          break;
        }
        case 7: {
          // RV32M: REMU
          auto dividen = rs1_data[t].u;
          auto divisor = rs2_data[t].u;
          if (rs2_data[t].i == 0) {
            rd_data[t].i = dividen;
          } else {
            rd_data[t].i = dividen % divisor;
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
            rd_data[t].i = rs1_data[t].i - rs2_data[t].i;
          } else {
            // RV32I: ADD
            rd_data[t].i = rs1_data[t].i + rs2_data[t].i;
          }
          break;
        }
        case 1: {
          // RV32I: SLL
          Word shamt_mask = (Word(1) << log2up(XLEN)) - 1;
          Word shamt = rs2_data[t].i & shamt_mask;
          rd_data[t].i = rs1_data[t].i << shamt;
          break;
        }
        case 2: {
          // RV32I: SLT
          rd_data[t].i = rs1_data[t].i < rs2_data[t].i;
          break;
        }
        case 3: {
          // RV32I: SLTU
          rd_data[t].i = rs1_data[t].u < rs2_data[t].u;
          break;
        }
        case 4: {
          // RV32I: XOR
          rd_data[t].i = rs1_data[t].i ^ rs2_data[t].i;
          break;
        }
        case 5: {
          Word shamt_mask = ((Word)1 << log2up(XLEN)) - 1;
          Word shamt = rs2_data[t].i & shamt_mask;
          if (func7 & 0x20) {
            // RV32I: SRA
            rd_data[t].i = rs1_data[t].i >> shamt;
          } else {
            // RV32I: SRL
            rd_data[t].i = rs1_data[t].u >> shamt;
          }
          break;
        }
        case 6: {
          // RV32I: OR
          rd_data[t].i = rs1_data[t].i | rs2_data[t].i;
          break;
        }
        case 7: {
          // RV32I: AND
          rd_data[t].i = rs1_data[t].i & rs2_data[t].i;
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
        rd_data[t].i = rs1_data[t].i + immsrc;
        break;
      }
      case 1: {
        // RV32I: SLLI
        rd_data[t].i = rs1_data[t].i << immsrc;
        break;
      }
      case 2: {
        // RV32I: SLTI
        rd_data[t].i = rs1_data[t].i < WordI(immsrc);
        break;
      }
      case 3: {
        // RV32I: SLTIU
        rd_data[t].i = rs1_data[t].u < immsrc;
        break;
      }
      case 4: {
        // RV32I: XORI
        rd_data[t].i = rs1_data[t].i ^ immsrc;
        break;
      }
      case 5: {
        if (func7 & 0x20) {
          // RV32I: SRAI
          Word result = rs1_data[t].i >> immsrc;
          rd_data[t].i = result;
        } else {
          // RV32I: SRLI
          Word result = rs1_data[t].u >> immsrc;
          rd_data[t].i = result;
        }
        break;
      }
      case 6: {
        // RV32I: ORI
        rd_data[t].i = rs1_data[t].i | immsrc;
        break;
      }
      case 7: {
        // RV32I: ANDI
        rd_data[t].i = rs1_data[t].i & immsrc;
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
            int32_t product = (int32_t)rs1_data[t].i * (int32_t)rs2_data[t].i;
            rd_data[t].i = sext((uint64_t)product, 32);
            trace->alu_type = AluType::IMUL;
            break;
          }
          case 4: {
            // RV64M: DIVW
            int32_t dividen = (int32_t)rs1_data[t].i;
            int32_t divisor = (int32_t)rs2_data[t].i;
            int32_t quotient;
            int32_t largest_negative = 0x80000000;
            if (divisor == 0){
              quotient = -1;
            } else if (dividen == largest_negative && divisor == -1) {
              quotient = dividen;
            } else {
              quotient = dividen / divisor;
            }
            rd_data[t].i = sext((uint64_t)quotient, 32);
            trace->alu_type = AluType::IDIV;
            break;
          }
          case 5: {
            // RV64M: DIVUW
            uint32_t dividen = (uint32_t)rs1_data[t].i;
            uint32_t divisor = (uint32_t)rs2_data[t].i;
            uint32_t quotient;
            if (divisor == 0){
              quotient = -1;
            } else {
              quotient = dividen / divisor;
            }
            rd_data[t].i = sext((uint64_t)quotient, 32);
            trace->alu_type = AluType::IDIV;
            break;
          }
          case 6: {
            // RV64M: REMW
            int32_t dividen = (uint32_t)rs1_data[t].i;
            int32_t divisor = (uint32_t)rs2_data[t].i;
            int32_t remainder;
            int32_t largest_negative = 0x80000000;
            if (divisor == 0){
              remainder = dividen;
            } else if (dividen == largest_negative && divisor == -1) {
              remainder = 0;
            } else {
              remainder = dividen % divisor;
            }
            rd_data[t].i = sext((uint64_t)remainder, 32);
            trace->alu_type = AluType::IDIV;
            break;
          }
          case 7: {
            // RV64M: REMUW
            uint32_t dividen = (uint32_t)rs1_data[t].i;
            uint32_t divisor = (uint32_t)rs2_data[t].i;
            uint32_t remainder;
            if (divisor == 0){
              remainder = dividen;
            } else {
              remainder = dividen % divisor;
            }
            rd_data[t].i = sext((uint64_t)remainder, 32);
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
            uint32_t result = (uint32_t)rs1_data[t].i - (uint32_t)rs2_data[t].i;
            rd_data[t].i = sext((uint64_t)result, 32);
          }
          else{
            // RV64I: ADDW
            uint32_t result = (uint32_t)rs1_data[t].i + (uint32_t)rs2_data[t].i;
            rd_data[t].i = sext((uint64_t)result, 32);
          }
          break;
        }
        case 1: {
          // RV64I: SLLW
          uint32_t shamt_mask = 0x1F;
          uint32_t shamt = rs2_data[t].i & shamt_mask;
          uint32_t result = (uint32_t)rs1_data[t].i << shamt;
          rd_data[t].i = sext((uint64_t)result, 32);
          break;
        }
        case 5: {
          uint32_t shamt_mask = 0x1F;
          uint32_t shamt = rs2_data[t].i & shamt_mask;
          uint32_t result;
          if (func7 & 0x20) {
            // RV64I: SRAW
            result = (int32_t)rs1_data[t].i >> shamt;
          } else {
            // RV64I: SRLW
            result = (uint32_t)rs1_data[t].i >> shamt;
          }
          rd_data[t].i = sext((uint64_t)result, 32);
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
          uint32_t result = (uint32_t)rs1_data[t].i + (uint32_t)immsrc;
          rd_data[t].i = sext((uint64_t)result, 32);
          break;
        }
        case 1: {
          // RV64I: SLLIW
          uint32_t shamt_mask = 0x1F;
          uint32_t shamt = immsrc & shamt_mask;
          uint32_t result = rs1_data[t].i << shamt;
          rd_data[t].i = sext((uint64_t)result, 32);
          break;
        }
        case 5: {
          uint32_t shamt_mask = 0x1F;
          uint32_t shamt = immsrc & shamt_mask;
          uint32_t result;
          if (func7 & 0x20) {
            // RV64I: SRAIW
            result = (int32_t)rs1_data[t].i >> shamt;
          } else {
            // RV64I: SRLIW
            result = (uint32_t)rs1_data[t].i >> shamt;
          }
          rd_data[t].i = sext((uint64_t)result, 32);
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
        if (rs1_data[t].i == rs2_data[t].i) {
          next_pc = warp.PC + immsrc;
          curr_taken = true;
        }
        break;
      }
      case 1: {
        // RV32I: BNE
        if (rs1_data[t].i != rs2_data[t].i) {
          next_pc = warp.PC + immsrc;
          curr_taken = true;
        }
        break;
      }
      case 4: {
        // RV32I: BLT
        if (rs1_data[t].i < rs2_data[t].i) {
          next_pc = warp.PC + immsrc;
          curr_taken = true;
        }
        break;
      }
      case 5: {
        // RV32I: BGE
        if (rs1_data[t].i >= rs2_data[t].i) {
          next_pc = warp.PC + immsrc;
          curr_taken = true;
        }
        break;
      }
      case 6: {
        // RV32I: BLTU
        if (rs1_data[t].u < rs2_data[t].u) {
          next_pc = warp.PC + immsrc;
          curr_taken = true;
        }
        break;
      }
      case 7: {
        // RV32I: BGEU
        if (rs1_data[t].u >= rs2_data[t].u) {
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
      rd_data[t].i = next_pc;
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
      rd_data[t].i = next_pc;
    }
    next_pc = rs1_data[thread_last].i + immsrc;
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
        uint64_t mem_addr = rs1_data[t].i + immsrc;
        uint64_t read_data = 0;
        this->dcache_read(&read_data, mem_addr, data_bytes);
        trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
        switch (func3) {
        case 0: // RV32I: LB
        case 1: // RV32I: LH
          rd_data[t].i = sext((Word)read_data, data_width);
          break;
        case 2:
          if (opcode == Opcode::L) {
            // RV32I: LW
            rd_data[t].i = sext((Word)read_data, data_width);
          } else {
            // RV32F: FLW
            rd_data[t].u64 = nan_box((uint32_t)read_data);
          }
          break;
        case 3: // RV64I: LD
                // RV32D: FLD
        case 4: // RV32I: LBU
        case 5: // RV32I: LHU
        case 6: // RV64I: LWU
          rd_data[t].u64 = read_data;
          break;
        default:
          std::abort();
        }
      }
      rd_write = true;
    }
  #ifdef EXT_V_ENABLE
    else {
      for (uint32_t t = thread_start; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        this->loadVector(instr, wid, t, rs1_data.at(t).i, rs2_data.at(t).i);
      }
    }
  #endif
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
        uint64_t mem_addr = rs1_data[t].i + immsrc;
        uint64_t write_data = rs2_data[t].u64;
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
    }
  #ifdef EXT_V_ENABLE
    else {
      for (uint32_t t = thread_start; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        this->storeVector(instr, wid, t, rs1_data.at(t).i, rs2_data.at(t).i);
      }
    }
  #endif
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
      uint64_t mem_addr = rs1_data[t].u;
      trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
      if (amo_type == 0x02) { // LR
        uint64_t read_data = 0;
        this->dcache_read(&read_data, mem_addr, data_bytes);
        this->dcache_amo_reserve(mem_addr);
        rd_data[t].i = sext((Word)read_data, data_width);
      } else
      if (amo_type == 0x03) { // SC
        if (this->dcache_amo_check(mem_addr)) {
          this->dcache_write(&rs2_data[t].u64, mem_addr, data_bytes);
          rd_data[t].i = 0;
        } else {
          rd_data[t].i = 1;
        }
      } else {
        uint64_t read_data = 0;
        this->dcache_read(&read_data, mem_addr, data_bytes);
        auto read_data_i = sext((WordI)read_data, data_width);
        auto rs1_data_i  = sext((WordI)rs2_data[t].u64, data_width);
        auto read_data_u = zext((Word)read_data, data_width);
        auto rs1_data_u  = zext((Word)rs2_data[t].u64, data_width);
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
        rd_data[t].i = read_data_i;
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
          rd_data[t].i = csr_value;
          this->set_csr(csr_addr, rs1_data[t].i, t, wid);
          trace->src_regs[0] = {RegType::Integer, rsrc0};
          trace->sfu_type = SfuType::CSRRW;
          rd_write = true;
          break;
        }
        case 2: {
          // RV32I: CSRRS
          rd_data[t].i = csr_value;
          if (rs1_data[t].i != 0) {
            this->set_csr(csr_addr, csr_value | rs1_data[t].i, t, wid);
          }
          trace->src_regs[0] = {RegType::Integer, rsrc0};
          trace->sfu_type = SfuType::CSRRS;
          rd_write = true;
          break;
        }
        case 3: {
          // RV32I: CSRRC
          rd_data[t].i = csr_value;
          if (rs1_data[t].i != 0) {
            this->set_csr(csr_addr, csr_value & ~rs1_data[t].i, t, wid);
          }
          trace->src_regs[0] = {RegType::Integer, rsrc0};
          trace->sfu_type = SfuType::CSRRC;
          rd_write = true;
          break;
        }
        case 5: {
          // RV32I: CSRRWI
          rd_data[t].i = csr_value;
          this->set_csr(csr_addr, rsrc0, t, wid);
          trace->sfu_type = SfuType::CSRRW;
          rd_write = true;
          break;
        }
        case 6: {
          // RV32I: CSRRSI;
          rd_data[t].i = csr_value;
          if (rsrc0 != 0) {
            this->set_csr(csr_addr, csr_value | rsrc0, t, wid);
          }
          trace->sfu_type = SfuType::CSRRS;
          rd_write = true;
          break;
        }
        case 7: {
          // RV32I: CSRRCI
          rd_data[t].i = csr_value;
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
      uint32_t frm = this->get_fpu_rm(func3, t, wid);
      uint32_t fflags = 0;
      switch (func7) {
      case 0x00: { // RV32F: FADD.S
        rd_data[t].u64 = nan_box(rv_fadd_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), frm, &fflags));
        trace->fpu_type = FpuType::FMA;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x01: { // RV32D: FADD.D
        rd_data[t].u64 = rv_fadd_d(rs1_data[t].u64, rs2_data[t].u64, frm, &fflags);
        trace->fpu_type = FpuType::FMA;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x04: { // RV32F: FSUB.S
        rd_data[t].u64 = nan_box(rv_fsub_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), frm, &fflags));
        trace->fpu_type = FpuType::FMA;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x05: { // RV32D: FSUB.D
        rd_data[t].u64 = rv_fsub_d(rs1_data[t].u64, rs2_data[t].u64, frm, &fflags);
        trace->fpu_type = FpuType::FMA;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x08: { // RV32F: FMUL.S
        rd_data[t].u64 = nan_box(rv_fmul_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), frm, &fflags));
        trace->fpu_type = FpuType::FMA;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x09: { // RV32D: FMUL.D
        rd_data[t].u64 = rv_fmul_d(rs1_data[t].u64, rs2_data[t].u64, frm, &fflags);
        trace->fpu_type = FpuType::FMA;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x0c: { // RV32F: FDIV.S
        rd_data[t].u64 = nan_box(rv_fdiv_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), frm, &fflags));
        trace->fpu_type = FpuType::FDIV;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x0d: { // RV32D: FDIV.D
        rd_data[t].u64 = rv_fdiv_d(rs1_data[t].u64, rs2_data[t].u64, frm, &fflags);
        trace->fpu_type = FpuType::FDIV;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x10: {
        switch (func3) {
        case 0: // RV32F: FSGNJ.S
          rd_data[t].u64 = nan_box(rv_fsgnj_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64)));
          break;
        case 1: // RV32F: FSGNJN.S
          rd_data[t].u64 = nan_box(rv_fsgnjn_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64)));
          break;
        case 2: // RV32F: FSGNJX.S
          rd_data[t].u64 = nan_box(rv_fsgnjx_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64)));
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
          rd_data[t].u64 = rv_fsgnj_d(rs1_data[t].u64, rs2_data[t].u64);
          break;
        case 1: // RV32D: FSGNJN.D
          rd_data[t].u64 = rv_fsgnjn_d(rs1_data[t].u64, rs2_data[t].u64);
          break;
        case 2: // RV32D: FSGNJX.D
          rd_data[t].u64 = rv_fsgnjx_d(rs1_data[t].u64, rs2_data[t].u64);
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
          rd_data[t].u64 = nan_box(rv_fmax_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags));
        } else {
          // RV32F: FMIN.S
          rd_data[t].u64 = nan_box(rv_fmin_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags));
        }
        trace->fpu_type = FpuType::FNCP;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x15: {
        if (func3) {
          // RV32D: FMAX.D
          rd_data[t].u64 = rv_fmax_d(rs1_data[t].u64, rs2_data[t].u64, &fflags);
        } else {
          // RV32D: FMIN.D
          rd_data[t].u64 = rv_fmin_d(rs1_data[t].u64, rs2_data[t].u64, &fflags);
        }
        trace->fpu_type = FpuType::FNCP;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        trace->src_regs[1] = {RegType::Float, rsrc1};
        break;
      }
      case 0x20: {
        // RV32D: FCVT.S.D
        rd_data[t].u64 = nan_box(rv_dtof(rs1_data[t].u64));
        trace->fpu_type = FpuType::FNCP;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        break;
      }
      case 0x21: {
        // RV32D: FCVT.D.S
        rd_data[t].u64 = rv_ftod(check_boxing(rs1_data[t].u64));
        trace->fpu_type = FpuType::FNCP;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        break;
      }
      case 0x2c: { // RV32F: FSQRT.S
        rd_data[t].u64 = nan_box(rv_fsqrt_s(check_boxing(rs1_data[t].u64), frm, &fflags));
        trace->fpu_type = FpuType::FSQRT;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        break;
      }
      case 0x2d: { // RV32D: FSQRT.D
        rd_data[t].u64 = rv_fsqrt_d(rs1_data[t].u64, frm, &fflags);
        trace->fpu_type = FpuType::FSQRT;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        break;
      }
      case 0x50: {
        switch (func3) {
        case 0:
          // RV32F: FLE.S
          rd_data[t].i = rv_fle_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags);
          break;
        case 1:
          // RV32F: FLT.S
          rd_data[t].i = rv_flt_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags);
          break;
        case 2:
          // RV32F: FEQ.S
          rd_data[t].i = rv_feq_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags);
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
          rd_data[t].i = rv_fle_d(rs1_data[t].u64, rs2_data[t].u64, &fflags);
          break;
        case 1:
          // RV32D: FLT.D
          rd_data[t].i = rv_flt_d(rs1_data[t].u64, rs2_data[t].u64, &fflags);
          break;
        case 2:
          // RV32D: FEQ.D
          rd_data[t].i = rv_feq_d(rs1_data[t].u64, rs2_data[t].u64, &fflags);
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
          rd_data[t].i = sext((uint64_t)rv_ftoi_s(check_boxing(rs1_data[t].u64), frm, &fflags), 32);
          break;
        case 1:
          // RV32F: FCVT.WU.S
          rd_data[t].i = sext((uint64_t)rv_ftou_s(check_boxing(rs1_data[t].u64), frm, &fflags), 32);
          break;
        case 2:
          // RV64F: FCVT.L.S
          rd_data[t].i = rv_ftol_s(check_boxing(rs1_data[t].u64), frm, &fflags);
          break;
        case 3:
          // RV64F: FCVT.LU.S
          rd_data[t].i = rv_ftolu_s(check_boxing(rs1_data[t].u64), frm, &fflags);
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
          rd_data[t].i = sext((uint64_t)rv_ftoi_d(rs1_data[t].u64, frm, &fflags), 32);
          break;
        case 1:
          // RV32D: FCVT.WU.D
          rd_data[t].i = sext((uint64_t)rv_ftou_d(rs1_data[t].u64, frm, &fflags), 32);
          break;
        case 2:
          // RV64D: FCVT.L.D
          rd_data[t].i = rv_ftol_d(rs1_data[t].u64, frm, &fflags);
          break;
        case 3:
          // RV64D: FCVT.LU.D
          rd_data[t].i = rv_ftolu_d(rs1_data[t].u64, frm, &fflags);
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
          rd_data[t].u64 = nan_box(rv_itof_s(rs1_data[t].i, frm, &fflags));
          break;
        case 1:
          // RV32F: FCVT.S.WU
          rd_data[t].u64 = nan_box(rv_utof_s(rs1_data[t].i, frm, &fflags));
          break;
        case 2:
          // RV64F: FCVT.S.L
          rd_data[t].u64 = nan_box(rv_ltof_s(rs1_data[t].i, frm, &fflags));
          break;
        case 3:
          // RV64F: FCVT.S.LU
          rd_data[t].u64 = nan_box(rv_lutof_s(rs1_data[t].i, frm, &fflags));
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
          rd_data[t].u64 = rv_itof_d(rs1_data[t].i, frm, &fflags);
          break;
        case 1:
          // RV32D: FCVT.D.WU
          rd_data[t].u64 = rv_utof_d(rs1_data[t].i, frm, &fflags);
          break;
        case 2:
          // RV64D: FCVT.D.L
          rd_data[t].u64 = rv_ltof_d(rs1_data[t].i, frm, &fflags);
          break;
        case 3:
          // RV64D: FCVT.D.LU
          rd_data[t].u64 = rv_lutof_d(rs1_data[t].i, frm, &fflags);
          break;
        }
        trace->fpu_type = FpuType::FCVT;
        trace->src_regs[0] = {RegType::Integer, rsrc0};
        break;
      }
      case 0x70: {
        if (func3) {
          // RV32F: FCLASS.S
          rd_data[t].i = rv_fclss_s(check_boxing(rs1_data[t].u64));
        } else {
          // RV32F: FMV.X.S
          uint32_t result = (uint32_t)rs1_data[t].u64;
          rd_data[t].i = sext((uint64_t)result, 32);
        }
        trace->fpu_type = FpuType::FNCP;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        break;
      }
      case 0x71: {
        if (func3) {
          // RV32D: FCLASS.D
          rd_data[t].i = rv_fclss_d(rs1_data[t].u64);
        } else {
          // RV64D: FMV.X.D
          rd_data[t].i = rs1_data[t].u64;
        }
        trace->fpu_type = FpuType::FNCP;
        trace->src_regs[0] = {RegType::Float, rsrc0};
        break;
      }
      case 0x78: { // RV32F: FMV.S.X
        rd_data[t].u64 = nan_box((uint32_t)rs1_data[t].i);
        trace->fpu_type = FpuType::FNCP;
        trace->src_regs[0] = {RegType::Integer, rsrc0};
        break;
      }
      case 0x79: { // RV64D: FMV.D.X
        rd_data[t].u64 = rs1_data[t].i;
        trace->fpu_type = FpuType::FNCP;
        trace->src_regs[0] = {RegType::Integer, rsrc0};
        break;
      }
      }
      this->update_fcrs(fflags, t, wid);
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
          rd_data[t].u64 = rv_fmadd_d(rs1_data[t].u64, rs2_data[t].u64, rs3_data[t].u64, frm, &fflags);
        else
          // RV32F: FMADD.S
          rd_data[t].u64 = nan_box(rv_fmadd_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), check_boxing(rs3_data[t].u64), frm, &fflags));
        break;
      case Opcode::FMSUB:
        if (func2)
          // RV32D: FMSUB.D
          rd_data[t].u64 = rv_fmsub_d(rs1_data[t].u64, rs2_data[t].u64, rs3_data[t].u64, frm, &fflags);
        else
          // RV32F: FMSUB.S
          rd_data[t].u64 = nan_box(rv_fmsub_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), check_boxing(rs3_data[t].u64), frm, &fflags));
        break;
      case Opcode::FMNMADD:
        if (func2)
          // RV32D: FNMADD.D
          rd_data[t].u64 = rv_fnmadd_d(rs1_data[t].u64, rs2_data[t].u64, rs3_data[t].u64, frm, &fflags);
        else
          // RV32F: FNMADD.S
          rd_data[t].u64 = nan_box(rv_fnmadd_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), check_boxing(rs3_data[t].u64), frm, &fflags));
        break;
      case Opcode::FMNMSUB:
        if (func2)
          // RV32D: FNMSUB.D
          rd_data[t].u64 = rv_fnmsub_d(rs1_data[t].u64, rs2_data[t].u64, rs3_data[t].u64, frm, &fflags);
        else
          // RV32F: FNMSUB.S
          rd_data[t].u64 = nan_box(rv_fnmsub_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), check_boxing(rs3_data[t].u64), frm, &fflags));
        break;
      default:
        break;
      }
      this->update_fcrs(fflags, t, wid);
    }
    rd_write = true;
    break;
  }
#ifdef EXT_V_ENABLE
  case Opcode::VSET:
    for (uint32_t t = thread_start; t < num_threads; ++t) {
      if (!warp.tmask.test(t))
        continue;
      rd_write |= executeVector(instr, wid, t, rs1_data.at(t).i, rs2_data.at(t).i, &rd_data.at(t).i);
    }
    break;
#endif
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
          next_tmask.set(t, rs1_data.at(thread_last).u & (1 << t));
        }
      } break;
      case 1: {
        // WSPAWN
        trace->fu_type = FUType::SFU;
        trace->sfu_type = SfuType::WSPAWN;
        trace->src_regs[0] = {RegType::Integer, rsrc0};
        trace->src_regs[1] = {RegType::Integer, rsrc1};
        trace->fetch_stall = true;
        trace->data = std::make_shared<SFUTraceData>(rs1_data.at(thread_last).u, rs2_data.at(thread_last).u);
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
          auto cond = (rs1_data.at(t).i & 0x1) ^ not_pred;
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
          rd_data[t].i = stack_size;
        }
        rd_write = true;
      } break;
      case 3: {
        // JOIN
        trace->fu_type = FUType::SFU;
        trace->sfu_type = SfuType::JOIN;
        trace->src_regs[0] = {RegType::Integer, rsrc0};
        trace->fetch_stall = true;

        auto stack_ptr = rs1_data.at(thread_last).u;
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
        trace->data = std::make_shared<SFUTraceData>(rs1_data[thread_last].i, rs2_data[thread_last].i);
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
          auto cond = (rs1_data.at(t).i & 0x1) ^ not_pred;
          pred[t] = warp.tmask.test(t) && cond;
        }
        if (pred.any()) {
          next_tmask &= pred;
        } else {
          next_tmask = rs2_data.at(thread_last).u;
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
  case Opcode::EXT2: {
    switch(func3) {
    case 0: // reserved
    case 1: // reserved
      std::abort();
    case 2: {
      trace->fu_type = FUType::SFU;
      trace->sfu_type = SfuType::MMADD;
      trace->src_regs[0] = {RegType::Integer, rsrc0};
      trace->src_regs[1] = {RegType::Integer, rsrc1};
      trace->src_regs[2] = {RegType::Integer, rsrc2};
      auto trace_data = std::make_shared<TensorUnit::TraceData>();
      trace->data = trace_data;

      TensorFormat from, to;
      switch (func2) {
      case 0: // INT8
        from = TensorFormat::Int4;
        to = TensorFormat::Int32;
        break;
      case 1: // INT16
        from = TensorFormat::Int8;
        to = TensorFormat::Int32;
        break;
      case 2: // FP16
        from = TensorFormat::FP16;
        to = TensorFormat::FP32;
        break;
      case 3: // BF16
        from = TensorFormat::BF16;
        to = TensorFormat::FP32;
        break;
      default:
        std::abort();
      }
      tensor_unit_->mmadd(from, to, rs1_data, rs2_data, rs3_data, rd_data, trace_data);
      rd_write = true;
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
    case RegType::None:
      break;
    case RegType::Integer:
      if (rdest) {
        DPH(2, "Dest Reg: " << type << rdest << "={");
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (t) DPN(2, ", ");
          if (!warp.tmask.test(t)) {
            DPN(2, "-");
            continue;
          }
          warp.ireg_file.at(rdest).at(t) = rd_data[t].i;
          DPN(2, "0x" << std::hex << rd_data[t].u << std::dec);
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
        warp.freg_file.at(rdest).at(t) = rd_data[t].u64;
        if ((rd_data[t].u64 >> 32) == 0xffffffff) {
          DPN(2, "0x" << std::hex << rd_data[t].u32 << std::dec);
        } else {
          DPN(2, "0x" << std::hex << rd_data[t].u64 << std::dec);
        }
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