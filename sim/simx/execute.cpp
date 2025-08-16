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
#include "types.h"
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

void Emulator::fetch_registers(std::vector<reg_data_t>& out, uint32_t wid, uint32_t src_index, const RegOpd& reg) {
  __unused(src_index);
  auto& warp = warps_.at(wid);
  uint32_t num_threads = warp.tmask.size();
  out.resize(num_threads);
  switch (reg.type) {
  case RegType::None:
#ifdef EXT_V_ENABLE
  case RegType::Vector:
    DPH(2, "Src" << src_index << " Reg: " << reg << "={");
    for (uint32_t t = 0; t < num_threads; ++t) {
      if (t) DPN(2, ", ");
      if (!warp.tmask.test(t)) {
        DPN(2, "-");
        continue;
      }
      DPN(2, vec_unit_->dumpRegister(wid, t, reg.idx));
    }
    DPN(2, "}" << std::endl);
#endif
    break;
  case RegType::Integer: {
    DPH(2, "Src" << src_index << " Reg: " << reg << "={");
    auto& reg_data = warp.ireg_file.at(reg.idx);
    for (uint32_t t = 0; t < num_threads; ++t) {
      if (t) DPN(2, ", ");
      if (!warp.tmask.test(t)) {
        DPN(2, "-");
        continue;
      }
      auto& value = out[t];
      value.u = reg_data.at(t);
      DPN(2, "0x" << std::hex << value.u << std::dec);
    }
    DPN(2, "}" << std::endl);
  } break;
  case RegType::Float: {
    DPH(2, "Src" << src_index << " Reg: " << reg << "={");
    auto& reg_data = warp.freg_file.at(reg.idx);
    for (uint32_t t = 0; t < num_threads; ++t) {
      if (t) DPN(2, ", ");
      if (!warp.tmask.test(t)) {
        DPN(2, "-");
        continue;
      }
      auto& value = out[t];
      value.u64 = reg_data.at(t);
      if ((value.u64 >> 32) == 0xffffffff) {
        DPN(2, "0x" << std::hex << value.u32 << std::dec);
      } else {
        DPN(2, "0x" << std::hex << value.u64 << std::dec);
      }
    }
    DPN(2, "}" << std::endl);
  } break;
  default:
    std::abort();
    break;
  }
}

instr_trace_t* Emulator::execute(const Instr &instr, uint32_t wid) {
  auto& warp = warps_.at(wid);
  assert(warp.tmask.any());

  auto next_pc = warp.PC + 4;
  auto next_tmask = warp.tmask;

  auto fu_type = instr.getFUType();
  auto op_type = instr.getOpType();
  auto instrArgs = instr.getArgs();
  auto rdest  = instr.getDestReg();
  auto rsrc0  = instr.getSrcReg(0);
  auto rsrc1  = instr.getSrcReg(1);
  auto rsrc2  = instr.getSrcReg(2);

  auto num_threads = arch_.num_threads();

  // create instruction trace
  auto trace_alloc = core_->trace_pool().allocate(1);
  auto trace = new (trace_alloc) instr_trace_t(instr.getUUID(), arch_);
  trace->fu_type  = fu_type;
  trace->op_type  = op_type;
  trace->cid      = core_->id();
  trace->wid      = wid;
  trace->PC       = warp.PC;
  trace->tmask    = warp.tmask;
  trace->dst_reg  = rdest;
  trace->src_regs = {rsrc0, rsrc1, rsrc2};

  std::vector<reg_data_t> rd_data(num_threads);
  std::vector<reg_data_t> rs1_data;
  std::vector<reg_data_t> rs2_data;
  std::vector<reg_data_t> rs3_data;

  DP(1, "Instr: " << instr << ", cid=" << core_->id() << ", wid=" << wid << ", tmask=" << warp.tmask
         << ", PC=0x" << std::hex << warp.PC << std::dec << " (#" << instr.getUUID() << ")");

  // fetch register values
  if (rsrc0.type != RegType::None) fetch_registers(rs1_data, wid, 0, rsrc0);
  if (rsrc1.type != RegType::None) fetch_registers(rs2_data, wid, 1, rsrc1);
  if (rsrc2.type != RegType::None) fetch_registers(rs3_data, wid, 2, rsrc2);

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

  bool is_w_enabled = false;
#ifdef XLEN_64
  is_w_enabled = true;
#endif // XLEN_64

  bool rd_write = false;

  visit_var(op_type,
    [&](AluType alu_type) {
      auto aluArgs = std::get<IntrAluArgs>(instrArgs);
      Word imm = sext<Word>(aluArgs.imm, 32);
      switch (alu_type) {
      case AluType::LUI: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          rd_data[t].i = imm;
        }
      } break;
      case AluType::AUIPC: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          rd_data[t].i = imm + warp.PC;
        }
      } break;
      case AluType::ADD: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          if (is_w_enabled && aluArgs.is_w) {
            auto result = rs1_data[t].i32 + (int32_t)(aluArgs.is_imm ? aluArgs.imm : rs2_data[t].i32);
            rd_data[t].i = sext((uint64_t)result, 32);
          } else {
            rd_data[t].i = rs1_data[t].i + (aluArgs.is_imm ? imm : rs2_data[t].i);
          }
        }
      } break;
      case AluType::SUB: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          if (is_w_enabled && aluArgs.is_w) {
            auto result = rs1_data[t].i32 - (int32_t)(aluArgs.is_imm ? aluArgs.imm : rs2_data[t].i32);
            rd_data[t].i = sext((uint64_t)result, 32);
          } else {
            rd_data[t].i = rs1_data[t].i - (aluArgs.is_imm ? imm : rs2_data[t].i);
          }
        }
      } break;
      case AluType::SLT: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          rd_data[t].i = rs1_data[t].i < (aluArgs.is_imm ? WordI(imm) : rs2_data[t].i);
        }
      } break;
      case AluType::SLTU: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          rd_data[t].i = rs1_data[t].u < (aluArgs.is_imm ? imm : rs2_data[t].u);
        }
      } break;
      case AluType::SLL: {
        Word shamt_mask = (Word(1) << log2up(XLEN)) - 1;
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          if (is_w_enabled && aluArgs.is_w) {
            uint32_t shamt = (aluArgs.is_imm ? aluArgs.imm : rs2_data[t].i32) & shamt_mask;
            uint32_t result = (uint32_t)rs1_data[t].i << shamt;
            rd_data[t].i = sext((uint64_t)result, 32);
          } else {
            Word shamt = (aluArgs.is_imm ? imm : rs2_data[t].i) & shamt_mask;
            rd_data[t].i = rs1_data[t].i << shamt;
          }
        }
      } break;
      case AluType::SRA: {
        Word shamt_mask = (Word(1) << log2up(XLEN)) - 1;
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          if (is_w_enabled && aluArgs.is_w) {
            uint32_t shamt = (aluArgs.is_imm ? aluArgs.imm : rs2_data[t].i32) & shamt_mask;
            uint32_t result = (int32_t)rs1_data[t].i >> shamt;
            rd_data[t].i = sext((uint64_t)result, 32);
          } else {
            Word shamt = (aluArgs.is_imm ? imm : rs2_data[t].i) & shamt_mask;
            rd_data[t].i = rs1_data[t].i >> shamt;
          }
        }
      } break;
      case AluType::SRL: {
        Word shamt_mask = (Word(1) << log2up(XLEN)) - 1;
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          if (is_w_enabled && aluArgs.is_w) {
            uint32_t shamt = (aluArgs.is_imm ? aluArgs.imm : rs2_data[t].i32) & shamt_mask;
            uint32_t result = (uint32_t)rs1_data[t].i >> shamt;
            rd_data[t].i = sext((uint64_t)result, 32);
          } else {
            Word shamt = (aluArgs.is_imm ? imm : rs2_data[t].i) & shamt_mask;
            rd_data[t].i = rs1_data[t].u >> shamt;
          }
        }
      } break;
      case AluType::AND: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          rd_data[t].i = rs1_data[t].i & (aluArgs.is_imm ? imm : rs2_data[t].i);
        }
      } break;
      case AluType::OR: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          rd_data[t].i = rs1_data[t].i | (aluArgs.is_imm ? imm : rs2_data[t].i);
        }
      } break;
      case AluType::XOR: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          rd_data[t].i = rs1_data[t].i ^ (aluArgs.is_imm ? imm : rs2_data[t].i);
        }
      } break;
      case AluType::CZERO: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          bool cond = (rs2_data[t].i == 0) ^ aluArgs.imm;
          rd_data[t].i = cond ? 0 : rs1_data[t].i;
        }
      } break;
      default:
        std::abort();
      }
      rd_write = true;
    },
    [&](VoteType vote_type) {
      bool has_vote_true = false;
      bool has_vote_false = false;
      Word ballot = 0;
      // compute votes
      for (uint32_t t = thread_start; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto is_pred = rs1_data[t].i & 0x1;
        if (is_pred) {
          has_vote_true = true;
          ballot |= (Word(1) << t);
        } else {
          has_vote_false = true;
        }
      }
      for (uint32_t t = thread_start; t < num_threads; ++t) {
        switch (vote_type) {
        case VoteType::ALL:
          rd_data[t].i = !has_vote_false;
          break;
        case VoteType::ANY:
          rd_data[t].i = has_vote_true;
          break;
        case VoteType::UNI:
          rd_data[t].i = !has_vote_true || !has_vote_false;
          break;
        case VoteType::BAL:
          rd_data[t].i = ballot;
          break;
        default:
          std::abort();
        }
      }
      rd_write = true;
    },
    [&](ShflType shfl_type) {
      for (uint32_t t = thread_start; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        auto bc  = rs2_data[t].i;
        int bval = (bc >>  0) & 0x3f;
        int cval = (bc >>  6) & 0x3f;
        int mask = (bc >> 12) & 0x3f;
        int maxLane = (t & mask) | (cval & ~mask);
        int minLane = (t & mask);
        int lane = 0;
        int pval = 0;
        switch (shfl_type) {
        case ShflType::UP: {
          lane = t - bval;
          pval = (lane >= minLane);
        } break;
        case ShflType::DOWN: {
          lane = t + bval;
          pval = (lane <= maxLane);
        } break;
        case ShflType::BFLY: {
          lane = t ^ bval;
          pval = (lane <= maxLane);
        } break;
        case ShflType::IDX: {
          lane = minLane | (bval & ~mask);
          pval = (lane <= maxLane);
        } break;
        default:
          std::abort();
        }
        if (!pval)
          lane = t;
        if (lane < num_threads) {
          rd_data[t].i = rs1_data[lane].i;
        } else {
          rd_data[t].i = rs1_data[t].i;
        }
      }
      rd_write = true;
    },
    [&](BrType br_type) {
      auto brArgs = std::get<IntrBrArgs>(instrArgs);
      Word offset = sext<Word>(brArgs.offset, 32);
      switch (br_type) {
      case BrType::BR: {
        bool all_taken = false;
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          bool curr_taken = false;
          switch (brArgs.cmp) {
          case 0: { // RV32I: BEQ
            if (rs1_data[t].i == rs2_data[t].i) {
              next_pc = warp.PC + offset;
              curr_taken = true;
            }
            break;
          }
          case 1: { // RV32I: BNE
            if (rs1_data[t].i != rs2_data[t].i) {
              next_pc = warp.PC + offset;
              curr_taken = true;
            }
            break;
          }
          case 4: { // RV32I: BLT
            if (rs1_data[t].i < rs2_data[t].i) {
              next_pc = warp.PC + offset;
              curr_taken = true;
            }
            break;
          }
          case 5: { // RV32I: BGE
            if (rs1_data[t].i >= rs2_data[t].i) {
              next_pc = warp.PC + offset;
              curr_taken = true;
            }
            break;
          }
          case 6: { // RV32I: BLTU
            if (rs1_data[t].u < rs2_data[t].u) {
              next_pc = warp.PC + offset;
              curr_taken = true;
            }
            break;
          }
          case 7: { // RV32I: BGEU
            if (rs1_data[t].u >= rs2_data[t].u) {
              next_pc = warp.PC + offset;
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
      } break;
      case BrType::JAL: { // RV32I: JAL
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          rd_data[t].i = next_pc;
        }
        next_pc = warp.PC + offset;
        trace->fetch_stall = true;
        rd_write = true;
      } break;
      case BrType::JALR: { // RV32I: JALR
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          rd_data[t].i = next_pc;
        }
        next_pc = rs1_data[thread_last].i + offset;
        trace->fetch_stall = true;
        rd_write = true;
      } break;
      case BrType::SYS:
        switch (brArgs.offset) {
        case 0x000: // RV32I: ECALL
          this->trigger_ecall();
          break;
        case 0x001: // RV32I: EBREAK
          this->trigger_ebreak();
          break;
        case 0x002: // RV32I: URET
        case 0x102: // RV32I: SRET
        case 0x302: // RV32I: MRET
          break;
        default:
          std::abort();
        }
        break;
      default:
        std::abort();
      }
    },
    [&](MdvType mdv_type) {
      auto mdvArgs = std::get<IntrMdvArgs>(instrArgs);
      switch (mdv_type) {
      case MdvType::MUL: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          if (is_w_enabled && mdvArgs.is_w) {
            auto product = rs1_data[t].i32 * rs2_data[t].i32;
            rd_data[t].i = sext((uint64_t)product, 32);
          } else {
            rd_data[t].i = rs1_data[t].i * rs2_data[t].i;
          }
        }
      } break;
      case MdvType::MULH: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          auto first = static_cast<DWordI>(rs1_data[t].i);
          auto second = static_cast<DWordI>(rs2_data[t].i);
          rd_data[t].i = (first * second) >> XLEN;
        }
      } break;
      case MdvType::MULHSU: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          auto first = static_cast<DWordI>(rs1_data[t].i);
          auto second = static_cast<DWord>(rs2_data[t].u);
          rd_data[t].i = (first * second) >> XLEN;
        }
      } break;
      case MdvType::MULHU: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          auto first = static_cast<DWord>(rs1_data[t].u);
          auto second = static_cast<DWord>(rs2_data[t].u);
          rd_data[t].i = (first * second) >> XLEN;
        }
      } break;
      case MdvType::DIV: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          if (is_w_enabled && mdvArgs.is_w) {
            auto dividen = rs1_data[t].i32;
            auto divisor = rs2_data[t].i32;
            int32_t largest_negative = 0x80000000;
            int32_t quotient;
            if (divisor == 0){
              quotient = -1;
            } else if (dividen == largest_negative && divisor == -1) {
              quotient = dividen;
            } else {
              quotient = dividen / divisor;
            }
            rd_data[t].i = sext((uint64_t)quotient, 32);
          } else {
            auto dividen = rs1_data[t].i;
            auto divisor = rs2_data[t].i;
            auto largest_negative = WordI(1) << (XLEN-1);
            WordI quotient;
            if (divisor == 0) {
              quotient = -1;
            } else if (dividen == largest_negative && divisor == -1) {
              quotient = dividen;
            } else {
              quotient = dividen / divisor;
            }
            rd_data[t].i = quotient;
          }
        }
      } break;
      case MdvType::DIVU: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          if (is_w_enabled && mdvArgs.is_w) {
            auto dividen = rs1_data[t].u32;
            auto divisor = rs2_data[t].u32;
            uint32_t quotient;
            if (divisor != 0){
              quotient = dividen / divisor;
            } else {
              quotient = -1;
            }
            rd_data[t].i = sext((uint64_t)quotient, 32);
          } else {
            auto dividen = rs1_data[t].u;
            auto divisor = rs2_data[t].u;
            Word quotient;
            if (divisor != 0) {
              quotient = dividen / divisor;
            } else {
              quotient = -1;
            }
            rd_data[t].i = quotient;
          }
        }
      } break;
      case MdvType::REM: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          if (is_w_enabled && mdvArgs.is_w) {
            auto dividen = rs1_data[t].i32;
            auto divisor = rs2_data[t].i32;
            int32_t largest_negative = 0x80000000;
            int32_t remainder;
            if (divisor == 0){
              remainder = dividen;
            } else if (dividen == largest_negative && divisor == -1) {
              remainder = 0;
            } else {
              remainder = dividen % divisor;
            }
            rd_data[t].i = sext((uint64_t)remainder, 32);
          } else {
            auto dividen = rs1_data[t].i;
            auto divisor = rs2_data[t].i;
            auto largest_negative = WordI(1) << (XLEN-1);
            WordI remainder;
            if (rs2_data[t].i == 0) {
              remainder = dividen;
            } else if (dividen == largest_negative && divisor == -1) {
              remainder = 0;
            } else {
              remainder = dividen % divisor;
            }
            rd_data[t].i = remainder;
          }
        }
      } break;
      case MdvType::REMU: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          if (is_w_enabled && mdvArgs.is_w) {
            auto dividen = (uint32_t)rs1_data[t].u32;
            auto divisor = (uint32_t)rs2_data[t].u32;
            uint32_t remainder;
            if (divisor != 0){
              remainder = dividen % divisor;
            } else {
              remainder = dividen;
            }
            rd_data[t].i = sext((uint64_t)remainder, 32);
          } else {
            auto dividen = rs1_data[t].u;
            auto divisor = rs2_data[t].u;
            Word remainder;
            if (rs2_data[t].i != 0) {
              remainder = dividen % divisor;
            } else {
              remainder = dividen;
            }
            rd_data[t].i = remainder;
          }
        }
      } break;
      default:
        std::abort();
      }
      rd_write = true;
    },
    [&](LsuType lsu_type) {
      auto lsuArgs = std::get<IntrLsuArgs>(instrArgs);
      switch (lsu_type) {
      case LsuType::LOAD: {
        auto trace_data = std::make_shared<LsuTraceData>(num_threads);
        trace->data = trace_data;
        uint32_t data_bytes = 1 << (lsuArgs.width & 0x3);
        uint32_t data_width = 8 * data_bytes;
        Word offset = sext<Word>(lsuArgs.offset, 32);
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint64_t mem_addr = rs1_data[t].i + offset;
          uint64_t read_data = 0;
          this->dcache_read(&read_data, mem_addr, data_bytes);
          trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
          switch (lsuArgs.width) {
          case 0: // RV32I: LB
          case 1: // RV32I: LH
            rd_data[t].i = sext((Word)read_data, data_width);
            break;
          case 2:
            if (lsuArgs.is_float) {
              // RV32F: FLW
              rd_data[t].u64 = nan_box((uint32_t)read_data);
            } else {
              // RV32I: LW
              rd_data[t].i = sext((Word)read_data, data_width);
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
      } break;
      case LsuType::STORE: {
        auto trace_data = std::make_shared<LsuTraceData>(num_threads);
        trace->data = trace_data;
        uint32_t data_bytes = 1 << (lsuArgs.width & 0x3);
        Word offset = sext<Word>(lsuArgs.offset, 32);
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint64_t mem_addr = rs1_data[t].i + offset;
          uint64_t write_data = rs2_data[t].u64;
          trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
          switch (lsuArgs.width) {
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
      } break;
      case LsuType::FENCE: {
        // no compute
      } break;
      default:
        std::abort();
      }
    },
    [&](AmoType amo_type) {
      auto amoArgs = std::get<IntrAmoArgs>(instrArgs);
      auto trace_data = std::make_shared<LsuTraceData>(num_threads);
      trace->data = trace_data;
      uint32_t data_bytes = 1 << (amoArgs.width & 0x3);
      uint32_t data_width = 8 * data_bytes;
      switch (amo_type) {
      case AmoType::LR: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint64_t mem_addr = rs1_data[t].u;
          trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
          uint64_t read_data = 0;
          this->dcache_read(&read_data, mem_addr, data_bytes);
          this->dcache_amo_reserve(mem_addr);
          rd_data[t].i = sext((Word)read_data, data_width);
        }
      } break;
      case AmoType::SC: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint64_t mem_addr = rs1_data[t].u;
          trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
          if (this->dcache_amo_check(mem_addr)) {
            this->dcache_write(&rs2_data[t].u64, mem_addr, data_bytes);
            rd_data[t].i = 0;
          } else {
            rd_data[t].i = 1;
          }
        }
      } break;
      case AmoType::AMOADD: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint64_t mem_addr = rs1_data[t].u;
          trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
          uint64_t read_data = 0;
          this->dcache_read(&read_data, mem_addr, data_bytes);
          auto read_data_i = sext((WordI)read_data, data_width);
          auto rs1_data_i  = sext((WordI)rs2_data[t].u64, data_width);
          uint64_t result = read_data_i + rs1_data_i;
          this->dcache_write(&result, mem_addr, data_bytes);
          rd_data[t].i = read_data_i;
        }
      } break;
      case AmoType::AMOSWAP: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint64_t mem_addr = rs1_data[t].u;
          trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
          uint64_t read_data = 0;
          this->dcache_read(&read_data, mem_addr, data_bytes);
          auto read_data_i = sext((WordI)read_data, data_width);
          auto rs1_data_u  = zext((Word)rs2_data[t].u64, data_width);
          uint64_t result = rs1_data_u;
          this->dcache_write(&result, mem_addr, data_bytes);
          rd_data[t].i = read_data_i;
        }
      } break;
      case AmoType::AMOXOR: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint64_t mem_addr = rs1_data[t].u;
          trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
          uint64_t read_data = 0;
          this->dcache_read(&read_data, mem_addr, data_bytes);
          auto read_data_i = sext((WordI)read_data, data_width);
          auto read_data_u = zext((Word)read_data, data_width);
          auto rs1_data_u  = zext((Word)rs2_data[t].u64, data_width);
          uint64_t result = read_data_u ^ rs1_data_u;
          this->dcache_write(&result, mem_addr, data_bytes);
          rd_data[t].i = read_data_i;
        }
      } break;
      case AmoType::AMOOR: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint64_t mem_addr = rs1_data[t].u;
          trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
          uint64_t read_data = 0;
          this->dcache_read(&read_data, mem_addr, data_bytes);
          auto read_data_i = sext((WordI)read_data, data_width);
          auto read_data_u = zext((Word)read_data, data_width);
          auto rs1_data_u  = zext((Word)rs2_data[t].u64, data_width);
          uint64_t result = read_data_u | rs1_data_u;
          this->dcache_write(&result, mem_addr, data_bytes);
          rd_data[t].i = read_data_i;
        }
      } break;
      case AmoType::AMOAND: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint64_t mem_addr = rs1_data[t].u;
          trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
          uint64_t read_data = 0;
          this->dcache_read(&read_data, mem_addr, data_bytes);
          auto read_data_i = sext((WordI)read_data, data_width);
          auto read_data_u = zext((Word)read_data, data_width);
          auto rs1_data_u  = zext((Word)rs2_data[t].u64, data_width);
          uint64_t result = read_data_u & rs1_data_u;
          this->dcache_write(&result, mem_addr, data_bytes);
          rd_data[t].i = read_data_i;
        }
      } break;
      case AmoType::AMOMIN: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint64_t mem_addr = rs1_data[t].u;
          trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
          uint64_t read_data = 0;
          this->dcache_read(&read_data, mem_addr, data_bytes);
          auto read_data_i = sext((WordI)read_data, data_width);
          auto rs1_data_i  = sext((WordI)rs2_data[t].u64, data_width);
          uint64_t result = std::min(read_data_i, rs1_data_i);
          this->dcache_write(&result, mem_addr, data_bytes);
          rd_data[t].i = read_data_i;
        }
      } break;
      case AmoType::AMOMAX: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint64_t mem_addr = rs1_data[t].u;
          trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
          uint64_t read_data = 0;
          this->dcache_read(&read_data, mem_addr, data_bytes);
          auto read_data_i = sext((WordI)read_data, data_width);
          auto rs1_data_i  = sext((WordI)rs2_data[t].u64, data_width);
          uint64_t result = std::max(read_data_i, rs1_data_i);
          this->dcache_write(&result, mem_addr, data_bytes);
          rd_data[t].i = read_data_i;
        }
      } break;
      case AmoType::AMOMINU: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint64_t mem_addr = rs1_data[t].u;
          trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
          uint64_t read_data = 0;
          this->dcache_read(&read_data, mem_addr, data_bytes);
          auto read_data_i = sext((WordI)read_data, data_width);
          auto read_data_u = zext((Word)read_data, data_width);
          auto rs1_data_u  = zext((Word)rs2_data[t].u64, data_width);
          uint64_t result = std::min(read_data_u, rs1_data_u);
          this->dcache_write(&result, mem_addr, data_bytes);
          rd_data[t].i = read_data_i;
        }
      } break;
      case AmoType::AMOMAXU: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint64_t mem_addr = rs1_data[t].u;
          trace_data->mem_addrs.at(t) = {mem_addr, data_bytes};
          uint64_t read_data = 0;
          this->dcache_read(&read_data, mem_addr, data_bytes);
          auto read_data_i = sext((WordI)read_data, data_width);
          auto read_data_u = zext((Word)read_data, data_width);
          auto rs1_data_u  = zext((Word)rs2_data[t].u64, data_width);
          uint64_t result = std::max(read_data_u, rs1_data_u);
          this->dcache_write(&result, mem_addr, data_bytes);
          rd_data[t].i = read_data_i;
        }
      } break;
      default:
        std::abort();
      }
      rd_write = true;
    },
    [&](FpuType fpu_type) {
      auto fpuArgs = std::get<IntrFpuArgs>(instrArgs);
      switch (fpu_type) {
      case FpuType::FADD: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t frm = this->get_fpu_rm(fpuArgs.frm, wid, t);
          uint32_t fflags = 0;
          if (fpuArgs.is_f64) {
            rd_data[t].u64 = rv_fadd_d(rs1_data[t].u64, rs2_data[t].u64, frm, &fflags);
          } else {
            rd_data[t].u64 = nan_box(rv_fadd_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), frm, &fflags));
          }
          this->update_fcrs(fflags, wid, t);
        }
      } break;
      case FpuType::FSUB: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t frm = this->get_fpu_rm(fpuArgs.frm, wid, t);
          uint32_t fflags = 0;
          if (fpuArgs.is_f64) {
            rd_data[t].u64 = rv_fsub_d(rs1_data[t].u64, rs2_data[t].u64, frm, &fflags);
          } else {
            rd_data[t].u64 = nan_box(rv_fsub_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), frm, &fflags));
          }
          this->update_fcrs(fflags, wid, t);
        }
      } break;
      case FpuType::FMUL: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t frm = this->get_fpu_rm(fpuArgs.frm, wid, t);
          uint32_t fflags = 0;
          if (fpuArgs.is_f64) {
            rd_data[t].u64 = rv_fmul_d(rs1_data[t].u64, rs2_data[t].u64, frm, &fflags);
          } else {
            rd_data[t].u64 = nan_box(rv_fmul_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), frm, &fflags));
          }
          this->update_fcrs(fflags, wid, t);
        }
      } break;
      case FpuType::FDIV: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t frm = this->get_fpu_rm(fpuArgs.frm, wid, t);
          uint32_t fflags = 0;
          if (fpuArgs.is_f64) {
            rd_data[t].u64 = rv_fdiv_d(rs1_data[t].u64, rs2_data[t].u64, frm, &fflags);
          } else {
            rd_data[t].u64 = nan_box(rv_fdiv_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), frm, &fflags));
          }
          this->update_fcrs(fflags, wid, t);
        }
      } break;
      case FpuType::FSQRT: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t frm = this->get_fpu_rm(fpuArgs.frm, wid, t);
          uint32_t fflags = 0;
          if (fpuArgs.is_f64) {
            rd_data[t].u64 = rv_fsqrt_d(rs1_data[t].u64, frm, &fflags);
          } else {
            rd_data[t].u64 = nan_box(rv_fsqrt_s(check_boxing(rs1_data[t].u64), frm, &fflags));
          }
          this->update_fcrs(fflags, wid, t);
        }
      } break;
      case FpuType::FSGNJ: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t fflags = 0;
          if (fpuArgs.is_f64) {
            switch (fpuArgs.frm) {
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
          } else {
            switch (fpuArgs.frm) {
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
          }
          this->update_fcrs(fflags, wid, t);
        }
      } break;
      case FpuType::FMINMAX: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t fflags = 0;
          if (fpuArgs.is_f64) {
            if (fpuArgs.frm) {
              rd_data[t].u64 = rv_fmax_d(rs1_data[t].u64, rs2_data[t].u64, &fflags);
            } else {
              rd_data[t].u64 = rv_fmin_d(rs1_data[t].u64, rs2_data[t].u64, &fflags);
            }
          } else {
            if (fpuArgs.frm) {
              rd_data[t].u64 = nan_box(rv_fmax_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags));
            } else {
              rd_data[t].u64 = nan_box(rv_fmin_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags));
            }
          }
          this->update_fcrs(fflags, wid, t);
        }
      } break;
      case FpuType::FCMP: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t fflags = 0;
          if (fpuArgs.is_f64) {
            switch (fpuArgs.frm) {
            case 0: // RV32D: FLE.D
              rd_data[t].i = rv_fle_d(rs1_data[t].u64, rs2_data[t].u64, &fflags);
              break;
            case 1: // RV32D: FLT.D
              rd_data[t].i = rv_flt_d(rs1_data[t].u64, rs2_data[t].u64, &fflags);
              break;
            case 2: // RV32D: FEQ.D
              rd_data[t].i = rv_feq_d(rs1_data[t].u64, rs2_data[t].u64, &fflags);
              break;
            }
          } else {
            switch (fpuArgs.frm) {
            case 0: // RV32F: FLE.S
              rd_data[t].i = rv_fle_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags);
              break;
            case 1: // RV32F: FLT.S
              rd_data[t].i = rv_flt_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags);
              break;
            case 2: // RV32F: FEQ.S
              rd_data[t].i = rv_feq_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), &fflags);
              break;
            }
          }
          this->update_fcrs(fflags, wid, t);
        }
      } break;
      case FpuType::F2I: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t frm = this->get_fpu_rm(fpuArgs.frm, wid, t);
          uint32_t fflags = 0;
          if (fpuArgs.is_f64) {
            switch (fpuArgs.cvt) {
            case 0: // RV32D: FCVT.W.D
              rd_data[t].i = sext((uint64_t)rv_ftoi_d(rs1_data[t].u64, frm, &fflags), 32);
              break;
            case 1: // RV32D: FCVT.WU.D
              rd_data[t].i = sext((uint64_t)rv_ftou_d(rs1_data[t].u64, frm, &fflags), 32);
              break;
            case 2: // RV64D: FCVT.L.D
              rd_data[t].i = rv_ftol_d(rs1_data[t].u64, frm, &fflags);
              break;
            case 3: // RV64D: FCVT.LU.D
              rd_data[t].i = rv_ftolu_d(rs1_data[t].u64, frm, &fflags);
              break;
            }
          } else {
            switch (fpuArgs.cvt) {
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
          }
          this->update_fcrs(fflags, wid, t);
        }
      } break;
      case FpuType::I2F: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t frm = this->get_fpu_rm(fpuArgs.frm, wid, t);
          uint32_t fflags = 0;
          if (fpuArgs.is_f64) {
            switch (fpuArgs.cvt) {
            case 0: // RV32D: FCVT.D.W
              rd_data[t].u64 = rv_itof_d(rs1_data[t].i, frm, &fflags);
              break;
            case 1: // RV32D: FCVT.D.WU
              rd_data[t].u64 = rv_utof_d(rs1_data[t].i, frm, &fflags);
              break;
            case 2: // RV64D: FCVT.D.L
              rd_data[t].u64 = rv_ltof_d(rs1_data[t].i, frm, &fflags);
              break;
            case 3: // RV64D: FCVT.D.LU
              rd_data[t].u64 = rv_lutof_d(rs1_data[t].i, frm, &fflags);
              break;
            }
          } else {
            switch (fpuArgs.cvt) {
            case 0: // RV32F: FCVT.S.W
              rd_data[t].u64 = nan_box(rv_itof_s(rs1_data[t].i, frm, &fflags));
              break;
            case 1: // RV32F: FCVT.S.WU
              rd_data[t].u64 = nan_box(rv_utof_s(rs1_data[t].i, frm, &fflags));
              break;
            case 2: // RV64F: FCVT.S.L
              rd_data[t].u64 = nan_box(rv_ltof_s(rs1_data[t].i, frm, &fflags));
              break;
            case 3: // RV64F: FCVT.S.LU
              rd_data[t].u64 = nan_box(rv_lutof_s(rs1_data[t].i, frm, &fflags));
              break;
            }
          }
          this->update_fcrs(fflags, wid, t);
        }
      } break;
      case FpuType::F2F: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t fflags = 0;
          if (fpuArgs.is_f64) {
            rd_data[t].u64 = rv_ftod(check_boxing(rs1_data[t].u64));
          } else {
            rd_data[t].u64 = nan_box(rv_dtof(rs1_data[t].u64));
          }
          this->update_fcrs(fflags, wid, t);
        }
      } break;
      case FpuType::FCLASS: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t fflags = 0;
          if (fpuArgs.is_f64) {
            rd_data[t].i = rv_fclss_d(rs1_data[t].u64);
          } else {
            rd_data[t].i = rv_fclss_s(check_boxing(rs1_data[t].u64));
          }
          this->update_fcrs(fflags, wid, t);
        }
      } break;
      case FpuType::FMVXW: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t fflags = 0;
          if (fpuArgs.is_f64) { // RV64D: FMV.X.D
            rd_data[t].u64 = rs1_data[t].u64;
          } else { // RV32F: FMV.X.S
            uint32_t result = (uint32_t)rs1_data[t].u64;
            rd_data[t].i = sext((uint64_t)result, 32);
          }
          this->update_fcrs(fflags, wid, t);
        }
      } break;
      case FpuType::FMVWX: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t fflags = 0;
          if (fpuArgs.is_f64) { // RV64D: FMV.D.X
            rd_data[t].u64 = rs1_data[t].i;
          } else { // RV32F: FMV.S.X
            rd_data[t].u64 = nan_box((uint32_t)rs1_data[t].i);
          }
          this->update_fcrs(fflags, wid, t);
        }
      } break;
      case FpuType::FMADD:
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t frm = this->get_fpu_rm(fpuArgs.frm, wid, t);
          uint32_t fflags = 0;
            if (fpuArgs.is_f64) {
            rd_data[t].u64 = rv_fmadd_d(rs1_data[t].u64, rs2_data[t].u64, rs3_data[t].u64, frm, &fflags);
          } else {
            rd_data[t].u64 = nan_box(rv_fmadd_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), check_boxing(rs3_data[t].u64), frm, &fflags));
          }
          this->update_fcrs(fflags, wid, t);
        }
        break;
      case FpuType::FMSUB: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t frm = this->get_fpu_rm(fpuArgs.frm, wid, t);
          uint32_t fflags = 0;
          if (fpuArgs.is_f64) {
            rd_data[t].u64 = rv_fmsub_d(rs1_data[t].u64, rs2_data[t].u64, rs3_data[t].u64, frm, &fflags);
          } else {
            rd_data[t].u64 = nan_box(rv_fmsub_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), check_boxing(rs3_data[t].u64), frm, &fflags));
          }
          this->update_fcrs(fflags, wid, t);
        }
      } break;
      case FpuType::FNMADD: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t frm = this->get_fpu_rm(fpuArgs.frm, wid, t);
          uint32_t fflags = 0;
          if (fpuArgs.is_f64) {
            rd_data[t].u64 = rv_fnmadd_d(rs1_data[t].u64, rs2_data[t].u64, rs3_data[t].u64, frm, &fflags);
          } else {
            rd_data[t].u64 = nan_box(rv_fnmadd_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), check_boxing(rs3_data[t].u64), frm, &fflags));
          }
          this->update_fcrs(fflags, wid, t);
        }
      } break;
      case FpuType::FNMSUB: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          uint32_t frm = this->get_fpu_rm(fpuArgs.frm, wid, t);
          uint32_t fflags = 0;
          if (fpuArgs.is_f64) {
            rd_data[t].u64 = rv_fnmsub_d(rs1_data[t].u64, rs2_data[t].u64, rs3_data[t].u64, frm, &fflags);
          } else {
            rd_data[t].u64 = nan_box(rv_fnmsub_s(check_boxing(rs1_data[t].u64), check_boxing(rs2_data[t].u64), check_boxing(rs3_data[t].u64), frm, &fflags));
          }
          this->update_fcrs(fflags, wid, t);
        }
      } break;
      default:
        std::abort();
      }
      rd_write = true;
    },
    [&](CsrType csr_type) {
      auto csrArgs = std::get<IntrCsrArgs>(instrArgs);
      uint32_t csr_addr = csrArgs.csr;
      switch (csr_type) {
      case CsrType::CSRRW: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          Word csr_value = this->get_csr(csr_addr, wid, t);
          auto src_data = csrArgs.is_imm ? csrArgs.imm : rs1_data[t].i;
          this->set_csr(csr_addr, src_data, wid, t);
          rd_data[t].i = csr_value;
        }
      } break;
      case CsrType::CSRRS: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          Word csr_value = this->get_csr(csr_addr, wid, t);
          auto src_data = csrArgs.is_imm ? csrArgs.imm : rs1_data[t].i;
          if (src_data != 0) {
            this->set_csr(csr_addr, csr_value | src_data, wid, t);
          }
          rd_data[t].i = csr_value;
        }
      } break;
      case CsrType::CSRRC: {
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          Word csr_value = this->get_csr(csr_addr, wid, t);
          auto src_data = csrArgs.is_imm ? csrArgs.imm : rs1_data[t].i;
          if (src_data != 0) {
            this->set_csr(csr_addr, csr_value & ~src_data, wid, t);
          }
          rd_data[t].i = csr_value;
        }
      } break;
      default:
        std::abort();
      }
      trace->fetch_stall = (csr_addr <= VX_CSR_FCSR);
      rd_write = true;
    },
    [&](WctlType wctl_type) {
      auto wctlArgs = std::get<IntrWctlArgs>(instrArgs);
      switch (wctl_type) {
      case WctlType::TMC: {
        trace->fetch_stall = true;
        next_tmask.reset();
        for (uint32_t t = 0; t < num_threads; ++t) {
          next_tmask.set(t, rs1_data.at(thread_last).u & (1 << t));
        }
      } break;
      case WctlType::WSPAWN: {
        trace->fetch_stall = true;
        trace->data = std::make_shared<SfuTraceData>(rs1_data.at(thread_last).u, rs2_data.at(thread_last).u);
      } break;
      case WctlType::SPLIT: {
        trace->fetch_stall = true;
        auto stack_size = warp.ipdom_stack.size();

        ThreadMask then_tmask(num_threads);
        ThreadMask else_tmask(num_threads);
        auto not_pred = wctlArgs.is_neg;
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
      case WctlType::JOIN: {
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
      case WctlType::BAR: {
        trace->fetch_stall = true;
        trace->data = std::make_shared<SfuTraceData>(rs1_data[thread_last].i, rs2_data[thread_last].i);
      } break;
      case WctlType::PRED: {
        trace->fetch_stall = true;
        ThreadMask pred(num_threads);
        auto not_pred = wctlArgs.is_neg;
        for (uint32_t t = 0; t < num_threads; ++t) {
          auto cond = (rs1_data.at(t).i & 0x1) ^ not_pred;
          pred[t] = warp.tmask.test(t) && cond;
        }
        if (pred.any()) {
          next_tmask &= pred;
        } else {
          next_tmask = ThreadMask(num_threads, rs2_data.at(thread_last).u);
        }
      } break;
      default:
        std::abort();
      }
    }
  #ifdef EXT_V_ENABLE
    ,[&](VsetType /*vset_type*/) {
      auto trace_data = std::make_shared<VecUnit::ExeTraceData>();
      trace->data = trace_data;
      for (uint32_t t = thread_start; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vec_unit_->configure(instr, wid, t, rs1_data, rs2_data, rd_data, trace_data.get());
      }
      rd_write = true;
    },
    [&](VlsType vls_type) {
      switch (vls_type) {
      case VlsType::VL:
      case VlsType::VLS:
      case VlsType::VLX: {
        auto trace_data = std::make_shared<VecUnit::MemTraceData>(num_threads);
        trace->data = trace_data;
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          vec_unit_->load(instr, wid, t, rs1_data, rs2_data, trace_data.get());
        }
        rd_write = true;
      } break;
      case VlsType::VS:
      case VlsType::VSS:
      case VlsType::VSX: {
        auto trace_data = std::make_shared<VecUnit::MemTraceData>(num_threads);
        trace->data = trace_data;
        for (uint32_t t = thread_start; t < num_threads; ++t) {
          if (!warp.tmask.test(t))
            continue;
          vec_unit_->store(instr, wid, t, rs1_data, rs2_data, trace_data.get());
        }
      } break;
      default:
        std::abort();
      }
    },
    [&](VopType /*vop_type*/) {
      auto trace_data = std::make_shared<VecUnit::ExeTraceData>();
      trace->data = trace_data;
      for (uint32_t t = thread_start; t < num_threads; ++t) {
        if (!warp.tmask.test(t))
          continue;
        vec_unit_->execute(instr, wid, t, rs1_data, rd_data, trace_data.get());
      }
      rd_write = true;
    }
  #endif // EXT_V_ENABLE
  #ifdef EXT_TCU_ENABLE
    ,[&](TcuType tcu_type) {
      auto tpuArgs = std::get<IntrTcuArgs>(instrArgs);
      switch (tcu_type) {
      case TcuType::WMMA: {
        auto trace_data = std::make_shared<TensorUnit::ExeTraceData>();
        trace->data = trace_data;
        assert(warp.tmask.count() == num_threads);
        tensor_unit_->wmma(wid, tpuArgs.fmt_s, tpuArgs.fmt_d, tpuArgs.step_m, tpuArgs.step_n, rs1_data, rs2_data, rs3_data, rd_data, trace_data.get());
        rd_write = true;
      } break;
      default:
        std::abort();
      }
    }
  #endif // EXT_TCU_ENABLE
  );

  if (rd_write) {
    trace->wb = true;
    switch (rdest.type) {
    case RegType::None:
      break;
    case RegType::Integer:
      if (rdest.idx != 0) {
        DPH(2, "Dest Reg: " << rdest << "={");
        for (uint32_t t = 0; t < num_threads; ++t) {
          if (t) DPN(2, ", ");
          if (!warp.tmask.test(t)) {
            DPN(2, "-");
            continue;
          }
          warp.ireg_file.at(rdest.idx).at(t) = rd_data[t].i;
          DPN(2, "0x" << std::hex << rd_data[t].u << std::dec);
        }
        DPN(2, "}" << std::endl);
      } else {
        // disable writes to x0
        trace->wb = false;
      }
      break;
    case RegType::Float:
      DPH(2, "Dest Reg: " << rdest << "={");
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (t) DPN(2, ", ");
        if (!warp.tmask.test(t)) {
          DPN(2, "-");
          continue;
        }
        warp.freg_file.at(rdest.idx).at(t) = rd_data[t].u64;
        if ((rd_data[t].u64 >> 32) == 0xffffffff) {
          DPN(2, "0x" << std::hex << rd_data[t].u32 << std::dec);
        } else {
          DPN(2, "0x" << std::hex << rd_data[t].u64 << std::dec);
        }
      }
      DPN(2, "}" << std::endl);
      break;
  #ifdef EXT_V_ENABLE
    case RegType::Vector:
      DPH(2, "Dest Reg: " << rdest << "={");
      for (uint32_t t = 0; t < num_threads; ++t) {
        if (t) DPN(2, ", ");
        if (!warp.tmask.test(t)) {
          DPN(2, "-");
          continue;
        }
        DPN(2, vec_unit_->dumpRegister(wid, t, rdest.idx));
      }
      DPN(2, "}" << std::endl);
      break;
  #endif
    default:
      std::cout << "Unrecognized register write back type: " << rdest.type << std::endl;
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
    DP(3, "*** New Tmask=" << next_tmask);
    warp.tmask = next_tmask;
    if (!next_tmask.any()) {
      active_warps_.reset(wid);
    }
  }

  DP(5, "Register state:");
  for (uint32_t i = 0; i < MAX_NUM_REGS; ++i) {
    DPN(5, "  %r" << std::setfill('0') << std::setw(2) << i << ':' << std::hex);
    // Integer register file
    for (uint32_t j = 0; j < arch_.num_threads(); ++j) {
      DPN(5, ' ' << std::setfill('0') << std::setw(XLEN/4) << warp.ireg_file.at(i).at(j) << std::setfill(' ') << ' ');
    }
    DPN(5, '|');
    // Floating point register file
    for (uint32_t j = 0; j < arch_.num_threads(); ++j) {
      DPN(5, ' ' << std::setfill('0') << std::setw(16) << warp.freg_file.at(i).at(j) << std::setfill(' ') << ' ');
    }
    DPN(5, std::dec << std::endl);
  }

  return trace;
}