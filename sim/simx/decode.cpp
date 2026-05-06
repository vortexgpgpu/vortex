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
#include <string>
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include <iomanip>
#include <vector>
#include <unordered_map>
#include <util.h>
#include "debug.h"
#include "types.h"
#include "decode.h"
#include "instr.h"

#ifdef EXT_TCU_ENABLE
#include "tensor_cfg.h"
#include "tcu_unit.h"
#endif

using namespace vortex;

static op_string_t op_string(const Instr &instr) {
  auto op_type = instr.get_op_type();
  auto instrArgs = instr.get_args();
  return visit_var(op_type,
    [&](AluType alu_type)-> op_string_t {
      auto aluArgs = std::get<IntrAluArgs>(instrArgs);
      switch (alu_type) {
      case AluType::LUI:   return {"LUI", to_hex_str(aluArgs.imm)};
      case AluType::AUIPC: return {"AUIPC", to_hex_str(aluArgs.imm)};
      case AluType::ADD: {
        if (aluArgs.is_imm) {
          return {aluArgs.is_w ? "ADDIW":"ADDI", to_hex_str(aluArgs.imm)};
        } else {
          return {aluArgs.is_w ? "ADDW":"ADD", ""};
        }
      }
      case AluType::SUB: {
        if (aluArgs.is_imm) {
          return {aluArgs.is_w ? "SUBW":"SUB", to_hex_str(aluArgs.imm)};
        } else {
          return {aluArgs.is_w ? "SUBW":"SUB", ""};
        }
      }
      case AluType::SLL: {
        if (aluArgs.is_imm) {
          return {aluArgs.is_w ? "SLLIW":"SLLI", to_hex_str(aluArgs.imm)};
        } else {
          return {aluArgs.is_w ? "SLLW":"SLL", ""};
        }
      }
      case AluType::SRL: {
        if (aluArgs.is_imm) {
          return {aluArgs.is_w ? "SRLIW":"SRLI", to_hex_str(aluArgs.imm)};
        } else {
          return {aluArgs.is_w ? "SRLW":"SRL", ""};
        }
      }
      case AluType::SRA: {
        if (aluArgs.is_imm) {
          return {aluArgs.is_w ? "SRAIW":"SRAI", to_hex_str(aluArgs.imm)};
        } else {
          return {aluArgs.is_w ? "SRAW":"SRA", ""};
        }
      }
      case AluType::SLT: {
        if (aluArgs.is_imm) {
          return {"SLTI", to_hex_str(aluArgs.imm)};
        } else {
          return {"SLT", ""};
        }
      }
      case AluType::SLTU: {
        if (aluArgs.is_imm) {
          return {"SLTIU", to_hex_str(aluArgs.imm)};
        } else {
          return {"SLTU", ""};
        }
      }
      case AluType::AND: {
        if (aluArgs.is_imm) {
          return {"ANDI", to_hex_str(aluArgs.imm)};
        } else {
          return {"AND", ""};
        }
      }
      case AluType::OR: {
        if (aluArgs.is_imm) {
          return {"ORI", to_hex_str(aluArgs.imm)};
        } else {
          return {"OR", ""};
        }
      }
      case AluType::XOR: {
        if (aluArgs.is_imm) {
          return {"XORI", to_hex_str(aluArgs.imm)};
        } else {
          return {"XOR", ""};
        }
      }
      case AluType::CZERO: return {aluArgs.imm ? "CZERO.NEZ":"CZERO.EQZ", ""};
      default:
        std::abort();
      }
    },
    [&](VoteType vote_type)-> op_string_t {
      switch (vote_type) {
      case VoteType::ALL: return {"VOTE.ALL", ""};
      case VoteType::ANY: return {"VOTE.ANY", ""};
      case VoteType::UNI: return {"VOTE.UNI", ""};
      case VoteType::BAL: return {"VOTE.BAL", ""};
      default:
        std::abort();
      }
    },
    [&](ShflType shfl_type)-> op_string_t {
      switch (shfl_type) {
      case ShflType::UP:   return {"SHFL.UP", ""};
      case ShflType::DOWN: return {"SHFL.DOWN", ""};
      case ShflType::BFLY: return {"SHFL.BFLY", ""};
      case ShflType::IDX:  return {"SHFL.IDX", ""};
      default:
        std::abort();
      }
    },
    [&](WgatherType)-> op_string_t {
      return {"WGATHER", ""};
    },
    [&](BrType br_type)-> op_string_t {
      auto brArgs = std::get<IntrBrArgs>(instrArgs);
      switch (br_type) {
      case BrType::BR: {
        switch (brArgs.cmp) {
        case 0: return {"BEQ",  to_hex_str(brArgs.offset)};
        case 1: return {"BNE",  to_hex_str(brArgs.offset)};
        case 4: return {"BLT",  to_hex_str(brArgs.offset)};
        case 5: return {"BGE",  to_hex_str(brArgs.offset)};
        case 6: return {"BLTU", to_hex_str(brArgs.offset)};
        case 7: return {"BGEU", to_hex_str(brArgs.offset)};
        default:
          std::abort();
        }
      }
      case BrType::JAL:  return {"JAL", to_hex_str(brArgs.offset)};
      case BrType::JALR: return {"JALR", to_hex_str(brArgs.offset)};
      case BrType::SYS:
        switch (brArgs.offset) {
        case 0x000: return {"ECALL", ""};
        case 0x001: return {"EBREAK", ""};
        case 0x002: return {"URET", ""};
        case 0x102: return {"SRET", ""};
        case 0x302: return {"MRET", ""};
        default:
          std::abort();
        }
      default:
        std::abort();
      }
    },
    [&](MdvType mdv_type)-> op_string_t {
      auto mdvArgs = std::get<IntrMdvArgs>(instrArgs);
      switch (mdv_type) {
      case MdvType::MUL:    return {mdvArgs.is_w ? "MULW":"MUL", ""};
      case MdvType::MULHU:  return {"MULHU", ""};
      case MdvType::MULH:   return {"MULH", ""};
      case MdvType::MULHSU: return {"MULHSU", ""};
      case MdvType::DIV:    return {mdvArgs.is_w ? "DIVW":"DIV", ""};
      case MdvType::DIVU:   return {mdvArgs.is_w ? "DIVUW":"DIVU", ""};
      case MdvType::REM:    return {mdvArgs.is_w ? "REMW":"REM", ""};
      case MdvType::REMU:   return {mdvArgs.is_w ? "REMUW":"REMU", ""};
      default:
        std::abort();
      }
    },
    [&](FpuType fpu_type)-> op_string_t {
      auto fpuArgs = std::get<IntrFpuArgs>(instrArgs);
      switch (fpu_type) {
      case FpuType::FADD:   return {fpuArgs.is_f64 ? "FADD.D":"FADD.S", ""};
      case FpuType::FSUB:   return {fpuArgs.is_f64 ? "FSUB.D":"FSUB.S", ""};
      case FpuType::FMUL:   return {fpuArgs.is_f64 ? "FMUL.D":"FMUL.S", ""};
      case FpuType::FDIV:   return {fpuArgs.is_f64 ? "FDIV.D":"FDIV.S", ""};
      case FpuType::FSQRT:  return {fpuArgs.is_f64 ? "FSQRT.D":"FSQRT.S", ""};
      case FpuType::FMADD:  return {fpuArgs.is_f64 ? "FMADD.D":"FMADD.S", ""};
      case FpuType::FMSUB:  return {fpuArgs.is_f64 ? "FMSUB.D":"FMSUB.S", ""};
      case FpuType::FNMADD: return {fpuArgs.is_f64 ? "FNMADD.D":"FNMADD.S", ""};
      case FpuType::FNMSUB: return {fpuArgs.is_f64 ? "FNMSUB.D":"FNMSUB.S", ""};
      case FpuType::F2I: {
        switch (fpuArgs.cvt) {
        case 0: return {fpuArgs.is_f64 ? "FCVT.W.D":"FCVT.W.S", ""};
        case 1: return {fpuArgs.is_f64 ? "FCVT.WU.D":"FCVT.WU.S", ""};
        case 2: return {fpuArgs.is_f64 ? "FCVT.L.D":"FCVT.L.S", ""};
        case 3: return {fpuArgs.is_f64 ? "FCVT.LU.D":"FCVT.LU.S", ""};
        default:
          std::abort();
        }
      }
      case FpuType::I2F: {
        switch (fpuArgs.cvt) {
        case 0: return {fpuArgs.is_f64 ? "FCVT.D.W":"FCVT.S.W", ""};
        case 1: return {fpuArgs.is_f64 ? "FCVT.D.WU":"FCVT.S.WU", ""};
        case 2: return {fpuArgs.is_f64 ? "FCVT.D.L":"FCVT.S.L", ""};
        case 3: return {fpuArgs.is_f64 ? "FCVT.D.LU":"FCVT.S.LU", ""};
        default:
          std::abort();
        }
      }
      case FpuType::F2F: return {fpuArgs.is_f64 ? "FCVT.D.S":"FCVT.S.D", ""};
      case FpuType::FCMP: {
        switch (fpuArgs.frm) {
        case 0: return {fpuArgs.is_f64 ? "FLE.D":"FLE.S", ""};
        case 1: return {fpuArgs.is_f64 ? "FLT.D":"FLT.S", ""};
        case 2: return {fpuArgs.is_f64 ? "FEQ.D":"FEQ.S", ""};
        default:
          std::abort();
        }
      }
      case FpuType::FSGNJ: {
        switch (fpuArgs.frm) {
        case 0: return {fpuArgs.is_f64 ? "FSGNJ.D":"FSGNJ.S", ""};
        case 1: return {fpuArgs.is_f64 ? "FSGNJN.D":"FSGNJN.S", ""};
        case 2: return {fpuArgs.is_f64 ? "FSGNJX.D":"FSGNJX.S", ""};
        default:
          std::abort();
        }
      }
      case FpuType::FCLASS: return {fpuArgs.is_f64 ? "FCLASS.D":"FCLASS.S", ""};
      case FpuType::FMVXW:  return {fpuArgs.is_f64 ? "FMV.X.D":"FMV.X.S", ""};
      case FpuType::FMVWX:  return {fpuArgs.is_f64 ? "FMV.D.X":"FMV.S.X", ""};
      case FpuType::FMINMAX: {
        switch (fpuArgs.frm) {
        case 0: return {fpuArgs.is_f64 ? "FMIN.D":"FMIN.S", ""};
        case 1: return {fpuArgs.is_f64 ? "FMAX.D":"FMAX.S", ""};
        default:
          std::abort();
        }
      }
      default:
        std::abort();
      }
    },
    [&](LsuType lsu_type)-> op_string_t {
      switch (lsu_type) {
      case LsuType::LOAD: {
        auto lsuArgs = std::get<IntrLsuArgs>(instrArgs);
        bool dst_float = (instr.get_dest_reg().type == RegType::Float);
        // packLD detection: Float dest with sub-word width (LB or LH).
        if (dst_float && lsuArgs.width == 0) return {"PACKLB.F", ""};
        if (dst_float && lsuArgs.width == 1) return {"PACKLH.F", ""};
        if (dst_float) {
          switch (lsuArgs.width) {
          case 2: return {"FLW", to_hex_str(lsuArgs.offset)};
          case 3: return {"FLD", to_hex_str(lsuArgs.offset)};
          default:
            std::abort();
          }
        } else {
          switch (lsuArgs.width) {
          case 0: return {"LB",  to_hex_str(lsuArgs.offset)};
          case 1: return {"LH",  to_hex_str(lsuArgs.offset)};
          case 2: return {"LW",  to_hex_str(lsuArgs.offset)};
          case 3: return {"LD",  to_hex_str(lsuArgs.offset)};
          case 4: return {"LBU", to_hex_str(lsuArgs.offset)};
          case 5: return {"LHU", to_hex_str(lsuArgs.offset)};
          case 6: return {"LWU", to_hex_str(lsuArgs.offset)};
          default:
            std::abort();
          }
        }
      }
      case LsuType::STORE: {
        auto lsuArgs = std::get<IntrLsuArgs>(instrArgs);
        // For stores, the data source (rs2) carries the type info.
        bool src_float = (instr.get_src_reg(1).type == RegType::Float);
        if (src_float) {
          switch (lsuArgs.width) {
          case 2: return {"FSW", to_hex_str(lsuArgs.offset)};
          case 3: return {"FSD", to_hex_str(lsuArgs.offset)};
          default:
            std::abort();
          }
        } else {
          switch (lsuArgs.width) {
          case 0: return {"SB", to_hex_str(lsuArgs.offset)};
          case 1: return {"SH", to_hex_str(lsuArgs.offset)};
          case 2: return {"SW", to_hex_str(lsuArgs.offset)};
          case 3: return {"SD", to_hex_str(lsuArgs.offset)};
          default:
            std::abort();
          }
        }
      }
      case LsuType::FENCE: return {"FENCE", ""};
      default:
        std::abort();
      }
    },
    [&](AmoType amo_type)-> op_string_t {
      auto amoArgs = std::get<IntrAmoArgs>(instrArgs);
      switch (amoArgs.width) {
      case 2: {
        switch (amo_type) {
        case AmoType::LR:      return {"LR.W", ""};
        case AmoType::SC:      return {"SC.W", ""};
        case AmoType::AMOADD:  return {"AMOADD.W", ""};
        case AmoType::AMOSWAP: return {"AMOSWAP.W", ""};
        case AmoType::AMOAND:  return {"AMOAND.W", ""};
        case AmoType::AMOOR:   return {"AMOOR.W", ""};
        case AmoType::AMOXOR:  return {"AMOXOR.W", ""};
        case AmoType::AMOMIN:  return {"AMOMIN.W", ""};
        case AmoType::AMOMAX:  return {"AMOMAX.W", ""};
        case AmoType::AMOMINU: return {"AMOMINU.W", ""};
        case AmoType::AMOMAXU: return {"AMOMAXU.W", ""};
        default:
          std::abort();
        }
      }
      case 3: {
        switch (amo_type) {
        case AmoType::LR:      return {"LR.D", ""};
        case AmoType::SC:      return {"SC.D", ""};
        case AmoType::AMOADD:  return {"AMOADD.D", ""};
        case AmoType::AMOSWAP: return {"AMOSWAP.D", ""};
        case AmoType::AMOAND:  return {"AMOAND.D", ""};
        case AmoType::AMOOR:   return {"AMOOR.D", ""};
        case AmoType::AMOXOR:  return {"AMOXOR.D", ""};
        case AmoType::AMOMIN:  return {"AMOMIN.D", ""};
        case AmoType::AMOMAX:  return {"AMOMAX.D", ""};
        case AmoType::AMOMINU: return {"AMOMINU.D", ""};
        case AmoType::AMOMAXU: return {"AMOMAXU.D", ""};
        default:
          std::abort();
        }
      }
      default:
        std::abort();
      }
    },
    [&](CsrType csr_type)-> op_string_t {
      auto csrArgs = std::get<IntrCsrArgs>(instrArgs);
      if (csrArgs.is_imm) {
        switch (csr_type) {
        case CsrType::CSRRW: return {"CSRRWI", to_hex_str(csrArgs.imm) + ", " + to_hex_str(csrArgs.csr)};
        case CsrType::CSRRS: return {"CSRRSI", to_hex_str(csrArgs.imm) + ", " + to_hex_str(csrArgs.csr)};
        case CsrType::CSRRC: return {"CSRRCI", to_hex_str(csrArgs.imm) + ", " + to_hex_str(csrArgs.csr)};
        default:
          std::abort();
        }
      } else {
        switch (csr_type) {
        case CsrType::CSRRW: return {"CSRRW", to_hex_str(csrArgs.csr)};
        case CsrType::CSRRS: return {"CSRRS", to_hex_str(csrArgs.csr)};
        case CsrType::CSRRC: return {"CSRRC", to_hex_str(csrArgs.csr)};
        default:
          std::abort();
        }
      }
    },
    [&](WctlType wctl_type)-> op_string_t {
      auto wctlArgs = std::get<IntrWctlArgs>(instrArgs);
      switch (wctl_type) {
      case WctlType::TMC:    return {"TMC", ""};
      case WctlType::WSPAWN: return {"WSPAWN", ""};
      case WctlType::SPLIT:  return {wctlArgs.is_cond_neg ? "SPLIT.N":"SPLIT", ""};
      case WctlType::JOIN:   return {"JOIN", ""};
      case WctlType::BAR: {
        if (wctlArgs.is_sync_bar) {
          return {"BAR", ""};
        } else {
          if (wctlArgs.is_bar_arrive) return {"BAR.ARRIVE", ""};
          else return {"BAR.WAIT", ""};
        }
      }
      case WctlType::PRED:   return {wctlArgs.is_cond_neg ? "PRED.N":"PRED", ""};
      case WctlType::WSYNC:  return {"WSYNC", ""};
      default:
        std::abort();
      }
    }
#ifdef EXT_DXA_ENABLE
    ,[&](DxaType /*dxa_type*/)-> op_string_t {
      return {"DXA.ISSUE", ""};
    }
#endif
  #ifdef EXT_TCU_ENABLE
    ,[&](TcuType tcu_type)-> op_string_t {
      auto tpuArgs = std::get<IntrTcuArgs>(instrArgs);
      return TcuUnit::op_string(tcu_type, tpuArgs);
    }
  #endif // EXT_TCU_ENABLE
  #ifdef EXT_TEX_ENABLE
    ,[&](TexType /*tex_type*/)-> op_string_t {
      return {"TEX.SAMPLE", ""};
    }
  #endif
  #ifdef EXT_OM_ENABLE
    ,[&](OmType /*om_type*/)-> op_string_t {
      return {"OM.WRITE", ""};
    }
  #endif
  #ifdef EXT_RASTER_ENABLE
    ,[&](RasterType /*raster_type*/)-> op_string_t {
      return {"RASTER.POP", ""};
    }
  #endif
 );
 return {"", ""};
}

namespace vortex {
std::ostream &operator<<(std::ostream &os, const Instr &instr) {
  auto sintr = ::op_string(instr);
  int sep = 0;
  os << sintr.op;
  auto rd = instr.get_dest_reg();
  if (rd.type != RegType::None) {
    if (sep++ != 0) { os << ", "; } else { os << " "; }
    os << rd;
  }
  for (uint32_t i = 0; i < Instr::MAX_REG_SOURCES; ++i) {
    auto rs = instr.get_src_reg(i);
    if (rs.type != RegType::None) {
      if (sep++ != 0) { os << ", "; } else { os << " "; }
      os << rs;
    }
  }
  if (sintr.arg != "") {
    if (sep++ != 0) { os << ", "; } else { os << " "; }
    os << sintr.arg;
  }
  if (instr.fu_lock_) {
    os << " [fu_lock";
    if (instr.fu_unlock_) {
      os << ",fu_unlock";
    }
    os << "]";
  }
  return os;
}
}

Decoder::Decoder(const SimContext& ctx, const char* name, PoolAllocator<Instr, 64>& instr_pool)
  : SimObject<Decoder>(ctx, name)
  , instr_pool_(instr_pool)
{}

Decoder::~Decoder() {}

Instr::Ptr Decoder::decode(uint32_t code, uint64_t uuid) {
  auto op = Opcode((code >> shift_opcode) & mask_opcode);
  auto funct2 = (code >> shift_funct2) & mask_funct2;
  auto funct3 = (code >> shift_funct3) & mask_funct3;
  auto funct5 = (code >> shift_funct5) & mask_funct5;
  auto funct6 = (code >> shift_funct6) & mask_funct6;
  auto funct7 = (code >> shift_funct7) & mask_funct7;
  __unused(funct6);

  auto rd  = (code >> shift_rd)  & mask_reg;
  auto rs1 = (code >> shift_rs1) & mask_reg;
  auto rs2 = (code >> shift_rs2) & mask_reg;
  auto rs3 = (code >> shift_rs3) & mask_reg;

  auto instr = std::allocate_shared<Instr>(instr_pool_, uuid, FUType::ALU);

  switch (op) {
  case Opcode::LUI:
  case Opcode::AUIPC: { // RV32I: LUI / AUIPC
    auto imm20 = (code >> shift_funct3) << shift_funct3;
    instr->set_op_type((op == Opcode::LUI) ? AluType::LUI : AluType::AUIPC);
    instr->set_args(IntrAluArgs{1, 0, imm20});
    instr->set_dest_reg(rd, RegType::Integer);
  } break;
#ifdef XLEN_64
  case Opcode::R_W:
  case Opcode::I_W:
#endif
  case Opcode::R:
  case Opcode::I: {
    bool is_w = (op == Opcode::R_W) || (op == Opcode::I_W);
    bool is_imm = (op == Opcode::I) || (op == Opcode::I_W);
    if (op == Opcode::R && funct7 == 0x7) {
      uint32_t imm;
      if (funct3 == 0x5) { // CZERO.EQZ
        imm = 0;
      } else
      if (funct3 == 0x7) { // CZERO.NEZ
        imm = 1;
      } else {
        std::abort();
      }
      instr->set_op_type(AluType::CZERO);
      instr->set_args(IntrAluArgs{0, 0, imm});
    } else
    if ((op == Opcode::R || op == Opcode::R_W) && (funct7 & 0x1)) {
      switch (funct3) {
      case 0: instr->set_op_type(MdvType::MUL); break;
      case 1: instr->set_op_type(MdvType::MULH); break;
      case 2: instr->set_op_type(MdvType::MULHSU); break;
      case 3: instr->set_op_type(MdvType::MULHU); break;
      case 4: instr->set_op_type(MdvType::DIV); break;
      case 5: instr->set_op_type(MdvType::DIVU); break;
      case 6: instr->set_op_type(MdvType::REM); break;
      case 7: instr->set_op_type(MdvType::REMU); break;
      default:
        std::abort();
      }
      instr->set_args(IntrMdvArgs{is_w});
    } else {
      uint32_t imm = 0;
      if (funct3 == 0x1 || funct3 == 0x5) {
        // Shift instructions
        imm = rs2; // uint5
      #ifdef XLEN_64
        imm |= ((funct7 & 0x1) << 5);
      #endif
      } else {
        auto imm12 = code >> shift_rs2;
        imm = sext(imm12, width_i_imm);
      }
      switch (funct3) {
      case 0: instr->set_op_type((!is_imm && funct7 == 0x20) ? AluType::SUB : AluType::ADD); break;
      case 1: instr->set_op_type(AluType::SLL); break;
      case 2: instr->set_op_type(AluType::SLT); break;
      case 3: instr->set_op_type(AluType::SLTU); break;
      case 4: instr->set_op_type(AluType::XOR); break;
      case 5: instr->set_op_type((funct7 == 0x20) ? AluType::SRA : AluType::SRL); break;
      case 6: instr->set_op_type(AluType::OR); break;
      case 7: instr->set_op_type(AluType::AND); break;
      default:
        std::abort();
      }
      instr->set_args(IntrAluArgs{is_imm, is_w, imm});
    }
    instr->set_dest_reg(rd, RegType::Integer);
    instr->set_src_reg(0, rs1, RegType::Integer);
    if (!is_imm) {
      instr->set_src_reg(1, rs2, RegType::Integer);
    }
  } break;
  case Opcode::B: {
    auto bit_11   = rd & 0x1;
    auto bits_4_1 = rd >> 1;
    auto bit_10_5 = funct7 & 0x3f;
    auto bit_12   = funct7 >> 6;
    auto imm12 = (bits_4_1 << 1) | (bit_10_5 << 5) | (bit_11 << 11) | (bit_12 << 12);
    auto addr = sext(imm12, width_i_imm+1);
    instr->set_op_type(BrType::BR);
    instr->set_args(IntrBrArgs{funct3, addr});
    instr->set_src_reg(0, rs1, RegType::Integer);
    instr->set_src_reg(1, rs2, RegType::Integer);
    instr->set_wstall(true);
  } break;
  case Opcode::JAL: {
    auto unordered  = code >> shift_funct3;
    auto bits_19_12 = unordered & 0xff;
    auto bit_11     = (unordered >> 8) & 0x1;
    auto bits_10_1  = (unordered >> 9) & 0x3ff;
    auto bit_20     = (unordered >> 19) & 0x1;
    auto imm20 = (bits_10_1 << 1) | (bit_11 << 11) | (bits_19_12 << 12) | (bit_20 << 20);
    auto addr = sext(imm20, width_j_imm+1);
    instr->set_op_type(BrType::JAL);
    instr->set_args(IntrBrArgs{0, addr});
    instr->set_dest_reg(rd, RegType::Integer);
    instr->set_wstall(true);
  } break;
  case Opcode::JALR: {
    auto imm12 = code >> shift_rs2;
    auto addr = sext(imm12, width_i_imm);
    instr->set_op_type(BrType::JALR);
    instr->set_args(IntrBrArgs{0, addr});
    instr->set_dest_reg(rd, RegType::Integer);
    instr->set_src_reg(0, rs1, RegType::Integer);
    instr->set_wstall(true);
  } break;
  case Opcode::L:
  case Opcode::FL:
  case Opcode::S:
  case Opcode::FS: {
    instr->set_fu_type(FUType::LSU);
    bool is_float = (op == Opcode::FL || op == Opcode::FS);
    bool is_load = (op == Opcode::L || op == Opcode::FL);
    {
      instr->set_src_reg(0, rs1, RegType::Integer);
      uint32_t imm12 = 0;
      if (is_load) {
        imm12 = code >> shift_rs2;
        instr->set_dest_reg(rd, is_float ? RegType::Float : RegType::Integer);
      } else {
        imm12 = (funct7 << width_reg) | rd;
        instr->set_src_reg(1, rs2, is_float ? RegType::Float : RegType::Integer);
      }
      auto offset = sext(imm12, width_i_imm);
      instr->set_op_type(is_load ? LsuType::LOAD : LsuType::STORE);
      instr->set_args(IntrLsuArgs{funct3, /*stride*/ 0, (int32_t)offset});
    }
  } break;
  case Opcode::FENCE: {
    instr->set_fu_type(FUType::LSU);
    instr->set_op_type(LsuType::FENCE);
    instr->set_args(IntrLsuArgs{0, 0, 0});
  } break;
  case Opcode::AMO: {
    instr->set_fu_type(FUType::LSU);
    uint32_t aq = (code >> shift_aq) & mask_aq;
    uint32_t rl = (code >> shift_rl) & mask_rl;
    switch (funct5) {
    case 0x00: instr->set_op_type(AmoType::AMOADD); break;
    case 0x01: instr->set_op_type(AmoType::AMOSWAP); break;
    case 0x02: instr->set_op_type(AmoType::LR); break;
    case 0x03: instr->set_op_type(AmoType::SC); break;
    case 0x04: instr->set_op_type(AmoType::AMOXOR); break;
    case 0x08: instr->set_op_type(AmoType::AMOOR); break;
    case 0x0c: instr->set_op_type(AmoType::AMOAND); break;
    case 0x10: instr->set_op_type(AmoType::AMOMIN); break;
    case 0x14: instr->set_op_type(AmoType::AMOMAX); break;
    case 0x18: instr->set_op_type(AmoType::AMOMINU); break;
    case 0x1c: instr->set_op_type(AmoType::AMOMAXU); break;
    default:
      std::abort();
    }
    instr->set_args(IntrAmoArgs{funct3, aq, rl});
    instr->set_dest_reg(rd, RegType::Integer);
    instr->set_src_reg(0, rs1, RegType::Integer);
    instr->set_src_reg(1, rs2, RegType::Integer);
  } break;
  case Opcode::SYS: {
    if (funct3 != 0) { // CSRRW/CSRRS/CSRRC — dispatched to the SFU's CSR sub-unit
      instr->set_fu_type(FUType::SFU);
      instr->set_dest_reg(rd, RegType::Integer);
      switch (funct3) {
      case 1: case 5: instr->set_op_type(CsrType::CSRRW); break;
      case 2: case 6: instr->set_op_type(CsrType::CSRRS); break;
      case 3: case 7: instr->set_op_type(CsrType::CSRRC); break;
      default:
        std::abort();
      }
      auto imm12 = code >> shift_rs2;
      if (funct3 < 5) {
        instr->set_src_reg(0, rs1, RegType::Integer);
        instr->set_args(IntrCsrArgs{0, 0, imm12});
      } else { // zimm
        instr->set_args(IntrCsrArgs{1, rs1, imm12});
      }
    } else { // ECALL/EBREACK/URET/SRET/MRET
      auto imm12 = code >> shift_rs2;
      instr->set_op_type(BrType::SYS);
      instr->set_args(IntrBrArgs{0, imm12});
      instr->set_wstall(true);
    }
  } break;
  case Opcode::FCI: {
    instr->set_fu_type(FUType::FPU);
    instr->set_args(IntrFpuArgs{funct3, rs2, (funct7 & 0x1)});
    switch (funct7) {
    case 0x00: // RV32F: FADD.S
    case 0x01: // RV32D: FADD.D
      instr->set_op_type(FpuType::FADD);
      instr->set_dest_reg(rd, RegType::Float);
      instr->set_src_reg(0, rs1, RegType::Float);
      instr->set_src_reg(1, rs2, RegType::Float);
      break;
    case 0x04: // RV32F: FSUB.S
    case 0x05: // RV32D: FSUB.D
      instr->set_op_type(FpuType::FSUB);
      instr->set_dest_reg(rd, RegType::Float);
      instr->set_src_reg(0, rs1, RegType::Float);
      instr->set_src_reg(1, rs2, RegType::Float);
      break;
    case 0x08: // RV32F: FMUL.S
    case 0x09: // RV32D: FMUL.D
      instr->set_op_type(FpuType::FMUL);
      instr->set_dest_reg(rd, RegType::Float);
      instr->set_src_reg(0, rs1, RegType::Float);
      instr->set_src_reg(1, rs2, RegType::Float);
      break;
    case 0x10: // RV32F: FSGNJ.S, FSGNJN.S, FSGNJX.S
    case 0x11: // RV32D: FSGNJ.D, FSGNJN.D, FSGNJX.D
      instr->set_op_type(FpuType::FSGNJ);
      instr->set_dest_reg(rd, RegType::Float);
      instr->set_src_reg(0, rs1, RegType::Float);
      instr->set_src_reg(1, rs2, RegType::Float);
      break;
    case 0x14: // RV32F: FMIN.S, FMAX.S
    case 0x15: // RV32D: FMIN.D, FMAX.D
      instr->set_op_type(FpuType::FMINMAX);
      instr->set_dest_reg(rd, RegType::Float);
      instr->set_src_reg(0, rs1, RegType::Float);
      instr->set_src_reg(1, rs2, RegType::Float);
      break;
    case 0x0c: // RV32F: FDIV.S
    case 0x0d: // RV32D: FDIV.D
      instr->set_op_type(FpuType::FDIV);
      instr->set_dest_reg(rd, RegType::Float);
      instr->set_src_reg(0, rs1, RegType::Float);
      instr->set_src_reg(1, rs2, RegType::Float);
      break;
    case 0x20: // FCVT.S.D
    case 0x21: // FCVT.D.S
      instr->set_op_type(FpuType::F2F);
      instr->set_dest_reg(rd, RegType::Float);
      instr->set_src_reg(0, rs1, RegType::Float);
      break;
    case 0x2c: // FSQRT.S
    case 0x2d: // FSQRT.D
      instr->set_op_type(FpuType::FSQRT);
      instr->set_dest_reg(rd, RegType::Float);
      instr->set_src_reg(0, rs1, RegType::Float);
      break;
    case 0x50: // FLE.S, FLT.S, FEQ.S
    case 0x51: // FLE.D, FLT.D, FEQ.D
      instr->set_op_type(FpuType::FCMP);
      instr->set_dest_reg(rd, RegType::Integer);
      instr->set_src_reg(0, rs1, RegType::Float);
      instr->set_src_reg(1, rs2, RegType::Float);
      break;
    case 0x60: // FCVT.W.D, FCVT.WU.D, FCVT.L.D, FCVT.LU.D
    case 0x61: // FCVT.W.S, FCVT.WU.S, FCVT.L.S, FCVT.LU.S
      instr->set_op_type(FpuType::F2I);
      instr->set_dest_reg(rd, RegType::Integer);
      instr->set_src_reg(0, rs1, RegType::Float);
      instr->set_src_reg(1, rs2, RegType::None);
      break;
    case 0x68: // FCVT.S.W, FCVT.S.WU, FCVT.S.L, FCVT.S.LU
    case 0x69: // FCVT.D.W, FCVT.D.WU, FCVT.D.L, FCVT.D.LU
      instr->set_op_type(FpuType::I2F);
      instr->set_dest_reg(rd, RegType::Float);
      instr->set_src_reg(0, rs1, RegType::Integer);
      instr->set_src_reg(1, rs2, RegType::None);
      break;
    case 0x70: // FCLASS.S, FMV.X.S
    case 0x71: // FCLASS.D, FMV.X.D
      instr->set_op_type((funct3 != 0) ? FpuType::FCLASS : FpuType::FMVXW);
      instr->set_dest_reg(rd, RegType::Integer);
      instr->set_src_reg(0, rs1, RegType::Float);
      break;
    case 0x78: // FMV.S.X
    case 0x79: // FMV.D.X
      instr->set_op_type(FpuType::FMVWX);
      instr->set_dest_reg(rd, RegType::Float);
      instr->set_src_reg(0, rs1, RegType::Integer);
      break;
    default:
      std::abort();
    }
  } break;
  case Opcode::FMADD:
  case Opcode::FMSUB:
  case Opcode::FNMADD:
  case Opcode::FNMSUB: {
    instr->set_fu_type(FUType::FPU);
    instr->set_op_type((op == Opcode::FMADD) ? FpuType::FMADD :
                     (op == Opcode::FMSUB) ? FpuType::FMSUB :
                     (op == Opcode::FNMADD) ? FpuType::FNMADD : FpuType::FNMSUB);
    instr->set_args(IntrFpuArgs{funct3, funct2, (funct7 & 0x1)});
    instr->set_dest_reg(rd, RegType::Float);
    instr->set_src_reg(0, rs1, RegType::Float);
    instr->set_src_reg(1, rs2, RegType::Float);
    instr->set_src_reg(2, rs3, RegType::Float);
  } break;
  case Opcode::EXT1: {
    switch (funct7) {
    case 0: {
      instr->set_fu_type(FUType::SFU);
      IntrWctlArgs wctlArgs{};
      switch (funct3) {
      case 0: // TMC
        instr->set_op_type(WctlType::TMC);
        instr->set_src_reg(0, rs1, RegType::Integer);
        instr->set_wstall(true);
        break;
      case 1: // WSPAWN
        instr->set_op_type(WctlType::WSPAWN);
        instr->set_src_reg(0, rs1, RegType::Integer);
        instr->set_src_reg(1, rs2, RegType::Integer);
        instr->set_wstall(true);
        break;
      case 2: // SPLIT
        instr->set_op_type(WctlType::SPLIT);
        instr->set_dest_reg(rd, RegType::Integer);
        instr->set_src_reg(0, rs1, RegType::Integer);
        wctlArgs.is_cond_neg = (rs2 != 0);
        instr->set_wstall(true);
        break;
      case 3: // JOIN
        instr->set_op_type(WctlType::JOIN);
        instr->set_src_reg(0, rs1, RegType::Integer);
        instr->set_wstall(true);
        break;
      case 4: // BAR (sync)
        instr->set_op_type(WctlType::BAR);
        instr->set_src_reg(0, rs1, RegType::Integer);
        instr->set_src_reg(1, rs2, RegType::Integer);
        wctlArgs.is_sync_bar = 1;
        wctlArgs.is_bar_arrive = 0;
        instr->set_wstall(true);
        break;
      case 5: // PRED
        instr->set_op_type(WctlType::PRED);
        instr->set_src_reg(0, rs1, RegType::Integer);
        instr->set_src_reg(1, rs2, RegType::Integer);
        wctlArgs.is_cond_neg = (rd != 0);
        instr->set_wstall(true);
        break;
      case 6: // BAR ARRIVE / WAIT
        instr->set_op_type(WctlType::BAR);
        instr->set_dest_reg(rd, RegType::Integer);
        instr->set_src_reg(0, rs1, RegType::Integer);
        instr->set_src_reg(1, rs2, RegType::Integer);
        wctlArgs.is_sync_bar = 0;
        wctlArgs.is_bar_arrive = (rd != 0);
        instr->set_wstall(rd == 0); // stall on wait, not on arrive
        break;
      case 7: // WSYNC
        instr->set_op_type(WctlType::WSYNC);
        instr->set_wstall(true);
        break;
      default:
        std::abort();
      }
      instr->set_args(wctlArgs);
    } break;
    case 1: { // VOTE
      instr->set_dest_reg(rd, RegType::Integer);
      instr->set_src_reg(0, rs1, RegType::Integer);
      switch (funct3) {
      case 0: instr->set_op_type(VoteType::ALL); break;
      case 1: instr->set_op_type(VoteType::ANY); break;
      case 2: instr->set_op_type(VoteType::UNI); break;
      case 3: instr->set_op_type(VoteType::BAL); break;
      case 4:
        instr->set_op_type(ShflType::UP);
        instr->set_src_reg(1, rs2, RegType::Integer);
        break;
      case 5:
        instr->set_op_type(ShflType::DOWN);
        instr->set_src_reg(1, rs2, RegType::Integer);
        break;
      case 6:
        instr->set_op_type(ShflType::BFLY);
        instr->set_src_reg(1, rs2, RegType::Integer);
        break;
      case 7:
        instr->set_op_type(ShflType::IDX);
        instr->set_src_reg(1, rs2, RegType::Integer);
        break;
      default:
        std::abort();
      }
    } break;
#ifdef EXT_DXA_ENABLE
    case 3: { // DXA issue
      instr->set_fu_type(FUType::SFU);
      IntrDxaArgs dxaArgs{};
      instr->set_args(dxaArgs);
      instr->set_op_type(DxaType::ISSUE);
      instr->set_src_reg(0, rs1, RegType::Integer);
      instr->set_src_reg(1, rs2, RegType::Integer);
    } break;
#endif
  #ifdef EXT_TCU_ENABLE
    case 2: {
      instr->set_fu_type(FUType::TCU);
      switch (funct3) {
      case 0: { // WMMA_SYNC / WMMA_SP_SYNC — single macro Instr, sequencer expands to micro-ops
        uint32_t fmt_d = rd, fmt_s = rs1;
        bool is_sparse = (rs2 & 1) != 0;
        instr->set_op_type(TcuType::WMMA);
        instr->set_args(IntrTcuArgs{is_sparse, 0, 0, fmt_s, fmt_d, 0, 0, 0, 0});
        instr->set_macro_op();
        instr->set_wstall(true);
      } break;
    #ifdef TCU_WGMMA_ENABLE
      case 1: { // WGMMA_SYNC — single macro Instr, sequencer expands to micro-ops
        uint32_t fmt_d = rd, fmt_s = rs1;
        bool is_sparse = (rs2 & 1) != 0;
        uint32_t cd_nregs = (rs2 >> 1) & 0x3;
        bool is_a_smem = (rs2 >> 3) & 1;
        instr->set_op_type(TcuType::WGMMA);
        instr->set_args(IntrTcuArgs{is_sparse, is_a_smem ? 1u : 0u, cd_nregs, fmt_s, fmt_d, 0, 0, 0, 0});
        instr->set_macro_op();
        instr->set_wstall(true);
      } break;
    #endif // TCU_WGMMA_ENABLE
      default:
        std::abort();
      }
    } break;
  #endif
    case 4: { // Load/Store Packing extensions — macro-ops, sequencer expands to N single-elem uops
      instr->set_fu_type(FUType::LSU);
      instr->set_op_type(LsuType::LOAD);
      instr->set_dest_reg(rd, RegType::Float);
      instr->set_src_reg(0, rs1, RegType::Integer);
      instr->set_src_reg(1, rs2, RegType::Integer);
      switch (funct3) {
      case 1: { // vx_packlb_f: 4 strided bytes (LB-width) → packed FP
        instr->set_args(IntrLsuArgs{/*width=LB*/ 0, /*stride*/ 0, /*offset*/ 0});
      } break;
      case 2: { // vx_packlh_f: 2 strided halfwords (LH-width) → packed FP
        instr->set_args(IntrLsuArgs{/*width=LH*/ 1, /*stride*/ 0, /*offset*/ 0});
      } break;
      default:
        std::abort();
      }
      instr->set_macro_op();
      instr->set_wstall(true);   // pause fetch while sequencer expands the N uops
    } break;
    default:
      std::abort();
    }
  } break;
  case Opcode::EXT2: {
    switch (funct3) {
    case 0: { // WGATHER
      instr->set_op_type(WgatherType::WGATHER);
      instr->set_dest_reg(rd, RegType::Integer);
      instr->set_src_reg(0, rs1, RegType::Integer);
      instr->set_src_reg(1, rs2, RegType::Integer);
      instr->set_src_reg(2, rs3, RegType::Integer);
      IntrWgatherArgs wgArgs{};
      wgArgs.src_lane = funct2;
      instr->set_args(wgArgs);
    } break;
#ifdef EXT_TEX_ENABLE
    case 1: { // vx_tex: R4-type, funct2=stage, rd=texel, rs1=u, rs2=v, rs3=lod
      instr->set_fu_type(FUType::SFU);
      instr->set_op_type(TexType::SAMPLE);
      instr->set_dest_reg(rd, RegType::Integer);
      instr->set_src_reg(0, rs1, RegType::Integer);
      instr->set_src_reg(1, rs2, RegType::Integer);
      instr->set_src_reg(2, rs3, RegType::Integer);
      IntrTexArgs texArgs{};
      texArgs.stage = funct2;
      instr->set_args(texArgs);
    } break;
#endif
#ifdef EXT_OM_ENABLE
    case 2: { // vx_om: R4-type, rs1=pos_face, rs2=color, rs3=depth
      instr->set_fu_type(FUType::SFU);
      instr->set_op_type(OmType::WRITE);
      instr->set_src_reg(0, rs1, RegType::Integer);
      instr->set_src_reg(1, rs2, RegType::Integer);
      instr->set_src_reg(2, rs3, RegType::Integer);
      IntrOmArgs omArgs{};
      instr->set_args(omArgs);
    } break;
#endif
#ifdef EXT_RASTER_ENABLE
    case 3: { // vx_rast: R-type, rd=quad descriptor
      instr->set_fu_type(FUType::SFU);
      instr->set_op_type(RasterType::POP);
      instr->set_dest_reg(rd, RegType::Integer);
      IntrRasterArgs rastArgs{};
      instr->set_args(rastArgs);
    } break;
#endif
    default:
      std::abort();
    }
  } break;
  default:
    std::abort();
  }

  return instr;
}
