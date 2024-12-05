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
#include <stdlib.h>
#include <string.h>
#include <iomanip>
#include <vector>
#include <unordered_map>
#include <util.h>
#include "debug.h"
#include "types.h"
#include "emulator.h"
#include "arch.h"
#include "instr.h"

using namespace vortex;

static const std::unordered_map<Opcode, InstType> sc_instTable = {
  {Opcode::R,       InstType::R},
  {Opcode::L,       InstType::I},
  {Opcode::I,       InstType::I},
  {Opcode::S,       InstType::S},
  {Opcode::B,       InstType::B},
  {Opcode::LUI,     InstType::U},
  {Opcode::AUIPC,   InstType::U},
  {Opcode::JAL,     InstType::J},
  {Opcode::JALR,    InstType::I},
  {Opcode::SYS,     InstType::I},
  {Opcode::FENCE,   InstType::I},
  {Opcode::AMO,     InstType::R},
  {Opcode::FL,      InstType::I},
  {Opcode::FS,      InstType::S},
  {Opcode::FCI,     InstType::R},
  {Opcode::FMADD,   InstType::R4},
  {Opcode::FMSUB,   InstType::R4},
  {Opcode::FMNMADD, InstType::R4},
  {Opcode::FMNMSUB, InstType::R4},
  {Opcode::VSET,    InstType::V},
  {Opcode::EXT1,    InstType::R},
  {Opcode::EXT2,    InstType::R4},
  {Opcode::R_W,     InstType::R},
  {Opcode::I_W,     InstType::I},
  {Opcode::TCU,     InstType::I},
};

static const char* op_string(const Instr &instr) {
  auto opcode = instr.getOpcode();
  auto func2  = instr.getFunc2();
  auto func3  = instr.getFunc3();
  auto func7  = instr.getFunc7();
  auto rd     = instr.getRDest();
  auto rs1    = instr.getRSrc(1);
  auto imm    = instr.getImm();

  switch (opcode) {
  case Opcode::LUI:   return "LUI";
  case Opcode::AUIPC: return "AUIPC";
  case Opcode::R:
    if (func7 == 0x7) {
      if (func3 == 0x5) {
        return "CZERO.EQZ";
      } else
      if (func3 == 0x7) {
        return "CZERO.NEZ";
      } else {
        std::abort();
      }
    } else
    if (func7 & 0x1) {
      switch (func3) {
      case 0: return "MUL";
      case 1: return "MULH";
      case 2: return "MULHSU";
      case 3: return "MULHU";
      case 4: return "DIV";
      case 5: return "DIVU";
      case 6: return "REM";
      case 7: return "REMU";
      default:
        std::abort();
      }
    } else {
      switch (func3) {
      case 0: return (func7 & 0x20) ? "SUB" : "ADD";
      case 1: return "SLL";
      case 2: return "SLT";
      case 3: return "SLTU";
      case 4: return "XOR";
      case 5: return (func7 & 0x20) ? "SRA" : "SRL";
      case 6: return "OR";
      case 7: return "AND";
      default:
        std::abort();
      }
    }
  case Opcode::I:
    switch (func3) {
    case 0: return "ADDI";
    case 1: return "SLLI";
    case 2: return "SLTI";
    case 3: return "SLTIU";
    case 4: return "XORI";
    case 5: return (func7 & 0x20) ? "SRAI" : "SRLI";
    case 6: return "ORI";
    case 7: return "ANDI";
    default:
      std::abort();
    }
  case Opcode::B:
    switch (func3) {
    case 0: return "BEQ";
    case 1: return "BNE";
    case 4: return "BLT";
    case 5: return "BGE";
    case 6: return "BLTU";
    case 7: return "BGEU";
    default:
      std::abort();
    }
  case Opcode::JAL:   return "JAL";
  case Opcode::JALR:  return "JALR";
  case Opcode::L:
    switch (func3) {
    case 0: return "LB";
    case 1: return "LH";
    case 2: return "LW";
    case 3: return "LD";
    case 4: return "LBU";
    case 5: return "LHU";
    case 6: return "LWU";
    default:
      std::abort();
    }
  case Opcode::S:
    switch (func3) {
    case 0: return "SB";
    case 1: return "SH";
    case 2: return "SW";
    case 3: return "SD";
    default:
      std::abort();
    }
  case Opcode::R_W:
    if (func7 & 0x1){
      switch (func3) {
      case 0: return "MULW";
      case 4: return "DIVW";
      case 5: return "DIVUW";
      case 6: return "REMW";
      case 7: return "REMUW";
      default:
        std::abort();
      }
    } else {
      switch (func3) {
      case 0: return (func7 & 0x20) ? "SUBW" : "ADDW";
      case 1: return "SLLW";
      case 5: return (func7 & 0x20) ? "SRAW" : "SRLW";
      default:
        std::abort();
      }
    }
  case Opcode::I_W:
    switch (func3) {
    case 0: return "ADDIW";
    case 1: return "SLLIW";
    case 5: return (func7 & 0x20) ? "SRAIW" : "SRLIW";
    default:
      std::abort();
    }
  case Opcode::SYS:
    switch (func3) {
    case 0:
      switch (imm) {
      case 0x000: return "ECALL";
      case 0x001: return "EBREAK";
      case 0x002: return "URET";
      case 0x102: return "SRET";
      case 0x302: return "MRET";
      default:
        std::abort();
      }
    case 1: return "CSRRW";
    case 2: return "CSRRS";
    case 3: return "CSRRC";
    case 5: return "CSRRWI";
    case 6: return "CSRRSI";
    case 7: return "CSRRCI";
    default:
      std::abort();
    }
  case Opcode::FENCE: return "FENCE";
  case Opcode::FL:
    switch (func3) {
    case 0x2: return "FLW";
    case 0x3: return "FLD";
    case 0x0: return "VL8";
    case 0x5: return "VL16";
    case 0x6: return "VL32";
    case 0x7: return "VL64";
    default:
      std::cout << "Could not decode float/vector load with func3: " << func3 << std::endl;
      std::abort();
    }
  case Opcode::FS:
    switch (func3) {
    case 0x1: return "VS";
    case 0x2: return "FSW";
    case 0x3: return "FSD";
    case 0x0: return "VS8";
    case 0x5: return "VS16";
    case 0x6: return "VS32";
    case 0x7: return "VS64";
    default:
      std::cout << "Could not decode float/vector store with func3: " << func3 << std::endl;
      std::abort();
    }
  case Opcode::AMO: {
    auto amo_type = func7 >> 2;
    switch (func3) {
      case 0x2:
        switch (amo_type) {
        case 0x00: return "AMOADD.W";
        case 0x01: return "AMOSWAP.W";
        case 0x02: return "LR.W";
        case 0x03: return "SC.W";
        case 0x04: return "AMOXOR.W";
        case 0x08: return "AMOOR.W";
        case 0x0c: return "AMOAND.W";
        case 0x10: return "AMOMIN.W";
        case 0x14: return "AMOMAX.W";
        case 0x18: return "AMOMINU.W";
        case 0x1c: return "AMOMAXU.W";
        default:
          std::abort();
        }
      case 0x3:
        switch (amo_type) {
        case 0x00: return "AMOADD.D";
        case 0x01: return "AMOSWAP.D";
        case 0x02: return "LR.D";
        case 0x03: return "SC.D";
        case 0x04: return "AMOXOR.D";
        case 0x08: return "AMOOR.D";
        case 0x0c: return "AMOAND.D";
        case 0x10: return "AMOMIN.D";
        case 0x14: return "AMOMAX.D";
        case 0x18: return "AMOMINU.D";
        case 0x1c: return "AMOMAXU.D";
        default:
          std::abort();
        }
      default:
        std::abort();
    }
  }
  case Opcode::FCI:
    switch (func7) {
    case 0x00: return "FADD.S";
    case 0x01: return "FADD.D";
    case 0x04: return "FSUB.S";
    case 0x05: return "FSUB.D";
    case 0x08: return "FMUL.S";
    case 0x09: return "FMUL.D";
    case 0x0c: return "FDIV.S";
    case 0x0d: return "FDIV.D";
    case 0x2c: return "FSQRT.S";
    case 0x2d: return "FSQRT.D";
    case 0x10:
      switch (func3) {
      case 0: return "FSGNJ.S";
      case 1: return "FSGNJN.S";
      case 2: return "FSGNJX.S";
      default:
        std::abort();
      }
    case 0x11:
      switch (func3) {
      case 0: return "FSGNJ.D";
      case 1: return "FSGNJN.D";
      case 2: return "FSGNJX.D";
      default:
        std::abort();
      }
    case 0x14:
      switch (func3) {
      case 0: return "FMIN.S";
      case 1: return "FMAX.S";
      default:
        std::abort();
      }
    case 0x15:
      switch (func3) {
      case 0: return "FMIN.D";
      case 1: return "FMAX.D";
      default:
        std::abort();
      }
    case 0x20: return "FCVT.S.D";
    case 0x21: return "FCVT.D.S";
    case 0x50:
      switch (func3) {
      case 0: return "FLE.S";
      case 1: return "FLT.S";
      case 2: return "FEQ.S";
      default:
        std::abort();
      }
    case 0x51:
      switch (func3) {
      case 0: return "FLE.D";
      case 1: return "FLT.D";
      case 2: return "FEQ.D";
      default:
        std::abort();
      }
    case 0x60:
      switch (rs1) {
      case 0: return "FCVT.W.S";
      case 1: return "FCVT.WU.S";
      case 2: return "FCVT.L.S";
      case 3: return "FCVT.LU.S";
      default:
        std::abort();
      }
    case 0x61:
      switch (rs1) {
      case 0: return "FCVT.W.D";
      case 1: return "FCVT.WU.D";
      case 2: return "FCVT.L.D";
      case 3: return "FCVT.LU.D";
      default:
        std::abort();
      }
    case 0x68:
      switch (rs1) {
      case 0: return "FCVT.S.W";
      case 1: return "FCVT.S.WU";
      case 2: return "FCVT.S.L";
      case 3: return "FCVT.S.LU";
      default:
        std::abort();
      }
    case 0x69:
      switch (rs1) {
      case 0: return "FCVT.D.W";
      case 1: return "FCVT.D.WU";
      case 2: return "FCVT.D.L";
      case 3: return "FCVT.D.LU";
      default:
        std::abort();
      }
    case 0x70: return func3 ? "FCLASS.S" : "FMV.X.S";
    case 0x71: return func3 ? "FCLASS.D" : "FMV.X.D";
    case 0x78: return "FMV.S.X";
    case 0x79: return "FMV.D.X";
    default:
      std::abort();
    }
  case Opcode::FMADD:   return func2 ? "FMADD.D" : "FMADD.S";
  case Opcode::FMSUB:   return func2 ? "FMSUB.D" : "FMSUB.S";
  case Opcode::FMNMADD: return func2 ? "FNMADD.D" : "FNMADD.S";
  case Opcode::FMNMSUB: return func2 ? "FNMSUB.D" : "FNMSUB.S";
  case Opcode::VSET:    return "VSET";
  case Opcode::EXT1:
    switch (func7) {
    case 0:
      switch (func3) {
      case 0: return "TMC";
      case 1: return "WSPAWN";
      case 2: return rs1 ? "SPLIT.N" : "SPLIT";
      case 3: return "JOIN";
      case 4: return "BAR";
      case 5: return rd ? "PRED.N" : "PRED";
      default:
        std::abort();
      }
    default:
      std::abort();
    }

  case Opcode::TCU:
    switch(func3)
    {
      case 0: return "ML";     // Matrix Load
      case 1: return "MS";     // Matrix Store
      case 2: return "MATMUL"; // Matrix Multiply
      default:
        std::abort();
    }
  default:
    std::abort();
  }
}

inline void print_vec_attr(std::ostream &os, const Instr &instr) {
  uint32_t mask = instr.getVattrMask();
  if (mask & vattr_vlswidth)
    os << ", width:" << instr.getVlsWidth();
  if (mask & vattr_vmop)
    os << ", mop:" << instr.getVmop();
  if (mask & vattr_vumop)
    os << ", umop:" << instr.getVumop();
  if (mask & vattr_vnf)
    os << ", nf:" << instr.getVnf();
  if (mask & vattr_vmask)
    os << ", vmask:" << instr.getVmask();
  if (mask & vattr_vs3)
    os << ", vs3:" << instr.getVs3();
  if (mask & vattr_zimm)
    os << ", zimm:" << ((instr.hasZimm()) ? "true" : "false");
  if (mask & vattr_vlmul)
    os << ", lmul:" << instr.getVlmul();
  if (mask & vattr_vsew)
    os << ", sew:" << instr.getVsew();
  if (mask & vattr_vta)
    os << ", ta:" << instr.getVta();
  if (mask & vattr_vma)
    os << ", ma:" << instr.getVma();
  if (mask & vattr_vediv)
    os << ", ediv:" << instr.getVediv();
}

namespace vortex {
std::ostream &operator<<(std::ostream &os, const Instr &instr) {
  os << op_string(instr);
  int sep = 0;
  if (instr.getRDType() != RegType::None) {
    if (sep++ != 0) { os << ", "; } else { os << " "; }
    os << instr.getRDType() << instr.getRDest();
  }
  for (uint32_t i = 0; i < instr.getNRSrc(); ++i) {
    if (sep++ != 0) { os << ", "; } else { os << " "; }
    if (instr.getRSType(i) != RegType::None) {
      os << instr.getRSType(i) << instr.getRSrc(i);
    } else {
      os << "0x" << std::hex << instr.getRSrc(0) << std::dec;
    }
  }
  if (instr.hasImm()) {
    if (sep++ != 0) { os << ", "; } else { os << " "; }
    os << "0x" << std::hex << instr.getImm() << std::dec;
  }
  if (instr.getOpcode() == Opcode::SYS && instr.getFunc3() >= 5) {
    // CSRs with immediate values
    if (sep++ != 0) { os << ", "; } else { os << " "; }
    os << "0x" << std::hex << instr.getRSrc(0);
  }
  // Log vector-specific attributes
  if (instr.getVattrMask() != 0) {
    print_vec_attr(os, instr);
  }
  return os;
}
}

std::shared_ptr<Instr> Emulator::decode(uint32_t code) const {
  auto instr = std::make_shared<Instr>();
  auto op = Opcode((code >> shift_opcode) & mask_opcode);
  instr->setOpcode(op);

  auto func2 = (code >> shift_func2) & mask_func2;
  auto func3 = (code >> shift_func3) & mask_func3;
  auto func6 = (code >> shift_func6) & mask_func6;
  auto func7 = (code >> shift_func7) & mask_func7;
  __unused(func6);

  auto rd  = (code >> shift_rd)  & mask_reg;
  auto rs1 = (code >> shift_rs1) & mask_reg;
  auto rs2 = (code >> shift_rs2) & mask_reg;
  auto rs3 = (code >> shift_rs3) & mask_reg;

  auto op_it = sc_instTable.find(op);
  if (op_it == sc_instTable.end()) {
    std::cout << "Error: invalid opcode: 0x" << std::hex << static_cast<int>(op) << std::dec << std::endl;
    return nullptr;
  }

  auto iType = op_it->second;
  if (op == Opcode::FL || op == Opcode::FS) {
    if (func3 != 0x2 && func3 != 0x3) {
      iType = InstType::V;
    }
  }

  switch (iType) {
  case InstType::R:
    switch (op) {
    case Opcode::FCI:
      switch (func7) {
      case 0x20: // FCVT.S.D
      case 0x21: // FCVT.D.S
        instr->setDestReg(rd, RegType::Float);
        instr->addSrcReg(rs1, RegType::Float);
        break;
      case 0x2c: // FSQRT.S
      case 0x2d: // FSQRT.D
        instr->setDestReg(rd, RegType::Float);
        instr->addSrcReg(rs1, RegType::Float);
        break;
      case 0x50: // FLE.S, FLT.S, FEQ.S
      case 0x51: // FLE.D, FLT.D, FEQ.D
        instr->setDestReg(rd, RegType::Integer);
        instr->addSrcReg(rs1, RegType::Float);
        instr->addSrcReg(rs2, RegType::Float);
        break;
      case 0x60: // FCVT.W.D, FCVT.WU.D, FCVT.L.D, FCVT.LU.D
      case 0x61: // FCVT.WU.S, FCVT.W.S, FCVT.L.S, FCVT.LU.S
        instr->setDestReg(rd, RegType::Integer);
        instr->addSrcReg(rs1, RegType::Float);
        instr->addSrcReg(rs2, RegType::None);
        break;
      case 0x68: // FCVT.S.W, FCVT.S.WU, FCVT.S.L, FCVT.S.LU
      case 0x69: // FCVT.D.W, FCVT.D.WU, FCVT.D.L, FCVT.D.LU
        instr->setDestReg(rd, RegType::Float);
        instr->addSrcReg(rs1, RegType::Integer);
        instr->addSrcReg(rs2, RegType::None);
        break;
      case 0x70: // FCLASS.S, FMV.X.S
      case 0x71: // FCLASS.D, FMV.X.D
        instr->setDestReg(rd, RegType::Integer);
        instr->addSrcReg(rs1, RegType::Float);
        break;
      case 0x78: // FMV.S.X
      case 0x79: // FMV.D.X
        instr->setDestReg(rd, RegType::Float);
        instr->addSrcReg(rs1, RegType::Integer);
        break;
      default:
        instr->setDestReg(rd, RegType::Float);
        instr->addSrcReg(rs1, RegType::Float);
        instr->addSrcReg(rs2, RegType::Float);
        break;
      }
      break;
    case Opcode::EXT1:
      switch (func7) {
      case 0:
        switch (func3) {
        case 0: // TMC
        case 3: // JOIN
          instr->addSrcReg(rs1, RegType::Integer);
          break;
        case 1: // WSPAWN
        case 4: // BAR
          instr->addSrcReg(rs1, RegType::Integer);
          instr->addSrcReg(rs2, RegType::Integer);
          break;
        case 5: // PRED
          instr->setDestReg(rd, RegType::None);
          instr->addSrcReg(rs1, RegType::Integer);
          instr->addSrcReg(rs2, RegType::Integer);
          break;
        case 2: // SPLIT
          instr->setDestReg(rd, RegType::Integer);
          instr->addSrcReg(rs1, RegType::Integer);
          instr->addSrcReg(rs2, RegType::None);
          break;
        default:
          std::abort();
        }
        break;
      default:
        std::abort();
      }
      break;
    default:
      instr->setDestReg(rd, RegType::Integer);
      instr->addSrcReg(rs1, RegType::Integer);
      instr->addSrcReg(rs2, RegType::Integer);
      break;
    }
    instr->setFunc3(func3);
    instr->setFunc7(func7);
    break;

  case InstType::I: {
    switch (op) {
    case Opcode::TCU: {
      instr->setDestReg(rs1, RegType::Integer);
      instr->addSrcReg(rs1, RegType::Integer);
      instr->setFunc3(func3);
      instr->setFunc7(func7);
      auto imm = code >> shift_rs2;
      instr->setImm(sext(imm, width_i_imm));
    } break;
    case Opcode::I:
    case Opcode::I_W:
    case Opcode::JALR:
      instr->setDestReg(rd, RegType::Integer);
      instr->addSrcReg(rs1, RegType::Integer);
      instr->setFunc3(func3);
      if (func3 == 0x1 || func3 == 0x5) {
        // Shift instructions
        auto shamt = rs2; // uint5
      #if (XLEN == 64)
        if (op == Opcode::I) {
          // uint6
          shamt |= ((func7 & 0x1) << 5);
        }
      #endif
        instr->setImm(shamt);
        instr->setFunc7(func7);
      } else {
        auto imm = code >> shift_rs2;
        instr->setImm(sext(imm, width_i_imm));
      }
      break;
    case Opcode::L:
    case Opcode::FL: {
      instr->setDestReg(rd, (op == Opcode::FL) ? RegType::Float : RegType::Integer);
      instr->addSrcReg(rs1, RegType::Integer);
      instr->setFunc3(func3);
      auto imm = code >> shift_rs2;
      instr->setImm(sext(imm, width_i_imm));
    } break;
    case Opcode::FENCE:
      instr->setFunc3(func3);
      instr->setImm(code >> shift_rs2);
      break;
    case Opcode::SYS:
      if (func3 != 0) {
        // CSR instructions
        instr->setDestReg(rd, RegType::Integer);
        instr->setFunc3(func3);
        if (func3 < 5) {
          instr->addSrcReg(rs1, RegType::Integer);
        } else {
          // zimm
          instr->addSrcReg(rs1, RegType::None);
        }
        instr->setImm(code >> shift_rs2);
      } else {
        // ECALL/EBREACK instructions
        instr->setImm(code >> shift_rs2);
      }
      break;
    default:
      std::abort();
      break;
    }
  } break;
  case InstType::S: {
    instr->addSrcReg(rs1, RegType::Integer);
    instr->addSrcReg(rs2, (op == Opcode::FS) ? RegType::Float : RegType::Integer);
    instr->setFunc3(func3);
    auto imm = (func7 << width_reg) | rd;
    instr->setImm(sext(imm, width_i_imm));
  } break;

  case InstType::B: {
    instr->addSrcReg(rs1, RegType::Integer);
    instr->addSrcReg(rs2, RegType::Integer);
    instr->setFunc3(func3);
    auto bit_11   = rd & 0x1;
    auto bits_4_1 = rd >> 1;
    auto bit_10_5 = func7 & 0x3f;
    auto bit_12   = func7 >> 6;
    auto imm = (bits_4_1 << 1) | (bit_10_5 << 5) | (bit_11 << 11) | (bit_12 << 12);
    instr->setImm(sext(imm, width_i_imm+1));
  } break;

  case InstType::U: {
    instr->setDestReg(rd, RegType::Integer);
    auto imm = (code >> shift_func3) << shift_func3;
    instr->setImm(imm);
  } break;

  case InstType::J: {
    instr->setDestReg(rd, RegType::Integer);
    auto unordered  = code >> shift_func3;
    auto bits_19_12 = unordered & 0xff;
    auto bit_11     = (unordered >> 8) & 0x1;
    auto bits_10_1  = (unordered >> 9) & 0x3ff;
    auto bit_20     = (unordered >> 19) & 0x1;
    auto imm = (bits_10_1 << 1) | (bit_11 << 11) | (bits_19_12 << 12) | (bit_20 << 20);
    instr->setImm(sext(imm, width_j_imm+1));
  } break;

  case InstType::R4: {
    instr->setDestReg(rd, RegType::Float);
    instr->addSrcReg(rs1, RegType::Float);
    instr->addSrcReg(rs2, RegType::Float);
    instr->addSrcReg(rs3, RegType::Float);
    instr->setFunc2(func2);
    instr->setFunc3(func3);
  } break;

#ifdef EXT_V_ENABLE
  case InstType::V:
    switch (op) {
    case Opcode::VSET: {
      instr->setDestReg(rd, RegType::Integer);
      instr->setFunc3(func3);
      switch (func3) {
        case 7: {
          if (code >> (shift_vset - 1) == 0b10) { // vsetvl
            instr->addSrcReg(rs1, RegType::Integer);
            instr->addSrcReg(rs2, RegType::Integer);
          } else {
            auto zimm = (code >> shift_rs2) & mask_v_zimm;
            instr->setZimm(true);
            instr->setVlmul(zimm & mask_v_lmul);
            instr->setVsew((zimm >> shift_v_sew) & mask_v_sew);
            instr->setVta((zimm >> shift_v_ta) & mask_v_ta);
            instr->setVma((zimm >> shift_v_ma) & mask_v_ma);
            if ((code >> shift_vset)) { // vsetivli
              instr->setImm(rs1);
            } else { // vsetvli
              instr->addSrcReg(rs1, RegType::Integer);
            }
          }
        } break;
        case 3: { // Vector - immediate arithmetic instructions
          instr->setDestReg(rd, RegType::Vector);
          instr->addSrcReg(rs2, RegType::Vector);
          instr->setImm(rs1);
          instr->setVmask((code >> shift_func7) & 0x1);
          instr->setFunc6(func6);
        } break;
        default: { // Vector - vector/scalar arithmetic instructions
          if (func3 == 1 && func6 == 16) {
            instr->setDestReg(rd, RegType::Float);
          } else if (func3 == 2 && func6 == 16) {
            instr->setDestReg(rd, RegType::Integer);
          } else {
            instr->setDestReg(rd, RegType::Vector);
          }
          instr->addSrcReg(rs1, RegType::Vector);
          instr->addSrcReg(rs2, RegType::Vector);
          instr->setVmask((code >> shift_func7) & 0x1);
          instr->setFunc6(func6);
        }
      }
    } break;
    case Opcode::FL:
      instr->addSrcReg(rs1, RegType::Integer);
      instr->setVmop((code >> shift_vmop) & 0b11);
      switch (instr->getVmop()) {
        case 0b00:
          instr->setVumop(rs2);
          break;
        case 0b10:
          instr->addSrcReg(rs2, RegType::Integer);
          break;
        case 0b01:
        case 0b11:
          instr->addSrcReg(rs2, RegType::Vector);
          break;
      }
      instr->setVsew(func3 & 0x3);
      instr->setDestReg(rd, RegType::Vector);
      instr->setVlsWidth(func3);
      instr->setVmask((code >> shift_func7) & 0x1);
      instr->setVnf((code >> shift_vnf) & mask_func3);
      break;

    case Opcode::FS:
      instr->addSrcReg(rs1, RegType::Integer);
      instr->setVmop((code >> shift_vmop) & 0b11);
      switch (instr->getVmop()) {
        case 0b00:
          instr->setVumop(rs2);
          break;
        case 0b10:
          instr->addSrcReg(rs2, RegType::Integer);
          break;
        case 0b01:
        case 0b11:
          instr->addSrcReg(rs2, RegType::Vector);
          break;
      }
      instr->setVsew(func3 & 0x3);
      instr->addSrcReg(rd, RegType::Vector);
      instr->setVlsWidth(func3);
      instr->setVmask((code >> shift_func7) & 0x1);
      instr->setVmop((code >> shift_vmop) & 0b11);
      instr->setVnf((code >> shift_vnf) & mask_func3);
      break;

    default:
      std::abort();
    }
    break;
  #endif

  default:
    std::abort();
  }

  return instr;
}
