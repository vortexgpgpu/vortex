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
#include "decode.h"
#include "arch.h"
#include "instr.h"

using namespace vortex;

static const std::unordered_map<Opcode, InstType> sc_instTable = {
  {Opcode::R_INST,     InstType::R_TYPE},
  {Opcode::L_INST,     InstType::I_TYPE},
  {Opcode::I_INST,     InstType::I_TYPE},
  {Opcode::S_INST,     InstType::S_TYPE},
  {Opcode::B_INST,     InstType::B_TYPE},
  {Opcode::LUI_INST,   InstType::U_TYPE},
  {Opcode::AUIPC_INST, InstType::U_TYPE},
  {Opcode::JAL_INST,   InstType::J_TYPE},
  {Opcode::JALR_INST,  InstType::I_TYPE},
  {Opcode::SYS_INST,   InstType::I_TYPE},
  {Opcode::FENCE,      InstType::I_TYPE},
  {Opcode::AMO,        InstType::R_TYPE},
  {Opcode::FL,         InstType::I_TYPE},
  {Opcode::FS,         InstType::S_TYPE},
  {Opcode::FCI,        InstType::R_TYPE}, 
  {Opcode::FMADD,      InstType::R4_TYPE},
  {Opcode::FMSUB,      InstType::R4_TYPE},
  {Opcode::FMNMADD,    InstType::R4_TYPE},
  {Opcode::FMNMSUB,    InstType::R4_TYPE},  
  {Opcode::VSET,       InstType::V_TYPE},
  {Opcode::EXT1,       InstType::R_TYPE},
  {Opcode::EXT2,       InstType::R4_TYPE},
  {Opcode::R_INST_W,   InstType::R_TYPE},
  {Opcode::I_INST_W,   InstType::I_TYPE},
};

enum Constants {
  width_opcode= 7,
  width_reg   = 5,
  width_func2 = 2,
  width_func3 = 3,
  width_func6 = 6,
  width_func7 = 7,
  width_mop   = 3,
  width_vmask = 1,
  width_i_imm = 12,
  width_j_imm = 20,
  width_v_imm = 11,
  width_aq    = 1,
  width_rl    = 1,

  shift_opcode= 0,
  shift_rd    = width_opcode,
  shift_func3 = shift_rd + width_reg,
  shift_rs1   = shift_func3 + width_func3,
  shift_rs2   = shift_rs1 + width_reg,
  shift_func2 = shift_rs2 + width_reg,
  shift_func7 = shift_rs2 + width_reg,
  shift_rs3   = shift_func7 + width_func2,
  shift_vmop  = shift_func7 + width_vmask,
  shift_vnf   = shift_vmop + width_mop,
  shift_func6 = shift_func7 + width_vmask,
  shift_vset  = shift_func7 + width_func6,

  mask_opcode = (1 << width_opcode) - 1,  
  mask_reg    = (1 << width_reg)   - 1,
  mask_func2  = (1 << width_func2) - 1,
  mask_func3  = (1 << width_func3) - 1,
  mask_func6  = (1 << width_func6) - 1,
  mask_func7  = (1 << width_func7) - 1,
  mask_i_imm  = (1 << width_i_imm) - 1,
  mask_j_imm  = (1 << width_j_imm) - 1,
  mask_v_imm  = (1 << width_v_imm) - 1,
};

static const char* op_string(const Instr &instr) {
  auto opcode = instr.getOpcode();
  auto func2  = instr.getFunc2();
  auto func3  = instr.getFunc3();
  auto func7  = instr.getFunc7();
  auto rs2    = instr.getRSrc(1);
  auto imm    = instr.getImm();

  switch (opcode) {
  case Opcode::LUI_INST:   return "LUI";
  case Opcode::AUIPC_INST: return "AUIPC";
  case Opcode::R_INST:
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
      case 0: return func7 ? "SUB" : "ADD";
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
  case Opcode::I_INST:
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
  case Opcode::B_INST:
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
  case Opcode::JAL_INST:   return "JAL";
  case Opcode::JALR_INST:  return "JALR";
  case Opcode::L_INST:
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
  case Opcode::S_INST:
    switch (func3) {
    case 0: return "SB";
    case 1: return "SH";
    case 2: return "SW";
    case 3: return "SD";
    default:
      std::abort();
    }
  case Opcode::R_INST_W:
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
      case 0: return func7 ? "SUBW" : "ADDW";
      case 1: return "SLLW";
      case 5: return func7 ? "SRAW" : "SRLW";  
      default:
        std::abort();
      }
    }
  case Opcode::I_INST_W:
    switch (func3) {
    case 0: return "ADDIW";
    case 1: return "SLLIW";
    case 5: return func7 ? "SRAIW" : "SRLIW";
    default:
      std::abort();
    }
  case Opcode::SYS_INST: 
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
    case 0x1: return "VL";
    case 0x2: return "FLW";
    case 0x3: return "FLD";
    default: 
      std::abort();
    }
  case Opcode::FS: 
    switch (func3) {
    case 0x1: return "VS";
    case 0x2: return "FSW";
    case 0x3: return "FSD";
    default: 
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
      switch (rs2) {
      case 0: return "FCVT.W.S";
      case 1: return "FCVT.WU.S";
      case 2: return "FCVT.L.S";
      case 3: return "FCVT.LU.S";
      default:
        std::abort();
      }
    case 0x61:
      switch (rs2) {
      case 0: return "FCVT.W.D";
      case 1: return "FCVT.WU.D";
      case 2: return "FCVT.L.D";
      case 3: return "FCVT.LU.D";
      default:
        std::abort();
      }
    case 0x68: 
      switch (rs2) {
      case 0: return "FCVT.S.W";
      case 1: return "FCVT.S.WU";
      case 2: return "FCVT.S.L";
      case 3: return "FCVT.S.LU";
      default:
        std::abort();
      }
    case 0x69:
      switch (rs2) {
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
      case 2: return "SPLIT";
      case 3: return "JOIN";
      case 4: return "BAR";
      case 5: return "PRED";
      default:
        std::abort();
      }
    default:
      std::abort();
    }
  case Opcode::EXT2:
    switch (func3) {
    case 1: {
      switch (func2) {
      case 0: return "CMOV"; 
      default:
        std::abort();
      }
    }
    default:
      std::abort();
    }
  default:
    std::abort();
  }
}

namespace vortex {
std::ostream &operator<<(std::ostream &os, const Instr &instr) {  
  auto opcode = instr.getOpcode();
  auto func3  = instr.getFunc3();

  os << op_string(instr);
  
  int sep = 0;
  if (instr.getRDType() != RegType::None) {
    if (sep++ != 0) { os << ", "; } else { os << " "; }
    os << instr.getRDType() << std::dec << instr.getRDest();
  }
  for (uint32_t i = 0; i < instr.getNRSrc(); ++i) {    
    if (instr.getRSType(i) == RegType::None)
      continue;
    if (sep++ != 0) { os << ", "; } else { os << " "; }
    os << instr.getRSType(i) << std::dec << instr.getRSrc(i);
  }
  if (instr.hasImm()) {
    if (sep++ != 0) { os << ", "; } else { os << " "; }
    os << "0x" << std::hex << instr.getImm();
  }
  if (opcode == Opcode::SYS_INST && func3 >= 5) {
    // CSRs with immediate values
    if (sep++ != 0) { os << ", "; } else { os << " "; }
    os << "0x" << std::hex << instr.getRSrc(0);
  }
  return os;
}
}

Decoder::Decoder(const Arch&) {}

std::shared_ptr<Instr> Decoder::decode(uint32_t code) const {  
  auto instr = std::make_shared<Instr>();
  auto op = Opcode((code >> shift_opcode) & mask_opcode);
  instr->setOpcode(op);

  auto func2 = (code >> shift_func2) & mask_func2;
  auto func3 = (code >> shift_func3) & mask_func3;
  auto func6 = (code >> shift_func6) & mask_func6;
  auto func7 = (code >> shift_func7) & mask_func7;

  auto rd  = (code >> shift_rd)  & mask_reg;
  auto rs1 = (code >> shift_rs1) & mask_reg;
  auto rs2 = (code >> shift_rs2) & mask_reg;
  auto rs3 = (code >> shift_rs3) & mask_reg;

  auto op_it = sc_instTable.find(op);
  if (op_it == sc_instTable.end()) {
    std::cout << std::hex << "Error: invalid opcode: 0x" << op << std::endl;
    return nullptr;
  }

  auto iType = op_it->second;
  if (op == Opcode::FL || op == Opcode::FS) { 
    if (func3 != 0x2 && func3 != 0x3) {
      iType = InstType::V_TYPE;
    }
  }

  switch (iType) {
  case InstType::R_TYPE:
    switch (op) {
    case Opcode::FCI:
      switch (func7) {  
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
        case 5: // PRED
          instr->addSrcReg(rs1, RegType::Integer);
          instr->addSrcReg(rs2, RegType::Integer);
          break;
        case 2: // SPLIT
          instr->setDestReg(rd, RegType::Integer);
          instr->addSrcReg(rs1, RegType::Integer);
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

  case InstType::I_TYPE: {
    instr->addSrcReg(rs1, RegType::Integer);
    if (op == Opcode::FL) {
      instr->setDestReg(rd, RegType::Float);      
    } else {
      instr->setDestReg(rd, RegType::Integer);
    }    
    instr->setFunc3(func3);
    instr->setFunc7(func7);
    switch (op) {
    case Opcode::SYS_INST:
      if (func3 != 0) {
        // RV32I: CSR
        if (func3 >= 5) {
          // rs1 holds zimm
          instr->setSrcReg(0, rs1, RegType::None);
        }        
      } else {        
        instr->setDestReg(rd, RegType::None);
        instr->setSrcReg(0, rs1, RegType::None);
      }      
      // uint12
      instr->setImm(code >> shift_rs2);
      break;
    case Opcode::FENCE:
      // uint12
      instr->setImm(code >> shift_rs2);
      instr->setDestReg(rd, RegType::None);
      instr->setSrcReg(0, rs1, RegType::None);
      break;
    case Opcode::I_INST:
    case Opcode::I_INST_W:
      if (func3 == 0x1 || func3 == 0x5) {
        auto shamt = rs2; // uint5
      #if (XLEN == 64)
        if (op == Opcode::I_INST) {
          // uint6
          shamt |= ((func7 & 0x1) << 5);
        }
      #endif
        instr->setImm(shamt);
      } else {
        // int12
        auto imm = code >> shift_rs2;
        instr->setImm(sext(imm, width_i_imm));
      }
      break;
    default:
      // int12
      auto imm = code >> shift_rs2;
      instr->setImm(sext(imm, width_i_imm));
      break;
    }
  } break;
  case InstType::S_TYPE: {    
    instr->addSrcReg(rs1, RegType::Integer);
    if (op == Opcode::FS) {
      instr->addSrcReg(rs2, RegType::Float);
    } else {
      instr->addSrcReg(rs2, RegType::Integer);
    }
    instr->setFunc3(func3);
    auto imm = (func7 << width_reg) | rd;
    instr->setImm(sext(imm, width_i_imm));
  } break;

  case InstType::B_TYPE: {
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

  case InstType::U_TYPE: {
    instr->setDestReg(rd, RegType::Integer);
    auto imm = code >> shift_func3;
    instr->setImm(sext(imm, width_j_imm));
  }  break;

  case InstType::J_TYPE: {
    instr->setDestReg(rd, RegType::Integer);
    auto unordered  = code >> shift_func3;
    auto bits_19_12 = unordered & 0xff;
    auto bit_11     = (unordered >> 8) & 0x1;
    auto bits_10_1  = (unordered >> 9) & 0x3ff;
    auto bit_20     = (unordered >> 19) & 0x1;
    auto imm = (bits_10_1 << 1) | (bit_11 << 11) | (bits_19_12 << 12) | (bit_20 << 20);
    instr->setImm(sext(imm, width_j_imm+1));
  } break;
    
  case InstType::V_TYPE:
    switch (op) {
    case Opcode::VSET: {
      instr->setDestReg(rd, RegType::Vector);
      instr->addSrcReg(rs1, RegType::Vector);
      instr->setFunc3(func3);
      if (func3 == 7) {
        instr->setImm(!(code >> shift_vset));
        if (instr->getImm()) {
          auto immed = (code >> shift_rs2) & mask_v_imm;
          instr->setImm(immed);
          instr->setVlmul(immed & 0x3);
          instr->setVediv((immed >> 4) & 0x3);
          instr->setVsew((immed >> 2) & 0x3);
        } else {
          instr->addSrcReg(rs2, RegType::Vector);
        }
      } else {
        instr->addSrcReg(rs2, RegType::Vector);
        instr->setVmask((code >> shift_func7) & 0x1);
        instr->setFunc6(func6);
      }
    } break;

    case Opcode::FL:
      instr->setDestReg(rd, RegType::Vector);
      instr->addSrcReg(rs1, RegType::Vector);
      instr->setVlsWidth(func3);
      instr->addSrcReg(rs2, RegType::Vector);
      instr->setVmask(code >> shift_func7);
      instr->setVmop((code >> shift_vmop) & mask_func3);
      instr->setVnf((code >> shift_vnf) & mask_func3);
      break;

    case Opcode::FS:
      instr->setVs3(rd);
      instr->addSrcReg(rs1, RegType::Vector);
      instr->setVlsWidth(func3);
      instr->addSrcReg(rs2, RegType::Vector);
      instr->setVmask(code >> shift_func7);
      instr->setVmop((code >> shift_vmop) & mask_func3);
      instr->setVnf((code >> shift_vnf) & mask_func3);
      break;

    default:
      std::abort();
    }
    break;
  case R4_TYPE:
    if (op == Opcode::EXT2) {
      switch (func3) {
      case 1:
        switch (func2) {
        case 0: // CMOV
          instr->setDestReg(rd, RegType::Integer);
          instr->addSrcReg(rs1, RegType::Integer);
          instr->addSrcReg(rs2, RegType::Integer);
          instr->addSrcReg(rs3, RegType::Integer);
          break;
        default:
          std::abort();
        }
        break;
      default:
        std::abort();
      }
    } else {
      instr->setDestReg(rd, RegType::Float);
      instr->addSrcReg(rs1, RegType::Float);
      instr->addSrcReg(rs2, RegType::Float);
      instr->addSrcReg(rs3, RegType::Float);
    }
    instr->setFunc2(func2);
    instr->setFunc3(func3);
    break;
  default:
    std::abort();
  }

  return instr;
}
