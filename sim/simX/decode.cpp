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
#include "archdef.h"
#include "instr.h"

using namespace vortex;

struct InstTableEntry_t {
  bool controlFlow;
  InstType iType;
};

static const std::unordered_map<int, struct InstTableEntry_t> sc_instTable = {
  {Opcode::NOP,        {false, InstType::N_TYPE}},
  {Opcode::R_INST,     {false, InstType::R_TYPE}},
  {Opcode::L_INST,     {false, InstType::I_TYPE}},
  {Opcode::I_INST,     {false, InstType::I_TYPE}},
  {Opcode::S_INST,     {false, InstType::S_TYPE}},
  {Opcode::B_INST,     {true , InstType::B_TYPE}},
  {Opcode::LUI_INST,   {false, InstType::U_TYPE}},
  {Opcode::AUIPC_INST, {false, InstType::U_TYPE}},
  {Opcode::JAL_INST,   {true , InstType::J_TYPE}},
  {Opcode::JALR_INST,  {true , InstType::I_TYPE}},
  {Opcode::SYS_INST,   {true , InstType::I_TYPE}},
  {Opcode::FENCE,      {true , InstType::I_TYPE}},
  {Opcode::FL,         {false, InstType::I_TYPE}},
  {Opcode::FS,         {false, InstType::S_TYPE}},
  {Opcode::FCI,        {false, InstType::R_TYPE}}, 
  {Opcode::FMADD,      {false, InstType::R4_TYPE}},
  {Opcode::FMSUB,      {false, InstType::R4_TYPE}},
  {Opcode::FMNMADD,    {false, InstType::R4_TYPE}},
  {Opcode::FMNMSUB,    {false, InstType::R4_TYPE}},  
  {Opcode::VSET,       {false, InstType::V_TYPE}}, 
  {Opcode::GPGPU,      {false, InstType::R_TYPE}},
};

static const char* op_string(const Instr &instr) {  
  Word func3 = instr.getFunc3();
  Word func7 = instr.getFunc7();
  Word rs2   = instr.getRSrc(1);
  Word imm   = instr.getImm();
  switch (instr.getOpcode()) {
  case Opcode::NOP:        return "NOP";
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
      }
    } else {
      switch (func3) {
      case 0: return func7 ? "SUB" : "ADD";
      case 1: return "SLL";
      case 2: return "SLT";
      case 3: return "SLTU";
      case 4: return "XOR";
      case 5: return func7 ? "SRA" : "SRL";
      case 6: return "OR";
      case 7: return "AND";
      }
    }
  case Opcode::I_INST:
    switch (func3) {
    case 0: return "ADDI";
    case 1: return "SLLI";
    case 2: return "SLTI";
    case 3: return "SLTIU";
    case 4: return "XORI";
    case 5: return func7 ? "SRAI" : "SRLI";
    case 6: return "ORI";
    case 7: return "ANDI";
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
    case 0: return "LBI";
    case 1: return "LHI";
    case 2: return "LW";
    case 4: return "LBU";
    case 5: return "LHU";
    default:
      std::abort();
    }
  case Opcode::S_INST:
    switch (func3) {
    case 0: return "SB";
    case 1: return "SH";
    case 2: return "SW";
    default:
      std::abort();
    }
  case Opcode::SYS_INST: 
    switch (func3) {
    case 0: return imm ? "EBREAK" : "ECALL";
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
  case Opcode::FL: return (func3 == 0x2) ? "FL" : "VL";
  case Opcode::FS: return (func3 == 0x2) ? "FS" : "VS";
  case Opcode::FCI: 
    switch (func7) {
    case 0x00: return "FADD";
    case 0x04: return "FSUB";
    case 0x08: return "FMUL";
    case 0x0c: return "FDIV";
    case 0x2c: return "FSQRT";
    case 0x10:
      switch (func3) {            
      case 0: return "FSGNJ";
      case 1: return "FSGNJN";
      case 2: return "FSGNJX";
      default:
        std::abort();
      }
    case 0x14:
      switch (func3) {            
      case 0: return "FMIM";
      case 1: return "FMAX";
      default:
        std::abort();
      }
    case 0x50: 
      switch (func3) {              
      case 0: return "FLE"; 
      case 1: return "FLT"; 
      case 2: return "FEQ";
      default:
        std::abort();
      }
    case 0x60: return rs2 ? "FCVT.WU.S" : "FCVT.W.S";
    case 0x68: return rs2 ? "FCVT.S.WU" : "FCVT.S.W";
    case 0x70: return func3 ? "FLASS" : "FMV.X.W";
    case 0x78: return "FMV.W.X";
    default:
      std::abort();
    }
  case Opcode::FMADD:   return "FMADD";
  case Opcode::FMSUB:   return "FMSUB";
  case Opcode::FMNMADD: return "FMNMADD";
  case Opcode::FMNMSUB: return "FMNMSUB";
  case Opcode::VSET:    return "VSET";
  case Opcode::GPGPU:
    switch (func3) {            
    case 0: return "TMC";
    case 1: return "WSPAWN";
    case 2: return "SPLIT";
    case 3: return "JOIN";
    case 4: return "BAR"; 
    case 6: return "PREFETCH";
    default:
      std::abort();
    }
  default:
    std::abort();
  }  
}

namespace vortex {
std::ostream &operator<<(std::ostream &os, const Instr &instr) {
  os << op_string(instr) << ": ";
  auto opcode = instr.getOpcode();
    
  auto rd_to_string = [&]() {
    int rdt = instr.getRDType();
    int rd = instr.getRDest();
    switch (rdt) {
    case 1: os << "r" << std::dec << rd << " <- "; break;
    case 2: os << "fr" << std::dec << rd << " <- "; break;
    case 3: os << "vr" << std::dec << rd << " <- "; break;
    default: break;
    }
  };

  auto rs_to_string = [&](int i) {
    int rst = instr.getRSType(i);
    int rs = instr.getRSrc(i);    
    switch (rst) {
    case 1: os << "r" << std::dec << rs; break;
    case 2: os << "fr" << std::dec << rs; break;
    case 3: os << "vr" << std::dec << rs; break;
    default: break;
    }
  };

  if (opcode == S_INST 
   || opcode == FS
   || opcode == VS) {     
     os << "M[r" << std::dec << instr.getRSrc(0) << " + 0x" << std::hex << instr.getImm() << "] <- ";
     rs_to_string(1);
  } else 
  if (opcode == L_INST 
   || opcode == FL
   || opcode == VL) {     
     rd_to_string();
     os << "M[r" << std::dec << instr.getRSrc(0) << " + 0x" << std::hex << instr.getImm() << "]";
  } else {
    rd_to_string();
    int i = 0;
    for (; i < instr.getNRSrc(); ++i) {    
      if (i) os << ", ";
      rs_to_string(i);
    }    
    if (instr.hasImm()) {
      if (i) os << ", ";
      os << "imm=0x" << std::hex << instr.getImm();
    }
  } 

  return os;
}
}

Decoder::Decoder(const ArchDef &arch) {
  inst_s_   = arch.wsize() * 8;
  opcode_s_ = 7;
  reg_s_    = 5;
  func2_s_  = 2;
  func3_s_  = 3;
  mop_s_    = 3;
  vmask_s_  = 1;

  shift_opcode_ = 0;
  shift_rd_     = opcode_s_;
  shift_func3_  = shift_rd_ + reg_s_;
  shift_rs1_    = shift_func3_ + func3_s_;
  shift_rs2_    = shift_rs1_ + reg_s_;
  shift_func7_  = shift_rs2_ + reg_s_;
  shift_rs3_    = shift_func7_ + func2_s_;
  shift_vmop_   = shift_func7_ + vmask_s_;
  shift_vnf_    = shift_vmop_ + mop_s_;
  shift_func6_  = shift_func7_ + 1;
  shift_vset_   = shift_func7_ + 6;

  reg_mask_    = 0x1f;
  func2_mask_  = 0x2;
  func3_mask_  = 0x7;
  func6_mask_  = 0x3f;
  func7_mask_  = 0x7f;
  opcode_mask_ = 0x7f;
  i_imm_mask_  = 0xfff;
  s_imm_mask_  = 0xfff;
  b_imm_mask_  = 0x1fff;
  u_imm_mask_  = 0xfffff;
  j_imm_mask_  = 0xfffff;
  v_imm_mask_  = 0x7ff;  
}

std::shared_ptr<Instr> Decoder::decode(Word code, Word PC) {  
  auto instr = std::make_shared<Instr>();
  Opcode op = (Opcode)((code >> shift_opcode_) & opcode_mask_);
  instr->setOpcode(op);

  Word func3 = (code >> shift_func3_) & func3_mask_;
  Word func6 = (code >> shift_func6_) & func6_mask_;
  Word func7 = (code >> shift_func7_) & func7_mask_;

  int rd  = (code >> shift_rd_)  & reg_mask_;
  int rs1 = (code >> shift_rs1_) & reg_mask_;
  int rs2 = (code >> shift_rs2_) & reg_mask_;
  int rs3 = (code >> shift_rs3_) & reg_mask_;

  auto op_it = sc_instTable.find(op);
  if (op_it == sc_instTable.end()) {
    std::cout << std::hex << "invalid opcode: 0x" << op << ", instruction=0x" << code << ", PC=" << PC << std::endl;
    std::abort();
  }

  auto iType = op_it->second.iType;
  if (op == Opcode::FL || op == Opcode::FS) { 
    if (func3 != 0x2) {
      iType = InstType::V_TYPE;
    }
  }

  switch (iType) {
  case InstType::N_TYPE:
    break;

  case InstType::R_TYPE:
    if (op == Opcode::FCI) {      
      switch (func7) {
      case 0x68: // FCVT.S.W, FCVT.S.WU
      case 0x78: // FMV.W.X
        instr->setSrcReg(rs1);
        break;
      default:
        instr->setSrcFReg(rs1);
      }      
      instr->setSrcFReg(rs2);
      switch (func7) {
      case 0x50: // FLE, FLT, FEQ
      case 0x60: // FCVT.WU.S, FCVT.W.S
      case 0x70: // FLASS, FMV.X.W
        instr->setDestReg(rd);
        break;
      default:
        instr->setDestFReg(rd);
      }
    } else {
      instr->setDestReg(rd);
      instr->setSrcReg(rs1);
      instr->setSrcReg(rs2);
    }
    instr->setFunc3(func3);
    instr->setFunc7(func7);
    break;

  case InstType::I_TYPE: {
    instr->setSrcReg(rs1);
    if (op == Opcode::FL) {
      instr->setDestFReg(rd);      
    } else {
      instr->setDestReg(rd);
    }    
    instr->setFunc3(func3);
    instr->setFunc7(func7);    
    if ((func3 == 5) && (op != L_INST) && (op != Opcode::FL)) {
      instr->setImm(signExt(rs2, 5, reg_mask_));
    } else {
      instr->setImm(signExt(code >> shift_rs2_, 12, i_imm_mask_));
    }
  } break;

  case InstType::S_TYPE: {    
    instr->setSrcReg(rs1);
    if (op == Opcode::FS) {
      instr->setSrcFReg(rs2);
    } else {
      instr->setSrcReg(rs2);
    }
    instr->setFunc3(func3);
    Word imeed = (func7 << reg_s_) | rd;
    instr->setImm(signExt(imeed, 12, s_imm_mask_));
  } break;

  case InstType::B_TYPE: {
    instr->setSrcReg(rs1);
    instr->setSrcReg(rs2);
    instr->setFunc3(func3);
    Word bit_11   = rd & 0x1;
    Word bits_4_1 = rd >> 1;
    Word bit_10_5 = func7 & 0x3f;
    Word bit_12   = func7 >> 6;
    Word imeed = (bits_4_1 << 1) | (bit_10_5 << 5) | (bit_11 << 11) | (bit_12 << 12);
    instr->setImm(signExt(imeed, 13, b_imm_mask_));
  } break;

  case InstType::U_TYPE:
    instr->setDestReg(rd);
    instr->setImm(signExt(code >> shift_func3_, 20, u_imm_mask_));
    break;

  case InstType::J_TYPE: {
    instr->setDestReg(rd);
    Word unordered = code >> shift_func3_;
    Word bits_19_12 = unordered & 0xff;
    Word bit_11 = (unordered >> 8) & 0x1;
    Word bits_10_1 = (unordered >> 9) & 0x3ff;
    Word bit_20 = (unordered >> 19) & 0x1;
    Word imeed = 0 | (bits_10_1 << 1) | (bit_11 << 11) | (bits_19_12 << 12) | (bit_20 << 20);
    if (bit_20) {
      imeed |= ~j_imm_mask_;
    }
    instr->setImm(imeed);
  } break;
    
  case InstType::V_TYPE:
    switch (op) {
    case Opcode::VSET: {
      instr->setDestVReg(rd);
      instr->setSrcVReg(rs1);
      instr->setFunc3(func3);
      if (func3 == 7) {
        instr->setImm(!(code >> shift_vset_));
        if (instr->getImm()) {
          Word immed = (code >> shift_rs2_) & v_imm_mask_;
          instr->setImm(immed);
          instr->setVlmul(immed & 0x3);
          instr->setVediv((immed >> 4) & 0x3);
          instr->setVsew((immed >> 2) & 0x3);
        } else {
          instr->setSrcVReg(rs2);
        }
      } else {
        instr->setSrcVReg(rs2);
        instr->setVmask((code >> shift_func7_) & 0x1);
        instr->setFunc6(func6);
      }
    } break;

    case Opcode::VL:
      instr->setDestVReg(rd);
      instr->setSrcVReg(rs1);
      instr->setVlsWidth(func3);
      instr->setSrcVReg(rs2);
      instr->setVmask(code >> shift_func7_);
      instr->setVmop((code >> shift_vmop_) & func3_mask_);
      instr->setVnf((code >> shift_vnf_) & func3_mask_);
      break;

    case Opcode::VS:
      instr->setVs3(rd);
      instr->setSrcVReg(rs1);
      instr->setVlsWidth(func3);
      instr->setSrcVReg(rs2);
      instr->setVmask(code >> shift_func7_);
      instr->setVmop((code >> shift_vmop_) & func3_mask_);
      instr->setVnf((code >> shift_vnf_) & func3_mask_);
      break;

    default:
      std::abort();
    }
    break;
  case R4_TYPE:
    instr->setDestFReg(rd);
    instr->setSrcFReg(rs1);
    instr->setSrcFReg(rs2);
    instr->setSrcFReg(rs3);
    instr->setFunc3(func3);
    break;
  default:
    std::abort();
  }

  D(2, "Instr 0x" << std::hex << code << ": " << *instr << std::flush);

  return instr;
}
