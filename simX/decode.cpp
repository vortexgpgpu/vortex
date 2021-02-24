#include <iostream>
#include <string>
#include <stdlib.h>
#include <string.h>
#include <iomanip>
#include <vector>
#include <unordered_map>
#include "debug.h"
#include "types.h"
#include "util.h"
#include "decode.h"
#include "archdef.h"
#include "instr.h"
#include "trace.h"

using namespace vortex;

struct InstTableEntry_t {
  const char *opString;
  bool controlFlow;
  InstType iType;
};

static const std::unordered_map<int, struct InstTableEntry_t> sc_instTable = {
  {Opcode::NOP,        {"nop"   , false, InstType::N_TYPE}},
  {Opcode::R_INST,     {"r_type", false, InstType::R_TYPE}},
  {Opcode::L_INST,     {"load"  , false, InstType::I_TYPE}},
  {Opcode::I_INST,     {"i_type", false, InstType::I_TYPE}},
  {Opcode::S_INST,     {"store" , false, InstType::S_TYPE}},
  {Opcode::B_INST,     {"branch", true , InstType::B_TYPE}},
  {Opcode::LUI_INST,   {"lui"   , false, InstType::U_TYPE}},
  {Opcode::AUIPC_INST, {"auipc" , false, InstType::U_TYPE}},
  {Opcode::JAL_INST,   {"jal"   , true , InstType::J_TYPE}},
  {Opcode::JALR_INST,  {"jalr"  , true , InstType::I_TYPE}},
  {Opcode::SYS_INST,   {"SYS"   , true , InstType::I_TYPE}},
  {Opcode::FENCE,      {"fence" , true , InstType::I_TYPE}},
  {Opcode::PJ_INST,    {"pred j", true , InstType::R_TYPE}},
  {Opcode::GPGPU,      {"gpgpu" , false, InstType::R_TYPE}},
  {Opcode::VSET_ARITH, {"vsetvl", false, InstType::V_TYPE}}, 
  {Opcode::VL,         {"vl"    , false, InstType::V_TYPE}}, 
  {Opcode::VS,         {"vs"    , false, InstType::V_TYPE}},
  {Opcode::FL,         {"fl"    , false, InstType::I_TYPE }},
  {Opcode::FS,         {"fs"    , false, InstType::S_TYPE }},
  {Opcode::FCI,        {"fci"   , false, InstType::R_TYPE }}, 
  {Opcode::FMADD,      {"fma"   , false, InstType::R4_TYPE }},
  {Opcode::FMSUB,      {"fms"   , false, InstType::R4_TYPE }},
  {Opcode::FMNMADD,    {"fmnma" , false, InstType::R4_TYPE }},
  {Opcode::FMNMSUB,    {"fmnms" , false, InstType::R4_TYPE }}   
};

std::ostream &vortex::operator<<(std::ostream &os, Instr &instr) {
  os << std::dec << sc_instTable.at(instr.opcode_).opString;
  return os;
}

Decoder::Decoder(const ArchDef &arch) {
  inst_s_   = arch.getWordSize() * 8;
  opcode_s_ = 7;
  reg_s_    = 5;
  func2_s_  = 2;
  func3_s_  = 3;
  mop_s_    = 3;
  vmask_s_  = 1;

  shift_opcode_ = 0;
  shift_rd_ = opcode_s_;
  shift_func3_ = opcode_s_ + reg_s_;
  shift_rs1_ = opcode_s_ + reg_s_ + func3_s_;
  shift_rs2_ = opcode_s_ + reg_s_ + func3_s_ + reg_s_;
  shift_func7_ = opcode_s_ + reg_s_ + func3_s_ + reg_s_ + reg_s_;
  shift_func2_ = opcode_s_ + reg_s_ + func3_s_ + reg_s_ + reg_s_;
  shift_rs3_   = opcode_s_ + reg_s_ + func3_s_ + reg_s_ + reg_s_ + func2_s_;
  shift_j_u_immed_ = opcode_s_ + reg_s_;
  shift_s_b_immed_ = opcode_s_ + reg_s_ + func3_s_ + reg_s_ + reg_s_;
  shift_i_immed_ = opcode_s_ + reg_s_ + func3_s_ + reg_s_;
  shift_vset_immed_ = opcode_s_ + reg_s_ + func3_s_ + reg_s_;
  shift_vmask_ = opcode_s_ + reg_s_ + func3_s_ + reg_s_ + reg_s_;
  shift_vmop_ = opcode_s_ + reg_s_ + func3_s_ + reg_s_ + reg_s_ + vmask_s_;
  shift_vnf_ = opcode_s_ + reg_s_ + func3_s_ + reg_s_ + reg_s_ + vmask_s_ + mop_s_;
  shift_func6_ = opcode_s_ + reg_s_ + func3_s_ + reg_s_ + reg_s_ + 1;
  shift_vset_ = opcode_s_ + reg_s_ + func3_s_ + reg_s_ + reg_s_ + 6;

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

std::shared_ptr<Instr> Decoder::decode(const std::vector<Byte> &v, Size &idx, trace_inst_t *trace_inst) {
  Word code(readWord(v, idx, inst_s_ / 8));

  // std::cout << "code: " << (int) code << "  v: " << v << " indx: " << idx << "\n";
  auto instr = std::make_shared<Instr>();

  Opcode op = (Opcode)((code >> shift_opcode_) & opcode_mask_);
  // std::cout << "opcode: " << op << "\n";
  instr->setOpcode(op);

  Word imeed, dest_bits, imm_bits, bit_11, bits_4_1, bit_10_5,
      bit_12, bits_19_12, bits_10_1, bit_20, unordered, func3;

  InstType curInstType = sc_instTable.at(op).iType; // get current inst type
  if (op == Opcode::FL || op == Opcode::FS) { // need to find out whether it is vector or floating point inst
    Word width_bits = (code >> shift_func3_)  & func3_mask_;
    if ((width_bits == 0x1) || (width_bits == 0x2) 
     || (width_bits == 0x3) || (width_bits == 0x4)) {
      curInstType = (op == Opcode::FL)? InstType::I_TYPE : InstType::S_TYPE;
    }
  }

  // std::cout << "op: " << std::hex << op << " what " << sc_instTable[op].iType << "\n";
  switch (curInstType) {
  case InstType::N_TYPE:
    break;

  case InstType::R_TYPE:
    instr->setDestReg((code >> shift_rd_) & reg_mask_);
    instr->setSrcReg((code >> shift_rs1_) & reg_mask_);
    instr->setSrcReg((code >> shift_rs2_) & reg_mask_);
    instr->setFunc3((code >> shift_func3_) & func3_mask_);
    instr->setFunc7((code >> shift_func7_) & func7_mask_);

    trace_inst->valid_inst = true;
    trace_inst->rs1 = ((code >> shift_rs1_) & reg_mask_);
    trace_inst->rs2 = ((code >> shift_rs2_) & reg_mask_);
    trace_inst->rd = ((code >> shift_rd_) & reg_mask_);
    break;

  case InstType::I_TYPE:
    instr->setDestReg((code >> shift_rd_) & reg_mask_);
    instr->setSrcReg((code >> shift_rs1_) & reg_mask_);
    instr->setFunc7((code >> shift_func7_) & func7_mask_);
    func3 = (code >> shift_func3_) & func3_mask_;
    instr->setFunc3(func3);

    if ((func3 == 5) && (op != L_INST) && (op != FL)) {
      // std::cout << "func7: " << func7 << "\n";
      instr->setSrcImm(signExt(((code >> shift_rs2_) & reg_mask_), 5, reg_mask_));
    } else {
      instr->setSrcImm(signExt(code >> shift_i_immed_, 12, i_imm_mask_));
    }

    trace_inst->valid_inst = true;
    trace_inst->rs1 = ((code >> shift_rs1_) & reg_mask_);
    trace_inst->rd = ((code >> shift_rd_) & reg_mask_);
    break;

  case InstType::S_TYPE:
    // std::cout << "************STORE\n";
    instr->setSrcReg((code >> shift_rs1_) & reg_mask_);
    instr->setSrcReg((code >> shift_rs2_) & reg_mask_);
    instr->setFunc3((code >> shift_func3_) & func3_mask_);

    dest_bits = (code >> shift_rd_) & reg_mask_;
    imm_bits = (code >> shift_s_b_immed_ & func7_mask_);
    imeed = (imm_bits << reg_s_) | dest_bits;
    // std::cout << "ENC: store imeed: " << imeed << "\n";
    instr->setSrcImm(signExt(imeed, 12, s_imm_mask_));

    trace_inst->valid_inst = true;
    trace_inst->rs1 = ((code >> shift_rs1_) & reg_mask_);
    trace_inst->rs2 = ((code >> shift_rs2_) & reg_mask_);
    break;

  case InstType::B_TYPE:
    instr->setSrcReg((code >> shift_rs1_) & reg_mask_);
    instr->setSrcReg((code >> shift_rs2_) & reg_mask_);
    instr->setFunc3((code >> shift_func3_) & func3_mask_);

    dest_bits = (code >> shift_rd_) & reg_mask_;
    imm_bits = (code >> shift_s_b_immed_ & func7_mask_);

    bit_11   = dest_bits & 0x1;
    bits_4_1 = dest_bits >> 1;
    bit_10_5 = imm_bits & 0x3f;
    bit_12   = imm_bits >> 6;

    imeed = 0 | (bits_4_1 << 1) | (bit_10_5 << 5) | (bit_11 << 11) | (bit_12 << 12);

    instr->setSrcImm(signExt(imeed, 13, b_imm_mask_));

    trace_inst->valid_inst = true;
    trace_inst->rs1 = ((code >> shift_rs1_) & reg_mask_);
    trace_inst->rs2 = ((code >> shift_rs2_) & reg_mask_);
    break;

  case InstType::U_TYPE:
    instr->setDestReg((code >> shift_rd_) & reg_mask_);
    instr->setSrcImm(signExt(code >> shift_j_u_immed_, 20, u_imm_mask_));
    trace_inst->valid_inst = true;
    trace_inst->rd = ((code >> shift_rd_) & reg_mask_);
    break;

  case InstType::J_TYPE:
    instr->setDestReg((code >> shift_rd_) & reg_mask_);

    // [20 | 10:1 | 11 | 19:12]

    unordered = code >> shift_j_u_immed_;

    bits_19_12 = unordered & 0xff;
    bit_11 = (unordered >> 8) & 0x1;
    bits_10_1 = (unordered >> 9) & 0x3ff;
    bit_20 = (unordered >> 19) & 0x1;

    imeed = 0 | (bits_10_1 << 1) | (bit_11 << 11) | (bits_19_12 << 12) | (bit_20 << 20);

    if (bit_20) {
      imeed |= ~j_imm_mask_;
    }

    instr->setSrcImm(imeed);

    trace_inst->valid_inst = true;
    trace_inst->rd = ((code >> shift_rd_) & reg_mask_);
    break;
    
  case InstType::V_TYPE:
    D(3, "Entered here: instr type = vector" << op);
    switch (op) {
    case Opcode::VSET_ARITH: //TODO: arithmetic ops
      instr->setDestReg((code >> shift_rd_) & reg_mask_);
      instr->setSrcReg((code >> shift_rs1_) & reg_mask_);
      func3 = (code >> shift_func3_) & func3_mask_;
      instr->setFunc3(func3);
      D(3, "Entered here: instr type = vector");

      if (func3 == 7) {
        D(3, "Entered here: imm instr");
        instr->setVsetImm(!(code >> shift_vset_));
        if (instr->getVsetImm()) {
          Word immed = (code >> shift_rs2_) & v_imm_mask_;
          D(3, "immed" << immed);
          instr->setSrcImm(immed); //TODO
          instr->setVlmul(immed & 0x3);
          D(3, "lmul " << (immed & 0x3));
          instr->setVediv((immed >> 4) & 0x3);
          D(3, "ediv " << ((immed >> 4) & 0x3));
          instr->setVsew((immed >> 2) & 0x3);
          D(3, "sew " << ((immed >> 2) & 0x3));
        } else {
          instr->setSrcReg((code >> shift_rs2_) & reg_mask_);
          trace_inst->rs2 = ((code >> shift_rs2_) & reg_mask_);
        }
        trace_inst->valid_inst = true;
        trace_inst->rs1 = ((code >> shift_rs1_) & reg_mask_);
        trace_inst->rd = ((code >> shift_rd_) & reg_mask_);
      } else {
        instr->setSrcReg((code >> shift_rs2_) & reg_mask_);
        instr->setVmask((code >> shift_vmask_) & 0x1);
        instr->setFunc6((code >> shift_func6_) & func6_mask_);

        trace_inst->valid_inst = true;
        trace_inst->rs1 = ((code >> shift_rs1_) & reg_mask_);
        trace_inst->rs2 = ((code >> shift_rs2_) & reg_mask_);
        trace_inst->rd = ((code >> shift_rd_) & reg_mask_);
      }
      break;

    case Opcode::VL:
      D(3, "vector load instr");
      instr->setDestReg((code >> shift_rd_) & reg_mask_);
      instr->setSrcReg((code >> shift_rs1_) & reg_mask_);
      instr->setVlsWidth((code >> shift_func3_) & func3_mask_);
      instr->setSrcReg((code >> shift_rs2_) & reg_mask_);
      instr->setVmask((code >> shift_vmask_));
      instr->setVmop((code >> shift_vmop_) & func3_mask_);
      instr->setVnf((code >> shift_vnf_) & func3_mask_);

      trace_inst->valid_inst = true;
      trace_inst->rs1 = ((code >> shift_rs1_) & reg_mask_);
      trace_inst->vd = ((code >> shift_rd_) & reg_mask_);
      //trace_inst->vs2        = ((code>>shift_rs2_)   & reg_mask_);
      break;

    case Opcode::VS:
      instr->setVs3((code >> shift_rd_) & reg_mask_);
      instr->setSrcReg((code >> shift_rs1_) & reg_mask_);
      instr->setVlsWidth((code >> shift_func3_) & func3_mask_);
      instr->setSrcReg((code >> shift_rs2_) & reg_mask_);
      instr->setVmask((code >> shift_vmask_));
      instr->setVmop((code >> shift_vmop_) & func3_mask_);
      instr->setVnf((code >> shift_vnf_) & func3_mask_);

      trace_inst->valid_inst = true;
      trace_inst->rs1 = ((code >> shift_rs1_) & reg_mask_);
      //trace_inst->vd = ((code>>shift_rd_) & reg_mask_);
      trace_inst->vs1 = ((code >> shift_rd_) & reg_mask_); //vs3
      break;

    default:
      std::cout << "Inavlid opcode.\n";
      std::abort();
    }
    break;
  case R4_TYPE:
    // RT: add R4_TYPE decoder    
    instr->setDestReg((code >> shift_rd_) & reg_mask_);
    instr->setSrcReg((code >> shift_rs1_) & reg_mask_);
    instr->setSrcReg((code >> shift_rs2_) & reg_mask_);
    instr->setSrcReg((code >> shift_rs3_) & reg_mask_);
    instr->setFunc3((code >> shift_func3_) & func3_mask_);
    
    trace_inst->valid_inst = true;
    trace_inst->rs1 = ((code >> shift_rs1_) & reg_mask_);
    trace_inst->rs2 = ((code >> shift_rs2_) & reg_mask_);
    trace_inst->rs3 = ((code >> shift_rs3_) & reg_mask_);
    trace_inst->rd  = ((code >> shift_rd_) & reg_mask_);
    break;
  default:
    std::cout << "Unrecognized argument class in word decoder.\n";
    std::abort();
  }

  D(2, "Decoded instr 0x" << std::hex << code << " into: " << instr << std::flush);

  return instr;
}
