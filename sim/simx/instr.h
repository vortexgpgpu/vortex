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

#pragma once

#include "types.h"

namespace vortex {

enum class Opcode {
  NONE      = 0,
  R         = 0x33,
  L         = 0x3,
  I         = 0x13,
  S         = 0x23,
  B         = 0x63,
  LUI       = 0x37,
  AUIPC     = 0x17,
  JAL       = 0x6f,
  JALR      = 0x67,
  SYS       = 0x73,
  FENCE     = 0x0f,
  AMO       = 0x2f,
  // F Extension
  FL        = 0x7,
  FS        = 0x27,
  FCI       = 0x53,
  FMADD     = 0x43,
  FMSUB     = 0x47,
  FMNMSUB   = 0x4b,
  FMNMADD   = 0x4f,
  // RV64 Standard Extension
  R_W       = 0x3b,
  I_W       = 0x1b,
  // Vector Extension
  VSET      = 0x57,
  // Custom Extensions
  EXT1      = 0x0b,
  EXT2      = 0x2b,
  EXT3      = 0x5b,
  EXT4      = 0x7b
};

enum class InstType {
  R,
  I,
  S,
  B,
  U,
  J,
  V,
  R4
};

enum DecodeConstants {
  width_opcode= 7,
  width_reg   = 5,
  width_funct2= 2,
  width_funct3= 3,
  width_funct6= 6,
  width_funct7= 7,
  width_mop   = 3,
  width_vmask = 1,
  width_i_imm = 12,
  width_j_imm = 20,
  width_v_zimm= 11,
  width_v_ma  = 1,
  width_v_ta  = 1,
  width_v_sew = 3,
  width_v_lmul= 3,
  width_aq    = 1,
  width_rl    = 1,

  shift_opcode= 0,
  shift_rd    = width_opcode,
  shift_funct3= shift_rd + width_reg,
  shift_rs1   = shift_funct3 + width_funct3,
  shift_rs2   = shift_rs1 + width_reg,
  shift_funct2= shift_rs2 + width_reg,
  shift_funct7= shift_rs2 + width_reg,
  shift_rs3   = shift_funct7 + width_funct2,
  shift_vmop  = shift_funct7 + width_vmask,
  shift_vnf   = shift_vmop + width_mop,
  shift_funct6= shift_funct7 + width_vmask,
  shift_vset  = shift_funct7 + width_funct6,
  shift_v_sew = width_v_lmul,
  shift_v_ta  = shift_v_sew + width_v_sew,
  shift_v_ma  = shift_v_ta + width_v_ta,

  mask_opcode = (1 << width_opcode) - 1,
  mask_reg    = (1 << width_reg)   - 1,
  mask_funct2 = (1 << width_funct2) - 1,
  mask_funct3 = (1 << width_funct3) - 1,
  mask_funct6 = (1 << width_funct6) - 1,
  mask_funct7 = (1 << width_funct7) - 1,
  mask_i_imm  = (1 << width_i_imm) - 1,
  mask_j_imm  = (1 << width_j_imm) - 1,
  mask_v_zimm = (1 << width_v_zimm) - 1,
  mask_v_ma   = (1 << width_v_ma) - 1,
  mask_v_ta   = (1 << width_v_ta) - 1,
  mask_v_sew  = (1 << width_v_sew) - 1,
  mask_v_lmul = (1 << width_v_lmul) - 1,
};

enum VectorAttrMask {
  vattr_vlswidth = (1 << 0),
  vattr_vmop     = (1 << 1),
  vattr_vumop    = (1 << 2),
  vattr_vnf      = (1 << 3),
  vattr_vmask    = (1 << 4),
  vattr_vs3      = (1 << 5),
  vattr_zimm     = (1 << 6),
  vattr_vediv    = (1 << 7)
};

class Instr {
public:
  Instr()
    : opcode_(Opcode::NONE)
    , num_rsrcs_(0)
    , has_imm_(false)
    , imm_(0)
    , funct2_(0)
    , funct3_(0)
    , funct6_(0)
    , funct7_(0)
  {}

  void setOpcode(Opcode opcode) {
    opcode_ = opcode;
  }

  void setDestReg(uint32_t destReg, RegType type) {
    rdest_ = {type, destReg };
  }

  void addSrcReg(uint32_t srcReg, RegType type) {
    rsrc_[num_rsrcs_] = {type, srcReg};
    ++num_rsrcs_;
  }

  void setSrcReg(uint32_t index, uint32_t srcReg, RegType type) {
    rsrc_[index] = { type, srcReg};
    num_rsrcs_ = std::max<uint32_t>(num_rsrcs_, index+1);
  }

  void setImm(uint32_t imm) { has_imm_ = true; imm_ = imm; }

  void setfunct2(uint32_t funct2) { funct2_ = funct2; }
  void setfunct3(uint32_t funct3) { funct3_ = funct3; }
  void setfunct6(uint32_t funct6) { funct6_ = funct6; }
  void setfunct7(uint32_t funct7) { funct7_ = funct7; }

  Opcode   getOpcode() const { return opcode_; }

  uint32_t getNumSrcRegs() const { return num_rsrcs_; }
  RegOpd   getSrcReg(uint32_t i) const { return rsrc_[i]; }

  RegOpd   getDestReg() const { return rdest_; }

  bool     hasImm() const { return has_imm_; }
  uint32_t getImm() const { return imm_; }

  uint32_t getFunct2() const { return funct2_; }
  uint32_t getFunct3() const { return funct3_; }
  uint32_t getFunct6() const { return funct6_; }
  uint32_t getFunct7() const { return funct7_; }

#ifdef EXT_V_ENABLE
  // Attributes for Vector instructions
  void setVlsWidth(uint32_t width) { vlsWidth_ = width; vattr_mask_ |= vattr_vlswidth; }
  void setVmop(uint32_t mop) { vmop_ = mop; vattr_mask_ |= vattr_vmop; }
  void setVumop(uint32_t umop) { vumop_ = umop; vattr_mask_ |= vattr_vumop; }
  void setVnf(uint32_t nf) { vnf_ = nf; vattr_mask_ |= vattr_vnf; }
  void setVmask(uint32_t vmask) { vmask_ = vmask; vattr_mask_ |= vattr_vmask; }
  void setVs3(uint32_t vs) { vs3_ = vs; vattr_mask_ |= vattr_vs3; }
  void setZimm(uint32_t zimm) { zimm_ = zimm; vattr_mask_ |= vattr_zimm; }
  void setVediv(uint32_t ediv) { vediv_ = 1 << ediv; vattr_mask_ |= vattr_vediv; }

  uint32_t getVlsWidth() const { return vlsWidth_; }
  uint32_t getVmop() const { return vmop_; }
  uint32_t getVumop() const { return vumop_; }
  uint32_t getVnf() const { return vnf_; }
  uint32_t getVmask() const { return vmask_; }
  uint32_t getVs3() const { return vs3_; }
  uint32_t getZimm() const { return zimm_; }
  uint32_t getVediv() const { return vediv_; }
  uint32_t getVattrMask() const { return vattr_mask_; }
  bool     hasVattrMask(VectorAttrMask mask) const { return vattr_mask_ & mask; }
#endif

private:

  enum {
    MAX_REG_SOURCES = 3
  };

  Opcode   opcode_;
  uint32_t num_rsrcs_;
  bool     has_imm_;
  RegOpd   rsrc_[MAX_REG_SOURCES];
  RegOpd   rdest_;
  uint32_t imm_;
  uint32_t funct2_;
  uint32_t funct3_;
  uint32_t funct6_;
  uint32_t funct7_;

#ifdef EXT_V_ENABLE
  // Vector
  uint32_t vmask_ = 0;
  uint32_t vlsWidth_ = 0;
  uint32_t vmop_ = 0;
  uint32_t vumop_ = 0;
  uint32_t vnf_ = 0;
  uint32_t vs3_ = 0;
  uint32_t zimm_ = 0;
  uint32_t vediv_ = 0;
  uint32_t vattr_mask_ = 0;
#endif

  friend std::ostream &operator<<(std::ostream &, const Instr&);
};

}