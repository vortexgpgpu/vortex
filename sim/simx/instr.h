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
  TCU       = 0x7b
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
  width_func2 = 2,
  width_func3 = 3,
  width_func6 = 6,
  width_func7 = 7,
  width_mop   = 3,
  width_vmask = 1,
  width_i_imm = 12,
  width_j_imm = 20,
  width_v_zimm = 11,
  width_v_ma = 1,
  width_v_ta = 1,
  width_v_sew = 3,
  width_v_lmul = 3,
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
  shift_v_sew = width_v_lmul,
  shift_v_ta  = shift_v_sew + width_v_sew,
  shift_v_ma  = shift_v_ta + width_v_ta,

  mask_opcode = (1 << width_opcode) - 1,
  mask_reg    = (1 << width_reg)   - 1,
  mask_func2  = (1 << width_func2) - 1,
  mask_func3  = (1 << width_func3) - 1,
  mask_func6  = (1 << width_func6) - 1,
  mask_func7  = (1 << width_func7) - 1,
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
  vattr_vlmul    = (1 << 7),
  vattr_vsew     = (1 << 8),
  vattr_vta      = (1 << 9),
  vattr_vma      = (1 << 10),
  vattr_vediv    = (1 << 11)
};

class Instr {
public:
  Instr()
    : opcode_(Opcode::NONE)
    , num_rsrcs_(0)
    , has_imm_(false)
    , rdest_type_(RegType::None)
    , imm_(0)
    , rdest_(0)
    , func2_(0)
    , func3_(0)
    , func6_(0)
    , func7_(0)
    , vmask_(0)
    , vlsWidth_(0)
    , vMop_(0)
    , vUmop_(0)
    , vNf_(0)
    , vs3_(0)
    , has_zimm_(false)
    , vlmul_(0)
    , vsew_(0)
    , vta_(0)
    , vma_(0)
    , vediv_(0)
    , vattr_mask_(0) {
    for (uint32_t i = 0; i < MAX_REG_SOURCES; ++i) {
       rsrc_type_[i] = RegType::None;
       rsrc_[i] = 0;
    }
  }

  void setOpcode(Opcode opcode) {
    opcode_ = opcode;
  }

  void setDestReg(uint32_t destReg, RegType type) {
    rdest_type_ = type;
    rdest_ = destReg;
  }

  void addSrcReg(uint32_t srcReg, RegType type) {
    rsrc_type_[num_rsrcs_] = type;
    rsrc_[num_rsrcs_] = srcReg;
    ++num_rsrcs_;
  }

  void setSrcReg(uint32_t index, uint32_t srcReg, RegType type) {
    rsrc_type_[index] = type;
    rsrc_[index] = srcReg;
    num_rsrcs_ = std::max<uint32_t>(num_rsrcs_, index+1);
  }

  void setImm(uint32_t imm) { has_imm_ = true; imm_ = imm; }

  void setFunc2(uint32_t func2) { func2_ = func2; }
  void setFunc3(uint32_t func3) { func3_ = func3; }
  void setFunc6(uint32_t func6) { func6_ = func6; }
  void setFunc7(uint32_t func7) { func7_ = func7; }

  // Attributes for Vector instructions
  void setVlsWidth(uint32_t width) { vlsWidth_ = width; vattr_mask_ |= vattr_vlswidth; }
  void setVmop(uint32_t mop) { vMop_ = mop; vattr_mask_ |= vattr_vmop; }
  void setVumop(uint32_t umop) { vUmop_ = umop; vattr_mask_ |= vattr_vumop; }
  void setVnf(uint32_t nf) { vNf_ = nf; vattr_mask_ |= vattr_vnf; }
  void setVmask(uint32_t mask) { vmask_ = mask; vattr_mask_ |= vattr_vmask; }
  void setVs3(uint32_t vs) { vs3_ = vs; vattr_mask_ |= vattr_vs3; }
  void setZimm(bool has_zimm) { has_zimm_ = has_zimm; vattr_mask_ |= vattr_zimm; }
  void setVlmul(uint32_t lmul) { vlmul_ = lmul; vattr_mask_ |= vattr_vlmul; }
  void setVsew(uint32_t sew) { vsew_ = sew; vattr_mask_ |= vattr_vsew; }
  void setVta(uint32_t vta) { vta_ = vta; vattr_mask_ |= vattr_vta; }
  void setVma(uint32_t vma) { vma_ = vma; vattr_mask_ |= vattr_vma; }
  void setVediv(uint32_t ediv) { vediv_ = 1 << ediv; vattr_mask_ |= vattr_vediv; }

  Opcode   getOpcode() const { return opcode_; }

  uint32_t getNRSrc() const { return num_rsrcs_; }
  uint32_t getRSrc(uint32_t i) const { return rsrc_[i]; }
  RegType  getRSType(uint32_t i) const { return rsrc_type_[i]; }

  uint32_t getRDest() const { return rdest_; }
  RegType  getRDType() const { return rdest_type_; }

  bool     hasImm() const { return has_imm_; }
  uint32_t getImm() const { return imm_; }

  uint32_t getFunc2() const { return func2_; }
  uint32_t getFunc3() const { return func3_; }
  uint32_t getFunc6() const { return func6_; }
  uint32_t getFunc7() const { return func7_; }

  uint32_t getVlsWidth() const { return vlsWidth_; }
  uint32_t getVmop() const { return vMop_; }
  uint32_t getVumop() const { return vUmop_; }
  uint32_t getVnf() const { return vNf_; }
  uint32_t getVmask() const { return vmask_; }
  uint32_t getVs3() const { return vs3_; }
  bool     hasZimm() const { return has_zimm_; }
  uint32_t getVlmul() const { return vlmul_; }
  uint32_t getVsew() const { return vsew_; }
  uint32_t getVta() const { return vta_; }
  uint32_t getVma() const { return vma_; }
  uint32_t getVediv() const { return vediv_; }
  uint32_t getVattrMask() const { return vattr_mask_; }

private:

  enum {
    MAX_REG_SOURCES = 3
  };

  Opcode opcode_;
  uint32_t num_rsrcs_;
  bool has_imm_;
  RegType rdest_type_;
  uint32_t imm_;
  RegType rsrc_type_[MAX_REG_SOURCES];
  uint32_t rsrc_[MAX_REG_SOURCES];
  uint32_t rdest_;
  uint32_t func2_;
  uint32_t func3_;
  uint32_t func6_;
  uint32_t func7_;

  // Vector
  uint32_t vmask_;
  uint32_t vlsWidth_;
  uint32_t vMop_;
  uint32_t vUmop_;
  uint32_t vNf_;
  uint32_t vs3_;
  bool     has_zimm_;
  uint32_t vlmul_;
  uint32_t vsew_;
  uint32_t vta_;
  uint32_t vma_;
  uint32_t vediv_;
  uint32_t vattr_mask_;

  friend std::ostream &operator<<(std::ostream &, const Instr&);
};

}