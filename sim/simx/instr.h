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

enum set_vuse_mask {
  set_func3 = (1 << 0),
  set_func6 = (1 << 1),
  set_imm = (1 << 2),
  set_vlswidth = (1 << 3),
  set_vmop = (1 << 4),
  set_vumop = (1 << 5),
  set_vnf = (1 << 6),
  set_vmask = (1 << 7),
  set_vs3 = (1 << 8),
  set_zimm = (1 << 9),
  set_vlmul = (1 << 10),
  set_vsew = (1 << 11),
  set_vta = (1 << 12),
  set_vma = (1 << 13),
  set_vediv = (1 << 14)
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
    , _vusemask(0)
    , _is_vec(false)   {
    for (uint32_t i = 0; i < MAX_REG_SOURCES; ++i) {
       rsrc_type_[i] = RegType::None;
       rsrc_[i] = 0;
    }
  }

  void setOpcode(Opcode opcode)  { opcode_ = opcode; }
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
  void setFunc2(uint32_t func2) { func2_ = func2; }
  void setFunc3(uint32_t func3) { func3_ = func3; _vusemask |= set_func3; }
  void setFunc6(uint32_t func6) { func6_ = func6; _vusemask |= set_func6; }
  void setFunc7(uint32_t func7) { func7_ = func7; }
  void setImm(uint32_t imm) { has_imm_ = true; imm_ = imm; _vusemask |= set_imm; }
  void setVlsWidth(uint32_t width) { vlsWidth_ = width; _vusemask |= set_vlswidth; }
  void setVmop(uint32_t mop) { vMop_ = mop; _vusemask |= set_vmop; }
  void setVumop(uint32_t umop) { vUmop_ = umop; _vusemask |= set_vumop; }
  void setVnf(uint32_t nf) { vNf_ = nf; _vusemask |= set_vnf; }
  void setVmask(uint32_t mask) { vmask_ = mask; _vusemask |= set_vmask; }
  void setVs3(uint32_t vs) { vs3_ = vs; _vusemask |= set_vs3; }
  void setZimm(bool has_zimm) { has_zimm_ = has_zimm; _vusemask |= set_zimm; }
  void setVlmul(uint32_t lmul) { vlmul_ = lmul; _vusemask |= set_vlmul; }
  void setVsew(uint32_t sew) { vsew_ = sew; _vusemask |= set_vsew; }
  void setVta(uint32_t vta) { vta_ = vta; _vusemask |= set_vta; }
  void setVma(uint32_t vma) { vma_ = vma; _vusemask |= set_vma; }
  void setVediv(uint32_t ediv) { vediv_ = 1 << ediv; _vusemask |= set_vediv; }
  void setVec(bool is_vec) { _is_vec = is_vec; }

  Opcode   getOpcode() const { return opcode_; }
  uint32_t getFunc2() const { return func2_; }
  uint32_t getFunc3() const { return func3_; }
  uint32_t getFunc6() const { return func6_; }
  uint32_t getFunc7() const { return func7_; }
  uint32_t getNRSrc() const { return num_rsrcs_; }
  uint32_t getRSrc(uint32_t i) const { return rsrc_[i]; }
  RegType  getRSType(uint32_t i) const { return rsrc_type_[i]; }
  uint32_t getRDest() const { return rdest_; }  
  RegType  getRDType() const { return rdest_type_; }  
  bool     hasImm() const { return has_imm_; }
  uint32_t getImm() const { return imm_; }
  uint32_t getVlsWidth() const { return vlsWidth_; }
  uint32_t getVmop() const { return vMop_; }
  uint32_t getVumop() const { return vUmop_; }
  uint32_t getVnf() const { return vNf_; }
  uint32_t getVmask() const { return vmask_; }
  uint32_t getVs3() const { return vs3_; }
  bool     hasZimm() const { return has_zimm_; }
  uint32_t getVlmul() const { return vlmul_; }
  uint32_t getVsew() const { return 1 << (3 + vsew_); }
  uint32_t getVsewO() const { return vsew_; }
  uint32_t getVta() const { return vta_; }
  uint32_t getVma() const { return vma_; }
  uint32_t getVediv() const { return vediv_; }
  uint32_t getVUseMask() const { return _vusemask; }
  bool     isVec() const { return _is_vec; }

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
  uint32_t _vusemask;
  bool     _is_vec;

  friend std::ostream &operator<<(std::ostream &, const Instr&);
};

}