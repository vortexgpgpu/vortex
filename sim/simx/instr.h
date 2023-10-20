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

class Warp;

enum Opcode {   
  NONE      = 0,    
  R_INST    = 0x33,
  L_INST    = 0x3,
  I_INST    = 0x13,
  S_INST    = 0x23,
  B_INST    = 0x63,
  LUI_INST  = 0x37,
  AUIPC_INST= 0x17,
  JAL_INST  = 0x6f,
  JALR_INST = 0x67,
  SYS_INST  = 0x73,
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
  R_INST_W  = 0x3b,
  I_INST_W  = 0x1b,
  // Vector Extension  
  VSET      = 0x57,
  // Custom Extensions
  EXT1      = 0x0b,
  EXT2      = 0x2b,
  EXT3      = 0x5b,
  EXT4      = 0x7b
};

enum InstType {
  R_TYPE, 
  I_TYPE, 
  S_TYPE, 
  B_TYPE, 
  U_TYPE, 
  J_TYPE,
  V_TYPE,
  R4_TYPE
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
    , vNf_(0)
    , vs3_(0)
    , vlmul_(0)
    , vsew_(0)
    , vediv_(0)   {
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
  void setFunc3(uint32_t func3) { func3_ = func3; }
  void setFunc7(uint32_t func7) { func7_ = func7; }
  void setImm(uint32_t imm) { has_imm_ = true; imm_ = imm; }
  void setVlsWidth(uint32_t width) { vlsWidth_ = width; }
  void setVmop(uint32_t mop) { vMop_ = mop; }
  void setVnf(uint32_t nf) { vNf_ = nf; }
  void setVmask(uint32_t mask) { vmask_ = mask; }
  void setVs3(uint32_t vs) { vs3_ = vs; }
  void setVlmul(uint32_t lmul) { vlmul_ = 1 << lmul; }
  void setVsew(uint32_t sew) { vsew_ = 1 << (3+sew); }
  void setVediv(uint32_t ediv) { vediv_ = 1 << ediv; }
  void setFunc6(uint32_t func6) { func6_ = func6; }

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
  uint32_t getvNf() const { return vNf_; }
  uint32_t getVmask() const { return vmask_; }
  uint32_t getVs3() const { return vs3_; }
  uint32_t getVlmul() const { return vlmul_; }
  uint32_t getVsew() const { return vsew_; }
  uint32_t getVediv() const { return vediv_; }

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
  uint32_t vNf_;
  uint32_t vs3_;
  uint32_t vlmul_;
  uint32_t vsew_;
  uint32_t vediv_;   

  friend std::ostream &operator<<(std::ostream &, const Instr&);
};

}