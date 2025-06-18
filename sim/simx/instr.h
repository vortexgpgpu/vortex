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

enum class Opcode : uint8_t {
  NONE      = 0b0000000,
  R         = 0b0110011,
  L         = 0b0000011,
  I         = 0b0010011,
  S         = 0b0100011,
  B         = 0b1100011,
  LUI       = 0b0110111,
  AUIPC     = 0b0010111,
  JAL       = 0b1101111,
  JALR      = 0b1100111,
  SYS       = 0b1110011,
  FENCE     = 0b0001111,
  AMO       = 0b0101111,
  // F Extension
  FL        = 0b0000111,
  FS        = 0b0100111,
  FCI       = 0b1010011,
  FMADD     = 0b1000011,
  FMSUB     = 0b1000111,
  FNMSUB    = 0b1001011,
  FNMADD    = 0b1001111,
  // RV64 Standard Extension
  R_W       = 0b0111011,
  I_W       = 0b0011011,
  // Vector Extension
  VSET      = 0b1010111,
  // Custom Extensions
  EXT1      = 0b0001011,
  EXT2      = 0b0101011,
  EXT3      = 0b1011011,
  EXT4      = 0b1111011
};;

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
  width_funct5= 5,
  width_funct6= 6,
  width_funct7= 7,
  width_i_imm = 12,
  width_j_imm = 20,
  width_vmop  = 2,
  width_vmew  = 1,
  width_vnf   = 3,
  width_vm    = 1,
  width_vzimm = 11,
  width_vma   = 1,
  width_vta   = 1,
  width_vsew  = 3,
  width_vlmul = 3,
  width_aq    = 1,
  width_rl    = 1,

  shift_opcode= 0,
  shift_rd    = width_opcode,
  shift_funct3= shift_rd + width_reg,
  shift_rs1   = shift_funct3 + width_funct3,
  shift_rs2   = shift_rs1 + width_reg,
  shift_funct2= shift_rs2 + width_reg,
  shift_funct5= shift_funct2 + width_funct2,
  shift_funct7= shift_funct2,
  shift_rl    = shift_funct2,
  shift_aq    = shift_rl + width_rl,
  shift_rs3   = shift_funct7 + width_funct2,
  shift_vm    = shift_funct7,
  shift_vmop  = shift_funct7 + width_vm,
  shift_vmew  = shift_vmop + width_vmop,
  shift_vnf   = shift_vmew + width_vmew,
  shift_funct6= shift_funct7 + width_vm,
  shift_vset  = shift_funct7 + width_funct6,
  shift_vsew  = width_vlmul,
  shift_vta   = shift_vsew + width_vsew,
  shift_vma   = shift_vta + width_vta,
  shift_vzimm = shift_rs2,

  mask_opcode = (1 << width_opcode) - 1,
  mask_reg    = (1 << width_reg)   - 1,
  mask_funct2 = (1 << width_funct2) - 1,
  mask_funct3 = (1 << width_funct3) - 1,
  mask_funct5 = (1 << width_funct5) - 1,
  mask_funct6 = (1 << width_funct6) - 1,
  mask_funct7 = (1 << width_funct7) - 1,
  mask_aq     = (1 << width_aq) - 1,
  mask_rl     = (1 << width_rl) - 1,
  mask_i_imm  = (1 << width_i_imm) - 1,
  mask_j_imm  = (1 << width_j_imm) - 1,
  mask_vmop   = (1 << width_vmop) - 1,
  mask_vmew   = (1 << width_vmew) - 1,
  mask_vnf    = (1 << width_vnf) - 1,
  mask_vm     = (1 << width_vm) - 1,
  mask_vzimm  = (1 << width_vzimm) - 1,
  mask_vma    = (1 << width_vma) - 1,
  mask_vta    = (1 << width_vta) - 1,
  mask_vsew   = (1 << width_vsew) - 1,
  mask_vlmul  = (1 << width_vlmul) - 1,
};

class Instr {
public:
  using Ptr = std::shared_ptr<Instr>;

  enum {
    MAX_REG_SOURCES = 3
  };

  Instr(uint64_t uuid, FUType fu_type = FUType::ALU)
    : uuid_(uuid)
    , fu_type_(fu_type)
  {}

  void setFUType(FUType fu_type) {
    fu_type_ = fu_type;
  }

  template <typename T> void setOpType(T op_type) {
    op_type_ = static_cast<T>(op_type);
  }

  template <typename T> void setArgs(T args) {
    args_ = static_cast<T>(args);
  }

  void setDestReg(uint32_t destReg, RegType type) {
    rdest_ = {type, destReg };
  }

  void setSrcReg(uint32_t index, uint32_t srcReg, RegType type) {
    rsrc_[index] = { type, srcReg};
  }

  FUType getFUType() const { return fu_type_; }

  OpType getOpType() const { return op_type_; }

  const IntrArgs& getArgs() const { return args_; }

  RegOpd getSrcReg(uint32_t i) const { return rsrc_[i]; }

  RegOpd getDestReg() const { return rdest_; }

  uint64_t getUUID() const { return uuid_; }

private:

  uint64_t uuid_;
  FUType   fu_type_;
  OpType   op_type_;
  IntrArgs args_;
  RegOpd   rsrc_[MAX_REG_SOURCES];
  RegOpd   rdest_;

  friend std::ostream &operator<<(std::ostream &, const Instr &);
};

}