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

#include <vector>
#include <memory>

namespace vortex {

class Arch;
class Instr;

class Decoder {
public:
  Decoder(const Arch &);    
  
  std::shared_ptr<Instr> decode(uint32_t code) const;
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
  mask_v_lmul  = (1 << width_v_lmul) - 1,
};

}