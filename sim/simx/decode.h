#pragma once

#include <vector>
#include <memory>

namespace vortex {

class ArchDef;
class Instr;
class Pipeline;

class Decoder {
public:
  Decoder(const ArchDef &);    
  
  std::shared_ptr<Instr> decode(uint32_t code) const;

private:

  uint32_t inst_s_;
  uint32_t opcode_s_;
  uint32_t reg_s_;
  uint32_t func2_s_;
  uint32_t func3_s_;
  uint32_t shift_opcode_;
  uint32_t shift_rd_;
  uint32_t shift_rs1_;
  uint32_t shift_rs2_;
  uint32_t shift_rs3_;
  uint32_t shift_func2_;
  uint32_t shift_func3_;
  uint32_t shift_func7_;
  uint32_t shift_j_u_immed_;
  uint32_t shift_s_b_immed_;
  uint32_t shift_i_immed_;

  uint32_t reg_mask_;
  uint32_t func2_mask_;
  uint32_t func3_mask_;
  uint32_t func6_mask_;
  uint32_t func7_mask_;
  uint32_t opcode_mask_;
  uint32_t i_imm_mask_;
  uint32_t s_imm_mask_;
  uint32_t b_imm_mask_;
  uint32_t u_imm_mask_;
  uint32_t j_imm_mask_;
  uint32_t v_imm_mask_;

  //Vector
  uint32_t shift_vset_;
  uint32_t shift_vset_immed_;
  uint32_t shift_vmask_;
  uint32_t shift_vmop_;
  uint32_t shift_vnf_;
  uint32_t shift_func6_;
  uint32_t vmask_s_;
  uint32_t mop_s_;
};

}