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
  
  std::shared_ptr<Instr> decode(HalfWord code, HalfWord PC);

private:

  HalfWord inst_s_;
  HalfWord opcode_s_;
  HalfWord reg_s_;
  HalfWord func2_s_;
  HalfWord func3_s_;
  HalfWord shift_opcode_;
  HalfWord shift_rd_;
  HalfWord shift_rs1_;
  HalfWord shift_rs2_;
  HalfWord shift_rs3_;
  HalfWord shift_func2_;
  HalfWord shift_func3_;
  HalfWord shift_func7_;
  HalfWord shift_j_u_immed_;
  HalfWord shift_s_b_immed_;
  HalfWord shift_i_immed_;

  HalfWord reg_mask_;
  HalfWord func2_mask_;
  HalfWord func3_mask_;
  HalfWord func6_mask_;
  HalfWord func7_mask_;
  HalfWord opcode_mask_;
  HalfWord i_imm_mask_;
  HalfWord s_imm_mask_;
  HalfWord b_imm_mask_;
  HalfWord u_imm_mask_;
  HalfWord j_imm_mask_;
  HalfWord v_imm_mask_;

  //Vector
  HalfWord shift_vset_;
  HalfWord shift_vset_immed_;
  HalfWord shift_vmask_;
  HalfWord shift_vmop_;
  HalfWord shift_vnf_;
  HalfWord shift_func6_;
  HalfWord vmask_s_;
  HalfWord mop_s_;
};

}