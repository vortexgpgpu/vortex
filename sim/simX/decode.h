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
  
  std::shared_ptr<Instr> decode(Word code, Word PC);

private:

  Word inst_s_;
  Word opcode_s_;
  Word reg_s_;
  Word func2_s_;
  Word func3_s_;
  Word shift_opcode_;
  Word shift_rd_;
  Word shift_rs1_;
  Word shift_rs2_;
  Word shift_rs3_;
  Word shift_func2_;
  Word shift_func3_;
  Word shift_func7_;
  Word shift_j_u_immed_;
  Word shift_s_b_immed_;
  Word shift_i_immed_;

  Word reg_mask_;
  Word func2_mask_;
  Word func3_mask_;
  Word func6_mask_;
  Word func7_mask_;
  Word opcode_mask_;
  Word i_imm_mask_;
  Word s_imm_mask_;
  Word b_imm_mask_;
  Word u_imm_mask_;
  Word j_imm_mask_;
  Word v_imm_mask_;

  //Vector
  Word shift_vset_;
  Word shift_vset_immed_;
  Word shift_vmask_;
  Word shift_vmop_;
  Word shift_vnf_;
  Word shift_func6_;
  Word vmask_s_;
  Word mop_s_;
};

}