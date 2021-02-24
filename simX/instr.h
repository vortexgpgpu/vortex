#pragma once

#include "types.h"
#include "trace.h"

namespace vortex {

class Warp;

enum Opcode {   
  NOP       = 0,    
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
  PJ_INST   = 0x7b,
  GPGPU     = 0x6b,
  VSET_ARITH= 0x57,
  VL        = 0x7,
  VS        = 0x27,
  // F-Extension
  FL        = 0x7,
  FS        = 0x27,
  FCI       = 0x53,
  FMADD     = 0x43,
  FMSUB     = 0x47,
  FMNMSUB   = 0x4b,
  FMNMADD   = 0x4f,
};

enum InstType { 
  N_TYPE, 
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
    : opcode_(Opcode::NOP)
    , nRsrc_(0)
    , nPsrc_(0)
    , hasImmSrc_(false)
    , hasRDest_(false)
    , func2_(0)
    , func3_(0)
    , func7_(0)
  {}


  friend std::ostream &operator<<(std::ostream &, Instr &);

  /* Setters used to "craft" the instruction. */
  void setOpcode(Opcode opcode)  { opcode_ = opcode; }
  void setDestReg(RegNum destReg) { hasRDest_ = true; rdest_ = destReg; }
  void setSrcReg(RegNum srcReg) { rsrc_[nRsrc_++] = srcReg; }
  void setFunc3(Word func3) { func3_ = func3; }
  void setFunc7(Word func7) { func7_ = func7; }
  void setSrcImm(Word srcImm) { hasImmSrc_ = true; immsrc_ = srcImm; }
  void setVsetImm(Word vset_imm) { if(vset_imm) vsetImm_ = true; else vsetImm_ = false; }
  void setVlsWidth(Word width) { vlsWidth_ = width; }
  void setVmop(Word mop) { vMop_ = mop; }
  void setVnf(Word nf) { vNf_ = nf; }
  void setVmask(Word mask) { vmask_ = mask; }
  void setVs3(Word vs) { vs3_ = vs; }
  void setVlmul(Word lmul);
  void setVsew(Word sew);
  void setVediv(Word ediv);
  void setFunc6(Word func6) { func6_ = func6; }

  /* Getters used by encoders. */
  Opcode getOpcode() const { return opcode_; }
  Word getFunc3() const { return func3_; }
  Word getFunc6() const { return func6_; }
  Word getFunc7() const { return func7_; }
  RegNum getNRSrc() const { return nRsrc_; }
  RegNum getRSrc(RegNum i) const { return rsrc_[i]; }
  bool hasRDest() const { return hasRDest_; }
  RegNum getRDest() const { return rdest_; }
  bool hasImm() const { return hasImmSrc_; }
  Word getImm() const { return immsrc_; }
  bool getVsetImm() const { return vsetImm_; }
  Word getVlsWidth() const { return vlsWidth_; }
  Word getVmop() const { return vMop_; }
  Word getvNf() const { return vNf_; }
  bool getVmask() const { return vmask_; }
  Word getVs3() const { return vs3_; }
  Word getVlmul() const { return vlmul_; }
  Word getVsew() const { return vsew_; }
  Word getVediv() const { return vediv_; }

private:

  enum {
    MAX_REG_SOURCES = 3
  };

  Opcode opcode_;
  int nRsrc_;
  int nPsrc_;
  bool hasImmSrc_;
  bool hasRDest_;
  Word immsrc_;
  Word func2_;
  Word func3_;
  Word func7_;
  RegNum rsrc_[MAX_REG_SOURCES];
  RegNum rdest_;

  //Vector
  bool vsetImm_;
  bool vmask_;
  Word vlsWidth_;
  Word vMop_;
  Word vNf_;
  Word vs3_;
  Word vlmul_;
  Word vsew_;
  Word vediv_;
  Word func6_;
};

std::ostream &operator<<(std::ostream &, Instr &);

}