#pragma once

#include "archdef.h"
#include "types.h"
#include "warp.h"

#include <array>
#include <vector>

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
  // F Extension
  FL        = 0x7,
  FS        = 0x27,
  FCI       = 0x53,
  FMADD     = 0x43,
  FMSUB     = 0x47,
  FMNMSUB   = 0x4b,
  FMNMADD   = 0x4f,
  // Vector Extension  
  VSET      = 0x57,
  VL        = 0x7,
  VS        = 0x27,
  // GPGPU Extension
  GPGPU     = 0x6b,
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
  enum {
    MAX_REG_SOURCES = 3
  };

public:
  Instr(const ArchDef& arch)
    : opcode_(Opcode::NOP)
    , num_rsrcs_(0)
    , has_imm_(false)
    , rdest_(0)
    , func3_(0)
    , func7_(0)
    , rsData(arch.num_threads())
    , rdData(arch.num_threads(), 0)
    , vrdData(arch.vsize(), 0)
  {
    for (int i = 0; i < MAX_REG_SOURCES; ++i) {
       rsrc_type_[i] = 0;
       vrsData[i].resize(arch.vsize(), 0);
       for (int tid = 0; tid < arch.num_threads(); ++tid) {
         rsData[tid][i] = 0;
       }
    }
  }

  /* Setters used to "craft" the instruction. */
  void setOpcode(Opcode opcode)  { opcode_ = opcode; }
  void setDestReg(int destReg) { rdest_type_ = 1; rdest_ = destReg; }
  void setSrcReg(int srcReg) { rsrc_type_[num_rsrcs_] = 1; rsrc_[num_rsrcs_++] = srcReg; }
  void setDestFReg(int destReg) { rdest_type_ = 2; rdest_ = destReg; }
  void setSrcFReg(int srcReg) { rsrc_type_[num_rsrcs_] = 2; rsrc_[num_rsrcs_++] = srcReg;  }
  void setDestVReg(int destReg) { rdest_type_ = 3; rdest_ = destReg; }
  void setSrcVReg(int srcReg) { rsrc_type_[num_rsrcs_] = 3; rsrc_[num_rsrcs_++] = srcReg;  }
  void setFunc3(Word func3) { func3_ = func3; }
  void setFunc7(Word func7) { func7_ = func7; }
  void setImm(Word imm) { has_imm_ = true; imm_ = imm; }
  void setVlsWidth(Word width) { vlsWidth_ = width; }
  void setVmop(Word mop) { vMop_ = mop; }
  void setVnf(Word nf) { vNf_ = nf; }
  void setVmask(Word mask) { vmask_ = mask; }
  void setVs3(Word vs) { vs3_ = vs; }
  void setVlmul(Word lmul) { vlmul_ = 1 << lmul; }
  void setVsew(Word sew) { vsew_ = 1 << (3+sew); }
  void setVediv(Word ediv) { vediv_ = 1 << ediv; }
  void setFunc6(Word func6) { func6_ = func6; }

  // tid - thread id
  void setRSData(Word data, const int tid, const int num_rs) {
    rsData[tid][num_rs] = data;
  }
  void setRDData(Word data, const int tid) {
    rdData[tid] = data;
  }
  void setVRSData(const std::vector<Byte>& data, const int num_rs) {
    vrsData[num_rs] = data;
  }
  void setVRDData(const std::vector<Byte>& data) {
    vrdData = data;
  }

  /* Getters used by encoders. */
  Opcode getOpcode() const { return opcode_; }
  Word getFunc3() const { return func3_; }
  Word getFunc6() const { return func6_; }
  Word getFunc7() const { return func7_; }
  int getNRSrc() const { return num_rsrcs_; }
  int getRSrc(int i) const { return rsrc_[i]; }
  int getRSType(int i) const { return rsrc_type_[i]; }
  int getRDest() const { return rdest_; }  
  int getRDType() const { return rdest_type_; }  
  bool hasImm() const { return has_imm_; }
  Word getImm() const { return imm_; }
  Word getVlsWidth() const { return vlsWidth_; }
  Word getVmop() const { return vMop_; }
  Word getvNf() const { return vNf_; }
  Word getVmask() const { return vmask_; }
  Word getVs3() const { return vs3_; }
  Word getVlmul() const { return vlmul_; }
  Word getVsew() const { return vsew_; }
  Word getVediv() const { return vediv_; }

  std::array<Word, MAX_REG_SOURCES> getRSData(int tid) {
    return rsData[tid];
  }
  std::array<std::vector<Byte>, MAX_REG_SOURCES> getVRSData() {
    return vrsData;
  }
  Word getRDData(int tid) const {
    return rdData.at(tid);
  }
  Word& getRDData(int tid) {
    return rdData[tid];
  }
  std::vector<Byte> getVRDData() const {
    return vrdData;
  }
  std::vector<Byte>& getVRDData() {
    return vrdData;
  }

private:
  Opcode opcode_;
  int num_rsrcs_;
  bool has_imm_;
  int rdest_type_;
  int isrc_mask_;
  int fsrc_mask_;  
  int vsrc_mask_;
  Word imm_;
  int rsrc_type_[MAX_REG_SOURCES];
  int rsrc_[MAX_REG_SOURCES];  
  int rdest_;
  Word func3_;
  Word func7_;

  // Vector
  Word vmask_;
  Word vlsWidth_;
  Word vMop_;
  Word vNf_;
  Word vs3_;
  Word vlmul_;
  Word vsew_;
  Word vediv_;
  Word func6_;

  // Src and dst values
  // Warp Source Registers
  std::vector<std::array<Word, MAX_REG_SOURCES>> rsData;
  // Warp Destination Registers
  std::vector<Word> rdData;
  // For vector instr
  // Warp Vector Source Registers
  std::array<std::vector<Byte>, MAX_REG_SOURCES> vrsData;
  // Warp Vector Destination Registers
  std::vector<Byte> vrdData;

  friend std::ostream &operator<<(std::ostream &, const Instr&);
};

}