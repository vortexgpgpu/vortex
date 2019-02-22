/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#ifndef __INSTRUCTION_H
#define __INSTRUCTION_H

#include <map>
#include <iostream>

#include "types.h"

namespace Harp {
  class Warp;
  class Ref;

  enum Opcode
  {   
      NOP = 0,    
      R_INST = 51,
      L_INST = 3,
      I_INST = 19,
      S_INST = 35,
      B_INST = 99,
      LUI_INST = 55,
      AUIPC_INST = 23,
      JAL_INST = 111,
      JALR_INST = 103,
      SYS_INST = 115,
      TRAP     = 0x7f,
      FENCE    = 0x0f,
      PJ_INST  = 0x7b,
      GPGPU    = 0x6b
   };

  enum InstType { N_TYPE, R_TYPE, I_TYPE, S_TYPE, B_TYPE, U_TYPE, J_TYPE};

  // We build a table of instruction information out of this.
  struct InstTableEntry_t {
    const char *opString;
    bool controlFlow, relAddress, allSrcArgs, privileged;
    InstType iType;

  };

  static std::map<int, struct InstTableEntry_t> instTable = 
  {
    {Opcode::NOP,        {"nop"   , false, false, false, false, InstType::N_TYPE }},
    {Opcode::R_INST,     {"r_type", false, false, false, false, InstType::R_TYPE }},
    {Opcode::L_INST,     {"load"  , false, false, false, false, InstType::I_TYPE }},
    {Opcode::I_INST,     {"i_type", false, false, false, false, InstType::I_TYPE }},
    {Opcode::S_INST,     {"store" , false, false, false, false, InstType::S_TYPE }},
    {Opcode::B_INST,     {"branch", true , false, false, false, InstType::B_TYPE }},
    {Opcode::LUI_INST,   {"lui"   , false, false, false, false, InstType::U_TYPE }},
    {Opcode::AUIPC_INST, {"auipc" , false, false, false, false, InstType::U_TYPE }},
    {Opcode::JAL_INST,   {"jal"   , true , false, false, false, InstType::J_TYPE }},
    {Opcode::JALR_INST,  {"jalr"  , true , false, false, false, InstType::I_TYPE }},
    {Opcode::SYS_INST,   {"SYS"   , true , false, false, false, InstType::I_TYPE }},
    {Opcode::TRAP,       {"TRAP"  , true , false, false, false, InstType::I_TYPE }},
    {Opcode::FENCE,      {"fence" , true , false, false, false, InstType::I_TYPE }},
    {Opcode::PJ_INST,    {"pred j", true , false, false, false, InstType::R_TYPE }},
    {Opcode::GPGPU,      {"gpgpu" , false, false, false, false, InstType::R_TYPE }}
  };

  static const Size MAX_REG_SOURCES(3);
  static const Size MAX_PRED_SOURCES(2);

  class Instruction;

  struct DivergentBranchException {};
  struct DomainException {};

  std::ostream &operator<<(std::ostream &, Instruction &);

  class Instruction {
  public:
    Instruction() : 
      predicated(false), nRsrc(0), nPsrc(0), immsrcPresent(false), 
      rdestPresent(false), pdestPresent(false), refLiteral(NULL)
      {
      }

    void executeOn(Warp &warp);
    friend std::ostream &operator<<(std::ostream &, Instruction &);

    /* Setters used to "craft" the instruction. */
    void  setOpcode  (Opcode opc)  { op = opc; }
    void  setPred    (RegNum pReg) { predicated = true; pred = pReg; }
    void  setDestReg (RegNum destReg) { rdestPresent = true; rdest = destReg; }
    void  setSrcReg  (RegNum srcReg) { rsrc[nRsrc++] = srcReg; }
    void  setFunc3  (Word func3) { this->func3 = func3; }
    void  setFunc7  (Word func7) { this->func7 = func7; }
    void  setDestPReg(RegNum dPReg) { pdestPresent = true; pdest = dPReg; }
    void  setSrcPReg (RegNum srcPReg) { psrc[nPsrc++] = srcPReg; }
    Word *setSrcImm  () { immsrcPresent = true; immsrc = 0xa5; return &immsrc;}
    void  setSrcImm  (Word srcImm) { immsrcPresent = true; immsrc = srcImm; }
    void  setImmRef  (Ref &r) { refLiteral = &r; }

    /* Getters used by encoders. */
    Opcode getOpcode() const { return op; }
    bool hasPred() const { return predicated; }
    RegNum getPred() const { return pred; }
    RegNum getNRSrc() const { return nRsrc; }
    RegNum getRSrc(RegNum i) const { return rsrc[i]; }
    RegNum getNPSrc() const { return nPsrc; }
    RegNum getPSrc(RegNum i) const { return psrc[i]; }
    bool hasRDest() const { return rdestPresent; }
    RegNum getRDest() const { return rdest; }
    bool hasPDest() const { return pdestPresent; }
    RegNum getPDest() const { return pdest; }
    bool hasImm() const { return immsrcPresent; }
    Word getImm() const { return immsrc; }
    bool hasRefLiteral() const { return refLiteral != NULL; }
    Ref *getRefLiteral() const { return refLiteral; }

    /* Getters used as table lookup. */
    bool hasRelImm() const { return (*(instTable.find(op))).second.relAddress; }

  private:
    bool predicated;
    RegNum pred;
    Opcode op;
    int nRsrc, nPsrc;
    RegNum rsrc[MAX_REG_SOURCES], psrc[MAX_PRED_SOURCES];
    bool immsrcPresent;
    Word immsrc;
    Word func3;
    Word func7;
    bool rdestPresent, pdestPresent;
    RegNum rdest, pdest;
    Ref *refLiteral;

  public:
    

  };
};

#endif

    // static struct InstTableEntry {
    //   const char *opString;
    //   bool controlFlow, relAddress, allSrcArgs, privileged;
    //   InstType iType;
    // };