/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#ifndef __INSTRUCTION_H
#define __INSTRUCTION_H

#include <iostream>

#include "types.h"

namespace Harp {
  class Core;
  class Ref;

  static const Size MAX_REG_SOURCES(3);
  static const Size MAX_PRED_SOURCES(2);

  class Instruction;

  struct DivergentBranchException {};
  struct DomainException {};

  std::ostream &operator<<(std::ostream &, Instruction &);

  class Instruction {
  public:
    enum Opcode { NOP, DI, EI, TLBADD, TLBFLUSH, NEG, NOT, 
                  AND, OR, XOR, ADD, SUB, MUL, DIV, MOD, SHL, SHR,
                  ANDI, ORI, XORI, ADDI, SUBI, MULI, DIVI, MODI, SHLI, SHRI,
                  JALI, JALR, JMPI, JMPR, CLONE, JALIS, JALRS, 
                  JMPRT, LD, ST, LDI, RTOP, ANDP, ORP, XORP, NOTP, ISNEG, 
                  ISZERO, HALT, TRAP, JMPRU, SKEP, RETI, TLBRM,
                  ITOF, FTOI, FADD, FSUB, FMUL, FDIV, FNEG };
    enum ArgClass {
      AC_NONE, AC_2REG, AC_2IMM, AC_3REG, AC_3PREG, AC_3IMM, AC_3REGSRC, 
      AC_1IMM, AC_1REG, AC_3IMMSRC, AC_PREG_REG, AC_2PREG
    };

    // We build a table of instruction information out of this.
    static struct InstTableEntry {
      const char *opString;
      bool controlFlow, relAddress, allSrcArgs, privileged;
      ArgClass argClass;
    } instTable[];

    Instruction() : 
      predicated(false), nRsrc(0), nPsrc(0), immsrcPresent(false), 
      rdestPresent(false), pdestPresent(false), refLiteral(NULL) {}

    void executeOn(Core &core);
    friend std::ostream &operator<<(std::ostream &, Instruction &);

    /* Setters used to "craft" the instruction. */
    void  setOpcode  (Opcode opc)  { op = opc; }
    void  setPred    (RegNum pReg) { predicated = true; pred = pReg; }
    void  setDestReg (RegNum destReg) { rdestPresent = true; rdest = destReg; }
    void  setSrcReg  (RegNum srcReg) { rsrc[nRsrc++] = srcReg; }
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
    bool hasRelImm() const { return instTable[op].relAddress; }

  private:
    bool predicated;
    RegNum pred;
    Opcode op;
    int nRsrc, nPsrc;
    RegNum rsrc[MAX_REG_SOURCES], psrc[MAX_PRED_SOURCES];
    bool immsrcPresent;
    Word immsrc;
    bool rdestPresent, pdestPresent;
    RegNum rdest, pdest;
    Ref *refLiteral;
  };
};

#endif
