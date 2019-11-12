/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#include <iostream>
#include <string>
#include <stdlib.h>
#include <string.h>
#include <iomanip>
#include <vector>

#include "include/debug.h"
#include "include/types.h"
#include "include/util.h"
#include "include/enc.h"
#include "include/archdef.h"
#include "include/instruction.h"

using namespace std;
using namespace Harp;

// ByteDecoder::ByteDecoder(const ArchDef &ad) {
//   wordSize = ad.getWordSize();
// }

static void decodeError(string msg) {
  cout << "Instruction decoder error: " << msg << '\n';
  exit(1);
}



static unsigned ceilLog2(RegNum x) {
  unsigned z = 0;
  bool nonZeroInnerValues(false);

  if (x == 0) return 0;

  while (x != 1) {
    z++;
    if (x&1) nonZeroInnerValues = true;
    x >>= 1;
  }

  if (nonZeroInnerValues) z++;

  return z;
}


WordDecoder::WordDecoder(const ArchDef &arch) {

    inst_s   = arch.getWordSize() * 8;
    opcode_s = 7;
    reg_s    = 5;
    func3_s  = 3;

    shift_opcode    = 0;
    shift_rd        = opcode_s;
    shift_func3     = opcode_s + reg_s;
    shift_rs1       = opcode_s + reg_s + func3_s;
    shift_rs2       = opcode_s + reg_s + func3_s + reg_s;
    shift_func7     = opcode_s + reg_s + func3_s + reg_s + reg_s;
    shift_j_u_immed = opcode_s + reg_s;
    shift_s_b_immed = opcode_s + reg_s + func3_s + reg_s + reg_s;
    shift_i_immed   = opcode_s + reg_s + func3_s + reg_s;

    reg_mask     = 0x1f;
    func3_mask   = 0x7;
    func7_mask   = 0x7f;
    opcode_mask  = 0x7f;
    i_immed_mask = 0xfff;
    s_immed_mask = 0xfff;
    b_immed_mask = 0x1fff;
    u_immed_mask = 0xfffff;
    j_immed_mask = 0xfffff;

}

static Word signExt(Word w, Size bit, Word mask) {
  if (w>>(bit-1)) w |= ~mask;
  return w;
}

Instruction *WordDecoder::decode(const std::vector<Byte> &v, Size &idx, trace_inst_t * trace_inst) {
  Word code(readWord(v, idx, inst_s/8));

  // std::cout << "code: " << (int) code << "  v: " << v << " indx: " << idx << "\n";
  

  Instruction &inst = * new Instruction();  

  // bool predicated = (code>>(n-1));
  bool predicated = false;
  if (predicated) { inst.setPred((code>>(inst_s-p-1))&pMask); }

  printf("CUrrent CODE: %x\n", code);

  Opcode op = (Opcode)((code>>shift_opcode)&opcode_mask);
  // std::cout << "opcode: " << op << "\n";
  inst.setOpcode(op);

  bool usedImm(false);
  Word imeed, dest_bits, imm_bits, bit_11, bits_4_1, bit_10_5,
          bit_12, bits_19_12, bits_10_1, bit_20, unordered, func3;

  // std::cout << "op: " << std::hex << op << " what " << instTable[op].iType << "\n";
  switch(instTable[op].iType)
  {
    case InstType::N_TYPE:
      break;
    case InstType::R_TYPE:
      inst.setPred((code>>shift_rs1)   & reg_mask);
      inst.setDestReg((code>>shift_rd)   & reg_mask);
      inst.setSrcReg((code>>shift_rs1)   & reg_mask);
      inst.setSrcReg((code>>shift_rs2)   & reg_mask);
      inst.setFunc3 ((code>>shift_func3) & func3_mask);
      inst.setFunc7 ((code>>shift_func7) & func7_mask);

      trace_inst->valid_inst = true;
      trace_inst->rs1        = ((code>>shift_rs1)   & reg_mask);
      trace_inst->rs2        = ((code>>shift_rs2)   & reg_mask);
      trace_inst->rd         = ((code>>shift_rd)    & reg_mask);

      break;
    case InstType::I_TYPE:
      inst.setDestReg((code>>shift_rd)   & reg_mask);
      inst.setSrcReg((code>>shift_rs1)   & reg_mask);
      inst.setFunc7 ((code>>shift_func7) & func7_mask);
      func3 = (code>>shift_func3) & func3_mask;
      inst.setFunc3 (func3);

      if ((func3 == 5) && (op != L_INST))
      {
        // std::cout << "func7: " << func7 << "\n";
        inst.setSrcImm(signExt(((code>>shift_rs2)&reg_mask), 5, reg_mask));
      }
      else
      {
        inst.setSrcImm(signExt(code>>shift_i_immed, 12, i_immed_mask));
      }
      usedImm = true;

      trace_inst->valid_inst = true;
      trace_inst->rs1        = ((code>>shift_rs1)   & reg_mask);
      trace_inst->rd         = ((code>>shift_rd)    & reg_mask);

      break;
    case InstType::S_TYPE:
      // std::cout << "************STORE\n";
      inst.setSrcReg((code>>shift_rs1)   & reg_mask);
      inst.setSrcReg((code>>shift_rs2)   & reg_mask);
      inst.setFunc3 ((code>>shift_func3) & func3_mask);

      dest_bits = (code>>shift_rd)  & reg_mask;
      imm_bits  = (code>>shift_s_b_immed & func7_mask);
      imeed     = (imm_bits << reg_s) | dest_bits;
      // std::cout << "ENC: store imeed: " << imeed << "\n";
      inst.setSrcImm(signExt(imeed, 12, s_immed_mask));
      usedImm = true;

      trace_inst->valid_inst = true;
      trace_inst->rs1        = ((code>>shift_rs1)   & reg_mask);
      trace_inst->rs2        = ((code>>shift_rs2)   & reg_mask);

      break;
    case InstType::B_TYPE:

      inst.setSrcReg((code>>shift_rs1)   & reg_mask);
      inst.setSrcReg((code>>shift_rs2)   & reg_mask);
      inst.setFunc3 ((code>>shift_func3) & func3_mask);

      dest_bits = (code>>shift_rd)  & reg_mask;
      imm_bits  = (code>>shift_s_b_immed & func7_mask);

      bit_11   = dest_bits & 0x1;
      bits_4_1 = dest_bits >> 1;
      bit_10_5 = imm_bits & 0x3f;
      bit_12   = imm_bits >> 6;

      imeed    = 0 | (bits_4_1 << 1) | (bit_10_5 << 5) | (bit_11 << 11) | (bit_12 << 12);

      inst.setSrcImm(signExt(imeed, 13, b_immed_mask));
      usedImm = true;

      trace_inst->valid_inst = true;
      trace_inst->rs1        = ((code>>shift_rs1)   & reg_mask);
      trace_inst->rs2        = ((code>>shift_rs2)   & reg_mask);

      break;
    case InstType::U_TYPE:
      inst.setDestReg((code>>shift_rd)   & reg_mask);
      inst.setSrcImm(signExt(code>>shift_j_u_immed, 20, u_immed_mask));
      usedImm = true;
      trace_inst->valid_inst = true;
      trace_inst->rd         = ((code>>shift_rd)    & reg_mask);
      break;
    case InstType::J_TYPE:
      inst.setDestReg((code>>shift_rd)   & reg_mask);

      // [20 | 10:1 | 11 | 19:12]

      unordered = code>>shift_j_u_immed;

      bits_19_12 = unordered & 0xff;
      bit_11     = (unordered>>8) & 0x1;
      bits_10_1  = (unordered >> 9) & 0x3ff;
      bit_20     = (unordered>>19) & 0x1;

      imeed  = 0 | (bits_10_1 << 1) | (bit_11 << 11) | (bits_19_12 << 12) | (bit_20 << 20);

      inst.setSrcImm(signExt(imeed, 20, j_immed_mask));
      usedImm = true;

      trace_inst->valid_inst = true;
      trace_inst->rd         = ((code>>shift_rd)    & reg_mask);

      break;
   defualt:
      cout << "Unrecognized argument class in word decoder.\n";
      exit(1);
  }

  if (haveRefs && usedImm && refMap.find(idx-n/8) != refMap.end()) {
    Ref *srcRef = refMap[idx-n/8];

    /* Create a new ref tied to this instruction. */
    Ref *r = new SimpleRef(srcRef->name, *(Addr*)inst.setSrcImm(),
                           inst.hasRelImm());
    inst.setImmRef(*r);
  }

  D(2, "Decoded 0x" << hex << code << " into: " << inst << '\n');

  return &inst;
}

