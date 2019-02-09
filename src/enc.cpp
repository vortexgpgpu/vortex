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

ByteDecoder::ByteDecoder(const ArchDef &ad) {
  wordSize = ad.getWordSize();
}

static void decodeError(string msg) {
  cout << "Instruction decoder error: " << msg << '\n';
  exit(1);
}

void Encoder::encodeChunk(DataChunk &dest, const TextChunk &src) {
  typedef vector<Instruction*>::const_iterator vec_it;
  const vector<Instruction*> &s(src.instructions);
  vector<Byte> &d(dest.contents);

  /* Keep encoding the instructions. */
  Size n = 0;

  /* For each instruction. */
  for (vec_it i = s.begin(); i != s.end(); i++) {
    Ref *ref;

    /* Perform the encoding. */
    n += encode(ref, d, n, **i);
    
    /* Add reference if necessary. */
    if (ref != NULL) {
      ref->ibase = n;
      dest.refs.push_back(ref);
    }
  }

  dest.alignment = src.alignment;
  dest.flags = src.flags;
  dest.address = src.address;
  dest.bound = src.bound;
  if (src.isGlobal()) dest.setGlobal();

  d.resize(n);
  dest.size = n;
}

void Decoder::decodeChunk(TextChunk &dest, const DataChunk &src) {
  typedef vector<Instruction*>::const_iterator vec_it;
  const vector<Byte> &v(src.contents);
  Size n = 0;

  setRefs(src.refs);

  while (n < src.contents.size()) {
    Instruction *inst = decode(v, n);
    if (inst->hasRefLiteral()) {
      dest.refs.push_back(inst->getRefLiteral());
    }

    dest.instructions.push_back(inst);
  }

  dest.alignment = src.alignment;
  dest.flags = src.flags;
  dest.address = src.address;
  dest.bound = src.bound;
  if (src.isGlobal()) dest.setGlobal();

  clearRefs();
}

void Decoder::setRefs(const std::vector<Ref*> &refVec) {
  haveRefs = true;

  typedef std::vector<Ref*>::const_iterator vec_ci;

  for (vec_ci i = refVec.begin(); i != refVec.end(); i++) {
    OffsetRef *oref = dynamic_cast<OffsetRef*>(*i);
    if (oref) {
      refMap[oref->getOffset()] = *i;
    } else {
      decodeError("Unknown Ref type in Decoder::setRefs");
    }
  }
}

Instruction *ByteDecoder::decode(const vector<Byte> &v, Size &n) {
  Instruction &inst = *(new Instruction());

  uint8_t pred = readByte(v, n);
  if (pred != 0xff) inst.setPred(pred);

  unsigned op = readByte(v, n);
  inst.setOpcode(Instruction::Opcode(op));

  bool usedImm = false;

  switch (Instruction::instTable[op].argClass) {
    case Instruction::AC_NONE:
      break;
    case Instruction::AC_2REG:
      inst.setDestReg(readByte(v, n));
      inst.setSrcReg(readByte(v, n));
      break;
    case Instruction::AC_2IMM:
      inst.setDestReg(readByte(v, n));
      inst.setSrcImm(readWord(v, n, wordSize));
      usedImm = true;
      break;
    case Instruction::AC_3REG:
      inst.setDestReg(readByte(v, n));
      inst.setSrcReg(readByte(v, n));
      inst.setSrcReg(readByte(v, n));
      break;
    case Instruction::AC_3PREG:
      inst.setDestPReg(readByte(v, n));
      inst.setSrcPReg(readByte(v, n));
      inst.setSrcPReg(readByte(v, n));
      break;
    case Instruction::AC_3IMM:
      inst.setDestReg(readByte(v, n));
      inst.setSrcReg(readByte(v, n));
      inst.setSrcImm(readWord(v, n, wordSize));
      usedImm = true;
      break;
    case Instruction::AC_3REGSRC:
      inst.setSrcReg(readByte(v, n));
      inst.setSrcReg(readByte(v, n));
      inst.setSrcReg(readByte(v, n));
      break;
    case Instruction::AC_1IMM:
      inst.setSrcImm(readWord(v, n, wordSize));
      usedImm = true;
      break;
    case Instruction::AC_1REG:
      inst.setSrcReg(readByte(v, n));
      break;
    case Instruction::AC_3IMMSRC:
      inst.setSrcReg(readByte(v, n));
      inst.setSrcReg(readByte(v, n));
      inst.setSrcImm(readWord(v, n, wordSize));
      usedImm = true;
      break;
    case Instruction::AC_PREG_REG:
      inst.setDestPReg(readByte(v, n));
      inst.setSrcReg(readByte(v, n));
      break;
    case Instruction::AC_2PREG:
      inst.setDestPReg(readByte(v, n));
      inst.setSrcPReg(readByte(v, n));
      break;
    case Instruction::AC_2REGSRC:
      inst.setSrcReg(readByte(v, n));
      inst.setSrcReg(readByte(v, n));
      break;
   default:
      decodeError("Unknown argument class.");
  }

  if (haveRefs && usedImm && 
      refMap.find(n - wordSize) != refMap.end()) {
    OffsetRef *oref = dynamic_cast<OffsetRef*>(refMap[n - wordSize]);
    if (!oref) {
      decodeError("Expected OffsetRef when decoding instruction stream.");
    }
    Ref *r = new SimpleRef(oref->name, *(Addr*)(inst.setSrcImm()), 
                           inst.hasRelImm());
    inst.setImmRef(*r);
  } 

  return &inst;
}

ByteEncoder::ByteEncoder(const ArchDef &ad) {
  wordSize = ad.getWordSize();
}

Size ByteEncoder::encode(Ref *&ref, vector<Byte> &v, Size n0, Instruction &i) {
  Size n(n0);

  if (i.hasPred()) writeByte(v, n, i.getPred());
  else writeByte(v, n, 0xff);

  writeByte(v, n, Byte(i.getOpcode()));

  if (i.hasRDest()) {
    writeByte(v, n, Byte(i.getRDest()));
  } else if (i.hasPDest()) {
    writeByte(v, n, Byte(i.getPDest()));
  }

  for (RegNum j = 0; j < i.getNRSrc(); j++) {
    writeByte(v, n, Byte(i.getRSrc(j)));
  }

  for (RegNum j = 0; j < i.getNPSrc(); j++) {
    writeByte(v, n, Byte(i.getPSrc(j)));
  }

  ref = NULL;
  if (i.hasImm()) {
    if (i.hasRefLiteral()) {
      Ref *r = i.getRefLiteral();
      ref = new OffsetRef(r->name, v, n, wordSize*8, wordSize, i.hasRelImm());
    }

    writeWord(v, n, wordSize, i.getImm());
  }

  return n - n0;
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

static Word mask(Size bits) {
  return (1ull<<bits)-1;
}

static void getSizes(const ArchDef &arch, Size &n, Size& o, Size &r, Size &p, 
                     Size &i1, Size &i2, Size &i3)
{
 n  = arch.getWordSize() * 8;
 o  = 7;
 r  = ceilLog2(arch.getNRegs());
 p  = 0;
 i1 = n - 1 - p - o;
 i2 = i1 - r;
 i3 = i2 - r;
}

WordDecoder::WordDecoder(const ArchDef &arch) {
  getSizes(arch, n, o, r, p, i1, i2, i3);
  if (p > r) r = p;
  oMask  = mask(o);  rMask  = mask(r);  pMask  = mask(p);
  i1Mask = mask(i1); i2Mask = mask(i2); i3Mask = mask(i3);
}

Word signExt(Word w, Size bit, Word mask) {
  if (w>>(bit-1)) w |= ~mask;
  return w;
}

Instruction *WordDecoder::decode(const std::vector<Byte> &v, Size &idx) {
  Word code(readWord(v, idx, n/8));
  Instruction &inst = * new Instruction();  

  // bool predicated = (code>>(n-1));
  bool predicated = false;
  if (predicated) { inst.setPred((code>>(n-p-1))&pMask); }

  Instruction::Opcode op = (Instruction::Opcode)((code>>i1)&oMask);
  inst.setOpcode(op);

  bool usedImm(false);
  switch(Instruction::instTable[op].argClass) {
    case Instruction::AC_NONE: 
      break;
    case Instruction::AC_1IMM:
      inst.setSrcImm(signExt(code&i1Mask, i1, i1Mask));
      usedImm = true;
      break;
    case Instruction::AC_2IMM:
      inst.setDestReg((code>>i2)&rMask);
      inst.setSrcImm(signExt(code&i2Mask, i2, i2Mask));
      usedImm = true;
      break;
    case Instruction::AC_3IMM:
      inst.setDestReg((code>>i2)&rMask);
      inst.setSrcReg((code>>i3)&rMask);
      inst.setSrcImm(signExt(code&i3Mask, i3, i3Mask));
      usedImm = true;
      break;
    case Instruction::AC_3IMMSRC:
      inst.setSrcReg((code>>i2)&rMask);
      inst.setSrcReg((code>>i3)&rMask);
      inst.setSrcImm(signExt(code&i3Mask, i3, i3Mask));
      usedImm = true;
      break;
    case Instruction::AC_1REG:
      inst.setSrcReg((code>>i2)&rMask);
      break;
    case Instruction::AC_2REG:
      inst.setDestReg((code>>i2)&rMask);
      inst.setSrcReg((code>>i3)&rMask);
      break;
    case Instruction::AC_3REG:
      inst.setDestReg((code>>i2)&rMask);
      inst.setSrcReg((code>>i3)&rMask);
      inst.setSrcReg((code>>(i3-r))&rMask);
      break;
    case Instruction::AC_3REGSRC:
      inst.setSrcReg((code>>i2)&rMask);
      inst.setSrcReg((code>>i3)&rMask);
      inst.setSrcReg((code>>(i3-r))&rMask);
      break;
    case Instruction::AC_PREG_REG:
      inst.setDestPReg((code>>i2)&pMask);
      inst.setSrcReg((code>>i3)&rMask);
      break;
    case Instruction::AC_2PREG:
      inst.setDestPReg((code>>i2)&pMask);
      inst.setSrcPReg((code>>i3)&pMask);
      break;
    case Instruction::AC_3PREG:
      inst.setDestPReg((code>>i2)&pMask);
      inst.setSrcPReg((code>>i3)&pMask);
      inst.setSrcPReg((code>>(i3-r))&pMask);
      break;
    case Instruction::AC_2REGSRC:
      inst.setSrcReg((code>>i2)&rMask);
      inst.setSrcReg((code>>i3)&rMask);
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

WordEncoder::WordEncoder(const ArchDef &arch) {
  getSizes(arch, n, o, r, p, i1, i2, i3);
  if (p > r) r = p;
  oMask  = mask(o);  rMask  = mask(r);  pMask  = mask(p);
  i1Mask = mask(i1); i2Mask = mask(i2); i3Mask = mask(i3);
}

Size WordEncoder::encode(Ref *&ref, std::vector<Byte> &v, 
                                 Size idx, Instruction &i)
{
  Word code = 0;
  Size bitsWritten = 0;

  /* Predicate/predicated bit */
  if (i.hasPred()) {
    code = 1 << p;
    code |= (i.getPred()&pMask);
    if (i.getPred() > pMask) {
      cout << "Predicate in " << i << " does not fit in encoding.\n";
      exit(1);
    }
  }
  bitsWritten += (1 + p);

  /* Opcode */
  code <<= o;
  code |= (i.getOpcode()&oMask);
  if (i.getOpcode() > oMask) {
    cout << "Opcode in " << i << " does not fit in encoding.\n";
    exit(1);
  }
  bitsWritten += o;

  if (i.hasRDest()) {
    code <<= r;
    code |= i.getRDest();
    bitsWritten += r;
    if (i.getRDest() > rMask) {
      cout << "Destination register in " << i << " does not fit in encoding.\n";
      exit(1);
    }
  }

  if (i.hasPDest()) {
    code <<= r;
    code |= i.getPDest();
    bitsWritten += r;
    if (i.getPDest() > rMask) {
      cout << "Destination predicate in " <<i<< " does not fit in encoding.\n";
      exit(1);
    }
  }

  for (Size j = 0; j < i.getNRSrc(); j++) {
    code <<= r;
    code |= i.getRSrc(j);
    bitsWritten += r;
    if (i.getRSrc(j) > rMask) {
      cout << "Source register " << j << " in " << i 
           << " does not fit in encoding.\n";
      exit(1);
    }
  }

  for (Size j = 0; j < i.getNPSrc(); j++) {
    code <<= r;
    code |= i.getPSrc(j);
    bitsWritten += r;
    if (i.getPSrc(j) > rMask) {
      cout << "Source predicate " << j << " in " << i
           << " does not fit in encoding.\n";
      exit(1);
    }
  }

  if (i.hasRefLiteral()) {
    Ref *r = i.getRefLiteral();
    ref = new OffsetRef(r->name, v, idx, n - bitsWritten, n, i.hasRelImm());
  } else {
    ref = NULL;
  }

  if (i.hasImm()) {
    if (bitsWritten == n - i1) {
      code <<= i1;
      code |= (i.getImm()&i1Mask);
      bitsWritten += i1;
      Word_s ws(i.getImm());
      if ((ws >> i1) != 0 && (ws >> i1) != -1) goto tooBigImm;
    } else if (bitsWritten == n - i2) {
      code <<= i2;
      code |= (i.getImm()&i2Mask);
      bitsWritten += i2;
      Word_s ws(i.getImm());
      if ((ws >> i2) != 0 && (ws >> i2) != -1) goto tooBigImm;
    } else if (bitsWritten == n - i3) {
      code <<= i3;
      code |= (i.getImm()&i3Mask);
      bitsWritten += i3;
      Word_s ws(i.getImm());
      if ((ws >> i3) != 0 && (ws >> i3) != -1) goto tooBigImm;
    } else {
      cout << "WordEncoder::encode() could not encode: " << i << '\n';
      exit(1);
    }
  }

  if (bitsWritten < n) code <<= (n - bitsWritten);

  writeWord(v, idx, n/8, code); 

  return n/8;

tooBigImm:
  cout << "Immediate in " << i << " too large to encode.\n";
  exit(1);
}
