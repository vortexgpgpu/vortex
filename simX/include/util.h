/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#ifndef __UTIL_H
#define __UTIL_H

#include <vector>
#include "types.h"

namespace Harp {
  Word_u bytesToWord(const Byte *b, Size wordSize);
  void wordToBytes(Byte *b, Word_u w, Size wordSize);
  Word_u flagsToWord(bool r, bool w, bool x);
  void wordToFlags(bool &r, bool &w, bool &x, Word_u f);

  class OutOfBytes {};

  Byte readByte(const std::vector<Byte> &b, Size &n);
  Word_u readWord(const std::vector<Byte> &b, Size &n, Size wordSize);
  void writeByte(std::vector<Byte> &p, Size &n, Byte b);
  void writeWord(std::vector<Byte> &p, Size &n, Size wordSize, Word w);
}

#endif
