#include <vector>
#include <iostream>
#include <stdexcept>
#include "types.h"
#include "util.h"

using namespace vortex;

Word vortex::signExt(Word w, Size bit, Word mask) {
  if (w >> (bit - 1))
    w |= ~mask;
  return w;
}

void vortex::wordToBytes(Byte *b, Word w, Size wordSize) {
  while (wordSize--) {
    *(b++) = w & 0xff;
    w >>= 8;
  }
}

Word vortex::bytesToWord(const Byte *b, Size wordSize) {
  Word w = 0;
  b += wordSize-1;
  while (wordSize--) {
    w <<= 8;
    w |= *(b--);
  }
  return w;
}

Word vortex::flagsToWord(bool r, bool w, bool x) {
  Word word = 0;
  if (r) word |= RD_USR;
  if (w) word |= WR_USR;
  if (x) word |= EX_USR;
  return word; 
}

void vortex::wordToFlags(bool &r, bool &w, bool &x, Word f) {
  r = f & RD_USR;
  w = f & WR_USR;
  x = f & EX_USR;
}

Byte vortex::readByte(const std::vector<Byte> &b, Size &n) {
  if (b.size() <= n) 
    throw std::out_of_range("out of range");
  return b[n++];
}

Word vortex::readWord(const std::vector<Byte> &b, Size &n, Size wordSize) {
  if (b.size() - n < wordSize) 
    throw std::out_of_range("out of range");
  Word w(0);
  n += wordSize;
  // std::cout << "wordSize: " << wordSize << "\n";
  for (Size i = 0; i < wordSize; i++) {
    w <<= 8;
    // cout << "index: " << n - i - 1 << "\n";
    w |= b[n - i - 1];
  }
  // cout << "b[0]" << std::hex << w << "\n";
  return w;
}

void vortex::writeByte(std::vector<Byte> &p, Size &n, Byte b) {
  if (p.size() <= n) p.resize(n+1);
  p[n++] = b;
}

void vortex::writeWord(std::vector<Byte> &p, Size &n, Size wordSize, Word w) {
  if (p.size() < (n+wordSize)) p.resize(n+wordSize);
  while (wordSize--) {
    p[n++] = w & 0xff;
    w >>= 8;
  }
}
