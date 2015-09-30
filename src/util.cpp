/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#include <vector>

#include "include/types.h"
#include "include/util.h"

using namespace Harp;
using namespace std;

// Make it easy for autotools-based build systems to detect this library.
extern "C" {
  int harplib_present = 1;
};

void Harp::wordToBytes(Byte *b, Word_u w, Size wordSize) {
  while (wordSize--) {
    *(b++) = w & 0xff;
    w >>= 8;
  }
}

Word_u Harp::bytesToWord(const Byte *b, Size wordSize) {
  Word_u w = 0;
  b += wordSize-1;
  while (wordSize--) {
    w <<= 8;
    w |= *(b--);
  }
  return w;
}

Word_u Harp::flagsToWord(bool r, bool w, bool x) {
  Word_u word = 0;
  if (r) word |= RD_USR;
  if (w) word |= WR_USR;
  if (x) word |= EX_USR;
  return word; 
}

void Harp::wordToFlags(bool &r, bool &w, bool &x, Word_u f) {
  r = f & RD_USR;
  w = f & WR_USR;
  x = f & EX_USR;
}

Byte Harp::readByte(const vector<Byte> &b, Size &n) {
  if (b.size() <= n) throw OutOfBytes();
  return b[n++];
}

Word_u Harp::readWord(const vector<Byte> &b, Size &n, Size wordSize) {
  if (b.size() - n < wordSize) throw OutOfBytes();
  Word_u w(0);
  n += wordSize;
  for (Size i = 0; i < wordSize; i++) {
    w <<= 8;
    w |= b[n - i - 1];
  }
  return w;
}

void Harp::writeByte(vector<Byte> &p, Size &n, Byte b) {
  if (p.size() <= n) p.resize(n+1);
  p[n++] = b;
}

void Harp::writeWord(vector<Byte> &p, Size &n, Size wordSize, Word w) {
  if (p.size() < (n+wordSize)) p.resize(n+wordSize);
  while (wordSize--) {
    p[n++] = w & 0xff;
    w >>= 8;
  }
}
