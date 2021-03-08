#include <vector>
#include <iostream>
#include <stdexcept>
#include <math.h>
#include <climits>
#include <fcntl.h>
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

// Convert 32-bit integer register file to IEEE-754 floating point number.
float vortex::intregToFloat(uint32_t input) {
  // 31th bit
  bool sign = input & 0x80000000;
  // Exponent: 23th ~ 30th bits -> 8 bits in total
  int32_t exp  = ((input & 0x7F800000)>>23);
  // printf("exp = %u\n", exp);
  // 0th ~ 22th bits -> 23 bits fraction
  uint32_t frac = input & 0x007FFFFF;
  // Frac_value= 1 + sum{i = 1}{23}{b_{23-i}*2^{-i}}
  double frac_value;
  if (exp == 0) {  // subnormal
    if (frac == 0) {
      // zero
      if (sign) 
        return -0.0;
      else 
        return 0.0;
    }
    frac_value = 0.0;
  } else
    frac_value = 1.0;

  for (int i = 0; i < 23; i++) {
    int bi = frac & 0x1;
    frac_value += static_cast<double>(bi * pow(2.0, i-23));
    frac = (frac >> 1);
  }
  
  return (float)((static_cast<double>(pow(-1.0, sign))) * (static_cast<double>(pow(2.0, exp - 127.0)))* frac_value);
}

// Convert a floating point number to IEEE-754 32-bit representation, 
// so that it could be stored in a 32-bit integer register file
// Reference: https://www.wikihow.com/Convert-a-Number-from-Decimal-to-IEEE-754-Floating-Point-Representation
 //            https://www.technical-recipes.com/2012/converting-between-binary-and-decimal-representations-of-ieee-754-floating-point-numbers-in-c/
uint32_t vortex::floatToBin(float in_value) {
  union  {
       float input;   // assumes sizeof(float) == sizeof(int)
       int   output;
  } data;

  data.input = in_value;

  std::bitset<sizeof(float) * CHAR_BIT>   bits(data.output);
  std::string mystring = bits.to_string<char, std::char_traits<char>, std::allocator<char> >();
  // Convert binary to uint32_t
  Word result = stoul(mystring, nullptr, 2);
  return result;
}

// https://en.wikipedia.org/wiki/Single-precision_floating-point_format
// check floating-point number in binary format is NaN
uint8_t vortex::fpBinIsNan(uint32_t din) {
  bool fsign = din & 0x80000000;
  uint32_t expo = (din>>23) & 0x000000FF;
  uint32_t fraction = din & 0x007FFFFF;
  uint32_t bit_22 = din & 0x00400000;

  if ((expo==0xFF) && (fraction!=0)) {
    // if (!fsign && (fraction == 0x00400000)) 
    if (!fsign && (bit_22))
      return 1; // quiet NaN, return 1
    else 
      return 2; // signaling NaN, return 2
  }
  return 0;
}

// check floating-point number in binary format is zero
uint8_t vortex::fpBinIsZero(uint32_t din) {
  bool fsign = din & 0x80000000;
  uint32_t expo = (din>>23) & 0x000000FF;
  uint32_t fraction = din & 0x007FFFFF;

  if ((expo==0) && (fraction==0)) {
    if (fsign)
      return 1; // negative 0
    else
      return 2; // positive 0
  }
  return 0;  // not zero
}

// check floating-point number in binary format is infinity
uint8_t vortex::fpBinIsInf(uint32_t din) {
  bool fsign = din & 0x80000000;
  uint32_t expo = (din>>23) & 0x000000FF;
  uint32_t fraction = din & 0x007FFFFF;

  if ((expo==0xFF) && (fraction==0)) {
    if (fsign)
      return 1; // negative infinity
    else
      return 2; // positive infinity
  }
  return 0;  // not infinity
}