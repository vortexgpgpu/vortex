#include "util.h"
#include <iostream>
#include <stdexcept>
#include <math.h>
#include <climits>
#include <string.h>
#include <bitset>
#include <fcntl.h>

using namespace vortex;

// Apply integer sign extension
uint32_t vortex::signExt(uint32_t w, uint32_t bit, uint32_t mask) {
  if (w >> (bit - 1))
    w |= ~mask;
  return w;
}

// Convert a floating point number to IEEE-754 32-bit representation, 
// so that it could be stored in a 32-bit integer register file
// Reference: https://www.wikihow.com/Convert-a-Number-from-Decimal-to-IEEE-754-Floating-Point-Representation
 //            https://www.technical-recipes.com/2012/converting-between-binary-and-decimal-representations-of-ieee-754-floating-point-numbers-in-c/
uint32_t vortex::floatToBin(float in_value) {
  union {
       float input;   // assumes sizeof(float) == sizeof(int)
       int   output;
  } data;

  data.input = in_value;

  std::bitset<sizeof(float) * CHAR_BIT> bits(data.output);
  std::string mystring = bits.to_string<char, std::char_traits<char>, std::allocator<char>>();
  // Convert binary to uint32_t
  uint32_t result = stoul(mystring, nullptr, 2);
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

// return file extension
const char* vortex::fileExtension(const char* filepath) {
    const char *ext = strrchr(filepath, '.');
    if (ext == NULL || ext == filepath) 
      return "";
    return ext + 1;
}