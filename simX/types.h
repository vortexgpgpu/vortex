#pragma once

#include <stdint.h>
#include <bitset>
#include <VX_config.h>

namespace vortex {

typedef uint8_t  Byte;
typedef uint32_t Word;
typedef int32_t  WordI;

typedef uint32_t Addr;
typedef uint32_t Size;

typedef std::bitset<32> RegMask;

typedef std::bitset<32> ThreadMask;

typedef std::bitset<32> WarpMask;

enum MemFlags {
  RD_USR = 1, 
  WR_USR = 2,  
  EX_USR = 4, 
  RD_SUP = 8, 
  WR_SUP = 16, 
  EX_SUP = 32
};

}