#pragma once

#include <stdint.h>
#include <VX_config.h>

namespace vortex {

typedef uint8_t  Byte;
typedef uint32_t Word;
typedef uint32_t Word_u;
typedef int32_t  Word_s;

typedef Word_u   Addr;
typedef Word_u   Size;

typedef unsigned RegNum;
typedef unsigned ThdNum;

enum MemFlags {
  RD_USR = 1, 
  WR_USR = 2,  
  EX_USR = 4, 
  RD_SUP = 8, 
  WR_SUP = 16, 
  EX_SUP = 32
};

}