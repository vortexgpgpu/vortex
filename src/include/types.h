/*******************************************************************************
 HARPtools by Chad D. Kersey, Summer 2011
*******************************************************************************/
#ifndef __TYPES_H
#define __TYPES_H

#include <stdint.h>

namespace Harp {
  typedef uint8_t  Byte;
  typedef uint64_t Word;
  typedef uint64_t Word_u;
  typedef int64_t  Word_s;

  typedef Word     Addr;
  typedef Word     Size;

  typedef unsigned RegNum;
  typedef unsigned ThdNum;

  enum MemFlags {RD_USR = 1, WR_USR = 2,  EX_USR = 4, 
                 RD_SUP = 8, WR_SUP = 16, EX_SUP = 32};
};

#endif
