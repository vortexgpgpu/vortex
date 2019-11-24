// Verilated -*- C++ -*-
#ifndef _Vcache_simX__Inlines_H_
#define _Vcache_simX__Inlines_H_

#include "verilated.h"

//======================

#ifndef VL_HAVE_CONST_W_16X
# define VL_HAVE_CONST_W_16X
static inline WDataOutP VL_CONST_W_16X(int obits, WDataOutP o
    	,IData d15,IData d14,IData d13,IData d12,IData d11,IData d10,IData d9,IData d8
    	,IData d7,IData d6,IData d5,IData d4,IData d3,IData d2,IData d1,IData d0) {
        o[15]=d15; o[14]=d14; o[13]=d13; o[12]=d12; o[11]=d11; o[10]=d10; o[9]=d9; o[8]=d8;
        o[7]=d7; o[6]=d6; o[5]=d5; o[4]=d4; o[3]=d3; o[2]=d2; o[1]=d1; o[0]=d0;
        for(int i=16;i<VL_WORDS_I(obits);i++) o[i] = (IData)0x0;
        return o;
}
#endif

//======================

#endif  /*guard*/
