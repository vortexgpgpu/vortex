// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef _Vrf2_32x128_wm1__Syms_H_
#define _Vrf2_32x128_wm1__Syms_H_

#include "verilated.h"

// INCLUDE MODULE CLASSES
#include "Vrf2_32x128_wm1.h"

// SYMS CLASS
class Vrf2_32x128_wm1__Syms : public VerilatedSyms {
  public:
    
    // LOCAL STATE
    const char* __Vm_namep;
    bool __Vm_didInit;
    
    // SUBCELL STATE
    Vrf2_32x128_wm1*               TOPp;
    
    // CREATORS
    Vrf2_32x128_wm1__Syms(Vrf2_32x128_wm1* topp, const char* namep);
    ~Vrf2_32x128_wm1__Syms() {}
    
    // METHODS
    inline const char* name() { return __Vm_namep; }
    
} VL_ATTR_ALIGNED(64);

#endif  // guard
