// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header

#ifndef _Vvortex__Syms_H_
#define _Vvortex__Syms_H_

#include "verilated_heavy.h"

// INCLUDE MODULE CLASSES
#include "Vvortex.h"

// SYMS CLASS
class Vvortex__Syms : public VerilatedSyms {
  public:
    
    // LOCAL STATE
    const char* __Vm_namep;
    bool __Vm_didInit;
    
    // SUBCELL STATE
    Vvortex*                       TOPp;
    
    // CREATORS
    Vvortex__Syms(Vvortex* topp, const char* namep);
    ~Vvortex__Syms() {}
    
    // METHODS
    inline const char* name() { return __Vm_namep; }
    
} VL_ATTR_ALIGNED(64);

#endif // guard
