// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header

#ifndef _VVortex__Syms_H_
#define _VVortex__Syms_H_

#include "verilated_heavy.h"

// INCLUDE MODULE CLASSES
#include "VVortex.h"

// SYMS CLASS
class VVortex__Syms : public VerilatedSyms {
  public:
    
    // LOCAL STATE
    const char* __Vm_namep;
    bool __Vm_didInit;
    
    // SUBCELL STATE
    VVortex*                       TOPp;
    
    // CREATORS
    VVortex__Syms(VVortex* topp, const char* namep);
    ~VVortex__Syms() {}
    
    // METHODS
    inline const char* name() { return __Vm_namep; }
    
} VL_ATTR_ALIGNED(64);

#endif // guard
