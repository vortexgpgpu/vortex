// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header

#ifndef _VVX_decode__Syms_H_
#define _VVX_decode__Syms_H_

#include "verilated_heavy.h"

// INCLUDE MODULE CLASSES
#include "VVX_decode.h"

// SYMS CLASS
class VVX_decode__Syms : public VerilatedSyms {
  public:
    
    // LOCAL STATE
    const char* __Vm_namep;
    bool __Vm_didInit;
    
    // SUBCELL STATE
    VVX_decode*                    TOPp;
    
    // CREATORS
    VVX_decode__Syms(VVX_decode* topp, const char* namep);
    ~VVX_decode__Syms() {}
    
    // METHODS
    inline const char* name() { return __Vm_namep; }
    
} VL_ATTR_ALIGNED(64);

#endif // guard
