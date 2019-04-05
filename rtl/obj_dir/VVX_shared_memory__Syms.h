// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header

#ifndef _VVX_shared_memory__Syms_H_
#define _VVX_shared_memory__Syms_H_

#include "verilated.h"

// INCLUDE MODULE CLASSES
#include "VVX_shared_memory.h"

// SYMS CLASS
class VVX_shared_memory__Syms : public VerilatedSyms {
  public:
    
    // LOCAL STATE
    const char* __Vm_namep;
    bool __Vm_didInit;
    
    // SUBCELL STATE
    VVX_shared_memory*             TOPp;
    
    // CREATORS
    VVX_shared_memory__Syms(VVX_shared_memory* topp, const char* namep);
    ~VVX_shared_memory__Syms() {}
    
    // METHODS
    inline const char* name() { return __Vm_namep; }
    
} VL_ATTR_ALIGNED(64);

#endif // guard
