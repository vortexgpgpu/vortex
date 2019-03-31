// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header

#ifndef _VVX_register_file__Syms_H_
#define _VVX_register_file__Syms_H_

#include "verilated_heavy.h"

// INCLUDE MODULE CLASSES
#include "VVX_register_file.h"

// SYMS CLASS
class VVX_register_file__Syms : public VerilatedSyms {
  public:
    
    // LOCAL STATE
    const char* __Vm_namep;
    bool __Vm_didInit;
    
    // SUBCELL STATE
    VVX_register_file*             TOPp;
    
    // CREATORS
    VVX_register_file__Syms(VVX_register_file* topp, const char* namep);
    ~VVX_register_file__Syms() {}
    
    // METHODS
    inline const char* name() { return __Vm_namep; }
    
} VL_ATTR_ALIGNED(64);

#endif // guard
