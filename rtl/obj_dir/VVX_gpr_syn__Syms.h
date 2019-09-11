// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef _VVX_gpr_syn__Syms_H_
#define _VVX_gpr_syn__Syms_H_

#include "verilated.h"

// INCLUDE MODULE CLASSES
#include "VVX_gpr_syn.h"

// SYMS CLASS
class VVX_gpr_syn__Syms : public VerilatedSyms {
  public:
    
    // LOCAL STATE
    const char* __Vm_namep;
    bool __Vm_didInit;
    
    // SUBCELL STATE
    VVX_gpr_syn*                   TOPp;
    
    // CREATORS
    VVX_gpr_syn__Syms(VVX_gpr_syn* topp, const char* namep);
    ~VVX_gpr_syn__Syms() {}
    
    // METHODS
    inline const char* name() { return __Vm_namep; }
    
} VL_ATTR_ALIGNED(64);

#endif  // guard
