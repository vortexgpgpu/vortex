// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header

#ifndef _VVortex__Syms_H_
#define _VVortex__Syms_H_

#include "verilated.h"

// INCLUDE MODULE CLASSES
#include "VVortex.h"
#include "VVortex_VX_context_slave.h"

// SYMS CLASS
class VVortex__Syms : public VerilatedSyms {
  public:
    
    // LOCAL STATE
    const char* __Vm_namep;
    bool __Vm_didInit;
    
    // SUBCELL STATE
    VVortex*                       TOPp;
    VVortex_VX_context_slave       TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__1__KET____DOT__VX_Context_one;
    VVortex_VX_context_slave       TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__2__KET____DOT__VX_Context_one;
    VVortex_VX_context_slave       TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__3__KET____DOT__VX_Context_one;
    VVortex_VX_context_slave       TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__4__KET____DOT__VX_Context_one;
    VVortex_VX_context_slave       TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__5__KET____DOT__VX_Context_one;
    VVortex_VX_context_slave       TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__6__KET____DOT__VX_Context_one;
    VVortex_VX_context_slave       TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__7__KET____DOT__VX_Context_one;
    
    // CREATORS
    VVortex__Syms(VVortex* topp, const char* namep);
    ~VVortex__Syms() {}
    
    // METHODS
    inline const char* name() { return __Vm_namep; }
    
} VL_ATTR_ALIGNED(64);

#endif // guard
