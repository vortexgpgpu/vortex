// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See VVortex.h for the primary calling header

#ifndef _VVortex___024unit_H_
#define _VVortex___024unit_H_

#include "verilated.h"

class VVortex__Syms;

//----------

VL_MODULE(VVortex___024unit) {
  public:
    
    // PORTS
    
    // LOCAL SIGNALS
    
    // LOCAL VARIABLES
    
    // INTERNAL VARIABLES
  private:
    VVortex__Syms* __VlSymsp;  // Symbol table
  public:
    
    // PARAMETERS
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(VVortex___024unit);  ///< Copying not allowed
  public:
    VVortex___024unit(const char* name="TOP");
    ~VVortex___024unit();
    
    // API METHODS
    
    // INTERNAL METHODS
    void __Vconfigure(VVortex__Syms* symsp, bool first);
  private:
    void _ctor_var_reset() VL_ATTR_COLD;
} VL_ATTR_ALIGNED(128);

#endif // guard
