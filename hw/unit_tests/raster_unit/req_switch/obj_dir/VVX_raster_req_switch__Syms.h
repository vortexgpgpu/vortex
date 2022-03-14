// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef _VVX_RASTER_REQ_SWITCH__SYMS_H_
#define _VVX_RASTER_REQ_SWITCH__SYMS_H_  // guard

#include "verilated_heavy.h"

// INCLUDE MODULE CLASSES
#include "VVX_raster_req_switch.h"
#include "VVX_raster_req_switch___024unit.h"

// DPI TYPES for DPI Export callbacks (Internal use)

// SYMS CLASS
class VVX_raster_req_switch__Syms : public VerilatedSyms {
  public:
    
    // LOCAL STATE
    const char* __Vm_namep;
    bool __Vm_activity;  ///< Used by trace routines to determine change occurred
    uint32_t __Vm_baseCode;  ///< Used by trace routines when tracing multiple models
    bool __Vm_didInit;
    
    // SUBCELL STATE
    VVX_raster_req_switch*         TOPp;
    VVX_raster_req_switch___024unit TOP____024unit;
    
    // CREATORS
    VVX_raster_req_switch__Syms(VVX_raster_req_switch* topp, const char* namep);
    ~VVX_raster_req_switch__Syms() {}
    
    // METHODS
    inline const char* name() { return __Vm_namep; }
    
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);

#endif  // guard
