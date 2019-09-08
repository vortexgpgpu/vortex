// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef _VVortex__Syms_H_
#define _VVortex__Syms_H_

#include "verilated.h"

// INCLUDE MODULE CLASSES
#include "VVortex.h"
#include "VVortex_VX_mem_req_inter.h"
#include "VVortex_VX_inst_mem_wb_inter.h"
#include "VVortex_VX_branch_response_inter.h"
#include "VVortex_VX_inst_meta_inter.h"
#include "VVortex_VX_frE_to_bckE_req_inter.h"
#include "VVortex_VX_warp_ctl_inter.h"
#include "VVortex_VX_wb_inter.h"

// SYMS CLASS
class VVortex__Syms : public VerilatedSyms {
  public:
    
    // LOCAL STATE
    const char* __Vm_namep;
    bool __Vm_didInit;
    
    // SUBCELL STATE
    VVortex*                       TOPp;
    VVortex_VX_branch_response_inter TOP__Vortex__DOT__VX_branch_rsp;
    VVortex_VX_mem_req_inter       TOP__Vortex__DOT__VX_exe_mem_req;
    VVortex_VX_inst_mem_wb_inter   TOP__Vortex__DOT__VX_mem_wb;
    VVortex_VX_warp_ctl_inter      TOP__Vortex__DOT__VX_warp_ctl;
    VVortex_VX_wb_inter            TOP__Vortex__DOT__VX_writeback_inter;
    VVortex_VX_inst_meta_inter     TOP__Vortex__DOT__fe_inst_meta_fd;
    VVortex_VX_frE_to_bckE_req_inter TOP__Vortex__DOT__vx_front_end__DOT__VX_frE_to_bckE_req;
    
    // CREATORS
    VVortex__Syms(VVortex* topp, const char* namep);
    ~VVortex__Syms() {}
    
    // METHODS
    inline const char* name() { return __Vm_namep; }
    
} VL_ATTR_ALIGNED(64);

#endif  // guard
