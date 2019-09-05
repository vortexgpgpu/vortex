// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See VVortex.h for the primary calling header

#ifndef _VVortex_VX_frE_to_bckE_req_inter_H_
#define _VVortex_VX_frE_to_bckE_req_inter_H_

#include "verilated.h"

class VVortex__Syms;

//----------

VL_MODULE(VVortex_VX_frE_to_bckE_req_inter) {
  public:
    
    // PORTS
    
    // LOCAL SIGNALS
    // Begin mtask footprint  all: 
    VL_SIG8(branch_type,2,0);
    VL_SIG8(jal,0,0);
    VL_SIG16(csr_address,11,0);
    VL_SIGW(a_reg_data,127,0,4);
    VL_SIGW(b_reg_data,127,0,4);
    VL_SIG(itype_immed,31,0);
    VL_SIG(jal_offset,31,0);
    
    // LOCAL VARIABLES
    
    // INTERNAL VARIABLES
  private:
    VVortex__Syms* __VlSymsp;  // Symbol table
  public:
    
    // PARAMETERS
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(VVortex_VX_frE_to_bckE_req_inter);  ///< Copying not allowed
  public:
    VVortex_VX_frE_to_bckE_req_inter(const char* name="TOP");
    ~VVortex_VX_frE_to_bckE_req_inter();
    
    // API METHODS
    
    // INTERNAL METHODS
    void __Vconfigure(VVortex__Syms* symsp, bool first);
  private:
    void _ctor_var_reset() VL_ATTR_COLD;
} VL_ATTR_ALIGNED(128);

#endif // guard
