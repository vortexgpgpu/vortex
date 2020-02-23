// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vcache_simX.h for the primary calling header

#include "Vcache_simX_VX_dcache_request_inter.h" // For This
#include "Vcache_simX__Syms.h"


//--------------------
// STATIC VARIABLES


//--------------------

VL_CTOR_IMP(Vcache_simX_VX_dcache_request_inter) {
    // Reset internal values
    // Reset structure values
    _ctor_var_reset();
}

void Vcache_simX_VX_dcache_request_inter::__Vconfigure(Vcache_simX__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

Vcache_simX_VX_dcache_request_inter::~Vcache_simX_VX_dcache_request_inter() {
}

//--------------------
// Internal Methods

void Vcache_simX_VX_dcache_request_inter::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vcache_simX_VX_dcache_request_inter::_ctor_var_reset\n"); );
    // Body
    VL_RAND_RESET_W(128,out_cache_driver_in_address);
    out_cache_driver_in_valid = VL_RAND_RESET_I(4);
}
