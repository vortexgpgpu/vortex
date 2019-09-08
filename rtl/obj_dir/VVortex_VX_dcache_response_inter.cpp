// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVortex.h for the primary calling header

#include "VVortex_VX_dcache_response_inter.h"
#include "VVortex__Syms.h"


//--------------------
// STATIC VARIABLES


//--------------------

VL_CTOR_IMP(VVortex_VX_dcache_response_inter) {
    // Reset internal values
    // Reset structure values
    _ctor_var_reset();
}

void VVortex_VX_dcache_response_inter::__Vconfigure(VVortex__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

VVortex_VX_dcache_response_inter::~VVortex_VX_dcache_response_inter() {
}

//--------------------
// Internal Methods

void VVortex_VX_dcache_response_inter::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+            VVortex_VX_dcache_response_inter::_ctor_var_reset\n"); );
    // Body
    { int __Vi0=0; for (; __Vi0<4; ++__Vi0) {
	    in_cache_driver_out_data[__Vi0] = VL_RAND_RESET_I(32);
    }}
}
