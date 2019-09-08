// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVortex.h for the primary calling header

#include "VVortex_VX_forward_reqeust_inter.h"
#include "VVortex__Syms.h"


//--------------------
// STATIC VARIABLES


//--------------------

VL_CTOR_IMP(VVortex_VX_forward_reqeust_inter) {
    // Reset internal values
    // Reset structure values
    _ctor_var_reset();
}

void VVortex_VX_forward_reqeust_inter::__Vconfigure(VVortex__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

VVortex_VX_forward_reqeust_inter::~VVortex_VX_forward_reqeust_inter() {
}

//--------------------
// Internal Methods

void VVortex_VX_forward_reqeust_inter::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VVortex_VX_forward_reqeust_inter::_ctor_var_reset\n"); );
    // Body
    src1 = VL_RAND_RESET_I(5);
    src2 = VL_RAND_RESET_I(5);
    warp_num = VL_RAND_RESET_I(4);
}
