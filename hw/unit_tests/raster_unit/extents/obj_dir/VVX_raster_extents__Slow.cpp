// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVX_raster_extents.h for the primary calling header

#include "VVX_raster_extents.h"
#include "VVX_raster_extents__Syms.h"

#include "verilated_dpi.h"

//==========

VL_CTOR_IMP(VVX_raster_extents) {
    VVX_raster_extents__Syms* __restrict vlSymsp = __VlSymsp = new VVX_raster_extents__Syms(this, name());
    VVX_raster_extents* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    VL_CELL(__PVT____024unit, VVX_raster_extents___024unit);
    // Reset internal values
    
    // Reset structure values
    _ctor_var_reset();
}

void VVX_raster_extents::__Vconfigure(VVX_raster_extents__Syms* vlSymsp, bool first) {
    if (false && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
    if (false && this->__VlSymsp) {}  // Prevent unused
    Verilated::timeunit(-12);
    Verilated::timeprecision(-12);
}

VVX_raster_extents::~VVX_raster_extents() {
    VL_DO_CLEAR(delete __VlSymsp, __VlSymsp = NULL);
}

void VVX_raster_extents::_eval_initial(VVX_raster_extents__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_raster_extents::_eval_initial\n"); );
    VVX_raster_extents* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void VVX_raster_extents::final() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_raster_extents::final\n"); );
    // Variables
    VVX_raster_extents__Syms* __restrict vlSymsp = this->__VlSymsp;
    VVX_raster_extents* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void VVX_raster_extents::_eval_settle(VVX_raster_extents__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_raster_extents::_eval_settle\n"); );
    VVX_raster_extents* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->_combo__TOP__1(vlSymsp);
}

void VVX_raster_extents::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_raster_extents::_ctor_var_reset\n"); );
    // Body
    { int __Vi0=0; for (; __Vi0<3; ++__Vi0) {
            edges[__Vi0] = VL_RAND_RESET_I(32);
    }}
    extents = VL_RAND_RESET_I(32);
    { int __Vi0=0; for (; __Vi0<1; ++__Vi0) {
            __Vm_traceActivity[__Vi0] = VL_RAND_RESET_I(1);
    }}
}
