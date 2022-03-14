// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "VVX_raster_extents__Syms.h"


void VVX_raster_extents::traceChgTop0(void* userp, VerilatedVcd* tracep) {
    VVX_raster_extents__Syms* __restrict vlSymsp = static_cast<VVX_raster_extents__Syms*>(userp);
    VVX_raster_extents* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    {
        vlTOPp->traceChgSub0(userp, tracep);
    }
}

void VVX_raster_extents::traceChgSub0(void* userp, VerilatedVcd* tracep) {
    VVX_raster_extents__Syms* __restrict vlSymsp = static_cast<VVX_raster_extents__Syms*>(userp);
    VVX_raster_extents* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    vluint32_t* const oldp = tracep->oldp(vlSymsp->__Vm_baseCode + 1);
    if (false && oldp) {}  // Prevent unused
    // Body
    {
        tracep->chgIData(oldp+0,(vlTOPp->edges[0]),32);
        tracep->chgIData(oldp+1,(vlTOPp->edges[1]),32);
        tracep->chgIData(oldp+2,(vlTOPp->edges[2]),32);
        tracep->chgIData(oldp+3,(vlTOPp->extents),32);
    }
}

void VVX_raster_extents::traceCleanup(void* userp, VerilatedVcd* /*unused*/) {
    VVX_raster_extents__Syms* __restrict vlSymsp = static_cast<VVX_raster_extents__Syms*>(userp);
    VVX_raster_extents* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    {
        vlSymsp->__Vm_activity = false;
        vlTOPp->__Vm_traceActivity[0U] = 0U;
    }
}
