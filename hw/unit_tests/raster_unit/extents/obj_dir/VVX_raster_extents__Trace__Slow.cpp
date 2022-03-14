// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "VVX_raster_extents__Syms.h"


//======================

void VVX_raster_extents::trace(VerilatedVcdC* tfp, int, int) {
    tfp->spTrace()->addInitCb(&traceInit, __VlSymsp);
    traceRegister(tfp->spTrace());
}

void VVX_raster_extents::traceInit(void* userp, VerilatedVcd* tracep, uint32_t code) {
    // Callback from tracep->open()
    VVX_raster_extents__Syms* __restrict vlSymsp = static_cast<VVX_raster_extents__Syms*>(userp);
    if (!Verilated::calcUnusedSigs()) {
        VL_FATAL_MT(__FILE__, __LINE__, __FILE__,
                        "Turning on wave traces requires Verilated::traceEverOn(true) call before time 0.");
    }
    vlSymsp->__Vm_baseCode = code;
    tracep->module(vlSymsp->name());
    tracep->scopeEscape(' ');
    VVX_raster_extents::traceInitTop(vlSymsp, tracep);
    tracep->scopeEscape('.');
}

//======================


void VVX_raster_extents::traceInitTop(void* userp, VerilatedVcd* tracep) {
    VVX_raster_extents__Syms* __restrict vlSymsp = static_cast<VVX_raster_extents__Syms*>(userp);
    VVX_raster_extents* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    {
        vlTOPp->traceInitSub0(userp, tracep);
    }
}

void VVX_raster_extents::traceInitSub0(void* userp, VerilatedVcd* tracep) {
    VVX_raster_extents__Syms* __restrict vlSymsp = static_cast<VVX_raster_extents__Syms*>(userp);
    VVX_raster_extents* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    const int c = vlSymsp->__Vm_baseCode;
    if (false && tracep && c) {}  // Prevent unused
    // Body
    {
        {int i; for (i=0; i<3; i++) {
                tracep->declBus(c+1+i*1,"edges", true,(i+0), 31,0);}}
        tracep->declBus(c+4,"extents", false,-1, 31,0);
        tracep->declBus(c+5,"VX_raster_extents RASTER_TILE_SIZE", false,-1, 31,0);
        {int i; for (i=0; i<3; i++) {
                tracep->declBus(c+1+i*1,"VX_raster_extents edges", true,(i+0), 31,0);}}
        tracep->declBus(c+4,"VX_raster_extents extents", false,-1, 31,0);
    }
}

void VVX_raster_extents::traceRegister(VerilatedVcd* tracep) {
    // Body
    {
        tracep->addFullCb(&traceFullTop0, __VlSymsp);
        tracep->addChgCb(&traceChgTop0, __VlSymsp);
        tracep->addCleanupCb(&traceCleanup, __VlSymsp);
    }
}

void VVX_raster_extents::traceFullTop0(void* userp, VerilatedVcd* tracep) {
    VVX_raster_extents__Syms* __restrict vlSymsp = static_cast<VVX_raster_extents__Syms*>(userp);
    VVX_raster_extents* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    {
        vlTOPp->traceFullSub0(userp, tracep);
    }
}

void VVX_raster_extents::traceFullSub0(void* userp, VerilatedVcd* tracep) {
    VVX_raster_extents__Syms* __restrict vlSymsp = static_cast<VVX_raster_extents__Syms*>(userp);
    VVX_raster_extents* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    vluint32_t* const oldp = tracep->oldp(vlSymsp->__Vm_baseCode);
    if (false && oldp) {}  // Prevent unused
    // Body
    {
        tracep->fullIData(oldp+1,(vlTOPp->edges[0]),32);
        tracep->fullIData(oldp+2,(vlTOPp->edges[1]),32);
        tracep->fullIData(oldp+3,(vlTOPp->edges[2]),32);
        tracep->fullIData(oldp+4,(vlTOPp->extents),32);
        tracep->fullIData(oldp+5,(0x40U),32);
    }
}
