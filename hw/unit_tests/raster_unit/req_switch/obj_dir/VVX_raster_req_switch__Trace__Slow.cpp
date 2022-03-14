// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "VVX_raster_req_switch__Syms.h"


//======================

void VVX_raster_req_switch::trace(VerilatedVcdC* tfp, int, int) {
    tfp->spTrace()->addInitCb(&traceInit, __VlSymsp);
    traceRegister(tfp->spTrace());
}

void VVX_raster_req_switch::traceInit(void* userp, VerilatedVcd* tracep, uint32_t code) {
    // Callback from tracep->open()
    VVX_raster_req_switch__Syms* __restrict vlSymsp = static_cast<VVX_raster_req_switch__Syms*>(userp);
    if (!Verilated::calcUnusedSigs()) {
        VL_FATAL_MT(__FILE__, __LINE__, __FILE__,
                        "Turning on wave traces requires Verilated::traceEverOn(true) call before time 0.");
    }
    vlSymsp->__Vm_baseCode = code;
    tracep->module(vlSymsp->name());
    tracep->scopeEscape(' ');
    VVX_raster_req_switch::traceInitTop(vlSymsp, tracep);
    tracep->scopeEscape('.');
}

//======================


void VVX_raster_req_switch::traceInitTop(void* userp, VerilatedVcd* tracep) {
    VVX_raster_req_switch__Syms* __restrict vlSymsp = static_cast<VVX_raster_req_switch__Syms*>(userp);
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    {
        vlTOPp->traceInitSub0(userp, tracep);
    }
}

void VVX_raster_req_switch::traceInitSub0(void* userp, VerilatedVcd* tracep) {
    VVX_raster_req_switch__Syms* __restrict vlSymsp = static_cast<VVX_raster_req_switch__Syms*>(userp);
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    const int c = vlSymsp->__Vm_baseCode;
    if (false && tracep && c) {}  // Prevent unused
    // Body
    {
        tracep->declBit(c+127,"clk", false,-1);
        tracep->declBit(c+128,"reset", false,-1);
        tracep->declBit(c+129,"input_valid", false,-1);
        tracep->declBus(c+130,"x_loc", false,-1, 15,0);
        tracep->declBus(c+131,"y_loc", false,-1, 15,0);
        {int i; for (i=0; i<3; i++) {
                tracep->declBus(c+132+i*1,"edge_func_val", true,(i+0), 31,0);}}
        tracep->declBus(c+135,"mem_base_addr", false,-1, 31,0);
        tracep->declBus(c+136,"mem_stride", false,-1, 31,0);
        tracep->declBus(c+137,"raster_slice_ready", false,-1, 3,0);
        tracep->declBus(c+138,"out_x_loc", false,-1, 15,0);
        tracep->declBus(c+139,"out_y_loc", false,-1, 15,0);
        tracep->declBus(c+140,"out_edges(0)(0)", false,-1, 31,0);
        tracep->declBus(c+141,"out_edges(0)(1)", false,-1, 31,0);
        tracep->declBus(c+142,"out_edges(0)(2)", false,-1, 31,0);
        tracep->declBus(c+143,"out_edges(1)(0)", false,-1, 31,0);
        tracep->declBus(c+144,"out_edges(1)(1)", false,-1, 31,0);
        tracep->declBus(c+145,"out_edges(1)(2)", false,-1, 31,0);
        tracep->declBus(c+146,"out_edges(2)(0)", false,-1, 31,0);
        tracep->declBus(c+147,"out_edges(2)(1)", false,-1, 31,0);
        tracep->declBus(c+148,"out_edges(2)(2)", false,-1, 31,0);
        {int i; for (i=0; i<3; i++) {
                tracep->declBus(c+149+i*1,"out_edge_func_val", true,(i+0), 31,0);}}
        {int i; for (i=0; i<3; i++) {
                tracep->declBus(c+152+i*1,"out_extents", true,(i+0), 31,0);}}
        tracep->declBus(c+155,"out_slice_index", false,-1, 1,0);
        tracep->declBit(c+156,"ready", false,-1);
        tracep->declBit(c+157,"mem_req_valid", false,-1);
        tracep->declBit(c+158,"mem_req_ready", false,-1);
        tracep->declBit(c+159,"mem_rsp_valid", false,-1);
        tracep->declBus(c+160,"mem_req_addr(0)", false,-1, 31,0);
        tracep->declBus(c+161,"mem_req_addr(1)", false,-1, 31,0);
        tracep->declBus(c+162,"mem_req_addr(2)", false,-1, 31,0);
        tracep->declBus(c+163,"mem_req_addr(3)", false,-1, 31,0);
        tracep->declBus(c+164,"mem_req_addr(4)", false,-1, 31,0);
        tracep->declBus(c+165,"mem_req_addr(5)", false,-1, 31,0);
        tracep->declBus(c+166,"mem_req_addr(6)", false,-1, 31,0);
        tracep->declBus(c+167,"mem_req_addr(7)", false,-1, 31,0);
        tracep->declBus(c+168,"mem_req_addr(8)", false,-1, 31,0);
        tracep->declBus(c+169,"mem_rsp_data(0)", false,-1, 31,0);
        tracep->declBus(c+170,"mem_rsp_data(1)", false,-1, 31,0);
        tracep->declBus(c+171,"mem_rsp_data(2)", false,-1, 31,0);
        tracep->declBus(c+172,"mem_rsp_data(3)", false,-1, 31,0);
        tracep->declBus(c+173,"mem_rsp_data(4)", false,-1, 31,0);
        tracep->declBus(c+174,"mem_rsp_data(5)", false,-1, 31,0);
        tracep->declBus(c+175,"mem_rsp_data(6)", false,-1, 31,0);
        tracep->declBus(c+176,"mem_rsp_data(7)", false,-1, 31,0);
        tracep->declBus(c+177,"mem_rsp_data(8)", false,-1, 31,0);
        tracep->declBus(c+178,"raster_mem_rsp_tag", false,-1, 2,0);
        tracep->declBus(c+179,"VX_raster_req_switch RASTER_SLICE_NUM", false,-1, 31,0);
        tracep->declBus(c+180,"VX_raster_req_switch RASTER_RS_SIZE", false,-1, 31,0);
        tracep->declBus(c+181,"VX_raster_req_switch RASTER_SLICE_BITS", false,-1, 31,0);
        tracep->declBit(c+127,"VX_raster_req_switch clk", false,-1);
        tracep->declBit(c+128,"VX_raster_req_switch reset", false,-1);
        tracep->declBit(c+129,"VX_raster_req_switch input_valid", false,-1);
        tracep->declBus(c+130,"VX_raster_req_switch x_loc", false,-1, 15,0);
        tracep->declBus(c+131,"VX_raster_req_switch y_loc", false,-1, 15,0);
        {int i; for (i=0; i<3; i++) {
                tracep->declBus(c+132+i*1,"VX_raster_req_switch edge_func_val", true,(i+0), 31,0);}}
        tracep->declBus(c+135,"VX_raster_req_switch mem_base_addr", false,-1, 31,0);
        tracep->declBus(c+136,"VX_raster_req_switch mem_stride", false,-1, 31,0);
        tracep->declBus(c+137,"VX_raster_req_switch raster_slice_ready", false,-1, 3,0);
        tracep->declBus(c+138,"VX_raster_req_switch out_x_loc", false,-1, 15,0);
        tracep->declBus(c+139,"VX_raster_req_switch out_y_loc", false,-1, 15,0);
        tracep->declBus(c+140,"VX_raster_req_switch out_edges(0)(0)", false,-1, 31,0);
        tracep->declBus(c+141,"VX_raster_req_switch out_edges(0)(1)", false,-1, 31,0);
        tracep->declBus(c+142,"VX_raster_req_switch out_edges(0)(2)", false,-1, 31,0);
        tracep->declBus(c+143,"VX_raster_req_switch out_edges(1)(0)", false,-1, 31,0);
        tracep->declBus(c+144,"VX_raster_req_switch out_edges(1)(1)", false,-1, 31,0);
        tracep->declBus(c+145,"VX_raster_req_switch out_edges(1)(2)", false,-1, 31,0);
        tracep->declBus(c+146,"VX_raster_req_switch out_edges(2)(0)", false,-1, 31,0);
        tracep->declBus(c+147,"VX_raster_req_switch out_edges(2)(1)", false,-1, 31,0);
        tracep->declBus(c+148,"VX_raster_req_switch out_edges(2)(2)", false,-1, 31,0);
        {int i; for (i=0; i<3; i++) {
                tracep->declBus(c+149+i*1,"VX_raster_req_switch out_edge_func_val", true,(i+0), 31,0);}}
        {int i; for (i=0; i<3; i++) {
                tracep->declBus(c+152+i*1,"VX_raster_req_switch out_extents", true,(i+0), 31,0);}}
        tracep->declBus(c+155,"VX_raster_req_switch out_slice_index", false,-1, 1,0);
        tracep->declBit(c+156,"VX_raster_req_switch ready", false,-1);
        tracep->declBit(c+157,"VX_raster_req_switch mem_req_valid", false,-1);
        tracep->declBit(c+158,"VX_raster_req_switch mem_req_ready", false,-1);
        tracep->declBit(c+159,"VX_raster_req_switch mem_rsp_valid", false,-1);
        tracep->declBus(c+160,"VX_raster_req_switch mem_req_addr(0)", false,-1, 31,0);
        tracep->declBus(c+161,"VX_raster_req_switch mem_req_addr(1)", false,-1, 31,0);
        tracep->declBus(c+162,"VX_raster_req_switch mem_req_addr(2)", false,-1, 31,0);
        tracep->declBus(c+163,"VX_raster_req_switch mem_req_addr(3)", false,-1, 31,0);
        tracep->declBus(c+164,"VX_raster_req_switch mem_req_addr(4)", false,-1, 31,0);
        tracep->declBus(c+165,"VX_raster_req_switch mem_req_addr(5)", false,-1, 31,0);
        tracep->declBus(c+166,"VX_raster_req_switch mem_req_addr(6)", false,-1, 31,0);
        tracep->declBus(c+167,"VX_raster_req_switch mem_req_addr(7)", false,-1, 31,0);
        tracep->declBus(c+168,"VX_raster_req_switch mem_req_addr(8)", false,-1, 31,0);
        tracep->declBus(c+169,"VX_raster_req_switch mem_rsp_data(0)", false,-1, 31,0);
        tracep->declBus(c+170,"VX_raster_req_switch mem_rsp_data(1)", false,-1, 31,0);
        tracep->declBus(c+171,"VX_raster_req_switch mem_rsp_data(2)", false,-1, 31,0);
        tracep->declBus(c+172,"VX_raster_req_switch mem_rsp_data(3)", false,-1, 31,0);
        tracep->declBus(c+173,"VX_raster_req_switch mem_rsp_data(4)", false,-1, 31,0);
        tracep->declBus(c+174,"VX_raster_req_switch mem_rsp_data(5)", false,-1, 31,0);
        tracep->declBus(c+175,"VX_raster_req_switch mem_rsp_data(6)", false,-1, 31,0);
        tracep->declBus(c+176,"VX_raster_req_switch mem_rsp_data(7)", false,-1, 31,0);
        tracep->declBus(c+177,"VX_raster_req_switch mem_rsp_data(8)", false,-1, 31,0);
        tracep->declBus(c+178,"VX_raster_req_switch raster_mem_rsp_tag", false,-1, 2,0);
        tracep->declBus(c+182,"VX_raster_req_switch RASTER_RS_DATA_WIDTH", false,-1, 31,0);
        tracep->declBus(c+183,"VX_raster_req_switch RASTER_RS_INDEX_BITS", false,-1, 31,0);
        {int i; for (i=0; i<8; i++) {
                tracep->declArray(c+6+i*13,"VX_raster_req_switch raster_rs", true,(i+0), 415,0);}}
        tracep->declBus(c+110,"VX_raster_req_switch raster_rs_valid", false,-1, 7,0);
        tracep->declBus(c+111,"VX_raster_req_switch raster_rs_empty", false,-1, 7,0);
        tracep->declBus(c+1,"VX_raster_req_switch raster_rs_empty_index", false,-1, 2,0);
        tracep->declBus(c+2,"VX_raster_req_switch raster_rs_index", false,-1, 2,0);
        tracep->declBit(c+3,"VX_raster_req_switch valid_raster_index", false,-1);
        tracep->declBit(c+4,"VX_raster_req_switch valid_rs_index", false,-1);
        tracep->declBit(c+5,"VX_raster_req_switch valid_rs_empty_index", false,-1);
        tracep->declBus(c+112,"VX_raster_req_switch unnamedblk1 i", false,-1, 31,0);
        tracep->declBus(c+113,"VX_raster_req_switch unnamedblk2 i", false,-1, 31,0);
        tracep->declBus(c+114,"VX_raster_req_switch unnamedblk2 unnamedblk3 j", false,-1, 31,0);
        {int i; for (i=0; i<3; i++) {
                tracep->declBus(c+115+i*1,"VX_raster_req_switch genblk1[0] extent_calc edges", true,(i+0), 31,0);}}
        tracep->declBus(c+118,"VX_raster_req_switch genblk1[0] extent_calc extents", false,-1, 31,0);
        {int i; for (i=0; i<3; i++) {
                tracep->declBus(c+119+i*1,"VX_raster_req_switch genblk1[1] extent_calc edges", true,(i+0), 31,0);}}
        tracep->declBus(c+122,"VX_raster_req_switch genblk1[1] extent_calc extents", false,-1, 31,0);
        {int i; for (i=0; i<3; i++) {
                tracep->declBus(c+123+i*1,"VX_raster_req_switch genblk1[2] extent_calc edges", true,(i+0), 31,0);}}
        tracep->declBus(c+126,"VX_raster_req_switch genblk1[2] extent_calc extents", false,-1, 31,0);
    }
}

void VVX_raster_req_switch::traceRegister(VerilatedVcd* tracep) {
    // Body
    {
        tracep->addFullCb(&traceFullTop0, __VlSymsp);
        tracep->addChgCb(&traceChgTop0, __VlSymsp);
        tracep->addCleanupCb(&traceCleanup, __VlSymsp);
    }
}

void VVX_raster_req_switch::traceFullTop0(void* userp, VerilatedVcd* tracep) {
    VVX_raster_req_switch__Syms* __restrict vlSymsp = static_cast<VVX_raster_req_switch__Syms*>(userp);
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    {
        vlTOPp->traceFullSub0(userp, tracep);
    }
}

void VVX_raster_req_switch::traceFullSub0(void* userp, VerilatedVcd* tracep) {
    VVX_raster_req_switch__Syms* __restrict vlSymsp = static_cast<VVX_raster_req_switch__Syms*>(userp);
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    vluint32_t* const oldp = tracep->oldp(vlSymsp->__Vm_baseCode);
    if (false && oldp) {}  // Prevent unused
    // Body
    {
        tracep->fullCData(oldp+1,((7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))),3);
        tracep->fullCData(oldp+2,((7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))),3);
        tracep->fullBit(oldp+3,((1U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo) 
                                       >> 3U))));
        tracep->fullBit(oldp+4,((1U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo) 
                                       >> 7U))));
        tracep->fullBit(oldp+5,((1U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo) 
                                       >> 7U))));
        tracep->fullWData(oldp+6,(vlTOPp->VX_raster_req_switch__DOT__raster_rs[0]),416);
        tracep->fullWData(oldp+19,(vlTOPp->VX_raster_req_switch__DOT__raster_rs[1]),416);
        tracep->fullWData(oldp+32,(vlTOPp->VX_raster_req_switch__DOT__raster_rs[2]),416);
        tracep->fullWData(oldp+45,(vlTOPp->VX_raster_req_switch__DOT__raster_rs[3]),416);
        tracep->fullWData(oldp+58,(vlTOPp->VX_raster_req_switch__DOT__raster_rs[4]),416);
        tracep->fullWData(oldp+71,(vlTOPp->VX_raster_req_switch__DOT__raster_rs[5]),416);
        tracep->fullWData(oldp+84,(vlTOPp->VX_raster_req_switch__DOT__raster_rs[6]),416);
        tracep->fullWData(oldp+97,(vlTOPp->VX_raster_req_switch__DOT__raster_rs[7]),416);
        tracep->fullCData(oldp+110,(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid),8);
        tracep->fullCData(oldp+111,(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty),8);
        tracep->fullIData(oldp+112,(vlTOPp->VX_raster_req_switch__DOT__unnamedblk1__DOT__i),32);
        tracep->fullIData(oldp+113,(vlTOPp->VX_raster_req_switch__DOT__unnamedblk2__DOT__i),32);
        tracep->fullIData(oldp+114,(vlTOPp->VX_raster_req_switch__DOT__unnamedblk2__DOT__unnamedblk3__DOT__j),32);
        tracep->fullIData(oldp+115,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges[0]),32);
        tracep->fullIData(oldp+116,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges[1]),32);
        tracep->fullIData(oldp+117,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges[2]),32);
        tracep->fullIData(oldp+118,((((0U < vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges
                                       [0U]) ? vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges
                                      [0U] : 0U) + 
                                     ((0U < vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges
                                       [1U]) ? vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges
                                      [1U] : 0U))),32);
        tracep->fullIData(oldp+119,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges[0]),32);
        tracep->fullIData(oldp+120,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges[1]),32);
        tracep->fullIData(oldp+121,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges[2]),32);
        tracep->fullIData(oldp+122,((((0U < vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges
                                       [0U]) ? vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges
                                      [0U] : 0U) + 
                                     ((0U < vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges
                                       [1U]) ? vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges
                                      [1U] : 0U))),32);
        tracep->fullIData(oldp+123,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges[0]),32);
        tracep->fullIData(oldp+124,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges[1]),32);
        tracep->fullIData(oldp+125,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges[2]),32);
        tracep->fullIData(oldp+126,((((0U < vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges
                                       [0U]) ? vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges
                                      [0U] : 0U) + 
                                     ((0U < vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges
                                       [1U]) ? vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges
                                      [1U] : 0U))),32);
        tracep->fullBit(oldp+127,(vlTOPp->clk));
        tracep->fullBit(oldp+128,(vlTOPp->reset));
        tracep->fullBit(oldp+129,(vlTOPp->input_valid));
        tracep->fullSData(oldp+130,(vlTOPp->x_loc),16);
        tracep->fullSData(oldp+131,(vlTOPp->y_loc),16);
        tracep->fullIData(oldp+132,(vlTOPp->edge_func_val[0]),32);
        tracep->fullIData(oldp+133,(vlTOPp->edge_func_val[1]),32);
        tracep->fullIData(oldp+134,(vlTOPp->edge_func_val[2]),32);
        tracep->fullIData(oldp+135,(vlTOPp->mem_base_addr),32);
        tracep->fullIData(oldp+136,(vlTOPp->mem_stride),32);
        tracep->fullCData(oldp+137,(vlTOPp->raster_slice_ready),4);
        tracep->fullSData(oldp+138,(vlTOPp->out_x_loc),16);
        tracep->fullSData(oldp+139,(vlTOPp->out_y_loc),16);
        tracep->fullIData(oldp+140,(vlTOPp->out_edges
                                    [0U][0U]),32);
        tracep->fullIData(oldp+141,(vlTOPp->out_edges
                                    [0U][1U]),32);
        tracep->fullIData(oldp+142,(vlTOPp->out_edges
                                    [0U][2U]),32);
        tracep->fullIData(oldp+143,(vlTOPp->out_edges
                                    [1U][0U]),32);
        tracep->fullIData(oldp+144,(vlTOPp->out_edges
                                    [1U][1U]),32);
        tracep->fullIData(oldp+145,(vlTOPp->out_edges
                                    [1U][2U]),32);
        tracep->fullIData(oldp+146,(vlTOPp->out_edges
                                    [2U][0U]),32);
        tracep->fullIData(oldp+147,(vlTOPp->out_edges
                                    [2U][1U]),32);
        tracep->fullIData(oldp+148,(vlTOPp->out_edges
                                    [2U][2U]),32);
        tracep->fullIData(oldp+149,(vlTOPp->out_edge_func_val[0]),32);
        tracep->fullIData(oldp+150,(vlTOPp->out_edge_func_val[1]),32);
        tracep->fullIData(oldp+151,(vlTOPp->out_edge_func_val[2]),32);
        tracep->fullIData(oldp+152,(vlTOPp->out_extents[0]),32);
        tracep->fullIData(oldp+153,(vlTOPp->out_extents[1]),32);
        tracep->fullIData(oldp+154,(vlTOPp->out_extents[2]),32);
        tracep->fullCData(oldp+155,(vlTOPp->out_slice_index),2);
        tracep->fullBit(oldp+156,(vlTOPp->ready));
        tracep->fullBit(oldp+157,(vlTOPp->mem_req_valid));
        tracep->fullBit(oldp+158,(vlTOPp->mem_req_ready));
        tracep->fullBit(oldp+159,(vlTOPp->mem_rsp_valid));
        tracep->fullIData(oldp+160,(vlTOPp->mem_req_addr[0U]),32);
        tracep->fullIData(oldp+161,(vlTOPp->mem_req_addr[1U]),32);
        tracep->fullIData(oldp+162,(vlTOPp->mem_req_addr[2U]),32);
        tracep->fullIData(oldp+163,(vlTOPp->mem_req_addr[3U]),32);
        tracep->fullIData(oldp+164,(vlTOPp->mem_req_addr[4U]),32);
        tracep->fullIData(oldp+165,(vlTOPp->mem_req_addr[5U]),32);
        tracep->fullIData(oldp+166,(vlTOPp->mem_req_addr[6U]),32);
        tracep->fullIData(oldp+167,(vlTOPp->mem_req_addr[7U]),32);
        tracep->fullIData(oldp+168,(vlTOPp->mem_req_addr[8U]),32);
        tracep->fullIData(oldp+169,(vlTOPp->mem_rsp_data[0U]),32);
        tracep->fullIData(oldp+170,(vlTOPp->mem_rsp_data[1U]),32);
        tracep->fullIData(oldp+171,(vlTOPp->mem_rsp_data[2U]),32);
        tracep->fullIData(oldp+172,(vlTOPp->mem_rsp_data[3U]),32);
        tracep->fullIData(oldp+173,(vlTOPp->mem_rsp_data[4U]),32);
        tracep->fullIData(oldp+174,(vlTOPp->mem_rsp_data[5U]),32);
        tracep->fullIData(oldp+175,(vlTOPp->mem_rsp_data[6U]),32);
        tracep->fullIData(oldp+176,(vlTOPp->mem_rsp_data[7U]),32);
        tracep->fullIData(oldp+177,(vlTOPp->mem_rsp_data[8U]),32);
        tracep->fullCData(oldp+178,(vlTOPp->raster_mem_rsp_tag),3);
        tracep->fullIData(oldp+179,(4U),32);
        tracep->fullIData(oldp+180,(8U),32);
        tracep->fullIData(oldp+181,(2U),32);
        tracep->fullIData(oldp+182,(0x1a0U),32);
        tracep->fullIData(oldp+183,(3U),32);
    }
}
