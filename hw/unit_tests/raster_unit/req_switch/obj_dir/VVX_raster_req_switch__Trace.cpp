// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "VVX_raster_req_switch__Syms.h"


void VVX_raster_req_switch::traceChgTop0(void* userp, VerilatedVcd* tracep) {
    VVX_raster_req_switch__Syms* __restrict vlSymsp = static_cast<VVX_raster_req_switch__Syms*>(userp);
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    if (VL_UNLIKELY(!vlSymsp->__Vm_activity)) return;
    // Body
    {
        vlTOPp->traceChgSub0(userp, tracep);
    }
}

void VVX_raster_req_switch::traceChgSub0(void* userp, VerilatedVcd* tracep) {
    VVX_raster_req_switch__Syms* __restrict vlSymsp = static_cast<VVX_raster_req_switch__Syms*>(userp);
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    vluint32_t* const oldp = tracep->oldp(vlSymsp->__Vm_baseCode + 1);
    if (false && oldp) {}  // Prevent unused
    // Body
    {
        if (VL_UNLIKELY(vlTOPp->__Vm_traceActivity[1U])) {
            tracep->chgCData(oldp+0,((7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))),3);
            tracep->chgCData(oldp+1,((7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))),3);
            tracep->chgBit(oldp+2,((1U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo) 
                                          >> 3U))));
            tracep->chgBit(oldp+3,((1U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo) 
                                          >> 7U))));
            tracep->chgBit(oldp+4,((1U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo) 
                                          >> 7U))));
        }
        if (VL_UNLIKELY(vlTOPp->__Vm_traceActivity[2U])) {
            tracep->chgWData(oldp+5,(vlTOPp->VX_raster_req_switch__DOT__raster_rs[0]),416);
            tracep->chgWData(oldp+18,(vlTOPp->VX_raster_req_switch__DOT__raster_rs[1]),416);
            tracep->chgWData(oldp+31,(vlTOPp->VX_raster_req_switch__DOT__raster_rs[2]),416);
            tracep->chgWData(oldp+44,(vlTOPp->VX_raster_req_switch__DOT__raster_rs[3]),416);
            tracep->chgWData(oldp+57,(vlTOPp->VX_raster_req_switch__DOT__raster_rs[4]),416);
            tracep->chgWData(oldp+70,(vlTOPp->VX_raster_req_switch__DOT__raster_rs[5]),416);
            tracep->chgWData(oldp+83,(vlTOPp->VX_raster_req_switch__DOT__raster_rs[6]),416);
            tracep->chgWData(oldp+96,(vlTOPp->VX_raster_req_switch__DOT__raster_rs[7]),416);
            tracep->chgCData(oldp+109,(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid),8);
            tracep->chgCData(oldp+110,(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty),8);
            tracep->chgIData(oldp+111,(vlTOPp->VX_raster_req_switch__DOT__unnamedblk1__DOT__i),32);
            tracep->chgIData(oldp+112,(vlTOPp->VX_raster_req_switch__DOT__unnamedblk2__DOT__i),32);
            tracep->chgIData(oldp+113,(vlTOPp->VX_raster_req_switch__DOT__unnamedblk2__DOT__unnamedblk3__DOT__j),32);
            tracep->chgIData(oldp+114,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges[0]),32);
            tracep->chgIData(oldp+115,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges[1]),32);
            tracep->chgIData(oldp+116,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges[2]),32);
            tracep->chgIData(oldp+117,((((0U < vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges
                                          [0U]) ? vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges
                                         [0U] : 0U) 
                                        + ((0U < vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges
                                            [1U]) ? 
                                           vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges
                                           [1U] : 0U))),32);
            tracep->chgIData(oldp+118,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges[0]),32);
            tracep->chgIData(oldp+119,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges[1]),32);
            tracep->chgIData(oldp+120,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges[2]),32);
            tracep->chgIData(oldp+121,((((0U < vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges
                                          [0U]) ? vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges
                                         [0U] : 0U) 
                                        + ((0U < vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges
                                            [1U]) ? 
                                           vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges
                                           [1U] : 0U))),32);
            tracep->chgIData(oldp+122,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges[0]),32);
            tracep->chgIData(oldp+123,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges[1]),32);
            tracep->chgIData(oldp+124,(vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges[2]),32);
            tracep->chgIData(oldp+125,((((0U < vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges
                                          [0U]) ? vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges
                                         [0U] : 0U) 
                                        + ((0U < vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges
                                            [1U]) ? 
                                           vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges
                                           [1U] : 0U))),32);
        }
        tracep->chgBit(oldp+126,(vlTOPp->clk));
        tracep->chgBit(oldp+127,(vlTOPp->reset));
        tracep->chgBit(oldp+128,(vlTOPp->input_valid));
        tracep->chgSData(oldp+129,(vlTOPp->x_loc),16);
        tracep->chgSData(oldp+130,(vlTOPp->y_loc),16);
        tracep->chgIData(oldp+131,(vlTOPp->edge_func_val[0]),32);
        tracep->chgIData(oldp+132,(vlTOPp->edge_func_val[1]),32);
        tracep->chgIData(oldp+133,(vlTOPp->edge_func_val[2]),32);
        tracep->chgIData(oldp+134,(vlTOPp->mem_base_addr),32);
        tracep->chgIData(oldp+135,(vlTOPp->mem_stride),32);
        tracep->chgCData(oldp+136,(vlTOPp->raster_slice_ready),4);
        tracep->chgSData(oldp+137,(vlTOPp->out_x_loc),16);
        tracep->chgSData(oldp+138,(vlTOPp->out_y_loc),16);
        tracep->chgIData(oldp+139,(vlTOPp->out_edges
                                   [0U][0U]),32);
        tracep->chgIData(oldp+140,(vlTOPp->out_edges
                                   [0U][1U]),32);
        tracep->chgIData(oldp+141,(vlTOPp->out_edges
                                   [0U][2U]),32);
        tracep->chgIData(oldp+142,(vlTOPp->out_edges
                                   [1U][0U]),32);
        tracep->chgIData(oldp+143,(vlTOPp->out_edges
                                   [1U][1U]),32);
        tracep->chgIData(oldp+144,(vlTOPp->out_edges
                                   [1U][2U]),32);
        tracep->chgIData(oldp+145,(vlTOPp->out_edges
                                   [2U][0U]),32);
        tracep->chgIData(oldp+146,(vlTOPp->out_edges
                                   [2U][1U]),32);
        tracep->chgIData(oldp+147,(vlTOPp->out_edges
                                   [2U][2U]),32);
        tracep->chgIData(oldp+148,(vlTOPp->out_edge_func_val[0]),32);
        tracep->chgIData(oldp+149,(vlTOPp->out_edge_func_val[1]),32);
        tracep->chgIData(oldp+150,(vlTOPp->out_edge_func_val[2]),32);
        tracep->chgIData(oldp+151,(vlTOPp->out_extents[0]),32);
        tracep->chgIData(oldp+152,(vlTOPp->out_extents[1]),32);
        tracep->chgIData(oldp+153,(vlTOPp->out_extents[2]),32);
        tracep->chgCData(oldp+154,(vlTOPp->out_slice_index),2);
        tracep->chgBit(oldp+155,(vlTOPp->ready));
        tracep->chgBit(oldp+156,(vlTOPp->mem_req_valid));
        tracep->chgBit(oldp+157,(vlTOPp->mem_req_ready));
        tracep->chgBit(oldp+158,(vlTOPp->mem_rsp_valid));
        tracep->chgIData(oldp+159,(vlTOPp->mem_req_addr[0U]),32);
        tracep->chgIData(oldp+160,(vlTOPp->mem_req_addr[1U]),32);
        tracep->chgIData(oldp+161,(vlTOPp->mem_req_addr[2U]),32);
        tracep->chgIData(oldp+162,(vlTOPp->mem_req_addr[3U]),32);
        tracep->chgIData(oldp+163,(vlTOPp->mem_req_addr[4U]),32);
        tracep->chgIData(oldp+164,(vlTOPp->mem_req_addr[5U]),32);
        tracep->chgIData(oldp+165,(vlTOPp->mem_req_addr[6U]),32);
        tracep->chgIData(oldp+166,(vlTOPp->mem_req_addr[7U]),32);
        tracep->chgIData(oldp+167,(vlTOPp->mem_req_addr[8U]),32);
        tracep->chgIData(oldp+168,(vlTOPp->mem_rsp_data[0U]),32);
        tracep->chgIData(oldp+169,(vlTOPp->mem_rsp_data[1U]),32);
        tracep->chgIData(oldp+170,(vlTOPp->mem_rsp_data[2U]),32);
        tracep->chgIData(oldp+171,(vlTOPp->mem_rsp_data[3U]),32);
        tracep->chgIData(oldp+172,(vlTOPp->mem_rsp_data[4U]),32);
        tracep->chgIData(oldp+173,(vlTOPp->mem_rsp_data[5U]),32);
        tracep->chgIData(oldp+174,(vlTOPp->mem_rsp_data[6U]),32);
        tracep->chgIData(oldp+175,(vlTOPp->mem_rsp_data[7U]),32);
        tracep->chgIData(oldp+176,(vlTOPp->mem_rsp_data[8U]),32);
        tracep->chgCData(oldp+177,(vlTOPp->raster_mem_rsp_tag),3);
    }
}

void VVX_raster_req_switch::traceCleanup(void* userp, VerilatedVcd* /*unused*/) {
    VVX_raster_req_switch__Syms* __restrict vlSymsp = static_cast<VVX_raster_req_switch__Syms*>(userp);
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    {
        vlSymsp->__Vm_activity = false;
        vlTOPp->__Vm_traceActivity[0U] = 0U;
        vlTOPp->__Vm_traceActivity[1U] = 0U;
        vlTOPp->__Vm_traceActivity[2U] = 0U;
    }
}
