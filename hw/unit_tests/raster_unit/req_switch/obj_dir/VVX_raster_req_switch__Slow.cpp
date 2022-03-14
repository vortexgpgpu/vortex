// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVX_raster_req_switch.h for the primary calling header

#include "VVX_raster_req_switch.h"
#include "VVX_raster_req_switch__Syms.h"

#include "verilated_dpi.h"

//==========

VL_CTOR_IMP(VVX_raster_req_switch) {
    VVX_raster_req_switch__Syms* __restrict vlSymsp = __VlSymsp = new VVX_raster_req_switch__Syms(this, name());
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    VL_CELL(__PVT____024unit, VVX_raster_req_switch___024unit);
    // Reset internal values
    
    // Reset structure values
    _ctor_var_reset();
}

void VVX_raster_req_switch::__Vconfigure(VVX_raster_req_switch__Syms* vlSymsp, bool first) {
    if (false && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
    if (false && this->__VlSymsp) {}  // Prevent unused
    Verilated::timeunit(-12);
    Verilated::timeprecision(-12);
}

VVX_raster_req_switch::~VVX_raster_req_switch() {
    VL_DO_CLEAR(delete __VlSymsp, __VlSymsp = NULL);
}

void VVX_raster_req_switch::_settle__TOP__1(VVX_raster_req_switch__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_raster_req_switch::_settle__TOP__1\n"); );
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices 
        = (0xfcU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices 
        = (4U | (0xf3U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices 
        = (0x20U | (0xcfU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices 
        = (0xc0U | (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
        = (0xfffff8U & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices);
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
        = (8U | (0xffffc7U & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
        = (0x80U | (0xfffe3fU & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
        = (0x600U | (0xfff1ffU & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
        = (0x4000U | (0xff8fffU & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
        = (0x28000U | (0xfc7fffU & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
        = (0x180000U | (0xe3ffffU & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
        = (0xe00000U | vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices);
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
        = (0xfffff8U & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices);
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
        = (8U | (0xffffc7U & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
        = (0x80U | (0xfffe3fU & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
        = (0x600U | (0xfff1ffU & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
        = (0x4000U | (0xff8fffU & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
        = (0x28000U | (0xfc7fffU & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
        = (0x180000U | (0xe3ffffU & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
        = (0xe00000U | vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices);
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x77U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (8U & ((IData)(vlTOPp->raster_slice_ready) 
                    << 3U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x6fU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x10U & ((IData)(vlTOPp->raster_slice_ready) 
                       << 3U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x5fU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x20U & ((IData)(vlTOPp->raster_slice_ready) 
                       << 3U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x3fU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x40U & ((IData)(vlTOPp->raster_slice_ready) 
                       << 3U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t 
        = ((0xff0U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t)) 
           | VL_STREAML_FAST_III(32,4,32,(IData)(vlTOPp->raster_slice_ready), 0));
    vlTOPp->ready = ((0U != (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty)) 
                     & (IData)(vlTOPp->mem_req_ready));
    vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges[2U] 
        = vlTOPp->out_edges[0U][2U];
    vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges[1U] 
        = vlTOPp->out_edges[0U][1U];
    vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges[0U] 
        = vlTOPp->out_edges[0U][0U];
    vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges[2U] 
        = vlTOPp->out_edges[1U][2U];
    vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges[1U] 
        = vlTOPp->out_edges[1U][1U];
    vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges[0U] 
        = vlTOPp->out_edges[1U][0U];
    vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges[2U] 
        = vlTOPp->out_edges[2U][2U];
    vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges[1U] 
        = vlTOPp->out_edges[2U][1U];
    vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges[0U] 
        = vlTOPp->out_edges[2U][0U];
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7f7fU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x80U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty) 
                       << 7U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7effU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x100U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty) 
                        << 7U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7dffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x200U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty) 
                        << 7U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7bffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x400U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty) 
                        << 7U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x77ffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x800U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty) 
                        << 7U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x6fffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x1000U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty) 
                         << 7U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x5fffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x2000U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty) 
                         << 7U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x3fffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x4000U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty) 
                         << 7U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7f7fU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x80U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid) 
                       << 7U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7effU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x100U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid) 
                        << 7U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7dffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x200U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid) 
                        << 7U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7bffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x400U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid) 
                        << 7U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x77ffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x800U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid) 
                        << 7U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x6fffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x1000U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid) 
                         << 7U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x5fffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x2000U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid) 
                         << 7U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x3fffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x4000U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid) 
                         << 7U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
        = ((0xffffff00U & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t) 
           | VL_STREAML_FAST_III(32,8,32,(IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty), 0));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
        = ((0xffffff00U & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t) 
           | VL_STREAML_FAST_III(32,8,32,(IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid), 0));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x3f3fU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n)) 
           | (0xc0U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices) 
                       << 6U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x3cffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n)) 
           | (0x300U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices) 
                        << 6U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x33ffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n)) 
           | (0xc00U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices) 
                        << 6U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0xfffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n)) 
           | (0x3000U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices) 
                         << 6U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x3f3fU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n)) 
           | (0xc0U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices) 
                       << 6U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x3cffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n)) 
           | (0x300U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices) 
                        << 6U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x33ffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n)) 
           | (0xc00U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices) 
                        << 6U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0xfffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n)) 
           | (0x3000U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices) 
                         << 6U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1fffff1fffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices))) 
              << 0x15U));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ffff8ffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 3U)))) << 0x18U));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1fffc7ffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 6U)))) << 0x1bU));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ffe3fffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 9U)))) << 0x1eU));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ff1ffffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 0xcU)))) << 0x21U));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1f8fffffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 0xfU)))) << 0x24U));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1c7fffffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 0x12U)))) << 0x27U));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x3ffffffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 0x15U)))) << 0x2aU));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1fffff1fffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices))) 
              << 0x15U));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ffff8ffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 3U)))) << 0x18U));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1fffc7ffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 6U)))) << 0x1bU));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ffe3fffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 9U)))) << 0x1eU));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ff1ffffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 0xcU)))) << 0x21U));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1f8fffffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 0xfU)))) << 0x24U));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1c7fffffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 0x12U)))) << 0x27U));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x3ffffffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 0x15U)))) << 0x2aU));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1fffff1fffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices))) 
              << 0x15U));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ffff8ffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 3U)))) << 0x18U));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1fffc7ffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 6U)))) << 0x1bU));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ffe3fffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 9U)))) << 0x1eU));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ff1ffffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 0xcU)))) << 0x21U));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1f8fffffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 0xfU)))) << 0x24U));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1c7fffffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 0x12U)))) << 0x27U));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x3ffffffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 0x15U)))) << 0x2aU));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1fffff1fffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices))) 
              << 0x15U));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ffff8ffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 3U)))) << 0x18U));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1fffc7ffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 6U)))) << 0x1bU));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ffe3fffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 9U)))) << 0x1eU));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ff1ffffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 0xcU)))) << 0x21U));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1f8fffffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 0xfU)))) << 0x24U));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1c7fffffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 0x12U)))) << 0x27U));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x3ffffffffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices 
                                     >> 0x15U)))) << 0x2aU));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7eU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (1U & (((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                     >> 1U) | ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                               >> 2U))));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7dU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (2U & ((0x3ffffffeU & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                    >> 2U)) | (0x1ffffffeU 
                                               & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                                  >> 3U)))));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7bU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (4U & ((0x1ffffffcU & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                    >> 3U)) | (0xffffffcU 
                                               & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                                  >> 4U)))));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__0__KET____DOT__shifted 
        = (7U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t) 
                 >> 1U));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__1__KET____DOT__shifted 
        = (3U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t) 
                 >> 6U));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo 
        = ((0xeU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo)) 
           | (1U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t) 
                    >> 0xbU)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo 
        = ((0xdU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo)) 
           | (2U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t) 
                    >> 9U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo 
        = ((0xbU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo)) 
           | (4U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t) 
                    >> 7U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo 
        = ((7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo)) 
           | (8U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t) 
                    >> 5U)));
    vlTOPp->out_extents[0U] = (((0U < vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges
                                 [0U]) ? vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges
                                [0U] : 0U) + ((0U < 
                                               vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges
                                               [1U])
                                               ? vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges
                                              [1U] : 0U));
    vlTOPp->out_extents[1U] = (((0U < vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges
                                 [0U]) ? vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges
                                [0U] : 0U) + ((0U < 
                                               vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges
                                               [1U])
                                               ? vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges
                                              [1U] : 0U));
    vlTOPp->out_extents[2U] = (((0U < vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges
                                 [0U]) ? vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges
                                [0U] : 0U) + ((0U < 
                                               vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges
                                               [1U])
                                               ? vlTOPp->VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges
                                              [1U] : 0U));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7ffeU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (1U & (((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                     >> 1U) | ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                               >> 2U))));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7ffdU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (2U & ((0x3ffffffeU & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                    >> 2U)) | (0x1ffffffeU 
                                               & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                                  >> 3U)))));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7ffbU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (4U & ((0x1ffffffcU & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                    >> 3U)) | (0xffffffcU 
                                               & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                                  >> 4U)))));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7ff7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (8U & ((0xffffff8U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                   >> 4U)) | (0x7fffff8U 
                                              & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                                 >> 5U)))));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7fefU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x10U & ((0x7fffff0U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                      >> 5U)) | (0x3fffff0U 
                                                 & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                                    >> 6U)))));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7fdfU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x20U & ((0x3ffffe0U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                      >> 6U)) | (0x1ffffe0U 
                                                 & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                                    >> 7U)))));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7fbfU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x40U & ((0x1ffffc0U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                      >> 7U)) | (0xffffc0U 
                                                 & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                                    >> 8U)))));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7ffeU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (1U & (((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                     >> 1U) | ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                               >> 2U))));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7ffdU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (2U & ((0x3ffffffeU & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                    >> 2U)) | (0x1ffffffeU 
                                               & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                                  >> 3U)))));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7ffbU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (4U & ((0x1ffffffcU & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                    >> 3U)) | (0xffffffcU 
                                               & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                                  >> 4U)))));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7ff7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (8U & ((0xffffff8U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                   >> 4U)) | (0x7fffff8U 
                                              & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                                 >> 5U)))));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7fefU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x10U & ((0x7fffff0U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                      >> 5U)) | (0x3fffff0U 
                                                 & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                                    >> 6U)))));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7fdfU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x20U & ((0x3ffffe0U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                      >> 6U)) | (0x1ffffe0U 
                                                 & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                                    >> 7U)))));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = ((0x7fbfU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)) 
           | (0x40U & ((0x1ffffc0U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                      >> 7U)) | (0xffffc0U 
                                                 & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n) 
                                                    >> 8U)))));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__0__KET____DOT__shifted 
        = (0x7fU & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
                    >> 1U));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__1__KET____DOT__shifted 
        = (0x3fU & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
                    >> 0xaU));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__2__KET____DOT__shifted 
        = (0xfU & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
                   >> 0x14U));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo 
        = ((0xfeU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo)) 
           | (1U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
                    >> 0x1fU)));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo 
        = ((0xfdU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo)) 
           | (2U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
                    >> 0x1dU)));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo 
        = ((0xfbU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo)) 
           | (4U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
                    >> 0x1bU)));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo 
        = ((0xf7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo)) 
           | (8U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
                    >> 0x19U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo 
        = ((0xefU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo)) 
           | (0x10U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
                       >> 0x17U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo 
        = ((0xdfU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo)) 
           | (0x20U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
                       >> 0x15U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo 
        = ((0xbfU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo)) 
           | (0x40U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
                       >> 0x13U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo 
        = ((0x7fU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo)) 
           | (0x80U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
                       >> 0x11U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__0__KET____DOT__shifted 
        = (0x7fU & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
                    >> 1U));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__1__KET____DOT__shifted 
        = (0x3fU & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
                    >> 0xaU));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__2__KET____DOT__shifted 
        = (0xfU & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
                   >> 0x14U));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo 
        = ((0xfeU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo)) 
           | (1U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
                    >> 0x1fU)));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo 
        = ((0xfdU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo)) 
           | (2U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
                    >> 0x1dU)));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo 
        = ((0xfbU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo)) 
           | (4U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
                    >> 0x1bU)));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo 
        = ((0xf7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo)) 
           | (8U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
                    >> 0x19U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo 
        = ((0xefU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo)) 
           | (0x10U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
                       >> 0x17U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo 
        = ((0xdfU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo)) 
           | (0x20U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
                       >> 0x15U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo 
        = ((0xbfU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo)) 
           | (0x40U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
                       >> 0x13U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo 
        = ((0x7fU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo)) 
           | (0x80U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
                       >> 0x11U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x3ffcU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n)) 
           | (3U & ((2U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))
                     ? ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
                        >> 2U) : ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
                                  >> 4U))));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x3ff3U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n)) 
           | (0xcU & (((8U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))
                        ? ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
                           >> 6U) : ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
                                     >> 8U)) << 2U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x3fcfU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n)) 
           | (0x30U & (((0x20U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))
                         ? ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
                            >> 0xaU) : ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
                                        >> 0xcU)) << 4U)));
    vlTOPp->out_slice_index = (3U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ffffffffff8ULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | (IData)((IData)((7U & ((2U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))
                                     ? (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                >> 3U))
                                     : (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                >> 6U)))))));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1fffffffffc7ULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & ((8U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))
                                      ? (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 9U))
                                      : (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0xcU)))))) 
              << 3U));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ffffffffe3fULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & ((0x20U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))
                                      ? (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0xfU))
                                      : (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x12U)))))) 
              << 6U));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ffffffff1ffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & ((0x80U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))
                                      ? (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x15U))
                                      : (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x18U)))))) 
              << 9U));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1fffffff8fffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & ((0x200U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))
                                      ? (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x1bU))
                                      : (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x1eU)))))) 
              << 0xcU));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ffffffc7fffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & ((0x800U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))
                                      ? (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x21U))
                                      : (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x24U)))))) 
              << 0xfU));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1fffffe3ffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & ((0x2000U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))
                                      ? (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x27U))
                                      : (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x2aU)))))) 
              << 0x12U));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ffffffffff8ULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | (IData)((IData)((7U & ((2U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))
                                     ? (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                >> 3U))
                                     : (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                >> 6U)))))));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1fffffffffc7ULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & ((8U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))
                                      ? (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 9U))
                                      : (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0xcU)))))) 
              << 3U));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ffffffffe3fULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & ((0x20U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))
                                      ? (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0xfU))
                                      : (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x12U)))))) 
              << 6U));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ffffffff1ffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & ((0x80U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))
                                      ? (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x15U))
                                      : (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x18U)))))) 
              << 9U));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1fffffff8fffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & ((0x200U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))
                                      ? (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x1bU))
                                      : (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x1eU)))))) 
              << 0xcU));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1ffffffc7fffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & ((0x800U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))
                                      ? (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x21U))
                                      : (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x24U)))))) 
              << 0xfU));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = ((0x1fffffe3ffffULL & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n) 
           | ((QData)((IData)((7U & ((0x2000U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))
                                      ? (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x27U))
                                      : (IData)((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
                                                 >> 0x2aU)))))) 
              << 0x12U));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t 
        = ((0xf0fU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t)) 
           | (0xf0U & (((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t) 
                        | (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__0__KET____DOT__shifted)) 
                       << 4U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t 
        = ((0xffU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t)) 
           | (0xf00U & ((0xffffff00U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t) 
                                        << 4U)) | ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__1__KET____DOT__shifted) 
                                                   << 8U))));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
        = ((0xffff00ffU & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t) 
           | (0xff00U & ((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
                          | (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__0__KET____DOT__shifted)) 
                         << 8U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
        = ((0xff00ffffU & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t) 
           | (0xff0000U & ((0xffff0000U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
                                           << 8U)) 
                           | ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__1__KET____DOT__shifted) 
                              << 0x10U))));
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
        = ((0xffffffU & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t) 
           | (0xff000000U & ((0xff000000U & (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
                                             << 8U)) 
                             | ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__2__KET____DOT__shifted) 
                                << 0x18U))));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
        = ((0xffff00ffU & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t) 
           | (0xff00U & ((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
                          | (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__0__KET____DOT__shifted)) 
                         << 8U)));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
        = ((0xff00ffffU & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t) 
           | (0xff0000U & ((0xffff0000U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
                                           << 8U)) 
                           | ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__1__KET____DOT__shifted) 
                              << 0x10U))));
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
        = ((0xffffffU & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t) 
           | (0xff000000U & ((0xff000000U & (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
                                             << 8U)) 
                             | ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__2__KET____DOT__shifted) 
                                << 0x18U))));
}

void VVX_raster_req_switch::_eval_initial(VVX_raster_req_switch__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_raster_req_switch::_eval_initial\n"); );
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
}

void VVX_raster_req_switch::final() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_raster_req_switch::final\n"); );
    // Variables
    VVX_raster_req_switch__Syms* __restrict vlSymsp = this->__VlSymsp;
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void VVX_raster_req_switch::_eval_settle(VVX_raster_req_switch__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_raster_req_switch::_eval_settle\n"); );
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->_settle__TOP__1(vlSymsp);
    vlTOPp->__Vm_traceActivity[2U] = 1U;
    vlTOPp->__Vm_traceActivity[1U] = 1U;
    vlTOPp->__Vm_traceActivity[0U] = 1U;
}

void VVX_raster_req_switch::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_raster_req_switch::_ctor_var_reset\n"); );
    // Body
    clk = VL_RAND_RESET_I(1);
    reset = VL_RAND_RESET_I(1);
    input_valid = VL_RAND_RESET_I(1);
    x_loc = VL_RAND_RESET_I(16);
    y_loc = VL_RAND_RESET_I(16);
    { int __Vi0=0; for (; __Vi0<3; ++__Vi0) {
            edge_func_val[__Vi0] = VL_RAND_RESET_I(32);
    }}
    mem_base_addr = VL_RAND_RESET_I(32);
    mem_stride = VL_RAND_RESET_I(32);
    raster_slice_ready = VL_RAND_RESET_I(4);
    out_x_loc = VL_RAND_RESET_I(16);
    out_y_loc = VL_RAND_RESET_I(16);
    { int __Vi0=0; for (; __Vi0<3; ++__Vi0) {
            { int __Vi1=0; for (; __Vi1<3; ++__Vi1) {
                    out_edges[__Vi0][__Vi1] = VL_RAND_RESET_I(32);
            }}
    }}
    { int __Vi0=0; for (; __Vi0<3; ++__Vi0) {
            out_edge_func_val[__Vi0] = VL_RAND_RESET_I(32);
    }}
    { int __Vi0=0; for (; __Vi0<3; ++__Vi0) {
            out_extents[__Vi0] = VL_RAND_RESET_I(32);
    }}
    out_slice_index = VL_RAND_RESET_I(2);
    ready = VL_RAND_RESET_I(1);
    mem_req_valid = VL_RAND_RESET_I(1);
    mem_req_ready = VL_RAND_RESET_I(1);
    mem_rsp_valid = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(288, mem_req_addr);
    VL_RAND_RESET_W(288, mem_rsp_data);
    raster_mem_rsp_tag = VL_RAND_RESET_I(3);
    { int __Vi0=0; for (; __Vi0<8; ++__Vi0) {
            VL_RAND_RESET_W(416, VX_raster_req_switch__DOT__raster_rs[__Vi0]);
    }}
    VX_raster_req_switch__DOT__raster_rs_valid = VL_RAND_RESET_I(8);
    VX_raster_req_switch__DOT__raster_rs_empty = VL_RAND_RESET_I(8);
    VX_raster_req_switch__DOT__unnamedblk1__DOT__i = 0;
    VX_raster_req_switch__DOT__unnamedblk2__DOT__i = 0;
    VX_raster_req_switch__DOT__unnamedblk2__DOT__unnamedblk3__DOT__j = 0;
    { int __Vi0=0; for (; __Vi0<3; ++__Vi0) {
            VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges[__Vi0] = VL_RAND_RESET_I(32);
    }}
    { int __Vi0=0; for (; __Vi0<3; ++__Vi0) {
            VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges[__Vi0] = VL_RAND_RESET_I(32);
    }}
    { int __Vi0=0; for (; __Vi0<3; ++__Vi0) {
            VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges[__Vi0] = VL_RAND_RESET_I(32);
    }}
    VX_raster_req_switch__DOT____Vlvbound1 = VL_RAND_RESET_I(32);
    VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo = VL_RAND_RESET_I(8);
    VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t = VL_RAND_RESET_I(32);
    VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__0__KET____DOT__shifted = VL_RAND_RESET_I(8);
    VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__1__KET____DOT__shifted = VL_RAND_RESET_I(8);
    VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__2__KET____DOT__shifted = VL_RAND_RESET_I(8);
    VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices = VL_RAND_RESET_I(24);
    VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n = VL_RAND_RESET_I(15);
    VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n = VL_RAND_RESET_Q(45);
    VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo = VL_RAND_RESET_I(8);
    VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t = VL_RAND_RESET_I(32);
    VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__0__KET____DOT__shifted = VL_RAND_RESET_I(8);
    VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__1__KET____DOT__shifted = VL_RAND_RESET_I(8);
    VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__2__KET____DOT__shifted = VL_RAND_RESET_I(8);
    VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices = VL_RAND_RESET_I(24);
    VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n = VL_RAND_RESET_I(15);
    VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n = VL_RAND_RESET_Q(45);
    VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo = VL_RAND_RESET_I(4);
    VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t = VL_RAND_RESET_I(12);
    VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__0__KET____DOT__shifted = VL_RAND_RESET_I(4);
    VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__1__KET____DOT__shifted = VL_RAND_RESET_I(4);
    VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices = VL_RAND_RESET_I(8);
    VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n = VL_RAND_RESET_I(7);
    VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n = VL_RAND_RESET_I(14);
    __Vchglast__TOP__VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t = VL_RAND_RESET_I(32);
    __Vchglast__TOP__VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n = VL_RAND_RESET_I(15);
    __Vchglast__TOP__VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n = VL_RAND_RESET_Q(45);
    __Vchglast__TOP__VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t = VL_RAND_RESET_I(32);
    __Vchglast__TOP__VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n = VL_RAND_RESET_I(15);
    __Vchglast__TOP__VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n = VL_RAND_RESET_Q(45);
    __Vchglast__TOP__VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t = VL_RAND_RESET_I(12);
    __Vchglast__TOP__VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n = VL_RAND_RESET_I(7);
    __Vchglast__TOP__VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n = VL_RAND_RESET_I(14);
    { int __Vi0=0; for (; __Vi0<3; ++__Vi0) {
            __Vm_traceActivity[__Vi0] = VL_RAND_RESET_I(1);
    }}
}
