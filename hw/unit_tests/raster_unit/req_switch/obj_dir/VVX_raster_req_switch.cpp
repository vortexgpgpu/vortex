// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVX_raster_req_switch.h for the primary calling header

#include "VVX_raster_req_switch.h"
#include "VVX_raster_req_switch__Syms.h"

#include "verilated_dpi.h"

//==========

void VVX_raster_req_switch::eval_step() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate VVX_raster_req_switch::eval\n"); );
    VVX_raster_req_switch__Syms* __restrict vlSymsp = this->__VlSymsp;  // Setup global symbol table
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
#ifdef VL_DEBUG
    // Debug assertions
    _eval_debug_assertions();
#endif  // VL_DEBUG
    // Initialize
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) _eval_initial_loop(vlSymsp);
    // Evaluate till stable
    int __VclockLoop = 0;
    QData __Vchange = 1;
    do {
        VL_DEBUG_IF(VL_DBG_MSGF("+ Clock loop\n"););
        vlSymsp->__Vm_activity = true;
        _eval(vlSymsp);
        if (VL_UNLIKELY(++__VclockLoop > 100)) {
            // About to fail, so enable debug to see what's not settling.
            // Note you must run make with OPT=-DVL_DEBUG for debug prints.
            int __Vsaved_debug = Verilated::debug();
            Verilated::debug(1);
            __Vchange = _change_request(vlSymsp);
            Verilated::debug(__Vsaved_debug);
            VL_FATAL_MT("../../../rtl/raster_unit/VX_raster_req_switch.sv", 3, "",
                "Verilated model didn't converge\n"
                "- See DIDNOTCONVERGE in the Verilator manual");
        } else {
            __Vchange = _change_request(vlSymsp);
        }
    } while (VL_UNLIKELY(__Vchange));
}

void VVX_raster_req_switch::_eval_initial_loop(VVX_raster_req_switch__Syms* __restrict vlSymsp) {
    vlSymsp->__Vm_didInit = true;
    _eval_initial(vlSymsp);
    vlSymsp->__Vm_activity = true;
    // Evaluate till stable
    int __VclockLoop = 0;
    QData __Vchange = 1;
    do {
        _eval_settle(vlSymsp);
        _eval(vlSymsp);
        if (VL_UNLIKELY(++__VclockLoop > 100)) {
            // About to fail, so enable debug to see what's not settling.
            // Note you must run make with OPT=-DVL_DEBUG for debug prints.
            int __Vsaved_debug = Verilated::debug();
            Verilated::debug(1);
            __Vchange = _change_request(vlSymsp);
            Verilated::debug(__Vsaved_debug);
            VL_FATAL_MT("../../../rtl/raster_unit/VX_raster_req_switch.sv", 3, "",
                "Verilated model didn't DC converge\n"
                "- See DIDNOTCONVERGE in the Verilator manual");
        } else {
            __Vchange = _change_request(vlSymsp);
        }
    } while (VL_UNLIKELY(__Vchange));
}

VL_INLINE_OPT void VVX_raster_req_switch::_combo__TOP__2(VVX_raster_req_switch__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_raster_req_switch::_combo__TOP__2\n"); );
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
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
    vlTOPp->out_slice_index = (3U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n));
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
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__0__KET____DOT__shifted 
        = (7U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t) 
                 >> 1U));
    vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__1__KET____DOT__shifted 
        = (3U & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t) 
                 >> 6U));
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
}

VL_INLINE_OPT void VVX_raster_req_switch::_sequent__TOP__3(VVX_raster_req_switch__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_raster_req_switch::_sequent__TOP__3\n"); );
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    CData/*2:0*/ __Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v0;
    CData/*0:0*/ __Vdlyvset__VX_raster_req_switch__DOT__raster_rs__v0;
    CData/*2:0*/ __Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v1;
    CData/*0:0*/ __Vdlyvset__VX_raster_req_switch__DOT__raster_rs__v1;
    CData/*0:0*/ __Vdlyvset__out_edge_func_val__v0;
    CData/*0:0*/ __Vdlyvset__out_edge_func_val__v3;
    SData/*8:0*/ __Vdlyvlsb__VX_raster_req_switch__DOT__raster_rs__v1;
    WData/*415:0*/ __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[13];
    WData/*287:0*/ __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v1[9];
    IData/*31:0*/ __Vdlyvval__out_edge_func_val__v0;
    IData/*31:0*/ __Vdlyvval__out_edges__v0;
    IData/*31:0*/ __Vdlyvval__out_edge_func_val__v1;
    IData/*31:0*/ __Vdlyvval__out_edges__v1;
    IData/*31:0*/ __Vdlyvval__out_edge_func_val__v2;
    IData/*31:0*/ __Vdlyvval__out_edges__v2;
    IData/*31:0*/ __Vdlyvval__out_edges__v3;
    IData/*31:0*/ __Vdlyvval__out_edges__v4;
    IData/*31:0*/ __Vdlyvval__out_edges__v5;
    IData/*31:0*/ __Vdlyvval__out_edges__v6;
    IData/*31:0*/ __Vdlyvval__out_edges__v7;
    IData/*31:0*/ __Vdlyvval__out_edges__v8;
    IData/*31:0*/ __Vdlyvval__out_edge_func_val__v3;
    IData/*31:0*/ __Vdlyvval__out_edges__v9;
    IData/*31:0*/ __Vdlyvval__out_edge_func_val__v4;
    IData/*31:0*/ __Vdlyvval__out_edges__v10;
    IData/*31:0*/ __Vdlyvval__out_edge_func_val__v5;
    IData/*31:0*/ __Vdlyvval__out_edges__v11;
    IData/*31:0*/ __Vdlyvval__out_edges__v12;
    IData/*31:0*/ __Vdlyvval__out_edges__v13;
    IData/*31:0*/ __Vdlyvval__out_edges__v14;
    IData/*31:0*/ __Vdlyvval__out_edges__v15;
    IData/*31:0*/ __Vdlyvval__out_edges__v16;
    IData/*31:0*/ __Vdlyvval__out_edges__v17;
    // Body
    __Vdlyvset__VX_raster_req_switch__DOT__raster_rs__v0 = 0U;
    __Vdlyvset__VX_raster_req_switch__DOT__raster_rs__v1 = 0U;
    __Vdlyvset__out_edge_func_val__v0 = 0U;
    __Vdlyvset__out_edge_func_val__v3 = 0U;
    if (vlTOPp->reset) {
        vlTOPp->VX_raster_req_switch__DOT__unnamedblk1__DOT__i = 8U;
    }
    if ((1U & (~ (IData)(vlTOPp->reset)))) {
        if ((((IData)(vlTOPp->ready) & (IData)(vlTOPp->input_valid)) 
             & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo) 
                >> 7U))) {
            vlTOPp->VX_raster_req_switch__DOT__unnamedblk2__DOT__unnamedblk3__DOT__j = 3U;
        }
    }
    if ((1U & (~ (IData)(vlTOPp->reset)))) {
        if ((((IData)(vlTOPp->ready) & (IData)(vlTOPp->input_valid)) 
             & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo) 
                >> 7U))) {
            vlTOPp->VX_raster_req_switch__DOT__unnamedblk2__DOT__i = 3U;
        }
    }
    vlTOPp->mem_req_valid = ((~ (IData)(vlTOPp->reset)) 
                             & (((IData)(vlTOPp->ready) 
                                 & (IData)(vlTOPp->input_valid)) 
                                & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo) 
                                   >> 7U)));
    if ((1U & (~ (IData)(vlTOPp->reset)))) {
        if ((((IData)(vlTOPp->ready) & (IData)(vlTOPp->input_valid)) 
             & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo) 
                >> 7U))) {
            vlTOPp->VX_raster_req_switch__DOT____Vlvbound1 
                = vlTOPp->mem_base_addr;
            vlTOPp->mem_req_addr[0U] = vlTOPp->VX_raster_req_switch__DOT____Vlvbound1;
            vlTOPp->VX_raster_req_switch__DOT____Vlvbound1 
                = (vlTOPp->mem_base_addr + vlTOPp->mem_stride);
            vlTOPp->mem_req_addr[1U] = vlTOPp->VX_raster_req_switch__DOT____Vlvbound1;
            vlTOPp->VX_raster_req_switch__DOT____Vlvbound1 
                = (vlTOPp->mem_base_addr + (vlTOPp->mem_stride 
                                            << 1U));
            vlTOPp->mem_req_addr[2U] = vlTOPp->VX_raster_req_switch__DOT____Vlvbound1;
            vlTOPp->VX_raster_req_switch__DOT____Vlvbound1 
                = (vlTOPp->mem_base_addr + ((IData)(3U) 
                                            * vlTOPp->mem_stride));
            vlTOPp->mem_req_addr[3U] = vlTOPp->VX_raster_req_switch__DOT____Vlvbound1;
            vlTOPp->VX_raster_req_switch__DOT____Vlvbound1 
                = (vlTOPp->mem_base_addr + (vlTOPp->mem_stride 
                                            << 2U));
            vlTOPp->mem_req_addr[4U] = vlTOPp->VX_raster_req_switch__DOT____Vlvbound1;
            vlTOPp->VX_raster_req_switch__DOT____Vlvbound1 
                = (vlTOPp->mem_base_addr + ((IData)(5U) 
                                            * vlTOPp->mem_stride));
            vlTOPp->mem_req_addr[5U] = vlTOPp->VX_raster_req_switch__DOT____Vlvbound1;
            vlTOPp->VX_raster_req_switch__DOT____Vlvbound1 
                = (vlTOPp->mem_base_addr + ((IData)(6U) 
                                            * vlTOPp->mem_stride));
            vlTOPp->mem_req_addr[6U] = vlTOPp->VX_raster_req_switch__DOT____Vlvbound1;
            vlTOPp->VX_raster_req_switch__DOT____Vlvbound1 
                = (vlTOPp->mem_base_addr + ((IData)(7U) 
                                            * vlTOPp->mem_stride));
            vlTOPp->mem_req_addr[7U] = vlTOPp->VX_raster_req_switch__DOT____Vlvbound1;
            vlTOPp->VX_raster_req_switch__DOT____Vlvbound1 
                = (vlTOPp->mem_base_addr + (vlTOPp->mem_stride 
                                            << 3U));
            vlTOPp->mem_req_addr[8U] = vlTOPp->VX_raster_req_switch__DOT____Vlvbound1;
        }
    }
    if ((1U & (~ (IData)(vlTOPp->reset)))) {
        if ((((IData)(vlTOPp->ready) & (IData)(vlTOPp->input_valid)) 
             & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo) 
                >> 7U))) {
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[0U] = 0U;
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[1U] = 0U;
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[2U] = 0U;
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[3U] = 0U;
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[4U] = 0U;
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[5U] = 0U;
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[6U] = 0U;
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[7U] = 0U;
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[8U] = 0U;
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[9U] 
                = vlTOPp->edge_func_val[2U];
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[0xaU] 
                = vlTOPp->edge_func_val[1U];
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[0xbU] 
                = (IData)((((QData)((IData)((((IData)(vlTOPp->x_loc) 
                                              << 0x10U) 
                                             | (IData)(vlTOPp->y_loc)))) 
                            << 0x20U) | (QData)((IData)(
                                                        vlTOPp->edge_func_val
                                                        [0U]))));
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[0xcU] 
                = (IData)(((((QData)((IData)((((IData)(vlTOPp->x_loc) 
                                               << 0x10U) 
                                              | (IData)(vlTOPp->y_loc)))) 
                             << 0x20U) | (QData)((IData)(
                                                         vlTOPp->edge_func_val
                                                         [0U]))) 
                           >> 0x20U));
            __Vdlyvset__VX_raster_req_switch__DOT__raster_rs__v0 = 1U;
            __Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v0 
                = (7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n));
        }
        if (vlTOPp->mem_rsp_valid) {
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v1[0U] 
                = vlTOPp->mem_rsp_data[8U];
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v1[1U] 
                = vlTOPp->mem_rsp_data[7U];
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v1[2U] 
                = vlTOPp->mem_rsp_data[6U];
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v1[3U] 
                = vlTOPp->mem_rsp_data[5U];
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v1[4U] 
                = vlTOPp->mem_rsp_data[4U];
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v1[5U] 
                = vlTOPp->mem_rsp_data[3U];
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v1[6U] 
                = vlTOPp->mem_rsp_data[2U];
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v1[7U] 
                = (IData)((((QData)((IData)(vlTOPp->mem_rsp_data[0U])) 
                            << 0x20U) | (QData)((IData)(
                                                        vlTOPp->mem_rsp_data[1U]))));
            __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v1[8U] 
                = (IData)(((((QData)((IData)(vlTOPp->mem_rsp_data[0U])) 
                             << 0x20U) | (QData)((IData)(
                                                         vlTOPp->mem_rsp_data[1U]))) 
                           >> 0x20U));
            __Vdlyvset__VX_raster_req_switch__DOT__raster_rs__v1 = 1U;
            __Vdlyvlsb__VX_raster_req_switch__DOT__raster_rs__v1 = 0U;
            __Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v1 
                = vlTOPp->raster_mem_rsp_tag;
        }
    }
    if (vlTOPp->reset) {
        vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid 
            = (0xfeU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid));
        vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid 
            = (0xfdU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid));
        vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid 
            = (0xfbU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid));
        vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid 
            = (0xf7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid));
        vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid 
            = (0xefU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid));
        vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid 
            = (0xdfU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid));
        vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid 
            = (0xbfU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid));
        vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid 
            = (0x7fU & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid));
    } else {
        if ((((IData)(vlTOPp->ready) & (IData)(vlTOPp->input_valid)) 
             & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo) 
                >> 7U))) {
            vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid 
                = ((~ ((IData)(1U) << (7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n)))) 
                   & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid));
        }
        if ((1U & (((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo) 
                    >> 3U) & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo) 
                              >> 7U)))) {
            vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid 
                = ((~ ((IData)(1U) << (7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n)))) 
                   & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid));
        }
        if (vlTOPp->mem_rsp_valid) {
            vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid 
                = ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid) 
                   | ((IData)(1U) << (IData)(vlTOPp->raster_mem_rsp_tag)));
            if ((1U & (((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo) 
                        >> 7U) & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo) 
                                  >> 3U)))) {
                vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid 
                    = ((~ ((IData)(1U) << (IData)(vlTOPp->raster_mem_rsp_tag))) 
                       & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid));
            }
        }
    }
    if (vlTOPp->reset) {
        vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty 
            = (1U | (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty));
        vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty 
            = (2U | (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty));
        vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty 
            = (4U | (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty));
        vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty 
            = (8U | (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty));
        vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty 
            = (0x10U | (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty));
        vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty 
            = (0x20U | (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty));
        vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty 
            = (0x40U | (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty));
        vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty 
            = (0x80U | (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty));
    } else {
        if ((((IData)(vlTOPp->ready) & (IData)(vlTOPp->input_valid)) 
             & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo) 
                >> 7U))) {
            vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty 
                = ((~ ((IData)(1U) << (7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n)))) 
                   & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty));
        }
        if ((1U & (((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo) 
                    >> 3U) & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo) 
                              >> 7U)))) {
            vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty 
                = ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty) 
                   | ((IData)(1U) << (7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))));
        }
        if (vlTOPp->mem_rsp_valid) {
            if ((1U & (((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo) 
                        >> 7U) & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo) 
                                  >> 3U)))) {
                vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty 
                    = ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty) 
                       | ((IData)(1U) << (IData)(vlTOPp->raster_mem_rsp_tag)));
            }
        }
    }
    if ((1U & (~ (IData)(vlTOPp->reset)))) {
        if ((1U & (((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo) 
                    >> 3U) & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo) 
                              >> 7U)))) {
            vlTOPp->out_x_loc = (0xffffU & (vlTOPp->VX_raster_req_switch__DOT__raster_rs
                                            [(7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))][0xcU] 
                                            >> 0x10U));
            vlTOPp->out_y_loc = (0xffffU & vlTOPp->VX_raster_req_switch__DOT__raster_rs
                                 [(7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))][0xcU]);
            __Vdlyvval__out_edge_func_val__v0 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                [(7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))][0xbU];
            __Vdlyvset__out_edge_func_val__v0 = 1U;
            __Vdlyvval__out_edges__v0 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                [(7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))][8U];
            __Vdlyvval__out_edge_func_val__v1 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                [(7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))][0xaU];
            __Vdlyvval__out_edges__v1 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                [(7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))][7U];
            __Vdlyvval__out_edge_func_val__v2 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                [(7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))][9U];
            __Vdlyvval__out_edges__v2 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                [(7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))][6U];
            __Vdlyvval__out_edges__v3 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                [(7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))][5U];
            __Vdlyvval__out_edges__v4 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                [(7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))][4U];
            __Vdlyvval__out_edges__v5 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                [(7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))][3U];
            __Vdlyvval__out_edges__v6 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                [(7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))][2U];
            __Vdlyvval__out_edges__v7 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                [(7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))][1U];
            __Vdlyvval__out_edges__v8 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                [(7U & (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))][0U];
        }
        if (vlTOPp->mem_rsp_valid) {
            if ((1U & (((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo) 
                        >> 7U) & ((IData)(vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo) 
                                  >> 3U)))) {
                vlTOPp->out_x_loc = (0xffffU & (vlTOPp->VX_raster_req_switch__DOT__raster_rs
                                                [vlTOPp->raster_mem_rsp_tag][0xcU] 
                                                >> 0x10U));
                vlTOPp->out_y_loc = (0xffffU & vlTOPp->VX_raster_req_switch__DOT__raster_rs
                                     [vlTOPp->raster_mem_rsp_tag][0xcU]);
                __Vdlyvval__out_edge_func_val__v3 = 
                    vlTOPp->VX_raster_req_switch__DOT__raster_rs
                    [vlTOPp->raster_mem_rsp_tag][0xbU];
                __Vdlyvset__out_edge_func_val__v3 = 1U;
                __Vdlyvval__out_edges__v9 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                    [vlTOPp->raster_mem_rsp_tag][8U];
                __Vdlyvval__out_edge_func_val__v4 = 
                    vlTOPp->VX_raster_req_switch__DOT__raster_rs
                    [vlTOPp->raster_mem_rsp_tag][0xaU];
                __Vdlyvval__out_edges__v10 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                    [vlTOPp->raster_mem_rsp_tag][7U];
                __Vdlyvval__out_edge_func_val__v5 = 
                    vlTOPp->VX_raster_req_switch__DOT__raster_rs
                    [vlTOPp->raster_mem_rsp_tag][9U];
                __Vdlyvval__out_edges__v11 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                    [vlTOPp->raster_mem_rsp_tag][6U];
                __Vdlyvval__out_edges__v12 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                    [vlTOPp->raster_mem_rsp_tag][5U];
                __Vdlyvval__out_edges__v13 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                    [vlTOPp->raster_mem_rsp_tag][4U];
                __Vdlyvval__out_edges__v14 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                    [vlTOPp->raster_mem_rsp_tag][3U];
                __Vdlyvval__out_edges__v15 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                    [vlTOPp->raster_mem_rsp_tag][2U];
                __Vdlyvval__out_edges__v16 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                    [vlTOPp->raster_mem_rsp_tag][1U];
                __Vdlyvval__out_edges__v17 = vlTOPp->VX_raster_req_switch__DOT__raster_rs
                    [vlTOPp->raster_mem_rsp_tag][0U];
            }
        }
    }
    if (__Vdlyvset__VX_raster_req_switch__DOT__raster_rs__v0) {
        vlTOPp->VX_raster_req_switch__DOT__raster_rs[__Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v0][0U] 
            = __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[0U];
        vlTOPp->VX_raster_req_switch__DOT__raster_rs[__Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v0][1U] 
            = __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[1U];
        vlTOPp->VX_raster_req_switch__DOT__raster_rs[__Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v0][2U] 
            = __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[2U];
        vlTOPp->VX_raster_req_switch__DOT__raster_rs[__Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v0][3U] 
            = __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[3U];
        vlTOPp->VX_raster_req_switch__DOT__raster_rs[__Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v0][4U] 
            = __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[4U];
        vlTOPp->VX_raster_req_switch__DOT__raster_rs[__Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v0][5U] 
            = __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[5U];
        vlTOPp->VX_raster_req_switch__DOT__raster_rs[__Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v0][6U] 
            = __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[6U];
        vlTOPp->VX_raster_req_switch__DOT__raster_rs[__Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v0][7U] 
            = __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[7U];
        vlTOPp->VX_raster_req_switch__DOT__raster_rs[__Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v0][8U] 
            = __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[8U];
        vlTOPp->VX_raster_req_switch__DOT__raster_rs[__Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v0][9U] 
            = __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[9U];
        vlTOPp->VX_raster_req_switch__DOT__raster_rs[__Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v0][0xaU] 
            = __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[0xaU];
        vlTOPp->VX_raster_req_switch__DOT__raster_rs[__Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v0][0xbU] 
            = __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[0xbU];
        vlTOPp->VX_raster_req_switch__DOT__raster_rs[__Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v0][0xcU] 
            = __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v0[0xcU];
    }
    if (__Vdlyvset__VX_raster_req_switch__DOT__raster_rs__v1) {
        VL_ASSIGNSEL_WIIW(288,(IData)(__Vdlyvlsb__VX_raster_req_switch__DOT__raster_rs__v1), 
                          vlTOPp->VX_raster_req_switch__DOT__raster_rs
                          [__Vdlyvdim0__VX_raster_req_switch__DOT__raster_rs__v1], __Vdlyvval__VX_raster_req_switch__DOT__raster_rs__v1);
    }
    if (__Vdlyvset__out_edge_func_val__v0) {
        vlTOPp->out_edge_func_val[0U] = __Vdlyvval__out_edge_func_val__v0;
        vlTOPp->out_edge_func_val[1U] = __Vdlyvval__out_edge_func_val__v1;
        vlTOPp->out_edge_func_val[2U] = __Vdlyvval__out_edge_func_val__v2;
    }
    if (__Vdlyvset__out_edge_func_val__v3) {
        vlTOPp->out_edge_func_val[0U] = __Vdlyvval__out_edge_func_val__v3;
        vlTOPp->out_edge_func_val[1U] = __Vdlyvval__out_edge_func_val__v4;
        vlTOPp->out_edge_func_val[2U] = __Vdlyvval__out_edge_func_val__v5;
    }
    if (__Vdlyvset__out_edge_func_val__v0) {
        vlTOPp->out_edges[0U][0U] = __Vdlyvval__out_edges__v0;
        vlTOPp->out_edges[0U][1U] = __Vdlyvval__out_edges__v1;
        vlTOPp->out_edges[0U][2U] = __Vdlyvval__out_edges__v2;
        vlTOPp->out_edges[1U][0U] = __Vdlyvval__out_edges__v3;
        vlTOPp->out_edges[1U][1U] = __Vdlyvval__out_edges__v4;
        vlTOPp->out_edges[1U][2U] = __Vdlyvval__out_edges__v5;
        vlTOPp->out_edges[2U][0U] = __Vdlyvval__out_edges__v6;
        vlTOPp->out_edges[2U][1U] = __Vdlyvval__out_edges__v7;
        vlTOPp->out_edges[2U][2U] = __Vdlyvval__out_edges__v8;
    }
    if (__Vdlyvset__out_edge_func_val__v3) {
        vlTOPp->out_edges[0U][0U] = __Vdlyvval__out_edges__v9;
        vlTOPp->out_edges[0U][1U] = __Vdlyvval__out_edges__v10;
        vlTOPp->out_edges[0U][2U] = __Vdlyvval__out_edges__v11;
        vlTOPp->out_edges[1U][0U] = __Vdlyvval__out_edges__v12;
        vlTOPp->out_edges[1U][1U] = __Vdlyvval__out_edges__v13;
        vlTOPp->out_edges[1U][2U] = __Vdlyvval__out_edges__v14;
        vlTOPp->out_edges[2U][0U] = __Vdlyvval__out_edges__v15;
        vlTOPp->out_edges[2U][1U] = __Vdlyvval__out_edges__v16;
        vlTOPp->out_edges[2U][2U] = __Vdlyvval__out_edges__v17;
    }
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
    vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
        = ((0xffffff00U & vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t) 
           | VL_STREAML_FAST_III(32,8,32,(IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_valid), 0));
    vlTOPp->ready = ((0U != (IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty)) 
                     & (IData)(vlTOPp->mem_req_ready));
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
    vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
        = ((0xffffff00U & vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t) 
           | VL_STREAML_FAST_III(32,8,32,(IData)(vlTOPp->VX_raster_req_switch__DOT__raster_rs_empty), 0));
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
}

VL_INLINE_OPT void VVX_raster_req_switch::_combo__TOP__4(VVX_raster_req_switch__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_raster_req_switch::_combo__TOP__4\n"); );
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
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
}

void VVX_raster_req_switch::_eval(VVX_raster_req_switch__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_raster_req_switch::_eval\n"); );
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->_combo__TOP__2(vlSymsp);
    vlTOPp->__Vm_traceActivity[1U] = 1U;
    if (((IData)(vlTOPp->clk) & (~ (IData)(vlTOPp->__Vclklast__TOP__clk)))) {
        vlTOPp->_sequent__TOP__3(vlSymsp);
        vlTOPp->__Vm_traceActivity[2U] = 1U;
    }
    vlTOPp->_combo__TOP__4(vlSymsp);
    // Final
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
}

VL_INLINE_OPT QData VVX_raster_req_switch::_change_request(VVX_raster_req_switch__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_raster_req_switch::_change_request\n"); );
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    return (vlTOPp->_change_request_1(vlSymsp));
}

VL_INLINE_OPT QData VVX_raster_req_switch::_change_request_1(VVX_raster_req_switch__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_raster_req_switch::_change_request_1\n"); );
    VVX_raster_req_switch* const __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // Change detection
    QData __req = false;  // Logically a bool
    __req |= ((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t)
         | (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)
         | (vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n)
         | (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t)
         | (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)
         | (vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n)
         | (vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t)
         | (vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n)
         | (vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n));
    VL_DEBUG_IF( if(__req && ((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t))) VL_DBG_MSGF("        CHANGE: ../../../rtl/libs/VX_scan.sv:19: VX_raster_req_switch.raster_empty_rs.genblk5.scan.t\n"); );
    VL_DEBUG_IF( if(__req && ((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))) VL_DBG_MSGF("        CHANGE: ../../../rtl/libs/VX_find_first.sv:19: VX_raster_req_switch.raster_empty_rs.genblk5.lzc.find_first.s_n\n"); );
    VL_DEBUG_IF( if(__req && ((vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))) VL_DBG_MSGF("        CHANGE: ../../../rtl/libs/VX_find_first.sv:20: VX_raster_req_switch.raster_empty_rs.genblk5.lzc.find_first.d_n\n"); );
    VL_DEBUG_IF( if(__req && ((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t))) VL_DBG_MSGF("        CHANGE: ../../../rtl/libs/VX_scan.sv:19: VX_raster_req_switch.raster_request_rs.genblk5.scan.t\n"); );
    VL_DEBUG_IF( if(__req && ((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))) VL_DBG_MSGF("        CHANGE: ../../../rtl/libs/VX_find_first.sv:19: VX_raster_req_switch.raster_request_rs.genblk5.lzc.find_first.s_n\n"); );
    VL_DEBUG_IF( if(__req && ((vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))) VL_DBG_MSGF("        CHANGE: ../../../rtl/libs/VX_find_first.sv:20: VX_raster_req_switch.raster_request_rs.genblk5.lzc.find_first.d_n\n"); );
    VL_DEBUG_IF( if(__req && ((vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t))) VL_DBG_MSGF("        CHANGE: ../../../rtl/libs/VX_scan.sv:19: VX_raster_req_switch.raster_ready_select.genblk5.scan.t\n"); );
    VL_DEBUG_IF( if(__req && ((vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n))) VL_DBG_MSGF("        CHANGE: ../../../rtl/libs/VX_find_first.sv:19: VX_raster_req_switch.raster_ready_select.genblk5.lzc.find_first.s_n\n"); );
    VL_DEBUG_IF( if(__req && ((vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n ^ vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n))) VL_DBG_MSGF("        CHANGE: ../../../rtl/libs/VX_find_first.sv:20: VX_raster_req_switch.raster_ready_select.genblk5.lzc.find_first.d_n\n"); );
    // Final
    vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t 
        = vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t;
    vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n;
    vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = vlTOPp->VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n;
    vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t 
        = vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t;
    vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n;
    vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = vlTOPp->VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n;
    vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t 
        = vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t;
    vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n 
        = vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n;
    vlTOPp->__Vchglast__TOP__VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n 
        = vlTOPp->VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n;
    return __req;
}

#ifdef VL_DEBUG
void VVX_raster_req_switch::_eval_debug_assertions() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_raster_req_switch::_eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((clk & 0xfeU))) {
        Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((reset & 0xfeU))) {
        Verilated::overWidthError("reset");}
    if (VL_UNLIKELY((input_valid & 0xfeU))) {
        Verilated::overWidthError("input_valid");}
    if (VL_UNLIKELY((raster_slice_ready & 0xf0U))) {
        Verilated::overWidthError("raster_slice_ready");}
}
#endif  // VL_DEBUG
