// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVX_raster_req_switch.h for the primary calling header

#include "VVX_raster_req_switch___024unit.h"
#include "VVX_raster_req_switch__Syms.h"

#include "verilated_dpi.h"

//==========

VL_INLINE_OPT void VVX_raster_req_switch___024unit::__Vdpiimwrap_dpi_imul_TOP____024unit(CData/*0:0*/ enable, IData/*31:0*/ a, IData/*31:0*/ b, CData/*0:0*/ is_signed_a, CData/*0:0*/ is_signed_b, IData/*31:0*/ (&resultl), IData/*31:0*/ (&resulth)) {
    VL_DEBUG_IF(VL_DBG_MSGF("+        VVX_raster_req_switch___024unit::__Vdpiimwrap_dpi_imul_TOP____024unit\n"); );
    // Body
    svLogic enable__Vcvt;
    enable__Vcvt = enable;
    int a__Vcvt;
    a__Vcvt = a;
    int b__Vcvt;
    b__Vcvt = b;
    svLogic is_signed_a__Vcvt;
    is_signed_a__Vcvt = is_signed_a;
    svLogic is_signed_b__Vcvt;
    is_signed_b__Vcvt = is_signed_b;
    int resultl__Vcvt;
    int resulth__Vcvt;
    dpi_imul(enable__Vcvt, a__Vcvt, b__Vcvt, is_signed_a__Vcvt, is_signed_b__Vcvt, &resultl__Vcvt, &resulth__Vcvt);
    resultl = resultl__Vcvt;
    resulth = resulth__Vcvt;
}

VL_INLINE_OPT void VVX_raster_req_switch___024unit::__Vdpiimwrap_dpi_idiv_TOP____024unit(CData/*0:0*/ enable, IData/*31:0*/ a, IData/*31:0*/ b, CData/*0:0*/ is_signed, IData/*31:0*/ (&quotient), IData/*31:0*/ (&remainder)) {
    VL_DEBUG_IF(VL_DBG_MSGF("+        VVX_raster_req_switch___024unit::__Vdpiimwrap_dpi_idiv_TOP____024unit\n"); );
    // Body
    svLogic enable__Vcvt;
    enable__Vcvt = enable;
    int a__Vcvt;
    a__Vcvt = a;
    int b__Vcvt;
    b__Vcvt = b;
    svLogic is_signed__Vcvt;
    is_signed__Vcvt = is_signed;
    int quotient__Vcvt;
    int remainder__Vcvt;
    dpi_idiv(enable__Vcvt, a__Vcvt, b__Vcvt, is_signed__Vcvt, &quotient__Vcvt, &remainder__Vcvt);
    quotient = quotient__Vcvt;
    remainder = remainder__Vcvt;
}

VL_INLINE_OPT void VVX_raster_req_switch___024unit::__Vdpiimwrap_dpi_register_TOP____024unit(IData/*31:0*/ (&dpi_register__Vfuncrtn)) {
    VL_DEBUG_IF(VL_DBG_MSGF("+        VVX_raster_req_switch___024unit::__Vdpiimwrap_dpi_register_TOP____024unit\n"); );
    // Body
    int dpi_register__Vfuncrtn__Vcvt;
    dpi_register__Vfuncrtn__Vcvt = dpi_register();
    dpi_register__Vfuncrtn = dpi_register__Vfuncrtn__Vcvt;
}

VL_INLINE_OPT void VVX_raster_req_switch___024unit::__Vdpiimwrap_dpi_assert_TOP____024unit(IData/*31:0*/ inst, CData/*0:0*/ cond, IData/*31:0*/ delay) {
    VL_DEBUG_IF(VL_DBG_MSGF("+        VVX_raster_req_switch___024unit::__Vdpiimwrap_dpi_assert_TOP____024unit\n"); );
    // Body
    int inst__Vcvt;
    inst__Vcvt = inst;
    svLogic cond__Vcvt;
    cond__Vcvt = cond;
    int delay__Vcvt;
    delay__Vcvt = delay;
    dpi_assert(inst__Vcvt, cond__Vcvt, delay__Vcvt);
}

VL_INLINE_OPT void VVX_raster_req_switch___024unit::__Vdpiimwrap_dpi_trace_TOP____024unit(std::string format) {
    VL_DEBUG_IF(VL_DBG_MSGF("+        VVX_raster_req_switch___024unit::__Vdpiimwrap_dpi_trace_TOP____024unit\n"); );
    // Body
    const char* format__Vcvt;
    format__Vcvt = format.c_str();
    dpi_trace(format__Vcvt);
}

VL_INLINE_OPT void VVX_raster_req_switch___024unit::__Vdpiimwrap_dpi_trace_start_TOP____024unit() {
    VL_DEBUG_IF(VL_DBG_MSGF("+        VVX_raster_req_switch___024unit::__Vdpiimwrap_dpi_trace_start_TOP____024unit\n"); );
    // Body
    dpi_trace_start();
}

VL_INLINE_OPT void VVX_raster_req_switch___024unit::__Vdpiimwrap_dpi_trace_stop_TOP____024unit() {
    VL_DEBUG_IF(VL_DBG_MSGF("+        VVX_raster_req_switch___024unit::__Vdpiimwrap_dpi_trace_stop_TOP____024unit\n"); );
    // Body
    dpi_trace_stop();
}
