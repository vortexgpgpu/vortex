// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See VVX_raster_req_switch.h for the primary calling header

#ifndef _VVX_RASTER_REQ_SWITCH___024UNIT_H_
#define _VVX_RASTER_REQ_SWITCH___024UNIT_H_  // guard

#include "verilated_heavy.h"
#include "VVX_raster_req_switch__Dpi.h"

//==========

class VVX_raster_req_switch__Syms;
class VVX_raster_req_switch_VerilatedVcd;


//----------

VL_MODULE(VVX_raster_req_switch___024unit) {
  public:
    
    // INTERNAL VARIABLES
  private:
    VVX_raster_req_switch__Syms* __VlSymsp;  // Symbol table
  public:
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(VVX_raster_req_switch___024unit);  ///< Copying not allowed
  public:
    VVX_raster_req_switch___024unit(const char* name = "TOP");
    ~VVX_raster_req_switch___024unit();
    
    // INTERNAL METHODS
    void __Vconfigure(VVX_raster_req_switch__Syms* symsp, bool first);
    void __Vdpiimwrap_dpi_assert_TOP____024unit(IData/*31:0*/ inst, CData/*0:0*/ cond, IData/*31:0*/ delay);
    void __Vdpiimwrap_dpi_idiv_TOP____024unit(CData/*0:0*/ enable, IData/*31:0*/ a, IData/*31:0*/ b, CData/*0:0*/ is_signed, IData/*31:0*/ (&quotient), IData/*31:0*/ (&remainder));
    void __Vdpiimwrap_dpi_imul_TOP____024unit(CData/*0:0*/ enable, IData/*31:0*/ a, IData/*31:0*/ b, CData/*0:0*/ is_signed_a, CData/*0:0*/ is_signed_b, IData/*31:0*/ (&resultl), IData/*31:0*/ (&resulth));
    void __Vdpiimwrap_dpi_register_TOP____024unit(IData/*31:0*/ (&dpi_register__Vfuncrtn));
    void __Vdpiimwrap_dpi_trace_TOP____024unit(std::string format);
    void __Vdpiimwrap_dpi_trace_start_TOP____024unit();
    void __Vdpiimwrap_dpi_trace_stop_TOP____024unit();
  private:
    void _ctor_var_reset() VL_ATTR_COLD;
    static void traceInit(void* userp, VerilatedVcd* tracep, uint32_t code) VL_ATTR_COLD;
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);

//----------


#endif  // guard
