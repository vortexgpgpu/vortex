// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary design header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef _VVX_RASTER_REQ_SWITCH_H_
#define _VVX_RASTER_REQ_SWITCH_H_  // guard

#include "verilated_heavy.h"
#include "VVX_raster_req_switch__Dpi.h"

//==========

class VVX_raster_req_switch__Syms;
class VVX_raster_req_switch_VerilatedVcd;
class VVX_raster_req_switch___024unit;


//----------

VL_MODULE(VVX_raster_req_switch) {
  public:
    // CELLS
    // Public to allow access to /*verilator_public*/ items;
    // otherwise the application code can consider these internals.
    VVX_raster_req_switch___024unit* __PVT____024unit;
    
    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    VL_IN8(clk,0,0);
    VL_IN8(reset,0,0);
    VL_IN8(input_valid,0,0);
    VL_IN8(raster_slice_ready,3,0);
    VL_OUT8(out_slice_index,1,0);
    VL_OUT8(ready,0,0);
    VL_OUT8(mem_req_valid,0,0);
    VL_OUT8(mem_req_ready,0,0);
    VL_OUT8(mem_rsp_valid,0,0);
    VL_OUT8(raster_mem_rsp_tag,2,0);
    VL_IN16(x_loc,15,0);
    VL_IN16(y_loc,15,0);
    VL_OUT16(out_x_loc,15,0);
    VL_OUT16(out_y_loc,15,0);
    VL_IN(mem_base_addr,31,0);
    VL_IN(mem_stride,31,0);
    VL_OUTW(mem_req_addr,287,0,9);
    VL_OUTW(mem_rsp_data,287,0,9);
    VL_IN(edge_func_val[3],31,0);
    VL_OUT(out_edges[3][3],31,0);
    VL_OUT(out_edge_func_val[3],31,0);
    VL_OUT(out_extents[3],31,0);
    
    // LOCAL SIGNALS
    // Internals; generally not touched by application code
    CData/*7:0*/ VX_raster_req_switch__DOT__raster_rs_valid;
    CData/*7:0*/ VX_raster_req_switch__DOT__raster_rs_empty;
    CData/*7:0*/ VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan_lo;
    IData/*31:0*/ VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t;
    CData/*7:0*/ VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__0__KET____DOT__shifted;
    CData/*7:0*/ VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__1__KET____DOT__shifted;
    CData/*7:0*/ VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__2__KET____DOT__shifted;
    IData/*23:0*/ VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__indices;
    QData/*44:0*/ VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n;
    CData/*7:0*/ VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan_lo;
    IData/*31:0*/ VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t;
    CData/*7:0*/ VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__0__KET____DOT__shifted;
    CData/*7:0*/ VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__1__KET____DOT__shifted;
    CData/*7:0*/ VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__2__KET____DOT__shifted;
    IData/*23:0*/ VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__indices;
    QData/*44:0*/ VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n;
    CData/*3:0*/ VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan_lo;
    SData/*11:0*/ VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t;
    CData/*3:0*/ VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__0__KET____DOT__shifted;
    CData/*3:0*/ VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__genblk6__DOT__genblk1__BRA__1__KET____DOT__shifted;
    CData/*7:0*/ VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__indices;
    CData/*6:0*/ VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n;
    SData/*13:0*/ VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n;
    SData/*14:0*/ VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n;
    SData/*14:0*/ VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n;
    IData/*31:0*/ VX_raster_req_switch__DOT__unnamedblk1__DOT__i;
    IData/*31:0*/ VX_raster_req_switch__DOT__unnamedblk2__DOT__i;
    IData/*31:0*/ VX_raster_req_switch__DOT__unnamedblk2__DOT__unnamedblk3__DOT__j;
    WData/*415:0*/ VX_raster_req_switch__DOT__raster_rs[8][13];
    
    // LOCAL VARIABLES
    // Internals; generally not touched by application code
    CData/*0:0*/ __Vclklast__TOP__clk;
    IData/*31:0*/ __Vchglast__TOP__VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__scan__DOT__t;
    QData/*44:0*/ __Vchglast__TOP__VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n;
    IData/*31:0*/ __Vchglast__TOP__VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__scan__DOT__t;
    QData/*44:0*/ __Vchglast__TOP__VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n;
    SData/*11:0*/ __Vchglast__TOP__VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__scan__DOT__t;
    CData/*6:0*/ __Vchglast__TOP__VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n;
    SData/*13:0*/ __Vchglast__TOP__VX_raster_req_switch__DOT__raster_ready_select__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__d_n;
    SData/*14:0*/ __Vchglast__TOP__VX_raster_req_switch__DOT__raster_empty_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n;
    SData/*14:0*/ __Vchglast__TOP__VX_raster_req_switch__DOT__raster_request_rs__DOT__genblk5__DOT__lzc__DOT__find_first__DOT__s_n;
    IData/*31:0*/ VX_raster_req_switch__DOT____Vlvbound1;
    IData/*31:0*/ VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__0__KET____DOT__extent_calc__edges[3];
    IData/*31:0*/ VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__1__KET____DOT__extent_calc__edges[3];
    IData/*31:0*/ VX_raster_req_switch__DOT____Vcellinp__genblk1__BRA__2__KET____DOT__extent_calc__edges[3];
    CData/*0:0*/ __Vm_traceActivity[3];
    
    // INTERNAL VARIABLES
    // Internals; generally not touched by application code
    VVX_raster_req_switch__Syms* __VlSymsp;  // Symbol table
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(VVX_raster_req_switch);  ///< Copying not allowed
  public:
    /// Construct the model; called by application code
    /// The special name  may be used to make a wrapper with a
    /// single model invisible with respect to DPI scope names.
    VVX_raster_req_switch(const char* name = "TOP");
    /// Destroy the model; called (often implicitly) by application code
    ~VVX_raster_req_switch();
    /// Trace signals in the model; called by application code
    void trace(VerilatedVcdC* tfp, int levels, int options = 0);
    
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval() { eval_step(); }
    /// Evaluate when calling multiple units/models per time step.
    void eval_step();
    /// Evaluate at end of a timestep for tracing, when using eval_step().
    /// Application must call after all eval() and before time changes.
    void eval_end_step() {}
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    
    // INTERNAL METHODS
  private:
    static void _eval_initial_loop(VVX_raster_req_switch__Syms* __restrict vlSymsp);
  public:
    void __Vconfigure(VVX_raster_req_switch__Syms* symsp, bool first);
  private:
    static QData _change_request(VVX_raster_req_switch__Syms* __restrict vlSymsp);
    static QData _change_request_1(VVX_raster_req_switch__Syms* __restrict vlSymsp);
  public:
    static void _combo__TOP__2(VVX_raster_req_switch__Syms* __restrict vlSymsp);
    static void _combo__TOP__4(VVX_raster_req_switch__Syms* __restrict vlSymsp);
  private:
    void _ctor_var_reset() VL_ATTR_COLD;
  public:
    static void _eval(VVX_raster_req_switch__Syms* __restrict vlSymsp);
  private:
#ifdef VL_DEBUG
    void _eval_debug_assertions();
#endif  // VL_DEBUG
  public:
    static void _eval_initial(VVX_raster_req_switch__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    static void _eval_settle(VVX_raster_req_switch__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    static void _sequent__TOP__3(VVX_raster_req_switch__Syms* __restrict vlSymsp);
    static void _settle__TOP__1(VVX_raster_req_switch__Syms* __restrict vlSymsp) VL_ATTR_COLD;
  private:
    static void traceChgSub0(void* userp, VerilatedVcd* tracep);
    static void traceChgTop0(void* userp, VerilatedVcd* tracep);
    static void traceCleanup(void* userp, VerilatedVcd* /*unused*/);
    static void traceFullSub0(void* userp, VerilatedVcd* tracep) VL_ATTR_COLD;
    static void traceFullTop0(void* userp, VerilatedVcd* tracep) VL_ATTR_COLD;
    static void traceInitSub0(void* userp, VerilatedVcd* tracep) VL_ATTR_COLD;
    static void traceInitTop(void* userp, VerilatedVcd* tracep) VL_ATTR_COLD;
    void traceRegister(VerilatedVcd* tracep) VL_ATTR_COLD;
    static void traceInit(void* userp, VerilatedVcd* tracep, uint32_t code) VL_ATTR_COLD;
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);

//----------


#endif  // guard
