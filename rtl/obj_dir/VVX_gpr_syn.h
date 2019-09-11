// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary design header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef _VVX_gpr_syn_H_
#define _VVX_gpr_syn_H_

#include "verilated.h"

class VVX_gpr_syn__Syms;

//----------

VL_MODULE(VVX_gpr_syn) {
  public:
    
    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    // Begin mtask footprint  all: 
    VL_IN8(clk,0,0);
    VL_IN8(rs1,4,0);
    VL_IN8(rs2,4,0);
    VL_IN8(warp_num,3,0);
    VL_IN8(rd,4,0);
    VL_IN8(wb,1,0);
    VL_IN8(wb_valid,3,0);
    VL_IN8(wb_warp_num,3,0);
    VL_OUT8(out_gpr_stall,0,0);
    VL_INW(write_data,127,0,4);
    VL_OUTW(out_a_reg_data,127,0,4);
    VL_OUTW(out_b_reg_data,127,0,4);
    
    // LOCAL SIGNALS
    // Internals; generally not touched by application code
    // Begin mtask footprint  all: 
    VL_SIG8(VX_gpr_wrapper__DOT__genblk1__BRA__0__KET____DOT__vx_gpr__DOT__write_enable,0,0);
    VL_SIG8(VX_gpr_wrapper__DOT__genblk1__BRA__1__KET____DOT__vx_gpr__DOT__write_enable,0,0);
    VL_SIG8(VX_gpr_wrapper__DOT__genblk1__BRA__2__KET____DOT__vx_gpr__DOT__write_enable,0,0);
    VL_SIG8(VX_gpr_wrapper__DOT__genblk1__BRA__3__KET____DOT__vx_gpr__DOT__write_enable,0,0);
    VL_SIG8(VX_gpr_wrapper__DOT__genblk1__BRA__4__KET____DOT__vx_gpr__DOT__write_enable,0,0);
    VL_SIG8(VX_gpr_wrapper__DOT__genblk1__BRA__5__KET____DOT__vx_gpr__DOT__write_enable,0,0);
    VL_SIG8(VX_gpr_wrapper__DOT__genblk1__BRA__6__KET____DOT__vx_gpr__DOT__write_enable,0,0);
    VL_SIG8(VX_gpr_wrapper__DOT__genblk1__BRA__7__KET____DOT__vx_gpr__DOT__write_enable,0,0);
    VL_SIGW(VX_gpr_wrapper__DOT__temp_a_reg_data,1023,0,32);
    VL_SIGW(VX_gpr_wrapper__DOT__temp_b_reg_data,1023,0,32);
    VL_SIGW(VX_gpr_wrapper__DOT__genblk1__BRA__0__KET____DOT__vx_gpr__DOT__gpr[32],127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT__genblk1__BRA__1__KET____DOT__vx_gpr__DOT__gpr[32],127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT__genblk1__BRA__2__KET____DOT__vx_gpr__DOT__gpr[32],127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT__genblk1__BRA__3__KET____DOT__vx_gpr__DOT__gpr[32],127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT__genblk1__BRA__4__KET____DOT__vx_gpr__DOT__gpr[32],127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT__genblk1__BRA__5__KET____DOT__vx_gpr__DOT__gpr[32],127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT__genblk1__BRA__6__KET____DOT__vx_gpr__DOT__gpr[32],127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT__genblk1__BRA__7__KET____DOT__vx_gpr__DOT__gpr[32],127,0,4);
    
    // LOCAL VARIABLES
    // Internals; generally not touched by application code
    // Begin mtask footprint  all: 
    VL_SIG8(__Vclklast__TOP__clk,0,0);
    VL_SIGW(VX_gpr_wrapper__DOT____Vcellout__genblk1__BRA__0__KET____DOT__vx_gpr__out_b_reg_data,127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT____Vcellout__genblk1__BRA__0__KET____DOT__vx_gpr__out_a_reg_data,127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT____Vcellout__genblk1__BRA__1__KET____DOT__vx_gpr__out_b_reg_data,127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT____Vcellout__genblk1__BRA__1__KET____DOT__vx_gpr__out_a_reg_data,127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT____Vcellout__genblk1__BRA__2__KET____DOT__vx_gpr__out_b_reg_data,127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT____Vcellout__genblk1__BRA__2__KET____DOT__vx_gpr__out_a_reg_data,127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT____Vcellout__genblk1__BRA__3__KET____DOT__vx_gpr__out_b_reg_data,127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT____Vcellout__genblk1__BRA__3__KET____DOT__vx_gpr__out_a_reg_data,127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT____Vcellout__genblk1__BRA__4__KET____DOT__vx_gpr__out_b_reg_data,127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT____Vcellout__genblk1__BRA__4__KET____DOT__vx_gpr__out_a_reg_data,127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT____Vcellout__genblk1__BRA__5__KET____DOT__vx_gpr__out_b_reg_data,127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT____Vcellout__genblk1__BRA__5__KET____DOT__vx_gpr__out_a_reg_data,127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT____Vcellout__genblk1__BRA__6__KET____DOT__vx_gpr__out_b_reg_data,127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT____Vcellout__genblk1__BRA__6__KET____DOT__vx_gpr__out_a_reg_data,127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT____Vcellout__genblk1__BRA__7__KET____DOT__vx_gpr__out_b_reg_data,127,0,4);
    VL_SIGW(VX_gpr_wrapper__DOT____Vcellout__genblk1__BRA__7__KET____DOT__vx_gpr__out_a_reg_data,127,0,4);
    
    // INTERNAL VARIABLES
    // Internals; generally not touched by application code
    VVX_gpr_syn__Syms* __VlSymsp;  // Symbol table
    
    // PARAMETERS
    // Parameters marked /*verilator public*/ for use by application code
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(VVX_gpr_syn);  ///< Copying not allowed
  public:
    /// Construct the model; called by application code
    /// The special name  may be used to make a wrapper with a
    /// single model invisible with respect to DPI scope names.
    VVX_gpr_syn(const char* name="TOP");
    /// Destroy the model; called (often implicitly) by application code
    ~VVX_gpr_syn();
    
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval();
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    
    // INTERNAL METHODS
  private:
    static void _eval_initial_loop(VVX_gpr_syn__Syms* __restrict vlSymsp);
  public:
    void __Vconfigure(VVX_gpr_syn__Syms* symsp, bool first);
  private:
    static QData _change_request(VVX_gpr_syn__Syms* __restrict vlSymsp);
  public:
    static void _combo__TOP__4(VVX_gpr_syn__Syms* __restrict vlSymsp);
  private:
    void _ctor_var_reset() VL_ATTR_COLD;
  public:
    static void _eval(VVX_gpr_syn__Syms* __restrict vlSymsp);
  private:
#ifdef VL_DEBUG
    void _eval_debug_assertions();
#endif // VL_DEBUG
  public:
    static void _eval_initial(VVX_gpr_syn__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    static void _eval_settle(VVX_gpr_syn__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    static void _initial__TOP__1(VVX_gpr_syn__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    static void _sequent__TOP__2(VVX_gpr_syn__Syms* __restrict vlSymsp);
    static void _settle__TOP__3(VVX_gpr_syn__Syms* __restrict vlSymsp) VL_ATTR_COLD;
} VL_ATTR_ALIGNED(128);

#endif // guard
