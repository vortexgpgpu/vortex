// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary design header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef _Vvortex_H_
#define _Vvortex_H_

#include "verilated_heavy.h"

class Vvortex__Syms;

//----------

VL_MODULE(Vvortex) {
  public:
    
    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    // Begin mtask footprint  all: 
    VL_IN8(clk,0,0);
    VL_IN8(reset,0,0);
    VL_OUT8(fe_delay,0,0);
    VL_IN(fe_instruction,31,0);
    VL_OUT(curr_PC,31,0);
    VL_OUT(de_instruction,31,0);
    
    // LOCAL SIGNALS
    // Internals; generally not touched by application code
    // Begin mtask footprint  all: 
    VL_SIG8(vortex__DOT__vx_fetch__DOT__stall_reg,0,0);
    VL_SIG8(vortex__DOT__vx_fetch__DOT__delay_reg,0,0);
    VL_SIG8(vortex__DOT__vx_fetch__DOT__state,4,0);
    VL_SIG8(vortex__DOT__vx_fetch__DOT__prev_debug,0,0);
    VL_SIG(vortex__DOT__vx_fetch__DOT__old,31,0);
    VL_SIG(vortex__DOT__vx_fetch__DOT__real_PC,31,0);
    VL_SIG(vortex__DOT__vx_fetch__DOT__JAL_reg,31,0);
    VL_SIG(vortex__DOT__vx_fetch__DOT__BR_reg,31,0);
    VL_SIG(vortex__DOT__vx_fetch__DOT__PC_to_use,31,0);
    VL_SIG(vortex__DOT__vx_f_d_reg__DOT__instruction,31,0);
    
    // LOCAL VARIABLES
    // Internals; generally not touched by application code
    // Begin mtask footprint  all: 
    VL_SIG8(__Vclklast__TOP__clk,0,0);
    VL_SIG8(__Vclklast__TOP__reset,0,0);
    
    // INTERNAL VARIABLES
    // Internals; generally not touched by application code
    Vvortex__Syms* __VlSymsp;  // Symbol table
    
    // PARAMETERS
    // Parameters marked /*verilator public*/ for use by application code
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(Vvortex);  ///< Copying not allowed
  public:
    /// Construct the model; called by application code
    /// The special name  may be used to make a wrapper with a
    /// single model invisible with respect to DPI scope names.
    Vvortex(const char* name="TOP");
    /// Destroy the model; called (often implicitly) by application code
    ~Vvortex();
    
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval();
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    
    // INTERNAL METHODS
  private:
    static void _eval_initial_loop(Vvortex__Syms* __restrict vlSymsp);
  public:
    void __Vconfigure(Vvortex__Syms* symsp, bool first);
  private:
    static QData _change_request(Vvortex__Syms* __restrict vlSymsp);
    void _ctor_var_reset();
  public:
    static void _eval(Vvortex__Syms* __restrict vlSymsp);
  private:
#ifdef VL_DEBUG
    void _eval_debug_assertions();
#endif // VL_DEBUG
  public:
    static void _eval_initial(Vvortex__Syms* __restrict vlSymsp);
    static void _eval_settle(Vvortex__Syms* __restrict vlSymsp);
    static void _initial__TOP__4(Vvortex__Syms* __restrict vlSymsp);
    static void _sequent__TOP__2(Vvortex__Syms* __restrict vlSymsp);
    static void _sequent__TOP__3(Vvortex__Syms* __restrict vlSymsp);
    static void _sequent__TOP__5(Vvortex__Syms* __restrict vlSymsp);
    static void _settle__TOP__1(Vvortex__Syms* __restrict vlSymsp);
    static void _settle__TOP__6(Vvortex__Syms* __restrict vlSymsp);
} VL_ATTR_ALIGNED(128);

#endif // guard
