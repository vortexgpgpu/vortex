// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary design header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef _Vrf2_32x128_wm1_H_
#define _Vrf2_32x128_wm1_H_

#include "verilated.h"

class Vrf2_32x128_wm1__Syms;

//----------

VL_MODULE(Vrf2_32x128_wm1) {
  public:
    
    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    // Begin mtask footprint  all: 
    VL_IN8(CLK,0,0);
    VL_IN8(A,4,0);
    VL_IN8(CEN,0,0);
    VL_IN8(DFTRAMBYP,0,0);
    VL_IN8(SE,0,0);
    VL_OUTW(Q_out,127,0,4);
    VL_INW(Q_in,127,0,4);
    
    // LOCAL SIGNALS
    // Internals; generally not touched by application code
    // Begin mtask footprint  all: 
    VL_SIG(rf2_32x128_wm1_error_injection__DOT__fault_entry,16,0);
    VL_SIG(rf2_32x128_wm1_error_injection__DOT__fault_table[16],16,0);
    
    // LOCAL VARIABLES
    // Internals; generally not touched by application code
    // Begin mtask footprint  all: 
    VL_SIG8(__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__list_complete,0,0);
    VL_SIG8(__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__row_address,3,0);
    VL_SIG8(__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__column_address,0,0);
    VL_SIG8(__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__bitPlace,6,0);
    VL_SIG8(__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__fault_type,1,0);
    VL_SIG8(__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__red_fault,1,0);
    VL_SIG8(__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__valid,0,0);
    VL_SIG8(__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__1__fault_type,1,0);
    VL_SIG8(__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__1__bitLoc,6,0);
    VL_SIG8(__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__2__fault_type,1,0);
    VL_SIG8(__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__2__bitLoc,6,0);
    VL_SIGW(__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__Q_output,127,0,4);
    VL_SIG(__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__i,31,0);
    
    // INTERNAL VARIABLES
    // Internals; generally not touched by application code
    Vrf2_32x128_wm1__Syms* __VlSymsp;  // Symbol table
    
    // PARAMETERS
    // Parameters marked /*verilator public*/ for use by application code
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(Vrf2_32x128_wm1);  ///< Copying not allowed
  public:
    /// Construct the model; called by application code
    /// The special name  may be used to make a wrapper with a
    /// single model invisible with respect to DPI scope names.
    Vrf2_32x128_wm1(const char* name="TOP");
    /// Destroy the model; called (often implicitly) by application code
    ~Vrf2_32x128_wm1();
    
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval();
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    
    // INTERNAL METHODS
  private:
    static void _eval_initial_loop(Vrf2_32x128_wm1__Syms* __restrict vlSymsp);
  public:
    void __Vconfigure(Vrf2_32x128_wm1__Syms* symsp, bool first);
  private:
    static QData _change_request(Vrf2_32x128_wm1__Syms* __restrict vlSymsp);
  public:
    static void _combo__TOP__1(Vrf2_32x128_wm1__Syms* __restrict vlSymsp);
  private:
    void _ctor_var_reset() VL_ATTR_COLD;
  public:
    static void _eval(Vrf2_32x128_wm1__Syms* __restrict vlSymsp);
  private:
#ifdef VL_DEBUG
    void _eval_debug_assertions();
#endif // VL_DEBUG
  public:
    static void _eval_initial(Vrf2_32x128_wm1__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    static void _eval_settle(Vrf2_32x128_wm1__Syms* __restrict vlSymsp) VL_ATTR_COLD;
} VL_ATTR_ALIGNED(128);

#endif // guard
