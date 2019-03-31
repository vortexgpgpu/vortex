// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary design header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef _VVX_register_file_slave_H_
#define _VVX_register_file_slave_H_

#include "verilated_heavy.h"

class VVX_register_file_slave__Syms;

//----------

VL_MODULE(VVX_register_file_slave) {
  public:
    
    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    // Begin mtask footprint  all: 
    VL_IN8(clk,0,0);
    VL_IN8(in_valid,0,0);
    VL_IN8(in_write_register,0,0);
    VL_IN8(in_rd,4,0);
    VL_IN8(in_src1,4,0);
    VL_IN8(in_src2,4,0);
    VL_IN8(in_clone,0,0);
    VL_IN(in_data,31,0);
    VL_OUT(out_src1_data,31,0);
    VL_OUT(out_src2_data,31,0);
    VL_IN(in_regs[32],31,0);
    
    // LOCAL SIGNALS
    // Internals; generally not touched by application code
    // Begin mtask footprint  all: 
    VL_SIG(VX_register_file_slave__DOT__registers[32],31,0);
    
    // LOCAL VARIABLES
    // Internals; generally not touched by application code
    // Begin mtask footprint  all: 
    VL_SIG8(__Vclklast__TOP__clk,0,0);
    
    // INTERNAL VARIABLES
    // Internals; generally not touched by application code
    VVX_register_file_slave__Syms* __VlSymsp;  // Symbol table
    
    // PARAMETERS
    // Parameters marked /*verilator public*/ for use by application code
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(VVX_register_file_slave);  ///< Copying not allowed
  public:
    /// Construct the model; called by application code
    /// The special name  may be used to make a wrapper with a
    /// single model invisible with respect to DPI scope names.
    VVX_register_file_slave(const char* name="TOP");
    /// Destroy the model; called (often implicitly) by application code
    ~VVX_register_file_slave();
    
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval();
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    
    // INTERNAL METHODS
  private:
    static void _eval_initial_loop(VVX_register_file_slave__Syms* __restrict vlSymsp);
  public:
    void __Vconfigure(VVX_register_file_slave__Syms* symsp, bool first);
  private:
    static QData _change_request(VVX_register_file_slave__Syms* __restrict vlSymsp);
    void _ctor_var_reset();
  public:
    static void _eval(VVX_register_file_slave__Syms* __restrict vlSymsp);
  private:
#ifdef VL_DEBUG
    void _eval_debug_assertions();
#endif // VL_DEBUG
  public:
    static void _eval_initial(VVX_register_file_slave__Syms* __restrict vlSymsp);
    static void _eval_settle(VVX_register_file_slave__Syms* __restrict vlSymsp);
    static void _sequent__TOP__1(VVX_register_file_slave__Syms* __restrict vlSymsp);
    static void _sequent__TOP__2(VVX_register_file_slave__Syms* __restrict vlSymsp);
} VL_ATTR_ALIGNED(128);

#endif // guard
