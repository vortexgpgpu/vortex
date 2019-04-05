// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary design header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef _VVX_shared_memory_H_
#define _VVX_shared_memory_H_

#include "verilated.h"

class VVX_shared_memory__Syms;

//----------

VL_MODULE(VVX_shared_memory) {
  public:
    
    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    // Begin mtask footprint  all: 
    VL_IN8(clk,0,0);
    VL_IN8(in_mem_read,2,0);
    VL_IN8(in_mem_write,2,0);
    VL_IN(in_address[1],31,0);
    VL_IN8(in_valid[1],0,0);
    VL_IN(in_data[1],31,0);
    VL_OUT(out_data[1],31,0);
    
    // LOCAL SIGNALS
    // Internals; generally not touched by application code
    // Begin mtask footprint  all: 
    VL_SIG(VX_shared_memory__DOT__mem[256],31,0);
    
    // LOCAL VARIABLES
    // Internals; generally not touched by application code
    // Begin mtask footprint  all: 
    VL_SIG8(__Vclklast__TOP__clk,0,0);
    
    // INTERNAL VARIABLES
    // Internals; generally not touched by application code
    VVX_shared_memory__Syms* __VlSymsp;  // Symbol table
    
    // PARAMETERS
    // Parameters marked /*verilator public*/ for use by application code
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(VVX_shared_memory);  ///< Copying not allowed
  public:
    /// Construct the model; called by application code
    /// The special name  may be used to make a wrapper with a
    /// single model invisible with respect to DPI scope names.
    VVX_shared_memory(const char* name="TOP");
    /// Destroy the model; called (often implicitly) by application code
    ~VVX_shared_memory();
    
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval();
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    
    // INTERNAL METHODS
  private:
    static void _eval_initial_loop(VVX_shared_memory__Syms* __restrict vlSymsp);
  public:
    void __Vconfigure(VVX_shared_memory__Syms* symsp, bool first);
  private:
    static QData _change_request(VVX_shared_memory__Syms* __restrict vlSymsp);
    void _ctor_var_reset();
  public:
    static void _eval(VVX_shared_memory__Syms* __restrict vlSymsp);
  private:
#ifdef VL_DEBUG
    void _eval_debug_assertions();
#endif // VL_DEBUG
  public:
    static void _eval_initial(VVX_shared_memory__Syms* __restrict vlSymsp);
    static void _eval_settle(VVX_shared_memory__Syms* __restrict vlSymsp);
    static void _sequent__TOP__1(VVX_shared_memory__Syms* __restrict vlSymsp);
} VL_ATTR_ALIGNED(128);

#endif // guard
