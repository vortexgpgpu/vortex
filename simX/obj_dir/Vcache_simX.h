// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary design header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef _Vcache_simX_H_
#define _Vcache_simX_H_

#include "verilated.h"
#include "Vcache_simX__Inlines.h"
class Vcache_simX__Syms;
class Vcache_simX_cache_simX;
class VerilatedVcd;

//----------

VL_MODULE(Vcache_simX) {
  public:
    // CELLS
    // Public to allow access to /*verilator_public*/ items;
    // otherwise the application code can consider these internals.
    Vcache_simX_cache_simX*	__PVT__v;
    
    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    VL_IN8(clk,0,0);
    VL_IN8(reset,0,0);
    VL_IN8(in_icache_valid_pc_addr,0,0);
    VL_OUT8(out_icache_stall,0,0);
    VL_IN8(in_dcache_mem_read,2,0);
    VL_IN8(in_dcache_mem_write,2,0);
    VL_OUT8(out_dcache_stall,0,0);
    //char	__VpadToAlign7[1];
    VL_IN(in_icache_pc_addr,31,0);
    VL_IN8(in_dcache_in_valid[4],0,0);
    VL_IN(in_dcache_in_address[4],31,0);
    
    // LOCAL SIGNALS
    // Internals; generally not touched by application code
    
    // LOCAL VARIABLES
    // Internals; generally not touched by application code
    VL_SIG8(__Vclklast__TOP__clk,0,0);
    VL_SIG8(__Vclklast__TOP__reset,0,0);
    //char	__VpadToAlign42[2];
    VL_SIG(__Vchglast__TOP__v__dmem_controller__shared_memory__DOT__block_addr,27,0);
    VL_SIG(__Vm_traceActivity,31,0);
    
    // INTERNAL VARIABLES
    // Internals; generally not touched by application code
    Vcache_simX__Syms*	__VlSymsp;		// Symbol table
    
    // PARAMETERS
    // Parameters marked /*verilator public*/ for use by application code
    
    // CONSTRUCTORS
  private:
    Vcache_simX& operator= (const Vcache_simX&);	///< Copying not allowed
    Vcache_simX(const Vcache_simX&);	///< Copying not allowed
  public:
    /// Construct the model; called by application code
    /// The special name  may be used to make a wrapper with a
    /// single model invisible WRT DPI scope names.
    Vcache_simX(const char* name="TOP");
    /// Destroy the model; called (often implicitly) by application code
    ~Vcache_simX();
    /// Trace signals in the model; called by application code
    void trace (VerilatedVcdC* tfp, int levels, int options=0);
    
    // USER METHODS
    
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval();
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    
    // INTERNAL METHODS
  private:
    static void _eval_initial_loop(Vcache_simX__Syms* __restrict vlSymsp);
  public:
    void __Vconfigure(Vcache_simX__Syms* symsp, bool first);
  private:
    static QData	_change_request(Vcache_simX__Syms* __restrict vlSymsp);
  public:
    static void	_combo__TOP__1(Vcache_simX__Syms* __restrict vlSymsp);
    static void	_combo__TOP__3(Vcache_simX__Syms* __restrict vlSymsp);
    static void	_combo__TOP__5(Vcache_simX__Syms* __restrict vlSymsp);
    static void	_eval(Vcache_simX__Syms* __restrict vlSymsp);
    static void	_eval_initial(Vcache_simX__Syms* __restrict vlSymsp);
    static void	_eval_settle(Vcache_simX__Syms* __restrict vlSymsp);
    static void	traceChgThis(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void	traceChgThis__2(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void	traceChgThis__3(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void	traceChgThis__4(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void	traceChgThis__5(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void	traceChgThis__6(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void	traceChgThis__7(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void	traceChgThis__8(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void	traceChgThis__9(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void	traceFullThis(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void	traceFullThis__1(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void	traceInitThis(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void	traceInitThis__1(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void traceInit (VerilatedVcd* vcdp, void* userthis, uint32_t code);
    static void traceFull (VerilatedVcd* vcdp, void* userthis, uint32_t code);
    static void traceChg  (VerilatedVcd* vcdp, void* userthis, uint32_t code);
} VL_ATTR_ALIGNED(128);

#endif  /*guard*/
