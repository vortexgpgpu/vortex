// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vcache_simX.h for the primary calling header

#ifndef _Vcache_simX_VX_icache_request_inter_H_
#define _Vcache_simX_VX_icache_request_inter_H_

#include "verilated.h"
#include "Vcache_simX__Inlines.h"
class Vcache_simX__Syms;
class VerilatedVcd;

//----------

VL_MODULE(Vcache_simX_VX_icache_request_inter) {
  public:
    // CELLS
    
    // PORTS
    
    // LOCAL SIGNALS
    
    // LOCAL VARIABLES
    
    // INTERNAL VARIABLES
  private:
    //char	__VpadToAlign12[4];
    Vcache_simX__Syms*	__VlSymsp;		// Symbol table
  public:
    
    // PARAMETERS
    
    // CONSTRUCTORS
  private:
    Vcache_simX_VX_icache_request_inter& operator= (const Vcache_simX_VX_icache_request_inter&);	///< Copying not allowed
    Vcache_simX_VX_icache_request_inter(const Vcache_simX_VX_icache_request_inter&);	///< Copying not allowed
  public:
    Vcache_simX_VX_icache_request_inter(const char* name="TOP");
    ~Vcache_simX_VX_icache_request_inter();
    void trace (VerilatedVcdC* tfp, int levels, int options=0);
    
    // USER METHODS
    
    // API METHODS
    
    // INTERNAL METHODS
    void __Vconfigure(Vcache_simX__Syms* symsp, bool first);
    static void traceInit (VerilatedVcd* vcdp, void* userthis, uint32_t code);
    static void traceFull (VerilatedVcd* vcdp, void* userthis, uint32_t code);
    static void traceChg  (VerilatedVcd* vcdp, void* userthis, uint32_t code);
} VL_ATTR_ALIGNED(128);

#endif  /*guard*/
