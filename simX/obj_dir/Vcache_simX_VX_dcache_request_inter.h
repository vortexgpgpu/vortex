// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vcache_simX.h for the primary calling header

#ifndef _Vcache_simX_VX_dcache_request_inter_H_
#define _Vcache_simX_VX_dcache_request_inter_H_

#include "verilated.h"

class Vcache_simX__Syms;
class VerilatedVcd;

//----------

VL_MODULE(Vcache_simX_VX_dcache_request_inter) {
  public:
    
    // PORTS
    
    // LOCAL SIGNALS
    CData/*3:0*/ out_cache_driver_in_valid;
    WData/*31:0*/ out_cache_driver_in_address[4];
    
    // LOCAL VARIABLES
    
    // INTERNAL VARIABLES
  private:
    Vcache_simX__Syms* __VlSymsp;  // Symbol table
  public:
    
    // PARAMETERS
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(Vcache_simX_VX_dcache_request_inter);  ///< Copying not allowed
  public:
    Vcache_simX_VX_dcache_request_inter(const char* name = "TOP");
    ~Vcache_simX_VX_dcache_request_inter();
    
    // API METHODS
    
    // INTERNAL METHODS
    void __Vconfigure(Vcache_simX__Syms* symsp, bool first);
  private:
    void _ctor_var_reset() VL_ATTR_COLD;
  public:
    static void traceInit(VerilatedVcd* vcdp, void* userthis, uint32_t code);
    static void traceFull(VerilatedVcd* vcdp, void* userthis, uint32_t code);
    static void traceChg(VerilatedVcd* vcdp, void* userthis, uint32_t code);
} VL_ATTR_ALIGNED(128);

#endif // guard
