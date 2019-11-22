// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vcache_simX.h for the primary calling header

#ifndef _Vcache_simX_VX_dram_req_rsp_inter__N4_NB4_H_
#define _Vcache_simX_VX_dram_req_rsp_inter__N4_NB4_H_

#include "verilated.h"

class Vcache_simX__Syms;
class VerilatedVcd;

//----------

VL_MODULE(Vcache_simX_VX_dram_req_rsp_inter__N4_NB4) {
  public:
    
    // PORTS
    
    // LOCAL SIGNALS
    WData/*31:0*/ i_m_readdata[16];
    
    // LOCAL VARIABLES
    
    // INTERNAL VARIABLES
  private:
    Vcache_simX__Syms* __VlSymsp;  // Symbol table
  public:
    
    // PARAMETERS
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(Vcache_simX_VX_dram_req_rsp_inter__N4_NB4);  ///< Copying not allowed
  public:
    Vcache_simX_VX_dram_req_rsp_inter__N4_NB4(const char* name = "TOP");
    ~Vcache_simX_VX_dram_req_rsp_inter__N4_NB4();
    
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
