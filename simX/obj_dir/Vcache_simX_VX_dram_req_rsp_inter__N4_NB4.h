// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vcache_simX.h for the primary calling header

#ifndef _Vcache_simX_VX_dram_req_rsp_inter__N4_NB4_H_
#define _Vcache_simX_VX_dram_req_rsp_inter__N4_NB4_H_

#include "verilated.h"
#include "Vcache_simX__Inlines.h"
class Vcache_simX__Syms;
class VerilatedVcd;

//----------

VL_MODULE(Vcache_simX_VX_dram_req_rsp_inter__N4_NB4) {
  public:
    // CELLS
    
    // PORTS
    
    // LOCAL SIGNALS
    //char	__VpadToAlign4[4];
    VL_SIGW(__PVT__i_m_readdata,511,0,16);
    
    // LOCAL VARIABLES
    
    // INTERNAL VARIABLES
  private:
    Vcache_simX__Syms*	__VlSymsp;		// Symbol table
  public:
    
    // PARAMETERS
    
    // CONSTRUCTORS
  private:
    Vcache_simX_VX_dram_req_rsp_inter__N4_NB4& operator= (const Vcache_simX_VX_dram_req_rsp_inter__N4_NB4&);	///< Copying not allowed
    Vcache_simX_VX_dram_req_rsp_inter__N4_NB4(const Vcache_simX_VX_dram_req_rsp_inter__N4_NB4&);	///< Copying not allowed
  public:
    Vcache_simX_VX_dram_req_rsp_inter__N4_NB4(const char* name="TOP");
    ~Vcache_simX_VX_dram_req_rsp_inter__N4_NB4();
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
