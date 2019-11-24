// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vcache_simX.h for the primary calling header

#ifndef _Vcache_simX_cache_simX_H_
#define _Vcache_simX_cache_simX_H_

#include "verilated.h"
#include "Vcache_simX__Inlines.h"
class Vcache_simX__Syms;
class Vcache_simX_VX_icache_request_inter;
class Vcache_simX_VX_icache_response_inter;
class Vcache_simX_VX_dram_req_rsp_inter__N1_NB4;
class Vcache_simX_VX_dcache_request_inter;
class Vcache_simX_VX_dcache_response_inter;
class Vcache_simX_VX_dram_req_rsp_inter__N4_NB4;
class Vcache_simX_VX_dmem_controller__V0_VB1000;
class VerilatedVcd;

//----------

VL_MODULE(Vcache_simX_cache_simX) {
  public:
    // CELLS
    Vcache_simX_VX_icache_request_inter*	__PVT__VX_icache_req;
    Vcache_simX_VX_icache_response_inter*	__PVT__VX_icache_rsp;
    Vcache_simX_VX_dram_req_rsp_inter__N1_NB4*	__PVT__VX_dram_req_rsp_icache;
    Vcache_simX_VX_dcache_request_inter*	__PVT__VX_dcache_req;
    Vcache_simX_VX_dcache_response_inter*	__PVT__VX_dcache_rsp;
    Vcache_simX_VX_dram_req_rsp_inter__N4_NB4*	__PVT__VX_dram_req_rsp;
    Vcache_simX_VX_dmem_controller__V0_VB1000*	__PVT__dmem_controller;
    
    // PORTS
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
    VL_SIG8(__PVT__icache_i_m_ready,0,0);
    VL_SIG8(__PVT__dcache_i_m_ready,0,0);
    //char	__VpadToAlign38[2];
    
    // LOCAL VARIABLES
    
    // INTERNAL VARIABLES
  private:
    Vcache_simX__Syms*	__VlSymsp;		// Symbol table
  public:
    
    // PARAMETERS
    
    // CONSTRUCTORS
  private:
    Vcache_simX_cache_simX& operator= (const Vcache_simX_cache_simX&);	///< Copying not allowed
    Vcache_simX_cache_simX(const Vcache_simX_cache_simX&);	///< Copying not allowed
  public:
    Vcache_simX_cache_simX(const char* name="TOP");
    ~Vcache_simX_cache_simX();
    void trace (VerilatedVcdC* tfp, int levels, int options=0);
    
    // USER METHODS
    
    // API METHODS
    
    // INTERNAL METHODS
    void __Vconfigure(Vcache_simX__Syms* symsp, bool first);
    static void	_combo__TOP__v__3(Vcache_simX__Syms* __restrict vlSymsp);
    static void	_sequent__TOP__v__2(Vcache_simX__Syms* __restrict vlSymsp);
    static void	_settle__TOP__v__1(Vcache_simX__Syms* __restrict vlSymsp);
    static void	_settle__TOP__v__4(Vcache_simX__Syms* __restrict vlSymsp);
    static void traceInit (VerilatedVcd* vcdp, void* userthis, uint32_t code);
    static void traceFull (VerilatedVcd* vcdp, void* userthis, uint32_t code);
    static void traceChg  (VerilatedVcd* vcdp, void* userthis, uint32_t code);
} VL_ATTR_ALIGNED(128);

#endif  /*guard*/
