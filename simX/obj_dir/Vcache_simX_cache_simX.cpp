// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vcache_simX.h for the primary calling header

#include "Vcache_simX_cache_simX.h" // For This
#include "Vcache_simX__Syms.h"

//--------------------
// STATIC VARIABLES


//--------------------

VL_CTOR_IMP(Vcache_simX_cache_simX) {
    VL_CELL (__PVT__VX_icache_req, Vcache_simX_VX_icache_request_inter);
    VL_CELL (__PVT__VX_icache_rsp, Vcache_simX_VX_icache_response_inter);
    VL_CELL (__PVT__VX_dram_req_rsp_icache, Vcache_simX_VX_dram_req_rsp_inter__N1_NB4);
    VL_CELL (__PVT__VX_dcache_req, Vcache_simX_VX_dcache_request_inter);
    VL_CELL (__PVT__VX_dcache_rsp, Vcache_simX_VX_dcache_response_inter);
    VL_CELL (__PVT__VX_dram_req_rsp, Vcache_simX_VX_dram_req_rsp_inter__N4_NB4);
    VL_CELL (__PVT__dmem_controller, Vcache_simX_VX_dmem_controller__V0_VB1000);
    // Reset internal values
    // Reset structure values
    clk = VL_RAND_RESET_I(1);
    reset = VL_RAND_RESET_I(1);
    in_icache_pc_addr = VL_RAND_RESET_I(32);
    in_icache_valid_pc_addr = VL_RAND_RESET_I(1);
    out_icache_stall = VL_RAND_RESET_I(1);
    in_dcache_mem_read = VL_RAND_RESET_I(3);
    in_dcache_mem_write = VL_RAND_RESET_I(3);
    { int __Vi0=0; for (; __Vi0<4; ++__Vi0) {
	    in_dcache_in_valid[__Vi0] = VL_RAND_RESET_I(1);
    }}
    { int __Vi0=0; for (; __Vi0<4; ++__Vi0) {
	    in_dcache_in_address[__Vi0] = VL_RAND_RESET_I(32);
    }}
    out_dcache_stall = VL_RAND_RESET_I(1);
    __PVT__icache_i_m_ready = VL_RAND_RESET_I(1);
    __PVT__dcache_i_m_ready = VL_RAND_RESET_I(1);
}

void Vcache_simX_cache_simX::__Vconfigure(Vcache_simX__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

Vcache_simX_cache_simX::~Vcache_simX_cache_simX() {
}

//--------------------
// Internal Methods

void Vcache_simX_cache_simX::_settle__TOP__v__1(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_PRINTF("      Vcache_simX_cache_simX::_settle__TOP__v__1\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_valid 
	= ((0xeU & (IData)(vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_valid)) 
	   | vlSymsp->TOP__v.in_dcache_in_valid[0U]);
    vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_valid 
	= ((0xdU & (IData)(vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_valid)) 
	   | (vlSymsp->TOP__v.in_dcache_in_valid[1U] 
	      << 1U));
    vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_valid 
	= ((0xbU & (IData)(vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_valid)) 
	   | (vlSymsp->TOP__v.in_dcache_in_valid[2U] 
	      << 2U));
    vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_valid 
	= ((7U & (IData)(vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_valid)) 
	   | (vlSymsp->TOP__v.in_dcache_in_valid[3U] 
	      << 3U));
}

VL_INLINE_OPT void Vcache_simX_cache_simX::_sequent__TOP__v__2(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_PRINTF("      Vcache_simX_cache_simX::_sequent__TOP__v__2\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // ALWAYS at cache_simX.v:93
    if (vlTOPp->reset) {
	vlSymsp->TOP__v.__PVT__icache_i_m_ready = 0U;
	vlSymsp->TOP__v.__PVT__dcache_i_m_ready = 0U;
    } else {
	vlSymsp->TOP__v.__PVT__icache_i_m_ready = (1U 
						   == (IData)(vlSymsp->TOP__v__dmem_controller.__PVT__icache__DOT__state));
	vlSymsp->TOP__v.__PVT__dcache_i_m_ready = (1U 
						   == (IData)(vlSymsp->TOP__v__dmem_controller.__PVT__dcache__DOT__state));
    }
}

VL_INLINE_OPT void Vcache_simX_cache_simX::_combo__TOP__v__3(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_PRINTF("      Vcache_simX_cache_simX::_combo__TOP__v__3\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_valid 
	= ((0xeU & (IData)(vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_valid)) 
	   | vlSymsp->TOP__v.in_dcache_in_valid[0U]);
    vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_valid 
	= ((0xdU & (IData)(vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_valid)) 
	   | (vlSymsp->TOP__v.in_dcache_in_valid[1U] 
	      << 1U));
    vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_valid 
	= ((0xbU & (IData)(vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_valid)) 
	   | (vlSymsp->TOP__v.in_dcache_in_valid[2U] 
	      << 2U));
    vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_valid 
	= ((7U & (IData)(vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_valid)) 
	   | (vlSymsp->TOP__v.in_dcache_in_valid[3U] 
	      << 3U));
    vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_address[0U] 
	= vlSymsp->TOP__v.in_dcache_in_address[0U];
    vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_address[1U] 
	= vlSymsp->TOP__v.in_dcache_in_address[1U];
    vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_address[2U] 
	= vlSymsp->TOP__v.in_dcache_in_address[2U];
    vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_address[3U] 
	= vlSymsp->TOP__v.in_dcache_in_address[3U];
}

void Vcache_simX_cache_simX::_settle__TOP__v__4(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_PRINTF("      Vcache_simX_cache_simX::_settle__TOP__v__4\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_address[0U] 
	= vlSymsp->TOP__v.in_dcache_in_address[0U];
    vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_address[1U] 
	= vlSymsp->TOP__v.in_dcache_in_address[1U];
    vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_address[2U] 
	= vlSymsp->TOP__v.in_dcache_in_address[2U];
    vlSymsp->TOP__v__VX_dcache_req.__PVT__out_cache_driver_in_address[3U] 
	= vlSymsp->TOP__v.in_dcache_in_address[3U];
}
