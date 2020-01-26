// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vcache_simX.h for the primary calling header

#include "Vcache_simX_VX_dram_req_rsp_inter__N1_NB4.h" // For This
#include "Vcache_simX__Syms.h"


//--------------------
// STATIC VARIABLES


//--------------------

VL_CTOR_IMP(Vcache_simX_VX_dram_req_rsp_inter__N1_NB4) {
    // Reset internal values
    // Reset structure values
    _ctor_var_reset();
}

void Vcache_simX_VX_dram_req_rsp_inter__N1_NB4::__Vconfigure(Vcache_simX__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

Vcache_simX_VX_dram_req_rsp_inter__N1_NB4::~Vcache_simX_VX_dram_req_rsp_inter__N1_NB4() {
}

//--------------------
// Internal Methods

void Vcache_simX_VX_dram_req_rsp_inter__N1_NB4::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+          Vcache_simX_VX_dram_req_rsp_inter__N1_NB4::_ctor_var_reset\n"); );
    // Body
    VL_RAND_RESET_W(128,i_m_readdata);
}
