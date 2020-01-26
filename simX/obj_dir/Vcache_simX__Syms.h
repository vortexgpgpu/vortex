// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header

#ifndef _Vcache_simX__Syms_H_
#define _Vcache_simX__Syms_H_

#include "verilated.h"

// INCLUDE MODULE CLASSES
#include "Vcache_simX.h"
#include "Vcache_simX_VX_dram_req_rsp_inter__N4_NB4.h"
#include "Vcache_simX_VX_dram_req_rsp_inter__N1_NB4.h"
#include "Vcache_simX_VX_dcache_request_inter.h"
#include "Vcache_simX_VX_Cache_Bank__pi8.h"

// SYMS CLASS
class Vcache_simX__Syms : public VerilatedSyms {
  public:
    
    // LOCAL STATE
    const char* __Vm_namep;
    bool __Vm_activity;  ///< Used by trace routines to determine change occurred
    bool __Vm_didInit;
    
    // SUBCELL STATE
    Vcache_simX*                   TOPp;
    Vcache_simX_VX_dcache_request_inter TOP__cache_simX__DOT__VX_dcache_req;
    Vcache_simX_VX_dram_req_rsp_inter__N4_NB4 TOP__cache_simX__DOT__VX_dram_req_rsp;
    Vcache_simX_VX_dram_req_rsp_inter__N1_NB4 TOP__cache_simX__DOT__VX_dram_req_rsp_icache;
    Vcache_simX_VX_Cache_Bank__pi8 TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure;
    Vcache_simX_VX_Cache_Bank__pi8 TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure;
    Vcache_simX_VX_Cache_Bank__pi8 TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure;
    Vcache_simX_VX_Cache_Bank__pi8 TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure;
    
    // CREATORS
    Vcache_simX__Syms(Vcache_simX* topp, const char* namep);
    ~Vcache_simX__Syms() {}
    
    // METHODS
    inline const char* name() { return __Vm_namep; }
    inline bool getClearActivity() { bool r=__Vm_activity; __Vm_activity=false; return r; }
    
} VL_ATTR_ALIGNED(64);

#endif // guard
