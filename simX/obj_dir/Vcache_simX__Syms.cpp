// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "Vcache_simX__Syms.h"
#include "Vcache_simX.h"
#include "Vcache_simX_VX_dram_req_rsp_inter__N4_NB4.h"
#include "Vcache_simX_VX_dram_req_rsp_inter__N1_NB4.h"
#include "Vcache_simX_VX_dcache_request_inter.h"
#include "Vcache_simX_VX_Cache_Bank__pi7.h"



// FUNCTIONS
Vcache_simX__Syms::Vcache_simX__Syms(Vcache_simX* topp, const char* namep)
    // Setup locals
    : __Vm_namep(namep)
    , __Vm_activity(false)
    , __Vm_didInit(false)
    // Setup submodule names
    , TOP__cache_simX__DOT__VX_dcache_req(Verilated::catName(topp->name(),"cache_simX.VX_dcache_req"))
    , TOP__cache_simX__DOT__VX_dram_req_rsp(Verilated::catName(topp->name(),"cache_simX.VX_dram_req_rsp"))
    , TOP__cache_simX__DOT__VX_dram_req_rsp_icache(Verilated::catName(topp->name(),"cache_simX.VX_dram_req_rsp_icache"))
    , TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure(Verilated::catName(topp->name(),"cache_simX.dmem_controller.dcache.genblk3[0].bank_structure"))
    , TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure(Verilated::catName(topp->name(),"cache_simX.dmem_controller.dcache.genblk3[1].bank_structure"))
    , TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure(Verilated::catName(topp->name(),"cache_simX.dmem_controller.dcache.genblk3[2].bank_structure"))
    , TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure(Verilated::catName(topp->name(),"cache_simX.dmem_controller.dcache.genblk3[3].bank_structure"))
{
    // Pointer to top level
    TOPp = topp;
    // Setup each module's pointers to their submodules
    TOPp->__PVT__cache_simX__DOT__VX_dcache_req  = &TOP__cache_simX__DOT__VX_dcache_req;
    TOPp->__PVT__cache_simX__DOT__VX_dram_req_rsp  = &TOP__cache_simX__DOT__VX_dram_req_rsp;
    TOPp->__PVT__cache_simX__DOT__VX_dram_req_rsp_icache  = &TOP__cache_simX__DOT__VX_dram_req_rsp_icache;
    TOPp->__PVT__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure  = &TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure;
    TOPp->__PVT__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure  = &TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure;
    TOPp->__PVT__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure  = &TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure;
    TOPp->__PVT__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure  = &TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure;
    // Setup each module's pointer back to symbol table (for public functions)
    TOPp->__Vconfigure(this, true);
    TOP__cache_simX__DOT__VX_dcache_req.__Vconfigure(this, true);
    TOP__cache_simX__DOT__VX_dram_req_rsp.__Vconfigure(this, true);
    TOP__cache_simX__DOT__VX_dram_req_rsp_icache.__Vconfigure(this, true);
    TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure.__Vconfigure(this, true);
    TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure.__Vconfigure(this, false);
    TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure.__Vconfigure(this, false);
    TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure.__Vconfigure(this, false);
}
