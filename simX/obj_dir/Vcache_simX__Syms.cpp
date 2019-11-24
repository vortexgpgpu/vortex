// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "Vcache_simX__Syms.h"
#include "Vcache_simX.h"
#include "Vcache_simX_cache_simX.h"
#include "Vcache_simX_VX_dmem_controller__V0_VB1000.h"
#include "Vcache_simX_VX_icache_request_inter.h"
#include "Vcache_simX_VX_icache_response_inter.h"
#include "Vcache_simX_VX_dram_req_rsp_inter__N4_NB4.h"
#include "Vcache_simX_VX_dram_req_rsp_inter__N1_NB4.h"
#include "Vcache_simX_VX_dcache_request_inter.h"
#include "Vcache_simX_VX_dcache_response_inter.h"

// FUNCTIONS
Vcache_simX__Syms::Vcache_simX__Syms(Vcache_simX* topp, const char* namep)
	// Setup locals
	: __Vm_namep(namep)
	, __Vm_activity(false)
	, __Vm_didInit(false)
	// Setup submodule names
	, TOP__v                         (Verilated::catName(topp->name(),"v"))
	, TOP__v__VX_dcache_req          (Verilated::catName(topp->name(),"v.VX_dcache_req"))
	, TOP__v__VX_dcache_rsp          (Verilated::catName(topp->name(),"v.VX_dcache_rsp"))
	, TOP__v__VX_dram_req_rsp        (Verilated::catName(topp->name(),"v.VX_dram_req_rsp"))
	, TOP__v__VX_dram_req_rsp_icache (Verilated::catName(topp->name(),"v.VX_dram_req_rsp_icache"))
	, TOP__v__VX_icache_req          (Verilated::catName(topp->name(),"v.VX_icache_req"))
	, TOP__v__VX_icache_rsp          (Verilated::catName(topp->name(),"v.VX_icache_rsp"))
	, TOP__v__dmem_controller        (Verilated::catName(topp->name(),"v.dmem_controller"))
{
    // Pointer to top level
    TOPp = topp;
    // Setup each module's pointers to their submodules
    TOPp->__PVT__v                  = &TOP__v;
    TOPp->__PVT__v->__PVT__VX_dcache_req  = &TOP__v__VX_dcache_req;
    TOPp->__PVT__v->__PVT__VX_dcache_rsp  = &TOP__v__VX_dcache_rsp;
    TOPp->__PVT__v->__PVT__VX_dram_req_rsp  = &TOP__v__VX_dram_req_rsp;
    TOPp->__PVT__v->__PVT__VX_dram_req_rsp_icache  = &TOP__v__VX_dram_req_rsp_icache;
    TOPp->__PVT__v->__PVT__VX_icache_req  = &TOP__v__VX_icache_req;
    TOPp->__PVT__v->__PVT__VX_icache_rsp  = &TOP__v__VX_icache_rsp;
    TOPp->__PVT__v->__PVT__dmem_controller  = &TOP__v__dmem_controller;
    // Setup each module's pointer back to symbol table (for public functions)
    TOPp->__Vconfigure(this, true);
    TOP__v.__Vconfigure(this, true);
    TOP__v__VX_dcache_req.__Vconfigure(this, true);
    TOP__v__VX_dcache_rsp.__Vconfigure(this, true);
    TOP__v__VX_dram_req_rsp.__Vconfigure(this, true);
    TOP__v__VX_dram_req_rsp_icache.__Vconfigure(this, true);
    TOP__v__VX_icache_req.__Vconfigure(this, true);
    TOP__v__VX_icache_rsp.__Vconfigure(this, true);
    TOP__v__dmem_controller.__Vconfigure(this, true);
    // Setup scope names
}
