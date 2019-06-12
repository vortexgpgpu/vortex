// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "VVortex__Syms.h"
#include "VVortex.h"
#include "VVortex_VX_context_slave.h"

// FUNCTIONS
VVortex__Syms::VVortex__Syms(VVortex* topp, const char* namep)
	// Setup locals
	: __Vm_namep(namep)
	, __Vm_didInit(false)
	// Setup submodule names
	, TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__1__KET____DOT__VX_Context_one (Verilated::catName(topp->name(),"Vortex.vx_decode.genblk1[1].VX_Context_one"))
	, TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__2__KET____DOT__VX_Context_one (Verilated::catName(topp->name(),"Vortex.vx_decode.genblk1[2].VX_Context_one"))
	, TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__3__KET____DOT__VX_Context_one (Verilated::catName(topp->name(),"Vortex.vx_decode.genblk1[3].VX_Context_one"))
	, TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__4__KET____DOT__VX_Context_one (Verilated::catName(topp->name(),"Vortex.vx_decode.genblk1[4].VX_Context_one"))
	, TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__5__KET____DOT__VX_Context_one (Verilated::catName(topp->name(),"Vortex.vx_decode.genblk1[5].VX_Context_one"))
	, TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__6__KET____DOT__VX_Context_one (Verilated::catName(topp->name(),"Vortex.vx_decode.genblk1[6].VX_Context_one"))
	, TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__7__KET____DOT__VX_Context_one (Verilated::catName(topp->name(),"Vortex.vx_decode.genblk1[7].VX_Context_one"))
{
    // Pointer to top level
    TOPp = topp;
    // Setup each module's pointers to their submodules
    TOPp->__PVT__Vortex__DOT__vx_decode__DOT__genblk1__BRA__1__KET____DOT__VX_Context_one  = &TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__1__KET____DOT__VX_Context_one;
    TOPp->__PVT__Vortex__DOT__vx_decode__DOT__genblk1__BRA__2__KET____DOT__VX_Context_one  = &TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__2__KET____DOT__VX_Context_one;
    TOPp->__PVT__Vortex__DOT__vx_decode__DOT__genblk1__BRA__3__KET____DOT__VX_Context_one  = &TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__3__KET____DOT__VX_Context_one;
    TOPp->__PVT__Vortex__DOT__vx_decode__DOT__genblk1__BRA__4__KET____DOT__VX_Context_one  = &TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__4__KET____DOT__VX_Context_one;
    TOPp->__PVT__Vortex__DOT__vx_decode__DOT__genblk1__BRA__5__KET____DOT__VX_Context_one  = &TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__5__KET____DOT__VX_Context_one;
    TOPp->__PVT__Vortex__DOT__vx_decode__DOT__genblk1__BRA__6__KET____DOT__VX_Context_one  = &TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__6__KET____DOT__VX_Context_one;
    TOPp->__PVT__Vortex__DOT__vx_decode__DOT__genblk1__BRA__7__KET____DOT__VX_Context_one  = &TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__7__KET____DOT__VX_Context_one;
    // Setup each module's pointer back to symbol table (for public functions)
    TOPp->__Vconfigure(this, true);
    TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__1__KET____DOT__VX_Context_one.__Vconfigure(this, true);
    TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__2__KET____DOT__VX_Context_one.__Vconfigure(this, false);
    TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__3__KET____DOT__VX_Context_one.__Vconfigure(this, false);
    TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__4__KET____DOT__VX_Context_one.__Vconfigure(this, false);
    TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__5__KET____DOT__VX_Context_one.__Vconfigure(this, false);
    TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__6__KET____DOT__VX_Context_one.__Vconfigure(this, false);
    TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__7__KET____DOT__VX_Context_one.__Vconfigure(this, false);
}
