// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "Vrf2_32x128_wm1__Syms.h"
#include "Vrf2_32x128_wm1.h"

// FUNCTIONS
Vrf2_32x128_wm1__Syms::Vrf2_32x128_wm1__Syms(Vrf2_32x128_wm1* topp, const char* namep)
	// Setup locals
	: __Vm_namep(namep)
	, __Vm_didInit(false)
	// Setup submodule names
{
    // Pointer to top level
    TOPp = topp;
    // Setup each module's pointers to their submodules
    // Setup each module's pointer back to symbol table (for public functions)
    TOPp->__Vconfigure(this, true);
}
