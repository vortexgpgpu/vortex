// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVX_shared_memory.h for the primary calling header

#include "VVX_shared_memory.h"
#include "VVX_shared_memory__Syms.h"


//--------------------
// STATIC VARIABLES


//--------------------

VL_CTOR_IMP(VVX_shared_memory) {
    VVX_shared_memory__Syms* __restrict vlSymsp = __VlSymsp = new VVX_shared_memory__Syms(this, name());
    VVX_shared_memory* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Reset internal values
    
    // Reset structure values
    _ctor_var_reset();
}

void VVX_shared_memory::__Vconfigure(VVX_shared_memory__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

VVX_shared_memory::~VVX_shared_memory() {
    delete __VlSymsp; __VlSymsp=NULL;
}

//--------------------


void VVX_shared_memory::eval() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate VVX_shared_memory::eval\n"); );
    VVX_shared_memory__Syms* __restrict vlSymsp = this->__VlSymsp;  // Setup global symbol table
    VVX_shared_memory* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
#ifdef VL_DEBUG
    // Debug assertions
    _eval_debug_assertions();
#endif // VL_DEBUG
    // Initialize
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) _eval_initial_loop(vlSymsp);
    // Evaluate till stable
    int __VclockLoop = 0;
    QData __Vchange = 1;
    do {
	VL_DEBUG_IF(VL_DBG_MSGF("+ Clock loop\n"););
	_eval(vlSymsp);
	if (VL_UNLIKELY(++__VclockLoop > 100)) {
	    // About to fail, so enable debug to see what's not settling.
	    // Note you must run make with OPT=-DVL_DEBUG for debug prints.
	    int __Vsaved_debug = Verilated::debug();
	    Verilated::debug(1);
	    __Vchange = _change_request(vlSymsp);
	    Verilated::debug(__Vsaved_debug);
	    VL_FATAL_MT(__FILE__,__LINE__,__FILE__,"Verilated model didn't converge");
	} else {
	    __Vchange = _change_request(vlSymsp);
	}
    } while (VL_UNLIKELY(__Vchange));
}

void VVX_shared_memory::_eval_initial_loop(VVX_shared_memory__Syms* __restrict vlSymsp) {
    vlSymsp->__Vm_didInit = true;
    _eval_initial(vlSymsp);
    // Evaluate till stable
    int __VclockLoop = 0;
    QData __Vchange = 1;
    do {
	_eval_settle(vlSymsp);
	_eval(vlSymsp);
	if (VL_UNLIKELY(++__VclockLoop > 100)) {
	    // About to fail, so enable debug to see what's not settling.
	    // Note you must run make with OPT=-DVL_DEBUG for debug prints.
	    int __Vsaved_debug = Verilated::debug();
	    Verilated::debug(1);
	    __Vchange = _change_request(vlSymsp);
	    Verilated::debug(__Vsaved_debug);
	    VL_FATAL_MT(__FILE__,__LINE__,__FILE__,"Verilated model didn't DC converge");
	} else {
	    __Vchange = _change_request(vlSymsp);
	}
    } while (VL_UNLIKELY(__Vchange));
}

//--------------------
// Internal Methods

VL_INLINE_OPT void VVX_shared_memory::_sequent__TOP__1(VVX_shared_memory__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_shared_memory::_sequent__TOP__1\n"); );
    VVX_shared_memory* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    // Begin mtask footprint  all: 
    VL_SIG8(__Vdlyvdim0__VX_shared_memory__DOT__mem__v0,7,0);
    VL_SIG8(__Vdlyvset__VX_shared_memory__DOT__mem__v0,0,0);
    VL_SIG(__Vdlyvval__VX_shared_memory__DOT__mem__v0[1],31,0);
    // Body
    // ALWAYS at VX_shared_memory.v:27
    vlTOPp->out_data[0U] = vlTOPp->VX_shared_memory__DOT__mem
	[(0xffU & (vlTOPp->in_address[0U] >> 2U))];
    __Vdlyvset__VX_shared_memory__DOT__mem__v0 = 0U;
    // ALWAYS at VX_shared_memory.v:27
    if (((2U == (IData)(vlTOPp->in_mem_write)) & (vlTOPp->in_valid 
						  & (0xffffU 
						     == 
						     (0xffffU 
						      & (vlTOPp->in_address
							 [0U] 
							 >> 0x10U)))))) {
	__Vdlyvval__VX_shared_memory__DOT__mem__v0 
	    = vlTOPp->in_data;
	__Vdlyvset__VX_shared_memory__DOT__mem__v0 = 1U;
	__Vdlyvdim0__VX_shared_memory__DOT__mem__v0 
	    = (0xffU & (vlTOPp->in_address[0U] >> 2U));
    }
    // ALWAYSPOST at VX_shared_memory.v:31
    if (__Vdlyvset__VX_shared_memory__DOT__mem__v0) {
	vlTOPp->VX_shared_memory__DOT__mem[__Vdlyvdim0__VX_shared_memory__DOT__mem__v0] 
	    = __Vdlyvval__VX_shared_memory__DOT__mem__v0;
    }
}

void VVX_shared_memory::_eval(VVX_shared_memory__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_shared_memory::_eval\n"); );
    VVX_shared_memory* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    if (((IData)(vlTOPp->clk) & (~ (IData)(vlTOPp->__Vclklast__TOP__clk)))) {
	vlTOPp->_sequent__TOP__1(vlSymsp);
    }
    // Final
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
}

void VVX_shared_memory::_eval_initial(VVX_shared_memory__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_shared_memory::_eval_initial\n"); );
    VVX_shared_memory* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
}

void VVX_shared_memory::final() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_shared_memory::final\n"); );
    // Variables
    VVX_shared_memory__Syms* __restrict vlSymsp = this->__VlSymsp;
    VVX_shared_memory* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void VVX_shared_memory::_eval_settle(VVX_shared_memory__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_shared_memory::_eval_settle\n"); );
    VVX_shared_memory* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

VL_INLINE_OPT QData VVX_shared_memory::_change_request(VVX_shared_memory__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_shared_memory::_change_request\n"); );
    VVX_shared_memory* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // Change detection
    QData __req = false;  // Logically a bool
    return __req;
}

#ifdef VL_DEBUG
void VVX_shared_memory::_eval_debug_assertions() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_shared_memory::_eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((clk & 0xfeU))) {
	Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((in_mem_read & 0xf8U))) {
	Verilated::overWidthError("in_mem_read");}
    if (VL_UNLIKELY((in_mem_write & 0xf8U))) {
	Verilated::overWidthError("in_mem_write");}
}
#endif // VL_DEBUG

void VVX_shared_memory::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_shared_memory::_ctor_var_reset\n"); );
    // Body
    clk = VL_RAND_RESET_I(1);
    { int __Vi0=0; for (; __Vi0<1; ++__Vi0) {
	    in_address[__Vi0] = VL_RAND_RESET_I(32);
    }}
    in_mem_read = VL_RAND_RESET_I(3);
    in_mem_write = VL_RAND_RESET_I(3);
    { int __Vi0=0; for (; __Vi0<1; ++__Vi0) {
	    in_valid[__Vi0] = VL_RAND_RESET_I(1);
    }}
    { int __Vi0=0; for (; __Vi0<1; ++__Vi0) {
	    in_data[__Vi0] = VL_RAND_RESET_I(32);
    }}
    { int __Vi0=0; for (; __Vi0<1; ++__Vi0) {
	    out_data[__Vi0] = VL_RAND_RESET_I(32);
    }}
    { int __Vi0=0; for (; __Vi0<256; ++__Vi0) {
	    VX_shared_memory__DOT__mem[__Vi0] = VL_RAND_RESET_I(32);
    }}
}
