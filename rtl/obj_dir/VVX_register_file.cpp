// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVX_register_file.h for the primary calling header

#include "VVX_register_file.h"
#include "VVX_register_file__Syms.h"


//--------------------
// STATIC VARIABLES


//--------------------

VL_CTOR_IMP(VVX_register_file) {
    VVX_register_file__Syms* __restrict vlSymsp = __VlSymsp = new VVX_register_file__Syms(this, name());
    VVX_register_file* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Reset internal values
    
    // Reset structure values
    _ctor_var_reset();
}

void VVX_register_file::__Vconfigure(VVX_register_file__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

VVX_register_file::~VVX_register_file() {
    delete __VlSymsp; __VlSymsp=NULL;
}

//--------------------


void VVX_register_file::eval() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate VVX_register_file::eval\n"); );
    VVX_register_file__Syms* __restrict vlSymsp = this->__VlSymsp;  // Setup global symbol table
    VVX_register_file* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
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

void VVX_register_file::_eval_initial_loop(VVX_register_file__Syms* __restrict vlSymsp) {
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

VL_INLINE_OPT void VVX_register_file::_sequent__TOP__1(VVX_register_file__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file::_sequent__TOP__1\n"); );
    VVX_register_file* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    // Begin mtask footprint  all: 
    VL_SIG8(__Vdlyvdim0__VX_register_file__DOT__registers__v0,4,0);
    VL_SIG8(__Vdlyvset__VX_register_file__DOT__registers__v0,0,0);
    VL_SIG(__Vdlyvval__VX_register_file__DOT__registers__v0,31,0);
    // Body
    __Vdlyvset__VX_register_file__DOT__registers__v0 = 0U;
    // ALWAYS at VX_register_file.v:30
    if (((IData)(vlTOPp->in_write_register) & (0U != (IData)(vlTOPp->in_rd)))) {
	__Vdlyvval__VX_register_file__DOT__registers__v0 
	    = vlTOPp->in_data;
	__Vdlyvset__VX_register_file__DOT__registers__v0 = 1U;
	__Vdlyvdim0__VX_register_file__DOT__registers__v0 
	    = vlTOPp->in_rd;
    }
    // ALWAYSPOST at VX_register_file.v:32
    if (__Vdlyvset__VX_register_file__DOT__registers__v0) {
	vlTOPp->VX_register_file__DOT__registers[__Vdlyvdim0__VX_register_file__DOT__registers__v0] 
	    = __Vdlyvval__VX_register_file__DOT__registers__v0;
    }
}

VL_INLINE_OPT void VVX_register_file::_settle__TOP__2(VVX_register_file__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file::_settle__TOP__2\n"); );
    VVX_register_file* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->out_src1_data = vlTOPp->VX_register_file__DOT__registers
	[vlTOPp->in_src1];
    vlTOPp->out_src2_data = vlTOPp->VX_register_file__DOT__registers
	[vlTOPp->in_src2];
}

void VVX_register_file::_eval(VVX_register_file__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file::_eval\n"); );
    VVX_register_file* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    if (((IData)(vlTOPp->clk) & (~ (IData)(vlTOPp->__Vclklast__TOP__clk)))) {
	vlTOPp->_sequent__TOP__1(vlSymsp);
    }
    vlTOPp->_settle__TOP__2(vlSymsp);
    // Final
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
}

void VVX_register_file::_eval_initial(VVX_register_file__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file::_eval_initial\n"); );
    VVX_register_file* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
}

void VVX_register_file::final() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file::final\n"); );
    // Variables
    VVX_register_file__Syms* __restrict vlSymsp = this->__VlSymsp;
    VVX_register_file* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void VVX_register_file::_eval_settle(VVX_register_file__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file::_eval_settle\n"); );
    VVX_register_file* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->_settle__TOP__2(vlSymsp);
}

VL_INLINE_OPT QData VVX_register_file::_change_request(VVX_register_file__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file::_change_request\n"); );
    VVX_register_file* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // Change detection
    QData __req = false;  // Logically a bool
    return __req;
}

#ifdef VL_DEBUG
void VVX_register_file::_eval_debug_assertions() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file::_eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((clk & 0xfeU))) {
	Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((in_write_register & 0xfeU))) {
	Verilated::overWidthError("in_write_register");}
    if (VL_UNLIKELY((in_rd & 0xe0U))) {
	Verilated::overWidthError("in_rd");}
    if (VL_UNLIKELY((in_src1 & 0xe0U))) {
	Verilated::overWidthError("in_src1");}
    if (VL_UNLIKELY((in_src2 & 0xe0U))) {
	Verilated::overWidthError("in_src2");}
}
#endif // VL_DEBUG

void VVX_register_file::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file::_ctor_var_reset\n"); );
    // Body
    clk = VL_RAND_RESET_I(1);
    in_write_register = VL_RAND_RESET_I(1);
    in_rd = VL_RAND_RESET_I(5);
    in_data = VL_RAND_RESET_I(32);
    in_src1 = VL_RAND_RESET_I(5);
    in_src2 = VL_RAND_RESET_I(5);
    out_src1_data = VL_RAND_RESET_I(32);
    out_src2_data = VL_RAND_RESET_I(32);
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
	    VX_register_file__DOT__registers[__Vi0] = VL_RAND_RESET_I(32);
    }}
}
