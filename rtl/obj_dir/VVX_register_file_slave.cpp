// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVX_register_file_slave.h for the primary calling header

#include "VVX_register_file_slave.h"
#include "VVX_register_file_slave__Syms.h"


//--------------------
// STATIC VARIABLES


//--------------------

VL_CTOR_IMP(VVX_register_file_slave) {
    VVX_register_file_slave__Syms* __restrict vlSymsp = __VlSymsp = new VVX_register_file_slave__Syms(this, name());
    VVX_register_file_slave* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Reset internal values
    
    // Reset structure values
    _ctor_var_reset();
}

void VVX_register_file_slave::__Vconfigure(VVX_register_file_slave__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

VVX_register_file_slave::~VVX_register_file_slave() {
    delete __VlSymsp; __VlSymsp=NULL;
}

//--------------------


void VVX_register_file_slave::eval() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate VVX_register_file_slave::eval\n"); );
    VVX_register_file_slave__Syms* __restrict vlSymsp = this->__VlSymsp;  // Setup global symbol table
    VVX_register_file_slave* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
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

void VVX_register_file_slave::_eval_initial_loop(VVX_register_file_slave__Syms* __restrict vlSymsp) {
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

VL_INLINE_OPT void VVX_register_file_slave::_sequent__TOP__1(VVX_register_file_slave__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file_slave::_sequent__TOP__1\n"); );
    VVX_register_file_slave* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // ALWAYS at VX_register_file_slave.v:51
    vlTOPp->out_src1_data = vlTOPp->VX_register_file_slave__DOT__registers
	[vlTOPp->in_src1];
    // ALWAYS at VX_register_file_slave.v:51
    vlTOPp->out_src2_data = vlTOPp->VX_register_file_slave__DOT__registers
	[vlTOPp->in_src2];
}

VL_INLINE_OPT void VVX_register_file_slave::_sequent__TOP__2(VVX_register_file_slave__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file_slave::_sequent__TOP__2\n"); );
    VVX_register_file_slave* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    // Begin mtask footprint  all: 
    VL_SIG8(__Vdlyvdim0__VX_register_file_slave__DOT__registers__v0,4,0);
    VL_SIG8(__Vdlyvset__VX_register_file_slave__DOT__registers__v0,0,0);
    VL_SIG8(__Vdlyvset__VX_register_file_slave__DOT__registers__v1,0,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v0,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v1,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v2,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v3,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v4,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v5,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v6,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v7,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v8,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v9,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v10,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v11,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v12,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v13,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v14,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v15,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v16,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v17,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v18,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v19,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v20,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v21,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v22,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v23,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v24,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v25,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v26,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v27,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v28,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v29,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v30,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v31,31,0);
    VL_SIG(__Vdlyvval__VX_register_file_slave__DOT__registers__v32,31,0);
    // Body
    __Vdlyvset__VX_register_file_slave__DOT__registers__v0 = 0U;
    __Vdlyvset__VX_register_file_slave__DOT__registers__v1 = 0U;
    // ALWAYS at VX_register_file_slave.v:42
    if (VL_UNLIKELY(((((IData)(vlTOPp->in_write_register) 
		       & (0U != (IData)(vlTOPp->in_rd))) 
		      & (IData)(vlTOPp->in_valid)) 
		     & (~ (IData)(vlTOPp->in_clone))))) {
	VL_WRITEF("RF: Writing %x to %2#\n",32,vlTOPp->in_data,
		  5,(IData)(vlTOPp->in_rd));
	__Vdlyvval__VX_register_file_slave__DOT__registers__v0 
	    = vlTOPp->in_data;
	__Vdlyvset__VX_register_file_slave__DOT__registers__v0 = 1U;
	__Vdlyvdim0__VX_register_file_slave__DOT__registers__v0 
	    = vlTOPp->in_rd;
    } else {
	if (vlTOPp->in_clone) {
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v1 
		= vlTOPp->in_regs[0x1fU];
	    __Vdlyvset__VX_register_file_slave__DOT__registers__v1 = 1U;
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v2 
		= vlTOPp->in_regs[0x1eU];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v3 
		= vlTOPp->in_regs[0x1dU];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v4 
		= vlTOPp->in_regs[0x1cU];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v5 
		= vlTOPp->in_regs[0x1bU];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v6 
		= vlTOPp->in_regs[0x1aU];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v7 
		= vlTOPp->in_regs[0x19U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v8 
		= vlTOPp->in_regs[0x18U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v9 
		= vlTOPp->in_regs[0x17U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v10 
		= vlTOPp->in_regs[0x16U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v11 
		= vlTOPp->in_regs[0x15U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v12 
		= vlTOPp->in_regs[0x14U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v13 
		= vlTOPp->in_regs[0x13U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v14 
		= vlTOPp->in_regs[0x12U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v15 
		= vlTOPp->in_regs[0x11U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v16 
		= vlTOPp->in_regs[0x10U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v17 
		= vlTOPp->in_regs[0xfU];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v18 
		= vlTOPp->in_regs[0xeU];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v19 
		= vlTOPp->in_regs[0xdU];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v20 
		= vlTOPp->in_regs[0xcU];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v21 
		= vlTOPp->in_regs[0xbU];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v22 
		= vlTOPp->in_regs[0xaU];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v23 
		= vlTOPp->in_regs[9U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v24 
		= vlTOPp->in_regs[8U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v25 
		= vlTOPp->in_regs[7U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v26 
		= vlTOPp->in_regs[6U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v27 
		= vlTOPp->in_regs[5U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v28 
		= vlTOPp->in_regs[4U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v29 
		= vlTOPp->in_regs[3U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v30 
		= vlTOPp->in_regs[2U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v31 
		= vlTOPp->in_regs[1U];
	    __Vdlyvval__VX_register_file_slave__DOT__registers__v32 
		= vlTOPp->in_regs[0U];
	}
    }
    // ALWAYSPOST at VX_register_file_slave.v:45
    if (__Vdlyvset__VX_register_file_slave__DOT__registers__v0) {
	vlTOPp->VX_register_file_slave__DOT__registers[__Vdlyvdim0__VX_register_file_slave__DOT__registers__v0] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v0;
    }
    if (__Vdlyvset__VX_register_file_slave__DOT__registers__v1) {
	vlTOPp->VX_register_file_slave__DOT__registers[0x1fU] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v1;
	vlTOPp->VX_register_file_slave__DOT__registers[0x1eU] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v2;
	vlTOPp->VX_register_file_slave__DOT__registers[0x1dU] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v3;
	vlTOPp->VX_register_file_slave__DOT__registers[0x1cU] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v4;
	vlTOPp->VX_register_file_slave__DOT__registers[0x1bU] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v5;
	vlTOPp->VX_register_file_slave__DOT__registers[0x1aU] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v6;
	vlTOPp->VX_register_file_slave__DOT__registers[0x19U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v7;
	vlTOPp->VX_register_file_slave__DOT__registers[0x18U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v8;
	vlTOPp->VX_register_file_slave__DOT__registers[0x17U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v9;
	vlTOPp->VX_register_file_slave__DOT__registers[0x16U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v10;
	vlTOPp->VX_register_file_slave__DOT__registers[0x15U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v11;
	vlTOPp->VX_register_file_slave__DOT__registers[0x14U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v12;
	vlTOPp->VX_register_file_slave__DOT__registers[0x13U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v13;
	vlTOPp->VX_register_file_slave__DOT__registers[0x12U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v14;
	vlTOPp->VX_register_file_slave__DOT__registers[0x11U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v15;
	vlTOPp->VX_register_file_slave__DOT__registers[0x10U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v16;
	vlTOPp->VX_register_file_slave__DOT__registers[0xfU] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v17;
	vlTOPp->VX_register_file_slave__DOT__registers[0xeU] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v18;
	vlTOPp->VX_register_file_slave__DOT__registers[0xdU] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v19;
	vlTOPp->VX_register_file_slave__DOT__registers[0xcU] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v20;
	vlTOPp->VX_register_file_slave__DOT__registers[0xbU] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v21;
	vlTOPp->VX_register_file_slave__DOT__registers[0xaU] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v22;
	vlTOPp->VX_register_file_slave__DOT__registers[9U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v23;
	vlTOPp->VX_register_file_slave__DOT__registers[8U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v24;
	vlTOPp->VX_register_file_slave__DOT__registers[7U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v25;
	vlTOPp->VX_register_file_slave__DOT__registers[6U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v26;
	vlTOPp->VX_register_file_slave__DOT__registers[5U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v27;
	vlTOPp->VX_register_file_slave__DOT__registers[4U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v28;
	vlTOPp->VX_register_file_slave__DOT__registers[3U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v29;
	vlTOPp->VX_register_file_slave__DOT__registers[2U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v30;
	vlTOPp->VX_register_file_slave__DOT__registers[1U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v31;
	vlTOPp->VX_register_file_slave__DOT__registers[0U] 
	    = __Vdlyvval__VX_register_file_slave__DOT__registers__v32;
    }
}

void VVX_register_file_slave::_eval(VVX_register_file_slave__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file_slave::_eval\n"); );
    VVX_register_file_slave* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    if (((~ (IData)(vlTOPp->clk)) & (IData)(vlTOPp->__Vclklast__TOP__clk))) {
	vlTOPp->_sequent__TOP__1(vlSymsp);
    }
    if (((IData)(vlTOPp->clk) & (~ (IData)(vlTOPp->__Vclklast__TOP__clk)))) {
	vlTOPp->_sequent__TOP__2(vlSymsp);
    }
    // Final
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
}

void VVX_register_file_slave::_eval_initial(VVX_register_file_slave__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file_slave::_eval_initial\n"); );
    VVX_register_file_slave* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
}

void VVX_register_file_slave::final() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file_slave::final\n"); );
    // Variables
    VVX_register_file_slave__Syms* __restrict vlSymsp = this->__VlSymsp;
    VVX_register_file_slave* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void VVX_register_file_slave::_eval_settle(VVX_register_file_slave__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file_slave::_eval_settle\n"); );
    VVX_register_file_slave* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

VL_INLINE_OPT QData VVX_register_file_slave::_change_request(VVX_register_file_slave__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file_slave::_change_request\n"); );
    VVX_register_file_slave* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // Change detection
    QData __req = false;  // Logically a bool
    return __req;
}

#ifdef VL_DEBUG
void VVX_register_file_slave::_eval_debug_assertions() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file_slave::_eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((clk & 0xfeU))) {
	Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((in_valid & 0xfeU))) {
	Verilated::overWidthError("in_valid");}
    if (VL_UNLIKELY((in_write_register & 0xfeU))) {
	Verilated::overWidthError("in_write_register");}
    if (VL_UNLIKELY((in_rd & 0xe0U))) {
	Verilated::overWidthError("in_rd");}
    if (VL_UNLIKELY((in_src1 & 0xe0U))) {
	Verilated::overWidthError("in_src1");}
    if (VL_UNLIKELY((in_src2 & 0xe0U))) {
	Verilated::overWidthError("in_src2");}
    if (VL_UNLIKELY((in_clone & 0xfeU))) {
	Verilated::overWidthError("in_clone");}
}
#endif // VL_DEBUG

void VVX_register_file_slave::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file_slave::_ctor_var_reset\n"); );
    // Body
    clk = VL_RAND_RESET_I(1);
    in_valid = VL_RAND_RESET_I(1);
    in_write_register = VL_RAND_RESET_I(1);
    in_rd = VL_RAND_RESET_I(5);
    in_data = VL_RAND_RESET_I(32);
    in_src1 = VL_RAND_RESET_I(5);
    in_src2 = VL_RAND_RESET_I(5);
    in_clone = VL_RAND_RESET_I(1);
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
	    in_regs[__Vi0] = VL_RAND_RESET_I(32);
    }}
    out_src1_data = VL_RAND_RESET_I(32);
    out_src2_data = VL_RAND_RESET_I(32);
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
	    VX_register_file_slave__DOT__registers[__Vi0] = VL_RAND_RESET_I(32);
    }}
}
