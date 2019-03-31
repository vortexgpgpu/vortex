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
    // Body
    // ALWAYS at VX_register_file.v:46
    vlTOPp->out_src1_data = vlTOPp->VX_register_file__DOT__registers
	[vlTOPp->in_src1];
    // ALWAYS at VX_register_file.v:46
    vlTOPp->out_src2_data = vlTOPp->VX_register_file__DOT__registers
	[vlTOPp->in_src2];
}

VL_INLINE_OPT void VVX_register_file::_sequent__TOP__2(VVX_register_file__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file::_sequent__TOP__2\n"); );
    VVX_register_file* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    // Begin mtask footprint  all: 
    VL_SIG8(__Vdlyvdim0__VX_register_file__DOT__registers__v0,4,0);
    VL_SIG8(__Vdlyvset__VX_register_file__DOT__registers__v0,0,0);
    VL_SIG(__Vdlyvval__VX_register_file__DOT__registers__v0,31,0);
    // Body
    __Vdlyvset__VX_register_file__DOT__registers__v0 = 0U;
    // ALWAYS at VX_register_file.v:39
    if (VL_UNLIKELY((((IData)(vlTOPp->in_write_register) 
		      & (0U != (IData)(vlTOPp->in_rd))) 
		     & (IData)(vlTOPp->in_valid)))) {
	VL_WRITEF("RF: Writing %x to %2#\n",32,vlTOPp->in_data,
		  5,(IData)(vlTOPp->in_rd));
	__Vdlyvval__VX_register_file__DOT__registers__v0 
	    = vlTOPp->in_data;
	__Vdlyvset__VX_register_file__DOT__registers__v0 = 1U;
	__Vdlyvdim0__VX_register_file__DOT__registers__v0 
	    = vlTOPp->in_rd;
    }
    // ALWAYSPOST at VX_register_file.v:42
    if (__Vdlyvset__VX_register_file__DOT__registers__v0) {
	vlTOPp->VX_register_file__DOT__registers[__Vdlyvdim0__VX_register_file__DOT__registers__v0] 
	    = __Vdlyvval__VX_register_file__DOT__registers__v0;
    }
    vlTOPp->out_regs[0x1fU] = vlTOPp->VX_register_file__DOT__registers
	[0x1fU];
    vlTOPp->out_regs[0x1eU] = vlTOPp->VX_register_file__DOT__registers
	[0x1eU];
    vlTOPp->out_regs[0x1dU] = vlTOPp->VX_register_file__DOT__registers
	[0x1dU];
    vlTOPp->out_regs[0x1cU] = vlTOPp->VX_register_file__DOT__registers
	[0x1cU];
    vlTOPp->out_regs[0x1bU] = vlTOPp->VX_register_file__DOT__registers
	[0x1bU];
    vlTOPp->out_regs[0x1aU] = vlTOPp->VX_register_file__DOT__registers
	[0x1aU];
    vlTOPp->out_regs[0x19U] = vlTOPp->VX_register_file__DOT__registers
	[0x19U];
    vlTOPp->out_regs[0x18U] = vlTOPp->VX_register_file__DOT__registers
	[0x18U];
    vlTOPp->out_regs[0x17U] = vlTOPp->VX_register_file__DOT__registers
	[0x17U];
    vlTOPp->out_regs[0x16U] = vlTOPp->VX_register_file__DOT__registers
	[0x16U];
    vlTOPp->out_regs[0x15U] = vlTOPp->VX_register_file__DOT__registers
	[0x15U];
    vlTOPp->out_regs[0x14U] = vlTOPp->VX_register_file__DOT__registers
	[0x14U];
    vlTOPp->out_regs[0x13U] = vlTOPp->VX_register_file__DOT__registers
	[0x13U];
    vlTOPp->out_regs[0x12U] = vlTOPp->VX_register_file__DOT__registers
	[0x12U];
    vlTOPp->out_regs[0x11U] = vlTOPp->VX_register_file__DOT__registers
	[0x11U];
    vlTOPp->out_regs[0x10U] = vlTOPp->VX_register_file__DOT__registers
	[0x10U];
    vlTOPp->out_regs[0xfU] = vlTOPp->VX_register_file__DOT__registers
	[0xfU];
    vlTOPp->out_regs[0xeU] = vlTOPp->VX_register_file__DOT__registers
	[0xeU];
    vlTOPp->out_regs[0xdU] = vlTOPp->VX_register_file__DOT__registers
	[0xdU];
    vlTOPp->out_regs[0xcU] = vlTOPp->VX_register_file__DOT__registers
	[0xcU];
    vlTOPp->out_regs[0xbU] = vlTOPp->VX_register_file__DOT__registers
	[0xbU];
    vlTOPp->out_regs[0xaU] = vlTOPp->VX_register_file__DOT__registers
	[0xaU];
    vlTOPp->out_regs[9U] = vlTOPp->VX_register_file__DOT__registers
	[9U];
    vlTOPp->out_regs[8U] = vlTOPp->VX_register_file__DOT__registers
	[8U];
    vlTOPp->out_regs[7U] = vlTOPp->VX_register_file__DOT__registers
	[7U];
    vlTOPp->out_regs[6U] = vlTOPp->VX_register_file__DOT__registers
	[6U];
    vlTOPp->out_regs[5U] = vlTOPp->VX_register_file__DOT__registers
	[5U];
    vlTOPp->out_regs[4U] = vlTOPp->VX_register_file__DOT__registers
	[4U];
    vlTOPp->out_regs[3U] = vlTOPp->VX_register_file__DOT__registers
	[3U];
    vlTOPp->out_regs[2U] = vlTOPp->VX_register_file__DOT__registers
	[2U];
    vlTOPp->out_regs[1U] = vlTOPp->VX_register_file__DOT__registers
	[1U];
    vlTOPp->out_regs[0U] = vlTOPp->VX_register_file__DOT__registers
	[0U];
}

void VVX_register_file::_settle__TOP__3(VVX_register_file__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file::_settle__TOP__3\n"); );
    VVX_register_file* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->out_regs[0x1fU] = vlTOPp->VX_register_file__DOT__registers
	[0x1fU];
    vlTOPp->out_regs[0x1eU] = vlTOPp->VX_register_file__DOT__registers
	[0x1eU];
    vlTOPp->out_regs[0x1dU] = vlTOPp->VX_register_file__DOT__registers
	[0x1dU];
    vlTOPp->out_regs[0x1cU] = vlTOPp->VX_register_file__DOT__registers
	[0x1cU];
    vlTOPp->out_regs[0x1bU] = vlTOPp->VX_register_file__DOT__registers
	[0x1bU];
    vlTOPp->out_regs[0x1aU] = vlTOPp->VX_register_file__DOT__registers
	[0x1aU];
    vlTOPp->out_regs[0x19U] = vlTOPp->VX_register_file__DOT__registers
	[0x19U];
    vlTOPp->out_regs[0x18U] = vlTOPp->VX_register_file__DOT__registers
	[0x18U];
    vlTOPp->out_regs[0x17U] = vlTOPp->VX_register_file__DOT__registers
	[0x17U];
    vlTOPp->out_regs[0x16U] = vlTOPp->VX_register_file__DOT__registers
	[0x16U];
    vlTOPp->out_regs[0x15U] = vlTOPp->VX_register_file__DOT__registers
	[0x15U];
    vlTOPp->out_regs[0x14U] = vlTOPp->VX_register_file__DOT__registers
	[0x14U];
    vlTOPp->out_regs[0x13U] = vlTOPp->VX_register_file__DOT__registers
	[0x13U];
    vlTOPp->out_regs[0x12U] = vlTOPp->VX_register_file__DOT__registers
	[0x12U];
    vlTOPp->out_regs[0x11U] = vlTOPp->VX_register_file__DOT__registers
	[0x11U];
    vlTOPp->out_regs[0x10U] = vlTOPp->VX_register_file__DOT__registers
	[0x10U];
    vlTOPp->out_regs[0xfU] = vlTOPp->VX_register_file__DOT__registers
	[0xfU];
    vlTOPp->out_regs[0xeU] = vlTOPp->VX_register_file__DOT__registers
	[0xeU];
    vlTOPp->out_regs[0xdU] = vlTOPp->VX_register_file__DOT__registers
	[0xdU];
    vlTOPp->out_regs[0xcU] = vlTOPp->VX_register_file__DOT__registers
	[0xcU];
    vlTOPp->out_regs[0xbU] = vlTOPp->VX_register_file__DOT__registers
	[0xbU];
    vlTOPp->out_regs[0xaU] = vlTOPp->VX_register_file__DOT__registers
	[0xaU];
    vlTOPp->out_regs[9U] = vlTOPp->VX_register_file__DOT__registers
	[9U];
    vlTOPp->out_regs[8U] = vlTOPp->VX_register_file__DOT__registers
	[8U];
    vlTOPp->out_regs[7U] = vlTOPp->VX_register_file__DOT__registers
	[7U];
    vlTOPp->out_regs[6U] = vlTOPp->VX_register_file__DOT__registers
	[6U];
    vlTOPp->out_regs[5U] = vlTOPp->VX_register_file__DOT__registers
	[5U];
    vlTOPp->out_regs[4U] = vlTOPp->VX_register_file__DOT__registers
	[4U];
    vlTOPp->out_regs[3U] = vlTOPp->VX_register_file__DOT__registers
	[3U];
    vlTOPp->out_regs[2U] = vlTOPp->VX_register_file__DOT__registers
	[2U];
    vlTOPp->out_regs[1U] = vlTOPp->VX_register_file__DOT__registers
	[1U];
    vlTOPp->out_regs[0U] = vlTOPp->VX_register_file__DOT__registers
	[0U];
}

void VVX_register_file::_eval(VVX_register_file__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file::_eval\n"); );
    VVX_register_file* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
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
    vlTOPp->_settle__TOP__3(vlSymsp);
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
}
#endif // VL_DEBUG

void VVX_register_file::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_register_file::_ctor_var_reset\n"); );
    // Body
    clk = VL_RAND_RESET_I(1);
    in_valid = VL_RAND_RESET_I(1);
    in_write_register = VL_RAND_RESET_I(1);
    in_rd = VL_RAND_RESET_I(5);
    in_data = VL_RAND_RESET_I(32);
    in_src1 = VL_RAND_RESET_I(5);
    in_src2 = VL_RAND_RESET_I(5);
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
	    out_regs[__Vi0] = VL_RAND_RESET_I(32);
    }}
    out_src1_data = VL_RAND_RESET_I(32);
    out_src2_data = VL_RAND_RESET_I(32);
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
	    VX_register_file__DOT__registers[__Vi0] = VL_RAND_RESET_I(32);
    }}
}
