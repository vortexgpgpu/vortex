// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vvortex.h for the primary calling header

#include "Vvortex.h"
#include "Vvortex__Syms.h"


//--------------------
// STATIC VARIABLES


//--------------------

VL_CTOR_IMP(Vvortex) {
    Vvortex__Syms* __restrict vlSymsp = __VlSymsp = new Vvortex__Syms(this, name());
    Vvortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Reset internal values
    
    // Reset structure values
    _ctor_var_reset();
}

void Vvortex::__Vconfigure(Vvortex__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

Vvortex::~Vvortex() {
    delete __VlSymsp; __VlSymsp=NULL;
}

//--------------------


void Vvortex::eval() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vvortex::eval\n"); );
    Vvortex__Syms* __restrict vlSymsp = this->__VlSymsp;  // Setup global symbol table
    Vvortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
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

void Vvortex::_eval_initial_loop(Vvortex__Syms* __restrict vlSymsp) {
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

void Vvortex::_settle__TOP__1(Vvortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vvortex::_settle__TOP__1\n"); );
    Vvortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->fe_delay = 0U;
    vlTOPp->de_instruction = vlTOPp->vortex__DOT__vx_f_d_reg__DOT__instruction;
}

VL_INLINE_OPT void Vvortex::_sequent__TOP__2(Vvortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vvortex::_sequent__TOP__2\n"); );
    Vvortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // ALWAYS at VX_fetch.v:128
    vlTOPp->vortex__DOT__vx_fetch__DOT__delay_reg = 0U;
    // ALWAYS at VX_fetch.v:128
    vlTOPp->vortex__DOT__vx_fetch__DOT__stall_reg = 0U;
    // ALWAYS at VX_fetch.v:128
    vlTOPp->vortex__DOT__vx_fetch__DOT__JAL_reg = ((IData)(vlTOPp->reset)
						    ? 0U
						    : 4U);
    // ALWAYS at VX_fetch.v:128
    vlTOPp->vortex__DOT__vx_fetch__DOT__BR_reg = ((IData)(vlTOPp->reset)
						   ? 0U
						   : 4U);
    // ALWAYS at VX_fetch.v:128
    vlTOPp->vortex__DOT__vx_fetch__DOT__real_PC = ((IData)(vlTOPp->reset)
						    ? 0U
						    : 
						   ((IData)(4U) 
						    + vlTOPp->vortex__DOT__vx_fetch__DOT__PC_to_use));
    // ALWAYS at VX_fetch.v:128
    vlTOPp->vortex__DOT__vx_fetch__DOT__old = ((IData)(vlTOPp->reset)
					        ? 0U
					        : vlTOPp->vortex__DOT__vx_fetch__DOT__PC_to_use);
    // ALWAYS at VX_fetch.v:128
    vlTOPp->vortex__DOT__vx_fetch__DOT__state = ((IData)(vlTOPp->reset)
						  ? 0U
						  : 
						 ((IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__prev_debug)
						   ? 4U
						   : 0U));
    // ALWAYS at VX_fetch.v:128
    vlTOPp->vortex__DOT__vx_fetch__DOT__prev_debug = 0U;
    // ALWAYS at VX_fetch.v:71
    vlTOPp->vortex__DOT__vx_fetch__DOT__PC_to_use = 
	((IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__delay_reg)
	  ? vlTOPp->vortex__DOT__vx_fetch__DOT__old
	  : ((IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__stall_reg)
	      ? vlTOPp->vortex__DOT__vx_fetch__DOT__old
	      : ((0x10U & (IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__state))
		  ? 0U : ((8U & (IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__state))
			   ? 0U : ((4U & (IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__state))
				    ? ((2U & (IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__state))
				        ? 0U : ((1U 
						 & (IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__state))
						 ? 0U
						 : vlTOPp->vortex__DOT__vx_fetch__DOT__old))
				    : ((2U & (IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__state))
				        ? ((1U & (IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__state))
					    ? vlTOPp->vortex__DOT__vx_fetch__DOT__real_PC
					    : vlTOPp->vortex__DOT__vx_fetch__DOT__BR_reg)
				        : ((1U & (IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__state))
					    ? vlTOPp->vortex__DOT__vx_fetch__DOT__JAL_reg
					    : vlTOPp->vortex__DOT__vx_fetch__DOT__real_PC)))))));
    vlTOPp->curr_PC = vlTOPp->vortex__DOT__vx_fetch__DOT__PC_to_use;
}

VL_INLINE_OPT void Vvortex::_sequent__TOP__3(Vvortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vvortex::_sequent__TOP__3\n"); );
    Vvortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // ALWAYS at VX_f_d_reg.v:17
    VL_WRITEF("Fetch Inst: %10#\tDecode Inst: %10#\n",
	      32,vlTOPp->fe_instruction,32,vlTOPp->vortex__DOT__vx_f_d_reg__DOT__instruction);
}

void Vvortex::_initial__TOP__4(Vvortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vvortex::_initial__TOP__4\n"); );
    Vvortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // INITIAL at VX_fetch.v:44
    vlTOPp->vortex__DOT__vx_fetch__DOT__stall_reg = 0U;
    vlTOPp->vortex__DOT__vx_fetch__DOT__delay_reg = 0U;
    vlTOPp->vortex__DOT__vx_fetch__DOT__old = 0U;
    vlTOPp->vortex__DOT__vx_fetch__DOT__state = 0U;
    vlTOPp->vortex__DOT__vx_fetch__DOT__real_PC = 0U;
    vlTOPp->vortex__DOT__vx_fetch__DOT__JAL_reg = 0U;
    vlTOPp->vortex__DOT__vx_fetch__DOT__BR_reg = 0U;
    vlTOPp->vortex__DOT__vx_fetch__DOT__prev_debug = 0U;
}

VL_INLINE_OPT void Vvortex::_sequent__TOP__5(Vvortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vvortex::_sequent__TOP__5\n"); );
    Vvortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // ALWAYS at VX_f_d_reg.v:26
    vlTOPp->vortex__DOT__vx_f_d_reg__DOT__instruction 
	= ((IData)(vlTOPp->reset) ? 0U : vlTOPp->fe_instruction);
    vlTOPp->de_instruction = vlTOPp->vortex__DOT__vx_f_d_reg__DOT__instruction;
}

void Vvortex::_settle__TOP__6(Vvortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vvortex::_settle__TOP__6\n"); );
    Vvortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // ALWAYS at VX_fetch.v:71
    vlTOPp->vortex__DOT__vx_fetch__DOT__PC_to_use = 
	((IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__delay_reg)
	  ? vlTOPp->vortex__DOT__vx_fetch__DOT__old
	  : ((IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__stall_reg)
	      ? vlTOPp->vortex__DOT__vx_fetch__DOT__old
	      : ((0x10U & (IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__state))
		  ? 0U : ((8U & (IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__state))
			   ? 0U : ((4U & (IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__state))
				    ? ((2U & (IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__state))
				        ? 0U : ((1U 
						 & (IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__state))
						 ? 0U
						 : vlTOPp->vortex__DOT__vx_fetch__DOT__old))
				    : ((2U & (IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__state))
				        ? ((1U & (IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__state))
					    ? vlTOPp->vortex__DOT__vx_fetch__DOT__real_PC
					    : vlTOPp->vortex__DOT__vx_fetch__DOT__BR_reg)
				        : ((1U & (IData)(vlTOPp->vortex__DOT__vx_fetch__DOT__state))
					    ? vlTOPp->vortex__DOT__vx_fetch__DOT__JAL_reg
					    : vlTOPp->vortex__DOT__vx_fetch__DOT__real_PC)))))));
    vlTOPp->curr_PC = vlTOPp->vortex__DOT__vx_fetch__DOT__PC_to_use;
}

void Vvortex::_eval(Vvortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vvortex::_eval\n"); );
    Vvortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    if ((((IData)(vlTOPp->clk) & (~ (IData)(vlTOPp->__Vclklast__TOP__clk))) 
	 | ((IData)(vlTOPp->reset) & (~ (IData)(vlTOPp->__Vclklast__TOP__reset))))) {
	vlTOPp->_sequent__TOP__2(vlSymsp);
    }
    if (((IData)(vlTOPp->clk) & (~ (IData)(vlTOPp->__Vclklast__TOP__clk)))) {
	vlTOPp->_sequent__TOP__3(vlSymsp);
    }
    if ((((IData)(vlTOPp->clk) & (~ (IData)(vlTOPp->__Vclklast__TOP__clk))) 
	 | ((IData)(vlTOPp->reset) & (~ (IData)(vlTOPp->__Vclklast__TOP__reset))))) {
	vlTOPp->_sequent__TOP__5(vlSymsp);
    }
    // Final
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
    vlTOPp->__Vclklast__TOP__reset = vlTOPp->reset;
}

void Vvortex::_eval_initial(Vvortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vvortex::_eval_initial\n"); );
    Vvortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
    vlTOPp->__Vclklast__TOP__reset = vlTOPp->reset;
    vlTOPp->_initial__TOP__4(vlSymsp);
}

void Vvortex::final() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vvortex::final\n"); );
    // Variables
    Vvortex__Syms* __restrict vlSymsp = this->__VlSymsp;
    Vvortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void Vvortex::_eval_settle(Vvortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vvortex::_eval_settle\n"); );
    Vvortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->_settle__TOP__1(vlSymsp);
    vlTOPp->_settle__TOP__6(vlSymsp);
}

VL_INLINE_OPT QData Vvortex::_change_request(Vvortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vvortex::_change_request\n"); );
    Vvortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // Change detection
    QData __req = false;  // Logically a bool
    return __req;
}

#ifdef VL_DEBUG
void Vvortex::_eval_debug_assertions() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vvortex::_eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((clk & 0xfeU))) {
	Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((reset & 0xfeU))) {
	Verilated::overWidthError("reset");}
}
#endif // VL_DEBUG

void Vvortex::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vvortex::_ctor_var_reset\n"); );
    // Body
    clk = VL_RAND_RESET_I(1);
    reset = VL_RAND_RESET_I(1);
    fe_instruction = VL_RAND_RESET_I(32);
    curr_PC = VL_RAND_RESET_I(32);
    de_instruction = VL_RAND_RESET_I(32);
    fe_delay = VL_RAND_RESET_I(1);
    vortex__DOT__vx_fetch__DOT__stall_reg = VL_RAND_RESET_I(1);
    vortex__DOT__vx_fetch__DOT__delay_reg = VL_RAND_RESET_I(1);
    vortex__DOT__vx_fetch__DOT__old = VL_RAND_RESET_I(32);
    vortex__DOT__vx_fetch__DOT__state = VL_RAND_RESET_I(5);
    vortex__DOT__vx_fetch__DOT__real_PC = VL_RAND_RESET_I(32);
    vortex__DOT__vx_fetch__DOT__JAL_reg = VL_RAND_RESET_I(32);
    vortex__DOT__vx_fetch__DOT__BR_reg = VL_RAND_RESET_I(32);
    vortex__DOT__vx_fetch__DOT__prev_debug = VL_RAND_RESET_I(1);
    vortex__DOT__vx_fetch__DOT__PC_to_use = VL_RAND_RESET_I(32);
    vortex__DOT__vx_f_d_reg__DOT__instruction = VL_RAND_RESET_I(32);
}
