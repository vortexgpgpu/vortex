// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vrf2_32x128_wm1.h for the primary calling header

#include "Vrf2_32x128_wm1.h"
#include "Vrf2_32x128_wm1__Syms.h"


//--------------------
// STATIC VARIABLES


//--------------------

VL_CTOR_IMP(Vrf2_32x128_wm1) {
    Vrf2_32x128_wm1__Syms* __restrict vlSymsp = __VlSymsp = new Vrf2_32x128_wm1__Syms(this, name());
    Vrf2_32x128_wm1* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Reset internal values
    
    // Reset structure values
    _ctor_var_reset();
}

void Vrf2_32x128_wm1::__Vconfigure(Vrf2_32x128_wm1__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

Vrf2_32x128_wm1::~Vrf2_32x128_wm1() {
    delete __VlSymsp; __VlSymsp=NULL;
}

//--------------------


void Vrf2_32x128_wm1::eval() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate Vrf2_32x128_wm1::eval\n"); );
    Vrf2_32x128_wm1__Syms* __restrict vlSymsp = this->__VlSymsp;  // Setup global symbol table
    Vrf2_32x128_wm1* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
#ifdef VL_DEBUG
    // Debug assertions
    _eval_debug_assertions();
#endif  // VL_DEBUG
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

void Vrf2_32x128_wm1::_eval_initial_loop(Vrf2_32x128_wm1__Syms* __restrict vlSymsp) {
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

VL_INLINE_OPT void Vrf2_32x128_wm1::_combo__TOP__1(Vrf2_32x128_wm1__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vrf2_32x128_wm1::_combo__TOP__1\n"); );
    Vrf2_32x128_wm1* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // ALWAYS at rf2_32x128_wm1.v:15356
    if ((1U & (((~ (IData)(vlTOPp->CEN)) & (~ (IData)(vlTOPp->DFTRAMBYP))) 
	       & (~ (IData)(vlTOPp->SE))))) {
	vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__list_complete = 0U;
	vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__i = 0U;
	vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__Q_output[0U] 
	    = vlTOPp->Q_in[0U];
	vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__Q_output[1U] 
	    = vlTOPp->Q_in[1U];
	vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__Q_output[2U] 
	    = vlTOPp->Q_in[2U];
	vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__Q_output[3U] 
	    = vlTOPp->Q_in[3U];
	while ((1U & (~ (IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__list_complete)))) {
	    vlTOPp->rf2_32x128_wm1_error_injection__DOT__fault_entry 
		= vlTOPp->rf2_32x128_wm1_error_injection__DOT__fault_table
		[(0xfU & vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__i)];
	    vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__row_address 
		= (0xfU & (vlTOPp->rf2_32x128_wm1_error_injection__DOT__fault_entry 
			   >> 0xdU));
	    vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__column_address 
		= (1U & (vlTOPp->rf2_32x128_wm1_error_injection__DOT__fault_entry 
			 >> 0xcU));
	    vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__bitPlace 
		= (0x7fU & (vlTOPp->rf2_32x128_wm1_error_injection__DOT__fault_entry 
			    >> 5U));
	    vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__fault_type 
		= (3U & (vlTOPp->rf2_32x128_wm1_error_injection__DOT__fault_entry 
			 >> 3U));
	    vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__red_fault 
		= (3U & (vlTOPp->rf2_32x128_wm1_error_injection__DOT__fault_entry 
			 >> 1U));
	    vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__valid 
		= (1U & vlTOPp->rf2_32x128_wm1_error_injection__DOT__fault_entry);
	    vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__i 
		= ((IData)(1U) + vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__i);
	    if (vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__valid) {
		if ((0U == (IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__red_fault))) {
		    if ((((IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__row_address) 
			  == (0xfU & ((IData)(vlTOPp->A) 
				      >> 1U))) & ((IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__column_address) 
						  == 
						  (1U 
						   & (IData)(vlTOPp->A))))) {
			if ((0x40U > (IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__bitPlace))) {
			    // Function: bit_error at rf2_32x128_wm1.v:15345
			    vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__1__bitLoc 
				= vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__bitPlace;
			    vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__1__fault_type 
				= vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__fault_type;
			    vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__Q_output[((IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__1__bitLoc) 
										>> 5U)] 
				= (((~ ((IData)(1U) 
					<< (0x1fU & (IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__1__bitLoc)))) 
				    & vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__Q_output[
				    ((IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__1__bitLoc) 
				     >> 5U)]) | (((0U 
						   != (IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__1__fault_type)) 
						  & ((1U 
						      == (IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__1__fault_type)) 
						     | (~ 
							(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__Q_output[
							 ((IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__1__bitLoc) 
							  >> 5U)] 
							 >> 
							 (0x1fU 
							  & (IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__1__bitLoc)))))) 
						 << 
						 (0x1fU 
						  & (IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__1__bitLoc))));
			} else {
			    if ((0x40U <= (IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__bitPlace))) {
				// Function: bit_error at rf2_32x128_wm1.v:15347
				vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__2__bitLoc 
				    = vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__bitPlace;
				vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__2__fault_type 
				    = vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__fault_type;
				vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__Q_output[((IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__2__bitLoc) 
										>> 5U)] 
				    = (((~ ((IData)(1U) 
					    << (0x1fU 
						& (IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__2__bitLoc)))) 
					& vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__Q_output[
					((IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__2__bitLoc) 
					 >> 5U)]) | 
				       (((0U != (IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__2__fault_type)) 
					 & ((1U == (IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__2__fault_type)) 
					    | (~ (vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__Q_output[
						  ((IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__2__bitLoc) 
						   >> 5U)] 
						  >> 
						  (0x1fU 
						   & (IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__2__bitLoc)))))) 
					<< (0x1fU & (IData)(vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__2__bitLoc))));
			    }
			}
		    }
		}
	    } else {
		vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__list_complete = 1U;
	    }
	}
	vlTOPp->Q_out[0U] = vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__Q_output[0U];
	vlTOPp->Q_out[1U] = vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__Q_output[1U];
	vlTOPp->Q_out[2U] = vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__Q_output[2U];
	vlTOPp->Q_out[3U] = vlTOPp->__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__Q_output[3U];
    } else {
	vlTOPp->Q_out[0U] = vlTOPp->Q_in[0U];
	vlTOPp->Q_out[1U] = vlTOPp->Q_in[1U];
	vlTOPp->Q_out[2U] = vlTOPp->Q_in[2U];
	vlTOPp->Q_out[3U] = vlTOPp->Q_in[3U];
    }
}

void Vrf2_32x128_wm1::_eval(Vrf2_32x128_wm1__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vrf2_32x128_wm1::_eval\n"); );
    Vrf2_32x128_wm1* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->_combo__TOP__1(vlSymsp);
}

void Vrf2_32x128_wm1::_eval_initial(Vrf2_32x128_wm1__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vrf2_32x128_wm1::_eval_initial\n"); );
    Vrf2_32x128_wm1* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void Vrf2_32x128_wm1::final() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vrf2_32x128_wm1::final\n"); );
    // Variables
    Vrf2_32x128_wm1__Syms* __restrict vlSymsp = this->__VlSymsp;
    Vrf2_32x128_wm1* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void Vrf2_32x128_wm1::_eval_settle(Vrf2_32x128_wm1__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vrf2_32x128_wm1::_eval_settle\n"); );
    Vrf2_32x128_wm1* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->_combo__TOP__1(vlSymsp);
}

VL_INLINE_OPT QData Vrf2_32x128_wm1::_change_request(Vrf2_32x128_wm1__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vrf2_32x128_wm1::_change_request\n"); );
    Vrf2_32x128_wm1* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // Change detection
    QData __req = false;  // Logically a bool
    return __req;
}

#ifdef VL_DEBUG
void Vrf2_32x128_wm1::_eval_debug_assertions() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vrf2_32x128_wm1::_eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((CLK & 0xfeU))) {
	Verilated::overWidthError("CLK");}
    if (VL_UNLIKELY((A & 0xe0U))) {
	Verilated::overWidthError("A");}
    if (VL_UNLIKELY((CEN & 0xfeU))) {
	Verilated::overWidthError("CEN");}
    if (VL_UNLIKELY((DFTRAMBYP & 0xfeU))) {
	Verilated::overWidthError("DFTRAMBYP");}
    if (VL_UNLIKELY((SE & 0xfeU))) {
	Verilated::overWidthError("SE");}
}
#endif // VL_DEBUG

void Vrf2_32x128_wm1::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    Vrf2_32x128_wm1::_ctor_var_reset\n"); );
    // Body
    VL_RAND_RESET_W(128,Q_out);
    VL_RAND_RESET_W(128,Q_in);
    CLK = VL_RAND_RESET_I(1);
    A = VL_RAND_RESET_I(5);
    CEN = VL_RAND_RESET_I(1);
    DFTRAMBYP = VL_RAND_RESET_I(1);
    SE = VL_RAND_RESET_I(1);
    { int __Vi0=0; for (; __Vi0<16; ++__Vi0) {
	    rf2_32x128_wm1_error_injection__DOT__fault_table[__Vi0] = VL_RAND_RESET_I(17);
    }}
    rf2_32x128_wm1_error_injection__DOT__fault_entry = VL_RAND_RESET_I(17);
    VL_RAND_RESET_W(128,__Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__Q_output);
    __Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__list_complete = VL_RAND_RESET_I(1);
    __Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__i = VL_RAND_RESET_I(32);
    __Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__row_address = VL_RAND_RESET_I(4);
    __Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__column_address = VL_RAND_RESET_I(1);
    __Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__bitPlace = VL_RAND_RESET_I(7);
    __Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__fault_type = VL_RAND_RESET_I(2);
    __Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__red_fault = VL_RAND_RESET_I(2);
    __Vtask_rf2_32x128_wm1_error_injection__DOT__error_injection_on_output__0__valid = VL_RAND_RESET_I(1);
    __Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__1__fault_type = VL_RAND_RESET_I(2);
    __Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__1__bitLoc = VL_RAND_RESET_I(7);
    __Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__2__fault_type = VL_RAND_RESET_I(2);
    __Vtask_rf2_32x128_wm1_error_injection__DOT__bit_error__2__bitLoc = VL_RAND_RESET_I(7);
}
