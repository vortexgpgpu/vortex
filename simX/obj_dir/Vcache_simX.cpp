// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vcache_simX.h for the primary calling header

#include "Vcache_simX.h"       // For This
#include "Vcache_simX__Syms.h"

//--------------------
// STATIC VARIABLES


//--------------------

VL_CTOR_IMP(Vcache_simX) {
    Vcache_simX__Syms* __restrict vlSymsp = __VlSymsp = new Vcache_simX__Syms(this, name());
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    VL_CELL (__PVT__v, Vcache_simX_cache_simX);
    // Reset internal values
    
    // Reset structure values
    clk = VL_RAND_RESET_I(1);
    reset = VL_RAND_RESET_I(1);
    in_icache_pc_addr = VL_RAND_RESET_I(32);
    in_icache_valid_pc_addr = VL_RAND_RESET_I(1);
    out_icache_stall = VL_RAND_RESET_I(1);
    in_dcache_mem_read = VL_RAND_RESET_I(3);
    in_dcache_mem_write = VL_RAND_RESET_I(3);
    { int __Vi0=0; for (; __Vi0<4; ++__Vi0) {
	    in_dcache_in_valid[__Vi0] = VL_RAND_RESET_I(1);
    }}
    { int __Vi0=0; for (; __Vi0<4; ++__Vi0) {
	    in_dcache_in_address[__Vi0] = VL_RAND_RESET_I(32);
    }}
    out_dcache_stall = VL_RAND_RESET_I(1);
    __Vclklast__TOP__clk = VL_RAND_RESET_I(1);
    __Vclklast__TOP__reset = VL_RAND_RESET_I(1);
    __Vchglast__TOP__v__dmem_controller__shared_memory__DOT__block_addr = VL_RAND_RESET_I(28);
    __Vm_traceActivity = VL_RAND_RESET_I(32);
}

void Vcache_simX::__Vconfigure(Vcache_simX__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

Vcache_simX::~Vcache_simX() {
    delete __VlSymsp; __VlSymsp=NULL;
}

//--------------------


void Vcache_simX::eval() {
    Vcache_simX__Syms* __restrict vlSymsp = this->__VlSymsp; // Setup global symbol table
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Initialize
    if (VL_UNLIKELY(!vlSymsp->__Vm_didInit)) _eval_initial_loop(vlSymsp);
    // Evaluate till stable
    VL_DEBUG_IF(VL_PRINTF("\n----TOP Evaluate Vcache_simX::eval\n"); );
    int __VclockLoop = 0;
    QData __Vchange=1;
    while (VL_LIKELY(__Vchange)) {
	VL_DEBUG_IF(VL_PRINTF(" Clock loop\n"););
	vlSymsp->__Vm_activity = true;
	_eval(vlSymsp);
	__Vchange = _change_request(vlSymsp);
	if (++__VclockLoop > 100) vl_fatal(__FILE__,__LINE__,__FILE__,"Verilated model didn't converge");
    }
}

void Vcache_simX::_eval_initial_loop(Vcache_simX__Syms* __restrict vlSymsp) {
    vlSymsp->__Vm_didInit = true;
    _eval_initial(vlSymsp);
    vlSymsp->__Vm_activity = true;
    int __VclockLoop = 0;
    QData __Vchange=1;
    while (VL_LIKELY(__Vchange)) {
	_eval_settle(vlSymsp);
	_eval(vlSymsp);
	__Vchange = _change_request(vlSymsp);
	if (++__VclockLoop > 100) vl_fatal(__FILE__,__LINE__,__FILE__,"Verilated model didn't DC converge");
    }
}

//--------------------
// Internal Methods

VL_INLINE_OPT void Vcache_simX::_combo__TOP__1(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_PRINTF("    Vcache_simX::_combo__TOP__1\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlSymsp->TOP__v.in_dcache_in_valid[3U] = vlTOPp->in_dcache_in_valid
	[3U];
    vlSymsp->TOP__v.in_dcache_in_valid[2U] = vlTOPp->in_dcache_in_valid
	[2U];
    vlSymsp->TOP__v.in_dcache_in_valid[1U] = vlTOPp->in_dcache_in_valid
	[1U];
    vlSymsp->TOP__v.in_dcache_in_valid[0U] = vlTOPp->in_dcache_in_valid
	[0U];
    vlSymsp->TOP__v.in_dcache_in_address[3U] = vlTOPp->in_dcache_in_address
	[3U];
    vlSymsp->TOP__v.in_dcache_in_address[2U] = vlTOPp->in_dcache_in_address
	[2U];
    vlSymsp->TOP__v.in_dcache_in_address[1U] = vlTOPp->in_dcache_in_address
	[1U];
    vlSymsp->TOP__v.in_dcache_in_address[0U] = vlTOPp->in_dcache_in_address
	[0U];
}

VL_INLINE_OPT void Vcache_simX::_combo__TOP__3(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_PRINTF("    Vcache_simX::_combo__TOP__3\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->out_icache_stall = ((IData)(vlSymsp->TOP__v__dmem_controller.__PVT__icache__DOT__new_stored_valid) 
				| (0U != (IData)(vlSymsp->TOP__v__dmem_controller.__PVT__icache__DOT__state)));
}

VL_INLINE_OPT void Vcache_simX::_combo__TOP__5(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_PRINTF("    Vcache_simX::_combo__TOP__5\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->out_dcache_stall = ((0U != (IData)(vlSymsp->TOP__v__dmem_controller.__PVT__shared_memory__DOT__vx_priority_encoder_sm__DOT__more_than_one_valid)) 
				| ((0U != (IData)(vlSymsp->TOP__v__dmem_controller.__PVT__dcache__DOT__new_stored_valid)) 
				   | (0U != (IData)(vlSymsp->TOP__v__dmem_controller.__PVT__dcache__DOT__state))));
}

void Vcache_simX::_eval(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_PRINTF("    Vcache_simX::_eval\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlSymsp->TOP__v__dmem_controller._combo__TOP__v__dmem_controller__1(vlSymsp);
    vlTOPp->__Vm_traceActivity = (2U | vlTOPp->__Vm_traceActivity);
    vlTOPp->_combo__TOP__1(vlSymsp);
    if ((((IData)(vlTOPp->clk) & (~ (IData)(vlTOPp->__Vclklast__TOP__clk))) 
	 | ((IData)(vlTOPp->reset) & (~ (IData)(vlTOPp->__Vclklast__TOP__reset))))) {
	vlSymsp->TOP__v__dmem_controller._sequent__TOP__v__dmem_controller__3(vlSymsp);
	vlTOPp->__Vm_traceActivity = (4U | vlTOPp->__Vm_traceActivity);
	vlSymsp->TOP__v._sequent__TOP__v__2(vlSymsp);
	vlSymsp->TOP__v__dmem_controller._sequent__TOP__v__dmem_controller__4(vlSymsp);
    }
    vlSymsp->TOP__v._combo__TOP__v__3(vlSymsp);
    vlSymsp->TOP__v__dmem_controller._combo__TOP__v__dmem_controller__5(vlSymsp);
    vlSymsp->TOP__v__dmem_controller._combo__TOP__v__dmem_controller__7(vlSymsp);
    if ((((IData)(vlTOPp->clk) & (~ (IData)(vlTOPp->__Vclklast__TOP__clk))) 
	 | ((IData)(vlTOPp->reset) & (~ (IData)(vlTOPp->__Vclklast__TOP__reset))))) {
	vlSymsp->TOP__v__dmem_controller._sequent__TOP__v__dmem_controller__8(vlSymsp);
	vlTOPp->__Vm_traceActivity = (8U | vlTOPp->__Vm_traceActivity);
    }
    vlSymsp->TOP__v__dmem_controller._combo__TOP__v__dmem_controller__10(vlSymsp);
    vlSymsp->TOP__v__dmem_controller._combo__TOP__v__dmem_controller__12(vlSymsp);
    vlSymsp->TOP__v__dmem_controller._combo__TOP__v__dmem_controller__14(vlSymsp);
    vlSymsp->TOP__v__dmem_controller._combo__TOP__v__dmem_controller__16(vlSymsp);
    vlSymsp->TOP__v__dmem_controller._combo__TOP__v__dmem_controller__18(vlSymsp);
    vlTOPp->_combo__TOP__3(vlSymsp);
    vlSymsp->TOP__v__dmem_controller._combo__TOP__v__dmem_controller__20(vlSymsp);
    vlSymsp->TOP__v__dmem_controller._combo__TOP__v__dmem_controller__22(vlSymsp);
    vlTOPp->_combo__TOP__5(vlSymsp);
    // Final
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
    vlTOPp->__Vclklast__TOP__reset = vlTOPp->reset;
}

void Vcache_simX::_eval_initial(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_PRINTF("    Vcache_simX::_eval_initial\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void Vcache_simX::final() {
    VL_DEBUG_IF(VL_PRINTF("    Vcache_simX::final\n"); );
    // Variables
    Vcache_simX__Syms* __restrict vlSymsp = this->__VlSymsp;
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void Vcache_simX::_eval_settle(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_PRINTF("    Vcache_simX::_eval_settle\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlSymsp->TOP__v__dmem_controller._combo__TOP__v__dmem_controller__1(vlSymsp);
    vlTOPp->__Vm_traceActivity = (1U | vlTOPp->__Vm_traceActivity);
    vlTOPp->_combo__TOP__1(vlSymsp);
    vlSymsp->TOP__v._settle__TOP__v__1(vlSymsp);
    vlSymsp->TOP__v._settle__TOP__v__4(vlSymsp);
    vlSymsp->TOP__v__dmem_controller._settle__TOP__v__dmem_controller__6(vlSymsp);
    vlSymsp->TOP__v__dmem_controller._settle__TOP__v__dmem_controller__9(vlSymsp);
    vlSymsp->TOP__v__dmem_controller._settle__TOP__v__dmem_controller__11(vlSymsp);
    vlSymsp->TOP__v__dmem_controller._settle__TOP__v__dmem_controller__13(vlSymsp);
    vlSymsp->TOP__v__dmem_controller._settle__TOP__v__dmem_controller__15(vlSymsp);
    vlSymsp->TOP__v__dmem_controller._settle__TOP__v__dmem_controller__17(vlSymsp);
    vlSymsp->TOP__v__dmem_controller._settle__TOP__v__dmem_controller__19(vlSymsp);
    vlTOPp->_combo__TOP__3(vlSymsp);
    vlSymsp->TOP__v__dmem_controller._settle__TOP__v__dmem_controller__21(vlSymsp);
    vlTOPp->_combo__TOP__5(vlSymsp);
}

VL_INLINE_OPT QData Vcache_simX::_change_request(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_PRINTF("    Vcache_simX::_change_request\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // Change detection
    QData __req = false;  // Logically a bool
    __req |= ((vlSymsp->TOP__v__dmem_controller.__PVT__shared_memory__DOT__block_addr ^ vlTOPp->__Vchglast__TOP__v__dmem_controller__shared_memory__DOT__block_addr));
    VL_DEBUG_IF( if(__req && ((vlSymsp->TOP__v__dmem_controller.__PVT__shared_memory__DOT__block_addr ^ vlTOPp->__Vchglast__TOP__v__dmem_controller__shared_memory__DOT__block_addr))) VL_PRINTF("	CHANGE: ../rtl/shared_memory/VX_shared_memory.v:49: shared_memory.block_addr\n"); );
    // Final
    vlTOPp->__Vchglast__TOP__v__dmem_controller__shared_memory__DOT__block_addr 
	= vlSymsp->TOP__v__dmem_controller.__PVT__shared_memory__DOT__block_addr;
    return __req;
}
