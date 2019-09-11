// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVortex.h for the primary calling header

#include "VVortex.h"
#include "VVortex__Syms.h"


//--------------------
// STATIC VARIABLES

// Begin mtask footprint  all: 
VL_ST_SIG8(VVortex::__Vtable1_Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu[8],4,0);

//--------------------

VL_CTOR_IMP(VVortex) {
    VVortex__Syms* __restrict vlSymsp = __VlSymsp = new VVortex__Syms(this, name());
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    VL_CELL(__PVT__Vortex__DOT__VX_dcache_rsp, VVortex_VX_dcache_response_inter);
    VL_CELL(__PVT__Vortex__DOT__VX_dcache_req, VVortex_VX_dcache_request_inter);
    VL_CELL(__PVT__Vortex__DOT__VX_bckE_req, VVortex_VX_frE_to_bckE_req_inter);
    VL_CELL(__PVT__Vortex__DOT__VX_writeback_inter, VVortex_VX_wb_inter);
    VL_CELL(__PVT__Vortex__DOT__VX_branch_rsp, VVortex_VX_branch_response_inter);
    VL_CELL(__PVT__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl, VVortex_VX_warp_ctl_inter);
    VL_CELL(__PVT__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd, VVortex_VX_inst_meta_inter);
    VL_CELL(__PVT__Vortex__DOT__vx_front_end__DOT__VX_frE_to_bckE_req, VVortex_VX_frE_to_bckE_req_inter);
    VL_CELL(__PVT__Vortex__DOT__vx_front_end__DOT__fd_inst_meta_de, VVortex_VX_inst_meta_inter);
    VL_CELL(__PVT__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req, VVortex_VX_mem_req_inter);
    VL_CELL(__PVT__Vortex__DOT__vx_back_end__DOT__VX_mem_req, VVortex_VX_mem_req_inter);
    VL_CELL(__PVT__Vortex__DOT__vx_back_end__DOT__VX_mem_wb, VVortex_VX_inst_mem_wb_inter);
    // Reset internal values
    
    // Reset structure values
    _ctor_var_reset();
}

void VVortex::__Vconfigure(VVortex__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

VVortex::~VVortex() {
    delete __VlSymsp; __VlSymsp=NULL;
}

//--------------------


void VVortex::eval() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate VVortex::eval\n"); );
    VVortex__Syms* __restrict vlSymsp = this->__VlSymsp;  // Setup global symbol table
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
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

void VVortex::_eval_initial_loop(VVortex__Syms* __restrict vlSymsp) {
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

void VVortex::_settle__TOP__1(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_settle__TOP__1\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jmprt_thread_mask 
	= (1U | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jmprt_thread_mask));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jmprt_thread_mask 
	= (0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jmprt_thread_mask));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jmprt_thread_mask 
	= (0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jmprt_thread_mask));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jmprt_thread_mask 
	= (7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jmprt_thread_mask));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_rsp.in_cache_driver_out_data[3U] 
	= vlTOPp->in_cache_driver_out_data[3U];
    vlSymsp->TOP__Vortex__DOT__VX_dcache_rsp.in_cache_driver_out_data[2U] 
	= vlTOPp->in_cache_driver_out_data[2U];
    vlSymsp->TOP__Vortex__DOT__VX_dcache_rsp.in_cache_driver_out_data[1U] 
	= vlTOPp->in_cache_driver_out_data[1U];
    vlSymsp->TOP__Vortex__DOT__VX_dcache_rsp.in_cache_driver_out_data[0U] 
	= vlTOPp->in_cache_driver_out_data[0U];
    vlTOPp->Vortex__DOT__execute_branch_stall = (1U 
						 & ((0U 
						     != 
						     (7U 
						      & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
							  << 3U) 
							 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
							    >> 0x1dU)))) 
						    | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
						       >> 8U)));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[0U] 
	= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[1U] 
	= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[2U] 
	= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[3U] 
	= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2 
	= ((0x40U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	    ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
		<< 0x1aU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
			     >> 6U)) : ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
					   >> 0xeU)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2 
	= ((0x40U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	    ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
		<< 0x1aU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
			     >> 6U)) : ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[7U] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
					   >> 0xeU)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2 
	= ((0x40U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	    ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
		<< 0x1aU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
			     >> 6U)) : ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[8U] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[7U] 
					   >> 0xeU)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2 
	= ((0x40U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	    ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
		<< 0x1aU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
			     >> 6U)) : ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[8U] 
					   >> 0xeU)));
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb 
	= (3U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
		  << 0x19U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
			       >> 7U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__write_enable 
	= ((0U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) 
	   & ((0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 8U)))) 
	      & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				  << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 0xaU))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__write_enable 
	= ((1U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) 
	   & ((0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 8U)))) 
	      & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				  << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 0xaU))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__write_enable 
	= ((2U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) 
	   & ((0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 8U)))) 
	      & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				  << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 0xaU))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__write_enable 
	= ((3U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) 
	   & ((0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 8U)))) 
	      & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				  << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 0xaU))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__write_enable 
	= ((4U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) 
	   & ((0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 8U)))) 
	      & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				  << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 0xaU))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__write_enable 
	= ((5U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) 
	   & ((0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 8U)))) 
	      & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				  << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 0xaU))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__write_enable 
	= ((6U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) 
	   & ((0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 8U)))) 
	      & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				  << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 0xaU))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__write_enable 
	= ((7U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) 
	   & ((0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 8U)))) 
	      & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				  << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 0xaU))))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data[0U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data[1U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data[2U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data[3U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_itype 
	= ((0x13U == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U)))) 
	   | (3U == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			       << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					    >> 8U)))));
    vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_frE_to_bckE_req.csr_address 
	= (0xfffU & (((0U != (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				     << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						 >> 0x14U)))) 
		      & (2U <= (0xfffU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					   << 4U) | 
					  (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					   >> 0x1cU)))))
		      ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			  << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				    >> 0x1cU)) : 0x55U));
    // ALWAYS at VX_decode.v:399
    vlTOPp->__Vtableidx1 = (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				   << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 0x14U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu 
	= vlTOPp->__Vtable1_Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu
	[vlTOPp->__Vtableidx1];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__alu_tempp 
	= (0xfffU & (((1U == (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				     << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						 >> 0x14U)))) 
		      | (5U == (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				       << 0xcU) | (
						   vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						   >> 0x14U)))))
		      ? (0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				   << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 0x1cU)))
		      : ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			  << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				    >> 0x1cU))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data[0U] 
	= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data[1U] 
	= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data[2U] 
	= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data[3U] 
	= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_csr 
	= ((0x73U == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U)))) 
	   & (0U != (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			    << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					>> 0x14U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[1U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[2U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[3U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[4U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[5U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[6U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[7U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[8U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[9U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0xaU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0xbU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0xcU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0xdU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0xeU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0xfU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x10U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x11U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x12U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x13U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x14U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x15U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x16U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x17U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x18U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x19U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x1aU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x1bU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x1cU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x1dU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x1eU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x1fU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_wspawn 
	= ((0x6bU == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U)))) 
	   & (0U == (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			    << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					>> 0x14U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jal_sys_jal 
	= ((0U == (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			  << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				      >> 0x14U)))) 
	   & (2U > (0xfffU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			       << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					 >> 0x1cU)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid[0U] 
	= (1U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid[1U] 
	= (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
		 >> 1U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid[2U] 
	= (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
		 >> 2U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid[3U] 
	= (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
		 >> 3U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jmprt 
	= ((0x6bU == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U)))) 
	   & (4U == (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			    << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					>> 0x14U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs 
	= ((0x6bU == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U)))) 
	   & (6U == (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			    << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					>> 0x14U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[1U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[2U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[3U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[4U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[5U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[6U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[7U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[8U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[9U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0xaU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0xbU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0xcU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0xdU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0xeU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0xfU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x10U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x11U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x12U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x13U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x14U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x15U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x16U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x17U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x18U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x19U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x1aU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x1bU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x1cU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x1dU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x1eU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x1fU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][3U];
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[0U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[1U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[2U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[3U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->out_cache_driver_in_mem_read = (7U & ((
						   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
						   << 1U) 
						  | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
						     >> 0x1fU)));
    vlTOPp->out_cache_driver_in_mem_write = (7U & (
						   (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
						    << 4U) 
						   | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
						      >> 0x1cU)));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_address[0U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
	    << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
			 >> 2U));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_data[0U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[4U] 
	    << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[3U] 
			 >> 0xbU));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_valid[0U] 
	= (1U & (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U] 
		 >> 4U));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_address[1U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xaU] 
	    << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
			 >> 2U));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_data[1U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[5U] 
	    << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[4U] 
			 >> 0xbU));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_valid[1U] 
	= (1U & (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U] 
		 >> 5U));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_address[2U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xbU] 
	    << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xaU] 
			 >> 2U));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_data[2U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[6U] 
	    << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[5U] 
			 >> 0xbU));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_valid[2U] 
	= (1U & (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U] 
		 >> 6U));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_address[3U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
	    << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xbU] 
			 >> 2U));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_data[3U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
	    << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[6U] 
			 >> 0xbU));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_valid[3U] 
	= (1U & (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U] 
		 >> 7U));
    vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[2U] 
	     << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[1U] 
			  >> 0xbU)) + ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[1U] 
					<< 0x16U) | 
				       (0x3ffffeU & 
					(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U] 
					 >> 0xaU))));
    // ALWAYS at VX_memory.v:59
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir 
	= (1U & ((0x400U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])
		  ? ((0x200U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])
		      ? ((~ (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U] 
			     >> 8U)) & (~ (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
					   >> 1U)))
		      : ((0x100U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])
			  ? (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
			     >> 1U) : (~ (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
					  >> 1U))))
		  : ((0x200U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])
		      ? ((0x100U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])
			  ? (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
			     >> 1U) : (0U != ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
					       << 0x1eU) 
					      | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
						 >> 2U))))
		      : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U] 
			  >> 8U) & (0U == ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
					    << 0x1eU) 
					   | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					      >> 2U)))))));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[0U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[3U] 
	    << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[2U] 
			 >> 0xbU));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[1U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[3U] 
	    << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[2U] 
			 >> 0xbU));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[2U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[3U] 
	    << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[2U] 
			 >> 0xbU));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[3U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[3U] 
	    << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[2U] 
			 >> 0xbU));
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[0U] 
	= vlSymsp->TOP__Vortex__DOT__VX_dcache_rsp.in_cache_driver_out_data
	[0U];
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[1U] 
	= vlSymsp->TOP__Vortex__DOT__VX_dcache_rsp.in_cache_driver_out_data
	[1U];
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[2U] 
	= vlSymsp->TOP__Vortex__DOT__VX_dcache_rsp.in_cache_driver_out_data
	[2U];
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[3U] 
	= vlSymsp->TOP__Vortex__DOT__VX_dcache_rsp.in_cache_driver_out_data
	[3U];
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__mult_signed_result 
	= VL_MULS_QQQ(64,64,64, VL_EXTENDS_QI(64,32, 
					      ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						<< 0x12U) 
					       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						  >> 0xeU))), 
		      VL_EXTENDS_QI(64,32, vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__mult_signed_result 
	= VL_MULS_QQQ(64,64,64, VL_EXTENDS_QI(64,32, 
					      ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						<< 0x12U) 
					       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						  >> 0xeU))), 
		      VL_EXTENDS_QI(64,32, vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__mult_signed_result 
	= VL_MULS_QQQ(64,64,64, VL_EXTENDS_QI(64,32, 
					      ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						<< 0x12U) 
					       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						  >> 0xeU))), 
		      VL_EXTENDS_QI(64,32, vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__mult_signed_result 
	= VL_MULS_QQQ(64,64,64, VL_EXTENDS_QI(64,32, 
					      ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						<< 0x12U) 
					       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						  >> 0xeU))), 
		      VL_EXTENDS_QI(64,32, vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd 
	= (((((0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			 << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				   >> 0x17U))) == (0x1fU 
						   & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
						       << 8U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
							 >> 0x18U)))) 
	     & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				 << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					   >> 0x17U))))) 
	    & (0U != (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))) 
	   & ((0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
				     >> 4U))) == (0xfU 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U])));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd 
	= (((((0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			 << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				   >> 0x1cU))) == (0x1fU 
						   & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
						       << 8U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
							 >> 0x18U)))) 
	     & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				 << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					   >> 0x1cU))))) 
	    & (0U != (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))) 
	   & ((0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
				     >> 4U))) == (0xfU 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U])));
    vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
	= ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			  << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 8U)))) ? 
	   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data[0U]
	    : ((1U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			      << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					   >> 8U))))
	        ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[6U] 
		    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[5U] 
				 >> 0xfU)) : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					       << 0x11U) 
					      | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						 >> 0xfU))));
    vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
	= ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			  << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 8U)))) ? 
	   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data[1U]
	    : ((1U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			      << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					   >> 8U))))
	        ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[7U] 
		    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[6U] 
				 >> 0xfU)) : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[3U] 
					       << 0x11U) 
					      | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						 >> 0xfU))));
    vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
	= ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			  << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 8U)))) ? 
	   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data[2U]
	    : ((1U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			      << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					   >> 8U))))
	        ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[8U] 
		    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[7U] 
				 >> 0xfU)) : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[4U] 
					       << 0x11U) 
					      | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[3U] 
						 >> 0xfU))));
    vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
	= ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			  << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 8U)))) ? 
	   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data[3U]
	    : ((1U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			      << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					   >> 8U))))
	        ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[9U] 
		    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[8U] 
				 >> 0xfU)) : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[5U] 
					       << 0x11U) 
					      | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[4U] 
						 >> 0xfU))));
    // ALWAYS at VX_decode.v:324
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_itype_immed 
	= ((0x4000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
	    ? ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
	        ? ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
		    ? 0xdeadbeefU : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
				      ? 0xdeadbeefU
				      : ((0x400U & 
					  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					  ? 0xdeadbeefU
					  : ((0x200U 
					      & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					      ? ((0x100U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						  ? 
						 ((0xfffff000U 
						   & (VL_NEGATE_I((IData)(
									  (1U 
									   & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									      >> 7U)))) 
						      << 0xcU)) 
						  | ((0x800U 
						      & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							 << 4U)) 
						     | ((0x400U 
							 & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							    >> 5U)) 
							| ((0x3f0U 
							    & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							       << 3U)) 
							   | (0xfU 
							      & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
								  << 0x10U) 
								 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
								    >> 0x10U)))))))
						  : 0xdeadbeefU)
					      : 0xdeadbeefU))))
	        : 0xdeadbeefU) : ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
				   ? ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
				       ? 0xdeadbeefU
				       : ((0x800U & 
					   vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					   ? 0xdeadbeefU
					   : ((0x400U 
					       & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					       ? 0xdeadbeefU
					       : ((0x200U 
						   & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						   ? 
						  ((0x100U 
						    & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						    ? 
						   ((0xfffff000U 
						     & (VL_NEGATE_I((IData)(
									    (1U 
									     & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										>> 7U)))) 
							<< 0xcU)) 
						    | ((0xfe0U 
							& (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							   << 4U)) 
						       | (0x1fU 
							  & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							      << 0x11U) 
							     | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
								>> 0xfU)))))
						    : 0xdeadbeefU)
						   : 0xdeadbeefU))))
				   : ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
				       ? ((0x800U & 
					   vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					   ? 0xdeadbeefU
					   : ((0x400U 
					       & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					       ? 0xdeadbeefU
					       : ((0x200U 
						   & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						   ? 
						  ((0x100U 
						    & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						    ? 
						   ((0xfffff000U 
						     & (VL_NEGATE_I((IData)(
									    (1U 
									     & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__alu_tempp) 
										>> 0xbU)))) 
							<< 0xcU)) 
						    | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__alu_tempp))
						    : 0xdeadbeefU)
						   : 0xdeadbeefU)))
				       : ((0x800U & 
					   vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					   ? 0xdeadbeefU
					   : ((0x400U 
					       & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					       ? 0xdeadbeefU
					       : ((0x200U 
						   & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						   ? 
						  ((0x100U 
						    & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						    ? 
						   ((0xfffff000U 
						     & (VL_NEGATE_I((IData)(
									    (1U 
									     & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										>> 7U)))) 
							<< 0xcU)) 
						    | (0xfffU 
						       & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							   << 4U) 
							  | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							     >> 0x1cU))))
						    : 0xdeadbeefU)
						   : 0xdeadbeefU))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_ebreak 
	= ((0x73U == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U)))) 
	   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jal_sys_jal) 
	      & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
	      [0U]));
    // ALWAYS at VX_decode.v:264
    if ((0x4000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
	if ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
	    if ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
		vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset 
		    = ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
		        ? 0xdeadbeefU : ((0x400U & 
					  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					  ? 0xdeadbeefU
					  : ((0x200U 
					      & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					      ? ((0x100U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						  ? 
						 (((0U 
						    == 
						    (7U 
						     & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							 << 0xcU) 
							| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							   >> 0x14U)))) 
						   & (2U 
						      > 
						      (0xfffU 
						       & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							   << 4U) 
							  | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							     >> 0x1cU)))))
						   ? 0xb0000000U
						   : 0xdeadbeefU)
						  : 0xdeadbeefU)
					      : 0xdeadbeefU)));
	    } else {
		if ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
		    if ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
			vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset 
			    = ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
			        ? ((0x100U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
				    ? ((0xffe00000U 
					& (VL_NEGATE_I((IData)(
							       (1U 
								& (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
								   >> 7U)))) 
					   << 0x15U)) 
				       | ((0x100000U 
					   & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					      << 0xdU)) 
					  | ((0xff000U 
					      & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						  << 0x18U) 
						 | (0xfff000U 
						    & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						       >> 8U)))) 
					     | ((0x800U 
						 & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						    >> 0x11U)) 
						| (0x7feU 
						   & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						       << 4U) 
						      | (0xeU 
							 & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							    >> 0x1cU))))))))
				    : 0xdeadbeefU) : 0xdeadbeefU);
		    } else {
			if ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
			    if ((0x100U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
				if (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs) 
				     | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jmprt))) {
				    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset = 0U;
				}
			    } else {
				vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset = 0xdeadbeefU;
			    }
			} else {
			    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset = 0xdeadbeefU;
			}
		    }
		} else {
		    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset 
			= ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
			    ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
			        ? ((0x100U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
				    ? ((0xfffff000U 
					& (VL_NEGATE_I((IData)(
							       (1U 
								& (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
								   >> 7U)))) 
					   << 0xcU)) 
				       | (0xfffU & 
					  ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					    << 4U) 
					   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					      >> 0x1cU))))
				    : 0xdeadbeefU) : 0xdeadbeefU)
			    : 0xdeadbeefU);
		}
	    }
	} else {
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset = 0xdeadbeefU;
	}
    } else {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset = 0xdeadbeefU;
    }
    // ALWAYS at VX_decode.v:264
    if ((0x4000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
	if ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
	    if ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
		vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal 
		    = ((~ (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			   >> 0xbU)) & ((~ (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					    >> 0xaU)) 
					& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					    >> 9U) 
					   & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 8U) 
					      & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jal_sys_jal) 
						 & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
						 [0U])))));
	    } else {
		if ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
		    if ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
			vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal 
			    = ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				>> 9U) & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					   >> 8U) & 
					  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
					  [0U]));
		    } else {
			if ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
			    if ((0x100U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
				if (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs) 
				     | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jmprt))) {
				    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal 
					= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
					[0U];
				}
			    } else {
				vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal = 0U;
			    }
			} else {
			    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal = 0U;
			}
		    }
		} else {
		    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal 
			= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			    >> 0xaU) & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					 >> 9U) & (
						   (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						    >> 8U) 
						   & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
						   [0U])));
		}
	    }
	} else {
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal = 0U;
	}
    } else {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal = 0U;
    }
    // ALWAYS at VX_decode.v:339
    if ((0x4000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
	if ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
	    if ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
		vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type = 0U;
	    } else {
		if ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
		    if ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
			vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type = 0U;
		    } else {
			if ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
			    if ((0x100U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
				if (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs) 
				     | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jmprt))) {
				    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type = 0U;
				}
			    } else {
				vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type = 0U;
			    }
			} else {
			    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type = 0U;
			}
		    }
		} else {
		    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type 
			= ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
			    ? 0U : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
				     ? ((0x100U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					 ? ((0x400000U 
					     & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					     ? ((0x200000U 
						 & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						 ? 
						((0x100000U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						  ? 6U
						  : 5U)
						 : 
						((0x100000U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						  ? 4U
						  : 3U))
					     : ((0x200000U 
						 & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						 ? 0U
						 : 
						((0x100000U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						  ? 2U
						  : 1U)))
					 : 0U) : 0U));
		}
	    }
	} else {
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type = 0U;
	}
    } else {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type = 0U;
    }
    vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs) 
	   | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jmprt));
    // ALWAYS at VX_decode.v:339
    if ((0x4000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
	if ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
	    if ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
		vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall = 0U;
	    } else {
		if ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
		    if ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
			vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall 
			    = ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				>> 9U) & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					   >> 8U) & 
					  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
					  [0U]));
		    } else {
			if ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
			    if ((0x100U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
				if (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs) 
				     | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jmprt))) {
				    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall 
					= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
					[0U];
				}
			    } else {
				vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall = 0U;
			    }
			} else {
			    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall = 0U;
			}
		    }
		} else {
		    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall 
			= ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
			    ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				>> 9U) & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					   >> 8U) & 
					  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
					  [0U])) : 
			   ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			     >> 9U) & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					>> 8U) & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
				       [0U])));
		}
	    }
	} else {
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall = 0U;
	}
    } else {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall = 0U;
    }
    vlTOPp->out_cache_driver_in_address[3U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_address
	[3U];
    vlTOPp->out_cache_driver_in_address[2U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_address
	[2U];
    vlTOPp->out_cache_driver_in_address[1U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_address
	[1U];
    vlTOPp->out_cache_driver_in_address[0U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_address
	[0U];
    vlTOPp->out_cache_driver_in_data[3U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_data
	[3U];
    vlTOPp->out_cache_driver_in_data[2U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_data
	[2U];
    vlTOPp->out_cache_driver_in_data[1U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_data
	[1U];
    vlTOPp->out_cache_driver_in_data[0U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_data
	[0U];
    vlTOPp->out_cache_driver_in_valid[3U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_valid
	[3U];
    vlTOPp->out_cache_driver_in_valid[2U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_valid
	[2U];
    vlTOPp->out_cache_driver_in_valid[1U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_valid
	[1U];
    vlTOPp->out_cache_driver_in_valid[0U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_valid
	[0U];
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd 
	= ((((((0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			  << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				    >> 0x17U))) == 
	       (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
			  << 9U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
				    >> 0x17U)))) & 
	      (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					  >> 0x17U))))) 
	     & (0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
			      << 0xbU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					  >> 0x15U))))) 
	    & (~ (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd))) 
	   & ((0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
				     >> 4U))) == (0xfU 
						  & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd 
	= ((((((0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			  << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				    >> 0x1cU))) == 
	       (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
			  << 9U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
				    >> 0x17U)))) & 
	      (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					  >> 0x1cU))))) 
	     & (0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
			      << 0xbU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					  >> 0x15U))))) 
	    & (~ (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd))) 
	   & ((0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
				     >> 4U))) == (0xfU 
						  & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])));
    vlTOPp->out_ebreak = ((0U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					   << 0x1cU) 
					  | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))) 
			  & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_ebreak));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__add_warp 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_wspawn) 
	   & (~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_ebreak)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_ebreak) 
	   & (~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_wspawn)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_final_alu 
	= ((0x63U == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U))))
	    ? ((5U > (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type))
	        ? 1U : 0xaU) : ((0x37U == (0x7fU & 
					   ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 8U))))
				 ? 0xbU : ((0x17U == 
					    (0x7fU 
					     & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						 << 0x18U) 
						| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						   >> 8U))))
					    ? 0xcU : 
					   ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_csr)
					     ? ((1U 
						 == 
						 (3U 
						  & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						      << 0xcU) 
						     | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							>> 0x14U))))
						 ? 0xdU
						 : 
						((2U 
						  == 
						  (3U 
						   & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						       << 0xcU) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							 >> 0x14U))))
						  ? 0xeU
						  : 0xfU))
					     : (((0x23U 
						  == 
						  (0x7fU 
						   & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						       << 0x18U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							 >> 8U)))) 
						 | (3U 
						    == 
						    (0x7fU 
						     & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							 << 0x18U) 
							| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							   >> 8U)))))
						 ? 0U
						 : 
						((0x400000U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						  ? 
						 ((0x200000U 
						   & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						   ? 
						  ((0x100000U 
						    & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						    ? 9U
						    : 8U)
						   : 
						  ((0x100000U 
						    & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						    ? 
						   ((0U 
						     == 
						     (0x7fU 
						      & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							 >> 1U)))
						     ? 6U
						     : 7U)
						    : 5U))
						  : 
						 ((0x200000U 
						   & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						   ? 
						  ((0x100000U 
						    & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						    ? 4U
						    : 3U)
						   : 
						  ((0x100000U 
						    & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						    ? 2U
						    : 
						   ((0x13U 
						     == 
						     (0x7fU 
						      & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							  << 0x18U) 
							 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							    >> 8U))))
						     ? 0U
						     : 
						    ((0U 
						      == 
						      (0x7fU 
						       & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							  >> 1U)))
						      ? 0U
						      : 1U))))))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_change_mask 
	= ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask) 
	   & (0U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					   >> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_change_mask 
	= ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask) 
	   & (1U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					   >> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_change_mask 
	= ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask) 
	   & (2U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					   >> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_change_mask 
	= ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask) 
	   & (3U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					   >> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_change_mask 
	= ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask) 
	   & (4U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					   >> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_change_mask 
	= ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask) 
	   & (5U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					   >> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_change_mask 
	= ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask) 
	   & (6U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					   >> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_change_mask 
	= ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask) 
	   & (7U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					   >> 4U)))));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd 
	= (((((((0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			   << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				     >> 0x17U))) == 
		(0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					>> 0xaU)))) 
	       & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				   << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 0x17U))))) 
	      & (0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			       << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 8U))))) 
	     & ((0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]) 
		== (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			     << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					  >> 4U))))) 
	    & (~ (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd))) 
	   & (~ (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)));
    vlTOPp->Vortex__DOT__forwarding_fwd_stall = ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd) 
						   | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd)) 
						  & (2U 
						     == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))) 
						 | (((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd) 
						     | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)) 
						    & (2U 
						       == 
						       (3U 
							& ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
							    << 0xbU) 
							   | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
							      >> 0x15U))))));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd 
	= (((((((0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			   << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				     >> 0x1cU))) == 
		(0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					>> 0xaU)))) 
	       & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				   << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 0x1cU))))) 
	      & (0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			       << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 8U))))) 
	     & (~ (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd))) 
	    & (~ (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd))) 
	   & ((0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]) 
	      == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			   << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					>> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall 
	= (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall) 
	    | (IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall)) 
	   | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall));
}

VL_INLINE_OPT void VVortex::_sequent__TOP__2(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_sequent__TOP__2\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // ALWAYS at VX_fetch.v:78
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_count 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__new_warp_count;
    // ALWAYS at VX_warp.v:71
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__real_PC 
	= ((IData)(vlTOPp->reset) ? 0U : ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_stall)
					   ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__temp_PC
					   : ((IData)(4U) 
					      + vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__temp_PC)));
    // ALWAYS at VX_warp.v:71
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__real_PC 
	= ((IData)(vlTOPp->reset) ? 0U : (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_wspawn) 
					   & (7U == 
					      ((IData)(1U) 
					       + (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_state))))
					   ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U]
					   : ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_stall)
					       ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__temp_PC
					       : ((IData)(4U) 
						  + vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__temp_PC))));
    // ALWAYS at VX_warp.v:71
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__real_PC 
	= ((IData)(vlTOPp->reset) ? 0U : (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_wspawn) 
					   & (6U == 
					      ((IData)(1U) 
					       + (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_state))))
					   ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U]
					   : ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_stall)
					       ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__temp_PC
					       : ((IData)(4U) 
						  + vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__temp_PC))));
    // ALWAYS at VX_warp.v:71
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__real_PC 
	= ((IData)(vlTOPp->reset) ? 0U : (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_wspawn) 
					   & (5U == 
					      ((IData)(1U) 
					       + (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_state))))
					   ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U]
					   : ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_stall)
					       ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__temp_PC
					       : ((IData)(4U) 
						  + vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__temp_PC))));
    // ALWAYS at VX_warp.v:71
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__real_PC 
	= ((IData)(vlTOPp->reset) ? 0U : (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_wspawn) 
					   & (4U == 
					      ((IData)(1U) 
					       + (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_state))))
					   ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U]
					   : ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_stall)
					       ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__temp_PC
					       : ((IData)(4U) 
						  + vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__temp_PC))));
    // ALWAYS at VX_warp.v:71
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__real_PC 
	= ((IData)(vlTOPp->reset) ? 0U : (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_wspawn) 
					   & (3U == 
					      ((IData)(1U) 
					       + (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_state))))
					   ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U]
					   : ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_stall)
					       ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__temp_PC
					       : ((IData)(4U) 
						  + vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__temp_PC))));
    // ALWAYS at VX_warp.v:71
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__real_PC 
	= ((IData)(vlTOPp->reset) ? 0U : (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_wspawn) 
					   & (2U == 
					      ((IData)(1U) 
					       + (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_state))))
					   ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U]
					   : ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_stall)
					       ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__temp_PC
					       : ((IData)(4U) 
						  + vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__temp_PC))));
    // ALWAYS at VX_warp.v:71
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__real_PC 
	= ((IData)(vlTOPp->reset) ? 0U : (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_wspawn) 
					   & (1U == 
					      ((IData)(1U) 
					       + (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_state))))
					   ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U]
					   : ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_stall)
					       ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__temp_PC
					       : ((IData)(4U) 
						  + vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__temp_PC))));
    // ALWAYS at VX_generic_register.v:20
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U] 
	= (((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall) 
				      | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall)))
	    ? 0U : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0U]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[1U] 
	= (((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall) 
				      | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall)))
	    ? 0U : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[1U]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
	= (((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall) 
				      | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall)))
	    ? 0U : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[2U]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
	= (((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall) 
				      | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall)))
	    ? 0U : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[3U]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
	= (((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall) 
				      | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall)))
	    ? 0U : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[4U]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
	= (((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall) 
				      | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall)))
	    ? 0U : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[5U]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
	= (((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall) 
				      | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall)))
	    ? 0U : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[6U]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[7U] 
	= (((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall) 
				      | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall)))
	    ? 0U : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[7U]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[8U] 
	= (((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall) 
				      | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall)))
	    ? 0U : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[8U]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
	= (((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall) 
				      | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall)))
	    ? 0U : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[9U]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
	= (((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall) 
				      | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall)))
	    ? 0U : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xaU]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
	= (((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall) 
				      | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall)))
	    ? 0U : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xbU]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
	= (((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall) 
				      | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall)))
	    ? 0U : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xcU]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
	= (((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall) 
				      | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall)))
	    ? 0U : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xdU]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
	= (((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall) 
				      | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall)))
	    ? 0U : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xeU]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xfU] 
	= (((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall) 
				      | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall)))
	    ? 0U : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xfU]);
    // ALWAYS at VX_fetch.v:78
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_state 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__new_warp_state;
    vlTOPp->Vortex__DOT__execute_branch_stall = (1U 
						 & ((0U 
						     != 
						     (7U 
						      & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
							  << 3U) 
							 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
							    >> 0x1dU)))) 
						    | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
						       >> 8U)));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[0U] 
	= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[1U] 
	= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[2U] 
	= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[3U] 
	= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2 
	= ((0x40U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	    ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
		<< 0x1aU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
			     >> 6U)) : ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
					   >> 0xeU)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2 
	= ((0x40U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	    ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
		<< 0x1aU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
			     >> 6U)) : ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[7U] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
					   >> 0xeU)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2 
	= ((0x40U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	    ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
		<< 0x1aU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
			     >> 6U)) : ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[8U] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[7U] 
					   >> 0xeU)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2 
	= ((0x40U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	    ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
		<< 0x1aU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
			     >> 6U)) : ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[8U] 
					   >> 0xeU)));
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb 
	= (3U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
		  << 0x19U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
			       >> 7U)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__mult_signed_result 
	= VL_MULS_QQQ(64,64,64, VL_EXTENDS_QI(64,32, 
					      ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						<< 0x12U) 
					       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						  >> 0xeU))), 
		      VL_EXTENDS_QI(64,32, vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__mult_signed_result 
	= VL_MULS_QQQ(64,64,64, VL_EXTENDS_QI(64,32, 
					      ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						<< 0x12U) 
					       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						  >> 0xeU))), 
		      VL_EXTENDS_QI(64,32, vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__mult_signed_result 
	= VL_MULS_QQQ(64,64,64, VL_EXTENDS_QI(64,32, 
					      ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						<< 0x12U) 
					       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						  >> 0xeU))), 
		      VL_EXTENDS_QI(64,32, vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__mult_signed_result 
	= VL_MULS_QQQ(64,64,64, VL_EXTENDS_QI(64,32, 
					      ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						<< 0x12U) 
					       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						  >> 0xeU))), 
		      VL_EXTENDS_QI(64,32, vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2));
}

VL_INLINE_OPT void VVortex::_combo__TOP__3(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_combo__TOP__3\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlSymsp->TOP__Vortex__DOT__VX_dcache_rsp.in_cache_driver_out_data[3U] 
	= vlTOPp->in_cache_driver_out_data[3U];
    vlSymsp->TOP__Vortex__DOT__VX_dcache_rsp.in_cache_driver_out_data[2U] 
	= vlTOPp->in_cache_driver_out_data[2U];
    vlSymsp->TOP__Vortex__DOT__VX_dcache_rsp.in_cache_driver_out_data[1U] 
	= vlTOPp->in_cache_driver_out_data[1U];
    vlSymsp->TOP__Vortex__DOT__VX_dcache_rsp.in_cache_driver_out_data[0U] 
	= vlTOPp->in_cache_driver_out_data[0U];
}

VL_INLINE_OPT void VVortex::_sequent__TOP__4(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_sequent__TOP__4\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    // Begin mtask footprint  all: 
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14,0,0);
    VL_SIG8(__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,4,0);
    VL_SIG8(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,6,0);
    VL_SIG8(__Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,7,0);
    VL_SIG8(__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15,0,0);
    // Body
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 0U;
    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 0U;
    // ALWAYS at VX_csr_handler.v:50
    if ((8U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xeU])) {
	vlTOPp->Vortex__DOT__vx_csr_handler__DOT____Vlvbound1 
	    = (0xfffU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xeU] 
			  << 0x1dU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
				       >> 3U)));
	if (VL_LIKELY((0x300bU >= (0x3fffU & ((IData)(0xcU) 
					      * (0xfffU 
						 & (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xeU] 
						    >> 4U))))))) {
	    VL_ASSIGNSEL_WIII(12,(0x3fffU & ((IData)(0xcU) 
					     * (0xfffU 
						& (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xeU] 
						   >> 4U)))), vlTOPp->Vortex__DOT__vx_csr_handler__DOT__csr, vlTOPp->Vortex__DOT__vx_csr_handler__DOT____Vlvbound1);
	}
    }
    // ALWAYS at VX_csr_handler.v:41
    vlTOPp->Vortex__DOT__vx_csr_handler__DOT__decode_csr_address 
	= vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_frE_to_bckE_req.csr_address;
    // ALWAYS at VX_warp.v:41
    if (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp) 
	 & (7U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			    << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					 >> 4U)))))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid_zero;
    } else {
	if (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_change_mask) {
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid 
		= vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask;
	}
    }
    // ALWAYS at VX_warp.v:41
    if (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp) 
	 & (6U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			    << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					 >> 4U)))))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid_zero;
    } else {
	if (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_change_mask) {
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid 
		= vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask;
	}
    }
    // ALWAYS at VX_warp.v:41
    if (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp) 
	 & (5U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			    << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					 >> 4U)))))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid_zero;
    } else {
	if (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_change_mask) {
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid 
		= vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask;
	}
    }
    // ALWAYS at VX_warp.v:41
    if (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp) 
	 & (4U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			    << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					 >> 4U)))))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid_zero;
    } else {
	if (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_change_mask) {
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid 
		= vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask;
	}
    }
    // ALWAYS at VX_warp.v:41
    if (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp) 
	 & (3U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			    << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					 >> 4U)))))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid_zero;
    } else {
	if (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_change_mask) {
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid 
		= vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask;
	}
    }
    // ALWAYS at VX_warp.v:41
    if (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp) 
	 & (2U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			    << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					 >> 4U)))))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid_zero;
    } else {
	if (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_change_mask) {
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid 
		= vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask;
	}
    }
    // ALWAYS at VX_warp.v:41
    if (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp) 
	 & (1U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			    << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					 >> 4U)))))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid_zero;
    } else {
	if (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_change_mask) {
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid 
		= vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask;
	}
    }
    // ALWAYS at VX_warp.v:41
    if (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp) 
	 & (0U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			    << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					 >> 4U)))))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid_zero;
    } else {
	if (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_change_mask) {
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid 
		= vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask;
	}
    }
    // ALWAYS at VX_csr_handler.v:41
    vlTOPp->Vortex__DOT__vx_csr_handler__DOT__cycle 
	= (VL_ULL(1) + vlTOPp->Vortex__DOT__vx_csr_handler__DOT__cycle);
    // ALWAYS at VX_csr_handler.v:41
    if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	vlTOPp->Vortex__DOT__vx_csr_handler__DOT__instret 
	    = (VL_ULL(1) + vlTOPp->Vortex__DOT__vx_csr_handler__DOT__instret);
    }
    // ALWAYS at byte_enabled_simple_dual_port_ram.v:16
    if (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__write_enable) {
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 0U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 8U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 0x10U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 0x18U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 0x20U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 0x28U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 0x30U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 0x38U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 0x40U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 0x48U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 0x50U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 0x58U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 0x60U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 8U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 0x68U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 0x10U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 0x70U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 0x18U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 0x78U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
    }
    // ALWAYS at byte_enabled_simple_dual_port_ram.v:16
    if (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__write_enable) {
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 0U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 8U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 0x10U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 0x18U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 0x20U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 0x28U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 0x30U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 0x38U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 0x40U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 0x48U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 0x50U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 0x58U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 0x60U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 8U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 0x68U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 0x10U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 0x70U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 0x18U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 0x78U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
    }
    // ALWAYS at byte_enabled_simple_dual_port_ram.v:16
    if (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__write_enable) {
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 0U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 8U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 0x10U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 0x18U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 0x20U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 0x28U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 0x30U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 0x38U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 0x40U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 0x48U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 0x50U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 0x58U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 0x60U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 8U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 0x68U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 0x10U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 0x70U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 0x18U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 0x78U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
    }
    // ALWAYS at byte_enabled_simple_dual_port_ram.v:16
    if (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__write_enable) {
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 0U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 8U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 0x10U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 0x18U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 0x20U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 0x28U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 0x30U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 0x38U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 0x40U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 0x48U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 0x50U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 0x58U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 0x60U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 8U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 0x68U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 0x10U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 0x70U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 0x18U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 0x78U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
    }
    // ALWAYS at byte_enabled_simple_dual_port_ram.v:16
    if (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__write_enable) {
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 0U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 8U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 0x10U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 0x18U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 0x20U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 0x28U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 0x30U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 0x38U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 0x40U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 0x48U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 0x50U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 0x58U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 0x60U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 8U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 0x68U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 0x10U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 0x70U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 0x18U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 0x78U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
    }
    // ALWAYS at byte_enabled_simple_dual_port_ram.v:16
    if (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__write_enable) {
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 0U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 8U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 0x10U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 0x18U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 0x20U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 0x28U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 0x30U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 0x38U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 0x40U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 0x48U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 0x50U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 0x58U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 0x60U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 8U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 0x68U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 0x10U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 0x70U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 0x18U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 0x78U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
    }
    // ALWAYS at byte_enabled_simple_dual_port_ram.v:16
    if (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__write_enable) {
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 0U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 8U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 0x10U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 0x18U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 0x20U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 0x28U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 0x30U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 0x38U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 0x40U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 0x48U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 0x50U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 0x58U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 0x60U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 8U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 0x68U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 0x10U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 0x70U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 0x18U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 0x78U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
    }
    // ALWAYS at byte_enabled_simple_dual_port_ram.v:16
    if (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__write_enable) {
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 = 0U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 = 8U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 = 0x10U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x10U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 = 0x18U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 = 0x20U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 = 0x28U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 = 0x30U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x20U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 = 0x38U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 = 0x40U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 0x18U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
					  >> 8U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 = 0x48U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 0x10U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
					  >> 0x10U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 = 0x50U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x40U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 
		= (0xffU & ((vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			     << 8U) | (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
				       >> 0x18U)));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 = 0x58U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 
		= (0xffU & vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U]);
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 = 0x60U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 8U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 = 0x68U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 0x10U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 = 0x70U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
	if ((0x80U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) {
	    __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 
		= (0xffU & (vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
			    >> 0x18U));
	    __Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 1U;
	    __Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 = 0x78U;
	    __Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15 
		= (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 0xaU)));
	}
    }
    // ALWAYSPOST at byte_enabled_simple_dual_port_ram.v:20
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15);
    }
    // ALWAYSPOST at byte_enabled_simple_dual_port_ram.v:20
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15);
    }
    // ALWAYSPOST at byte_enabled_simple_dual_port_ram.v:20
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15);
    }
    // ALWAYSPOST at byte_enabled_simple_dual_port_ram.v:20
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15);
    }
    // ALWAYSPOST at byte_enabled_simple_dual_port_ram.v:20
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15);
    }
    // ALWAYSPOST at byte_enabled_simple_dual_port_ram.v:20
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15);
    }
    // ALWAYSPOST at byte_enabled_simple_dual_port_ram.v:20
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15);
    }
    // ALWAYSPOST at byte_enabled_simple_dual_port_ram.v:20
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v0);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v1);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v2);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v3);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v4);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v5);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v6);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v7);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v8);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v9);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v10);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v11);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v12);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v13);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v14);
    }
    if (__Vdlyvset__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15) {
	VL_ASSIGNSEL_WIII(8,(IData)(__Vdlyvlsb__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15), 
			  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
			  [__Vdlyvdim0__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15], __Vdlyvval__Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR__v15);
    }
}

VL_INLINE_OPT void VVortex::_sequent__TOP__5(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_sequent__TOP__5\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // ALWAYS at VX_csr_handler.v:56
    vlTOPp->Vortex__DOT__vx_csr_handler__DOT__data_read 
	= (0xfffU & ((0x300bU >= (0x3fffU & ((IData)(0xcU) 
					     * (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__decode_csr_address))))
		      ? (((0U == (0x1fU & ((IData)(0xcU) 
					   * (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__decode_csr_address))))
			   ? 0U : (vlTOPp->Vortex__DOT__vx_csr_handler__DOT__csr[
				   ((IData)(1U) + (0x1ffU 
						   & (((IData)(0xcU) 
						       * (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__decode_csr_address)) 
						      >> 5U)))] 
				   << ((IData)(0x20U) 
				       - (0x1fU & ((IData)(0xcU) 
						   * (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__decode_csr_address)))))) 
			 | (vlTOPp->Vortex__DOT__vx_csr_handler__DOT__csr[
			    (0x1ffU & (((IData)(0xcU) 
					* (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__decode_csr_address)) 
				       >> 5U))] >> 
			    (0x1fU & ((IData)(0xcU) 
				      * (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__decode_csr_address)))))
		      : 0U));
}

void VVortex::_initial__TOP__6(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_initial__TOP__6\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // INITIAL at VX_warp.v:30
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__real_PC = 0U;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid 
	= (0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid 
	= (0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid 
	= (7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid_zero 
	= (7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid 
	= (1U | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid_zero));
    // INITIAL at VX_warp.v:30
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__real_PC = 0U;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid 
	= (0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid 
	= (0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid 
	= (7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid_zero 
	= (7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid 
	= (1U | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid_zero));
    // INITIAL at VX_warp.v:30
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__real_PC = 0U;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid 
	= (0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid 
	= (0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid 
	= (7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid_zero 
	= (7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid 
	= (1U | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid_zero));
    // INITIAL at VX_warp.v:30
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__real_PC = 0U;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid 
	= (0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid 
	= (0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid 
	= (7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid_zero 
	= (7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid 
	= (1U | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid_zero));
    // INITIAL at VX_warp.v:30
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__real_PC = 0U;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid 
	= (0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid 
	= (0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid 
	= (7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid_zero 
	= (7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid 
	= (1U | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid_zero));
    // INITIAL at VX_warp.v:30
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__real_PC = 0U;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid 
	= (0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid 
	= (0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid 
	= (7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid_zero 
	= (7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid 
	= (1U | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid_zero));
    // INITIAL at VX_warp.v:30
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__real_PC = 0U;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid 
	= (0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid 
	= (0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid 
	= (7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid_zero 
	= (7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid 
	= (1U | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid_zero));
    // INITIAL at VX_warp.v:30
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__real_PC = 0U;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid 
	= (0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid 
	= (0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid 
	= (7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid_zero 
	= (7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid_zero));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid 
	= (1U | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid_zero 
	= (0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid_zero));
    // INITIAL at VX_csr_handler.v:34
    vlTOPp->Vortex__DOT__vx_csr_handler__DOT__cycle = VL_ULL(0);
    vlTOPp->Vortex__DOT__vx_csr_handler__DOT__instret = VL_ULL(0);
    vlTOPp->Vortex__DOT__vx_csr_handler__DOT__decode_csr_address = 0U;
    // INITIAL at VX_fetch.v:50
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num = 0U;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_state = 0U;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_count = 1U;
}

VL_INLINE_OPT void VVortex::_sequent__TOP__7(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_sequent__TOP__7\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // ALWAYS at VX_generic_register.v:20
    if (vlTOPp->reset) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] = 0U;
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] = 0U;
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] = 0U;
    } else {
	if ((1U & (~ (IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall)))) {
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
		= ((0xffffff00U & ((IData)((((QData)((IData)(
							     ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall)
							       ? 0U
							       : vlTOPp->icache_response_instruction))) 
					     << 0x20U) 
					    | (QData)((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var)))) 
				   << 8U)) | (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num) 
					       << 4U) 
					      | (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd.valid)));
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
		= ((0xffU & ((IData)((((QData)((IData)(
						       ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall)
							 ? 0U
							 : vlTOPp->icache_response_instruction))) 
				       << 0x20U) | (QData)((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var)))) 
			     >> 0x18U)) | (0xffffff00U 
					   & ((IData)(
						      ((((QData)((IData)(
									 ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall)
									   ? 0U
									   : vlTOPp->icache_response_instruction))) 
							 << 0x20U) 
							| (QData)((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var))) 
						       >> 0x20U)) 
					      << 8U)));
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		= (0xffU & ((IData)(((((QData)((IData)(
						       ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall)
							 ? 0U
							 : vlTOPp->icache_response_instruction))) 
				       << 0x20U) | (QData)((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var))) 
				     >> 0x20U)) >> 0x18U));
	}
    }
    // ALWAYS at VX_generic_register.v:20
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	= ((IData)(vlTOPp->reset) ? 0U : (IData)((((QData)((IData)(
								   (0x1fU 
								    & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
									<< 9U) 
								       | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
									  >> 0x17U))))) 
						   << 0x2aU) 
						  | (((QData)((IData)(
								      (3U 
								       & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
									   << 0xbU) 
									  | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
									     >> 0x15U))))) 
						      << 0x28U) 
						     | (((QData)((IData)(
									 ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[3U] 
									   << 0x15U) 
									  | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[2U] 
									     >> 0xbU)))) 
							 << 8U) 
							| (QData)((IData)(
									  (0xffU 
									   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U]))))))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
	= ((IData)(vlTOPp->reset) ? 0U : ((0xffff8000U 
					   & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[0U] 
					      << 0xfU)) 
					  | (IData)(
						    ((((QData)((IData)(
								       (0x1fU 
									& ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
									    << 9U) 
									   | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
									      >> 0x17U))))) 
						       << 0x2aU) 
						      | (((QData)((IData)(
									  (3U 
									   & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
									       << 0xbU) 
									      | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
										>> 0x15U))))) 
							  << 0x28U) 
							 | (((QData)((IData)(
									     ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[3U] 
									       << 0x15U) 
									      | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[2U] 
										>> 0xbU)))) 
							     << 8U) 
							    | (QData)((IData)(
									      (0xffU 
									       & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))))) 
						     >> 0x20U))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
	= ((IData)(vlTOPp->reset) ? 0U : ((0x7fffU 
					   & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[0U] 
					      >> 0x11U)) 
					  | (0xffff8000U 
					     & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[1U] 
						<< 0xfU))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[3U] 
	= ((IData)(vlTOPp->reset) ? 0U : ((0x7fffU 
					   & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[1U] 
					      >> 0x11U)) 
					  | (0xffff8000U 
					     & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[2U] 
						<< 0xfU))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[4U] 
	= ((IData)(vlTOPp->reset) ? 0U : ((0x7fffU 
					   & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[2U] 
					      >> 0x11U)) 
					  | (0xffff8000U 
					     & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[3U] 
						<< 0xfU))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[5U] 
	= ((IData)(vlTOPp->reset) ? 0U : ((0x7fffU 
					   & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[3U] 
					      >> 0x11U)) 
					  | (0xffff8000U 
					     & (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
						<< 0xdU))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[6U] 
	= ((IData)(vlTOPp->reset) ? 0U : ((0x7fffU 
					   & ((0x6000U 
					       & (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
						  << 0xdU)) 
					      | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
						 >> 0x13U))) 
					  | (0xffff8000U 
					     & (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
						<< 0xdU))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[7U] 
	= ((IData)(vlTOPp->reset) ? 0U : ((0x7fffU 
					   & ((0x6000U 
					       & (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xaU] 
						  << 0xdU)) 
					      | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
						 >> 0x13U))) 
					  | (0xffff8000U 
					     & (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xaU] 
						<< 0xdU))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[8U] 
	= ((IData)(vlTOPp->reset) ? 0U : ((0x7fffU 
					   & ((0x6000U 
					       & (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xbU] 
						  << 0xdU)) 
					      | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xaU] 
						 >> 0x13U))) 
					  | (0xffff8000U 
					     & (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xbU] 
						<< 0xdU))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[9U] 
	= ((IData)(vlTOPp->reset) ? 0U : (0x7fffU & 
					  ((0x6000U 
					    & (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
					       << 0xdU)) 
					   | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xbU] 
					      >> 0x13U))));
    // ALWAYS at VX_fetch.v:78
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__new_warp_num;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_itype 
	= ((0x13U == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U)))) 
	   | (3U == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			       << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					    >> 8U)))));
    vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_frE_to_bckE_req.csr_address 
	= (0xfffU & (((0U != (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				     << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						 >> 0x14U)))) 
		      & (2U <= (0xfffU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					   << 4U) | 
					  (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					   >> 0x1cU)))))
		      ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			  << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				    >> 0x1cU)) : 0x55U));
    // ALWAYS at VX_decode.v:399
    vlTOPp->__Vtableidx1 = (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				   << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 0x14U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu 
	= vlTOPp->__Vtable1_Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu
	[vlTOPp->__Vtableidx1];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__alu_tempp 
	= (0xfffU & (((1U == (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				     << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						 >> 0x14U)))) 
		      | (5U == (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				       << 0xcU) | (
						   vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						   >> 0x14U)))))
		      ? (0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				   << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 0x1cU)))
		      : ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			  << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				    >> 0x1cU))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data[0U] 
	= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data[1U] 
	= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data[2U] 
	= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data[3U] 
	= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_csr 
	= ((0x73U == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U)))) 
	   & (0U != (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			    << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					>> 0x14U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[1U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[2U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[3U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[4U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[5U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[6U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[7U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[8U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[9U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0xaU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0xbU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0xcU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0xdU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0xeU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0xfU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x10U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x11U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x12U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x13U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x14U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x15U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x16U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x17U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x18U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x19U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x1aU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x1bU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x1cU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x1dU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x1eU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[0x1fU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x17U)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_wspawn 
	= ((0x6bU == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U)))) 
	   & (0U == (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			    << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					>> 0x14U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jal_sys_jal 
	= ((0U == (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			  << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				      >> 0x14U)))) 
	   & (2U > (0xfffU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			       << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					 >> 0x1cU)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid[0U] 
	= (1U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid[1U] 
	= (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
		 >> 1U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid[2U] 
	= (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
		 >> 2U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid[3U] 
	= (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
		 >> 3U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jmprt 
	= ((0x6bU == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U)))) 
	   & (4U == (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			    << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					>> 0x14U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs 
	= ((0x6bU == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U)))) 
	   & (6U == (7U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			    << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					>> 0x14U)))));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd 
	= (((((0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			 << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				   >> 0x17U))) == (0x1fU 
						   & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
						       << 8U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
							 >> 0x18U)))) 
	     & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				 << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					   >> 0x17U))))) 
	    & (0U != (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))) 
	   & ((0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
				     >> 4U))) == (0xfU 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U])));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[1U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[2U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[3U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[4U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[5U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[6U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[7U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[8U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[9U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0xaU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0xbU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0xcU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0xdU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0xeU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0xfU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x10U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x11U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x12U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x13U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x14U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x15U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x16U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x17U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x18U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x19U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x1aU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x1bU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x1cU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][0U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x1dU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][1U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x1eU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][2U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[0x1fU] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR
	[(0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
		    << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      >> 0x1cU)))][3U];
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd 
	= (((((0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			 << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				   >> 0x1cU))) == (0x1fU 
						   & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
						       << 8U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
							 >> 0x18U)))) 
	     & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				 << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					   >> 0x1cU))))) 
	    & (0U != (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))) 
	   & ((0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
				     >> 4U))) == (0xfU 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U])));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__write_enable 
	= ((0U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) 
	   & ((0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 8U)))) 
	      & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				  << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 0xaU))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__write_enable 
	= ((1U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) 
	   & ((0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 8U)))) 
	      & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				  << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 0xaU))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__write_enable 
	= ((2U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) 
	   & ((0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 8U)))) 
	      & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				  << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 0xaU))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__write_enable 
	= ((3U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) 
	   & ((0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 8U)))) 
	      & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				  << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 0xaU))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__write_enable 
	= ((4U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) 
	   & ((0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 8U)))) 
	      & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				  << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 0xaU))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__write_enable 
	= ((5U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) 
	   & ((0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 8U)))) 
	      & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				  << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 0xaU))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__write_enable 
	= ((6U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) 
	   & ((0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 8U)))) 
	      & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				  << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 0xaU))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__write_enable 
	= ((7U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U])) 
	   & ((0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			     << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					  >> 8U)))) 
	      & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				  << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 0xaU))))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data[0U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data[1U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data[2U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data[3U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[0U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[1U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[2U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
			 >> 8U));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[3U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
	    << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
			 >> 8U));
    // ALWAYS at VX_generic_register.v:20
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U] 
	= ((IData)(vlTOPp->reset) ? 0U : vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0U]);
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[1U] 
	= ((IData)(vlTOPp->reset) ? 0U : vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[1U]);
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[2U] 
	= ((IData)(vlTOPp->reset) ? 0U : vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[2U]);
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[3U] 
	= ((IData)(vlTOPp->reset) ? 0U : vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[3U]);
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[4U] 
	= ((IData)(vlTOPp->reset) ? 0U : vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[4U]);
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[5U] 
	= ((IData)(vlTOPp->reset) ? 0U : vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[5U]);
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[6U] 
	= ((IData)(vlTOPp->reset) ? 0U : vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[6U]);
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
	= ((IData)(vlTOPp->reset) ? 0U : vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[7U]);
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
	= ((IData)(vlTOPp->reset) ? 0U : vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[8U]);
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
	= ((IData)(vlTOPp->reset) ? 0U : vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[9U]);
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xaU] 
	= ((IData)(vlTOPp->reset) ? 0U : vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0xaU]);
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xbU] 
	= ((IData)(vlTOPp->reset) ? 0U : vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0xbU]);
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
	= ((IData)(vlTOPp->reset) ? 0U : vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0xcU]);
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
	= ((IData)(vlTOPp->reset) ? 0U : vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0xdU]);
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xeU] 
	= ((IData)(vlTOPp->reset) ? 0U : vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0xeU]);
    // ALWAYS at VX_decode.v:324
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_itype_immed 
	= ((0x4000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
	    ? ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
	        ? ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
		    ? 0xdeadbeefU : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
				      ? 0xdeadbeefU
				      : ((0x400U & 
					  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					  ? 0xdeadbeefU
					  : ((0x200U 
					      & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					      ? ((0x100U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						  ? 
						 ((0xfffff000U 
						   & (VL_NEGATE_I((IData)(
									  (1U 
									   & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									      >> 7U)))) 
						      << 0xcU)) 
						  | ((0x800U 
						      & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							 << 4U)) 
						     | ((0x400U 
							 & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							    >> 5U)) 
							| ((0x3f0U 
							    & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							       << 3U)) 
							   | (0xfU 
							      & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
								  << 0x10U) 
								 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
								    >> 0x10U)))))))
						  : 0xdeadbeefU)
					      : 0xdeadbeefU))))
	        : 0xdeadbeefU) : ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
				   ? ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
				       ? 0xdeadbeefU
				       : ((0x800U & 
					   vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					   ? 0xdeadbeefU
					   : ((0x400U 
					       & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					       ? 0xdeadbeefU
					       : ((0x200U 
						   & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						   ? 
						  ((0x100U 
						    & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						    ? 
						   ((0xfffff000U 
						     & (VL_NEGATE_I((IData)(
									    (1U 
									     & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										>> 7U)))) 
							<< 0xcU)) 
						    | ((0xfe0U 
							& (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							   << 4U)) 
						       | (0x1fU 
							  & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							      << 0x11U) 
							     | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
								>> 0xfU)))))
						    : 0xdeadbeefU)
						   : 0xdeadbeefU))))
				   : ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
				       ? ((0x800U & 
					   vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					   ? 0xdeadbeefU
					   : ((0x400U 
					       & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					       ? 0xdeadbeefU
					       : ((0x200U 
						   & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						   ? 
						  ((0x100U 
						    & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						    ? 
						   ((0xfffff000U 
						     & (VL_NEGATE_I((IData)(
									    (1U 
									     & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__alu_tempp) 
										>> 0xbU)))) 
							<< 0xcU)) 
						    | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__alu_tempp))
						    : 0xdeadbeefU)
						   : 0xdeadbeefU)))
				       : ((0x800U & 
					   vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					   ? 0xdeadbeefU
					   : ((0x400U 
					       & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					       ? 0xdeadbeefU
					       : ((0x200U 
						   & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						   ? 
						  ((0x100U 
						    & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						    ? 
						   ((0xfffff000U 
						     & (VL_NEGATE_I((IData)(
									    (1U 
									     & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										>> 7U)))) 
							<< 0xcU)) 
						    | (0xfffU 
						       & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							   << 4U) 
							  | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							     >> 0x1cU))))
						    : 0xdeadbeefU)
						   : 0xdeadbeefU))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_ebreak 
	= ((0x73U == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U)))) 
	   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jal_sys_jal) 
	      & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
	      [0U]));
    // ALWAYS at VX_decode.v:264
    if ((0x4000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
	if ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
	    if ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
		vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset 
		    = ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
		        ? 0xdeadbeefU : ((0x400U & 
					  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					  ? 0xdeadbeefU
					  : ((0x200U 
					      & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					      ? ((0x100U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						  ? 
						 (((0U 
						    == 
						    (7U 
						     & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							 << 0xcU) 
							| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							   >> 0x14U)))) 
						   & (2U 
						      > 
						      (0xfffU 
						       & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							   << 4U) 
							  | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							     >> 0x1cU)))))
						   ? 0xb0000000U
						   : 0xdeadbeefU)
						  : 0xdeadbeefU)
					      : 0xdeadbeefU)));
	    } else {
		if ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
		    if ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
			vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset 
			    = ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
			        ? ((0x100U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
				    ? ((0xffe00000U 
					& (VL_NEGATE_I((IData)(
							       (1U 
								& (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
								   >> 7U)))) 
					   << 0x15U)) 
				       | ((0x100000U 
					   & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					      << 0xdU)) 
					  | ((0xff000U 
					      & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						  << 0x18U) 
						 | (0xfff000U 
						    & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						       >> 8U)))) 
					     | ((0x800U 
						 & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						    >> 0x11U)) 
						| (0x7feU 
						   & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						       << 4U) 
						      | (0xeU 
							 & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							    >> 0x1cU))))))))
				    : 0xdeadbeefU) : 0xdeadbeefU);
		    } else {
			if ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
			    if ((0x100U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
				if (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs) 
				     | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jmprt))) {
				    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset = 0U;
				}
			    } else {
				vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset = 0xdeadbeefU;
			    }
			} else {
			    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset = 0xdeadbeefU;
			}
		    }
		} else {
		    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset 
			= ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
			    ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
			        ? ((0x100U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
				    ? ((0xfffff000U 
					& (VL_NEGATE_I((IData)(
							       (1U 
								& (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
								   >> 7U)))) 
					   << 0xcU)) 
				       | (0xfffU & 
					  ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					    << 4U) 
					   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					      >> 0x1cU))))
				    : 0xdeadbeefU) : 0xdeadbeefU)
			    : 0xdeadbeefU);
		}
	    }
	} else {
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset = 0xdeadbeefU;
	}
    } else {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset = 0xdeadbeefU;
    }
    // ALWAYS at VX_decode.v:264
    if ((0x4000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
	if ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
	    if ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
		vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal 
		    = ((~ (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			   >> 0xbU)) & ((~ (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					    >> 0xaU)) 
					& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					    >> 9U) 
					   & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 8U) 
					      & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jal_sys_jal) 
						 & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
						 [0U])))));
	    } else {
		if ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
		    if ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
			vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal 
			    = ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				>> 9U) & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					   >> 8U) & 
					  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
					  [0U]));
		    } else {
			if ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
			    if ((0x100U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
				if (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs) 
				     | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jmprt))) {
				    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal 
					= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
					[0U];
				}
			    } else {
				vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal = 0U;
			    }
			} else {
			    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal = 0U;
			}
		    }
		} else {
		    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal 
			= ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			    >> 0xaU) & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					 >> 9U) & (
						   (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						    >> 8U) 
						   & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
						   [0U])));
		}
	    }
	} else {
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal = 0U;
	}
    } else {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal = 0U;
    }
    // ALWAYS at VX_decode.v:339
    if ((0x4000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
	if ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
	    if ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
		vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type = 0U;
	    } else {
		if ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
		    if ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
			vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type = 0U;
		    } else {
			if ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
			    if ((0x100U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
				if (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs) 
				     | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jmprt))) {
				    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type = 0U;
				}
			    } else {
				vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type = 0U;
			    }
			} else {
			    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type = 0U;
			}
		    }
		} else {
		    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type 
			= ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
			    ? 0U : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
				     ? ((0x100U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					 ? ((0x400000U 
					     & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
					     ? ((0x200000U 
						 & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						 ? 
						((0x100000U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						  ? 6U
						  : 5U)
						 : 
						((0x100000U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						  ? 4U
						  : 3U))
					     : ((0x200000U 
						 & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						 ? 0U
						 : 
						((0x100000U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						  ? 2U
						  : 1U)))
					 : 0U) : 0U));
		}
	    }
	} else {
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type = 0U;
	}
    } else {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type = 0U;
    }
    vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs) 
	   | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jmprt));
    // ALWAYS at VX_decode.v:339
    if ((0x4000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
	if ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
	    if ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
		vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall = 0U;
	    } else {
		if ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
		    if ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
			vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall 
			    = ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				>> 9U) & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					   >> 8U) & 
					  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
					  [0U]));
		    } else {
			if ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
			    if ((0x100U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])) {
				if (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs) 
				     | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jmprt))) {
				    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall 
					= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
					[0U];
				}
			    } else {
				vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall = 0U;
			    }
			} else {
			    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall = 0U;
			}
		    }
		} else {
		    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall 
			= ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
			    ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				>> 9U) & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					   >> 8U) & 
					  vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
					  [0U])) : 
			   ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			     >> 9U) & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					>> 8U) & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid
				       [0U])));
		}
	    }
	} else {
	    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall = 0U;
	}
    } else {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall = 0U;
    }
    vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U] 
	= ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			  << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 8U)))) ? 
	   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data[0U]
	    : ((1U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			      << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					   >> 8U))))
	        ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[6U] 
		    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[5U] 
				 >> 0xfU)) : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					       << 0x11U) 
					      | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						 >> 0xfU))));
    vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U] 
	= ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			  << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 8U)))) ? 
	   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data[1U]
	    : ((1U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			      << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					   >> 8U))))
	        ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[7U] 
		    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[6U] 
				 >> 0xfU)) : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[3U] 
					       << 0x11U) 
					      | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						 >> 0xfU))));
    vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U] 
	= ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			  << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 8U)))) ? 
	   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data[2U]
	    : ((1U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			      << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					   >> 8U))))
	        ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[8U] 
		    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[7U] 
				 >> 0xfU)) : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[4U] 
					       << 0x11U) 
					      | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[3U] 
						 >> 0xfU))));
    vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U] 
	= ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			  << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 8U)))) ? 
	   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data[3U]
	    : ((1U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			      << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					   >> 8U))))
	        ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[9U] 
		    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[8U] 
				 >> 0xfU)) : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[5U] 
					       << 0x11U) 
					      | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[4U] 
						 >> 0xfU))));
    vlTOPp->out_ebreak = ((0U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					   << 0x1cU) 
					  | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))) 
			  & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_ebreak));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__add_warp 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_wspawn) 
	   & (~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_ebreak)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_ebreak) 
	   & (~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_wspawn)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_final_alu 
	= ((0x63U == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U))))
	    ? ((5U > (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type))
	        ? 1U : 0xaU) : ((0x37U == (0x7fU & 
					   ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 8U))))
				 ? 0xbU : ((0x17U == 
					    (0x7fU 
					     & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						 << 0x18U) 
						| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						   >> 8U))))
					    ? 0xcU : 
					   ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_csr)
					     ? ((1U 
						 == 
						 (3U 
						  & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						      << 0xcU) 
						     | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							>> 0x14U))))
						 ? 0xdU
						 : 
						((2U 
						  == 
						  (3U 
						   & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						       << 0xcU) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							 >> 0x14U))))
						  ? 0xeU
						  : 0xfU))
					     : (((0x23U 
						  == 
						  (0x7fU 
						   & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						       << 0x18U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							 >> 8U)))) 
						 | (3U 
						    == 
						    (0x7fU 
						     & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							 << 0x18U) 
							| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							   >> 8U)))))
						 ? 0U
						 : 
						((0x400000U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						  ? 
						 ((0x200000U 
						   & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						   ? 
						  ((0x100000U 
						    & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						    ? 9U
						    : 8U)
						   : 
						  ((0x100000U 
						    & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						    ? 
						   ((0U 
						     == 
						     (0x7fU 
						      & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							 >> 1U)))
						     ? 6U
						     : 7U)
						    : 5U))
						  : 
						 ((0x200000U 
						   & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						   ? 
						  ((0x100000U 
						    & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						    ? 4U
						    : 3U)
						   : 
						  ((0x100000U 
						    & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U])
						    ? 2U
						    : 
						   ((0x13U 
						     == 
						     (0x7fU 
						      & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							  << 0x18U) 
							 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							    >> 8U))))
						     ? 0U
						     : 
						    ((0U 
						      == 
						      (0x7fU 
						       & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							  >> 1U)))
						      ? 0U
						      : 1U))))))))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_change_mask 
	= ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask) 
	   & (0U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					   >> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_change_mask 
	= ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask) 
	   & (1U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					   >> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_change_mask 
	= ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask) 
	   & (2U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					   >> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_change_mask 
	= ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask) 
	   & (3U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					   >> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_change_mask 
	= ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask) 
	   & (4U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					   >> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_change_mask 
	= ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask) 
	   & (5U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					   >> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_change_mask 
	= ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask) 
	   & (6U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					   >> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_change_mask 
	= ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.change_mask) 
	   & (7U == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			      << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					   >> 4U)))));
    vlTOPp->out_cache_driver_in_mem_read = (7U & ((
						   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
						   << 1U) 
						  | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
						     >> 0x1fU)));
    vlTOPp->out_cache_driver_in_mem_write = (7U & (
						   (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
						    << 4U) 
						   | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
						      >> 0x1cU)));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_address[0U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
	    << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
			 >> 2U));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_data[0U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[4U] 
	    << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[3U] 
			 >> 0xbU));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_valid[0U] 
	= (1U & (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U] 
		 >> 4U));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_address[1U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xaU] 
	    << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
			 >> 2U));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_data[1U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[5U] 
	    << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[4U] 
			 >> 0xbU));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_valid[1U] 
	= (1U & (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U] 
		 >> 5U));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_address[2U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xbU] 
	    << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xaU] 
			 >> 2U));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_data[2U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[6U] 
	    << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[5U] 
			 >> 0xbU));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_valid[2U] 
	= (1U & (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U] 
		 >> 6U));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_address[3U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
	    << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xbU] 
			 >> 2U));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_data[3U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
	    << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[6U] 
			 >> 0xbU));
    vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_valid[3U] 
	= (1U & (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U] 
		 >> 7U));
    vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[2U] 
	     << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[1U] 
			  >> 0xbU)) + ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[1U] 
					<< 0x16U) | 
				       (0x3ffffeU & 
					(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U] 
					 >> 0xaU))));
    // ALWAYS at VX_memory.v:59
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir 
	= (1U & ((0x400U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])
		  ? ((0x200U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])
		      ? ((~ (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U] 
			     >> 8U)) & (~ (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
					   >> 1U)))
		      : ((0x100U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])
			  ? (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
			     >> 1U) : (~ (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
					  >> 1U))))
		  : ((0x200U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])
		      ? ((0x100U & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])
			  ? (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
			     >> 1U) : (0U != ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
					       << 0x1eU) 
					      | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
						 >> 2U))))
		      : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U] 
			  >> 8U) & (0U == ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
					    << 0x1eU) 
					   | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					      >> 2U)))))));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd 
	= ((((((0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			  << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				    >> 0x17U))) == 
	       (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
			  << 9U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
				    >> 0x17U)))) & 
	      (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					  >> 0x17U))))) 
	     & (0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
			      << 0xbU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					  >> 0x15U))))) 
	    & (~ (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd))) 
	   & ((0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
				     >> 4U))) == (0xfU 
						  & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[0U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[3U] 
	    << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[2U] 
			 >> 0xbU));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[1U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[3U] 
	    << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[2U] 
			 >> 0xbU));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[2U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[3U] 
	    << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[2U] 
			 >> 0xbU));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[3U] 
	= ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[3U] 
	    << 0x15U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[2U] 
			 >> 0xbU));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd 
	= ((((((0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			  << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				    >> 0x1cU))) == 
	       (0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
			  << 9U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
				    >> 0x17U)))) & 
	      (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					  >> 0x1cU))))) 
	     & (0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
			      << 0xbU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					  >> 0x15U))))) 
	    & (~ (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd))) 
	   & ((0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
				     >> 4U))) == (0xfU 
						  & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__new_warp_count 
	= (0xfU & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__add_warp)
		    ? ((IData)(1U) + (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_count))
		    : ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp)
		        ? ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_count) 
			   - (IData)(1U)) : (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_count))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__new_warp_state 
	= (0xfU & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__add_warp)
		    ? ((IData)(1U) + (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_state))
		    : (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp) 
			& (3U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_count)))
		        ? 0U : (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_state))));
    vlTOPp->out_cache_driver_in_address[3U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_address
	[3U];
    vlTOPp->out_cache_driver_in_address[2U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_address
	[2U];
    vlTOPp->out_cache_driver_in_address[1U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_address
	[1U];
    vlTOPp->out_cache_driver_in_address[0U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_address
	[0U];
    vlTOPp->out_cache_driver_in_data[3U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_data
	[3U];
    vlTOPp->out_cache_driver_in_data[2U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_data
	[2U];
    vlTOPp->out_cache_driver_in_data[1U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_data
	[1U];
    vlTOPp->out_cache_driver_in_data[0U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_data
	[0U];
    vlTOPp->out_cache_driver_in_valid[3U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_valid
	[3U];
    vlTOPp->out_cache_driver_in_valid[2U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_valid
	[2U];
    vlTOPp->out_cache_driver_in_valid[1U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_valid
	[1U];
    vlTOPp->out_cache_driver_in_valid[0U] = vlSymsp->TOP__Vortex__DOT__VX_dcache_req.out_cache_driver_in_valid
	[0U];
    // ALWAYS at VX_warp.v:57
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__temp_PC 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
	     >> 2U) & (0U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
	    ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
		<< 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
			     >> 2U)) : (((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir) 
					 & (0U == (0xfU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
					 ? vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest
					 : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__real_PC));
    // ALWAYS at VX_warp.v:57
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__temp_PC 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
	     >> 2U) & (1U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
	    ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
		<< 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
			     >> 2U)) : (((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir) 
					 & (1U == (0xfU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
					 ? vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest
					 : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__real_PC));
    // ALWAYS at VX_warp.v:57
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__temp_PC 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
	     >> 2U) & (2U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
	    ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
		<< 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
			     >> 2U)) : (((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir) 
					 & (2U == (0xfU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
					 ? vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest
					 : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__real_PC));
    // ALWAYS at VX_warp.v:57
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__temp_PC 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
	     >> 2U) & (3U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
	    ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
		<< 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
			     >> 2U)) : (((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir) 
					 & (3U == (0xfU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
					 ? vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest
					 : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__real_PC));
    // ALWAYS at VX_warp.v:57
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__temp_PC 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
	     >> 2U) & (4U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
	    ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
		<< 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
			     >> 2U)) : (((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir) 
					 & (4U == (0xfU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
					 ? vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest
					 : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__real_PC));
    // ALWAYS at VX_warp.v:57
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__temp_PC 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
	     >> 2U) & (5U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
	    ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
		<< 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
			     >> 2U)) : (((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir) 
					 & (5U == (0xfU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
					 ? vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest
					 : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__real_PC));
    // ALWAYS at VX_warp.v:57
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__temp_PC 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
	     >> 2U) & (6U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
	    ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
		<< 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
			     >> 2U)) : (((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir) 
					 & (6U == (0xfU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
					 ? vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest
					 : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__real_PC));
    // ALWAYS at VX_warp.v:57
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__temp_PC 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
	     >> 2U) & (7U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
	    ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
		<< 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
			     >> 2U)) : (((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir) 
					 & (7U == (0xfU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
					 ? vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest
					 : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__real_PC));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd 
	= (((((((0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			   << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				     >> 0x17U))) == 
		(0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					>> 0xaU)))) 
	       & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				   << 9U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 0x17U))))) 
	      & (0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			       << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 8U))))) 
	     & ((0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]) 
		== (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			     << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					  >> 4U))))) 
	    & (~ (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd))) 
	   & (~ (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)));
    vlTOPp->Vortex__DOT__forwarding_fwd_stall = ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd) 
						   | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd)) 
						  & (2U 
						     == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))) 
						 | (((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd) 
						     | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)) 
						    & (2U 
						       == 
						       (3U 
							& ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
							    << 0xbU) 
							   | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
							      >> 0x15U))))));
    vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd 
	= (((((((0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
			   << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				     >> 0x1cU))) == 
		(0x1fU & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0x16U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					>> 0xaU)))) 
	       & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				   << 4U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 0x1cU))))) 
	      & (0U != (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			       << 0x18U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 8U))))) 
	     & (~ (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd))) 
	    & (~ (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd))) 
	   & ((0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]) 
	      == (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
			   << 0x1cU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					>> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[0U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__temp_PC;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[1U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__temp_PC;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[2U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__temp_PC;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[3U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__temp_PC;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[4U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__temp_PC;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[5U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__temp_PC;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[6U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__temp_PC;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[7U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__temp_PC;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall 
	= (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall) 
	    | (IData)(vlTOPp->Vortex__DOT__forwarding_fwd_stall)) 
	   | (IData)(vlTOPp->Vortex__DOT__execute_branch_stall));
    // ALWAYS at VX_fetch.v:152
    if ((0U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[0U];
    }
    if ((1U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[1U];
    }
    if ((2U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[2U];
    }
    if ((3U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[3U];
    }
    if ((4U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[4U];
    }
    if ((5U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[5U];
    }
    if ((6U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[6U];
    }
    if ((7U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[7U];
    }
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_stall 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall) 
	   | (0U != (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_stall 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall) 
	   | (1U != (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_stall 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall) 
	   | (2U != (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_stall 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall) 
	   | (3U != (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_stall 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall) 
	   | (4U != (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_stall 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall) 
	   | (5U != (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_stall 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall) 
	   | (6U != (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_stall 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall) 
	   | (7U != (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num)));
    vlTOPp->icache_request_pc_address = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var;
}

VL_INLINE_OPT void VVortex::_multiclk__TOP__8(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_multiclk__TOP__8\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->Vortex__DOT__csr_decode_csr_data = ((0xc00U 
						 == (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__decode_csr_address))
						 ? (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__cycle)
						 : 
						((0xc80U 
						  == (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__decode_csr_address))
						  ? (IData)(
							    (vlTOPp->Vortex__DOT__vx_csr_handler__DOT__cycle 
							     >> 0x20U))
						  : 
						 ((0xc02U 
						   == (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__decode_csr_address))
						   ? (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__instret)
						   : 
						  ((0xc82U 
						    == (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__decode_csr_address))
						    ? (IData)(
							      (vlTOPp->Vortex__DOT__vx_csr_handler__DOT__instret 
							       >> 0x20U))
						    : (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__data_read)))));
}

void VVortex::_settle__TOP__9(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_settle__TOP__9\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    // Begin mtask footprint  all: 
    VL_SIGW(__Vtemp61,223,0,7);
    VL_SIGW(__Vtemp70,479,0,15);
    VL_SIGW(__Vtemp85,127,0,4);
    VL_SIGW(__Vtemp106,127,0,4);
    VL_SIGW(__Vtemp112,319,0,10);
    // Body
    // ALWAYS at VX_warp.v:57
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__temp_PC 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
	     >> 2U) & (0U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
	    ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
		<< 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
			     >> 2U)) : (((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir) 
					 & (0U == (0xfU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
					 ? vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest
					 : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__real_PC));
    // ALWAYS at VX_warp.v:57
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__temp_PC 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
	     >> 2U) & (1U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
	    ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
		<< 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
			     >> 2U)) : (((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir) 
					 & (1U == (0xfU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
					 ? vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest
					 : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__real_PC));
    // ALWAYS at VX_warp.v:57
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__temp_PC 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
	     >> 2U) & (2U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
	    ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
		<< 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
			     >> 2U)) : (((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir) 
					 & (2U == (0xfU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
					 ? vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest
					 : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__real_PC));
    // ALWAYS at VX_warp.v:57
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__temp_PC 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
	     >> 2U) & (3U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
	    ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
		<< 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
			     >> 2U)) : (((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir) 
					 & (3U == (0xfU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
					 ? vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest
					 : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__real_PC));
    // ALWAYS at VX_warp.v:57
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__temp_PC 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
	     >> 2U) & (4U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
	    ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
		<< 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
			     >> 2U)) : (((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir) 
					 & (4U == (0xfU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
					 ? vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest
					 : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__real_PC));
    // ALWAYS at VX_warp.v:57
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__temp_PC 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
	     >> 2U) & (5U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
	    ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
		<< 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
			     >> 2U)) : (((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir) 
					 & (5U == (0xfU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
					 ? vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest
					 : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__real_PC));
    // ALWAYS at VX_warp.v:57
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__temp_PC 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
	     >> 2U) & (6U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
	    ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
		<< 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
			     >> 2U)) : (((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir) 
					 & (6U == (0xfU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
					 ? vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest
					 : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__real_PC));
    // ALWAYS at VX_warp.v:57
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__temp_PC 
	= (((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
	     >> 2U) & (7U == (0xfU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
	    ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xdU] 
		<< 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
			     >> 2U)) : (((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir) 
					 & (7U == (0xfU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0U])))
					 ? vlSymsp->TOP__Vortex__DOT__VX_branch_rsp.branch_dest
					 : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__real_PC));
    vlTOPp->Vortex__DOT__csr_decode_csr_data = ((0xc00U 
						 == (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__decode_csr_address))
						 ? (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__cycle)
						 : 
						((0xc80U 
						  == (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__decode_csr_address))
						  ? (IData)(
							    (vlTOPp->Vortex__DOT__vx_csr_handler__DOT__cycle 
							     >> 0x20U))
						  : 
						 ((0xc02U 
						   == (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__decode_csr_address))
						   ? (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__instret)
						   : 
						  ((0xc82U 
						    == (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__decode_csr_address))
						    ? (IData)(
							      (vlTOPp->Vortex__DOT__vx_csr_handler__DOT__instret 
							       >> 0x20U))
						    : (IData)(vlTOPp->Vortex__DOT__vx_csr_handler__DOT__data_read)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__new_warp_count 
	= (0xfU & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__add_warp)
		    ? ((IData)(1U) + (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_count))
		    : ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp)
		        ? ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_count) 
			   - (IData)(1U)) : (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_count))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__new_warp_state 
	= (0xfU & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__add_warp)
		    ? ((IData)(1U) + (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_state))
		    : (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp) 
			& (3U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_count)))
		        ? 0U : (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_state))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__new_warp_num 
	= (0xfU & (((((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num) 
						>= (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_state))) 
		     | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp)) 
		    | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__add_warp))
		    ? 0U : ((IData)(1U) + (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_stall 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall) 
	   | (0U != (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_stall 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall) 
	   | (1U != (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_stall 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall) 
	   | (2U != (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_stall 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall) 
	   | (3U != (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_stall 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall) 
	   | (4U != (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_stall 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall) 
	   | (5U != (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_stall 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall) 
	   | (6U != (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_stall 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall) 
	   | (7U != (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[0U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__temp_PC;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[1U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__temp_PC;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[2U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__temp_PC;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[3U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__temp_PC;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[4U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__temp_PC;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[5U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__temp_PC;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[6U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__temp_PC;
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[7U] 
	= vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__temp_PC;
    // ALWAYS at VX_alu.v:48
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__0__KET____DOT__vx_alu__out_alu_result 
	= ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	    ? ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	        ? 0U : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			 ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			     ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
					   >> 0xeU))
				     : VL_MODDIV_III(32, 
						     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						       << 0x12U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
							 >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2))
				 : ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
					   >> 0xeU))
				     : VL_MODDIVS_III(32, 
						      ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
							<< 0x12U) 
						       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
							  >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)))
			     : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? 0xffffffffU : 
				    VL_DIV_III(32, 
					       ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						 << 0x12U) 
						| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						   >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2))
				 : ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? 0xffffffffU : 
				    VL_DIVS_III(32, 
						((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						  << 0x12U) 
						 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						    >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2))))
			 : ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			     ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? (IData)((((QData)((IData)(
							     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
							       << 0x12U) 
							      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
								 >> 0xeU)))) 
					     * (QData)((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2))) 
					    >> 0x20U))
				 : (IData)((((((QData)((IData)(
							       VL_NEGATE_I((IData)(
										(1U 
										& (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
										>> 0xdU)))))) 
					       << 0x20U) 
					      | (QData)((IData)(
								((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
								  << 0x12U) 
								 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
								    >> 0xeU))))) 
					     * (QData)((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2))) 
					    >> 0x20U)))
			     : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? (IData)((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__mult_signed_result 
					    >> 0x20U))
				 : (IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__mult_signed_result)))))
	    : ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	        ? ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		    ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? vlTOPp->Vortex__DOT__csr_decode_csr_data
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? vlTOPp->Vortex__DOT__csr_decode_csr_data
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
				 << 0x17U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
					      >> 9U)) 
			       + (0xfffff000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
						 << 3U)))))
		    : ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (0xfffff000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
					      << 3U))
			    : ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
				  << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
					       >> 0xeU)) 
				>= vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
			        ? 0U : 0xffffffffU))
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2 
			       & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
				   << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						>> 0xeU)))
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
					      >> 0xeU)) 
			       | vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2))))
	        : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		    ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? VL_SHIFTRS_III(32,32,5, 
					     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					       << 0x12U) 
					      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						 >> 0xeU)), 
					     (0x1fU 
					      & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2))
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
					      >> 0xeU)) 
			       >> (0x1fU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)))
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
					      >> 0xeU)) 
			       ^ vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
			    : ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
				  << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
					       >> 0xeU)) 
				< vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
			        ? 1U : 0U))) : ((0x400U 
						 & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						 ? 
						((0x200U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						  ? 
						 (VL_LTS_III(1,32,32, 
							     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
							       << 0x12U) 
							      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
								 >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
						   ? 1U
						   : 0U)
						  : 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						      >> 0xeU)) 
						  << 
						  (0x1fU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)))
						 : 
						((0x200U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						  ? 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						      >> 0xeU)) 
						  - vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
						  : 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						      >> 0xeU)) 
						  + vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2))))));
    // ALWAYS at VX_alu.v:48
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__1__KET____DOT__vx_alu__out_alu_result 
	= ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	    ? ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	        ? 0U : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			 ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			     ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					   >> 0xeU))
				     : VL_MODDIV_III(32, 
						     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						       << 0x12U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
							 >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2))
				 : ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					   >> 0xeU))
				     : VL_MODDIVS_III(32, 
						      ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
							<< 0x12U) 
						       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
							  >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)))
			     : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? 0xffffffffU : 
				    VL_DIV_III(32, 
					       ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						 << 0x12U) 
						| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						   >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2))
				 : ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? 0xffffffffU : 
				    VL_DIVS_III(32, 
						((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						  << 0x12U) 
						 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						    >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2))))
			 : ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			     ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? (IData)((((QData)((IData)(
							     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
							       << 0x12U) 
							      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
								 >> 0xeU)))) 
					     * (QData)((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2))) 
					    >> 0x20U))
				 : (IData)((((((QData)((IData)(
							       VL_NEGATE_I((IData)(
										(1U 
										& (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
										>> 0xdU)))))) 
					       << 0x20U) 
					      | (QData)((IData)(
								((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
								  << 0x12U) 
								 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
								    >> 0xeU))))) 
					     * (QData)((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2))) 
					    >> 0x20U)))
			     : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? (IData)((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__mult_signed_result 
					    >> 0x20U))
				 : (IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__mult_signed_result)))))
	    : ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	        ? ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		    ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? vlTOPp->Vortex__DOT__csr_decode_csr_data
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? vlTOPp->Vortex__DOT__csr_decode_csr_data
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
				 << 0x17U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
					      >> 9U)) 
			       + (0xfffff000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
						 << 3U)))))
		    : ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (0xfffff000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
					      << 3U))
			    : ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
				  << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					       >> 0xeU)) 
				>= vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
			        ? 0U : 0xffffffffU))
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2 
			       & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
				   << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						>> 0xeU)))
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					      >> 0xeU)) 
			       | vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2))))
	        : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		    ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? VL_SHIFTRS_III(32,32,5, 
					     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					       << 0x12U) 
					      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						 >> 0xeU)), 
					     (0x1fU 
					      & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2))
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					      >> 0xeU)) 
			       >> (0x1fU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)))
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					      >> 0xeU)) 
			       ^ vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
			    : ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
				  << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					       >> 0xeU)) 
				< vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
			        ? 1U : 0U))) : ((0x400U 
						 & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						 ? 
						((0x200U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						  ? 
						 (VL_LTS_III(1,32,32, 
							     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
							       << 0x12U) 
							      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
								 >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
						   ? 1U
						   : 0U)
						  : 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						      >> 0xeU)) 
						  << 
						  (0x1fU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)))
						 : 
						((0x200U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						  ? 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						      >> 0xeU)) 
						  - vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
						  : 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						      >> 0xeU)) 
						  + vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2))))));
    // ALWAYS at VX_alu.v:48
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__2__KET____DOT__vx_alu__out_alu_result 
	= ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	    ? ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	        ? 0U : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			 ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			     ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					   >> 0xeU))
				     : VL_MODDIV_III(32, 
						     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						       << 0x12U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
							 >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2))
				 : ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					   >> 0xeU))
				     : VL_MODDIVS_III(32, 
						      ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
							<< 0x12U) 
						       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
							  >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)))
			     : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? 0xffffffffU : 
				    VL_DIV_III(32, 
					       ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						 << 0x12U) 
						| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						   >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2))
				 : ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? 0xffffffffU : 
				    VL_DIVS_III(32, 
						((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						  << 0x12U) 
						 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						    >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2))))
			 : ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			     ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? (IData)((((QData)((IData)(
							     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
							       << 0x12U) 
							      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
								 >> 0xeU)))) 
					     * (QData)((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2))) 
					    >> 0x20U))
				 : (IData)((((((QData)((IData)(
							       VL_NEGATE_I((IData)(
										(1U 
										& (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
										>> 0xdU)))))) 
					       << 0x20U) 
					      | (QData)((IData)(
								((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
								  << 0x12U) 
								 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
								    >> 0xeU))))) 
					     * (QData)((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2))) 
					    >> 0x20U)))
			     : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? (IData)((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__mult_signed_result 
					    >> 0x20U))
				 : (IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__mult_signed_result)))))
	    : ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	        ? ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		    ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? vlTOPp->Vortex__DOT__csr_decode_csr_data
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? vlTOPp->Vortex__DOT__csr_decode_csr_data
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
				 << 0x17U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
					      >> 9U)) 
			       + (0xfffff000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
						 << 3U)))))
		    : ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (0xfffff000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
					      << 3U))
			    : ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
				  << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					       >> 0xeU)) 
				>= vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
			        ? 0U : 0xffffffffU))
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2 
			       & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
				   << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						>> 0xeU)))
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					      >> 0xeU)) 
			       | vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2))))
	        : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		    ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? VL_SHIFTRS_III(32,32,5, 
					     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					       << 0x12U) 
					      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						 >> 0xeU)), 
					     (0x1fU 
					      & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2))
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					      >> 0xeU)) 
			       >> (0x1fU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)))
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					      >> 0xeU)) 
			       ^ vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
			    : ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
				  << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					       >> 0xeU)) 
				< vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
			        ? 1U : 0U))) : ((0x400U 
						 & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						 ? 
						((0x200U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						  ? 
						 (VL_LTS_III(1,32,32, 
							     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
							       << 0x12U) 
							      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
								 >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
						   ? 1U
						   : 0U)
						  : 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						      >> 0xeU)) 
						  << 
						  (0x1fU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)))
						 : 
						((0x200U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						  ? 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						      >> 0xeU)) 
						  - vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
						  : 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						      >> 0xeU)) 
						  + vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2))))));
    // ALWAYS at VX_alu.v:48
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__3__KET____DOT__vx_alu__out_alu_result 
	= ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	    ? ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	        ? 0U : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			 ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			     ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					   >> 0xeU))
				     : VL_MODDIV_III(32, 
						     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						       << 0x12U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
							 >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2))
				 : ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					   >> 0xeU))
				     : VL_MODDIVS_III(32, 
						      ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
							<< 0x12U) 
						       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
							  >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)))
			     : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? 0xffffffffU : 
				    VL_DIV_III(32, 
					       ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						 << 0x12U) 
						| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						   >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2))
				 : ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? 0xffffffffU : 
				    VL_DIVS_III(32, 
						((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						  << 0x12U) 
						 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						    >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2))))
			 : ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			     ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? (IData)((((QData)((IData)(
							     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
							       << 0x12U) 
							      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
								 >> 0xeU)))) 
					     * (QData)((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2))) 
					    >> 0x20U))
				 : (IData)((((((QData)((IData)(
							       VL_NEGATE_I((IData)(
										(1U 
										& (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
										>> 0xdU)))))) 
					       << 0x20U) 
					      | (QData)((IData)(
								((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
								  << 0x12U) 
								 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
								    >> 0xeU))))) 
					     * (QData)((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2))) 
					    >> 0x20U)))
			     : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? (IData)((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__mult_signed_result 
					    >> 0x20U))
				 : (IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__mult_signed_result)))))
	    : ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	        ? ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		    ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? vlTOPp->Vortex__DOT__csr_decode_csr_data
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? vlTOPp->Vortex__DOT__csr_decode_csr_data
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
				 << 0x17U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
					      >> 9U)) 
			       + (0xfffff000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
						 << 3U)))))
		    : ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (0xfffff000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
					      << 3U))
			    : ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
				  << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					       >> 0xeU)) 
				>= vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
			        ? 0U : 0xffffffffU))
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2 
			       & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
				   << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						>> 0xeU)))
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					      >> 0xeU)) 
			       | vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2))))
	        : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		    ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? VL_SHIFTRS_III(32,32,5, 
					     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
					       << 0x12U) 
					      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						 >> 0xeU)), 
					     (0x1fU 
					      & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2))
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					      >> 0xeU)) 
			       >> (0x1fU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)))
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					      >> 0xeU)) 
			       ^ vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
			    : ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
				  << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					       >> 0xeU)) 
				< vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
			        ? 1U : 0U))) : ((0x400U 
						 & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						 ? 
						((0x200U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						  ? 
						 (VL_LTS_III(1,32,32, 
							     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
							       << 0x12U) 
							      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
								 >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
						   ? 1U
						   : 0U)
						  : 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						      >> 0xeU)) 
						  << 
						  (0x1fU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)))
						 : 
						((0x200U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						  ? 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						      >> 0xeU)) 
						  - vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
						  : 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						      >> 0xeU)) 
						  + vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2))))));
    // ALWAYS at VX_fetch.v:152
    if ((0U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[0U];
    }
    if ((1U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[1U];
    }
    if ((2U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[2U];
    }
    if ((3U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[3U];
    }
    if ((4U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[4U];
    }
    if ((5U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[5U];
    }
    if ((6U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[6U];
    }
    if ((7U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var 
	    = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc[7U];
    }
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[0U] 
	= vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__0__KET____DOT__vx_alu__out_alu_result;
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[1U] 
	= vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__1__KET____DOT__vx_alu__out_alu_result;
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[2U] 
	= vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__2__KET____DOT__vx_alu__out_alu_result;
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[3U] 
	= vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__3__KET____DOT__vx_alu__out_alu_result;
    vlTOPp->icache_request_pc_address = vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var;
    __Vtemp61[5U] = ((0xfff80000U & (((0xdU == (0x1fU 
						& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
						    << 0x17U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
						      >> 9U))))
				       ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
					   << 3U) | 
					  (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
					   >> 0x1dU))
				       : ((0xeU == 
					   (0x1fU & 
					    ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
					      << 0x17U) 
					     | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
						>> 9U))))
					   ? (vlTOPp->Vortex__DOT__csr_decode_csr_data 
					      | ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
						  << 3U) 
						 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						    >> 0x1dU)))
					   : ((0xfU 
					       == (0x1fU 
						   & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
						       << 0x17U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
							 >> 9U))))
					       ? (vlTOPp->Vortex__DOT__csr_decode_csr_data 
						  & ((IData)(0xffffffffU) 
						     - 
						     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
						       << 3U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
							 >> 0x1dU))))
					       : 0xdeadbeefU))) 
				     << 0x13U)) | (
						   (0x40000U 
						    & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
						       << 0xaU)) 
						   | (0x3ffffU 
						      & ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
							    << 0x12U) 
							   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
							      >> 0xeU)) 
							  + 
							  ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
							    << 0x18U) 
							   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[1U] 
							      >> 8U))) 
							 >> 0xeU))));
    __Vtemp61[6U] = (0x7ffffU & (((0xdU == (0x1fU & 
					    ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
					      << 0x17U) 
					     | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
						>> 9U))))
				   ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
				       << 3U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						 >> 0x1dU))
				   : ((0xeU == (0x1fU 
						& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
						    << 0x17U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
						      >> 9U))))
				       ? (vlTOPp->Vortex__DOT__csr_decode_csr_data 
					  | ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
					      << 3U) 
					     | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						>> 0x1dU)))
				       : ((0xfU == 
					   (0x1fU & 
					    ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
					      << 0x17U) 
					     | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
						>> 9U))))
					   ? (vlTOPp->Vortex__DOT__csr_decode_csr_data 
					      & ((IData)(0xffffffffU) 
						 - 
						 ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
						   << 3U) 
						  | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						     >> 0x1dU))))
					   : 0xdeadbeefU))) 
				 >> 0xdU));
    __Vtemp70[7U] = ((0xffff0000U & ((0x80000000U & 
				      (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
				       << 0x1cU)) | 
				     ((0x70000000U 
				       & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
					  << 0x1cU)) 
				      | ((0xf800000U 
					  & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
					      << 0x1fU) 
					     | (0x7f800000U 
						& (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						   >> 1U)))) 
					 | (((IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb) 
					     << 0x15U) 
					    | (0x1f0000U 
					       & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
						   << 0x1dU) 
						  | (0x1fff0000U 
						     & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
							>> 3U))))))))) 
		     | ((0xf800U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
				     << 0x1dU) | (0x1ffff800U 
						  & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						     >> 3U)))) 
			| (0x7ffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
				     >> 3U))));
    __Vtemp70[8U] = ((0xffffU & ((0xfffcU & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[0U] 
					     << 2U)) 
				 | ((3U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
					   >> 4U)) 
				    | (0xffffU & ((IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb) 
						  >> 0xbU))))) 
		     | (0xffff0000U & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[0U] 
				       << 2U)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0U] 
	= ((0xfffff800U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
			   << 5U)) | ((0x700U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
						  << 0xbU) 
						 | (0x700U 
						    & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
						       >> 0x15U)))) 
				      | (0xffU & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U])));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[1U] 
	= ((0x7ffU & ((0x7e0U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
				 << 5U)) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
					    >> 0x1bU))) 
	   | (0xfffff800U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
			     << 2U)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[2U] 
	= ((0x7ffU & ((0x7fcU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
				 << 2U)) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
					    >> 0x1eU))) 
	   | (0xfffff800U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U] 
			     << 3U)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[3U] 
	= ((0x7ffU & ((0x7f8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[1U] 
				 << 3U)) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U] 
					    >> 0x1dU))) 
	   | (0xfffff800U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
			      << 0x1dU) | (0x1ffff800U 
					   & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
					      >> 3U)))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[4U] 
	= ((0x7ffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
		      >> 3U)) | (0xfffff800U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[7U] 
						 << 0x1dU) 
						| (0x1ffff800U 
						   & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
						      >> 3U)))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[5U] 
	= ((0x7ffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[7U] 
		      >> 3U)) | (0xfffff800U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[8U] 
						 << 0x1dU) 
						| (0x1ffff800U 
						   & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[7U] 
						      >> 3U)))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[6U] 
	= ((0x7ffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[8U] 
		      >> 3U)) | (0xfffff800U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						 << 0x1dU) 
						| (0x1ffff800U 
						   & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[8U] 
						      >> 3U)))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[7U] 
	= __Vtemp70[7U];
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[8U] 
	= __Vtemp70[8U];
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[9U] 
	= ((0xffffU & ((3U & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[0U] 
			      >> 0x1eU)) | (0xfffcU 
					    & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[1U] 
					       << 2U)))) 
	   | (0xffff0000U & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[1U] 
			     << 2U)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0xaU] 
	= ((0xffffU & ((3U & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[1U] 
			      >> 0x1eU)) | (0xfffcU 
					    & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[2U] 
					       << 2U)))) 
	   | (0xffff0000U & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[2U] 
			     << 2U)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0xbU] 
	= ((0xffffU & ((3U & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[2U] 
			      >> 0x1eU)) | (0xfffcU 
					    & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[3U] 
					       << 2U)))) 
	   | (0xffff0000U & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[3U] 
			     << 2U)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0xcU] 
	= ((0xffffU & ((3U & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[3U] 
			      >> 0x1eU)) | (0xfffcU 
					    & ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						  << 0x12U) 
						 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						    >> 0xeU)) 
						+ (
						   (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
						    << 0x18U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[1U] 
						      >> 8U))) 
					       << 2U)))) 
	   | (0xffff0000U & (__Vtemp61[5U] << 0x10U)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0xdU] 
	= ((0xffffU & (__Vtemp61[5U] >> 0x10U)) | (0xffff0000U 
						   & (__Vtemp61[6U] 
						      << 0x10U)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0xeU] 
	= (0xffffU & ((0xfff0U & ((0xffc0U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xfU] 
					      << 6U)) 
				  | (0x30U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
					      >> 0x1aU)))) 
		      | ((8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
				>> 0x1aU)) | (__Vtemp61[6U] 
					      >> 0x10U))));
    __Vtemp85[0U] = ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)
		      ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					<< 0xbU) | 
				       (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					>> 0x15U))))
			  ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[0U]
			  : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					    << 0xbU) 
					   | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					      >> 0x15U))))
			      ? vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[0U]
			      : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
				  << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					       >> 2U))))
		      : ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd)
			  ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					    << 0x18U) 
					   | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 8U))))
			      ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[0U]
			      : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						<< 0x18U) 
					       | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						  >> 8U))))
				  ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				      << 0x11U) | (
						   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						   >> 0xfU))
				  : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[6U] 
				      << 0x11U) | (
						   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[5U] 
						   >> 0xfU))))
			  : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[0U]));
    __Vtemp85[1U] = ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)
		      ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					<< 0xbU) | 
				       (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					>> 0x15U))))
			  ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[1U]
			  : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					    << 0xbU) 
					   | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					      >> 0x15U))))
			      ? vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[1U]
			      : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xaU] 
				  << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
					       >> 2U))))
		      : ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd)
			  ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					    << 0x18U) 
					   | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 8U))))
			      ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[1U]
			      : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						<< 0x18U) 
					       | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						  >> 8U))))
				  ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[3U] 
				      << 0x11U) | (
						   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						   >> 0xfU))
				  : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[7U] 
				      << 0x11U) | (
						   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[6U] 
						   >> 0xfU))))
			  : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[1U]));
    __Vtemp85[2U] = ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)
		      ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					<< 0xbU) | 
				       (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					>> 0x15U))))
			  ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[2U]
			  : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					    << 0xbU) 
					   | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					      >> 0x15U))))
			      ? vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[2U]
			      : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xbU] 
				  << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xaU] 
					       >> 2U))))
		      : ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd)
			  ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					    << 0x18U) 
					   | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 8U))))
			      ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[2U]
			      : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						<< 0x18U) 
					       | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						  >> 8U))))
				  ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[4U] 
				      << 0x11U) | (
						   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[3U] 
						   >> 0xfU))
				  : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[8U] 
				      << 0x11U) | (
						   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[7U] 
						   >> 0xfU))))
			  : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[2U]));
    __Vtemp85[3U] = ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)
		      ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					<< 0xbU) | 
				       (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					>> 0x15U))))
			  ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[3U]
			  : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					    << 0xbU) 
					   | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					      >> 0x15U))))
			      ? vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[3U]
			      : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
				  << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xbU] 
					       >> 2U))))
		      : ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd)
			  ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					    << 0x18U) 
					   | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 8U))))
			      ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[3U]
			      : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						<< 0x18U) 
					       | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						  >> 8U))))
				  ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[5U] 
				      << 0x11U) | (
						   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[4U] 
						   >> 0xfU))
				  : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[9U] 
				      << 0x11U) | (
						   vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[8U] 
						   >> 0xfU))))
			  : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[3U]));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U] 
	= ((0x6fU == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U))))
	    ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data[0U]
	    : ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd) 
		 | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)) 
		| (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd))
	        ? ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd)
		    ? ((3U == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))
		        ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[0U]
		        : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[0U])
		    : __Vtemp85[0U]) : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[
	       (0x1cU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
			 >> 2U))]));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[1U] 
	= ((0x6fU == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U))))
	    ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data[1U]
	    : ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd) 
		 | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)) 
		| (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd))
	        ? ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd)
		    ? ((3U == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))
		        ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[1U]
		        : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[1U])
		    : __Vtemp85[1U]) : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[
	       ((IData)(1U) + (0x1cU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					>> 2U)))]));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[2U] 
	= ((0x6fU == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U))))
	    ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data[2U]
	    : ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd) 
		 | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)) 
		| (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd))
	        ? ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd)
		    ? ((3U == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))
		        ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[2U]
		        : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[2U])
		    : __Vtemp85[2U]) : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[
	       ((IData)(2U) + (0x1cU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					>> 2U)))]));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[3U] 
	= ((0x6fU == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U))))
	    ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data[3U]
	    : ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd) 
		 | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)) 
		| (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd))
	        ? ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd)
		    ? ((3U == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))
		        ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[3U]
		        : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[3U])
		    : __Vtemp85[3U]) : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[
	       ((IData)(3U) + (0x1cU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					>> 2U)))]));
    __Vtemp106[0U] = ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)
		       ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					 << 0xbU) | 
					(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					 >> 0x15U))))
			   ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[0U]
			   : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					     << 0xbU) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					       >> 0x15U))))
			       ? vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[0U]
			       : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
				   << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
						>> 2U))))
		       : ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd)
			   ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 8U))))
			       ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[0U]
			       : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						 << 0x18U) 
						| (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						   >> 8U))))
				   ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				       << 0x11U) | 
				      (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0xfU)) : 
				  ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[6U] 
				    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[5U] 
						 >> 0xfU))))
			   : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[0U]));
    __Vtemp106[1U] = ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)
		       ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					 << 0xbU) | 
					(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					 >> 0x15U))))
			   ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[1U]
			   : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					     << 0xbU) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					       >> 0x15U))))
			       ? vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[1U]
			       : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xaU] 
				   << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
						>> 2U))))
		       : ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd)
			   ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 8U))))
			       ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[1U]
			       : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						 << 0x18U) 
						| (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						   >> 8U))))
				   ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[3U] 
				       << 0x11U) | 
				      (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				       >> 0xfU)) : 
				  ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[7U] 
				    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[6U] 
						 >> 0xfU))))
			   : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[1U]));
    __Vtemp106[2U] = ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)
		       ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					 << 0xbU) | 
					(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					 >> 0x15U))))
			   ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[2U]
			   : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					     << 0xbU) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					       >> 0x15U))))
			       ? vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[2U]
			       : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xbU] 
				   << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xaU] 
						>> 2U))))
		       : ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd)
			   ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 8U))))
			       ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[2U]
			       : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						 << 0x18U) 
						| (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						   >> 8U))))
				   ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[4U] 
				       << 0x11U) | 
				      (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[3U] 
				       >> 0xfU)) : 
				  ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[8U] 
				    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[7U] 
						 >> 0xfU))))
			   : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[2U]));
    __Vtemp106[3U] = ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)
		       ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					 << 0xbU) | 
					(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					 >> 0x15U))))
			   ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[3U]
			   : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					     << 0xbU) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					       >> 0x15U))))
			       ? vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[3U]
			       : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
				   << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xbU] 
						>> 2U))))
		       : ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd)
			   ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 8U))))
			       ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[3U]
			       : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						 << 0x18U) 
						| (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						   >> 8U))))
				   ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[5U] 
				       << 0x11U) | 
				      (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[4U] 
				       >> 0xfU)) : 
				  ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[9U] 
				    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[8U] 
						 >> 0xfU))))
			   : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[3U]));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[0U] 
	= ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd) 
	     | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)) 
	    | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd))
	    ? ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd)
	        ? ((3U == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))
		    ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[0U]
		    : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[0U])
	        : __Vtemp106[0U]) : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[
	   (0x1cU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
		     >> 2U))]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[1U] 
	= ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd) 
	     | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)) 
	    | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd))
	    ? ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd)
	        ? ((3U == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))
		    ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[1U]
		    : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[1U])
	        : __Vtemp106[1U]) : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[
	   ((IData)(1U) + (0x1cU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
				    >> 2U)))]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[2U] 
	= ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd) 
	     | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)) 
	    | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd))
	    ? ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd)
	        ? ((3U == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))
		    ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[2U]
		    : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[2U])
	        : __Vtemp106[2U]) : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[
	   ((IData)(2U) + (0x1cU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
				    >> 2U)))]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[3U] 
	= ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd) 
	     | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)) 
	    | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd))
	    ? ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd)
	        ? ((3U == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))
		    ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[3U]
		    : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[3U])
	        : __Vtemp106[3U]) : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[
	   ((IData)(3U) + (0x1cU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
				    >> 2U)))]);
    __Vtemp112[0U] = (IData)((((QData)((IData)((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						  >> 1U) 
						 & (0x33U 
						    == 
						    (0x7fU 
						     & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							 << 0x18U) 
							| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							   >> 8U)))))
						 ? (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu)
						 : (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_final_alu)))) 
			       << 0x2cU) | (((QData)((IData)(
							     (((((0x6fU 
								  == 
								  (0x7fU 
								   & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
								       << 0x18U) 
								      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									 >> 8U)))) 
								 | (0x67U 
								    == 
								    (0x7fU 
								     & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									 << 0x18U) 
									| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									   >> 8U))))) 
								| (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs)) 
							       | ((0x73U 
								   == 
								   (0x7fU 
								    & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									<< 0x18U) 
								       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									  >> 8U)))) 
								  & (0U 
								     == 
								     (7U 
								      & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									  << 0xcU) 
									 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									    >> 0x14U))))))
							       ? 3U
							       : 
							      ((3U 
								== 
								(0x7fU 
								 & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
								     << 0x18U) 
								    | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
								       >> 8U))))
							        ? 2U
							        : 
							       ((((((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_itype) 
								    | (0x33U 
								       == 
								       (0x7fU 
									& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									    << 0x18U) 
									   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									      >> 8U))))) 
								   | (0x37U 
								      == 
								      (0x7fU 
								       & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									   << 0x18U) 
									  | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									     >> 8U))))) 
								  | (0x17U 
								     == 
								     (0x7fU 
								      & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									  << 0x18U) 
									 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									    >> 8U))))) 
								 | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_csr))
								 ? 1U
								 : 0U))))) 
					     << 0x2aU) 
					    | (((QData)((IData)(
								(1U 
								 & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_itype) 
								     | (0x23U 
									== 
									(0x7fU 
									 & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									     << 0x18U) 
									    | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									       >> 8U)))))
								     ? 1U
								     : 0U)))) 
						<< 0x29U) 
					       | (((QData)((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_itype_immed)) 
						   << 9U) 
						  | (QData)((IData)(
								    ((0x1c0U 
								      & (((3U 
									   == 
									   (0x7fU 
									    & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
									       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))
									   ? 
									  ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									    << 0xcU) 
									   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									      >> 0x14U))
									   : 7U) 
									 << 6U)) 
								     | ((0x38U 
									 & (((0x23U 
									      == 
									      (0x7fU 
									       & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))
									      ? 
									     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									       << 0xcU) 
									      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 0x14U))
									      : 7U) 
									    << 3U)) 
									| (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type))))))))));
    __Vtemp112[1U] = ((0xfffe0000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[0U] 
				      << 0x11U)) | (IData)(
							   ((((QData)((IData)(
									      (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										>> 1U) 
										& (0x33U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U)))))
									        ? (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu)
									        : (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_final_alu)))) 
							      << 0x2cU) 
							     | (((QData)((IData)(
										(((((0x6fU 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U)))) 
										| (0x67U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))) 
										| (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs)) 
										| ((0x73U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U)))) 
										& (0U 
										== 
										(7U 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0xcU) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 0x14U))))))
										 ? 3U
										 : 
										((3U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))
										 ? 2U
										 : 
										((((((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_itype) 
										| (0x33U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))) 
										| (0x37U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))) 
										| (0x17U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))) 
										| (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_csr))
										 ? 1U
										 : 0U))))) 
								 << 0x2aU) 
								| (((QData)((IData)(
										(1U 
										& (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_itype) 
										| (0x23U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U)))))
										 ? 1U
										 : 0U)))) 
								    << 0x29U) 
								   | (((QData)((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_itype_immed)) 
								       << 9U) 
								      | (QData)((IData)(
										((0x1c0U 
										& (((3U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))
										 ? 
										((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0xcU) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 0x14U))
										 : 7U) 
										<< 6U)) 
										| ((0x38U 
										& (((0x23U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))
										 ? 
										((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0xcU) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 0x14U))
										 : 7U) 
										<< 3U)) 
										| (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type))))))))) 
							    >> 0x20U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0U] 
	= ((0xffffff00U & ((IData)((((QData)((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset)) 
				     << 0x20U) | (QData)((IData)(
								 ((IData)(4U) 
								  + 
								  ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
								    << 0x18U) 
								   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
								      >> 8U))))))) 
			   << 8U)) | ((0xf0U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
						<< 4U)) 
				      | (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						  << 0x1cU) 
						 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
						    >> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[1U] 
	= ((0xffU & ((IData)((((QData)((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset)) 
			       << 0x20U) | (QData)((IData)(
							   ((IData)(4U) 
							    + 
							    ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							      << 0x18U) 
							     | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
								>> 8U))))))) 
		     >> 0x18U)) | (0xffffff00U & ((IData)(
							  ((((QData)((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset)) 
							     << 0x20U) 
							    | (QData)((IData)(
									      ((IData)(4U) 
									       + 
									       ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
										>> 8U)))))) 
							   >> 0x20U)) 
						  << 8U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[2U] 
	= ((0xfffffe00U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
			   << 1U)) | ((0xffffff00U 
				       & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal) 
					  << 8U)) | 
				      (0xffU & ((IData)(
							((((QData)((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset)) 
							   << 0x20U) 
							  | (QData)((IData)(
									    ((IData)(4U) 
									     + 
									     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									       << 0x18U) 
									      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
										>> 8U)))))) 
							 >> 0x20U)) 
						>> 0x18U))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[3U] 
	= ((0xe0000000U & (__Vtemp112[0U] << 0x1dU)) 
	   | ((0x1ffffe00U & (((0x37U == (0x7fU & (
						   (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						    << 0x18U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						      >> 8U))))
			        ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				    << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						>> 0x14U))
			        : ((0x17U == (0x7fU 
					      & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						  << 0x18U) 
						 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						    >> 8U))))
				    ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					<< 0xcU) | 
				       (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					>> 0x14U)) : 0U)) 
			      << 9U)) | (0x1ffU & (
						   (0x1feU 
						    & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						       << 1U)) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
						      >> 0x1fU)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[4U] 
	= ((0x1fffffffU & (__Vtemp112[0U] >> 3U)) | 
	   (0xe0000000U & (__Vtemp112[1U] << 0x1dU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[5U] 
	= ((0x1fffffffU & (__Vtemp112[1U] >> 3U)) | 
	   (0xe0000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[0U] 
			   << 0xeU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[6U] 
	= ((0x1fffffffU & ((0x3fffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[0U] 
				       >> 0x12U)) | 
			   (0x1fffc000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[1U] 
					   << 0xeU)))) 
	   | (0xe0000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[1U] 
			     << 0xeU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[7U] 
	= ((0x1fffffffU & ((0x3fffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[1U] 
				       >> 0x12U)) | 
			   (0x1fffc000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[2U] 
					   << 0xeU)))) 
	   | (0xe0000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[2U] 
			     << 0xeU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[8U] 
	= ((0x1fffffffU & ((0x3fffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[2U] 
				       >> 0x12U)) | 
			   (0x1fffc000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[3U] 
					   << 0xeU)))) 
	   | (0xe0000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[3U] 
			     << 0xeU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[9U] 
	= ((0x1fffffffU & ((0x3fffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[3U] 
				       >> 0x12U)) | 
			   (0x1fffc000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U] 
					   << 0xeU)))) 
	   | (0xe0000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U] 
			     << 0xeU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xaU] 
	= ((0x1fffffffU & ((0x3fffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U] 
				       >> 0x12U)) | 
			   (0x1fffc000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[1U] 
					   << 0xeU)))) 
	   | (0xe0000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[1U] 
			     << 0xeU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xbU] 
	= ((0x1fffffffU & ((0x3fffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[1U] 
				       >> 0x12U)) | 
			   (0x1fffc000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[2U] 
					   << 0xeU)))) 
	   | (0xe0000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[2U] 
			     << 0xeU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xcU] 
	= ((0x1fffffffU & ((0x3fffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[2U] 
				       >> 0x12U)) | 
			   (0x1fffc000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[3U] 
					   << 0xeU)))) 
	   | (0xe0000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[3U] 
			     << 0xeU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xdU] 
	= ((0x1fffffffU & ((0x1f000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					   << 9U)) 
			   | ((0xf80000U & ((0x10000000U 
					     & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						<< 0x1cU)) 
					    | (0xff80000U 
					       & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						  >> 4U)))) 
			      | ((0x7c000U & ((0x1ffc0000U 
					       & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						  << 0x12U)) 
					      | (0x3c000U 
						 & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						    >> 0xeU)))) 
				 | (0x3fffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[3U] 
					       >> 0x12U)))))) 
	   | (0xe0000000U & ((((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_csr) 
			       & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				  >> 0x16U)) ? (0x1fU 
						& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						    << 9U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						      >> 0x17U)))
			       : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U]) 
			     << 0x1dU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xeU] 
	= ((0xc0000000U & ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_frE_to_bckE_req.csr_address) 
			   << 0x1eU)) | ((0xe0000000U 
					  & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_csr) 
					     << 0x1dU)) 
					 | (0x1fffffffU 
					    & ((((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_csr) 
						 & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						    >> 0x16U))
						 ? 
						(0x1fU 
						 & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						     << 9U) 
						    | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						       >> 0x17U)))
						 : 
						vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U]) 
					       >> 3U))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xfU] 
	= (0x3fffffffU & ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_frE_to_bckE_req.csr_address) 
			  >> 2U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask)) 
	   | VL_LTES_III(32,32,32, 0U, vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[0U]));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask)) 
	   | (VL_LTES_III(1,32,32, 1U, vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[0U]) 
	      << 1U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask)) 
	   | (VL_LTES_III(1,32,32, 2U, vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[0U]) 
	      << 2U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask)) 
	   | (VL_LTES_III(1,32,32, 3U, vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[0U]) 
	      << 3U));
    vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs)
	    ? (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask)
	    : (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jmprt_thread_mask));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid)) 
	   | (1U & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_change_mask)
		     ? (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask)
		     : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_stall)) 
			& (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid)) 
	   | (2U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 1U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid) 
				      >> 1U))) << 1U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid)) 
	   | (4U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 2U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid) 
				      >> 2U))) << 2U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid)) 
	   | (8U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 3U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid) 
				      >> 3U))) << 3U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid)) 
	   | (1U & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_change_mask)
		     ? (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask)
		     : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_stall)) 
			& (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid)) 
	   | (2U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 1U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid) 
				      >> 1U))) << 1U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid)) 
	   | (4U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 2U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid) 
				      >> 2U))) << 2U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid)) 
	   | (8U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 3U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid) 
				      >> 3U))) << 3U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid)) 
	   | (1U & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_change_mask)
		     ? (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask)
		     : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_stall)) 
			& (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid)) 
	   | (2U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 1U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid) 
				      >> 1U))) << 1U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid)) 
	   | (4U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 2U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid) 
				      >> 2U))) << 2U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid)) 
	   | (8U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 3U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid) 
				      >> 3U))) << 3U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid)) 
	   | (1U & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_change_mask)
		     ? (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask)
		     : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_stall)) 
			& (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid)) 
	   | (2U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 1U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid) 
				      >> 1U))) << 1U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid)) 
	   | (4U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 2U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid) 
				      >> 2U))) << 2U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid)) 
	   | (8U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 3U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid) 
				      >> 3U))) << 3U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid)) 
	   | (1U & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_change_mask)
		     ? (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask)
		     : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_stall)) 
			& (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid)) 
	   | (2U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 1U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid) 
				      >> 1U))) << 1U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid)) 
	   | (4U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 2U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid) 
				      >> 2U))) << 2U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid)) 
	   | (8U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 3U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid) 
				      >> 3U))) << 3U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid)) 
	   | (1U & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_change_mask)
		     ? (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask)
		     : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_stall)) 
			& (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid)) 
	   | (2U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 1U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid) 
				      >> 1U))) << 1U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid)) 
	   | (4U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 2U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid) 
				      >> 2U))) << 2U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid)) 
	   | (8U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 3U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid) 
				      >> 3U))) << 3U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid)) 
	   | (1U & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_change_mask)
		     ? (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask)
		     : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_stall)) 
			& (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid)) 
	   | (2U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 1U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid) 
				      >> 1U))) << 1U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid)) 
	   | (4U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 2U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid) 
				      >> 2U))) << 2U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid)) 
	   | (8U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 3U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid) 
				      >> 3U))) << 3U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid)) 
	   | (1U & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_change_mask)
		     ? (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask)
		     : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_stall)) 
			& (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid)) 
	   | (2U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 1U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid) 
				      >> 1U))) << 1U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid)) 
	   | (4U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 2U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid) 
				      >> 2U))) << 2U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid)) 
	   | (8U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 3U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid) 
				      >> 3U))) << 3U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
	= ((0xfffffff0U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid) 
	   | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
	= ((0xffffff0fU & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid) 
	   | ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid) 
	      << 4U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
	= ((0xfffff0ffU & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid) 
	   | ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid) 
	      << 8U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
	= ((0xffff0fffU & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid) 
	   | ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid) 
	      << 0xcU));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
	= ((0xfff0ffffU & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid) 
	   | ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid) 
	      << 0x10U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
	= ((0xff0fffffU & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid) 
	   | ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid) 
	      << 0x14U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
	= ((0xf0ffffffU & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid) 
	   | ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid) 
	      << 0x18U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
	= ((0xfffffffU & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid) 
	   | ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid) 
	      << 0x1cU));
    // ALWAYS at VX_fetch.v:152
    if ((0U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (1U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (2U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (4U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (8U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid));
    }
    if ((1U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 4U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (2U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 4U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (4U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 4U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 4U)));
    }
    if ((2U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 8U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (2U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 8U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (4U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 8U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 8U)));
    }
    if ((3U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0xcU)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (2U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0xcU)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (4U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0xcU)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0xcU)));
    }
    if ((4U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x10U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (2U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x10U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (4U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x10U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x10U)));
    }
    if ((5U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x14U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (2U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x14U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (4U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x14U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x14U)));
    }
    if ((6U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x18U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (2U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x18U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (4U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x18U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x18U)));
    }
    if ((7U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x1cU)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (2U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x1cU)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (4U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x1cU)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x1cU)));
    }
    vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd.valid 
	= ((0xeU & (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd.valid)) 
	   | (1U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)));
    vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd.valid 
	= ((0xdU & (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd.valid)) 
	   | (2U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)));
    vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd.valid 
	= ((0xbU & (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd.valid)) 
	   | (4U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)));
    vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd.valid 
	= ((7U & (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd.valid)) 
	   | (8U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)));
}

VL_INLINE_OPT void VVortex::_combo__TOP__10(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_combo__TOP__10\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[0U] 
	= vlSymsp->TOP__Vortex__DOT__VX_dcache_rsp.in_cache_driver_out_data
	[0U];
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[1U] 
	= vlSymsp->TOP__Vortex__DOT__VX_dcache_rsp.in_cache_driver_out_data
	[1U];
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[2U] 
	= vlSymsp->TOP__Vortex__DOT__VX_dcache_rsp.in_cache_driver_out_data
	[2U];
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[3U] 
	= vlSymsp->TOP__Vortex__DOT__VX_dcache_rsp.in_cache_driver_out_data
	[3U];
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__new_warp_num 
	= (0xfU & (((((IData)(vlTOPp->reset) | ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num) 
						>= (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_state))) 
		     | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp)) 
		    | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__add_warp))
		    ? 0U : ((IData)(1U) + (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))));
}

VL_INLINE_OPT void VVortex::_multiclk__TOP__11(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_multiclk__TOP__11\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    // Begin mtask footprint  all: 
    VL_SIGW(__Vtemp127,223,0,7);
    VL_SIGW(__Vtemp136,479,0,15);
    // Body
    // ALWAYS at VX_alu.v:48
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__0__KET____DOT__vx_alu__out_alu_result 
	= ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	    ? ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	        ? 0U : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			 ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			     ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
					   >> 0xeU))
				     : VL_MODDIV_III(32, 
						     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						       << 0x12U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
							 >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2))
				 : ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
					   >> 0xeU))
				     : VL_MODDIVS_III(32, 
						      ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
							<< 0x12U) 
						       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
							  >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)))
			     : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? 0xffffffffU : 
				    VL_DIV_III(32, 
					       ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						 << 0x12U) 
						| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						   >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2))
				 : ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? 0xffffffffU : 
				    VL_DIVS_III(32, 
						((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						  << 0x12U) 
						 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						    >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2))))
			 : ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			     ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? (IData)((((QData)((IData)(
							     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
							       << 0x12U) 
							      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
								 >> 0xeU)))) 
					     * (QData)((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2))) 
					    >> 0x20U))
				 : (IData)((((((QData)((IData)(
							       VL_NEGATE_I((IData)(
										(1U 
										& (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
										>> 0xdU)))))) 
					       << 0x20U) 
					      | (QData)((IData)(
								((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
								  << 0x12U) 
								 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
								    >> 0xeU))))) 
					     * (QData)((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2))) 
					    >> 0x20U)))
			     : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? (IData)((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__mult_signed_result 
					    >> 0x20U))
				 : (IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__mult_signed_result)))))
	    : ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	        ? ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		    ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? vlTOPp->Vortex__DOT__csr_decode_csr_data
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? vlTOPp->Vortex__DOT__csr_decode_csr_data
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
				 << 0x17U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
					      >> 9U)) 
			       + (0xfffff000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
						 << 3U)))))
		    : ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (0xfffff000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
					      << 3U))
			    : ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
				  << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
					       >> 0xeU)) 
				>= vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
			        ? 0U : 0xffffffffU))
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2 
			       & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
				   << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						>> 0xeU)))
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
					      >> 0xeU)) 
			       | vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2))))
	        : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		    ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? VL_SHIFTRS_III(32,32,5, 
					     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					       << 0x12U) 
					      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						 >> 0xeU)), 
					     (0x1fU 
					      & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2))
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
					      >> 0xeU)) 
			       >> (0x1fU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)))
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
					      >> 0xeU)) 
			       ^ vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
			    : ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
				  << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
					       >> 0xeU)) 
				< vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
			        ? 1U : 0U))) : ((0x400U 
						 & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						 ? 
						((0x200U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						  ? 
						 (VL_LTS_III(1,32,32, 
							     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
							       << 0x12U) 
							      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
								 >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
						   ? 1U
						   : 0U)
						  : 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						      >> 0xeU)) 
						  << 
						  (0x1fU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)))
						 : 
						((0x200U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						  ? 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						      >> 0xeU)) 
						  - vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2)
						  : 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						      >> 0xeU)) 
						  + vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2))))));
    // ALWAYS at VX_alu.v:48
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__1__KET____DOT__vx_alu__out_alu_result 
	= ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	    ? ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	        ? 0U : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			 ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			     ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					   >> 0xeU))
				     : VL_MODDIV_III(32, 
						     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						       << 0x12U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
							 >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2))
				 : ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					   >> 0xeU))
				     : VL_MODDIVS_III(32, 
						      ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
							<< 0x12U) 
						       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
							  >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)))
			     : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? 0xffffffffU : 
				    VL_DIV_III(32, 
					       ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						 << 0x12U) 
						| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						   >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2))
				 : ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? 0xffffffffU : 
				    VL_DIVS_III(32, 
						((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						  << 0x12U) 
						 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						    >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2))))
			 : ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			     ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? (IData)((((QData)((IData)(
							     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
							       << 0x12U) 
							      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
								 >> 0xeU)))) 
					     * (QData)((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2))) 
					    >> 0x20U))
				 : (IData)((((((QData)((IData)(
							       VL_NEGATE_I((IData)(
										(1U 
										& (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
										>> 0xdU)))))) 
					       << 0x20U) 
					      | (QData)((IData)(
								((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
								  << 0x12U) 
								 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
								    >> 0xeU))))) 
					     * (QData)((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2))) 
					    >> 0x20U)))
			     : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? (IData)((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__mult_signed_result 
					    >> 0x20U))
				 : (IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__mult_signed_result)))))
	    : ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	        ? ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		    ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? vlTOPp->Vortex__DOT__csr_decode_csr_data
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? vlTOPp->Vortex__DOT__csr_decode_csr_data
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
				 << 0x17U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
					      >> 9U)) 
			       + (0xfffff000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
						 << 3U)))))
		    : ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (0xfffff000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
					      << 3U))
			    : ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
				  << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					       >> 0xeU)) 
				>= vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
			        ? 0U : 0xffffffffU))
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2 
			       & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
				   << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						>> 0xeU)))
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					      >> 0xeU)) 
			       | vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2))))
	        : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		    ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? VL_SHIFTRS_III(32,32,5, 
					     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					       << 0x12U) 
					      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						 >> 0xeU)), 
					     (0x1fU 
					      & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2))
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					      >> 0xeU)) 
			       >> (0x1fU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)))
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					      >> 0xeU)) 
			       ^ vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
			    : ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
				  << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
					       >> 0xeU)) 
				< vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
			        ? 1U : 0U))) : ((0x400U 
						 & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						 ? 
						((0x200U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						  ? 
						 (VL_LTS_III(1,32,32, 
							     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
							       << 0x12U) 
							      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
								 >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
						   ? 1U
						   : 0U)
						  : 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						      >> 0xeU)) 
						  << 
						  (0x1fU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)))
						 : 
						((0x200U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						  ? 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						      >> 0xeU)) 
						  - vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2)
						  : 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						      >> 0xeU)) 
						  + vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2))))));
    // ALWAYS at VX_alu.v:48
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__2__KET____DOT__vx_alu__out_alu_result 
	= ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	    ? ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	        ? 0U : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			 ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			     ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					   >> 0xeU))
				     : VL_MODDIV_III(32, 
						     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						       << 0x12U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
							 >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2))
				 : ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					   >> 0xeU))
				     : VL_MODDIVS_III(32, 
						      ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
							<< 0x12U) 
						       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
							  >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)))
			     : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? 0xffffffffU : 
				    VL_DIV_III(32, 
					       ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						 << 0x12U) 
						| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						   >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2))
				 : ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? 0xffffffffU : 
				    VL_DIVS_III(32, 
						((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						  << 0x12U) 
						 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						    >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2))))
			 : ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			     ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? (IData)((((QData)((IData)(
							     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
							       << 0x12U) 
							      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
								 >> 0xeU)))) 
					     * (QData)((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2))) 
					    >> 0x20U))
				 : (IData)((((((QData)((IData)(
							       VL_NEGATE_I((IData)(
										(1U 
										& (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
										>> 0xdU)))))) 
					       << 0x20U) 
					      | (QData)((IData)(
								((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
								  << 0x12U) 
								 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
								    >> 0xeU))))) 
					     * (QData)((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2))) 
					    >> 0x20U)))
			     : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? (IData)((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__mult_signed_result 
					    >> 0x20U))
				 : (IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__mult_signed_result)))))
	    : ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	        ? ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		    ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? vlTOPp->Vortex__DOT__csr_decode_csr_data
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? vlTOPp->Vortex__DOT__csr_decode_csr_data
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
				 << 0x17U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
					      >> 9U)) 
			       + (0xfffff000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
						 << 3U)))))
		    : ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (0xfffff000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
					      << 3U))
			    : ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
				  << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					       >> 0xeU)) 
				>= vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
			        ? 0U : 0xffffffffU))
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2 
			       & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
				   << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						>> 0xeU)))
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					      >> 0xeU)) 
			       | vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2))))
	        : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		    ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? VL_SHIFTRS_III(32,32,5, 
					     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					       << 0x12U) 
					      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						 >> 0xeU)), 
					     (0x1fU 
					      & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2))
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					      >> 0xeU)) 
			       >> (0x1fU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)))
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					      >> 0xeU)) 
			       ^ vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
			    : ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
				  << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
					       >> 0xeU)) 
				< vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
			        ? 1U : 0U))) : ((0x400U 
						 & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						 ? 
						((0x200U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						  ? 
						 (VL_LTS_III(1,32,32, 
							     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
							       << 0x12U) 
							      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
								 >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
						   ? 1U
						   : 0U)
						  : 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						      >> 0xeU)) 
						  << 
						  (0x1fU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)))
						 : 
						((0x200U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						  ? 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						      >> 0xeU)) 
						  - vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2)
						  : 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xbU] 
						      >> 0xeU)) 
						  + vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2))))));
    // ALWAYS at VX_alu.v:48
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__3__KET____DOT__vx_alu__out_alu_result 
	= ((0x2000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	    ? ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	        ? 0U : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			 ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			     ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					   >> 0xeU))
				     : VL_MODDIV_III(32, 
						     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						       << 0x12U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
							 >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2))
				 : ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
					 << 0x12U) 
					| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					   >> 0xeU))
				     : VL_MODDIVS_III(32, 
						      ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
							<< 0x12U) 
						       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
							  >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)))
			     : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? 0xffffffffU : 
				    VL_DIV_III(32, 
					       ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						 << 0x12U) 
						| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						   >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2))
				 : ((0U == vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
				     ? 0xffffffffU : 
				    VL_DIVS_III(32, 
						((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						  << 0x12U) 
						 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						    >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2))))
			 : ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			     ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? (IData)((((QData)((IData)(
							     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
							       << 0x12U) 
							      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
								 >> 0xeU)))) 
					     * (QData)((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2))) 
					    >> 0x20U))
				 : (IData)((((((QData)((IData)(
							       VL_NEGATE_I((IData)(
										(1U 
										& (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
										>> 0xdU)))))) 
					       << 0x20U) 
					      | (QData)((IData)(
								((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
								  << 0x12U) 
								 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
								    >> 0xeU))))) 
					     * (QData)((IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2))) 
					    >> 0x20U)))
			     : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
				 ? (IData)((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__mult_signed_result 
					    >> 0x20U))
				 : (IData)(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__mult_signed_result)))))
	    : ((0x1000U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
	        ? ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		    ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? vlTOPp->Vortex__DOT__csr_decode_csr_data
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? vlTOPp->Vortex__DOT__csr_decode_csr_data
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
				 << 0x17U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
					      >> 9U)) 
			       + (0xfffff000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
						 << 3U)))))
		    : ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (0xfffff000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
					      << 3U))
			    : ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
				  << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					       >> 0xeU)) 
				>= vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
			        ? 0U : 0xffffffffU))
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2 
			       & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
				   << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						>> 0xeU)))
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					      >> 0xeU)) 
			       | vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2))))
	        : ((0x800U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		    ? ((0x400U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
		        ? ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? VL_SHIFTRS_III(32,32,5, 
					     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
					       << 0x12U) 
					      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						 >> 0xeU)), 
					     (0x1fU 
					      & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2))
			    : (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					      >> 0xeU)) 
			       >> (0x1fU & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)))
		        : ((0x200U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
			    ? (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
				 << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					      >> 0xeU)) 
			       ^ vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
			    : ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
				  << 0x12U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
					       >> 0xeU)) 
				< vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
			        ? 1U : 0U))) : ((0x400U 
						 & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						 ? 
						((0x200U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						  ? 
						 (VL_LTS_III(1,32,32, 
							     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
							       << 0x12U) 
							      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
								 >> 0xeU)), vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
						   ? 1U
						   : 0U)
						  : 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						      >> 0xeU)) 
						  << 
						  (0x1fU 
						   & vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)))
						 : 
						((0x200U 
						  & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U])
						  ? 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						      >> 0xeU)) 
						  - vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2)
						  : 
						 (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						    << 0x12U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xcU] 
						      >> 0xeU)) 
						  + vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2))))));
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[0U] 
	= vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__0__KET____DOT__vx_alu__out_alu_result;
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[1U] 
	= vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__1__KET____DOT__vx_alu__out_alu_result;
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[2U] 
	= vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__2__KET____DOT__vx_alu__out_alu_result;
    vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[3U] 
	= vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__3__KET____DOT__vx_alu__out_alu_result;
    __Vtemp127[5U] = ((0xfff80000U & (((0xdU == (0x1fU 
						 & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
						     << 0x17U) 
						    | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
						       >> 9U))))
				        ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
					    << 3U) 
					   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
					      >> 0x1dU))
				        : ((0xeU == 
					    (0x1fU 
					     & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
						 << 0x17U) 
						| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
						   >> 9U))))
					    ? (vlTOPp->Vortex__DOT__csr_decode_csr_data 
					       | ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
						   << 3U) 
						  | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						     >> 0x1dU)))
					    : ((0xfU 
						== 
						(0x1fU 
						 & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
						     << 0x17U) 
						    | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
						       >> 9U))))
					        ? (vlTOPp->Vortex__DOT__csr_decode_csr_data 
						   & ((IData)(0xffffffffU) 
						      - 
						      ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
							<< 3U) 
						       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
							  >> 0x1dU))))
					        : 0xdeadbeefU))) 
				      << 0x13U)) | 
		      ((0x40000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
				    << 0xaU)) | (0x3ffffU 
						 & ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						       << 0x12U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
							 >> 0xeU)) 
						     + 
						     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
						       << 0x18U) 
						      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[1U] 
							 >> 8U))) 
						    >> 0xeU))));
    __Vtemp127[6U] = (0x7ffffU & (((0xdU == (0x1fU 
					     & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
						 << 0x17U) 
						| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
						   >> 9U))))
				    ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
					<< 3U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						  >> 0x1dU))
				    : ((0xeU == (0x1fU 
						 & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
						     << 0x17U) 
						    | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
						       >> 9U))))
				        ? (vlTOPp->Vortex__DOT__csr_decode_csr_data 
					   | ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
					       << 3U) 
					      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						 >> 0x1dU)))
				        : ((0xfU == 
					    (0x1fU 
					     & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
						 << 0x17U) 
						| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
						   >> 9U))))
					    ? (vlTOPp->Vortex__DOT__csr_decode_csr_data 
					       & ((IData)(0xffffffffU) 
						  - 
						  ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
						    << 3U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						      >> 0x1dU))))
					    : 0xdeadbeefU))) 
				  >> 0xdU));
    __Vtemp136[7U] = ((0xffff0000U & ((0x80000000U 
				       & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
					  << 0x1cU)) 
				      | ((0x70000000U 
					  & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
					     << 0x1cU)) 
					 | ((0xf800000U 
					     & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
						 << 0x1fU) 
						| (0x7f800000U 
						   & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						      >> 1U)))) 
					    | (((IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb) 
						<< 0x15U) 
					       | (0x1f0000U 
						  & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
						      << 0x1dU) 
						     | (0x1fff0000U 
							& (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
							   >> 3U))))))))) 
		      | ((0xf800U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
				      << 0x1dU) | (0x1ffff800U 
						   & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xdU] 
						      >> 3U)))) 
			 | (0x7ffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
				      >> 3U))));
    __Vtemp136[8U] = ((0xffffU & ((0xfffcU & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[0U] 
					      << 2U)) 
				  | ((3U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
					    >> 4U)) 
				     | (0xffffU & ((IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb) 
						   >> 0xbU))))) 
		      | (0xffff0000U & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[0U] 
					<< 2U)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0U] 
	= ((0xfffff800U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
			   << 5U)) | ((0x700U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
						  << 0xbU) 
						 | (0x700U 
						    & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
						       >> 0x15U)))) 
				      | (0xffU & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U])));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[1U] 
	= ((0x7ffU & ((0x7e0U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
				 << 5U)) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[4U] 
					    >> 0x1bU))) 
	   | (0xfffff800U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
			     << 2U)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[2U] 
	= ((0x7ffU & ((0x7fcU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[3U] 
				 << 2U)) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
					    >> 0x1eU))) 
	   | (0xfffff800U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U] 
			     << 3U)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[3U] 
	= ((0x7ffU & ((0x7f8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[1U] 
				 << 3U)) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0U] 
					    >> 0x1dU))) 
	   | (0xfffff800U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
			      << 0x1dU) | (0x1ffff800U 
					   & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[5U] 
					      >> 3U)))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[4U] 
	= ((0x7ffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
		      >> 3U)) | (0xfffff800U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[7U] 
						 << 0x1dU) 
						| (0x1ffff800U 
						   & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[6U] 
						      >> 3U)))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[5U] 
	= ((0x7ffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[7U] 
		      >> 3U)) | (0xfffff800U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[8U] 
						 << 0x1dU) 
						| (0x1ffff800U 
						   & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[7U] 
						      >> 3U)))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[6U] 
	= ((0x7ffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[8U] 
		      >> 3U)) | (0xfffff800U & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						 << 0x1dU) 
						| (0x1ffff800U 
						   & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[8U] 
						      >> 3U)))));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[7U] 
	= __Vtemp136[7U];
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[8U] 
	= __Vtemp136[8U];
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[9U] 
	= ((0xffffU & ((3U & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[0U] 
			      >> 0x1eU)) | (0xfffcU 
					    & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[1U] 
					       << 2U)))) 
	   | (0xffff0000U & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[1U] 
			     << 2U)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0xaU] 
	= ((0xffffU & ((3U & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[1U] 
			      >> 0x1eU)) | (0xfffcU 
					    & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[2U] 
					       << 2U)))) 
	   | (0xffff0000U & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[2U] 
			     << 2U)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0xbU] 
	= ((0xffffU & ((3U & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[2U] 
			      >> 0x1eU)) | (0xfffcU 
					    & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[3U] 
					       << 2U)))) 
	   | (0xffff0000U & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[3U] 
			     << 2U)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0xcU] 
	= ((0xffffU & ((3U & (vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[3U] 
			      >> 0x1eU)) | (0xfffcU 
					    & ((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xaU] 
						  << 0x12U) 
						 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[9U] 
						    >> 0xeU)) 
						+ (
						   (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[2U] 
						    << 0x18U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[1U] 
						      >> 8U))) 
					       << 2U)))) 
	   | (0xffff0000U & (__Vtemp127[5U] << 0x10U)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0xdU] 
	= ((0xffffU & (__Vtemp127[5U] >> 0x10U)) | 
	   (0xffff0000U & (__Vtemp127[6U] << 0x10U)));
    vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in[0xeU] 
	= (0xffffU & ((0xfff0U & ((0xffc0U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xfU] 
					      << 6U)) 
				  | (0x30U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
					      >> 0x1aU)))) 
		      | ((8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value[0xeU] 
				>> 0x1aU)) | (__Vtemp127[6U] 
					      >> 0x10U))));
}

VL_INLINE_OPT void VVortex::_combo__TOP__12(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_combo__TOP__12\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    // Begin mtask footprint  all: 
    VL_SIGW(__Vtemp151,127,0,4);
    VL_SIGW(__Vtemp172,127,0,4);
    VL_SIGW(__Vtemp178,319,0,10);
    // Body
    __Vtemp151[0U] = ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)
		       ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					 << 0xbU) | 
					(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					 >> 0x15U))))
			   ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[0U]
			   : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					     << 0xbU) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					       >> 0x15U))))
			       ? vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[0U]
			       : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
				   << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
						>> 2U))))
		       : ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd)
			   ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 8U))))
			       ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[0U]
			       : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						 << 0x18U) 
						| (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						   >> 8U))))
				   ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				       << 0x11U) | 
				      (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0xfU)) : 
				  ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[6U] 
				    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[5U] 
						 >> 0xfU))))
			   : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[0U]));
    __Vtemp151[1U] = ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)
		       ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					 << 0xbU) | 
					(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					 >> 0x15U))))
			   ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[1U]
			   : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					     << 0xbU) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					       >> 0x15U))))
			       ? vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[1U]
			       : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xaU] 
				   << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
						>> 2U))))
		       : ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd)
			   ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 8U))))
			       ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[1U]
			       : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						 << 0x18U) 
						| (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						   >> 8U))))
				   ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[3U] 
				       << 0x11U) | 
				      (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				       >> 0xfU)) : 
				  ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[7U] 
				    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[6U] 
						 >> 0xfU))))
			   : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[1U]));
    __Vtemp151[2U] = ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)
		       ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					 << 0xbU) | 
					(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					 >> 0x15U))))
			   ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[2U]
			   : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					     << 0xbU) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					       >> 0x15U))))
			       ? vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[2U]
			       : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xbU] 
				   << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xaU] 
						>> 2U))))
		       : ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd)
			   ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 8U))))
			       ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[2U]
			       : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						 << 0x18U) 
						| (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						   >> 8U))))
				   ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[4U] 
				       << 0x11U) | 
				      (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[3U] 
				       >> 0xfU)) : 
				  ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[8U] 
				    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[7U] 
						 >> 0xfU))))
			   : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[2U]));
    __Vtemp151[3U] = ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)
		       ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					 << 0xbU) | 
					(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					 >> 0x15U))))
			   ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[3U]
			   : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					     << 0xbU) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					       >> 0x15U))))
			       ? vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[3U]
			       : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
				   << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xbU] 
						>> 2U))))
		       : ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd)
			   ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 8U))))
			       ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[3U]
			       : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						 << 0x18U) 
						| (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						   >> 8U))))
				   ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[5U] 
				       << 0x11U) | 
				      (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[4U] 
				       >> 0xfU)) : 
				  ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[9U] 
				    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[8U] 
						 >> 0xfU))))
			   : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[3U]));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U] 
	= ((0x6fU == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U))))
	    ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data[0U]
	    : ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd) 
		 | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)) 
		| (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd))
	        ? ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd)
		    ? ((3U == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))
		        ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[0U]
		        : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[0U])
		    : __Vtemp151[0U]) : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[
	       (0x1cU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
			 >> 2U))]));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[1U] 
	= ((0x6fU == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U))))
	    ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data[1U]
	    : ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd) 
		 | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)) 
		| (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd))
	        ? ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd)
		    ? ((3U == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))
		        ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[1U]
		        : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[1U])
		    : __Vtemp151[1U]) : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[
	       ((IData)(1U) + (0x1cU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					>> 2U)))]));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[2U] 
	= ((0x6fU == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U))))
	    ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data[2U]
	    : ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd) 
		 | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)) 
		| (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd))
	        ? ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd)
		    ? ((3U == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))
		        ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[2U]
		        : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[2U])
		    : __Vtemp151[2U]) : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[
	       ((IData)(2U) + (0x1cU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					>> 2U)))]));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[3U] 
	= ((0x6fU == (0x7fU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				<< 0x18U) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 8U))))
	    ? vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data[3U]
	    : ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd) 
		 | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd)) 
		| (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd))
	        ? ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd)
		    ? ((3U == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))
		        ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[3U]
		        : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[3U])
		    : __Vtemp151[3U]) : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data[
	       ((IData)(3U) + (0x1cU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					>> 2U)))]));
    __Vtemp172[0U] = ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)
		       ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					 << 0xbU) | 
					(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					 >> 0x15U))))
			   ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[0U]
			   : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					     << 0xbU) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					       >> 0x15U))))
			       ? vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[0U]
			       : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
				   << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
						>> 2U))))
		       : ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd)
			   ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 8U))))
			       ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[0U]
			       : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						 << 0x18U) 
						| (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						   >> 8U))))
				   ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				       << 0x11U) | 
				      (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0xfU)) : 
				  ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[6U] 
				    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[5U] 
						 >> 0xfU))))
			   : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[0U]));
    __Vtemp172[1U] = ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)
		       ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					 << 0xbU) | 
					(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					 >> 0x15U))))
			   ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[1U]
			   : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					     << 0xbU) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					       >> 0x15U))))
			       ? vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[1U]
			       : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xaU] 
				   << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[9U] 
						>> 2U))))
		       : ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd)
			   ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 8U))))
			       ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[1U]
			       : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						 << 0x18U) 
						| (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						   >> 8U))))
				   ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[3U] 
				       << 0x11U) | 
				      (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				       >> 0xfU)) : 
				  ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[7U] 
				    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[6U] 
						 >> 0xfU))))
			   : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[1U]));
    __Vtemp172[2U] = ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)
		       ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					 << 0xbU) | 
					(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					 >> 0x15U))))
			   ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[2U]
			   : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					     << 0xbU) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					       >> 0x15U))))
			       ? vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[2U]
			       : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xbU] 
				   << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xaU] 
						>> 2U))))
		       : ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd)
			   ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 8U))))
			       ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[2U]
			       : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						 << 0x18U) 
						| (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						   >> 8U))))
				   ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[4U] 
				       << 0x11U) | 
				      (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[3U] 
				       >> 0xfU)) : 
				  ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[8U] 
				    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[7U] 
						 >> 0xfU))))
			   : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[2U]));
    __Vtemp172[3U] = ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)
		       ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					 << 0xbU) | 
					(vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					 >> 0x15U))))
			   ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[3U]
			   : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[8U] 
					     << 0xbU) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[7U] 
					       >> 0x15U))))
			       ? vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_mem_wb.mem_result[3U]
			       : ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xcU] 
				   << 0x1eU) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value[0xbU] 
						>> 2U))))
		       : ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd)
			   ? ((3U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					       >> 8U))))
			       ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[3U]
			       : ((2U == (3U & ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
						 << 0x18U) 
						| (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
						   >> 8U))))
				   ? ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[5U] 
				       << 0x11U) | 
				      (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[4U] 
				       >> 0xfU)) : 
				  ((vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[9U] 
				    << 0x11U) | (vlTOPp->Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[8U] 
						 >> 0xfU))))
			   : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[3U]));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[0U] 
	= ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd) 
	     | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)) 
	    | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd))
	    ? ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd)
	        ? ((3U == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))
		    ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[0U]
		    : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[0U])
	        : __Vtemp172[0U]) : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[
	   (0x1cU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
		     >> 2U))]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[1U] 
	= ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd) 
	     | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)) 
	    | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd))
	    ? ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd)
	        ? ((3U == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))
		    ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[1U]
		    : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[1U])
	        : __Vtemp172[1U]) : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[
	   ((IData)(1U) + (0x1cU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
				    >> 2U)))]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[2U] 
	= ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd) 
	     | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)) 
	    | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd))
	    ? ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd)
	        ? ((3U == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))
		    ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[2U]
		    : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[2U])
	        : __Vtemp172[2U]) : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[
	   ((IData)(2U) + (0x1cU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
				    >> 2U)))]);
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[3U] 
	= ((((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd) 
	     | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd)) 
	    | (IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd))
	    ? ((IData)(vlTOPp->Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd)
	        ? ((3U == (IData)(vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.wb))
		    ? vlTOPp->Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[3U]
		    : vlSymsp->TOP__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req.alu_result[3U])
	        : __Vtemp172[3U]) : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data[
	   ((IData)(3U) + (0x1cU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
				    >> 2U)))]);
    __Vtemp178[0U] = (IData)((((QData)((IData)((((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						  >> 1U) 
						 & (0x33U 
						    == 
						    (0x7fU 
						     & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
							 << 0x18U) 
							| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							   >> 8U)))))
						 ? (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu)
						 : (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_final_alu)))) 
			       << 0x2cU) | (((QData)((IData)(
							     (((((0x6fU 
								  == 
								  (0x7fU 
								   & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
								       << 0x18U) 
								      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									 >> 8U)))) 
								 | (0x67U 
								    == 
								    (0x7fU 
								     & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									 << 0x18U) 
									| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									   >> 8U))))) 
								| (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs)) 
							       | ((0x73U 
								   == 
								   (0x7fU 
								    & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									<< 0x18U) 
								       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									  >> 8U)))) 
								  & (0U 
								     == 
								     (7U 
								      & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									  << 0xcU) 
									 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									    >> 0x14U))))))
							       ? 3U
							       : 
							      ((3U 
								== 
								(0x7fU 
								 & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
								     << 0x18U) 
								    | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
								       >> 8U))))
							        ? 2U
							        : 
							       ((((((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_itype) 
								    | (0x33U 
								       == 
								       (0x7fU 
									& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									    << 0x18U) 
									   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									      >> 8U))))) 
								   | (0x37U 
								      == 
								      (0x7fU 
								       & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									   << 0x18U) 
									  | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									     >> 8U))))) 
								  | (0x17U 
								     == 
								     (0x7fU 
								      & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									  << 0x18U) 
									 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									    >> 8U))))) 
								 | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_csr))
								 ? 1U
								 : 0U))))) 
					     << 0x2aU) 
					    | (((QData)((IData)(
								(1U 
								 & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_itype) 
								     | (0x23U 
									== 
									(0x7fU 
									 & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									     << 0x18U) 
									    | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									       >> 8U)))))
								     ? 1U
								     : 0U)))) 
						<< 0x29U) 
					       | (((QData)((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_itype_immed)) 
						   << 9U) 
						  | (QData)((IData)(
								    ((0x1c0U 
								      & (((3U 
									   == 
									   (0x7fU 
									    & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
									       | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))
									   ? 
									  ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									    << 0xcU) 
									   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									      >> 0x14U))
									   : 7U) 
									 << 6U)) 
								     | ((0x38U 
									 & (((0x23U 
									      == 
									      (0x7fU 
									       & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))
									      ? 
									     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
									       << 0xcU) 
									      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 0x14U))
									      : 7U) 
									    << 3U)) 
									| (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type))))))))));
    __Vtemp178[1U] = ((0xfffe0000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[0U] 
				      << 0x11U)) | (IData)(
							   ((((QData)((IData)(
									      (((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										>> 1U) 
										& (0x33U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U)))))
									        ? (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu)
									        : (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_final_alu)))) 
							      << 0x2cU) 
							     | (((QData)((IData)(
										(((((0x6fU 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U)))) 
										| (0x67U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))) 
										| (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs)) 
										| ((0x73U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U)))) 
										& (0U 
										== 
										(7U 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0xcU) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 0x14U))))))
										 ? 3U
										 : 
										((3U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))
										 ? 2U
										 : 
										((((((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_itype) 
										| (0x33U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))) 
										| (0x37U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))) 
										| (0x17U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))) 
										| (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_csr))
										 ? 1U
										 : 0U))))) 
								 << 0x2aU) 
								| (((QData)((IData)(
										(1U 
										& (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_itype) 
										| (0x23U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U)))))
										 ? 1U
										 : 0U)))) 
								    << 0x29U) 
								   | (((QData)((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_itype_immed)) 
								       << 9U) 
								      | (QData)((IData)(
										((0x1c0U 
										& (((3U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))
										 ? 
										((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0xcU) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 0x14U))
										 : 7U) 
										<< 6U)) 
										| ((0x38U 
										& (((0x23U 
										== 
										(0x7fU 
										& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 8U))))
										 ? 
										((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
										<< 0xcU) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										>> 0x14U))
										 : 7U) 
										<< 3U)) 
										| (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type))))))))) 
							    >> 0x20U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0U] 
	= ((0xffffff00U & ((IData)((((QData)((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset)) 
				     << 0x20U) | (QData)((IData)(
								 ((IData)(4U) 
								  + 
								  ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
								    << 0x18U) 
								   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
								      >> 8U))))))) 
			   << 8U)) | ((0xf0U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
						<< 4U)) 
				      | (0xfU & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						  << 0x1cU) 
						 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
						    >> 4U)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[1U] 
	= ((0xffU & ((IData)((((QData)((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset)) 
			       << 0x20U) | (QData)((IData)(
							   ((IData)(4U) 
							    + 
							    ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
							      << 0x18U) 
							     | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
								>> 8U))))))) 
		     >> 0x18U)) | (0xffffff00U & ((IData)(
							  ((((QData)((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset)) 
							     << 0x20U) 
							    | (QData)((IData)(
									      ((IData)(4U) 
									       + 
									       ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
										<< 0x18U) 
										| (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
										>> 8U)))))) 
							   >> 0x20U)) 
						  << 8U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[2U] 
	= ((0xfffffe00U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
			   << 1U)) | ((0xffffff00U 
				       & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal) 
					  << 8U)) | 
				      (0xffU & ((IData)(
							((((QData)((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset)) 
							   << 0x20U) 
							  | (QData)((IData)(
									    ((IData)(4U) 
									     + 
									     ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
									       << 0x18U) 
									      | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
										>> 8U)))))) 
							 >> 0x20U)) 
						>> 0x18U))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[3U] 
	= ((0xe0000000U & (__Vtemp178[0U] << 0x1dU)) 
	   | ((0x1ffffe00U & (((0x37U == (0x7fU & (
						   (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						    << 0x18U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						      >> 8U))))
			        ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				    << 0xcU) | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						>> 0x14U))
			        : ((0x17U == (0x7fU 
					      & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						  << 0x18U) 
						 | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						    >> 8U))))
				    ? ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					<< 0xcU) | 
				       (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					>> 0x14U)) : 0U)) 
			      << 9U)) | (0x1ffU & (
						   (0x1feU 
						    & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						       << 1U)) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
						      >> 0x1fU)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[4U] 
	= ((0x1fffffffU & (__Vtemp178[0U] >> 3U)) | 
	   (0xe0000000U & (__Vtemp178[1U] << 0x1dU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[5U] 
	= ((0x1fffffffU & (__Vtemp178[1U] >> 3U)) | 
	   (0xe0000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[0U] 
			   << 0xeU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[6U] 
	= ((0x1fffffffU & ((0x3fffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[0U] 
				       >> 0x12U)) | 
			   (0x1fffc000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[1U] 
					   << 0xeU)))) 
	   | (0xe0000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[1U] 
			     << 0xeU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[7U] 
	= ((0x1fffffffU & ((0x3fffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[1U] 
				       >> 0x12U)) | 
			   (0x1fffc000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[2U] 
					   << 0xeU)))) 
	   | (0xe0000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[2U] 
			     << 0xeU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[8U] 
	= ((0x1fffffffU & ((0x3fffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[2U] 
				       >> 0x12U)) | 
			   (0x1fffc000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[3U] 
					   << 0xeU)))) 
	   | (0xe0000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[3U] 
			     << 0xeU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[9U] 
	= ((0x1fffffffU & ((0x3fffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[3U] 
				       >> 0x12U)) | 
			   (0x1fffc000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U] 
					   << 0xeU)))) 
	   | (0xe0000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U] 
			     << 0xeU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xaU] 
	= ((0x1fffffffU & ((0x3fffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U] 
				       >> 0x12U)) | 
			   (0x1fffc000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[1U] 
					   << 0xeU)))) 
	   | (0xe0000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[1U] 
			     << 0xeU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xbU] 
	= ((0x1fffffffU & ((0x3fffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[1U] 
				       >> 0x12U)) | 
			   (0x1fffc000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[2U] 
					   << 0xeU)))) 
	   | (0xe0000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[2U] 
			     << 0xeU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xcU] 
	= ((0x1fffffffU & ((0x3fffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[2U] 
				       >> 0x12U)) | 
			   (0x1fffc000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[3U] 
					   << 0xeU)))) 
	   | (0xe0000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[3U] 
			     << 0xeU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xdU] 
	= ((0x1fffffffU & ((0x1f000000U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					   << 9U)) 
			   | ((0xf80000U & ((0x10000000U 
					     & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						<< 0x1cU)) 
					    | (0xff80000U 
					       & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						  >> 4U)))) 
			      | ((0x7c000U & ((0x1ffc0000U 
					       & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						  << 0x12U)) 
					      | (0x3c000U 
						 & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						    >> 0xeU)))) 
				 | (0x3fffU & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[3U] 
					       >> 0x12U)))))) 
	   | (0xe0000000U & ((((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_csr) 
			       & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				  >> 0x16U)) ? (0x1fU 
						& ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						    << 9U) 
						   | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						      >> 0x17U)))
			       : vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U]) 
			     << 0x1dU)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xeU] 
	= ((0xc0000000U & ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_frE_to_bckE_req.csr_address) 
			   << 0x1eU)) | ((0xe0000000U 
					  & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_csr) 
					     << 0x1dU)) 
					 | (0x1fffffffU 
					    & ((((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_csr) 
						 & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						    >> 0x16U))
						 ? 
						(0x1fU 
						 & ((vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
						     << 9U) 
						    | (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
						       >> 0x17U)))
						 : 
						vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data[0U]) 
					       >> 3U))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in[0xfU] 
	= (0x3fffffffU & ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_frE_to_bckE_req.csr_address) 
			  >> 2U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask)) 
	   | VL_LTES_III(32,32,32, 0U, vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[0U]));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask)) 
	   | (VL_LTES_III(1,32,32, 1U, vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[0U]) 
	      << 1U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask)) 
	   | (VL_LTES_III(1,32,32, 2U, vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[0U]) 
	      << 2U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask)) 
	   | (VL_LTES_III(1,32,32, 3U, vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data[0U]) 
	      << 3U));
    vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask 
	= ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs)
	    ? (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask)
	    : (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jmprt_thread_mask));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid)) 
	   | (1U & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_change_mask)
		     ? (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask)
		     : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_stall)) 
			& (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid)) 
	   | (2U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 1U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid) 
				      >> 1U))) << 1U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid)) 
	   | (4U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 2U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid) 
				      >> 2U))) << 2U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid)) 
	   | (8U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 3U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid) 
				      >> 3U))) << 3U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid)) 
	   | (1U & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_change_mask)
		     ? (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask)
		     : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_stall)) 
			& (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid)) 
	   | (2U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 1U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid) 
				      >> 1U))) << 1U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid)) 
	   | (4U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 2U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid) 
				      >> 2U))) << 2U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid)) 
	   | (8U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 3U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid) 
				      >> 3U))) << 3U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid)) 
	   | (1U & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_change_mask)
		     ? (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask)
		     : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_stall)) 
			& (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid)) 
	   | (2U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 1U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid) 
				      >> 1U))) << 1U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid)) 
	   | (4U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 2U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid) 
				      >> 2U))) << 2U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid)) 
	   | (8U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 3U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid) 
				      >> 3U))) << 3U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid)) 
	   | (1U & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_change_mask)
		     ? (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask)
		     : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_stall)) 
			& (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid)) 
	   | (2U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 1U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid) 
				      >> 1U))) << 1U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid)) 
	   | (4U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 2U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid) 
				      >> 2U))) << 2U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid)) 
	   | (8U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 3U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid) 
				      >> 3U))) << 3U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid)) 
	   | (1U & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_change_mask)
		     ? (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask)
		     : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_stall)) 
			& (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid)) 
	   | (2U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 1U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid) 
				      >> 1U))) << 1U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid)) 
	   | (4U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 2U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid) 
				      >> 2U))) << 2U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid)) 
	   | (8U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 3U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid) 
				      >> 3U))) << 3U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid)) 
	   | (1U & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_change_mask)
		     ? (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask)
		     : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_stall)) 
			& (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid)) 
	   | (2U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 1U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid) 
				      >> 1U))) << 1U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid)) 
	   | (4U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 2U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid) 
				      >> 2U))) << 2U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid)) 
	   | (8U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 3U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid) 
				      >> 3U))) << 3U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid)) 
	   | (1U & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_change_mask)
		     ? (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask)
		     : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_stall)) 
			& (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid)) 
	   | (2U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 1U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid) 
				      >> 1U))) << 1U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid)) 
	   | (4U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 2U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid) 
				      >> 2U))) << 2U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid)) 
	   | (8U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 3U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid) 
				      >> 3U))) << 3U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid 
	= ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid)) 
	   | (1U & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_change_mask)
		     ? (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask)
		     : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_stall)) 
			& (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid)))));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid 
	= ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid)) 
	   | (2U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 1U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid) 
				      >> 1U))) << 1U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid 
	= ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid)) 
	   | (4U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 2U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid) 
				      >> 2U))) << 2U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid 
	= ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid)) 
	   | (8U & (((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_change_mask)
		      ? ((IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl.thread_mask) 
			 >> 3U) : ((~ (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_stall)) 
				   & ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid) 
				      >> 3U))) << 3U)));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
	= ((0xfffffff0U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid) 
	   | (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
	= ((0xffffff0fU & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid) 
	   | ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid) 
	      << 4U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
	= ((0xfffff0ffU & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid) 
	   | ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid) 
	      << 8U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
	= ((0xffff0fffU & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid) 
	   | ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid) 
	      << 0xcU));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
	= ((0xfff0ffffU & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid) 
	   | ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid) 
	      << 0x10U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
	= ((0xff0fffffU & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid) 
	   | ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid) 
	      << 0x14U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
	= ((0xf0ffffffU & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid) 
	   | ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid) 
	      << 0x18U));
    vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
	= ((0xfffffffU & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid) 
	   | ((IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid) 
	      << 0x1cU));
    // ALWAYS at VX_fetch.v:152
    if ((0U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (1U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (2U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (4U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (8U & vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid));
    }
    if ((1U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 4U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (2U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 4U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (4U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 4U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 4U)));
    }
    if ((2U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 8U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (2U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 8U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (4U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 8U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 8U)));
    }
    if ((3U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0xcU)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (2U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0xcU)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (4U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0xcU)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0xcU)));
    }
    if ((4U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x10U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (2U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x10U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (4U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x10U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x10U)));
    }
    if ((5U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x14U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (2U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x14U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (4U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x14U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x14U)));
    }
    if ((6U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x18U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (2U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x18U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (4U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x18U)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x18U)));
    }
    if ((7U == (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num))) {
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xeU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (1U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x1cU)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xdU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (2U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x1cU)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((0xbU & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (4U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x1cU)));
	vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var 
	    = ((7U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)) 
	       | (8U & (vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid 
			>> 0x1cU)));
    }
    vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd.valid 
	= ((0xeU & (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd.valid)) 
	   | (1U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)));
    vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd.valid 
	= ((0xdU & (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd.valid)) 
	   | (2U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)));
    vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd.valid 
	= ((0xbU & (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd.valid)) 
	   | (4U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)));
    vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd.valid 
	= ((7U & (IData)(vlSymsp->TOP__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd.valid)) 
	   | (8U & (IData)(vlTOPp->Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var)));
}

void VVortex::_eval(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_eval\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    if ((((IData)(vlTOPp->clk) & (~ (IData)(vlTOPp->__Vclklast__TOP__clk))) 
	 | ((IData)(vlTOPp->reset) & (~ (IData)(vlTOPp->__Vclklast__TOP__reset))))) {
	vlTOPp->_sequent__TOP__2(vlSymsp);
    }
    vlTOPp->_combo__TOP__3(vlSymsp);
    if (((IData)(vlTOPp->clk) & (~ (IData)(vlTOPp->__Vclklast__TOP__clk)))) {
	vlTOPp->_sequent__TOP__4(vlSymsp);
    }
    if (((~ (IData)(vlTOPp->clk)) & (IData)(vlTOPp->__Vclklast__TOP__clk))) {
	vlTOPp->_sequent__TOP__5(vlSymsp);
    }
    if ((((IData)(vlTOPp->clk) & (~ (IData)(vlTOPp->__Vclklast__TOP__clk))) 
	 | ((IData)(vlTOPp->reset) & (~ (IData)(vlTOPp->__Vclklast__TOP__reset))))) {
	vlTOPp->_sequent__TOP__7(vlSymsp);
    }
    if (((IData)(vlTOPp->clk) ^ (IData)(vlTOPp->__Vclklast__TOP__clk))) {
	vlTOPp->_multiclk__TOP__8(vlSymsp);
    }
    vlTOPp->_combo__TOP__10(vlSymsp);
    if ((((IData)(vlTOPp->clk) ^ (IData)(vlTOPp->__Vclklast__TOP__clk)) 
	 | ((IData)(vlTOPp->reset) & (~ (IData)(vlTOPp->__Vclklast__TOP__reset))))) {
	vlTOPp->_multiclk__TOP__11(vlSymsp);
    }
    vlTOPp->_combo__TOP__12(vlSymsp);
    // Final
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
    vlTOPp->__Vclklast__TOP__reset = vlTOPp->reset;
}

void VVortex::_eval_initial(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_eval_initial\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
    vlTOPp->__Vclklast__TOP__reset = vlTOPp->reset;
    vlTOPp->_initial__TOP__6(vlSymsp);
}

void VVortex::final() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::final\n"); );
    // Variables
    VVortex__Syms* __restrict vlSymsp = this->__VlSymsp;
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void VVortex::_eval_settle(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_eval_settle\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->_settle__TOP__1(vlSymsp);
    vlTOPp->_settle__TOP__9(vlSymsp);
}

VL_INLINE_OPT QData VVortex::_change_request(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_change_request\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // Change detection
    QData __req = false;  // Logically a bool
    return __req;
}

#ifdef VL_DEBUG
void VVortex::_eval_debug_assertions() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((clk & 0xfeU))) {
	Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((reset & 0xfeU))) {
	Verilated::overWidthError("reset");}
}
#endif // VL_DEBUG

void VVortex::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVortex::_ctor_var_reset\n"); );
    // Body
    clk = VL_RAND_RESET_I(1);
    reset = VL_RAND_RESET_I(1);
    icache_response_instruction = VL_RAND_RESET_I(32);
    icache_request_pc_address = VL_RAND_RESET_I(32);
    { int __Vi0=0; for (; __Vi0<4; ++__Vi0) {
	    in_cache_driver_out_data[__Vi0] = VL_RAND_RESET_I(32);
    }}
    { int __Vi0=0; for (; __Vi0<4; ++__Vi0) {
	    out_cache_driver_in_address[__Vi0] = VL_RAND_RESET_I(32);
    }}
    out_cache_driver_in_mem_read = VL_RAND_RESET_I(3);
    out_cache_driver_in_mem_write = VL_RAND_RESET_I(3);
    { int __Vi0=0; for (; __Vi0<4; ++__Vi0) {
	    out_cache_driver_in_valid[__Vi0] = VL_RAND_RESET_I(1);
    }}
    { int __Vi0=0; for (; __Vi0<4; ++__Vi0) {
	    out_cache_driver_in_data[__Vi0] = VL_RAND_RESET_I(32);
    }}
    out_ebreak = VL_RAND_RESET_I(1);
    Vortex__DOT__execute_branch_stall = VL_RAND_RESET_I(1);
    Vortex__DOT__forwarding_fwd_stall = VL_RAND_RESET_I(1);
    Vortex__DOT__csr_decode_csr_data = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_state = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_count = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__add_warp = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__new_warp_state = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__new_warp_count = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__new_warp_num = VL_RAND_RESET_I(4);
    VL_RAND_RESET_W(256,Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_change_mask = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_stall = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_change_mask = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_stall = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_change_mask = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_stall = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_change_mask = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_stall = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_change_mask = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_stall = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_change_mask = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_stall = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_change_mask = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_stall = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_change_mask = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_stall = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__real_PC = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__temp_PC = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid_zero = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__real_PC = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__temp_PC = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid_zero = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__real_PC = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__temp_PC = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid_zero = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__real_PC = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__temp_PC = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid_zero = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__real_PC = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__temp_PC = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid_zero = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__real_PC = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__temp_PC = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid_zero = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__real_PC = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__temp_PC = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid_zero = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__real_PC = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__temp_PC = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid_zero = VL_RAND_RESET_I(4);
    VL_RAND_RESET_W(72,Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value);
    { int __Vi0=0; for (; __Vi0<4; ++__Vi0) {
	    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid[__Vi0] = VL_RAND_RESET_I(1);
    }}
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_itype = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_csr = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jmprt = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_wspawn = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jal_sys_jal = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__alu_tempp = VL_RAND_RESET_I(12);
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu = VL_RAND_RESET_I(5);
    VL_RAND_RESET_W(128,Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data);
    VL_RAND_RESET_W(128,Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data);
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jmprt_thread_mask = VL_RAND_RESET_I(4);
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_jal_offset = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_ebreak = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_itype_immed = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_type = VL_RAND_RESET_I(3);
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_branch_stall = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_final_alu = VL_RAND_RESET_I(5);
    VL_RAND_RESET_W(1024,Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data);
    VL_RAND_RESET_W(1024,Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data);
    VL_RAND_RESET_W(128,Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data);
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__write_enable = VL_RAND_RESET_I(1);
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
	    VL_RAND_RESET_W(128,Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR[__Vi0]);
    }}
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__write_enable = VL_RAND_RESET_I(1);
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
	    VL_RAND_RESET_W(128,Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR[__Vi0]);
    }}
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__write_enable = VL_RAND_RESET_I(1);
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
	    VL_RAND_RESET_W(128,Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR[__Vi0]);
    }}
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__write_enable = VL_RAND_RESET_I(1);
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
	    VL_RAND_RESET_W(128,Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR[__Vi0]);
    }}
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__write_enable = VL_RAND_RESET_I(1);
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
	    VL_RAND_RESET_W(128,Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR[__Vi0]);
    }}
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__write_enable = VL_RAND_RESET_I(1);
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
	    VL_RAND_RESET_W(128,Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR[__Vi0]);
    }}
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__write_enable = VL_RAND_RESET_I(1);
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
	    VL_RAND_RESET_W(128,Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR[__Vi0]);
    }}
    Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__write_enable = VL_RAND_RESET_I(1);
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
	    VL_RAND_RESET_W(128,Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__first_ram__DOT__GPR[__Vi0]);
    }}
    VL_RAND_RESET_W(490,Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in);
    VL_RAND_RESET_W(490,Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value);
    Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__0__KET____DOT__vx_alu__out_alu_result = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__1__KET____DOT__vx_alu__out_alu_result = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__2__KET____DOT__vx_alu__out_alu_result = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__3__KET____DOT__vx_alu__out_alu_result = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2 = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__mult_signed_result = VL_RAND_RESET_Q(64);
    Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2 = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__mult_signed_result = VL_RAND_RESET_Q(64);
    Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2 = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__mult_signed_result = VL_RAND_RESET_Q(64);
    Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2 = VL_RAND_RESET_I(32);
    Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__mult_signed_result = VL_RAND_RESET_Q(64);
    VL_RAND_RESET_W(464,Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in);
    VL_RAND_RESET_W(464,Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value);
    Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(303,Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value);
    VL_RAND_RESET_W(128,Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data);
    Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd = VL_RAND_RESET_I(1);
    Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(128,Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next);
    VL_RAND_RESET_W(128,Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next);
    VL_RAND_RESET_W(128,Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next);
    VL_RAND_RESET_W(12300,Vortex__DOT__vx_csr_handler__DOT__csr);
    Vortex__DOT__vx_csr_handler__DOT__cycle = VL_RAND_RESET_Q(64);
    Vortex__DOT__vx_csr_handler__DOT__instret = VL_RAND_RESET_Q(64);
    Vortex__DOT__vx_csr_handler__DOT__decode_csr_address = VL_RAND_RESET_I(12);
    Vortex__DOT__vx_csr_handler__DOT__data_read = VL_RAND_RESET_I(12);
    Vortex__DOT__vx_csr_handler__DOT____Vlvbound1 = VL_RAND_RESET_I(12);
    __Vtableidx1 = VL_RAND_RESET_I(3);
    __Vtable1_Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu[0] = 0x10U;
    __Vtable1_Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu[1] = 0x11U;
    __Vtable1_Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu[2] = 0x12U;
    __Vtable1_Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu[3] = 0x13U;
    __Vtable1_Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu[4] = 0x14U;
    __Vtable1_Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu[5] = 0x15U;
    __Vtable1_Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu[6] = 0x16U;
    __Vtable1_Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu[7] = 0x17U;
}
