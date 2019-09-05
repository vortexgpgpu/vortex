// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVortex.h for the primary calling header

#include "VVortex_VX_context_slave.h"
#include "VVortex__Syms.h"


//--------------------
// STATIC VARIABLES


//--------------------

VL_CTOR_IMP(VVortex_VX_context_slave) {
    // Reset internal values
    // Reset structure values
    _ctor_var_reset();
}

void VVortex_VX_context_slave::__Vconfigure(VVortex__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

VVortex_VX_context_slave::~VVortex_VX_context_slave() {
}

//--------------------
// Internal Methods

void VVortex_VX_context_slave::_initial__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__VX_Context_one__1(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VVortex_VX_context_slave::_initial__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__VX_Context_one__1\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // INITIAL at VX_context_slave.v:38
    // INITIAL at VX_context_slave.v:39
    // INITIAL at VX_context_slave.v:41
    this->__PVT__clone_state_stall = 0U;
    this->__PVT__wspawn_state_stall = 0U;
}

void VVortex_VX_context_slave::_settle__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__VX_Context_one__8(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VVortex_VX_context_slave::_settle__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__VX_Context_one__8\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    this->__PVT__rd2_register[0U] = this->__Vcellout__vx_register_file_master__out_src2_data;
    this->__PVT__rd2_register[1U] = this->__Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src2_data;
    this->__PVT__rd2_register[2U] = this->__Vcellout__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__out_src2_data;
    this->__PVT__rd2_register[3U] = this->__Vcellout__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__out_src2_data;
    this->__PVT__rd1_register[0U] = this->__Vcellout__vx_register_file_master__out_src1_data;
    this->__PVT__rd1_register[1U] = this->__Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src1_data;
    this->__PVT__rd1_register[2U] = this->__Vcellout__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__out_src1_data;
    this->__PVT__rd1_register[3U] = this->__Vcellout__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__out_src1_data;
    this->out_b_reg_data[0U] = ((IData)(vlTOPp->Vortex__DOT__forwarding_src2_fwd)
				 ? vlTOPp->Vortex__DOT____Vcellout__vx_forwarding__out_src2_fwd_data[0U]
				 : this->__PVT__rd2_register[0U]);
    this->out_b_reg_data[1U] = ((IData)(vlTOPp->Vortex__DOT__forwarding_src2_fwd)
				 ? vlTOPp->Vortex__DOT____Vcellout__vx_forwarding__out_src2_fwd_data[1U]
				 : this->__PVT__rd2_register[1U]);
    this->out_b_reg_data[2U] = ((IData)(vlTOPp->Vortex__DOT__forwarding_src2_fwd)
				 ? vlTOPp->Vortex__DOT____Vcellout__vx_forwarding__out_src2_fwd_data[2U]
				 : this->__PVT__rd2_register[2U]);
    this->out_b_reg_data[3U] = ((IData)(vlTOPp->Vortex__DOT__forwarding_src2_fwd)
				 ? vlTOPp->Vortex__DOT____Vcellout__vx_forwarding__out_src2_fwd_data[3U]
				 : this->__PVT__rd2_register[3U]);
    this->out_a_reg_data[0U] = ((0x6fU == (0x7fU & 
					   ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 8U))))
				 ? ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				     << 0x18U) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
						  >> 8U))
				 : ((IData)(vlTOPp->Vortex__DOT__forwarding_src1_fwd)
				     ? vlTOPp->Vortex__DOT____Vcellout__vx_forwarding__out_src1_fwd_data[0U]
				     : this->__PVT__rd1_register[0U]));
    this->out_a_reg_data[1U] = ((0x6fU == (0x7fU & 
					   ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 8U))))
				 ? ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				     << 0x18U) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
						  >> 8U))
				 : ((IData)(vlTOPp->Vortex__DOT__forwarding_src1_fwd)
				     ? vlTOPp->Vortex__DOT____Vcellout__vx_forwarding__out_src1_fwd_data[1U]
				     : this->__PVT__rd1_register[1U]));
    this->out_a_reg_data[2U] = ((0x6fU == (0x7fU & 
					   ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 8U))))
				 ? ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				     << 0x18U) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
						  >> 8U))
				 : ((IData)(vlTOPp->Vortex__DOT__forwarding_src1_fwd)
				     ? vlTOPp->Vortex__DOT____Vcellout__vx_forwarding__out_src1_fwd_data[2U]
				     : this->__PVT__rd1_register[2U]));
    this->out_a_reg_data[3U] = ((0x6fU == (0x7fU & 
					   ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 8U))))
				 ? ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				     << 0x18U) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
						  >> 8U))
				 : ((IData)(vlTOPp->Vortex__DOT__forwarding_src1_fwd)
				     ? vlTOPp->Vortex__DOT____Vcellout__vx_forwarding__out_src1_fwd_data[3U]
				     : this->__PVT__rd1_register[3U]));
}

VL_INLINE_OPT void VVortex_VX_context_slave::_sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__VX_Context_one__15(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VVortex_VX_context_slave::_sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__VX_Context_one__15\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    this->__Vdly__wspawn_state_stall = this->__PVT__wspawn_state_stall;
    this->__Vdly__clone_state_stall = this->__PVT__clone_state_stall;
    // ALWAYS at VX_context_slave.v:119
    if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__real_wspawn) 
	 & (0U == (IData)(this->__PVT__wspawn_state_stall)))) {
	this->__Vdly__wspawn_state_stall = 0xaU;
    } else {
	if ((1U == (IData)(this->__PVT__wspawn_state_stall))) {
	    this->__Vdly__wspawn_state_stall = 0U;
	} else {
	    if ((0U < (IData)(this->__PVT__wspawn_state_stall))) {
		this->__Vdly__wspawn_state_stall = 
		    (0x3fU & ((IData)(this->__PVT__wspawn_state_stall) 
			      - (IData)(1U)));
	    }
	}
    }
    // ALWAYS at VX_context_slave.v:104
    if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__real_isclone) 
	 & (0U == (IData)(this->__PVT__clone_state_stall)))) {
	this->__Vdly__clone_state_stall = 0xaU;
    } else {
	if ((1U == (IData)(this->__PVT__clone_state_stall))) {
	    this->__Vdly__clone_state_stall = 0U;
	} else {
	    if ((0U < (IData)(this->__PVT__clone_state_stall))) {
		this->__Vdly__clone_state_stall = (0x3fU 
						   & ((IData)(this->__PVT__clone_state_stall) 
						      - (IData)(1U)));
	    }
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 7U)) & (1U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__real_isclone) 
	      & ((3U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (1U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 6U)) & (1U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__real_isclone) 
	      & ((2U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (1U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 5U)) & (1U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__real_isclone) 
	      & ((1U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (1U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    this->__PVT__clone_state_stall = this->__Vdly__clone_state_stall;
    // ALWAYS at VX_register_file_master_slave.v:50
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 4U)) & (1U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__real_wspawn)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__vx_register_file_master__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U]);
    } else {
	if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__real_wspawn) 
	     & (2U == (IData)(this->__PVT__wspawn_state_stall)))) {
	    this->__PVT__vx_register_file_master__DOT__registers[0U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__vx_register_file_master__DOT__registers[1U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__vx_register_file_master__DOT__registers[2U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__vx_register_file_master__DOT__registers[3U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__vx_register_file_master__DOT__registers[4U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__vx_register_file_master__DOT__registers[5U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__vx_register_file_master__DOT__registers[6U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__vx_register_file_master__DOT__registers[7U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__vx_register_file_master__DOT__registers[8U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__vx_register_file_master__DOT__registers[9U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__vx_register_file_master__DOT__registers[0xaU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xbU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xcU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xdU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xeU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xfU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x10U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x11U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x12U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x13U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x14U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x15U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x16U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x17U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x18U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x19U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1aU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1bU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1cU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1dU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1eU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1fU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    this->__PVT__wspawn_state_stall = this->__Vdly__wspawn_state_stall;
}

VL_INLINE_OPT void VVortex_VX_context_slave::_sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__VX_Context_one__22(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VVortex_VX_context_slave::_sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__VX_Context_one__22\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // ALWAYS at VX_register_file_master_slave.v:66
    this->__Vcellout__vx_register_file_master__out_src1_data 
	= this->__PVT__vx_register_file_master__DOT__registers[
	(0x1fU & ((0x7fffe00U & (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				 << 9U)) | (0x1ffU 
					    & (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 0x17U))))];
    // ALWAYS at VX_register_file_master_slave.v:66
    this->__Vcellout__vx_register_file_master__out_src2_data 
	= this->__PVT__vx_register_file_master__DOT__registers[
	(0x1fU & ((0x7fffff0U & (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				 << 4U)) | (0xfU & 
					    (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 0x1cU))))];
    // ALWAYS at VX_register_file_slave.v:68
    this->__Vcellout__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__out_src1_data 
	= this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[
	(0x1fU & ((0x7fffe00U & (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				 << 9U)) | (0x1ffU 
					    & (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 0x17U))))];
    // ALWAYS at VX_register_file_slave.v:68
    this->__Vcellout__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__out_src1_data 
	= this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[
	(0x1fU & ((0x7fffe00U & (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				 << 9U)) | (0x1ffU 
					    & (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 0x17U))))];
    // ALWAYS at VX_register_file_slave.v:68
    this->__Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src1_data 
	= this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[
	(0x1fU & ((0x7fffe00U & (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				 << 9U)) | (0x1ffU 
					    & (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 0x17U))))];
    // ALWAYS at VX_register_file_slave.v:68
    this->__Vcellout__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__out_src2_data 
	= this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[
	(0x1fU & ((0x7fffff0U & (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				 << 4U)) | (0xfU & 
					    (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 0x1cU))))];
    // ALWAYS at VX_register_file_slave.v:68
    this->__Vcellout__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__out_src2_data 
	= this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[
	(0x1fU & ((0x7fffff0U & (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				 << 4U)) | (0xfU & 
					    (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 0x1cU))))];
    // ALWAYS at VX_register_file_slave.v:68
    this->__Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src2_data 
	= this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[
	(0x1fU & ((0x7fffff0U & (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
				 << 4U)) | (0xfU & 
					    (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					     >> 0x1cU))))];
    this->__PVT__rd1_register[0U] = this->__Vcellout__vx_register_file_master__out_src1_data;
    this->__PVT__rd2_register[0U] = this->__Vcellout__vx_register_file_master__out_src2_data;
    this->__PVT__rd1_register[3U] = this->__Vcellout__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__out_src1_data;
    this->__PVT__rd1_register[2U] = this->__Vcellout__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__out_src1_data;
    this->__PVT__rd1_register[1U] = this->__Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src1_data;
    this->__PVT__rd2_register[3U] = this->__Vcellout__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__out_src2_data;
    this->__PVT__rd2_register[2U] = this->__Vcellout__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__out_src2_data;
    this->__PVT__rd2_register[1U] = this->__Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src2_data;
}

VL_INLINE_OPT void VVortex_VX_context_slave::_combo__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__VX_Context_one__29(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VVortex_VX_context_slave::_combo__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__VX_Context_one__29\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    this->out_a_reg_data[0U] = ((0x6fU == (0x7fU & 
					   ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 8U))))
				 ? ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				     << 0x18U) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
						  >> 8U))
				 : ((IData)(vlTOPp->Vortex__DOT__forwarding_src1_fwd)
				     ? vlTOPp->Vortex__DOT____Vcellout__vx_forwarding__out_src1_fwd_data[0U]
				     : this->__PVT__rd1_register[0U]));
    this->out_a_reg_data[1U] = ((0x6fU == (0x7fU & 
					   ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 8U))))
				 ? ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				     << 0x18U) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
						  >> 8U))
				 : ((IData)(vlTOPp->Vortex__DOT__forwarding_src1_fwd)
				     ? vlTOPp->Vortex__DOT____Vcellout__vx_forwarding__out_src1_fwd_data[1U]
				     : this->__PVT__rd1_register[1U]));
    this->out_a_reg_data[2U] = ((0x6fU == (0x7fU & 
					   ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 8U))))
				 ? ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				     << 0x18U) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
						  >> 8U))
				 : ((IData)(vlTOPp->Vortex__DOT__forwarding_src1_fwd)
				     ? vlTOPp->Vortex__DOT____Vcellout__vx_forwarding__out_src1_fwd_data[2U]
				     : this->__PVT__rd1_register[2U]));
    this->out_a_reg_data[3U] = ((0x6fU == (0x7fU & 
					   ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[2U] 
					     << 0x18U) 
					    | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
					       >> 8U))))
				 ? ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				     << 0x18U) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
						  >> 8U))
				 : ((IData)(vlTOPp->Vortex__DOT__forwarding_src1_fwd)
				     ? vlTOPp->Vortex__DOT____Vcellout__vx_forwarding__out_src1_fwd_data[3U]
				     : this->__PVT__rd1_register[3U]));
    this->out_b_reg_data[0U] = ((IData)(vlTOPp->Vortex__DOT__forwarding_src2_fwd)
				 ? vlTOPp->Vortex__DOT____Vcellout__vx_forwarding__out_src2_fwd_data[0U]
				 : this->__PVT__rd2_register[0U]);
    this->out_b_reg_data[1U] = ((IData)(vlTOPp->Vortex__DOT__forwarding_src2_fwd)
				 ? vlTOPp->Vortex__DOT____Vcellout__vx_forwarding__out_src2_fwd_data[1U]
				 : this->__PVT__rd2_register[1U]);
    this->out_b_reg_data[2U] = ((IData)(vlTOPp->Vortex__DOT__forwarding_src2_fwd)
				 ? vlTOPp->Vortex__DOT____Vcellout__vx_forwarding__out_src2_fwd_data[2U]
				 : this->__PVT__rd2_register[2U]);
    this->out_b_reg_data[3U] = ((IData)(vlTOPp->Vortex__DOT__forwarding_src2_fwd)
				 ? vlTOPp->Vortex__DOT____Vcellout__vx_forwarding__out_src2_fwd_data[3U]
				 : this->__PVT__rd2_register[3U]);
}

VL_INLINE_OPT void VVortex_VX_context_slave::_sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__2__KET____DOT__VX_Context_one__16(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VVortex_VX_context_slave::_sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__2__KET____DOT__VX_Context_one__16\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    this->__Vdly__wspawn_state_stall = this->__PVT__wspawn_state_stall;
    this->__Vdly__clone_state_stall = this->__PVT__clone_state_stall;
    // ALWAYS at VX_context_slave.v:119
    if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__2__KET____DOT__real_wspawn) 
	 & (0U == (IData)(this->__PVT__wspawn_state_stall)))) {
	this->__Vdly__wspawn_state_stall = 0xaU;
    } else {
	if ((1U == (IData)(this->__PVT__wspawn_state_stall))) {
	    this->__Vdly__wspawn_state_stall = 0U;
	} else {
	    if ((0U < (IData)(this->__PVT__wspawn_state_stall))) {
		this->__Vdly__wspawn_state_stall = 
		    (0x3fU & ((IData)(this->__PVT__wspawn_state_stall) 
			      - (IData)(1U)));
	    }
	}
    }
    // ALWAYS at VX_context_slave.v:104
    if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__2__KET____DOT__real_isclone) 
	 & (0U == (IData)(this->__PVT__clone_state_stall)))) {
	this->__Vdly__clone_state_stall = 0xaU;
    } else {
	if ((1U == (IData)(this->__PVT__clone_state_stall))) {
	    this->__Vdly__clone_state_stall = 0U;
	} else {
	    if ((0U < (IData)(this->__PVT__clone_state_stall))) {
		this->__Vdly__clone_state_stall = (0x3fU 
						   & ((IData)(this->__PVT__clone_state_stall) 
						      - (IData)(1U)));
	    }
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 7U)) & (2U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__2__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__2__KET____DOT__real_isclone) 
	      & ((3U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (2U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 6U)) & (2U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__2__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__2__KET____DOT__real_isclone) 
	      & ((2U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (2U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 5U)) & (2U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__2__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__2__KET____DOT__real_isclone) 
	      & ((1U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (2U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    this->__PVT__clone_state_stall = this->__Vdly__clone_state_stall;
    // ALWAYS at VX_register_file_master_slave.v:50
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 4U)) & (2U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__2__KET____DOT__real_wspawn)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__vx_register_file_master__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U]);
    } else {
	if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__2__KET____DOT__real_wspawn) 
	     & (2U == (IData)(this->__PVT__wspawn_state_stall)))) {
	    this->__PVT__vx_register_file_master__DOT__registers[0U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__vx_register_file_master__DOT__registers[1U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__vx_register_file_master__DOT__registers[2U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__vx_register_file_master__DOT__registers[3U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__vx_register_file_master__DOT__registers[4U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__vx_register_file_master__DOT__registers[5U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__vx_register_file_master__DOT__registers[6U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__vx_register_file_master__DOT__registers[7U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__vx_register_file_master__DOT__registers[8U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__vx_register_file_master__DOT__registers[9U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__vx_register_file_master__DOT__registers[0xaU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xbU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xcU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xdU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xeU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xfU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x10U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x11U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x12U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x13U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x14U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x15U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x16U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x17U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x18U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x19U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1aU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1bU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1cU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1dU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1eU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1fU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    this->__PVT__wspawn_state_stall = this->__Vdly__wspawn_state_stall;
}

VL_INLINE_OPT void VVortex_VX_context_slave::_sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__3__KET____DOT__VX_Context_one__17(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VVortex_VX_context_slave::_sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__3__KET____DOT__VX_Context_one__17\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    this->__Vdly__wspawn_state_stall = this->__PVT__wspawn_state_stall;
    this->__Vdly__clone_state_stall = this->__PVT__clone_state_stall;
    // ALWAYS at VX_context_slave.v:119
    if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__3__KET____DOT__real_wspawn) 
	 & (0U == (IData)(this->__PVT__wspawn_state_stall)))) {
	this->__Vdly__wspawn_state_stall = 0xaU;
    } else {
	if ((1U == (IData)(this->__PVT__wspawn_state_stall))) {
	    this->__Vdly__wspawn_state_stall = 0U;
	} else {
	    if ((0U < (IData)(this->__PVT__wspawn_state_stall))) {
		this->__Vdly__wspawn_state_stall = 
		    (0x3fU & ((IData)(this->__PVT__wspawn_state_stall) 
			      - (IData)(1U)));
	    }
	}
    }
    // ALWAYS at VX_context_slave.v:104
    if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__3__KET____DOT__real_isclone) 
	 & (0U == (IData)(this->__PVT__clone_state_stall)))) {
	this->__Vdly__clone_state_stall = 0xaU;
    } else {
	if ((1U == (IData)(this->__PVT__clone_state_stall))) {
	    this->__Vdly__clone_state_stall = 0U;
	} else {
	    if ((0U < (IData)(this->__PVT__clone_state_stall))) {
		this->__Vdly__clone_state_stall = (0x3fU 
						   & ((IData)(this->__PVT__clone_state_stall) 
						      - (IData)(1U)));
	    }
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 7U)) & (3U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__3__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__3__KET____DOT__real_isclone) 
	      & ((3U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (3U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 6U)) & (3U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__3__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__3__KET____DOT__real_isclone) 
	      & ((2U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (3U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 5U)) & (3U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__3__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__3__KET____DOT__real_isclone) 
	      & ((1U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (3U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    this->__PVT__clone_state_stall = this->__Vdly__clone_state_stall;
    // ALWAYS at VX_register_file_master_slave.v:50
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 4U)) & (3U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__3__KET____DOT__real_wspawn)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__vx_register_file_master__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U]);
    } else {
	if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__3__KET____DOT__real_wspawn) 
	     & (2U == (IData)(this->__PVT__wspawn_state_stall)))) {
	    this->__PVT__vx_register_file_master__DOT__registers[0U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__vx_register_file_master__DOT__registers[1U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__vx_register_file_master__DOT__registers[2U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__vx_register_file_master__DOT__registers[3U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__vx_register_file_master__DOT__registers[4U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__vx_register_file_master__DOT__registers[5U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__vx_register_file_master__DOT__registers[6U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__vx_register_file_master__DOT__registers[7U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__vx_register_file_master__DOT__registers[8U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__vx_register_file_master__DOT__registers[9U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__vx_register_file_master__DOT__registers[0xaU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xbU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xcU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xdU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xeU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xfU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x10U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x11U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x12U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x13U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x14U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x15U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x16U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x17U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x18U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x19U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1aU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1bU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1cU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1dU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1eU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1fU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    this->__PVT__wspawn_state_stall = this->__Vdly__wspawn_state_stall;
}

VL_INLINE_OPT void VVortex_VX_context_slave::_sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__4__KET____DOT__VX_Context_one__18(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VVortex_VX_context_slave::_sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__4__KET____DOT__VX_Context_one__18\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    this->__Vdly__wspawn_state_stall = this->__PVT__wspawn_state_stall;
    this->__Vdly__clone_state_stall = this->__PVT__clone_state_stall;
    // ALWAYS at VX_context_slave.v:119
    if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__4__KET____DOT__real_wspawn) 
	 & (0U == (IData)(this->__PVT__wspawn_state_stall)))) {
	this->__Vdly__wspawn_state_stall = 0xaU;
    } else {
	if ((1U == (IData)(this->__PVT__wspawn_state_stall))) {
	    this->__Vdly__wspawn_state_stall = 0U;
	} else {
	    if ((0U < (IData)(this->__PVT__wspawn_state_stall))) {
		this->__Vdly__wspawn_state_stall = 
		    (0x3fU & ((IData)(this->__PVT__wspawn_state_stall) 
			      - (IData)(1U)));
	    }
	}
    }
    // ALWAYS at VX_context_slave.v:104
    if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__4__KET____DOT__real_isclone) 
	 & (0U == (IData)(this->__PVT__clone_state_stall)))) {
	this->__Vdly__clone_state_stall = 0xaU;
    } else {
	if ((1U == (IData)(this->__PVT__clone_state_stall))) {
	    this->__Vdly__clone_state_stall = 0U;
	} else {
	    if ((0U < (IData)(this->__PVT__clone_state_stall))) {
		this->__Vdly__clone_state_stall = (0x3fU 
						   & ((IData)(this->__PVT__clone_state_stall) 
						      - (IData)(1U)));
	    }
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 7U)) & (4U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__4__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__4__KET____DOT__real_isclone) 
	      & ((3U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (4U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 6U)) & (4U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__4__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__4__KET____DOT__real_isclone) 
	      & ((2U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (4U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 5U)) & (4U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__4__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__4__KET____DOT__real_isclone) 
	      & ((1U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (4U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    this->__PVT__clone_state_stall = this->__Vdly__clone_state_stall;
    // ALWAYS at VX_register_file_master_slave.v:50
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 4U)) & (4U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__4__KET____DOT__real_wspawn)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__vx_register_file_master__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U]);
    } else {
	if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__4__KET____DOT__real_wspawn) 
	     & (2U == (IData)(this->__PVT__wspawn_state_stall)))) {
	    this->__PVT__vx_register_file_master__DOT__registers[0U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__vx_register_file_master__DOT__registers[1U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__vx_register_file_master__DOT__registers[2U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__vx_register_file_master__DOT__registers[3U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__vx_register_file_master__DOT__registers[4U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__vx_register_file_master__DOT__registers[5U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__vx_register_file_master__DOT__registers[6U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__vx_register_file_master__DOT__registers[7U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__vx_register_file_master__DOT__registers[8U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__vx_register_file_master__DOT__registers[9U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__vx_register_file_master__DOT__registers[0xaU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xbU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xcU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xdU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xeU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xfU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x10U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x11U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x12U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x13U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x14U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x15U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x16U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x17U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x18U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x19U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1aU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1bU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1cU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1dU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1eU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1fU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    this->__PVT__wspawn_state_stall = this->__Vdly__wspawn_state_stall;
}

VL_INLINE_OPT void VVortex_VX_context_slave::_sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__5__KET____DOT__VX_Context_one__19(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VVortex_VX_context_slave::_sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__5__KET____DOT__VX_Context_one__19\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    this->__Vdly__wspawn_state_stall = this->__PVT__wspawn_state_stall;
    this->__Vdly__clone_state_stall = this->__PVT__clone_state_stall;
    // ALWAYS at VX_context_slave.v:119
    if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__5__KET____DOT__real_wspawn) 
	 & (0U == (IData)(this->__PVT__wspawn_state_stall)))) {
	this->__Vdly__wspawn_state_stall = 0xaU;
    } else {
	if ((1U == (IData)(this->__PVT__wspawn_state_stall))) {
	    this->__Vdly__wspawn_state_stall = 0U;
	} else {
	    if ((0U < (IData)(this->__PVT__wspawn_state_stall))) {
		this->__Vdly__wspawn_state_stall = 
		    (0x3fU & ((IData)(this->__PVT__wspawn_state_stall) 
			      - (IData)(1U)));
	    }
	}
    }
    // ALWAYS at VX_context_slave.v:104
    if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__5__KET____DOT__real_isclone) 
	 & (0U == (IData)(this->__PVT__clone_state_stall)))) {
	this->__Vdly__clone_state_stall = 0xaU;
    } else {
	if ((1U == (IData)(this->__PVT__clone_state_stall))) {
	    this->__Vdly__clone_state_stall = 0U;
	} else {
	    if ((0U < (IData)(this->__PVT__clone_state_stall))) {
		this->__Vdly__clone_state_stall = (0x3fU 
						   & ((IData)(this->__PVT__clone_state_stall) 
						      - (IData)(1U)));
	    }
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 7U)) & (5U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__5__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__5__KET____DOT__real_isclone) 
	      & ((3U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (5U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 6U)) & (5U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__5__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__5__KET____DOT__real_isclone) 
	      & ((2U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (5U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 5U)) & (5U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__5__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__5__KET____DOT__real_isclone) 
	      & ((1U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (5U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    this->__PVT__clone_state_stall = this->__Vdly__clone_state_stall;
    // ALWAYS at VX_register_file_master_slave.v:50
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 4U)) & (5U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__5__KET____DOT__real_wspawn)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__vx_register_file_master__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U]);
    } else {
	if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__5__KET____DOT__real_wspawn) 
	     & (2U == (IData)(this->__PVT__wspawn_state_stall)))) {
	    this->__PVT__vx_register_file_master__DOT__registers[0U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__vx_register_file_master__DOT__registers[1U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__vx_register_file_master__DOT__registers[2U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__vx_register_file_master__DOT__registers[3U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__vx_register_file_master__DOT__registers[4U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__vx_register_file_master__DOT__registers[5U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__vx_register_file_master__DOT__registers[6U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__vx_register_file_master__DOT__registers[7U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__vx_register_file_master__DOT__registers[8U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__vx_register_file_master__DOT__registers[9U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__vx_register_file_master__DOT__registers[0xaU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xbU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xcU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xdU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xeU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xfU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x10U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x11U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x12U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x13U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x14U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x15U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x16U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x17U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x18U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x19U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1aU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1bU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1cU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1dU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1eU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1fU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    this->__PVT__wspawn_state_stall = this->__Vdly__wspawn_state_stall;
}

VL_INLINE_OPT void VVortex_VX_context_slave::_sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__6__KET____DOT__VX_Context_one__20(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VVortex_VX_context_slave::_sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__6__KET____DOT__VX_Context_one__20\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    this->__Vdly__wspawn_state_stall = this->__PVT__wspawn_state_stall;
    this->__Vdly__clone_state_stall = this->__PVT__clone_state_stall;
    // ALWAYS at VX_context_slave.v:119
    if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__6__KET____DOT__real_wspawn) 
	 & (0U == (IData)(this->__PVT__wspawn_state_stall)))) {
	this->__Vdly__wspawn_state_stall = 0xaU;
    } else {
	if ((1U == (IData)(this->__PVT__wspawn_state_stall))) {
	    this->__Vdly__wspawn_state_stall = 0U;
	} else {
	    if ((0U < (IData)(this->__PVT__wspawn_state_stall))) {
		this->__Vdly__wspawn_state_stall = 
		    (0x3fU & ((IData)(this->__PVT__wspawn_state_stall) 
			      - (IData)(1U)));
	    }
	}
    }
    // ALWAYS at VX_context_slave.v:104
    if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__6__KET____DOT__real_isclone) 
	 & (0U == (IData)(this->__PVT__clone_state_stall)))) {
	this->__Vdly__clone_state_stall = 0xaU;
    } else {
	if ((1U == (IData)(this->__PVT__clone_state_stall))) {
	    this->__Vdly__clone_state_stall = 0U;
	} else {
	    if ((0U < (IData)(this->__PVT__clone_state_stall))) {
		this->__Vdly__clone_state_stall = (0x3fU 
						   & ((IData)(this->__PVT__clone_state_stall) 
						      - (IData)(1U)));
	    }
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 7U)) & (6U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__6__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__6__KET____DOT__real_isclone) 
	      & ((3U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (6U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 6U)) & (6U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__6__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__6__KET____DOT__real_isclone) 
	      & ((2U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (6U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 5U)) & (6U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__6__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__6__KET____DOT__real_isclone) 
	      & ((1U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (6U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    this->__PVT__clone_state_stall = this->__Vdly__clone_state_stall;
    // ALWAYS at VX_register_file_master_slave.v:50
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 4U)) & (6U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__6__KET____DOT__real_wspawn)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__vx_register_file_master__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U]);
    } else {
	if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__6__KET____DOT__real_wspawn) 
	     & (2U == (IData)(this->__PVT__wspawn_state_stall)))) {
	    this->__PVT__vx_register_file_master__DOT__registers[0U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__vx_register_file_master__DOT__registers[1U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__vx_register_file_master__DOT__registers[2U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__vx_register_file_master__DOT__registers[3U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__vx_register_file_master__DOT__registers[4U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__vx_register_file_master__DOT__registers[5U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__vx_register_file_master__DOT__registers[6U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__vx_register_file_master__DOT__registers[7U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__vx_register_file_master__DOT__registers[8U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__vx_register_file_master__DOT__registers[9U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__vx_register_file_master__DOT__registers[0xaU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xbU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xcU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xdU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xeU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xfU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x10U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x11U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x12U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x13U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x14U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x15U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x16U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x17U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x18U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x19U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1aU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1bU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1cU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1dU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1eU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1fU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    this->__PVT__wspawn_state_stall = this->__Vdly__wspawn_state_stall;
}

VL_INLINE_OPT void VVortex_VX_context_slave::_sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__7__KET____DOT__VX_Context_one__21(VVortex__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VVortex_VX_context_slave::_sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__7__KET____DOT__VX_Context_one__21\n"); );
    VVortex* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    this->__Vdly__wspawn_state_stall = this->__PVT__wspawn_state_stall;
    this->__Vdly__clone_state_stall = this->__PVT__clone_state_stall;
    // ALWAYS at VX_context_slave.v:119
    if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__7__KET____DOT__real_wspawn) 
	 & (0U == (IData)(this->__PVT__wspawn_state_stall)))) {
	this->__Vdly__wspawn_state_stall = 0xaU;
    } else {
	if ((1U == (IData)(this->__PVT__wspawn_state_stall))) {
	    this->__Vdly__wspawn_state_stall = 0U;
	} else {
	    if ((0U < (IData)(this->__PVT__wspawn_state_stall))) {
		this->__Vdly__wspawn_state_stall = 
		    (0x3fU & ((IData)(this->__PVT__wspawn_state_stall) 
			      - (IData)(1U)));
	    }
	}
    }
    // ALWAYS at VX_context_slave.v:104
    if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__7__KET____DOT__real_isclone) 
	 & (0U == (IData)(this->__PVT__clone_state_stall)))) {
	this->__Vdly__clone_state_stall = 0xaU;
    } else {
	if ((1U == (IData)(this->__PVT__clone_state_stall))) {
	    this->__Vdly__clone_state_stall = 0U;
	} else {
	    if ((0U < (IData)(this->__PVT__clone_state_stall))) {
		this->__Vdly__clone_state_stall = (0x3fU 
						   & ((IData)(this->__PVT__clone_state_stall) 
						      - (IData)(1U)));
	    }
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 7U)) & (7U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__7__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[3U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__7__KET____DOT__real_isclone) 
	      & ((3U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (7U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 6U)) & (7U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__7__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[2U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__7__KET____DOT__real_isclone) 
	      & ((2U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (7U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    // ALWAYS at VX_register_file_slave.v:53
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 5U)) & (7U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__7__KET____DOT__real_isclone)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[1U]);
    } else {
	if ((((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__7__KET____DOT__real_isclone) 
	      & ((1U == this->__PVT__rd1_register[0U]) 
		 & (1U == (IData)(this->__PVT__clone_state_stall)))) 
	     & (7U == (0xfU & ((vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[1U] 
				<< 0x1cU) | (vlTOPp->Vortex__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value[0U] 
					     >> 4U)))))) {
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[1U] 
		= this->__PVT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[2U] 
		= this->__PVT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[3U] 
		= this->__PVT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[4U] 
		= this->__PVT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[5U] 
		= this->__PVT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[6U] 
		= this->__PVT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[7U] 
		= this->__PVT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[8U] 
		= this->__PVT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[9U] 
		= this->__PVT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xaU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xbU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xcU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xdU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xeU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0xfU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x10U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x11U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x12U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x13U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x14U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x15U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x16U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x17U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x18U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x19U] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1aU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1bU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1cU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1dU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1eU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[0x1fU] 
		= this->__PVT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    this->__PVT__clone_state_stall = this->__Vdly__clone_state_stall;
    // ALWAYS at VX_register_file_master_slave.v:50
    if ((((((0U != (3U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
			   << 0xeU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
				       >> 0x12U)))) 
	    & (0U != (0x1fU & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
				<< 0xcU) | (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					    >> 0x14U))))) 
	   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U] 
	      >> 4U)) & (7U == (0xfU & vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[0U]))) 
	 & (~ (IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__7__KET____DOT__real_wspawn)))) {
	VL_ASSIGNSEL_WIII(32,(0x3e0U & ((vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[2U] 
					 << 0x11U) 
					| (0x1ffe0U 
					   & (vlTOPp->Vortex__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value[1U] 
					      >> 0xfU)))), this->__PVT__vx_register_file_master__DOT__registers, 
			  vlSymsp->TOP__Vortex__DOT__VX_writeback_inter.write_data[0U]);
    } else {
	if (((IData)(vlTOPp->Vortex__DOT__vx_decode__DOT__genblk2__BRA__7__KET____DOT__real_wspawn) 
	     & (2U == (IData)(this->__PVT__wspawn_state_stall)))) {
	    this->__PVT__vx_register_file_master__DOT__registers[0U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0U];
	    this->__PVT__vx_register_file_master__DOT__registers[1U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[1U];
	    this->__PVT__vx_register_file_master__DOT__registers[2U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[2U];
	    this->__PVT__vx_register_file_master__DOT__registers[3U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[3U];
	    this->__PVT__vx_register_file_master__DOT__registers[4U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[4U];
	    this->__PVT__vx_register_file_master__DOT__registers[5U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[5U];
	    this->__PVT__vx_register_file_master__DOT__registers[6U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[6U];
	    this->__PVT__vx_register_file_master__DOT__registers[7U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[7U];
	    this->__PVT__vx_register_file_master__DOT__registers[8U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[8U];
	    this->__PVT__vx_register_file_master__DOT__registers[9U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[9U];
	    this->__PVT__vx_register_file_master__DOT__registers[0xaU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xaU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xbU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xbU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xcU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xcU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xdU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xdU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xeU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xeU];
	    this->__PVT__vx_register_file_master__DOT__registers[0xfU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0xfU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x10U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x10U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x11U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x11U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x12U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x12U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x13U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x13U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x14U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x14U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x15U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x15U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x16U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x16U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x17U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x17U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x18U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x18U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x19U] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x19U];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1aU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1aU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1bU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1bU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1cU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1cU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1dU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1dU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1eU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1eU];
	    this->__PVT__vx_register_file_master__DOT__registers[0x1fU] 
		= vlTOPp->Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[0x1fU];
	}
    }
    this->__PVT__wspawn_state_stall = this->__Vdly__wspawn_state_stall;
}

void VVortex_VX_context_slave::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+          VVortex_VX_context_slave::_ctor_var_reset\n"); );
    // Body
    clk = VL_RAND_RESET_I(1);
    in_warp = VL_RAND_RESET_I(1);
    in_wb_warp = VL_RAND_RESET_I(1);
    in_valid = VL_RAND_RESET_I(4);
    in_write_register = VL_RAND_RESET_I(1);
    in_rd = VL_RAND_RESET_I(5);
    VL_RAND_RESET_W(128,in_write_data);
    in_src1 = VL_RAND_RESET_I(5);
    in_src2 = VL_RAND_RESET_I(5);
    in_curr_PC = VL_RAND_RESET_I(32);
    in_is_clone = VL_RAND_RESET_I(1);
    in_is_jal = VL_RAND_RESET_I(1);
    in_src1_fwd = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(128,in_src1_fwd_data);
    in_src2_fwd = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(128,in_src2_fwd_data);
    VL_RAND_RESET_W(1024,in_wspawn_regs);
    in_wspawn = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(128,out_a_reg_data);
    VL_RAND_RESET_W(128,out_b_reg_data);
    out_clone_stall = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(128,__PVT__rd1_register);
    VL_RAND_RESET_W(128,__PVT__rd2_register);
    __PVT__clone_state_stall = VL_RAND_RESET_I(6);
    __PVT__wspawn_state_stall = VL_RAND_RESET_I(6);
    __Vcellout__vx_register_file_master__out_src2_data = VL_RAND_RESET_I(32);
    __Vcellout__vx_register_file_master__out_src1_data = VL_RAND_RESET_I(32);
    __Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src2_data = VL_RAND_RESET_I(32);
    __Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src1_data = VL_RAND_RESET_I(32);
    __Vcellout__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__out_src2_data = VL_RAND_RESET_I(32);
    __Vcellout__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__out_src1_data = VL_RAND_RESET_I(32);
    __Vcellout__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__out_src2_data = VL_RAND_RESET_I(32);
    __Vcellout__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__out_src1_data = VL_RAND_RESET_I(32);
    VL_RAND_RESET_W(1024,__PVT__vx_register_file_master__DOT__registers);
    VL_RAND_RESET_W(1024,__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers);
    VL_RAND_RESET_W(1024,__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers);
    VL_RAND_RESET_W(1024,__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers);
    __Vdly__clone_state_stall = VL_RAND_RESET_I(6);
    __Vdly__wspawn_state_stall = VL_RAND_RESET_I(6);
}
