// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See VVX_decode.h for the primary calling header

#include "VVX_decode.h"
#include "VVX_decode__Syms.h"


//--------------------
// STATIC VARIABLES


//--------------------

VL_CTOR_IMP(VVX_decode) {
    VVX_decode__Syms* __restrict vlSymsp = __VlSymsp = new VVX_decode__Syms(this, name());
    VVX_decode* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Reset internal values
    
    // Reset structure values
    _ctor_var_reset();
}

void VVX_decode::__Vconfigure(VVX_decode__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

VVX_decode::~VVX_decode() {
    delete __VlSymsp; __VlSymsp=NULL;
}

//--------------------


void VVX_decode::eval() {
    VL_DEBUG_IF(VL_DBG_MSGF("+++++TOP Evaluate VVX_decode::eval\n"); );
    VVX_decode__Syms* __restrict vlSymsp = this->__VlSymsp;  // Setup global symbol table
    VVX_decode* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
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

void VVX_decode::_eval_initial_loop(VVX_decode__Syms* __restrict vlSymsp) {
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

VL_INLINE_OPT void VVX_decode::_combo__TOP__1(VVX_decode__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_decode::_combo__TOP__1\n"); );
    VVX_decode* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->out_PC_next = ((IData)(4U) + vlTOPp->in_curr_PC);
    vlTOPp->out_mem_read = (7U & ((3U == (0x7fU & vlTOPp->in_instruction))
				   ? (vlTOPp->in_instruction 
				      >> 0xcU) : 7U));
    vlTOPp->out_mem_write = (7U & ((0x23U == (0x7fU 
					      & vlTOPp->in_instruction))
				    ? (vlTOPp->in_instruction 
				       >> 0xcU) : 7U));
    // ALWAYS at VX_decode.v:247
    vlTOPp->out_jal = ((0x6fU == (0x7fU & vlTOPp->in_instruction)) 
		       | ((0x67U == (0x7fU & vlTOPp->in_instruction)) 
			  | ((0x73U == (0x7fU & vlTOPp->in_instruction)) 
			     & ((0U == (7U & (vlTOPp->in_instruction 
					      >> 0xcU))) 
				& (2U > (0xfffU & (vlTOPp->in_instruction 
						   >> 0x14U)))))));
    vlTOPp->out_csr_address = (0xfffU & (((0U != (7U 
						  & (vlTOPp->in_instruction 
						     >> 0xcU))) 
					  & (2U <= 
					     (0xfffU 
					      & (vlTOPp->in_instruction 
						 >> 0x14U))))
					  ? (vlTOPp->in_instruction 
					     >> 0x14U)
					  : 0x55U));
    // ALWAYS at VX_decode.v:306
    vlTOPp->out_branch_stall = ((0x63U == (0x7fU & vlTOPp->in_instruction)) 
				| ((0x6fU == (0x7fU 
					      & vlTOPp->in_instruction)) 
				   | (0x67U == (0x7fU 
						& vlTOPp->in_instruction))));
    vlTOPp->out_rd = (0x1fU & (vlTOPp->in_instruction 
			       >> 7U));
    // ALWAYS at VX_decode.v:306
    vlTOPp->out_branch_type = ((0x63U == (0x7fU & vlTOPp->in_instruction))
			        ? ((0x4000U & vlTOPp->in_instruction)
				    ? ((0x2000U & vlTOPp->in_instruction)
				        ? ((0x1000U 
					    & vlTOPp->in_instruction)
					    ? 6U : 5U)
				        : ((0x1000U 
					    & vlTOPp->in_instruction)
					    ? 4U : 3U))
				    : ((0x2000U & vlTOPp->in_instruction)
				        ? 0U : ((0x1000U 
						 & vlTOPp->in_instruction)
						 ? 2U
						 : 1U)))
			        : 0U);
    vlTOPp->VX_decode__DOT__is_itype = ((0x13U == (0x7fU 
						   & vlTOPp->in_instruction)) 
					| (3U == (0x7fU 
						  & vlTOPp->in_instruction)));
    vlTOPp->VX_decode__DOT__is_csr = ((0x73U == (0x7fU 
						 & vlTOPp->in_instruction)) 
				      & (0U != (7U 
						& (vlTOPp->in_instruction 
						   >> 0xcU))));
    vlTOPp->out_rs1 = (0x1fU & (vlTOPp->in_instruction 
				>> 0xfU));
    vlTOPp->out_rs2 = (0x1fU & (vlTOPp->in_instruction 
				>> 0x14U));
    vlTOPp->out_rs2_src = (1U & (((IData)(vlTOPp->VX_decode__DOT__is_itype) 
				  | (0x23U == (0x7fU 
					       & vlTOPp->in_instruction)))
				  ? 1U : 0U));
    vlTOPp->out_is_csr = vlTOPp->VX_decode__DOT__is_csr;
    vlTOPp->out_wb = ((((0x6fU == (0x7fU & vlTOPp->in_instruction)) 
			| (0x67U == (0x7fU & vlTOPp->in_instruction))) 
		       | ((0x73U == (0x7fU & vlTOPp->in_instruction)) 
			  & (0U == (7U & (vlTOPp->in_instruction 
					  >> 0xcU)))))
		       ? 3U : ((3U == (0x7fU & vlTOPp->in_instruction))
			        ? 2U : ((((((IData)(vlTOPp->VX_decode__DOT__is_itype) 
					    | (0x33U 
					       == (0x7fU 
						   & vlTOPp->in_instruction))) 
					   | (0x37U 
					      == (0x7fU 
						  & vlTOPp->in_instruction))) 
					  | (0x17U 
					     == (0x7fU 
						 & vlTOPp->in_instruction))) 
					 | (IData)(vlTOPp->VX_decode__DOT__is_csr))
					 ? 1U : 0U)));
    vlTOPp->out_alu_op = ((0x63U == (0x7fU & vlTOPp->in_instruction))
			   ? ((5U > (IData)(vlTOPp->out_branch_type))
			       ? 1U : 0xaU) : ((0x37U 
						== 
						(0x7fU 
						 & vlTOPp->in_instruction))
					        ? 0xbU
					        : (
						   (0x17U 
						    == 
						    (0x7fU 
						     & vlTOPp->in_instruction))
						    ? 0xcU
						    : 
						   ((IData)(vlTOPp->VX_decode__DOT__is_csr)
						     ? 
						    ((1U 
						      == 
						      (3U 
						       & (vlTOPp->in_instruction 
							  >> 0xcU)))
						      ? 0xdU
						      : 
						     ((2U 
						       == 
						       (3U 
							& (vlTOPp->in_instruction 
							   >> 0xcU)))
						       ? 0xeU
						       : 0xfU))
						     : 
						    (((0x23U 
						       == 
						       (0x7fU 
							& vlTOPp->in_instruction)) 
						      | (3U 
							 == 
							 (0x7fU 
							  & vlTOPp->in_instruction)))
						      ? 0U
						      : 
						     ((0x4000U 
						       & vlTOPp->in_instruction)
						       ? 
						      ((0x2000U 
							& vlTOPp->in_instruction)
						        ? 
						       ((0x1000U 
							 & vlTOPp->in_instruction)
							 ? 9U
							 : 8U)
						        : 
						       ((0x1000U 
							 & vlTOPp->in_instruction)
							 ? 
							((0U 
							  == 
							  (0x7fU 
							   & (vlTOPp->in_instruction 
							      >> 0x19U)))
							  ? 6U
							  : 7U)
							 : 5U))
						       : 
						      ((0x2000U 
							& vlTOPp->in_instruction)
						        ? 
						       ((0x1000U 
							 & vlTOPp->in_instruction)
							 ? 4U
							 : 3U)
						        : 
						       ((0x1000U 
							 & vlTOPp->in_instruction)
							 ? 2U
							 : 
							((0x13U 
							  == 
							  (0x7fU 
							   & vlTOPp->in_instruction))
							  ? 0U
							  : 
							 ((0U 
							   == 
							   (0x7fU 
							    & (vlTOPp->in_instruction 
							       >> 0x19U)))
							   ? 0U
							   : 1U))))))))));
    // ALWAYS at VX_decode.v:201
    vlTOPp->out_upper_immed = ((0x37U == (0x7fU & vlTOPp->in_instruction))
			        ? ((0xfe000U & (vlTOPp->in_instruction 
						>> 0xcU)) 
				   | (((IData)(vlTOPp->out_rs2) 
				       << 8U) | (((IData)(vlTOPp->out_rs1) 
						  << 3U) 
						 | (7U 
						    & (vlTOPp->in_instruction 
						       >> 0xcU)))))
			        : ((0x17U == (0x7fU 
					      & vlTOPp->in_instruction))
				    ? ((0xfe000U & 
					(vlTOPp->in_instruction 
					 >> 0xcU)) 
				       | (((IData)(vlTOPp->out_rs2) 
					   << 8U) | 
					  (((IData)(vlTOPp->out_rs1) 
					    << 3U) 
					   | (7U & 
					      (vlTOPp->in_instruction 
					       >> 0xcU)))))
				    : 0U));
    vlTOPp->VX_decode__DOT__jalr_immed = ((0xfe0U & 
					   (vlTOPp->in_instruction 
					    >> 0x14U)) 
					  | (IData)(vlTOPp->out_rs2));
    vlTOPp->VX_decode__DOT__alu_tempp = (0xfffU & (
						   ((1U 
						     == 
						     (7U 
						      & (vlTOPp->in_instruction 
							 >> 0xcU))) 
						    | (5U 
						       == 
						       (7U 
							& (vlTOPp->in_instruction 
							   >> 0xcU))))
						    ? (IData)(vlTOPp->out_rs2)
						    : 
						   (vlTOPp->in_instruction 
						    >> 0x14U)));
    // ALWAYS at VX_decode.v:247
    vlTOPp->out_jal_offset = ((0x6fU == (0x7fU & vlTOPp->in_instruction))
			       ? ((0xffe00000U & (VL_NEGATE_I((IData)(
								      (1U 
								       & (vlTOPp->in_instruction 
									  >> 0x1fU)))) 
						  << 0x15U)) 
				  | ((0x100000U & (vlTOPp->in_instruction 
						   >> 0xbU)) 
				     | ((0xff000U & vlTOPp->in_instruction) 
					| ((0x800U 
					    & (vlTOPp->in_instruction 
					       >> 9U)) 
					   | (0x7feU 
					      & (vlTOPp->in_instruction 
						 >> 0x14U))))))
			       : ((0x67U == (0x7fU 
					     & vlTOPp->in_instruction))
				   ? ((0xfffff000U 
				       & (VL_NEGATE_I((IData)(
							      (1U 
							       & ((IData)(vlTOPp->VX_decode__DOT__jalr_immed) 
								  >> 0xbU)))) 
					  << 0xcU)) 
				      | (IData)(vlTOPp->VX_decode__DOT__jalr_immed))
				   : ((0x73U == (0x7fU 
						 & vlTOPp->in_instruction))
				       ? (((0U == (7U 
						   & (vlTOPp->in_instruction 
						      >> 0xcU))) 
					   & (2U > 
					      (0xfffU 
					       & (vlTOPp->in_instruction 
						  >> 0x14U))))
					   ? 0xb0000000U
					   : 0xdeadbeefU)
				       : 0xdeadbeefU)));
    // ALWAYS at VX_decode.v:295
    vlTOPp->out_itype_immed = ((0x40U & vlTOPp->in_instruction)
			        ? ((0x20U & vlTOPp->in_instruction)
				    ? ((0x10U & vlTOPp->in_instruction)
				        ? 0xdeadbeefU
				        : ((8U & vlTOPp->in_instruction)
					    ? 0xdeadbeefU
					    : ((4U 
						& vlTOPp->in_instruction)
					        ? 0xdeadbeefU
					        : (
						   (2U 
						    & vlTOPp->in_instruction)
						    ? 
						   ((1U 
						     & vlTOPp->in_instruction)
						     ? 
						    ((0xfffff000U 
						      & (VL_NEGATE_I((IData)(
									     (1U 
									      & (vlTOPp->in_instruction 
										>> 0x1fU)))) 
							 << 0xcU)) 
						     | ((0x800U 
							 & (vlTOPp->in_instruction 
							    >> 0x14U)) 
							| ((0x400U 
							    & (vlTOPp->in_instruction 
							       << 3U)) 
							   | ((0x3f0U 
							       & (vlTOPp->in_instruction 
								  >> 0x15U)) 
							      | (0xfU 
								 & (vlTOPp->in_instruction 
								    >> 8U))))))
						     : 0xdeadbeefU)
						    : 0xdeadbeefU))))
				    : 0xdeadbeefU) : 
			       ((0x20U & vlTOPp->in_instruction)
				 ? ((0x10U & vlTOPp->in_instruction)
				     ? 0xdeadbeefU : 
				    ((8U & vlTOPp->in_instruction)
				      ? 0xdeadbeefU
				      : ((4U & vlTOPp->in_instruction)
					  ? 0xdeadbeefU
					  : ((2U & vlTOPp->in_instruction)
					      ? ((1U 
						  & vlTOPp->in_instruction)
						  ? 
						 ((0xfffff000U 
						   & (VL_NEGATE_I((IData)(
									  (1U 
									   & (vlTOPp->in_instruction 
									      >> 0x1fU)))) 
						      << 0xcU)) 
						  | ((0xfe0U 
						      & (vlTOPp->in_instruction 
							 >> 0x14U)) 
						     | (IData)(vlTOPp->out_rd)))
						  : 0xdeadbeefU)
					      : 0xdeadbeefU))))
				 : ((0x10U & vlTOPp->in_instruction)
				     ? ((8U & vlTOPp->in_instruction)
					 ? 0xdeadbeefU
					 : ((4U & vlTOPp->in_instruction)
					     ? 0xdeadbeefU
					     : ((2U 
						 & vlTOPp->in_instruction)
						 ? 
						((1U 
						  & vlTOPp->in_instruction)
						  ? 
						 ((0xfffff000U 
						   & (VL_NEGATE_I((IData)(
									  (1U 
									   & ((IData)(vlTOPp->VX_decode__DOT__alu_tempp) 
									      >> 0xbU)))) 
						      << 0xcU)) 
						  | (IData)(vlTOPp->VX_decode__DOT__alu_tempp))
						  : 0xdeadbeefU)
						 : 0xdeadbeefU)))
				     : ((8U & vlTOPp->in_instruction)
					 ? 0xdeadbeefU
					 : ((4U & vlTOPp->in_instruction)
					     ? 0xdeadbeefU
					     : ((2U 
						 & vlTOPp->in_instruction)
						 ? 
						((1U 
						  & vlTOPp->in_instruction)
						  ? 
						 ((0xfffff000U 
						   & (VL_NEGATE_I((IData)(
									  (1U 
									   & (vlTOPp->in_instruction 
									      >> 0x1fU)))) 
						      << 0xcU)) 
						  | (0xfffU 
						     & (vlTOPp->in_instruction 
							>> 0x14U)))
						  : 0xdeadbeefU)
						 : 0xdeadbeefU))))));
}

void VVX_decode::_settle__TOP__2(VVX_decode__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_decode::_settle__TOP__2\n"); );
    VVX_decode* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->out_PC_next = ((IData)(4U) + vlTOPp->in_curr_PC);
    vlTOPp->out_mem_read = (7U & ((3U == (0x7fU & vlTOPp->in_instruction))
				   ? (vlTOPp->in_instruction 
				      >> 0xcU) : 7U));
    vlTOPp->out_mem_write = (7U & ((0x23U == (0x7fU 
					      & vlTOPp->in_instruction))
				    ? (vlTOPp->in_instruction 
				       >> 0xcU) : 7U));
    // ALWAYS at VX_decode.v:247
    vlTOPp->out_jal = ((0x6fU == (0x7fU & vlTOPp->in_instruction)) 
		       | ((0x67U == (0x7fU & vlTOPp->in_instruction)) 
			  | ((0x73U == (0x7fU & vlTOPp->in_instruction)) 
			     & ((0U == (7U & (vlTOPp->in_instruction 
					      >> 0xcU))) 
				& (2U > (0xfffU & (vlTOPp->in_instruction 
						   >> 0x14U)))))));
    vlTOPp->out_csr_address = (0xfffU & (((0U != (7U 
						  & (vlTOPp->in_instruction 
						     >> 0xcU))) 
					  & (2U <= 
					     (0xfffU 
					      & (vlTOPp->in_instruction 
						 >> 0x14U))))
					  ? (vlTOPp->in_instruction 
					     >> 0x14U)
					  : 0x55U));
    // ALWAYS at VX_decode.v:306
    vlTOPp->out_branch_stall = ((0x63U == (0x7fU & vlTOPp->in_instruction)) 
				| ((0x6fU == (0x7fU 
					      & vlTOPp->in_instruction)) 
				   | (0x67U == (0x7fU 
						& vlTOPp->in_instruction))));
    vlTOPp->out_rd = (0x1fU & (vlTOPp->in_instruction 
			       >> 7U));
    // ALWAYS at VX_decode.v:306
    vlTOPp->out_branch_type = ((0x63U == (0x7fU & vlTOPp->in_instruction))
			        ? ((0x4000U & vlTOPp->in_instruction)
				    ? ((0x2000U & vlTOPp->in_instruction)
				        ? ((0x1000U 
					    & vlTOPp->in_instruction)
					    ? 6U : 5U)
				        : ((0x1000U 
					    & vlTOPp->in_instruction)
					    ? 4U : 3U))
				    : ((0x2000U & vlTOPp->in_instruction)
				        ? 0U : ((0x1000U 
						 & vlTOPp->in_instruction)
						 ? 2U
						 : 1U)))
			        : 0U);
    vlTOPp->VX_decode__DOT__is_itype = ((0x13U == (0x7fU 
						   & vlTOPp->in_instruction)) 
					| (3U == (0x7fU 
						  & vlTOPp->in_instruction)));
    vlTOPp->VX_decode__DOT__is_csr = ((0x73U == (0x7fU 
						 & vlTOPp->in_instruction)) 
				      & (0U != (7U 
						& (vlTOPp->in_instruction 
						   >> 0xcU))));
    vlTOPp->out_rs1 = (0x1fU & (vlTOPp->in_instruction 
				>> 0xfU));
    vlTOPp->out_rs2 = (0x1fU & (vlTOPp->in_instruction 
				>> 0x14U));
    vlTOPp->out_rs2_src = (1U & (((IData)(vlTOPp->VX_decode__DOT__is_itype) 
				  | (0x23U == (0x7fU 
					       & vlTOPp->in_instruction)))
				  ? 1U : 0U));
    vlTOPp->out_is_csr = vlTOPp->VX_decode__DOT__is_csr;
    vlTOPp->out_wb = ((((0x6fU == (0x7fU & vlTOPp->in_instruction)) 
			| (0x67U == (0x7fU & vlTOPp->in_instruction))) 
		       | ((0x73U == (0x7fU & vlTOPp->in_instruction)) 
			  & (0U == (7U & (vlTOPp->in_instruction 
					  >> 0xcU)))))
		       ? 3U : ((3U == (0x7fU & vlTOPp->in_instruction))
			        ? 2U : ((((((IData)(vlTOPp->VX_decode__DOT__is_itype) 
					    | (0x33U 
					       == (0x7fU 
						   & vlTOPp->in_instruction))) 
					   | (0x37U 
					      == (0x7fU 
						  & vlTOPp->in_instruction))) 
					  | (0x17U 
					     == (0x7fU 
						 & vlTOPp->in_instruction))) 
					 | (IData)(vlTOPp->VX_decode__DOT__is_csr))
					 ? 1U : 0U)));
    vlTOPp->out_alu_op = ((0x63U == (0x7fU & vlTOPp->in_instruction))
			   ? ((5U > (IData)(vlTOPp->out_branch_type))
			       ? 1U : 0xaU) : ((0x37U 
						== 
						(0x7fU 
						 & vlTOPp->in_instruction))
					        ? 0xbU
					        : (
						   (0x17U 
						    == 
						    (0x7fU 
						     & vlTOPp->in_instruction))
						    ? 0xcU
						    : 
						   ((IData)(vlTOPp->VX_decode__DOT__is_csr)
						     ? 
						    ((1U 
						      == 
						      (3U 
						       & (vlTOPp->in_instruction 
							  >> 0xcU)))
						      ? 0xdU
						      : 
						     ((2U 
						       == 
						       (3U 
							& (vlTOPp->in_instruction 
							   >> 0xcU)))
						       ? 0xeU
						       : 0xfU))
						     : 
						    (((0x23U 
						       == 
						       (0x7fU 
							& vlTOPp->in_instruction)) 
						      | (3U 
							 == 
							 (0x7fU 
							  & vlTOPp->in_instruction)))
						      ? 0U
						      : 
						     ((0x4000U 
						       & vlTOPp->in_instruction)
						       ? 
						      ((0x2000U 
							& vlTOPp->in_instruction)
						        ? 
						       ((0x1000U 
							 & vlTOPp->in_instruction)
							 ? 9U
							 : 8U)
						        : 
						       ((0x1000U 
							 & vlTOPp->in_instruction)
							 ? 
							((0U 
							  == 
							  (0x7fU 
							   & (vlTOPp->in_instruction 
							      >> 0x19U)))
							  ? 6U
							  : 7U)
							 : 5U))
						       : 
						      ((0x2000U 
							& vlTOPp->in_instruction)
						        ? 
						       ((0x1000U 
							 & vlTOPp->in_instruction)
							 ? 4U
							 : 3U)
						        : 
						       ((0x1000U 
							 & vlTOPp->in_instruction)
							 ? 2U
							 : 
							((0x13U 
							  == 
							  (0x7fU 
							   & vlTOPp->in_instruction))
							  ? 0U
							  : 
							 ((0U 
							   == 
							   (0x7fU 
							    & (vlTOPp->in_instruction 
							       >> 0x19U)))
							   ? 0U
							   : 1U))))))))));
    vlTOPp->out_rd1 = ((0x6fU == (0x7fU & vlTOPp->in_instruction))
		        ? vlTOPp->in_curr_PC : ((IData)(vlTOPp->in_src1_fwd)
						 ? vlTOPp->in_src1_fwd_data
						 : 
						vlTOPp->VX_decode__DOT__vx_register_file__DOT__registers
						[vlTOPp->out_rs1]));
    // ALWAYS at VX_decode.v:201
    vlTOPp->out_upper_immed = ((0x37U == (0x7fU & vlTOPp->in_instruction))
			        ? ((0xfe000U & (vlTOPp->in_instruction 
						>> 0xcU)) 
				   | (((IData)(vlTOPp->out_rs2) 
				       << 8U) | (((IData)(vlTOPp->out_rs1) 
						  << 3U) 
						 | (7U 
						    & (vlTOPp->in_instruction 
						       >> 0xcU)))))
			        : ((0x17U == (0x7fU 
					      & vlTOPp->in_instruction))
				    ? ((0xfe000U & 
					(vlTOPp->in_instruction 
					 >> 0xcU)) 
				       | (((IData)(vlTOPp->out_rs2) 
					   << 8U) | 
					  (((IData)(vlTOPp->out_rs1) 
					    << 3U) 
					   | (7U & 
					      (vlTOPp->in_instruction 
					       >> 0xcU)))))
				    : 0U));
    vlTOPp->VX_decode__DOT__jalr_immed = ((0xfe0U & 
					   (vlTOPp->in_instruction 
					    >> 0x14U)) 
					  | (IData)(vlTOPp->out_rs2));
    vlTOPp->VX_decode__DOT__alu_tempp = (0xfffU & (
						   ((1U 
						     == 
						     (7U 
						      & (vlTOPp->in_instruction 
							 >> 0xcU))) 
						    | (5U 
						       == 
						       (7U 
							& (vlTOPp->in_instruction 
							   >> 0xcU))))
						    ? (IData)(vlTOPp->out_rs2)
						    : 
						   (vlTOPp->in_instruction 
						    >> 0x14U)));
    vlTOPp->out_rd2 = ((IData)(vlTOPp->in_src2_fwd)
		        ? vlTOPp->in_src2_fwd_data : 
		       vlTOPp->VX_decode__DOT__vx_register_file__DOT__registers
		       [vlTOPp->out_rs2]);
    vlTOPp->out_csr_mask = (((IData)(vlTOPp->VX_decode__DOT__is_csr) 
			     & (vlTOPp->in_instruction 
				>> 0xeU)) ? (IData)(vlTOPp->out_rs1)
			     : vlTOPp->out_rd1);
    // ALWAYS at VX_decode.v:247
    vlTOPp->out_jal_offset = ((0x6fU == (0x7fU & vlTOPp->in_instruction))
			       ? ((0xffe00000U & (VL_NEGATE_I((IData)(
								      (1U 
								       & (vlTOPp->in_instruction 
									  >> 0x1fU)))) 
						  << 0x15U)) 
				  | ((0x100000U & (vlTOPp->in_instruction 
						   >> 0xbU)) 
				     | ((0xff000U & vlTOPp->in_instruction) 
					| ((0x800U 
					    & (vlTOPp->in_instruction 
					       >> 9U)) 
					   | (0x7feU 
					      & (vlTOPp->in_instruction 
						 >> 0x14U))))))
			       : ((0x67U == (0x7fU 
					     & vlTOPp->in_instruction))
				   ? ((0xfffff000U 
				       & (VL_NEGATE_I((IData)(
							      (1U 
							       & ((IData)(vlTOPp->VX_decode__DOT__jalr_immed) 
								  >> 0xbU)))) 
					  << 0xcU)) 
				      | (IData)(vlTOPp->VX_decode__DOT__jalr_immed))
				   : ((0x73U == (0x7fU 
						 & vlTOPp->in_instruction))
				       ? (((0U == (7U 
						   & (vlTOPp->in_instruction 
						      >> 0xcU))) 
					   & (2U > 
					      (0xfffU 
					       & (vlTOPp->in_instruction 
						  >> 0x14U))))
					   ? 0xb0000000U
					   : 0xdeadbeefU)
				       : 0xdeadbeefU)));
    // ALWAYS at VX_decode.v:295
    vlTOPp->out_itype_immed = ((0x40U & vlTOPp->in_instruction)
			        ? ((0x20U & vlTOPp->in_instruction)
				    ? ((0x10U & vlTOPp->in_instruction)
				        ? 0xdeadbeefU
				        : ((8U & vlTOPp->in_instruction)
					    ? 0xdeadbeefU
					    : ((4U 
						& vlTOPp->in_instruction)
					        ? 0xdeadbeefU
					        : (
						   (2U 
						    & vlTOPp->in_instruction)
						    ? 
						   ((1U 
						     & vlTOPp->in_instruction)
						     ? 
						    ((0xfffff000U 
						      & (VL_NEGATE_I((IData)(
									     (1U 
									      & (vlTOPp->in_instruction 
										>> 0x1fU)))) 
							 << 0xcU)) 
						     | ((0x800U 
							 & (vlTOPp->in_instruction 
							    >> 0x14U)) 
							| ((0x400U 
							    & (vlTOPp->in_instruction 
							       << 3U)) 
							   | ((0x3f0U 
							       & (vlTOPp->in_instruction 
								  >> 0x15U)) 
							      | (0xfU 
								 & (vlTOPp->in_instruction 
								    >> 8U))))))
						     : 0xdeadbeefU)
						    : 0xdeadbeefU))))
				    : 0xdeadbeefU) : 
			       ((0x20U & vlTOPp->in_instruction)
				 ? ((0x10U & vlTOPp->in_instruction)
				     ? 0xdeadbeefU : 
				    ((8U & vlTOPp->in_instruction)
				      ? 0xdeadbeefU
				      : ((4U & vlTOPp->in_instruction)
					  ? 0xdeadbeefU
					  : ((2U & vlTOPp->in_instruction)
					      ? ((1U 
						  & vlTOPp->in_instruction)
						  ? 
						 ((0xfffff000U 
						   & (VL_NEGATE_I((IData)(
									  (1U 
									   & (vlTOPp->in_instruction 
									      >> 0x1fU)))) 
						      << 0xcU)) 
						  | ((0xfe0U 
						      & (vlTOPp->in_instruction 
							 >> 0x14U)) 
						     | (IData)(vlTOPp->out_rd)))
						  : 0xdeadbeefU)
					      : 0xdeadbeefU))))
				 : ((0x10U & vlTOPp->in_instruction)
				     ? ((8U & vlTOPp->in_instruction)
					 ? 0xdeadbeefU
					 : ((4U & vlTOPp->in_instruction)
					     ? 0xdeadbeefU
					     : ((2U 
						 & vlTOPp->in_instruction)
						 ? 
						((1U 
						  & vlTOPp->in_instruction)
						  ? 
						 ((0xfffff000U 
						   & (VL_NEGATE_I((IData)(
									  (1U 
									   & ((IData)(vlTOPp->VX_decode__DOT__alu_tempp) 
									      >> 0xbU)))) 
						      << 0xcU)) 
						  | (IData)(vlTOPp->VX_decode__DOT__alu_tempp))
						  : 0xdeadbeefU)
						 : 0xdeadbeefU)))
				     : ((8U & vlTOPp->in_instruction)
					 ? 0xdeadbeefU
					 : ((4U & vlTOPp->in_instruction)
					     ? 0xdeadbeefU
					     : ((2U 
						 & vlTOPp->in_instruction)
						 ? 
						((1U 
						  & vlTOPp->in_instruction)
						  ? 
						 ((0xfffff000U 
						   & (VL_NEGATE_I((IData)(
									  (1U 
									   & (vlTOPp->in_instruction 
									      >> 0x1fU)))) 
						      << 0xcU)) 
						  | (0xfffU 
						     & (vlTOPp->in_instruction 
							>> 0x14U)))
						  : 0xdeadbeefU)
						 : 0xdeadbeefU))))));
}

VL_INLINE_OPT void VVX_decode::_sequent__TOP__3(VVX_decode__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_decode::_sequent__TOP__3\n"); );
    VVX_decode* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    // Begin mtask footprint  all: 
    VL_SIG8(__Vdlyvdim0__VX_decode__DOT__vx_register_file__DOT__registers__v0,4,0);
    VL_SIG8(__Vdlyvset__VX_decode__DOT__vx_register_file__DOT__registers__v0,0,0);
    VL_SIG(__Vdlyvval__VX_decode__DOT__vx_register_file__DOT__registers__v0,31,0);
    // Body
    __Vdlyvset__VX_decode__DOT__vx_register_file__DOT__registers__v0 = 0U;
    // ALWAYS at VX_register_file.v:30
    if (((0U != (IData)(vlTOPp->in_wb)) & (0U != (IData)(vlTOPp->in_rd)))) {
	__Vdlyvval__VX_decode__DOT__vx_register_file__DOT__registers__v0 
	    = vlTOPp->in_write_data;
	__Vdlyvset__VX_decode__DOT__vx_register_file__DOT__registers__v0 = 1U;
	__Vdlyvdim0__VX_decode__DOT__vx_register_file__DOT__registers__v0 
	    = vlTOPp->in_rd;
    }
    // ALWAYSPOST at VX_register_file.v:32
    if (__Vdlyvset__VX_decode__DOT__vx_register_file__DOT__registers__v0) {
	vlTOPp->VX_decode__DOT__vx_register_file__DOT__registers[__Vdlyvdim0__VX_decode__DOT__vx_register_file__DOT__registers__v0] 
	    = __Vdlyvval__VX_decode__DOT__vx_register_file__DOT__registers__v0;
    }
}

VL_INLINE_OPT void VVX_decode::_combo__TOP__4(VVX_decode__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_decode::_combo__TOP__4\n"); );
    VVX_decode* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->out_rd2 = ((IData)(vlTOPp->in_src2_fwd)
		        ? vlTOPp->in_src2_fwd_data : 
		       vlTOPp->VX_decode__DOT__vx_register_file__DOT__registers
		       [vlTOPp->out_rs2]);
    vlTOPp->out_rd1 = ((0x6fU == (0x7fU & vlTOPp->in_instruction))
		        ? vlTOPp->in_curr_PC : ((IData)(vlTOPp->in_src1_fwd)
						 ? vlTOPp->in_src1_fwd_data
						 : 
						vlTOPp->VX_decode__DOT__vx_register_file__DOT__registers
						[vlTOPp->out_rs1]));
    vlTOPp->out_csr_mask = (((IData)(vlTOPp->VX_decode__DOT__is_csr) 
			     & (vlTOPp->in_instruction 
				>> 0xeU)) ? (IData)(vlTOPp->out_rs1)
			     : vlTOPp->out_rd1);
}

void VVX_decode::_eval(VVX_decode__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_decode::_eval\n"); );
    VVX_decode* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->_combo__TOP__1(vlSymsp);
    if (((IData)(vlTOPp->clk) & (~ (IData)(vlTOPp->__Vclklast__TOP__clk)))) {
	vlTOPp->_sequent__TOP__3(vlSymsp);
    }
    vlTOPp->_combo__TOP__4(vlSymsp);
    // Final
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
}

void VVX_decode::_eval_initial(VVX_decode__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_decode::_eval_initial\n"); );
    VVX_decode* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->__Vclklast__TOP__clk = vlTOPp->clk;
}

void VVX_decode::final() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_decode::final\n"); );
    // Variables
    VVX_decode__Syms* __restrict vlSymsp = this->__VlSymsp;
    VVX_decode* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
}

void VVX_decode::_eval_settle(VVX_decode__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_decode::_eval_settle\n"); );
    VVX_decode* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    vlTOPp->_settle__TOP__2(vlSymsp);
}

VL_INLINE_OPT QData VVX_decode::_change_request(VVX_decode__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_decode::_change_request\n"); );
    VVX_decode* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    // Change detection
    QData __req = false;  // Logically a bool
    return __req;
}

#ifdef VL_DEBUG
void VVX_decode::_eval_debug_assertions() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_decode::_eval_debug_assertions\n"); );
    // Body
    if (VL_UNLIKELY((clk & 0xfeU))) {
	Verilated::overWidthError("clk");}
    if (VL_UNLIKELY((in_rd & 0xe0U))) {
	Verilated::overWidthError("in_rd");}
    if (VL_UNLIKELY((in_wb & 0xfcU))) {
	Verilated::overWidthError("in_wb");}
    if (VL_UNLIKELY((in_src1_fwd & 0xfeU))) {
	Verilated::overWidthError("in_src1_fwd");}
    if (VL_UNLIKELY((in_src2_fwd & 0xfeU))) {
	Verilated::overWidthError("in_src2_fwd");}
}
#endif // VL_DEBUG

void VVX_decode::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+    VVX_decode::_ctor_var_reset\n"); );
    // Body
    clk = VL_RAND_RESET_I(1);
    in_instruction = VL_RAND_RESET_I(32);
    in_curr_PC = VL_RAND_RESET_I(32);
    in_write_data = VL_RAND_RESET_I(32);
    in_rd = VL_RAND_RESET_I(5);
    in_wb = VL_RAND_RESET_I(2);
    in_src1_fwd = VL_RAND_RESET_I(1);
    in_src1_fwd_data = VL_RAND_RESET_I(32);
    in_src2_fwd = VL_RAND_RESET_I(1);
    in_src2_fwd_data = VL_RAND_RESET_I(32);
    out_csr_address = VL_RAND_RESET_I(12);
    out_is_csr = VL_RAND_RESET_I(1);
    out_csr_mask = VL_RAND_RESET_I(32);
    out_rd = VL_RAND_RESET_I(5);
    out_rs1 = VL_RAND_RESET_I(5);
    out_rd1 = VL_RAND_RESET_I(32);
    out_rs2 = VL_RAND_RESET_I(5);
    out_rd2 = VL_RAND_RESET_I(32);
    out_wb = VL_RAND_RESET_I(2);
    out_alu_op = VL_RAND_RESET_I(4);
    out_rs2_src = VL_RAND_RESET_I(1);
    out_itype_immed = VL_RAND_RESET_I(32);
    out_mem_read = VL_RAND_RESET_I(3);
    out_mem_write = VL_RAND_RESET_I(3);
    out_branch_type = VL_RAND_RESET_I(3);
    out_branch_stall = VL_RAND_RESET_I(1);
    out_jal = VL_RAND_RESET_I(1);
    out_jal_offset = VL_RAND_RESET_I(32);
    out_upper_immed = VL_RAND_RESET_I(20);
    out_PC_next = VL_RAND_RESET_I(32);
    VX_decode__DOT__is_itype = VL_RAND_RESET_I(1);
    VX_decode__DOT__is_csr = VL_RAND_RESET_I(1);
    VX_decode__DOT__jalr_immed = VL_RAND_RESET_I(12);
    VX_decode__DOT__alu_tempp = VL_RAND_RESET_I(12);
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
	    VX_decode__DOT__vx_register_file__DOT__registers[__Vi0] = VL_RAND_RESET_I(32);
    }}
}
