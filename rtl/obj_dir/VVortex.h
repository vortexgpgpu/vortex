// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary design header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef _VVortex_H_
#define _VVortex_H_

#include "verilated.h"

class VVortex__Syms;
class VVortex_VX_dcache_response_inter;
class VVortex_VX_dcache_request_inter;
class VVortex_VX_frE_to_bckE_req_inter;
class VVortex_VX_wb_inter;
class VVortex_VX_branch_response_inter;
class VVortex_VX_warp_ctl_inter;
class VVortex_VX_inst_meta_inter;
class VVortex_VX_mem_req_inter;
class VVortex_VX_inst_mem_wb_inter;

//----------

VL_MODULE(VVortex) {
  public:
    // CELLS
    // Public to allow access to /*verilator_public*/ items;
    // otherwise the application code can consider these internals.
    VVortex_VX_dcache_response_inter*	__PVT__Vortex__DOT__VX_dcache_rsp;
    VVortex_VX_dcache_request_inter*	__PVT__Vortex__DOT__VX_dcache_req;
    VVortex_VX_frE_to_bckE_req_inter*	__PVT__Vortex__DOT__VX_bckE_req;
    VVortex_VX_wb_inter*	__PVT__Vortex__DOT__VX_writeback_inter;
    VVortex_VX_branch_response_inter*	__PVT__Vortex__DOT__VX_branch_rsp;
    VVortex_VX_warp_ctl_inter*	__PVT__Vortex__DOT__vx_front_end__DOT__VX_warp_ctl;
    VVortex_VX_inst_meta_inter*	__PVT__Vortex__DOT__vx_front_end__DOT__fe_inst_meta_fd;
    VVortex_VX_frE_to_bckE_req_inter*	__PVT__Vortex__DOT__vx_front_end__DOT__VX_frE_to_bckE_req;
    VVortex_VX_inst_meta_inter*	__PVT__Vortex__DOT__vx_front_end__DOT__fd_inst_meta_de;
    VVortex_VX_mem_req_inter*	__PVT__Vortex__DOT__vx_back_end__DOT__VX_exe_mem_req;
    VVortex_VX_mem_req_inter*	__PVT__Vortex__DOT__vx_back_end__DOT__VX_mem_req;
    VVortex_VX_inst_mem_wb_inter*	__PVT__Vortex__DOT__vx_back_end__DOT__VX_mem_wb;
    
    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    // Begin mtask footprint  all: 
    VL_IN8(clk,0,0);
    VL_IN8(reset,0,0);
    VL_OUT8(out_cache_driver_in_mem_read,2,0);
    VL_OUT8(out_cache_driver_in_mem_write,2,0);
    VL_OUT8(out_ebreak,0,0);
    VL_IN(icache_response_instruction,31,0);
    VL_OUT(icache_request_pc_address,31,0);
    VL_IN(in_cache_driver_out_data[4],31,0);
    VL_OUT(out_cache_driver_in_address[4],31,0);
    VL_OUT8(out_cache_driver_in_valid[4],0,0);
    VL_OUT(out_cache_driver_in_data[4],31,0);
    
    // LOCAL SIGNALS
    // Internals; generally not touched by application code
    // Anonymous structures to workaround compiler member-count bugs
    struct {
	// Begin mtask footprint  all: 
	VL_SIG8(Vortex__DOT__execute_branch_stall,0,0);
	VL_SIG8(Vortex__DOT__forwarding_fwd_stall,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__decode_branch_stall,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__stall,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_state,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_count,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__add_warp,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__remove_warp,0,0);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_valid,31,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_valid_var,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_change_mask,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__warp_zero_stall,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_change_mask,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__warp_zero_stall,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_change_mask,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__warp_zero_stall,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_change_mask,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__warp_zero_stall,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_change_mask,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__warp_zero_stall,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_change_mask,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__warp_zero_stall,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_change_mask,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__warp_zero_stall,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_change_mask,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__warp_zero_stall,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__valid_zero,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__valid_zero,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__valid_zero,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__valid_zero,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__valid_zero,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__valid_zero,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__valid_zero,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__valid_zero,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_itype,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_csr,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jalrs,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_jmprt,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_wspawn,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jal_sys_jal,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu,4,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jalrs_thread_mask,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__jmprt_thread_mask,3,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__is_ebreak,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__temp_final_alu,4,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__write_enable,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__write_enable,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__write_enable,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__write_enable,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__write_enable,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__write_enable,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__write_enable,0,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__write_enable,0,0);
	VL_SIG8(Vortex__DOT__vx_back_end__DOT__vx_memory__DOT__temp_branch_dir,0,0);
	VL_SIG8(Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd,0,0);
    };
    struct {
	VL_SIG8(Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd,0,0);
	VL_SIG8(Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd,0,0);
	VL_SIG8(Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd,0,0);
	VL_SIG8(Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd,0,0);
	VL_SIG8(Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd,0,0);
	VL_SIG16(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__alu_tempp,11,0);
	VL_SIGW(Vortex__DOT__vx_csr_handler__DOT__csr,12299,0,385);
	VL_SIG16(Vortex__DOT__vx_csr_handler__DOT__decode_csr_address,11,0);
	VL_SIG16(Vortex__DOT__vx_csr_handler__DOT__data_read,11,0);
	VL_SIG(Vortex__DOT__csr_decode_csr_data,31,0);
	VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_glob_pc,255,0,8);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__out_PC_var,31,0);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__real_PC,31,0);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__0__KET____DOT__VX_Warp__DOT__temp_PC,31,0);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__real_PC,31,0);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__1__KET____DOT__VX_Warp__DOT__temp_PC,31,0);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__real_PC,31,0);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__2__KET____DOT__VX_Warp__DOT__temp_PC,31,0);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__real_PC,31,0);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__3__KET____DOT__VX_Warp__DOT__temp_PC,31,0);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__real_PC,31,0);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__4__KET____DOT__VX_Warp__DOT__temp_PC,31,0);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__real_PC,31,0);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__5__KET____DOT__VX_Warp__DOT__temp_PC,31,0);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__real_PC,31,0);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__6__KET____DOT__VX_Warp__DOT__temp_PC,31,0);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__real_PC,31,0);
	VL_SIG(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__genblk1__BRA__7__KET____DOT__VX_Warp__DOT__temp_PC,31,0);
	VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_f_d_reg__DOT__f_d_reg__DOT__value,71,0,3);
	VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_a_reg_data,1023,0,32);
	VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__temp_b_reg_data,1023,0,32);
	VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__jal_data,127,0,4);
	VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT__d_e_reg__DOT__value,489,0,16);
	VL_SIG(Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2,31,0);
	VL_SIG(Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2,31,0);
	VL_SIG(Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__ALU_in2,31,0);
	VL_SIG(Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__ALU_in2,31,0);
	VL_SIGW(Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT__f_d_reg__DOT__value,463,0,15);
	VL_SIGW(Vortex__DOT__vx_back_end__DOT__vx_m_w_reg__DOT__m_w_reg__DOT__value,302,0,10);
	VL_SIGW(Vortex__DOT__vx_back_end__DOT__vx_writeback__DOT__out_pc_data,127,0,4);
	VL_SIGW(Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next,127,0,4);
	VL_SIGW(Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next,127,0,4);
	VL_SIGW(Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next,127,0,4);
	VL_SIG64(Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__mult_signed_result,63,0);
	VL_SIG64(Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__mult_signed_result,63,0);
	VL_SIG64(Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__2__KET____DOT__vx_alu__DOT__mult_signed_result,63,0);
	VL_SIG64(Vortex__DOT__vx_back_end__DOT__vx_execute__DOT__genblk1__BRA__3__KET____DOT__vx_alu__DOT__mult_signed_result,63,0);
	VL_SIG64(Vortex__DOT__vx_csr_handler__DOT__cycle,63,0);
	VL_SIG64(Vortex__DOT__vx_csr_handler__DOT__instret,63,0);
	VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__in_valid[4],0,0);
	VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__0__KET____DOT__vx_gpr__DOT__gpr[32],127,0,4);
	VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__1__KET____DOT__vx_gpr__DOT__gpr[32],127,0,4);
	VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__2__KET____DOT__vx_gpr__DOT__gpr[32],127,0,4);
	VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__3__KET____DOT__vx_gpr__DOT__gpr[32],127,0,4);
	VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__4__KET____DOT__vx_gpr__DOT__gpr[32],127,0,4);
	VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__5__KET____DOT__vx_gpr__DOT__gpr[32],127,0,4);
	VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__6__KET____DOT__vx_gpr__DOT__gpr[32],127,0,4);
	VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT__genblk2__BRA__7__KET____DOT__vx_gpr__DOT__gpr[32],127,0,4);
    };
    
    // LOCAL VARIABLES
    // Internals; generally not touched by application code
    // Begin mtask footprint  all: 
    VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__0__KET____DOT__VX_Warp__out_valid,3,0);
    VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__1__KET____DOT__VX_Warp__out_valid,3,0);
    VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__2__KET____DOT__VX_Warp__out_valid,3,0);
    VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__3__KET____DOT__VX_Warp__out_valid,3,0);
    VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__4__KET____DOT__VX_Warp__out_valid,3,0);
    VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__5__KET____DOT__VX_Warp__out_valid,3,0);
    VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__6__KET____DOT__VX_Warp__out_valid,3,0);
    VL_SIG8(Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT____Vcellout__genblk1__BRA__7__KET____DOT__VX_Warp__out_valid,3,0);
    VL_SIG8(__Vtableidx1,2,0);
    VL_SIG8(__Vdly__Vortex__DOT__vx_front_end__DOT__vx_fetch__DOT__warp_num,3,0);
    VL_SIG8(__Vclklast__TOP__clk,0,0);
    VL_SIG8(__Vclklast__TOP__reset,0,0);
    VL_SIG16(Vortex__DOT__vx_csr_handler__DOT____Vlvbound1,11,0);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_b_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT____Vcellout__vx_grp_wrapper__out_a_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT____Vcellout__genblk2__BRA__0__KET____DOT__vx_gpr__out_b_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT____Vcellout__genblk2__BRA__0__KET____DOT__vx_gpr__out_a_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT____Vcellout__genblk2__BRA__1__KET____DOT__vx_gpr__out_b_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT____Vcellout__genblk2__BRA__1__KET____DOT__vx_gpr__out_a_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT____Vcellout__genblk2__BRA__2__KET____DOT__vx_gpr__out_b_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT____Vcellout__genblk2__BRA__2__KET____DOT__vx_gpr__out_a_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT____Vcellout__genblk2__BRA__3__KET____DOT__vx_gpr__out_b_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT____Vcellout__genblk2__BRA__3__KET____DOT__vx_gpr__out_a_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT____Vcellout__genblk2__BRA__4__KET____DOT__vx_gpr__out_b_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT____Vcellout__genblk2__BRA__4__KET____DOT__vx_gpr__out_a_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT____Vcellout__genblk2__BRA__5__KET____DOT__vx_gpr__out_b_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT____Vcellout__genblk2__BRA__5__KET____DOT__vx_gpr__out_a_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT____Vcellout__genblk2__BRA__6__KET____DOT__vx_gpr__out_b_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT____Vcellout__genblk2__BRA__6__KET____DOT__vx_gpr__out_a_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT____Vcellout__genblk2__BRA__7__KET____DOT__vx_gpr__out_b_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__vx_grp_wrapper__DOT____Vcellout__genblk2__BRA__7__KET____DOT__vx_gpr__out_a_reg_data,127,0,4);
    VL_SIGW(Vortex__DOT__vx_front_end__DOT__vx_d_e_reg__DOT____Vcellinp__d_e_reg__in,489,0,16);
    VL_SIG(Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__0__KET____DOT__vx_alu__out_alu_result,31,0);
    VL_SIG(Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__1__KET____DOT__vx_alu__out_alu_result,31,0);
    VL_SIG(Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__2__KET____DOT__vx_alu__out_alu_result,31,0);
    VL_SIG(Vortex__DOT__vx_back_end__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__3__KET____DOT__vx_alu__out_alu_result,31,0);
    VL_SIGW(Vortex__DOT__vx_back_end__DOT__vx_e_m_reg__DOT____Vcellinp__f_d_reg__in,463,0,15);
    static VL_ST_SIG8(__Vtable1_Vortex__DOT__vx_front_end__DOT__vx_decode__DOT__mul_alu[8],4,0);
    
    // INTERNAL VARIABLES
    // Internals; generally not touched by application code
    VVortex__Syms* __VlSymsp;  // Symbol table
    
    // PARAMETERS
    // Parameters marked /*verilator public*/ for use by application code
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(VVortex);  ///< Copying not allowed
  public:
    /// Construct the model; called by application code
    /// The special name  may be used to make a wrapper with a
    /// single model invisible with respect to DPI scope names.
    VVortex(const char* name="TOP");
    /// Destroy the model; called (often implicitly) by application code
    ~VVortex();
    
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval();
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    
    // INTERNAL METHODS
  private:
    static void _eval_initial_loop(VVortex__Syms* __restrict vlSymsp);
  public:
    void __Vconfigure(VVortex__Syms* symsp, bool first);
  private:
    static QData _change_request(VVortex__Syms* __restrict vlSymsp);
  public:
    static void _combo__TOP__11(VVortex__Syms* __restrict vlSymsp);
    static void _combo__TOP__4(VVortex__Syms* __restrict vlSymsp);
    static void _combo__TOP__9(VVortex__Syms* __restrict vlSymsp);
  private:
    void _ctor_var_reset() VL_ATTR_COLD;
  public:
    static void _eval(VVortex__Syms* __restrict vlSymsp);
  private:
#ifdef VL_DEBUG
    void _eval_debug_assertions();
#endif // VL_DEBUG
  public:
    static void _eval_initial(VVortex__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    static void _eval_settle(VVortex__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    static void _initial__TOP__1(VVortex__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    static void _multiclk__TOP__10(VVortex__Syms* __restrict vlSymsp);
    static void _multiclk__TOP__8(VVortex__Syms* __restrict vlSymsp);
    static void _sequent__TOP__3(VVortex__Syms* __restrict vlSymsp);
    static void _sequent__TOP__5(VVortex__Syms* __restrict vlSymsp);
    static void _sequent__TOP__6(VVortex__Syms* __restrict vlSymsp);
    static void _sequent__TOP__7(VVortex__Syms* __restrict vlSymsp);
    static void _settle__TOP__2(VVortex__Syms* __restrict vlSymsp) VL_ATTR_COLD;
} VL_ATTR_ALIGNED(128);

#endif // guard
