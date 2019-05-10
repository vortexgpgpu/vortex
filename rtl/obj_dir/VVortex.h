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

//----------

VL_MODULE(VVortex) {
  public:
    
    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    // Begin mtask footprint  all: 
    VL_IN8(clk,0,0);
    VL_IN8(reset,0,0);
    VL_OUT8(out_cache_driver_in_mem_read,2,0);
    VL_OUT8(out_cache_driver_in_mem_write,2,0);
    VL_OUT8(out_ebreak,0,0);
    VL_IN(fe_instruction,31,0);
    VL_OUT(curr_PC,31,0);
    VL_IN(in_cache_driver_out_data[2],31,0);
    VL_OUT(out_cache_driver_in_address[2],31,0);
    VL_OUT8(out_cache_driver_in_valid[2],0,0);
    VL_OUT(out_cache_driver_in_data[2],31,0);
    
    // LOCAL SIGNALS
    // Internals; generally not touched by application code
    // Anonymous structures to workaround compiler member-count bugs
    struct {
	// Begin mtask footprint  all: 
	VL_SIG8(Vortex__DOT__decode_branch_stall,0,0);
	VL_SIG8(Vortex__DOT__decode_branch_type,2,0);
	VL_SIG8(Vortex__DOT__decode_jal,0,0);
	VL_SIG8(Vortex__DOT__decode_clone_stall,0,0);
	VL_SIG8(Vortex__DOT__decode_change_mask,0,0);
	VL_SIG8(Vortex__DOT__execute_branch_stall,0,0);
	VL_SIG8(Vortex__DOT__memory_branch_dir,0,0);
	VL_SIG8(Vortex__DOT__forwarding_fwd_stall,0,0);
	VL_SIG8(Vortex__DOT__forwarding_src1_fwd,0,0);
	VL_SIG8(Vortex__DOT__forwarding_src2_fwd,0,0);
	VL_SIG8(Vortex__DOT__vx_fetch__DOT__stall,0,0);
	VL_SIG8(Vortex__DOT__vx_fetch__DOT__warp_num,1,0);
	VL_SIG8(Vortex__DOT__vx_fetch__DOT__warp_state,1,0);
	VL_SIG8(Vortex__DOT__vx_fetch__DOT__add_warp,0,0);
	VL_SIG8(Vortex__DOT__vx_fetch__DOT__remove_warp,0,0);
	VL_SIG8(Vortex__DOT__vx_fetch__DOT__warp_zero_change_mask,0,0);
	VL_SIG8(Vortex__DOT__vx_fetch__DOT__warp_zero_stall,0,0);
	VL_SIG8(Vortex__DOT__vx_fetch__DOT__warp_one_change_mask,0,0);
	VL_SIG8(Vortex__DOT__vx_fetch__DOT__warp_one_stall,0,0);
	VL_SIG8(Vortex__DOT__vx_f_d_reg__DOT__warp_num,1,0);
	VL_SIG8(Vortex__DOT__vx_decode__DOT__is_itype,0,0);
	VL_SIG8(Vortex__DOT__vx_decode__DOT__is_csr,0,0);
	VL_SIG8(Vortex__DOT__vx_decode__DOT__is_clone,0,0);
	VL_SIG8(Vortex__DOT__vx_decode__DOT__is_jalrs,0,0);
	VL_SIG8(Vortex__DOT__vx_decode__DOT__is_jmprt,0,0);
	VL_SIG8(Vortex__DOT__vx_decode__DOT__is_wspawn,0,0);
	VL_SIG8(Vortex__DOT__vx_decode__DOT__jal_sys_jal,0,0);
	VL_SIG8(Vortex__DOT__vx_decode__DOT__mul_alu,4,0);
	VL_SIG8(Vortex__DOT__vx_decode__DOT__is_ebreak,0,0);
	VL_SIG8(Vortex__DOT__vx_decode__DOT__temp_final_alu,4,0);
	VL_SIG8(Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__state_stall,5,0);
	VL_SIG8(Vortex__DOT__vx_decode__DOT__VX_Context_one__DOT__clone_state_stall,5,0);
	VL_SIG8(Vortex__DOT__vx_decode__DOT__VX_Context_one__DOT__wspawn_state_stall,5,0);
	VL_SIG8(Vortex__DOT__vx_d_e_reg__DOT__rd,4,0);
	VL_SIG8(Vortex__DOT__vx_d_e_reg__DOT__alu_op,4,0);
	VL_SIG8(Vortex__DOT__vx_d_e_reg__DOT__wb,1,0);
	VL_SIG8(Vortex__DOT__vx_d_e_reg__DOT__rs2_src,0,0);
	VL_SIG8(Vortex__DOT__vx_d_e_reg__DOT__mem_read,2,0);
	VL_SIG8(Vortex__DOT__vx_d_e_reg__DOT__mem_write,2,0);
	VL_SIG8(Vortex__DOT__vx_d_e_reg__DOT__branch_type,2,0);
	VL_SIG8(Vortex__DOT__vx_d_e_reg__DOT__is_csr,0,0);
	VL_SIG8(Vortex__DOT__vx_d_e_reg__DOT__jal,0,0);
	VL_SIG8(Vortex__DOT__vx_d_e_reg__DOT__warp_num,1,0);
	VL_SIG8(Vortex__DOT__vx_d_e_reg__DOT__stalling,0,0);
	VL_SIG8(Vortex__DOT__vx_e_m_reg__DOT__rd,4,0);
	VL_SIG8(Vortex__DOT__vx_e_m_reg__DOT__wb,1,0);
	VL_SIG8(Vortex__DOT__vx_e_m_reg__DOT__mem_read,2,0);
	VL_SIG8(Vortex__DOT__vx_e_m_reg__DOT__mem_write,2,0);
	VL_SIG8(Vortex__DOT__vx_e_m_reg__DOT__is_csr,0,0);
	VL_SIG8(Vortex__DOT__vx_e_m_reg__DOT__branch_type,2,0);
	VL_SIG8(Vortex__DOT__vx_e_m_reg__DOT__jal,0,0);
	VL_SIG8(Vortex__DOT__vx_e_m_reg__DOT__warp_num,1,0);
	VL_SIG8(Vortex__DOT__vx_m_w_reg__DOT__rd,4,0);
	VL_SIG8(Vortex__DOT__vx_m_w_reg__DOT__wb,1,0);
	VL_SIG8(Vortex__DOT__vx_m_w_reg__DOT__warp_num,1,0);
	VL_SIG8(Vortex__DOT__vx_forwarding__DOT__src1_exe_fwd,0,0);
	VL_SIG8(Vortex__DOT__vx_forwarding__DOT__src1_mem_fwd,0,0);
	VL_SIG8(Vortex__DOT__vx_forwarding__DOT__src1_wb_fwd,0,0);
	VL_SIG8(Vortex__DOT__vx_forwarding__DOT__src2_exe_fwd,0,0);
	VL_SIG8(Vortex__DOT__vx_forwarding__DOT__src2_mem_fwd,0,0);
	VL_SIG8(Vortex__DOT__vx_forwarding__DOT__src2_wb_fwd,0,0);
	VL_SIG16(Vortex__DOT__decode_csr_address,11,0);
	VL_SIG16(Vortex__DOT__vx_decode__DOT__alu_tempp,11,0);
	VL_SIG16(Vortex__DOT__vx_d_e_reg__DOT__csr_address,11,0);
    };
    struct {
	VL_SIG16(Vortex__DOT__vx_e_m_reg__DOT__csr_address,11,0);
	VL_SIG16(Vortex__DOT__vx_csr_handler__DOT__decode_csr_address,11,0);
	VL_SIG(Vortex__DOT__decode_itype_immed,31,0);
	VL_SIG(Vortex__DOT__decode_jal_offset,31,0);
	VL_SIG(Vortex__DOT__memory_branch_dest,31,0);
	VL_SIG(Vortex__DOT__csr_decode_csr_data,31,0);
	VL_SIG(Vortex__DOT__vx_fetch__DOT__out_PC,31,0);
	VL_SIG(Vortex__DOT__vx_fetch__DOT__VX_Warp_zero__DOT__real_PC,31,0);
	VL_SIG(Vortex__DOT__vx_fetch__DOT__VX_Warp_zero__DOT__temp_PC,31,0);
	VL_SIG(Vortex__DOT__vx_fetch__DOT__VX_Warp_one__DOT__real_PC,31,0);
	VL_SIG(Vortex__DOT__vx_fetch__DOT__VX_Warp_one__DOT__temp_PC,31,0);
	VL_SIG(Vortex__DOT__vx_f_d_reg__DOT__instruction,31,0);
	VL_SIG(Vortex__DOT__vx_f_d_reg__DOT__curr_PC,31,0);
	VL_SIG(Vortex__DOT__vx_d_e_reg__DOT__PC_next_out,31,0);
	VL_SIG(Vortex__DOT__vx_d_e_reg__DOT__itype_immed,31,0);
	VL_SIG(Vortex__DOT__vx_d_e_reg__DOT__upper_immed,19,0);
	VL_SIG(Vortex__DOT__vx_d_e_reg__DOT__csr_mask,31,0);
	VL_SIG(Vortex__DOT__vx_d_e_reg__DOT__curr_PC,31,0);
	VL_SIG(Vortex__DOT__vx_d_e_reg__DOT__jal_offset,31,0);
	VL_SIG(Vortex__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__ALU_in2,31,0);
	VL_SIG(Vortex__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__ALU_in2,31,0);
	VL_SIG(Vortex__DOT__vx_e_m_reg__DOT__PC_next,31,0);
	VL_SIG(Vortex__DOT__vx_e_m_reg__DOT__csr_result,31,0);
	VL_SIG(Vortex__DOT__vx_e_m_reg__DOT__curr_PC,31,0);
	VL_SIG(Vortex__DOT__vx_e_m_reg__DOT__branch_offset,31,0);
	VL_SIG(Vortex__DOT__vx_e_m_reg__DOT__jal_dest,31,0);
	VL_SIG(Vortex__DOT__vx_m_w_reg__DOT__PC_next,31,0);
	VL_SIG64(Vortex__DOT__vx_execute__DOT__genblk1__BRA__0__KET____DOT__vx_alu__DOT__mult_signed_result,63,0);
	VL_SIG64(Vortex__DOT__vx_execute__DOT__genblk1__BRA__1__KET____DOT__vx_alu__DOT__mult_signed_result,63,0);
	VL_SIG64(Vortex__DOT__vx_csr_handler__DOT__cycle,63,0);
	VL_SIG64(Vortex__DOT__vx_csr_handler__DOT__instret,63,0);
	VL_SIG8(Vortex__DOT__fetch_valid[2],0,0);
	VL_SIG8(Vortex__DOT__f_d_valid[2],0,0);
	VL_SIG(Vortex__DOT__decode_a_reg_data[2],31,0);
	VL_SIG(Vortex__DOT__decode_b_reg_data[2],31,0);
	VL_SIG8(Vortex__DOT__decode_valid[2],0,0);
	VL_SIG8(Vortex__DOT__decode_thread_mask[2],0,0);
	VL_SIG(Vortex__DOT__d_e_a_reg_data[2],31,0);
	VL_SIG(Vortex__DOT__d_e_b_reg_data[2],31,0);
	VL_SIG8(Vortex__DOT__d_e_valid[2],0,0);
	VL_SIG(Vortex__DOT__execute_alu_result[2],31,0);
	VL_SIG(Vortex__DOT__execute_b_reg_data[2],31,0);
	VL_SIG8(Vortex__DOT__execute_valid[2],0,0);
	VL_SIG(Vortex__DOT__e_m_alu_result[2],31,0);
	VL_SIG(Vortex__DOT__e_m_b_reg_data[2],31,0);
	VL_SIG8(Vortex__DOT__e_m_valid[2],0,0);
	VL_SIG(Vortex__DOT__memory_alu_result[2],31,0);
	VL_SIG(Vortex__DOT__memory_mem_result[2],31,0);
	VL_SIG8(Vortex__DOT__memory_valid[2],0,0);
	VL_SIG(Vortex__DOT__m_w_alu_result[2],31,0);
	VL_SIG(Vortex__DOT__m_w_mem_result[2],31,0);
	VL_SIG8(Vortex__DOT__m_w_valid[2],0,0);
	VL_SIG(Vortex__DOT__writeback_write_data[2],31,0);
	VL_SIG(Vortex__DOT__forwarding_src1_fwd_data[2],31,0);
	VL_SIG(Vortex__DOT__forwarding_src2_fwd_data[2],31,0);
	VL_SIG8(Vortex__DOT__vx_fetch__DOT__warp_zero_valid[2],0,0);
	VL_SIG8(Vortex__DOT__vx_fetch__DOT__warp_one_valid[2],0,0);
	VL_SIG8(Vortex__DOT__vx_fetch__DOT__VX_Warp_zero__DOT__valid[2],0,0);
	VL_SIG8(Vortex__DOT__vx_fetch__DOT__VX_Warp_one__DOT__valid[2],0,0);
	VL_SIG8(Vortex__DOT__vx_f_d_reg__DOT__valid[2],0,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__w0_t0_registers[32],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__zero_a_reg_data[2],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__zero_b_reg_data[2],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__one_a_reg_data[2],31,0);
    };
    struct {
	VL_SIG(Vortex__DOT__vx_decode__DOT__one_b_reg_data[2],31,0);
	VL_SIG8(Vortex__DOT__vx_decode__DOT__jalrs_thread_mask[2],0,0);
	VL_SIG8(Vortex__DOT__vx_decode__DOT__jmprt_thread_mask[2],0,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__rd1_register[2],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__rd2_register[2],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__clone_regsiters[32],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__vx_register_file_master__DOT__registers[32],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[32],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_one__DOT__rd1_register[2],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_one__DOT__rd2_register[2],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_one__DOT__clone_regsiters[32],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_one__DOT__vx_register_file_master__DOT__registers[32],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_one__DOT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[32],31,0);
	VL_SIG(Vortex__DOT__vx_d_e_reg__DOT__a_reg_data[2],31,0);
	VL_SIG(Vortex__DOT__vx_d_e_reg__DOT__b_reg_data[2],31,0);
	VL_SIG8(Vortex__DOT__vx_d_e_reg__DOT__valid[2],0,0);
	VL_SIG(Vortex__DOT__vx_d_e_reg__DOT__reg_data_z[2],31,0);
	VL_SIG8(Vortex__DOT__vx_d_e_reg__DOT__valid_z[2],0,0);
	VL_SIG(Vortex__DOT__vx_e_m_reg__DOT__alu_result[2],31,0);
	VL_SIG(Vortex__DOT__vx_e_m_reg__DOT__a_reg_data[2],31,0);
	VL_SIG(Vortex__DOT__vx_e_m_reg__DOT__b_reg_data[2],31,0);
	VL_SIG8(Vortex__DOT__vx_e_m_reg__DOT__valid[2],0,0);
	VL_SIG(Vortex__DOT__vx_m_w_reg__DOT__alu_result[2],31,0);
	VL_SIG(Vortex__DOT__vx_m_w_reg__DOT__mem_result[2],31,0);
	VL_SIG8(Vortex__DOT__vx_m_w_reg__DOT__valid[2],0,0);
	VL_SIG(Vortex__DOT__vx_writeback__DOT__out_pc_data[2],31,0);
	VL_SIG(Vortex__DOT__vx_forwarding__DOT__use_execute_PC_next[2],31,0);
	VL_SIG(Vortex__DOT__vx_forwarding__DOT__use_memory_PC_next[2],31,0);
	VL_SIG(Vortex__DOT__vx_forwarding__DOT__use_writeback_PC_next[2],31,0);
	VL_SIG16(Vortex__DOT__vx_csr_handler__DOT__csr[4096],11,0);
    };
    
    // LOCAL VARIABLES
    // Internals; generally not touched by application code
    // Anonymous structures to workaround compiler member-count bugs
    struct {
	// Begin mtask footprint  all: 
	VL_SIG8(__Vtableidx1,2,0);
	VL_SIG8(__Vclklast__TOP__clk,0,0);
	VL_SIG8(__Vclklast__TOP__reset,0,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT____Vcellout__vx_register_file_master__out_src2_data,31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT____Vcellout__vx_register_file_master__out_src1_data,31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT____Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src2_data,31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT____Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src1_data,31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_one__DOT____Vcellout__vx_register_file_master__out_src2_data,31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_one__DOT____Vcellout__vx_register_file_master__out_src1_data,31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_one__DOT____Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src2_data,31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_one__DOT____Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src1_data,31,0);
	VL_SIG(Vortex__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__0__KET____DOT__vx_alu__out_alu_result,31,0);
	VL_SIG(Vortex__DOT__vx_execute__DOT____Vcellout__genblk1__BRA__1__KET____DOT__vx_alu__out_alu_result,31,0);
	VL_SIG8(Vortex__DOT____Vcellout__vx_fetch__out_valid[2],0,0);
	VL_SIG8(Vortex__DOT____Vcellinp__vx_fetch__in_thread_mask[2],0,0);
	VL_SIG8(Vortex__DOT____Vcellout__vx_f_d_reg__out_valid[2],0,0);
	VL_SIG8(Vortex__DOT____Vcellinp__vx_f_d_reg__in_valid[2],0,0);
	VL_SIG8(Vortex__DOT____Vcellout__vx_decode__out_thread_mask[2],0,0);
	VL_SIG8(Vortex__DOT____Vcellout__vx_decode__out_valid[2],0,0);
	VL_SIG(Vortex__DOT____Vcellout__vx_decode__out_b_reg_data[2],31,0);
	VL_SIG(Vortex__DOT____Vcellout__vx_decode__out_a_reg_data[2],31,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_decode__in_src2_fwd_data[2],31,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_decode__in_src1_fwd_data[2],31,0);
	VL_SIG8(Vortex__DOT____Vcellinp__vx_decode__in_wb_valid[2],0,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_decode__in_write_data[2],31,0);
	VL_SIG8(Vortex__DOT____Vcellinp__vx_decode__in_valid[2],0,0);
	VL_SIG8(Vortex__DOT____Vcellout__vx_d_e_reg__out_valid[2],0,0);
	VL_SIG(Vortex__DOT____Vcellout__vx_d_e_reg__out_b_reg_data[2],31,0);
	VL_SIG(Vortex__DOT____Vcellout__vx_d_e_reg__out_a_reg_data[2],31,0);
	VL_SIG8(Vortex__DOT____Vcellinp__vx_d_e_reg__in_valid[2],0,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_d_e_reg__in_b_reg_data[2],31,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_d_e_reg__in_a_reg_data[2],31,0);
	VL_SIG8(Vortex__DOT____Vcellout__vx_execute__out_valid[2],0,0);
	VL_SIG(Vortex__DOT____Vcellout__vx_execute__out_b_reg_data[2],31,0);
	VL_SIG(Vortex__DOT____Vcellout__vx_execute__out_alu_result[2],31,0);
	VL_SIG8(Vortex__DOT____Vcellinp__vx_execute__in_valid[2],0,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_execute__in_b_reg_data[2],31,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_execute__in_a_reg_data[2],31,0);
	VL_SIG8(Vortex__DOT____Vcellout__vx_e_m_reg__out_valid[2],0,0);
	VL_SIG(Vortex__DOT____Vcellout__vx_e_m_reg__out_b_reg_data[2],31,0);
	VL_SIG(Vortex__DOT____Vcellout__vx_e_m_reg__out_alu_result[2],31,0);
	VL_SIG8(Vortex__DOT____Vcellinp__vx_e_m_reg__in_valid[2],0,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_e_m_reg__in_b_reg_data[2],31,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_e_m_reg__in_alu_result[2],31,0);
	VL_SIG8(Vortex__DOT____Vcellout__vx_memory__out_cache_driver_in_valid[2],0,0);
	VL_SIG(Vortex__DOT____Vcellout__vx_memory__out_cache_driver_in_data[2],31,0);
	VL_SIG(Vortex__DOT____Vcellout__vx_memory__out_cache_driver_in_address[2],31,0);
	VL_SIG8(Vortex__DOT____Vcellout__vx_memory__out_valid[2],0,0);
	VL_SIG(Vortex__DOT____Vcellout__vx_memory__out_mem_result[2],31,0);
	VL_SIG(Vortex__DOT____Vcellout__vx_memory__out_alu_result[2],31,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_memory__in_cache_driver_out_data[2],31,0);
	VL_SIG8(Vortex__DOT____Vcellinp__vx_memory__in_valid[2],0,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_memory__in_rd2[2],31,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_memory__in_alu_result[2],31,0);
	VL_SIG8(Vortex__DOT____Vcellout__vx_m_w_reg__out_valid[2],0,0);
	VL_SIG(Vortex__DOT____Vcellout__vx_m_w_reg__out_mem_result[2],31,0);
	VL_SIG(Vortex__DOT____Vcellout__vx_m_w_reg__out_alu_result[2],31,0);
	VL_SIG8(Vortex__DOT____Vcellinp__vx_m_w_reg__in_valid[2],0,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_m_w_reg__in_mem_result[2],31,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_m_w_reg__in_alu_result[2],31,0);
	VL_SIG(Vortex__DOT____Vcellout__vx_writeback__out_write_data[2],31,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_writeback__in_mem_result[2],31,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_writeback__in_alu_result[2],31,0);
	VL_SIG(Vortex__DOT____Vcellout__vx_forwarding__out_src2_fwd_data[2],31,0);
    };
    struct {
	VL_SIG(Vortex__DOT____Vcellout__vx_forwarding__out_src1_fwd_data[2],31,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_forwarding__in_writeback_mem_data[2],31,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_forwarding__in_writeback_alu_result[2],31,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_forwarding__in_memory_mem_data[2],31,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_forwarding__in_memory_alu_result[2],31,0);
	VL_SIG(Vortex__DOT____Vcellinp__vx_forwarding__in_execute_alu_result[2],31,0);
	VL_SIG8(Vortex__DOT__vx_fetch__DOT____Vcellout__VX_Warp_zero__out_valid[2],0,0);
	VL_SIG8(Vortex__DOT__vx_fetch__DOT____Vcellinp__VX_Warp_zero__in_thread_mask[2],0,0);
	VL_SIG8(Vortex__DOT__vx_fetch__DOT____Vcellout__VX_Warp_one__out_valid[2],0,0);
	VL_SIG8(Vortex__DOT__vx_fetch__DOT____Vcellinp__VX_Warp_one__in_thread_mask[2],0,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT____Vcellout__VX_Context_zero__w0_t0_registers[32],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT____Vcellout__VX_Context_zero__out_b_reg_data[2],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT____Vcellout__VX_Context_zero__out_a_reg_data[2],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT____Vcellinp__VX_Context_zero__in_write_data[2],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT____Vcellinp__VX_Context_zero__in_src2_fwd_data[2],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT____Vcellinp__VX_Context_zero__in_src1_fwd_data[2],31,0);
	VL_SIG8(Vortex__DOT__vx_decode__DOT____Vcellinp__VX_Context_zero__in_valid[2],0,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT____Vcellout__VX_Context_one__out_b_reg_data[2],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT____Vcellout__VX_Context_one__out_a_reg_data[2],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT____Vcellinp__VX_Context_one__in_wspawn_regs[32],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT____Vcellinp__VX_Context_one__in_write_data[2],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT____Vcellinp__VX_Context_one__in_src2_fwd_data[2],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT____Vcellinp__VX_Context_one__in_src1_fwd_data[2],31,0);
	VL_SIG8(Vortex__DOT__vx_decode__DOT____Vcellinp__VX_Context_one__in_valid[2],0,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT____Vcellout__vx_register_file_master__out_regs[32],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_zero__DOT____Vcellinp__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__in_regs[32],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_one__DOT____Vcellout__vx_register_file_master__out_regs[32],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_one__DOT____Vcellinp__vx_register_file_master__in_wspawn_regs[32],31,0);
	VL_SIG(Vortex__DOT__vx_decode__DOT__VX_Context_one__DOT____Vcellinp__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__in_regs[32],31,0);
    };
    static VL_ST_SIG8(__Vtable1_Vortex__DOT__vx_decode__DOT__mul_alu[8],4,0);
    
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
    static void _combo__TOP__5(VVortex__Syms* __restrict vlSymsp);
    static void _combo__TOP__8(VVortex__Syms* __restrict vlSymsp);
  private:
    void _ctor_var_reset();
  public:
    static void _eval(VVortex__Syms* __restrict vlSymsp);
  private:
#ifdef VL_DEBUG
    void _eval_debug_assertions();
#endif // VL_DEBUG
  public:
    static void _eval_initial(VVortex__Syms* __restrict vlSymsp);
    static void _eval_settle(VVortex__Syms* __restrict vlSymsp);
    static void _initial__TOP__1(VVortex__Syms* __restrict vlSymsp);
    static void _sequent__TOP__3(VVortex__Syms* __restrict vlSymsp);
    static void _sequent__TOP__4(VVortex__Syms* __restrict vlSymsp);
    static void _sequent__TOP__6(VVortex__Syms* __restrict vlSymsp);
    static void _sequent__TOP__7(VVortex__Syms* __restrict vlSymsp);
    static void _settle__TOP__2(VVortex__Syms* __restrict vlSymsp);
} VL_ATTR_ALIGNED(128);

#endif // guard
