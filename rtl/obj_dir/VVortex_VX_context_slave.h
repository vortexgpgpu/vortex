// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See VVortex.h for the primary calling header

#ifndef _VVortex_VX_context_slave_H_
#define _VVortex_VX_context_slave_H_

#include "verilated.h"

class VVortex__Syms;

//----------

VL_MODULE(VVortex_VX_context_slave) {
  public:
    
    // PORTS
    // Begin mtask footprint  all: 
    VL_IN8(clk,0,0);
    VL_IN8(in_warp,0,0);
    VL_IN8(in_wb_warp,0,0);
    VL_IN8(in_write_register,0,0);
    VL_IN8(in_rd,4,0);
    VL_IN8(in_src1,4,0);
    VL_IN8(in_src2,4,0);
    VL_IN8(in_is_clone,0,0);
    VL_IN8(in_is_jal,0,0);
    VL_IN8(in_src1_fwd,0,0);
    VL_IN8(in_src2_fwd,0,0);
    VL_IN8(in_wspawn,0,0);
    VL_OUT8(out_clone_stall,0,0);
    VL_IN(in_curr_PC,31,0);
    VL_IN8(in_valid[4],0,0);
    VL_IN(in_write_data[4],31,0);
    VL_IN(in_src1_fwd_data[4],31,0);
    VL_IN(in_src2_fwd_data[4],31,0);
    VL_IN(in_wspawn_regs[32],31,0);
    VL_OUT(out_a_reg_data[4],31,0);
    VL_OUT(out_b_reg_data[4],31,0);
    
    // LOCAL SIGNALS
    // Begin mtask footprint  all: 
    VL_SIG8(__PVT__clone_state_stall,5,0);
    VL_SIG8(__PVT__wspawn_state_stall,5,0);
    VL_SIG(__PVT__rd1_register[4],31,0);
    VL_SIG(__PVT__rd2_register[4],31,0);
    VL_SIG(__PVT__clone_regsiters[32],31,0);
    VL_SIG(__PVT__vx_register_file_master__DOT__registers[32],31,0);
    VL_SIG(__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[32],31,0);
    VL_SIG(__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[32],31,0);
    VL_SIG(__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[32],31,0);
    
    // LOCAL VARIABLES
    // Anonymous structures to workaround compiler member-count bugs
    struct {
	// Begin mtask footprint  all: 
	VL_SIG8(__Vdly__clone_state_stall,5,0);
	VL_SIG8(__Vdly__wspawn_state_stall,5,0);
	VL_SIG8(__Vdlyvdim0__vx_register_file_master__DOT__registers__v0,4,0);
	VL_SIG8(__Vdlyvset__vx_register_file_master__DOT__registers__v0,0,0);
	VL_SIG8(__Vdlyvset__vx_register_file_master__DOT__registers__v1,0,0);
	VL_SIG8(__Vdlyvdim0__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v0,4,0);
	VL_SIG8(__Vdlyvset__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v0,0,0);
	VL_SIG8(__Vdlyvset__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v1,0,0);
	VL_SIG8(__Vdlyvdim0__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v0,4,0);
	VL_SIG8(__Vdlyvset__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v0,0,0);
	VL_SIG8(__Vdlyvset__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v1,0,0);
	VL_SIG8(__Vdlyvdim0__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v0,4,0);
	VL_SIG8(__Vdlyvset__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v0,0,0);
	VL_SIG8(__Vdlyvset__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v1,0,0);
	VL_SIG(__Vcellout__vx_register_file_master__out_src2_data,31,0);
	VL_SIG(__Vcellout__vx_register_file_master__out_src1_data,31,0);
	VL_SIG(__Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src2_data,31,0);
	VL_SIG(__Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src1_data,31,0);
	VL_SIG(__Vcellout__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__out_src2_data,31,0);
	VL_SIG(__Vcellout__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__out_src1_data,31,0);
	VL_SIG(__Vcellout__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__out_src2_data,31,0);
	VL_SIG(__Vcellout__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__out_src1_data,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v0,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v1,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v2,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v3,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v4,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v5,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v6,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v7,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v8,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v9,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v10,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v11,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v12,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v13,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v14,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v15,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v16,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v17,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v18,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v19,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v20,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v21,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v22,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v23,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v24,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v25,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v26,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v27,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v28,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v29,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v30,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v31,31,0);
	VL_SIG(__Vdlyvval__vx_register_file_master__DOT__registers__v32,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v0,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v1,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v2,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v3,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v4,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v5,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v6,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v7,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v8,31,0);
    };
    struct {
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v9,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v10,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v11,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v12,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v13,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v14,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v15,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v16,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v17,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v18,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v19,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v20,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v21,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v22,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v23,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v24,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v25,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v26,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v27,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v28,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v29,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v30,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v31,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers__v32,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v0,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v1,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v2,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v3,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v4,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v5,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v6,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v7,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v8,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v9,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v10,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v11,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v12,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v13,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v14,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v15,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v16,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v17,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v18,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v19,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v20,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v21,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v22,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v23,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v24,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v25,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v26,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v27,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v28,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v29,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v30,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v31,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers__v32,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v0,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v1,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v2,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v3,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v4,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v5,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v6,31,0);
    };
    struct {
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v7,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v8,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v9,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v10,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v11,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v12,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v13,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v14,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v15,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v16,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v17,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v18,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v19,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v20,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v21,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v22,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v23,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v24,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v25,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v26,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v27,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v28,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v29,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v30,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v31,31,0);
	VL_SIG(__Vdlyvval__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers__v32,31,0);
	VL_SIG(__Vcellout__vx_register_file_master__out_regs[32],31,0);
	VL_SIG(__Vcellinp__vx_register_file_master__in_wspawn_regs[32],31,0);
	VL_SIG(__Vcellinp__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__in_regs[32],31,0);
	VL_SIG(__Vcellinp__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__in_regs[32],31,0);
	VL_SIG(__Vcellinp__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__in_regs[32],31,0);
    };
    
    // INTERNAL VARIABLES
  private:
    VVortex__Syms* __VlSymsp;  // Symbol table
  public:
    
    // PARAMETERS
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(VVortex_VX_context_slave);  ///< Copying not allowed
  public:
    VVortex_VX_context_slave(const char* name="TOP");
    ~VVortex_VX_context_slave();
    
    // API METHODS
    
    // INTERNAL METHODS
    void __Vconfigure(VVortex__Syms* symsp, bool first);
    void _combo__TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__1__KET____DOT__VX_Context_one__29(VVortex__Syms* __restrict vlSymsp);
    void _combo__TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__1__KET____DOT__VX_Context_one__36(VVortex__Syms* __restrict vlSymsp);
  private:
    void _ctor_var_reset();
  public:
    void _initial__TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__1__KET____DOT__VX_Context_one__1(VVortex__Syms* __restrict vlSymsp);
    void _sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__1__KET____DOT__VX_Context_one__15(VVortex__Syms* __restrict vlSymsp);
    void _sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__1__KET____DOT__VX_Context_one__22(VVortex__Syms* __restrict vlSymsp);
    void _sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__2__KET____DOT__VX_Context_one__16(VVortex__Syms* __restrict vlSymsp);
    void _sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__3__KET____DOT__VX_Context_one__17(VVortex__Syms* __restrict vlSymsp);
    void _sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__4__KET____DOT__VX_Context_one__18(VVortex__Syms* __restrict vlSymsp);
    void _sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__5__KET____DOT__VX_Context_one__19(VVortex__Syms* __restrict vlSymsp);
    void _sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__6__KET____DOT__VX_Context_one__20(VVortex__Syms* __restrict vlSymsp);
    void _sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__7__KET____DOT__VX_Context_one__21(VVortex__Syms* __restrict vlSymsp);
    void _settle__TOP__Vortex__DOT__vx_decode__DOT__genblk1__BRA__1__KET____DOT__VX_Context_one__8(VVortex__Syms* __restrict vlSymsp);
} VL_ATTR_ALIGNED(128);

#endif // guard
