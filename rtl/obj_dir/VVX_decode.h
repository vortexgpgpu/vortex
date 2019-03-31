// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary design header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef _VVX_decode_H_
#define _VVX_decode_H_

#include "verilated_heavy.h"

class VVX_decode__Syms;

//----------

VL_MODULE(VVX_decode) {
  public:
    
    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    // Begin mtask footprint  all: 
    VL_IN8(clk,0,0);
    VL_IN8(in_rd,4,0);
    VL_IN8(in_wb,1,0);
    VL_IN8(in_src1_fwd,0,0);
    VL_IN8(in_src2_fwd,0,0);
    VL_OUT8(out_is_csr,0,0);
    VL_OUT8(out_rd,4,0);
    VL_OUT8(out_rs1,4,0);
    VL_OUT8(out_rs2,4,0);
    VL_OUT8(out_wb,1,0);
    VL_OUT8(out_alu_op,4,0);
    VL_OUT8(out_rs2_src,0,0);
    VL_OUT8(out_mem_read,2,0);
    VL_OUT8(out_mem_write,2,0);
    VL_OUT8(out_branch_type,2,0);
    VL_OUT8(out_branch_stall,0,0);
    VL_OUT8(out_jal,0,0);
    VL_OUT8(out_clone_stall,0,0);
    VL_OUT16(out_csr_address,11,0);
    VL_IN(in_instruction,31,0);
    VL_IN(in_curr_PC,31,0);
    VL_OUT(out_csr_mask,31,0);
    VL_OUT(out_itype_immed,31,0);
    VL_OUT(out_jal_offset,31,0);
    VL_OUT(out_upper_immed,19,0);
    VL_OUT(out_PC_next,31,0);
    VL_IN8(in_valid[5],0,0);
    VL_IN(in_write_data[5],31,0);
    VL_IN8(in_wb_valid[5],0,0);
    VL_IN(in_src1_fwd_data[5],31,0);
    VL_IN(in_src2_fwd_data[5],31,0);
    VL_OUT(out_a_reg_data[5],31,0);
    VL_OUT(out_b_reg_data[5],31,0);
    VL_OUT8(out_valid[5],0,0);
    
    // LOCAL SIGNALS
    // Internals; generally not touched by application code
    // Begin mtask footprint  all: 
    VL_SIG8(VX_decode__DOT__is_itype,0,0);
    VL_SIG8(VX_decode__DOT__is_csr,0,0);
    VL_SIG8(VX_decode__DOT__is_clone,0,0);
    VL_SIG8(VX_decode__DOT__mul_alu,4,0);
    VL_SIG8(VX_decode__DOT__state_stall,2,0);
    VL_SIG8(VX_decode__DOT__temp_final_alu,4,0);
    VL_SIG16(VX_decode__DOT__jalr_immed,11,0);
    VL_SIG16(VX_decode__DOT__alu_tempp,11,0);
    VL_SIG(VX_decode__DOT__rd1_register[5],31,0);
    VL_SIG(VX_decode__DOT__rd2_register[5],31,0);
    VL_SIG(VX_decode__DOT__clone_regsiters[32],31,0);
    VL_SIG(VX_decode__DOT__vx_register_file_master__DOT__registers[32],31,0);
    VL_SIG(VX_decode__DOT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers[32],31,0);
    VL_SIG(VX_decode__DOT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers[32],31,0);
    VL_SIG(VX_decode__DOT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers[32],31,0);
    VL_SIG(VX_decode__DOT__gen_code_label__BRA__4__KET____DOT__vx_register_file_slave__DOT__registers[32],31,0);
    
    // LOCAL VARIABLES
    // Internals; generally not touched by application code
    // Begin mtask footprint  all: 
    VL_SIG8(__Vtableidx1,2,0);
    VL_SIG8(__Vclklast__TOP__clk,0,0);
    VL_SIG(VX_decode__DOT____Vcellout__vx_register_file_master__out_src2_data,31,0);
    VL_SIG(VX_decode__DOT____Vcellout__vx_register_file_master__out_src1_data,31,0);
    VL_SIG(VX_decode__DOT____Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src2_data,31,0);
    VL_SIG(VX_decode__DOT____Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src1_data,31,0);
    VL_SIG(VX_decode__DOT____Vcellout__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__out_src2_data,31,0);
    VL_SIG(VX_decode__DOT____Vcellout__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__out_src1_data,31,0);
    VL_SIG(VX_decode__DOT____Vcellout__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__out_src2_data,31,0);
    VL_SIG(VX_decode__DOT____Vcellout__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__out_src1_data,31,0);
    VL_SIG(VX_decode__DOT____Vcellout__gen_code_label__BRA__4__KET____DOT__vx_register_file_slave__out_src2_data,31,0);
    VL_SIG(VX_decode__DOT____Vcellout__gen_code_label__BRA__4__KET____DOT__vx_register_file_slave__out_src1_data,31,0);
    VL_SIG(VX_decode__DOT____Vcellout__vx_register_file_master__out_regs[32],31,0);
    VL_SIG(VX_decode__DOT____Vcellinp__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__in_regs[32],31,0);
    VL_SIG(VX_decode__DOT____Vcellinp__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__in_regs[32],31,0);
    VL_SIG(VX_decode__DOT____Vcellinp__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__in_regs[32],31,0);
    VL_SIG(VX_decode__DOT____Vcellinp__gen_code_label__BRA__4__KET____DOT__vx_register_file_slave__in_regs[32],31,0);
    static VL_ST_SIG8(__Vtable1_VX_decode__DOT__mul_alu[8],4,0);
    
    // INTERNAL VARIABLES
    // Internals; generally not touched by application code
    VVX_decode__Syms* __VlSymsp;  // Symbol table
    
    // PARAMETERS
    // Parameters marked /*verilator public*/ for use by application code
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(VVX_decode);  ///< Copying not allowed
  public:
    /// Construct the model; called by application code
    /// The special name  may be used to make a wrapper with a
    /// single model invisible with respect to DPI scope names.
    VVX_decode(const char* name="TOP");
    /// Destroy the model; called (often implicitly) by application code
    ~VVX_decode();
    
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval();
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    
    // INTERNAL METHODS
  private:
    static void _eval_initial_loop(VVX_decode__Syms* __restrict vlSymsp);
  public:
    void __Vconfigure(VVX_decode__Syms* symsp, bool first);
  private:
    static QData _change_request(VVX_decode__Syms* __restrict vlSymsp);
  public:
    static void _combo__TOP__1(VVX_decode__Syms* __restrict vlSymsp);
    static void _combo__TOP__6(VVX_decode__Syms* __restrict vlSymsp);
  private:
    void _ctor_var_reset();
  public:
    static void _eval(VVX_decode__Syms* __restrict vlSymsp);
  private:
#ifdef VL_DEBUG
    void _eval_debug_assertions();
#endif // VL_DEBUG
  public:
    static void _eval_initial(VVX_decode__Syms* __restrict vlSymsp);
    static void _eval_settle(VVX_decode__Syms* __restrict vlSymsp);
    static void _initial__TOP__5(VVX_decode__Syms* __restrict vlSymsp);
    static void _sequent__TOP__3(VVX_decode__Syms* __restrict vlSymsp);
    static void _sequent__TOP__4(VVX_decode__Syms* __restrict vlSymsp);
    static void _settle__TOP__2(VVX_decode__Syms* __restrict vlSymsp);
} VL_ATTR_ALIGNED(128);

#endif // guard
