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
    VL_IN8(in_valid,3,0);
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
    VL_INW(in_write_data,127,0,4);
    VL_IN(in_curr_PC,31,0);
    VL_INW(in_src1_fwd_data,127,0,4);
    VL_INW(in_src2_fwd_data,127,0,4);
    VL_INW(in_wspawn_regs,1023,0,32);
    VL_OUTW(out_a_reg_data,127,0,4);
    VL_OUTW(out_b_reg_data,127,0,4);
    
    // LOCAL SIGNALS
    // Begin mtask footprint  all: 
    VL_SIG8(__PVT__clone_state_stall,5,0);
    VL_SIG8(__PVT__wspawn_state_stall,5,0);
    VL_SIGW(__PVT__rd1_register,127,0,4);
    VL_SIGW(__PVT__rd2_register,127,0,4);
    VL_SIGW(__PVT__vx_register_file_master__DOT__registers,1023,0,32);
    VL_SIGW(__PVT__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__DOT__registers,1023,0,32);
    VL_SIGW(__PVT__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__DOT__registers,1023,0,32);
    VL_SIGW(__PVT__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__DOT__registers,1023,0,32);
    
    // LOCAL VARIABLES
    // Begin mtask footprint  all: 
    VL_SIG8(__Vdly__clone_state_stall,5,0);
    VL_SIG8(__Vdly__wspawn_state_stall,5,0);
    VL_SIG(__Vcellout__vx_register_file_master__out_src2_data,31,0);
    VL_SIG(__Vcellout__vx_register_file_master__out_src1_data,31,0);
    VL_SIG(__Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src2_data,31,0);
    VL_SIG(__Vcellout__gen_code_label__BRA__1__KET____DOT__vx_register_file_slave__out_src1_data,31,0);
    VL_SIG(__Vcellout__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__out_src2_data,31,0);
    VL_SIG(__Vcellout__gen_code_label__BRA__2__KET____DOT__vx_register_file_slave__out_src1_data,31,0);
    VL_SIG(__Vcellout__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__out_src2_data,31,0);
    VL_SIG(__Vcellout__gen_code_label__BRA__3__KET____DOT__vx_register_file_slave__out_src1_data,31,0);
    
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
    void _combo__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__VX_Context_one__29(VVortex__Syms* __restrict vlSymsp);
  private:
    void _ctor_var_reset() VL_ATTR_COLD;
  public:
    void _initial__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__VX_Context_one__1(VVortex__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    void _sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__VX_Context_one__15(VVortex__Syms* __restrict vlSymsp);
    void _sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__VX_Context_one__22(VVortex__Syms* __restrict vlSymsp);
    void _sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__2__KET____DOT__VX_Context_one__16(VVortex__Syms* __restrict vlSymsp);
    void _sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__3__KET____DOT__VX_Context_one__17(VVortex__Syms* __restrict vlSymsp);
    void _sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__4__KET____DOT__VX_Context_one__18(VVortex__Syms* __restrict vlSymsp);
    void _sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__5__KET____DOT__VX_Context_one__19(VVortex__Syms* __restrict vlSymsp);
    void _sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__6__KET____DOT__VX_Context_one__20(VVortex__Syms* __restrict vlSymsp);
    void _sequent__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__7__KET____DOT__VX_Context_one__21(VVortex__Syms* __restrict vlSymsp);
    void _settle__TOP__Vortex__DOT__vx_decode__DOT__genblk2__BRA__1__KET____DOT__VX_Context_one__8(VVortex__Syms* __restrict vlSymsp) VL_ATTR_COLD;
} VL_ATTR_ALIGNED(128);

#endif // guard
