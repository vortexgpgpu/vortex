// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vcache_simX.h for the primary calling header

#ifndef _Vcache_simX_VX_Cache_Bank__pi8_H_
#define _Vcache_simX_VX_Cache_Bank__pi8_H_

#include "verilated.h"

class Vcache_simX__Syms;
class VerilatedVcd;

//----------

VL_MODULE(Vcache_simX_VX_Cache_Bank__pi8) {
  public:
    
    // PORTS
    VL_IN8(rst,0,0);
    VL_IN8(clk,0,0);
    VL_IN8(state,3,0);
    VL_IN8(actual_index,4,0);
    VL_IN8(block_offset,1,0);
    VL_IN8(valid_in,0,0);
    VL_IN8(read_or_write,0,0);
    VL_IN8(i_p_mem_read,2,0);
    VL_IN8(i_p_mem_write,2,0);
    VL_IN8(byte_select,1,0);
    VL_IN8(evicted_way,0,0);
    VL_OUT8(hit,0,0);
    VL_OUT8(eviction_wb,0,0);
    VL_IN(o_tag,20,0);
    VL_IN(writedata,31,0);
    VL_INW(fetched_writedata,127,0,4);
    VL_OUT(readdata,31,0);
    VL_OUT(eviction_addr,31,0);
    VL_OUTW(data_evicted,127,0,4);
    
    // LOCAL SIGNALS
    VL_SIG8(__PVT__valid_use,0,0);
    VL_SIG8(__PVT__access,0,0);
    VL_SIG8(__PVT__write_from_mem,0,0);
    VL_SIG8(__PVT__way_to_update,0,0);
    VL_SIG8(__PVT__sb_mask,3,0);
    VL_SIG16(__PVT__we,15,0);
    VL_SIG8(__PVT__genblk1__BRA__0__KET____DOT__normal_write,0,0);
    VL_SIG8(__PVT__data_structures__DOT__valid_use_per_way,1,0);
    VL_SIG8(__PVT__data_structures__DOT__dirty_use_per_way,1,0);
    VL_SIG8(__PVT__data_structures__DOT__hit_per_way,1,0);
    VL_SIG(__PVT__data_structures__DOT__we_per_way,31,0);
    VL_SIG8(__PVT__data_structures__DOT__write_from_mem_per_way,1,0);
    VL_SIG8(__PVT__data_structures__DOT__invalid_found,0,0);
    VL_SIG8(__PVT__data_structures__DOT__way_index,0,0);
    VL_SIG8(__PVT__data_structures__DOT__invalid_index,0,0);
    VL_SIG8(__PVT__data_structures__DOT__way_use_Qual,0,0);
    VL_SIG8(__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found,0,0);
    VL_SIG8(__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__update_dirty,0,0);
    VL_SIG8(__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__update_dirty,0,0);
    VL_SIG(__PVT__tag_use,20,0);
    VL_SIG(__PVT__data_unQual,31,0);
    VL_SIG(__PVT__use_write_data,31,0);
    VL_SIGW(__PVT__data_write,127,0,4);
    VL_SIG64(__PVT__data_structures__DOT__tag_use_per_way,41,0);
    VL_SIGW(__PVT__data_structures__DOT__data_use_per_way,255,0,8);
    VL_SIGW(__PVT__data_structures__DOT__data_write_per_way,255,0,8);
    VL_SIG(__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__f,31,0);
    VL_SIG(__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__ini_ind,31,0);
    VL_SIG(__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__f,31,0);
    VL_SIG(__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__ini_ind,31,0);
    VL_SIGW(__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[32],127,0,4);
    VL_SIG(__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[32],20,0);
    VL_SIG8(__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[32],0,0);
    VL_SIG8(__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[32],0,0);
    VL_SIGW(__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[32],127,0,4);
    VL_SIG(__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[32],20,0);
    VL_SIG8(__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[32],0,0);
    VL_SIG8(__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[32],0,0);
    
    // LOCAL VARIABLES
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v0,0,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32,0,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32,0,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32,0,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47,0,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v0,0,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32,0,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32,0,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32,0,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46,0,0);
    VL_SIG8(__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47,6,0);
    VL_SIG8(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47,7,0);
    VL_SIG8(__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47,0,0);
    VL_SIGW(__Vcellout__data_structures__data_use,127,0,4);
    VL_SIG(__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32,20,0);
    VL_SIG(__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32,20,0);
    
    // INTERNAL VARIABLES
  private:
    Vcache_simX__Syms* __VlSymsp;  // Symbol table
  public:
    
    // PARAMETERS
    
    // CONSTRUCTORS
  private:
    Vcache_simX_VX_Cache_Bank__pi8& operator= (const Vcache_simX_VX_Cache_Bank__pi8&);  ///< Copying not allowed
    Vcache_simX_VX_Cache_Bank__pi8(const Vcache_simX_VX_Cache_Bank__pi8&);  ///< Copying not allowed
  public:
    Vcache_simX_VX_Cache_Bank__pi8(const char* name="TOP");
    ~Vcache_simX_VX_Cache_Bank__pi8();
    void trace (VerilatedVcdC* tfp, int levels, int options=0);
    
    // API METHODS
    
    // INTERNAL METHODS
    void __Vconfigure(Vcache_simX__Syms* symsp, bool first);
    void _combo__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__9(Vcache_simX__Syms* __restrict vlSymsp);
    void _combo__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure__10(Vcache_simX__Syms* __restrict vlSymsp);
    void _combo__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure__11(Vcache_simX__Syms* __restrict vlSymsp);
    void _combo__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure__12(Vcache_simX__Syms* __restrict vlSymsp);
  private:
    void _ctor_var_reset();
  public:
    void _sequent__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__5(Vcache_simX__Syms* __restrict vlSymsp);
    void _sequent__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure__6(Vcache_simX__Syms* __restrict vlSymsp);
    void _sequent__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure__7(Vcache_simX__Syms* __restrict vlSymsp);
    void _sequent__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure__8(Vcache_simX__Syms* __restrict vlSymsp);
    void _settle__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__1(Vcache_simX__Syms* __restrict vlSymsp);
    void _settle__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure__2(Vcache_simX__Syms* __restrict vlSymsp);
    void _settle__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure__3(Vcache_simX__Syms* __restrict vlSymsp);
    void _settle__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure__4(Vcache_simX__Syms* __restrict vlSymsp);
    static void traceInit (VerilatedVcd* vcdp, void* userthis, uint32_t code);
    static void traceFull (VerilatedVcd* vcdp, void* userthis, uint32_t code);
    static void traceChg  (VerilatedVcd* vcdp, void* userthis, uint32_t code);
} VL_ATTR_ALIGNED(128);

#endif // guard
