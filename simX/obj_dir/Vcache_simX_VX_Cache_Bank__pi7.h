// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design internal header
// See Vcache_simX.h for the primary calling header

#ifndef _Vcache_simX_VX_Cache_Bank__pi7_H_
#define _Vcache_simX_VX_Cache_Bank__pi7_H_

#include "verilated.h"

class Vcache_simX__Syms;
class VerilatedVcd;

//----------

VL_MODULE(Vcache_simX_VX_Cache_Bank__pi7) {
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
    CData/*0:0*/ __PVT__valid_use;
    CData/*0:0*/ __PVT__access;
    CData/*0:0*/ __PVT__write_from_mem;
    CData/*0:0*/ __PVT__miss;
    CData/*0:0*/ __PVT__way_to_update;
    CData/*3:0*/ __PVT__sb_mask;
    SData/*3:0*/ __PVT__we;
    CData/*0:0*/ __PVT__genblk1__BRA__0__KET____DOT__normal_write;
    CData/*0:0*/ __PVT__genblk1__BRA__1__KET____DOT__normal_write;
    CData/*0:0*/ __PVT__genblk1__BRA__2__KET____DOT__normal_write;
    CData/*0:0*/ __PVT__genblk1__BRA__3__KET____DOT__normal_write;
    CData/*1:0*/ __PVT__data_structures__DOT__valid_use_per_way;
    CData/*1:0*/ __PVT__data_structures__DOT__dirty_use_per_way;
    CData/*1:0*/ __PVT__data_structures__DOT__hit_per_way;
    IData/*3:0*/ __PVT__data_structures__DOT__we_per_way;
    CData/*1:0*/ __PVT__data_structures__DOT__write_from_mem_per_way;
    CData/*0:0*/ __PVT__data_structures__DOT__invalid_found;
    CData/*0:0*/ __PVT__data_structures__DOT__way_index;
    CData/*0:0*/ __PVT__data_structures__DOT__invalid_index;
    CData/*0:0*/ __PVT__data_structures__DOT__way_use_Qual;
    CData/*0:0*/ __PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found;
    IData/*20:0*/ __PVT__tag_use;
    IData/*31:0*/ __PVT__data_unQual;
    IData/*31:0*/ __PVT__use_write_data;
    WData/*31:0*/ __PVT__data_write[4];
    QData/*20:0*/ __PVT__data_structures__DOT__tag_use_per_way;
    WData/*31:0*/ __PVT__data_structures__DOT__data_use_per_way[8];
    WData/*31:0*/ __PVT__data_structures__DOT__data_write_per_way[8];
    IData/*31:0*/ __PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__f;
    IData/*31:0*/ __PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__ini_ind;
    IData/*31:0*/ __PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__f;
    IData/*31:0*/ __PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__ini_ind;
    WData/*7:0*/ __PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[32][4];
    IData/*20:0*/ __PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[32];
    CData/*0:0*/ __PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[32];
    CData/*0:0*/ __PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[32];
    WData/*7:0*/ __PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[32][4];
    IData/*20:0*/ __PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[32];
    CData/*0:0*/ __PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[32];
    CData/*0:0*/ __PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[32];
    
    // LOCAL VARIABLES
    // Anonymous structures to workaround compiler member-count bugs
    struct {
        CData/*0:0*/ data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use;
        CData/*0:0*/ data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v0;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v0;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v0;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45;
    };
    struct {
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v0;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32;
        CData/*0:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v0;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v0;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v0;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42;
    };
    struct {
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47;
        CData/*6:0*/ __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47;
        CData/*7:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v0;
        CData/*4:0*/ __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32;
        CData/*0:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32;
        CData/*0:0*/ __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32;
        WData/*31:0*/ __Vcellout__data_structures__data_use[4];
        IData/*20:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32;
        IData/*20:0*/ __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32;
    };
    
    // INTERNAL VARIABLES
  private:
    Vcache_simX__Syms* __VlSymsp;  // Symbol table
  public:
    
    // PARAMETERS
    
    // CONSTRUCTORS
  private:
    VL_UNCOPYABLE(Vcache_simX_VX_Cache_Bank__pi7);  ///< Copying not allowed
  public:
    Vcache_simX_VX_Cache_Bank__pi7(const char* name = "TOP");
    ~Vcache_simX_VX_Cache_Bank__pi7();
    
    // API METHODS
    
    // INTERNAL METHODS
    void __Vconfigure(Vcache_simX__Syms* symsp, bool first);
    void _combo__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__9(Vcache_simX__Syms* __restrict vlSymsp);
    void _combo__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure__10(Vcache_simX__Syms* __restrict vlSymsp);
    void _combo__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure__11(Vcache_simX__Syms* __restrict vlSymsp);
    void _combo__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure__12(Vcache_simX__Syms* __restrict vlSymsp);
  private:
    void _ctor_var_reset() VL_ATTR_COLD;
  public:
    void _sequent__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__5(Vcache_simX__Syms* __restrict vlSymsp);
    void _sequent__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure__6(Vcache_simX__Syms* __restrict vlSymsp);
    void _sequent__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure__7(Vcache_simX__Syms* __restrict vlSymsp);
    void _sequent__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure__8(Vcache_simX__Syms* __restrict vlSymsp);
    void _settle__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__1(Vcache_simX__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    void _settle__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure__2(Vcache_simX__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    void _settle__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure__3(Vcache_simX__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    void _settle__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure__4(Vcache_simX__Syms* __restrict vlSymsp) VL_ATTR_COLD;
    static void traceInit(VerilatedVcd* vcdp, void* userthis, uint32_t code);
    static void traceFull(VerilatedVcd* vcdp, void* userthis, uint32_t code);
    static void traceChg(VerilatedVcd* vcdp, void* userthis, uint32_t code);
} VL_ATTR_ALIGNED(128);

#endif // guard
