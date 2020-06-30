// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "VVX_cache__Syms.h"


//======================

void VVX_cache::traceChg(VerilatedVcd* vcdp, void* userthis, uint32_t code) {
    // Callback from vcd->dump()
    VVX_cache* t = (VVX_cache*)userthis;
    VVX_cache__Syms* __restrict vlSymsp = t->__VlSymsp;  // Setup global symbol table
    if (vlSymsp->getClearActivity()) {
        t->traceChgThis(vlSymsp, vcdp, code);
    }
}

//======================


void VVX_cache::traceChgThis(VVX_cache__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code) {
    VVX_cache* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    int c = code;
    if (0 && vcdp && c) {}  // Prevent unused
    // Body
    {
        if (VL_UNLIKELY((1U & (vlTOPp->__Vm_traceActivity 
                               | (vlTOPp->__Vm_traceActivity 
                                  >> 1U))))) {
            vlTOPp->traceChgThis__2(vlSymsp, vcdp, code);
        }
        if (VL_UNLIKELY((1U & ((vlTOPp->__Vm_traceActivity 
                                | (vlTOPp->__Vm_traceActivity 
                                   >> 1U)) | (vlTOPp->__Vm_traceActivity 
                                              >> 2U))))) {
            vlTOPp->traceChgThis__3(vlSymsp, vcdp, code);
        }
        if (VL_UNLIKELY((1U & (vlTOPp->__Vm_traceActivity 
                               | (vlTOPp->__Vm_traceActivity 
                                  >> 2U))))) {
            vlTOPp->traceChgThis__4(vlSymsp, vcdp, code);
        }
        if (VL_UNLIKELY((4U & vlTOPp->__Vm_traceActivity))) {
            vlTOPp->traceChgThis__5(vlSymsp, vcdp, code);
        }
        vlTOPp->traceChgThis__6(vlSymsp, vcdp, code);
    }
    // Final
    vlTOPp->__Vm_traceActivity = 0U;
}

void VVX_cache::traceChgThis__2(VVX_cache__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code) {
    VVX_cache* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    int c = code;
    if (0 && vcdp && c) {}  // Prevent unused
    // Body
    {
        vcdp->chgBus(c+1,(vlTOPp->VX_cache__DOT____Vcellout__cache_core_req_bank_sel__per_bank_valid),32);
        vcdp->chgBus(c+9,(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready),8);
        vcdp->chgBus(c+17,(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_ready),8);
        vcdp->chgBus(c+25,(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_ready),8);
        vcdp->chgBus(c+33,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__curr_bank_core_req_valid),4);
        vcdp->chgBit(c+41,((1U & (IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready))));
        vcdp->chgBit(c+49,((1U & (IData)(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_ready))));
        vcdp->chgBit(c+57,((1U & (IData)(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_ready))));
        vcdp->chgBus(c+65,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__curr_bank_core_req_valid),4);
        vcdp->chgBit(c+73,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                  >> 1U))));
        vcdp->chgBit(c+81,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_ready) 
                                  >> 1U))));
        vcdp->chgBit(c+89,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_ready) 
                                  >> 1U))));
        vcdp->chgBus(c+97,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__curr_bank_core_req_valid),4);
        vcdp->chgBit(c+105,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                   >> 2U))));
        vcdp->chgBit(c+113,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_ready) 
                                   >> 2U))));
        vcdp->chgBit(c+121,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_ready) 
                                   >> 2U))));
        vcdp->chgBus(c+129,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__curr_bank_core_req_valid),4);
        vcdp->chgBit(c+137,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                   >> 3U))));
        vcdp->chgBit(c+145,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_ready) 
                                   >> 3U))));
        vcdp->chgBit(c+153,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_ready) 
                                   >> 3U))));
        vcdp->chgBus(c+161,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__curr_bank_core_req_valid),4);
        vcdp->chgBit(c+169,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                   >> 4U))));
        vcdp->chgBit(c+177,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_ready) 
                                   >> 4U))));
        vcdp->chgBit(c+185,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_ready) 
                                   >> 4U))));
        vcdp->chgBus(c+193,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__curr_bank_core_req_valid),4);
        vcdp->chgBit(c+201,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                   >> 5U))));
        vcdp->chgBit(c+209,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_ready) 
                                   >> 5U))));
        vcdp->chgBit(c+217,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_ready) 
                                   >> 5U))));
        vcdp->chgBus(c+225,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__curr_bank_core_req_valid),4);
        vcdp->chgBit(c+233,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                   >> 6U))));
        vcdp->chgBit(c+241,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_ready) 
                                   >> 6U))));
        vcdp->chgBit(c+249,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_ready) 
                                   >> 6U))));
        vcdp->chgBus(c+257,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__curr_bank_core_req_valid),4);
        vcdp->chgBit(c+265,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                   >> 7U))));
        vcdp->chgBit(c+273,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_ready) 
                                   >> 7U))));
        vcdp->chgBit(c+281,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_ready) 
                                   >> 7U))));
        vcdp->chgBus(c+289,(vlTOPp->VX_cache__DOT__cache_core_req_bank_sel__DOT__genblk2__DOT__per_bank_ready_sel),8);
        vcdp->chgBit(c+297,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dfqq_pop));
        vcdp->chgBit(c+305,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__reading));
        vcdp->chgBit(c+313,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dram_wb_req_fire));
        vcdp->chgBit(c+321,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_rsp_fire));
        vcdp->chgQuad(c+329,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__snp_req_queue__data_in),54);
        vcdp->chgBit(c+345,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__writing));
        vcdp->chgArray(c+353,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__dfp_queue__data_in),153);
        vcdp->chgBit(c+393,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__writing));
        vcdp->chgArray(c+401,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellinp__reqq_queue__data_in),314);
        vcdp->chgBit(c+481,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__writing));
        vcdp->chgBit(c+489,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__reading));
        vcdp->chgBit(c+497,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__reading));
        vcdp->chgBit(c+505,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dram_wb_req_fire));
        vcdp->chgBit(c+513,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_rsp_fire));
        vcdp->chgQuad(c+521,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__snp_req_queue__data_in),54);
        vcdp->chgBit(c+537,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__writing));
        vcdp->chgArray(c+545,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__dfp_queue__data_in),153);
        vcdp->chgBit(c+585,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__writing));
        vcdp->chgArray(c+593,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellinp__reqq_queue__data_in),314);
        vcdp->chgBit(c+673,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__writing));
        vcdp->chgBit(c+681,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__reading));
        vcdp->chgBit(c+689,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__reading));
        vcdp->chgBit(c+697,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dram_wb_req_fire));
        vcdp->chgBit(c+705,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_rsp_fire));
        vcdp->chgQuad(c+713,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__snp_req_queue__data_in),54);
        vcdp->chgBit(c+729,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__writing));
        vcdp->chgArray(c+737,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__dfp_queue__data_in),153);
        vcdp->chgBit(c+777,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__writing));
        vcdp->chgArray(c+785,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellinp__reqq_queue__data_in),314);
        vcdp->chgBit(c+865,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__writing));
        vcdp->chgBit(c+873,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__reading));
        vcdp->chgBit(c+881,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__reading));
        vcdp->chgBit(c+889,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dram_wb_req_fire));
        vcdp->chgBit(c+897,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_rsp_fire));
        vcdp->chgQuad(c+905,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__snp_req_queue__data_in),54);
        vcdp->chgBit(c+921,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__writing));
        vcdp->chgArray(c+929,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__dfp_queue__data_in),153);
        vcdp->chgBit(c+969,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__writing));
        vcdp->chgArray(c+977,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellinp__reqq_queue__data_in),314);
        vcdp->chgBit(c+1057,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__writing));
        vcdp->chgBit(c+1065,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__reading));
        vcdp->chgBit(c+1073,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__reading));
        vcdp->chgBit(c+1081,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dram_wb_req_fire));
        vcdp->chgBit(c+1089,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_rsp_fire));
        vcdp->chgQuad(c+1097,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellinp__snp_req_queue__data_in),54);
        vcdp->chgBit(c+1113,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__writing));
        vcdp->chgArray(c+1121,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellinp__dfp_queue__data_in),153);
        vcdp->chgBit(c+1161,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__writing));
        vcdp->chgArray(c+1169,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellinp__reqq_queue__data_in),314);
        vcdp->chgBit(c+1249,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__writing));
        vcdp->chgBit(c+1257,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__reading));
        vcdp->chgBit(c+1265,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwb_queue__DOT__reading));
        vcdp->chgBit(c+1273,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dram_wb_req_fire));
        vcdp->chgBit(c+1281,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_rsp_fire));
        vcdp->chgQuad(c+1289,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellinp__snp_req_queue__data_in),54);
        vcdp->chgBit(c+1305,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__writing));
        vcdp->chgArray(c+1313,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellinp__dfp_queue__data_in),153);
        vcdp->chgBit(c+1353,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__writing));
        vcdp->chgArray(c+1361,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellinp__reqq_queue__data_in),314);
        vcdp->chgBit(c+1441,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__writing));
        vcdp->chgBit(c+1449,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__reading));
        vcdp->chgBit(c+1457,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwb_queue__DOT__reading));
        vcdp->chgBit(c+1465,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dram_wb_req_fire));
        vcdp->chgBit(c+1473,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_rsp_fire));
        vcdp->chgQuad(c+1481,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellinp__snp_req_queue__data_in),54);
        vcdp->chgBit(c+1497,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__writing));
        vcdp->chgArray(c+1505,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellinp__dfp_queue__data_in),153);
        vcdp->chgBit(c+1545,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__writing));
        vcdp->chgArray(c+1553,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellinp__reqq_queue__data_in),314);
        vcdp->chgBit(c+1633,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__writing));
        vcdp->chgBit(c+1641,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__reading));
        vcdp->chgBit(c+1649,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwb_queue__DOT__reading));
        vcdp->chgBit(c+1657,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dram_wb_req_fire));
        vcdp->chgBit(c+1665,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_rsp_fire));
        vcdp->chgQuad(c+1673,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellinp__snp_req_queue__data_in),54);
        vcdp->chgBit(c+1689,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__writing));
        vcdp->chgArray(c+1697,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellinp__dfp_queue__data_in),153);
        vcdp->chgBit(c+1737,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__writing));
        vcdp->chgArray(c+1745,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellinp__reqq_queue__data_in),314);
        vcdp->chgBit(c+1825,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__writing));
        vcdp->chgBit(c+1833,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__reading));
        vcdp->chgBit(c+1841,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwb_queue__DOT__reading));
    }
}

void VVX_cache::traceChgThis__3(VVX_cache__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code) {
    VVX_cache* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    int c = code;
    if (0 && vcdp && c) {}  // Prevent unused
    // Body
    {
        vcdp->chgBit(c+1849,((((IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dfqq_pop) 
                               & (~ (IData)((0U != (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bank_dram_fill_req_valid))))) 
                              & (~ ((~ (IData)((0U 
                                                != 
                                                (0xffU 
                                                 & vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[7U])))) 
                                    | (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r))))));
        vcdp->chgBit(c+1857,(((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__curr_bank_core_req_valid)) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+1865,((1U & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)) 
                                    & (IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready)))));
        vcdp->chgBit(c+1873,((((((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                  >> 6U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                               >> 5U))) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dram_wb_req_fire)) 
                               | (((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 5U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                                 >> 6U))) 
                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_rsp_fire))) 
                              | (((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                   >> 6U) & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                             >> 5U)) 
                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_rsp_fire)))));
        vcdp->chgBit(c+1881,((((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__curr_bank_core_req_valid)) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+1889,(((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__curr_bank_core_req_valid)) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+1897,((1U & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)) 
                                    & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                       >> 1U)))));
        vcdp->chgBit(c+1905,((((((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                  >> 6U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                               >> 5U))) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dram_wb_req_fire)) 
                               | (((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 5U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                                 >> 6U))) 
                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_rsp_fire))) 
                              | (((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                   >> 6U) & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                             >> 5U)) 
                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_rsp_fire)))));
        vcdp->chgBit(c+1913,((((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__curr_bank_core_req_valid)) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+1921,(((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__curr_bank_core_req_valid)) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+1929,((1U & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)) 
                                    & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                       >> 2U)))));
        vcdp->chgBit(c+1937,((((((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                  >> 6U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                               >> 5U))) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dram_wb_req_fire)) 
                               | (((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 5U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                                 >> 6U))) 
                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_rsp_fire))) 
                              | (((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                   >> 6U) & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                             >> 5U)) 
                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_rsp_fire)))));
        vcdp->chgBit(c+1945,((((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__curr_bank_core_req_valid)) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+1953,(((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__curr_bank_core_req_valid)) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+1961,((1U & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)) 
                                    & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                       >> 3U)))));
        vcdp->chgBit(c+1969,((((((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                  >> 6U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                               >> 5U))) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dram_wb_req_fire)) 
                               | (((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 5U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                                 >> 6U))) 
                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_rsp_fire))) 
                              | (((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                   >> 6U) & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                             >> 5U)) 
                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_rsp_fire)))));
        vcdp->chgBit(c+1977,((((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__curr_bank_core_req_valid)) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+1985,(((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__curr_bank_core_req_valid)) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+1993,((1U & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)) 
                                    & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                       >> 4U)))));
        vcdp->chgBit(c+2001,((((((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                  >> 6U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                               >> 5U))) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dram_wb_req_fire)) 
                               | (((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 5U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                                 >> 6U))) 
                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_rsp_fire))) 
                              | (((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                   >> 6U) & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                             >> 5U)) 
                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_rsp_fire)))));
        vcdp->chgBit(c+2009,((((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__curr_bank_core_req_valid)) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+2017,(((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__curr_bank_core_req_valid)) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+2025,((1U & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)) 
                                    & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                       >> 5U)))));
        vcdp->chgBit(c+2033,((((((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                  >> 6U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                               >> 5U))) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dram_wb_req_fire)) 
                               | (((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 5U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                                 >> 6U))) 
                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_rsp_fire))) 
                              | (((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                   >> 6U) & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                             >> 5U)) 
                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_rsp_fire)))));
        vcdp->chgBit(c+2041,((((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__curr_bank_core_req_valid)) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+2049,(((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__curr_bank_core_req_valid)) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+2057,((1U & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)) 
                                    & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                       >> 6U)))));
        vcdp->chgBit(c+2065,((((((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                  >> 6U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                               >> 5U))) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dram_wb_req_fire)) 
                               | (((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 5U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                                 >> 6U))) 
                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_rsp_fire))) 
                              | (((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                   >> 6U) & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                             >> 5U)) 
                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_rsp_fire)))));
        vcdp->chgBit(c+2073,((((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__curr_bank_core_req_valid)) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+2081,(((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__curr_bank_core_req_valid)) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+2089,((1U & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)) 
                                    & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                       >> 7U)))));
        vcdp->chgBit(c+2097,((((((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                  >> 6U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                               >> 5U))) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dram_wb_req_fire)) 
                               | (((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 5U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                                 >> 6U))) 
                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_rsp_fire))) 
                              | (((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                   >> 6U) & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                             >> 5U)) 
                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_rsp_fire)))));
        vcdp->chgBit(c+2105,((((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__curr_bank_core_req_valid)) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
    }
}

void VVX_cache::traceChgThis__4(VVX_cache__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code) {
    VVX_cache* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    int c = code;
    if (0 && vcdp && c) {}  // Prevent unused
    // Variables
    WData/*127:0*/ __Vtemp1366[4];
    WData/*127:0*/ __Vtemp1385[4];
    WData/*127:0*/ __Vtemp1404[4];
    WData/*127:0*/ __Vtemp1423[4];
    WData/*127:0*/ __Vtemp1442[4];
    WData/*127:0*/ __Vtemp1461[4];
    WData/*127:0*/ __Vtemp1480[4];
    WData/*127:0*/ __Vtemp1499[4];
    WData/*127:0*/ __Vtemp1339[4];
    WData/*127:0*/ __Vtemp1340[4];
    WData/*127:0*/ __Vtemp1341[4];
    WData/*127:0*/ __Vtemp1342[4];
    WData/*127:0*/ __Vtemp1343[4];
    WData/*127:0*/ __Vtemp1344[4];
    WData/*127:0*/ __Vtemp1345[4];
    WData/*127:0*/ __Vtemp1346[4];
    WData/*223:0*/ __Vtemp1347[7];
    WData/*127:0*/ __Vtemp1348[4];
    WData/*127:0*/ __Vtemp1353[4];
    WData/*127:0*/ __Vtemp1359[4];
    WData/*127:0*/ __Vtemp1360[4];
    WData/*127:0*/ __Vtemp1363[4];
    WData/*127:0*/ __Vtemp1364[4];
    WData/*127:0*/ __Vtemp1365[4];
    WData/*127:0*/ __Vtemp1367[4];
    WData/*127:0*/ __Vtemp1372[4];
    WData/*127:0*/ __Vtemp1378[4];
    WData/*127:0*/ __Vtemp1379[4];
    WData/*127:0*/ __Vtemp1382[4];
    WData/*127:0*/ __Vtemp1383[4];
    WData/*127:0*/ __Vtemp1384[4];
    WData/*127:0*/ __Vtemp1386[4];
    WData/*127:0*/ __Vtemp1391[4];
    WData/*127:0*/ __Vtemp1397[4];
    WData/*127:0*/ __Vtemp1398[4];
    WData/*127:0*/ __Vtemp1401[4];
    WData/*127:0*/ __Vtemp1402[4];
    WData/*127:0*/ __Vtemp1403[4];
    WData/*127:0*/ __Vtemp1405[4];
    WData/*127:0*/ __Vtemp1410[4];
    WData/*127:0*/ __Vtemp1416[4];
    WData/*127:0*/ __Vtemp1417[4];
    WData/*127:0*/ __Vtemp1420[4];
    WData/*127:0*/ __Vtemp1421[4];
    WData/*127:0*/ __Vtemp1422[4];
    WData/*127:0*/ __Vtemp1424[4];
    WData/*127:0*/ __Vtemp1429[4];
    WData/*127:0*/ __Vtemp1435[4];
    WData/*127:0*/ __Vtemp1436[4];
    WData/*127:0*/ __Vtemp1439[4];
    WData/*127:0*/ __Vtemp1440[4];
    WData/*127:0*/ __Vtemp1441[4];
    WData/*127:0*/ __Vtemp1443[4];
    WData/*127:0*/ __Vtemp1448[4];
    WData/*127:0*/ __Vtemp1454[4];
    WData/*127:0*/ __Vtemp1455[4];
    WData/*127:0*/ __Vtemp1458[4];
    WData/*127:0*/ __Vtemp1459[4];
    WData/*127:0*/ __Vtemp1460[4];
    WData/*127:0*/ __Vtemp1462[4];
    WData/*127:0*/ __Vtemp1467[4];
    WData/*127:0*/ __Vtemp1473[4];
    WData/*127:0*/ __Vtemp1474[4];
    WData/*127:0*/ __Vtemp1477[4];
    WData/*127:0*/ __Vtemp1478[4];
    WData/*127:0*/ __Vtemp1479[4];
    WData/*127:0*/ __Vtemp1481[4];
    WData/*127:0*/ __Vtemp1486[4];
    WData/*127:0*/ __Vtemp1492[4];
    WData/*127:0*/ __Vtemp1493[4];
    WData/*127:0*/ __Vtemp1496[4];
    WData/*127:0*/ __Vtemp1497[4];
    WData/*127:0*/ __Vtemp1498[4];
    // Body
    {
        vcdp->chgBus(c+2113,(vlTOPp->VX_cache__DOT__per_bank_core_req_ready),8);
        vcdp->chgBus(c+2121,(vlTOPp->VX_cache__DOT__per_bank_core_rsp_valid),8);
        vcdp->chgBus(c+2129,(vlTOPp->VX_cache__DOT__per_bank_core_rsp_tid),16);
        vcdp->chgArray(c+2137,(vlTOPp->VX_cache__DOT__per_bank_core_rsp_data),256);
        vcdp->chgArray(c+2201,(vlTOPp->VX_cache__DOT__per_bank_core_rsp_tag),336);
        vcdp->chgBus(c+2289,(vlTOPp->VX_cache__DOT__per_bank_dram_fill_req_valid),8);
        vcdp->chgArray(c+2297,(vlTOPp->VX_cache__DOT__per_bank_dram_fill_req_addr),224);
        vcdp->chgBus(c+2353,(vlTOPp->VX_cache__DOT__per_bank_dram_fill_rsp_ready),8);
        vcdp->chgBus(c+2361,(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_valid),8);
        vcdp->chgArray(c+2369,(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_byteen),128);
        vcdp->chgArray(c+2401,(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_addr),224);
        vcdp->chgArray(c+2457,(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_data),1024);
        vcdp->chgBus(c+2713,(vlTOPp->VX_cache__DOT__per_bank_snp_req_ready),8);
        vcdp->chgBus(c+2721,(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_valid),8);
        vcdp->chgArray(c+2729,(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_tag),224);
        vcdp->chgBus(c+2785,((3U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U] 
                                    >> 0xaU))),2);
        vcdp->chgBus(c+2793,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[0U]),32);
        vcdp->chgQuad(c+2801,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[1U]))))),42);
        vcdp->chgBit(c+2817,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU))))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwbq_push_stall))))));
        vcdp->chgBit(c+2825,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__curr_bank_dram_wb_req_valid));
        vcdp->chgBus(c+2833,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                          << 0xbU) 
                                         | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                            >> 0x15U)))),16);
        vcdp->chgBus(c+2841,((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                             << 4U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                                               >> 0x1cU)))),25);
        __Vtemp1339[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U] 
                                      >> 0x1cU));
        __Vtemp1339[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                                      >> 0x1cU));
        __Vtemp1339[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                                      >> 0x1cU));
        __Vtemp1339[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                                      >> 0x1cU));
        vcdp->chgArray(c+2849,(__Vtemp1339),128);
        vcdp->chgBit(c+2881,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__curr_bank_snp_rsp_valid));
        vcdp->chgBus(c+2889,((0xfffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U])),28);
        vcdp->chgBus(c+2897,((3U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U] 
                                    >> 0xaU))),2);
        vcdp->chgBus(c+2905,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[0U]),32);
        vcdp->chgQuad(c+2913,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[1U]))))),42);
        vcdp->chgBit(c+2929,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU))))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwbq_push_stall))))));
        vcdp->chgBit(c+2937,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__curr_bank_dram_wb_req_valid));
        vcdp->chgBus(c+2945,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                          << 0xbU) 
                                         | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                            >> 0x15U)))),16);
        vcdp->chgBus(c+2953,((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                             << 4U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                                               >> 0x1cU)))),25);
        __Vtemp1340[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U] 
                                      >> 0x1cU));
        __Vtemp1340[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                                      >> 0x1cU));
        __Vtemp1340[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                                      >> 0x1cU));
        __Vtemp1340[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                                      >> 0x1cU));
        vcdp->chgArray(c+2961,(__Vtemp1340),128);
        vcdp->chgBit(c+2993,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__curr_bank_snp_rsp_valid));
        vcdp->chgBus(c+3001,((0xfffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U])),28);
        vcdp->chgBus(c+3009,((3U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U] 
                                    >> 0xaU))),2);
        vcdp->chgBus(c+3017,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[0U]),32);
        vcdp->chgQuad(c+3025,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[1U]))))),42);
        vcdp->chgBit(c+3041,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU))))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwbq_push_stall))))));
        vcdp->chgBit(c+3049,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__curr_bank_dram_wb_req_valid));
        vcdp->chgBus(c+3057,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                          << 0xbU) 
                                         | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                            >> 0x15U)))),16);
        vcdp->chgBus(c+3065,((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                             << 4U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                                               >> 0x1cU)))),25);
        __Vtemp1341[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U] 
                                      >> 0x1cU));
        __Vtemp1341[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                                      >> 0x1cU));
        __Vtemp1341[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                                      >> 0x1cU));
        __Vtemp1341[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                                      >> 0x1cU));
        vcdp->chgArray(c+3073,(__Vtemp1341),128);
        vcdp->chgBit(c+3105,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__curr_bank_snp_rsp_valid));
        vcdp->chgBus(c+3113,((0xfffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U])),28);
        vcdp->chgBus(c+3121,((3U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U] 
                                    >> 0xaU))),2);
        vcdp->chgBus(c+3129,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[0U]),32);
        vcdp->chgQuad(c+3137,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[1U]))))),42);
        vcdp->chgBit(c+3153,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU))))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwbq_push_stall))))));
        vcdp->chgBit(c+3161,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__curr_bank_dram_wb_req_valid));
        vcdp->chgBus(c+3169,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                          << 0xbU) 
                                         | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                            >> 0x15U)))),16);
        vcdp->chgBus(c+3177,((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                             << 4U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                                               >> 0x1cU)))),25);
        __Vtemp1342[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U] 
                                      >> 0x1cU));
        __Vtemp1342[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                                      >> 0x1cU));
        __Vtemp1342[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                                      >> 0x1cU));
        __Vtemp1342[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                                      >> 0x1cU));
        vcdp->chgArray(c+3185,(__Vtemp1342),128);
        vcdp->chgBit(c+3217,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__curr_bank_snp_rsp_valid));
        vcdp->chgBus(c+3225,((0xfffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U])),28);
        vcdp->chgBus(c+3233,((3U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U] 
                                    >> 0xaU))),2);
        vcdp->chgBus(c+3241,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[0U]),32);
        vcdp->chgQuad(c+3249,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[1U]))))),42);
        vcdp->chgBit(c+3265,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU))))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwbq_push_stall))))));
        vcdp->chgBit(c+3273,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__curr_bank_dram_wb_req_valid));
        vcdp->chgBus(c+3281,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                          << 0xbU) 
                                         | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                            >> 0x15U)))),16);
        vcdp->chgBus(c+3289,((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                             << 4U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                                               >> 0x1cU)))),25);
        __Vtemp1343[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U] 
                                      >> 0x1cU));
        __Vtemp1343[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                                      >> 0x1cU));
        __Vtemp1343[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                                      >> 0x1cU));
        __Vtemp1343[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                                      >> 0x1cU));
        vcdp->chgArray(c+3297,(__Vtemp1343),128);
        vcdp->chgBit(c+3329,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__curr_bank_snp_rsp_valid));
        vcdp->chgBus(c+3337,((0xfffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U])),28);
        vcdp->chgBus(c+3345,((3U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U] 
                                    >> 0xaU))),2);
        vcdp->chgBus(c+3353,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[0U]),32);
        vcdp->chgQuad(c+3361,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[1U]))))),42);
        vcdp->chgBit(c+3377,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU))))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwbq_push_stall))))));
        vcdp->chgBit(c+3385,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__curr_bank_dram_wb_req_valid));
        vcdp->chgBus(c+3393,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                          << 0xbU) 
                                         | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                            >> 0x15U)))),16);
        vcdp->chgBus(c+3401,((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                             << 4U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                                               >> 0x1cU)))),25);
        __Vtemp1344[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U] 
                                      >> 0x1cU));
        __Vtemp1344[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                                      >> 0x1cU));
        __Vtemp1344[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                                      >> 0x1cU));
        __Vtemp1344[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                                      >> 0x1cU));
        vcdp->chgArray(c+3409,(__Vtemp1344),128);
        vcdp->chgBit(c+3441,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__curr_bank_snp_rsp_valid));
        vcdp->chgBus(c+3449,((0xfffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U])),28);
        vcdp->chgBus(c+3457,((3U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U] 
                                    >> 0xaU))),2);
        vcdp->chgBus(c+3465,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[0U]),32);
        vcdp->chgQuad(c+3473,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[1U]))))),42);
        vcdp->chgBit(c+3489,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU))))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwbq_push_stall))))));
        vcdp->chgBit(c+3497,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__curr_bank_dram_wb_req_valid));
        vcdp->chgBus(c+3505,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                          << 0xbU) 
                                         | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                            >> 0x15U)))),16);
        vcdp->chgBus(c+3513,((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                             << 4U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                                               >> 0x1cU)))),25);
        __Vtemp1345[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U] 
                                      >> 0x1cU));
        __Vtemp1345[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                                      >> 0x1cU));
        __Vtemp1345[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                                      >> 0x1cU));
        __Vtemp1345[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                                      >> 0x1cU));
        vcdp->chgArray(c+3521,(__Vtemp1345),128);
        vcdp->chgBit(c+3553,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__curr_bank_snp_rsp_valid));
        vcdp->chgBus(c+3561,((0xfffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U])),28);
        vcdp->chgBus(c+3569,((3U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U] 
                                    >> 0xaU))),2);
        vcdp->chgBus(c+3577,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[0U]),32);
        vcdp->chgQuad(c+3585,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[1U]))))),42);
        vcdp->chgBit(c+3601,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU))))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwbq_push_stall))))));
        vcdp->chgBit(c+3609,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__curr_bank_dram_wb_req_valid));
        vcdp->chgBus(c+3617,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                          << 0xbU) 
                                         | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                            >> 0x15U)))),16);
        vcdp->chgBus(c+3625,((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                             << 4U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                                               >> 0x1cU)))),25);
        __Vtemp1346[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U] 
                                      >> 0x1cU));
        __Vtemp1346[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                                      >> 0x1cU));
        __Vtemp1346[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                                      >> 0x1cU));
        __Vtemp1346[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                            << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                                      >> 0x1cU));
        vcdp->chgArray(c+3633,(__Vtemp1346),128);
        vcdp->chgBit(c+3665,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__curr_bank_snp_rsp_valid));
        vcdp->chgBus(c+3673,((0xfffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U])),28);
        vcdp->chgBit(c+3681,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dwb_valid));
        vcdp->chgBit(c+3689,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dfqq_req));
        vcdp->chgBus(c+3697,(((0xdfU >= (0xffU & ((IData)(0x1cU) 
                                                  * (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_request_index))))
                               ? (0xfffffffU & (((0U 
                                                  == 
                                                  (0x1fU 
                                                   & ((IData)(0x1cU) 
                                                      * (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_request_index))))
                                                  ? 0U
                                                  : 
                                                 (vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_bank_dram_fill_req_addr[
                                                  ((IData)(1U) 
                                                   + 
                                                   (7U 
                                                    & (((IData)(0x1cU) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_request_index)) 
                                                       >> 5U)))] 
                                                  << 
                                                  ((IData)(0x20U) 
                                                   - 
                                                   (0x1fU 
                                                    & ((IData)(0x1cU) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_request_index)))))) 
                                                | (vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_bank_dram_fill_req_addr[
                                                   (7U 
                                                    & (((IData)(0x1cU) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_request_index)) 
                                                       >> 5U))] 
                                                   >> 
                                                   (0x1fU 
                                                    & ((IData)(0x1cU) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_request_index))))))
                               : 0U)),28);
        vcdp->chgBit(c+3705,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_has_request)))));
        vcdp->chgBit(c+3713,((0U != (IData)(vlTOPp->VX_cache__DOT__per_bank_dram_fill_req_valid))));
        vcdp->chgBus(c+3721,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dwb_bank),3);
        vcdp->chgBit(c+3729,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__update_use));
        vcdp->chgBit(c+3737,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__reading));
        vcdp->chgBit(c+3745,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__writing));
        vcdp->chgBus(c+3753,((0xffU & vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[7U])),8);
        __Vtemp1347[0U] = vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[0U];
        __Vtemp1347[1U] = vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[1U];
        __Vtemp1347[2U] = vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[2U];
        __Vtemp1347[3U] = vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[3U];
        __Vtemp1347[4U] = vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[4U];
        __Vtemp1347[5U] = vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[5U];
        __Vtemp1347[6U] = vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[6U];
        vcdp->chgArray(c+3761,(__Vtemp1347),224);
        vcdp->chgBus(c+3817,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bqual_bank_dram_fill_req_valid),8);
        vcdp->chgArray(c+3825,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_bank_dram_fill_req_addr),224);
        vcdp->chgBus(c+3881,(((IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bqual_bank_dram_fill_req_valid) 
                              & (~ ((IData)(1U) << (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_request_index))))),8);
        vcdp->chgBit(c+3889,((1U & ((~ (IData)((0U 
                                                != 
                                                (0xffU 
                                                 & vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[7U])))) 
                                    | (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->chgBit(c+3897,(((0U != (IData)(vlTOPp->VX_cache__DOT__per_bank_dram_fill_req_valid)) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBus(c+3905,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_request_index),3);
        vcdp->chgBit(c+3913,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_has_request));
        vcdp->chgArray(c+3921,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellinp__dfqq_queue__data_in),232);
        vcdp->chgArray(c+3985,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out),232);
        vcdp->chgBit(c+4049,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__writing));
        vcdp->chgBus(c+4057,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),8);
        vcdp->chgBus(c+4065,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgBus(c+4073,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__sel_dwb__DOT__genblk2__DOT__grant_onehot_r),8);
        vcdp->chgBus(c+4081,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__sel_dwb__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgArray(c+4089,(vlTOPp->VX_cache__DOT____Vcellout__cache_core_rsp_merge__core_rsp_data),128);
        vcdp->chgQuad(c+4121,(((0x14fU >= (0x1ffU & 
                                           ((IData)(0x2aU) 
                                            * (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__main_bank_index))))
                                ? (VL_ULL(0x3ffffffffff) 
                                   & (((0U == (0x1fU 
                                               & ((IData)(0x2aU) 
                                                  * (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__main_bank_index))))
                                        ? VL_ULL(0)
                                        : ((QData)((IData)(
                                                           vlTOPp->VX_cache__DOT__per_bank_core_rsp_tag[
                                                           ((IData)(2U) 
                                                            + 
                                                            (0xfU 
                                                             & (((IData)(0x2aU) 
                                                                 * (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__main_bank_index)) 
                                                                >> 5U)))])) 
                                           << ((IData)(0x40U) 
                                               - (0x1fU 
                                                  & ((IData)(0x2aU) 
                                                     * (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__main_bank_index)))))) 
                                      | (((QData)((IData)(
                                                          vlTOPp->VX_cache__DOT__per_bank_core_rsp_tag[
                                                          ((IData)(1U) 
                                                           + 
                                                           (0xfU 
                                                            & (((IData)(0x2aU) 
                                                                * (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__main_bank_index)) 
                                                               >> 5U)))])) 
                                          << ((0U == 
                                               (0x1fU 
                                                & ((IData)(0x2aU) 
                                                   * (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__main_bank_index))))
                                               ? 0x20U
                                               : ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(0x2aU) 
                                                      * (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__main_bank_index)))))) 
                                         | ((QData)((IData)(
                                                            vlTOPp->VX_cache__DOT__per_bank_core_rsp_tag[
                                                            (0xfU 
                                                             & (((IData)(0x2aU) 
                                                                 * (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__main_bank_index)) 
                                                                >> 5U))])) 
                                            >> (0x1fU 
                                                & ((IData)(0x2aU) 
                                                   * (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__main_bank_index)))))))
                                : VL_ULL(0))),42);
        vcdp->chgBus(c+4137,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__per_bank_core_rsp_pop_unqual),8);
        vcdp->chgBus(c+4145,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__main_bank_index),3);
        vcdp->chgBit(c+4153,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__grant_valid));
        vcdp->chgBus(c+4161,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),8);
        vcdp->chgBus(c+4169,((((IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__requests_use) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r))) 
                              | (((IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__refill_original) 
                                  ^ (IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_valid)) 
                                 & (~ (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__refill_original))))),8);
        vcdp->chgBus(c+4177,((((IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__refill_original) 
                               ^ (IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_valid)) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__refill_original)))),8);
        vcdp->chgBus(c+4185,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgBus(c+4193,(vlTOPp->VX_cache__DOT__snp_rsp_arb__DOT__fsq_bank),3);
        vcdp->chgBit(c+4201,(vlTOPp->VX_cache__DOT__snp_rsp_arb__DOT__fsq_valid));
        vcdp->chgBus(c+4209,(vlTOPp->VX_cache__DOT__snp_rsp_arb__DOT__sel_ffsq__DOT__genblk2__DOT__grant_onehot_r),8);
        vcdp->chgBus(c+4217,(vlTOPp->VX_cache__DOT__snp_rsp_arb__DOT__sel_ffsq__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgBit(c+4225,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop));
        vcdp->chgBus(c+4233,((0x1ffffffU & (IData)(
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                    >> 0x1dU)))),25);
        vcdp->chgBit(c+4241,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                            >> 0x1cU)))));
        vcdp->chgBus(c+4249,((0xfffffffU & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out))),28);
        vcdp->chgBit(c+4257,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop));
        vcdp->chgBus(c+4265,((0x1ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])),25);
        __Vtemp1348[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U];
        __Vtemp1348[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U];
        __Vtemp1348[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U];
        __Vtemp1348[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U];
        vcdp->chgArray(c+4273,(__Vtemp1348),128);
        vcdp->chgBit(c+4305,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop));
        vcdp->chgBit(c+4313,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request)))));
        vcdp->chgBit(c+4321,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request));
        vcdp->chgBus(c+4329,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index),2);
        vcdp->chgBit(c+4337,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_req_rw_st0));
        vcdp->chgBus(c+4345,((0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                      >> (0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                  << 2U))))),4);
        vcdp->chgBus(c+4353,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_req_addr_st0),30);
        vcdp->chgBus(c+4361,((((0U == (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                << 5U)))
                                ? 0U : (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                        ((IData)(1U) 
                                         + (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index)))] 
                                        << ((IData)(0x20U) 
                                            - (0x1fU 
                                               & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                  << 5U))))) 
                              | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                 (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))] 
                                 >> (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                              << 5U))))),32);
        vcdp->chgBit(c+4369,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_pop));
        vcdp->chgBit(c+4377,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible));
        vcdp->chgBit(c+4385,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_rw_st0));
        vcdp->chgBit(c+4393,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match))));
        vcdp->chgBit(c+4401,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_because_miss) 
                              & (((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 6U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x1aU))) 
                                  == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                  [0U]) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              >> 0x14U))))));
        vcdp->chgBit(c+4409,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__force_request_miss_st1e));
        vcdp->chgBit(c+4417,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__recover_mrvq_state_st2));
        vcdp->chgBit(c+4425,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_push_stall));
        vcdp->chgBit(c+4433,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwbq_push_stall));
        vcdp->chgBit(c+4441,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_push_stall));
        vcdp->chgBit(c+4449,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dram_fill_req_stall));
        vcdp->chgBit(c+4457,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__stall_bank_pipe));
        vcdp->chgBit(c+4465,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_in_pipe));
        vcdp->chgBit(c+4473,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1[0]));
        vcdp->chgBit(c+4481,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__going_to_write_st1[0]));
        vcdp->chgBit(c+4489,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual));
        vcdp->chgBit(c+4497,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop_unqual));
        vcdp->chgBit(c+4505,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop_unqual));
        vcdp->chgBit(c+4513,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop) 
                                | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_pop)) 
                               | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop)) 
                              | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop))));
        vcdp->chgBus(c+4521,((0x1ffffffU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                             ? (0x1ffffffU 
                                                & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])
                                             : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                                 ? 
                                                ((0x18fU 
                                                  >= 
                                                  (0x1ffU 
                                                   & ((IData)(0x19U) 
                                                      * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                  ? 
                                                 (0x1ffffffU 
                                                  & (((0U 
                                                       == 
                                                       (0x1fU 
                                                        & ((IData)(0x19U) 
                                                           * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                       ? 0U
                                                       : 
                                                      (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                       ((IData)(1U) 
                                                        + 
                                                        (0xfU 
                                                         & (((IData)(0x19U) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                            >> 5U)))] 
                                                       << 
                                                       ((IData)(0x20U) 
                                                        - 
                                                        (0x1fU 
                                                         & ((IData)(0x19U) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                     | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                        (0xfU 
                                                         & (((IData)(0x19U) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                            >> 5U))] 
                                                        >> 
                                                        (0x1fU 
                                                         & ((IData)(0x19U) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                                  : 0U)
                                                 : 
                                                ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop_unqual)
                                                  ? 
                                                 (0x1ffffffU 
                                                  & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_req_addr_st0 
                                                     >> 5U))
                                                  : 
                                                 ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop_unqual)
                                                   ? 
                                                  (0x1ffffffU 
                                                   & (IData)(
                                                             (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                              >> 0x1dU)))
                                                   : 0U)))))),25);
        vcdp->chgBus(c+4529,((3U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop_unqual)
                                     ? (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_req_addr_st0)
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                         ? (3U & ((
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                   << 0x1eU) 
                                                  | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                     [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                     >> 2U)))
                                         : 0U)))),2);
        vcdp->chgBus(c+4537,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                               ? ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                   << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                               [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                               >> 0x15U))
                               : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop_unqual)
                                   ? (((0U == (0x1fU 
                                               & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                  << 5U)))
                                        ? 0U : (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                                ((IData)(1U) 
                                                 + 
                                                 (3U 
                                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index)))] 
                                                << 
                                                ((IData)(0x20U) 
                                                 - 
                                                 (0x1fU 
                                                  & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                     << 5U))))) 
                                      | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                         (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))] 
                                         >> (0x1fU 
                                             & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                << 5U))))
                                   : 0U))),32);
        __Vtemp1353[0U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U]
                            : 0x39U);
        __Vtemp1353[1U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U]
                            : 0U);
        __Vtemp1353[2U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U]
                            : 0U);
        __Vtemp1353[3U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U]
                            : 0U);
        vcdp->chgArray(c+4545,(__Vtemp1353),128);
        vcdp->chgQuad(c+4577,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                ? ((VL_ULL(0x1ffffffffff80) 
                                    & (((QData)((IData)(
                                                        vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                        << 0x3eU) | 
                                       (((QData)((IData)(
                                                         vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U])) 
                                         << 0x1eU) 
                                        | (VL_ULL(0x3fffffffffffff80) 
                                           & ((QData)((IData)(
                                                              vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                              [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])) 
                                              >> 2U))))) 
                                   | (QData)((IData)(
                                                     (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_rw_st0) 
                                                       << 6U) 
                                                      | ((0x3cU 
                                                          & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                              [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                              << 0x1eU) 
                                                             | (0x3ffffffcU 
                                                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                                   >> 2U)))) 
                                                         | (3U 
                                                            & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                                                << 0xdU) 
                                                               | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                                  >> 0x13U))))))))
                                : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop_unqual)
                                    ? ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag 
                                        << 7U) | (QData)((IData)(
                                                                 (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_req_rw_st0) 
                                                                   << 6U) 
                                                                  | ((0x3cU 
                                                                      & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                                                          >> 
                                                                          (0xfU 
                                                                           & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                                              << 2U))) 
                                                                         << 2U)) 
                                                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))))))
                                    : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop_unqual)
                                        ? ((QData)((IData)(
                                                           (0xfffffffU 
                                                            & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out)))) 
                                           << 7U) : VL_ULL(0))))),49);
        vcdp->chgBit(c+4593,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
                               ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible) 
                                        & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_rw_st0))
                                        ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop_unqual) 
                                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_req_rw_st0))
                                                 ? 1U
                                                 : 0U)))));
        vcdp->chgBit(c+4601,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                     ? (1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                              [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                              >> 1U))
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? 1U : 0U)))));
        vcdp->chgBit(c+4609,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                     ? (1U & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? (1U & (IData)(
                                                         (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                          >> 0x1cU)))
                                         : 0U)))));
        vcdp->chgBit(c+4617,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1[0]));
        vcdp->chgBus(c+4625,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1[0]),25);
        vcdp->chgBus(c+4633,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1[0]),2);
        vcdp->chgBus(c+4641,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writeword_st1[0]),32);
        vcdp->chgQuad(c+4649,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__inst_meta_st1[0]),49);
        vcdp->chgArray(c+4665,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writedata_st1[0]),128);
        vcdp->chgBit(c+4697,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1[0]));
        vcdp->chgBit(c+4705,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_invalidate_st1[0]));
        vcdp->chgBit(c+4713,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_mrvq_st1[0]));
        __Vtemp1359[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp1359[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp1359[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp1359[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgBus(c+4721,((((0U == (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                                [0U] 
                                                << 5U)))
                                ? 0U : (__Vtemp1359[
                                        ((IData)(1U) 
                                         + (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                            [0U]))] 
                                        << ((IData)(0x20U) 
                                            - (0x1fU 
                                               & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                                  [0U] 
                                                  << 5U))))) 
                              | (__Vtemp1359[(3U & 
                                              vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                              [0U])] 
                                 >> (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                              [0U] 
                                              << 5U))))),32);
        __Vtemp1360[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp1360[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp1360[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp1360[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgArray(c+4729,(__Vtemp1360),128);
        vcdp->chgBus(c+4761,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c
                             [0U]),21);
        vcdp->chgBit(c+4769,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_st1e));
        vcdp->chgBit(c+4777,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dirty_st1e));
        vcdp->chgBus(c+4785,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c
                             [0U]),16);
        vcdp->chgQuad(c+4793,((VL_ULL(0x3ffffffffff) 
                               & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__inst_meta_st1
                                  [0U] >> 7U))),42);
        vcdp->chgBus(c+4809,((3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__inst_meta_st1
                                           [0U]))),2);
        vcdp->chgBit(c+4817,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__inst_meta_st1
                                            [0U] >> 6U)))));
        vcdp->chgBus(c+4825,((0xfU & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__inst_meta_st1
                                              [0U] 
                                              >> 2U)))),4);
        vcdp->chgBit(c+4833,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dirty_st1e))));
        vcdp->chgBit(c+4841,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1
                             [0U]));
        vcdp->chgBit(c+4849,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_invalidate_st1
                             [0U]));
        vcdp->chgBit(c+4857,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_to_mrvq_st1e));
        vcdp->chgBit(c+4865,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_to_mrvq_st1e) 
                              | ((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                   & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1
                                      [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                                                [0U])) 
                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                                 [0U]))));
        vcdp->chgBit(c+4873,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_because_miss));
        vcdp->chgBit(c+4881,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                             [0U]));
        vcdp->chgBit(c+4889,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_mrvq_st1
                             [0U]));
        vcdp->chgBit(c+4897,((((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                                [0U] & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_mrvq_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__recover_mrvq_state_st2)) 
                              & ((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                 << 6U) 
                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                   >> 0x1aU))) 
                                 == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                 [0U]))));
        vcdp->chgBus(c+4905,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                             [0U]),25);
        vcdp->chgBit(c+4913,((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                              [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                                      [0U]))));
        vcdp->chgBit(c+4921,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_init_ready_state_st2));
        vcdp->chgBit(c+4929,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_unqual) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)) 
                              & ((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                 << 6U) 
                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                   >> 0x1aU))) 
                                 == (0x1ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])))));
        vcdp->chgBit(c+4937,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_unqual) 
                               & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                               [0U]) & ((0x1ffffffU 
                                         & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                             << 6U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                               >> 0x1aU))) 
                                        == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                        [0U]))));
        vcdp->chgBit(c+4945,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_unqual));
        vcdp->chgBit(c+4953,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add));
        vcdp->chgBit(c+4961,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_is_mrvq));
        vcdp->chgBit(c+4969,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwbq_push_unqual) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))) & (~ 
                                                 (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_push_stall) 
                                                   | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                                  | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+4977,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwbq_push_unqual));
        vcdp->chgBit(c+4985,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_unqual) 
                              & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                 | ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU) & (~ 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                   >> 0x1bU)))))));
        vcdp->chgBit(c+4993,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_push_unqual) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+5001,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_is_dwb_in));
        vcdp->chgBit(c+5009,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_is_snp_in));
        vcdp->chgBit(c+5017,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 6U))));
        vcdp->chgBit(c+5025,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 5U))));
        vcdp->chgBit(c+5033,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_push_unqual));
        vcdp->chgQuad(c+5041,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out),54);
        vcdp->chgBit(c+5057,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__reading));
        vcdp->chgArray(c+5065,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out),153);
        vcdp->chgBit(c+5105,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__reading));
        vcdp->chgBus(c+5113,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                      >> 0x16U))),4);
        vcdp->chgBus(c+5121,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                      >> 0x12U))),4);
        vcdp->chgBus(c+5129,((0xffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                         >> 2U))),16);
        __Vtemp1363[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                                         >> 0xaU));
        __Vtemp1363[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                                         >> 0xaU));
        __Vtemp1363[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                                         >> 0xaU));
        __Vtemp1363[3U] = (0xffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                         << 0x16U) 
                                        | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                                           >> 0xaU)));
        vcdp->chgArray(c+5137,(__Vtemp1363),120);
        __Vtemp1364[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U] 
                                         >> 0xaU));
        __Vtemp1364[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                                         >> 0xaU));
        __Vtemp1364[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                                         >> 0xaU));
        __Vtemp1364[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                                         >> 0xaU));
        vcdp->chgArray(c+5169,(__Vtemp1364),128);
        vcdp->chgQuad(c+5201,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[0U]))))),42);
        vcdp->chgBit(c+5217,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty));
        vcdp->chgBit(c+5225,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__pop_qual));
        vcdp->chgBus(c+5233,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x16U) & 
                                      VL_NEGATE_I((IData)(
                                                          (1U 
                                                           & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty)))))))),4);
        vcdp->chgArray(c+5241,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out),314);
        vcdp->chgBit(c+5321,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__reading));
        vcdp->chgBus(c+5329,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->chgBus(c+5337,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgArray(c+5345,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__s0_1_c0__in),242);
        vcdp->chgBus(c+5409,((0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                              [0U])),4);
        vcdp->chgBit(c+5417,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                             [0U]));
        vcdp->chgBus(c+5425,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writeword_st1
                             [0U]),32);
        __Vtemp1365[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writedata_st1
            [0U][0U];
        __Vtemp1365[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writedata_st1
            [0U][1U];
        __Vtemp1365[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writedata_st1
            [0U][2U];
        __Vtemp1365[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writedata_st1
            [0U][3U];
        vcdp->chgArray(c+5433,(__Vtemp1365),128);
        vcdp->chgBus(c+5465,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                             [0U]),2);
        vcdp->chgBit(c+5473,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c[0]));
        vcdp->chgBit(c+5481,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c[0]));
        vcdp->chgBus(c+5489,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c[0]),16);
        vcdp->chgBus(c+5497,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c[0]),21);
        vcdp->chgArray(c+5505,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c[0]),128);
        vcdp->chgBit(c+5537,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid) 
                                    >> (0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                        [0U])))));
        vcdp->chgBit(c+5545,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty) 
                                    >> (0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                        [0U])))));
        vcdp->chgBus(c+5553,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                             [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                               [0U])]),16);
        vcdp->chgBus(c+5561,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag
                             [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                               [0U])]),21);
        __Vtemp1366[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
              [0U])][0U];
        __Vtemp1366[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
              [0U])][1U];
        __Vtemp1366[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
              [0U])][2U];
        __Vtemp1366[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
              [0U])][3U];
        vcdp->chgArray(c+5569,(__Vtemp1366),128);
        vcdp->chgBit(c+5601,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                             [0U]));
        vcdp->chgBit(c+5609,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c
                             [0U]));
        vcdp->chgBus(c+5617,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable),16);
        vcdp->chgArray(c+5625,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__data_write),128);
        vcdp->chgBit(c+5657,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__snoop_hit_no_pending));
        vcdp->chgBit(c+5665,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match));
        vcdp->chgBit(c+5673,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill));
        vcdp->chgBus(c+5681,((0x1fffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                           [0U] >> 4U))),21);
        vcdp->chgBus(c+5689,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__we),16);
        vcdp->chgBit(c+5697,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__should_write));
        vcdp->chgBit(c+5705,((((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                                [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1
                                        [0U])) & (~ 
                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                                  [0U])) 
                              & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                                 [0U]))));
        vcdp->chgBit(c+5713,(((((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                                 [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1
                                         [0U])) & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                [0U]) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                                         [0U])) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match)))));
        vcdp->chgBit(c+5721,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss));
        vcdp->chgBit(c+5729,((((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                 & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1
                                    [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                                              [0U])) 
                               & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                               [0U]) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss)))));
        vcdp->chgBit(c+5737,(((0U == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+5745,(((1U == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+5753,(((2U == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+5761,(((3U == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+5769,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable))));
        vcdp->chgArray(c+5777,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT____Vcellinp__s0_1_c0__in),167);
        vcdp->chgArray(c+5825,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__st_1e_2__in),316);
        vcdp->chgBus(c+5905,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready),16);
        vcdp->chgBus(c+5913,((0xffffU & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                                          & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                         << (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr)))),16);
        vcdp->chgBus(c+5921,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match),16);
        vcdp->chgBit(c+5929,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push));
        vcdp->chgBit(c+5937,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_pop));
        vcdp->chgBit(c+5945,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_is_mrvq))));
        vcdp->chgBit(c+5953,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__increment_head));
        vcdp->chgBit(c+5961,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready))));
        vcdp->chgBit(c+5969,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_init_ready_state_st2))));
        vcdp->chgArray(c+5977,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__cwb_queue__data_in),76);
        vcdp->chgArray(c+6001,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out),76);
        vcdp->chgBit(c+6025,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__writing));
        vcdp->chgArray(c+6033,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__dwb_queue__data_in),199);
        vcdp->chgArray(c+6089,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out),199);
        vcdp->chgBit(c+6145,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__writing));
        vcdp->chgBit(c+6153,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop));
        vcdp->chgBus(c+6161,((0x1ffffffU & (IData)(
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                    >> 0x1dU)))),25);
        vcdp->chgBit(c+6169,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                            >> 0x1cU)))));
        vcdp->chgBus(c+6177,((0xfffffffU & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out))),28);
        vcdp->chgBit(c+6185,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop));
        vcdp->chgBus(c+6193,((0x1ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])),25);
        __Vtemp1367[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U];
        __Vtemp1367[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U];
        __Vtemp1367[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U];
        __Vtemp1367[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U];
        vcdp->chgArray(c+6201,(__Vtemp1367),128);
        vcdp->chgBit(c+6233,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop));
        vcdp->chgBit(c+6241,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request)))));
        vcdp->chgBit(c+6249,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request));
        vcdp->chgBus(c+6257,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index),2);
        vcdp->chgBit(c+6265,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_req_rw_st0));
        vcdp->chgBus(c+6273,((0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                      >> (0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                  << 2U))))),4);
        vcdp->chgBus(c+6281,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_req_addr_st0),30);
        vcdp->chgBus(c+6289,((((0U == (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                << 5U)))
                                ? 0U : (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                        ((IData)(1U) 
                                         + (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index)))] 
                                        << ((IData)(0x20U) 
                                            - (0x1fU 
                                               & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                  << 5U))))) 
                              | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                 (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))] 
                                 >> (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                              << 5U))))),32);
        vcdp->chgBit(c+6297,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_pop));
        vcdp->chgBit(c+6305,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible));
        vcdp->chgBit(c+6313,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_rw_st0));
        vcdp->chgBit(c+6321,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match))));
        vcdp->chgBit(c+6329,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_because_miss) 
                              & (((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 6U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x1aU))) 
                                  == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                  [0U]) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              >> 0x14U))))));
        vcdp->chgBit(c+6337,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__force_request_miss_st1e));
        vcdp->chgBit(c+6345,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__recover_mrvq_state_st2));
        vcdp->chgBit(c+6353,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_push_stall));
        vcdp->chgBit(c+6361,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwbq_push_stall));
        vcdp->chgBit(c+6369,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_push_stall));
        vcdp->chgBit(c+6377,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dram_fill_req_stall));
        vcdp->chgBit(c+6385,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__stall_bank_pipe));
        vcdp->chgBit(c+6393,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_in_pipe));
        vcdp->chgBit(c+6401,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1[0]));
        vcdp->chgBit(c+6409,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__going_to_write_st1[0]));
        vcdp->chgBit(c+6417,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual));
        vcdp->chgBit(c+6425,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop_unqual));
        vcdp->chgBit(c+6433,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop_unqual));
        vcdp->chgBit(c+6441,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop) 
                                | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_pop)) 
                               | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop)) 
                              | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop))));
        vcdp->chgBus(c+6449,((0x1ffffffU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                             ? (0x1ffffffU 
                                                & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])
                                             : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                                 ? 
                                                ((0x18fU 
                                                  >= 
                                                  (0x1ffU 
                                                   & ((IData)(0x19U) 
                                                      * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                  ? 
                                                 (0x1ffffffU 
                                                  & (((0U 
                                                       == 
                                                       (0x1fU 
                                                        & ((IData)(0x19U) 
                                                           * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                       ? 0U
                                                       : 
                                                      (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                       ((IData)(1U) 
                                                        + 
                                                        (0xfU 
                                                         & (((IData)(0x19U) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                            >> 5U)))] 
                                                       << 
                                                       ((IData)(0x20U) 
                                                        - 
                                                        (0x1fU 
                                                         & ((IData)(0x19U) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                     | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                        (0xfU 
                                                         & (((IData)(0x19U) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                            >> 5U))] 
                                                        >> 
                                                        (0x1fU 
                                                         & ((IData)(0x19U) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                                  : 0U)
                                                 : 
                                                ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop_unqual)
                                                  ? 
                                                 (0x1ffffffU 
                                                  & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_req_addr_st0 
                                                     >> 5U))
                                                  : 
                                                 ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop_unqual)
                                                   ? 
                                                  (0x1ffffffU 
                                                   & (IData)(
                                                             (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                              >> 0x1dU)))
                                                   : 0U)))))),25);
        vcdp->chgBus(c+6457,((3U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop_unqual)
                                     ? (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_req_addr_st0)
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                         ? (3U & ((
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                   << 0x1eU) 
                                                  | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                     [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                     >> 2U)))
                                         : 0U)))),2);
        vcdp->chgBus(c+6465,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                               ? ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                   << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                               [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                               >> 0x15U))
                               : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop_unqual)
                                   ? (((0U == (0x1fU 
                                               & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                  << 5U)))
                                        ? 0U : (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                                ((IData)(1U) 
                                                 + 
                                                 (3U 
                                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index)))] 
                                                << 
                                                ((IData)(0x20U) 
                                                 - 
                                                 (0x1fU 
                                                  & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                     << 5U))))) 
                                      | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                         (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))] 
                                         >> (0x1fU 
                                             & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                << 5U))))
                                   : 0U))),32);
        __Vtemp1372[0U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U]
                            : 0x39U);
        __Vtemp1372[1U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U]
                            : 0U);
        __Vtemp1372[2U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U]
                            : 0U);
        __Vtemp1372[3U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U]
                            : 0U);
        vcdp->chgArray(c+6473,(__Vtemp1372),128);
        vcdp->chgQuad(c+6505,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                ? ((VL_ULL(0x1ffffffffff80) 
                                    & (((QData)((IData)(
                                                        vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                        << 0x3eU) | 
                                       (((QData)((IData)(
                                                         vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U])) 
                                         << 0x1eU) 
                                        | (VL_ULL(0x3fffffffffffff80) 
                                           & ((QData)((IData)(
                                                              vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                              [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])) 
                                              >> 2U))))) 
                                   | (QData)((IData)(
                                                     (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_rw_st0) 
                                                       << 6U) 
                                                      | ((0x3cU 
                                                          & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                              [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                              << 0x1eU) 
                                                             | (0x3ffffffcU 
                                                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                                   >> 2U)))) 
                                                         | (3U 
                                                            & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                                                << 0xdU) 
                                                               | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                                  >> 0x13U))))))))
                                : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop_unqual)
                                    ? ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag 
                                        << 7U) | (QData)((IData)(
                                                                 (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_req_rw_st0) 
                                                                   << 6U) 
                                                                  | ((0x3cU 
                                                                      & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                                                          >> 
                                                                          (0xfU 
                                                                           & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                                              << 2U))) 
                                                                         << 2U)) 
                                                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))))))
                                    : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop_unqual)
                                        ? ((QData)((IData)(
                                                           (0xfffffffU 
                                                            & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out)))) 
                                           << 7U) : VL_ULL(0))))),49);
        vcdp->chgBit(c+6521,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
                               ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible) 
                                        & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_rw_st0))
                                        ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop_unqual) 
                                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_req_rw_st0))
                                                 ? 1U
                                                 : 0U)))));
        vcdp->chgBit(c+6529,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                     ? (1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                              [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                              >> 1U))
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? 1U : 0U)))));
        vcdp->chgBit(c+6537,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                     ? (1U & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? (1U & (IData)(
                                                         (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                          >> 0x1cU)))
                                         : 0U)))));
        vcdp->chgBit(c+6545,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1[0]));
        vcdp->chgBus(c+6553,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1[0]),25);
        vcdp->chgBus(c+6561,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1[0]),2);
        vcdp->chgBus(c+6569,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writeword_st1[0]),32);
        vcdp->chgQuad(c+6577,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__inst_meta_st1[0]),49);
        vcdp->chgArray(c+6593,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writedata_st1[0]),128);
        vcdp->chgBit(c+6625,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1[0]));
        vcdp->chgBit(c+6633,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_invalidate_st1[0]));
        vcdp->chgBit(c+6641,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_mrvq_st1[0]));
        __Vtemp1378[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp1378[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp1378[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp1378[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgBus(c+6649,((((0U == (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                                [0U] 
                                                << 5U)))
                                ? 0U : (__Vtemp1378[
                                        ((IData)(1U) 
                                         + (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                            [0U]))] 
                                        << ((IData)(0x20U) 
                                            - (0x1fU 
                                               & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                                  [0U] 
                                                  << 5U))))) 
                              | (__Vtemp1378[(3U & 
                                              vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                              [0U])] 
                                 >> (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                              [0U] 
                                              << 5U))))),32);
        __Vtemp1379[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp1379[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp1379[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp1379[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgArray(c+6657,(__Vtemp1379),128);
        vcdp->chgBus(c+6689,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c
                             [0U]),21);
        vcdp->chgBit(c+6697,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_st1e));
        vcdp->chgBit(c+6705,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dirty_st1e));
        vcdp->chgBus(c+6713,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c
                             [0U]),16);
        vcdp->chgQuad(c+6721,((VL_ULL(0x3ffffffffff) 
                               & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__inst_meta_st1
                                  [0U] >> 7U))),42);
        vcdp->chgBus(c+6737,((3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__inst_meta_st1
                                           [0U]))),2);
        vcdp->chgBit(c+6745,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__inst_meta_st1
                                            [0U] >> 6U)))));
        vcdp->chgBus(c+6753,((0xfU & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__inst_meta_st1
                                              [0U] 
                                              >> 2U)))),4);
        vcdp->chgBit(c+6761,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dirty_st1e))));
        vcdp->chgBit(c+6769,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1
                             [0U]));
        vcdp->chgBit(c+6777,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_invalidate_st1
                             [0U]));
        vcdp->chgBit(c+6785,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_to_mrvq_st1e));
        vcdp->chgBit(c+6793,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_to_mrvq_st1e) 
                              | ((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                   & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1
                                      [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                                                [0U])) 
                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                                 [0U]))));
        vcdp->chgBit(c+6801,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_because_miss));
        vcdp->chgBit(c+6809,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                             [0U]));
        vcdp->chgBit(c+6817,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_mrvq_st1
                             [0U]));
        vcdp->chgBit(c+6825,((((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                                [0U] & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_mrvq_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__recover_mrvq_state_st2)) 
                              & ((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                 << 6U) 
                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                   >> 0x1aU))) 
                                 == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                 [0U]))));
        vcdp->chgBus(c+6833,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                             [0U]),25);
        vcdp->chgBit(c+6841,((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                              [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                                      [0U]))));
        vcdp->chgBit(c+6849,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_init_ready_state_st2));
        vcdp->chgBit(c+6857,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_unqual) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)) 
                              & ((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                 << 6U) 
                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                   >> 0x1aU))) 
                                 == (0x1ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])))));
        vcdp->chgBit(c+6865,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_unqual) 
                               & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                               [0U]) & ((0x1ffffffU 
                                         & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                             << 6U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                               >> 0x1aU))) 
                                        == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                        [0U]))));
        vcdp->chgBit(c+6873,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_unqual));
        vcdp->chgBit(c+6881,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add));
        vcdp->chgBit(c+6889,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_is_mrvq));
        vcdp->chgBit(c+6897,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwbq_push_unqual) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))) & (~ 
                                                 (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_push_stall) 
                                                   | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                                  | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+6905,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwbq_push_unqual));
        vcdp->chgBit(c+6913,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_unqual) 
                              & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                 | ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU) & (~ 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                   >> 0x1bU)))))));
        vcdp->chgBit(c+6921,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_push_unqual) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+6929,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_is_dwb_in));
        vcdp->chgBit(c+6937,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_is_snp_in));
        vcdp->chgBit(c+6945,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 6U))));
        vcdp->chgBit(c+6953,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 5U))));
        vcdp->chgBit(c+6961,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_push_unqual));
        vcdp->chgQuad(c+6969,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out),54);
        vcdp->chgBit(c+6985,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__reading));
        vcdp->chgArray(c+6993,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out),153);
        vcdp->chgBit(c+7033,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__reading));
        vcdp->chgBus(c+7041,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                      >> 0x16U))),4);
        vcdp->chgBus(c+7049,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                      >> 0x12U))),4);
        vcdp->chgBus(c+7057,((0xffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                         >> 2U))),16);
        __Vtemp1382[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                                         >> 0xaU));
        __Vtemp1382[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                                         >> 0xaU));
        __Vtemp1382[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                                         >> 0xaU));
        __Vtemp1382[3U] = (0xffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                         << 0x16U) 
                                        | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                                           >> 0xaU)));
        vcdp->chgArray(c+7065,(__Vtemp1382),120);
        __Vtemp1383[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U] 
                                         >> 0xaU));
        __Vtemp1383[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                                         >> 0xaU));
        __Vtemp1383[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                                         >> 0xaU));
        __Vtemp1383[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                                         >> 0xaU));
        vcdp->chgArray(c+7097,(__Vtemp1383),128);
        vcdp->chgQuad(c+7129,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[0U]))))),42);
        vcdp->chgBit(c+7145,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty));
        vcdp->chgBit(c+7153,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__pop_qual));
        vcdp->chgBus(c+7161,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x16U) & 
                                      VL_NEGATE_I((IData)(
                                                          (1U 
                                                           & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty)))))))),4);
        vcdp->chgArray(c+7169,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out),314);
        vcdp->chgBit(c+7249,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__reading));
        vcdp->chgBus(c+7257,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->chgBus(c+7265,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgArray(c+7273,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__s0_1_c0__in),242);
        vcdp->chgBus(c+7337,((0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                              [0U])),4);
        vcdp->chgBit(c+7345,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                             [0U]));
        vcdp->chgBus(c+7353,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writeword_st1
                             [0U]),32);
        __Vtemp1384[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writedata_st1
            [0U][0U];
        __Vtemp1384[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writedata_st1
            [0U][1U];
        __Vtemp1384[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writedata_st1
            [0U][2U];
        __Vtemp1384[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writedata_st1
            [0U][3U];
        vcdp->chgArray(c+7361,(__Vtemp1384),128);
        vcdp->chgBus(c+7393,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                             [0U]),2);
        vcdp->chgBit(c+7401,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c[0]));
        vcdp->chgBit(c+7409,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c[0]));
        vcdp->chgBus(c+7417,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c[0]),16);
        vcdp->chgBus(c+7425,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c[0]),21);
        vcdp->chgArray(c+7433,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c[0]),128);
        vcdp->chgBit(c+7465,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid) 
                                    >> (0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                        [0U])))));
        vcdp->chgBit(c+7473,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty) 
                                    >> (0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                        [0U])))));
        vcdp->chgBus(c+7481,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                             [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                               [0U])]),16);
        vcdp->chgBus(c+7489,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag
                             [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                               [0U])]),21);
        __Vtemp1385[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
              [0U])][0U];
        __Vtemp1385[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
              [0U])][1U];
        __Vtemp1385[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
              [0U])][2U];
        __Vtemp1385[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
              [0U])][3U];
        vcdp->chgArray(c+7497,(__Vtemp1385),128);
        vcdp->chgBit(c+7529,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                             [0U]));
        vcdp->chgBit(c+7537,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c
                             [0U]));
        vcdp->chgBus(c+7545,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable),16);
        vcdp->chgArray(c+7553,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__data_write),128);
        vcdp->chgBit(c+7585,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__snoop_hit_no_pending));
        vcdp->chgBit(c+7593,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match));
        vcdp->chgBit(c+7601,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill));
        vcdp->chgBus(c+7609,((0x1fffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                           [0U] >> 4U))),21);
        vcdp->chgBus(c+7617,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__we),16);
        vcdp->chgBit(c+7625,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__should_write));
        vcdp->chgBit(c+7633,((((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                                [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1
                                        [0U])) & (~ 
                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                                  [0U])) 
                              & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                                 [0U]))));
        vcdp->chgBit(c+7641,(((((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                                 [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1
                                         [0U])) & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                [0U]) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                                         [0U])) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match)))));
        vcdp->chgBit(c+7649,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss));
        vcdp->chgBit(c+7657,((((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                 & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1
                                    [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                                              [0U])) 
                               & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                               [0U]) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss)))));
        vcdp->chgBit(c+7665,(((0U == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+7673,(((1U == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+7681,(((2U == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+7689,(((3U == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+7697,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable))));
        vcdp->chgArray(c+7705,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT____Vcellinp__s0_1_c0__in),167);
        vcdp->chgArray(c+7753,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__st_1e_2__in),316);
        vcdp->chgBus(c+7833,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready),16);
        vcdp->chgBus(c+7841,((0xffffU & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                                          & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                         << (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr)))),16);
        vcdp->chgBus(c+7849,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match),16);
        vcdp->chgBit(c+7857,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push));
        vcdp->chgBit(c+7865,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_pop));
        vcdp->chgBit(c+7873,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_is_mrvq))));
        vcdp->chgBit(c+7881,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__increment_head));
        vcdp->chgBit(c+7889,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready))));
        vcdp->chgBit(c+7897,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_init_ready_state_st2))));
        vcdp->chgArray(c+7905,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__cwb_queue__data_in),76);
        vcdp->chgArray(c+7929,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out),76);
        vcdp->chgBit(c+7953,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__writing));
        vcdp->chgArray(c+7961,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__dwb_queue__data_in),199);
        vcdp->chgArray(c+8017,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out),199);
        vcdp->chgBit(c+8073,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__writing));
        vcdp->chgBit(c+8081,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop));
        vcdp->chgBus(c+8089,((0x1ffffffU & (IData)(
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                    >> 0x1dU)))),25);
        vcdp->chgBit(c+8097,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                            >> 0x1cU)))));
        vcdp->chgBus(c+8105,((0xfffffffU & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out))),28);
        vcdp->chgBit(c+8113,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop));
        vcdp->chgBus(c+8121,((0x1ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])),25);
        __Vtemp1386[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U];
        __Vtemp1386[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U];
        __Vtemp1386[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U];
        __Vtemp1386[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U];
        vcdp->chgArray(c+8129,(__Vtemp1386),128);
        vcdp->chgBit(c+8161,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop));
        vcdp->chgBit(c+8169,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request)))));
        vcdp->chgBit(c+8177,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request));
        vcdp->chgBus(c+8185,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index),2);
        vcdp->chgBit(c+8193,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_req_rw_st0));
        vcdp->chgBus(c+8201,((0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                      >> (0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                  << 2U))))),4);
        vcdp->chgBus(c+8209,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_req_addr_st0),30);
        vcdp->chgBus(c+8217,((((0U == (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                << 5U)))
                                ? 0U : (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                        ((IData)(1U) 
                                         + (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index)))] 
                                        << ((IData)(0x20U) 
                                            - (0x1fU 
                                               & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                  << 5U))))) 
                              | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                 (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))] 
                                 >> (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                              << 5U))))),32);
        vcdp->chgBit(c+8225,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_pop));
        vcdp->chgBit(c+8233,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible));
        vcdp->chgBit(c+8241,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_rw_st0));
        vcdp->chgBit(c+8249,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match))));
        vcdp->chgBit(c+8257,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_because_miss) 
                              & (((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 6U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x1aU))) 
                                  == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                  [0U]) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              >> 0x14U))))));
        vcdp->chgBit(c+8265,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__force_request_miss_st1e));
        vcdp->chgBit(c+8273,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__recover_mrvq_state_st2));
        vcdp->chgBit(c+8281,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_push_stall));
        vcdp->chgBit(c+8289,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwbq_push_stall));
        vcdp->chgBit(c+8297,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_push_stall));
        vcdp->chgBit(c+8305,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dram_fill_req_stall));
        vcdp->chgBit(c+8313,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__stall_bank_pipe));
        vcdp->chgBit(c+8321,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_in_pipe));
        vcdp->chgBit(c+8329,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1[0]));
        vcdp->chgBit(c+8337,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__going_to_write_st1[0]));
        vcdp->chgBit(c+8345,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual));
        vcdp->chgBit(c+8353,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop_unqual));
        vcdp->chgBit(c+8361,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop_unqual));
        vcdp->chgBit(c+8369,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop) 
                                | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_pop)) 
                               | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop)) 
                              | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop))));
        vcdp->chgBus(c+8377,((0x1ffffffU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                             ? (0x1ffffffU 
                                                & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])
                                             : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                                 ? 
                                                ((0x18fU 
                                                  >= 
                                                  (0x1ffU 
                                                   & ((IData)(0x19U) 
                                                      * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                  ? 
                                                 (0x1ffffffU 
                                                  & (((0U 
                                                       == 
                                                       (0x1fU 
                                                        & ((IData)(0x19U) 
                                                           * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                       ? 0U
                                                       : 
                                                      (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                       ((IData)(1U) 
                                                        + 
                                                        (0xfU 
                                                         & (((IData)(0x19U) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                            >> 5U)))] 
                                                       << 
                                                       ((IData)(0x20U) 
                                                        - 
                                                        (0x1fU 
                                                         & ((IData)(0x19U) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                     | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                        (0xfU 
                                                         & (((IData)(0x19U) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                            >> 5U))] 
                                                        >> 
                                                        (0x1fU 
                                                         & ((IData)(0x19U) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                                  : 0U)
                                                 : 
                                                ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop_unqual)
                                                  ? 
                                                 (0x1ffffffU 
                                                  & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_req_addr_st0 
                                                     >> 5U))
                                                  : 
                                                 ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop_unqual)
                                                   ? 
                                                  (0x1ffffffU 
                                                   & (IData)(
                                                             (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                              >> 0x1dU)))
                                                   : 0U)))))),25);
        vcdp->chgBus(c+8385,((3U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop_unqual)
                                     ? (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_req_addr_st0)
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                         ? (3U & ((
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                   << 0x1eU) 
                                                  | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                     [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                     >> 2U)))
                                         : 0U)))),2);
        vcdp->chgBus(c+8393,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                               ? ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                   << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                               [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                               >> 0x15U))
                               : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop_unqual)
                                   ? (((0U == (0x1fU 
                                               & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                  << 5U)))
                                        ? 0U : (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                                ((IData)(1U) 
                                                 + 
                                                 (3U 
                                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index)))] 
                                                << 
                                                ((IData)(0x20U) 
                                                 - 
                                                 (0x1fU 
                                                  & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                     << 5U))))) 
                                      | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                         (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))] 
                                         >> (0x1fU 
                                             & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                << 5U))))
                                   : 0U))),32);
        __Vtemp1391[0U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U]
                            : 0x39U);
        __Vtemp1391[1U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U]
                            : 0U);
        __Vtemp1391[2U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U]
                            : 0U);
        __Vtemp1391[3U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U]
                            : 0U);
        vcdp->chgArray(c+8401,(__Vtemp1391),128);
        vcdp->chgQuad(c+8433,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                ? ((VL_ULL(0x1ffffffffff80) 
                                    & (((QData)((IData)(
                                                        vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                        << 0x3eU) | 
                                       (((QData)((IData)(
                                                         vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U])) 
                                         << 0x1eU) 
                                        | (VL_ULL(0x3fffffffffffff80) 
                                           & ((QData)((IData)(
                                                              vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                              [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])) 
                                              >> 2U))))) 
                                   | (QData)((IData)(
                                                     (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_rw_st0) 
                                                       << 6U) 
                                                      | ((0x3cU 
                                                          & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                              [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                              << 0x1eU) 
                                                             | (0x3ffffffcU 
                                                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                                   >> 2U)))) 
                                                         | (3U 
                                                            & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                                                << 0xdU) 
                                                               | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                                  >> 0x13U))))))))
                                : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop_unqual)
                                    ? ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag 
                                        << 7U) | (QData)((IData)(
                                                                 (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_req_rw_st0) 
                                                                   << 6U) 
                                                                  | ((0x3cU 
                                                                      & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                                                          >> 
                                                                          (0xfU 
                                                                           & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                                              << 2U))) 
                                                                         << 2U)) 
                                                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))))))
                                    : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop_unqual)
                                        ? ((QData)((IData)(
                                                           (0xfffffffU 
                                                            & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out)))) 
                                           << 7U) : VL_ULL(0))))),49);
        vcdp->chgBit(c+8449,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
                               ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible) 
                                        & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_rw_st0))
                                        ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop_unqual) 
                                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_req_rw_st0))
                                                 ? 1U
                                                 : 0U)))));
        vcdp->chgBit(c+8457,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                     ? (1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                              [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                              >> 1U))
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? 1U : 0U)))));
        vcdp->chgBit(c+8465,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                     ? (1U & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? (1U & (IData)(
                                                         (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                          >> 0x1cU)))
                                         : 0U)))));
        vcdp->chgBit(c+8473,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1[0]));
        vcdp->chgBus(c+8481,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1[0]),25);
        vcdp->chgBus(c+8489,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1[0]),2);
        vcdp->chgBus(c+8497,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writeword_st1[0]),32);
        vcdp->chgQuad(c+8505,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__inst_meta_st1[0]),49);
        vcdp->chgArray(c+8521,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writedata_st1[0]),128);
        vcdp->chgBit(c+8553,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1[0]));
        vcdp->chgBit(c+8561,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_invalidate_st1[0]));
        vcdp->chgBit(c+8569,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_mrvq_st1[0]));
        __Vtemp1397[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp1397[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp1397[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp1397[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgBus(c+8577,((((0U == (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                                [0U] 
                                                << 5U)))
                                ? 0U : (__Vtemp1397[
                                        ((IData)(1U) 
                                         + (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                            [0U]))] 
                                        << ((IData)(0x20U) 
                                            - (0x1fU 
                                               & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                                  [0U] 
                                                  << 5U))))) 
                              | (__Vtemp1397[(3U & 
                                              vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                              [0U])] 
                                 >> (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                              [0U] 
                                              << 5U))))),32);
        __Vtemp1398[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp1398[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp1398[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp1398[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgArray(c+8585,(__Vtemp1398),128);
        vcdp->chgBus(c+8617,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c
                             [0U]),21);
        vcdp->chgBit(c+8625,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_st1e));
        vcdp->chgBit(c+8633,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dirty_st1e));
        vcdp->chgBus(c+8641,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c
                             [0U]),16);
        vcdp->chgQuad(c+8649,((VL_ULL(0x3ffffffffff) 
                               & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__inst_meta_st1
                                  [0U] >> 7U))),42);
        vcdp->chgBus(c+8665,((3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__inst_meta_st1
                                           [0U]))),2);
        vcdp->chgBit(c+8673,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__inst_meta_st1
                                            [0U] >> 6U)))));
        vcdp->chgBus(c+8681,((0xfU & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__inst_meta_st1
                                              [0U] 
                                              >> 2U)))),4);
        vcdp->chgBit(c+8689,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dirty_st1e))));
        vcdp->chgBit(c+8697,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1
                             [0U]));
        vcdp->chgBit(c+8705,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_invalidate_st1
                             [0U]));
        vcdp->chgBit(c+8713,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_to_mrvq_st1e));
        vcdp->chgBit(c+8721,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_to_mrvq_st1e) 
                              | ((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                   & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1
                                      [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                                                [0U])) 
                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                                 [0U]))));
        vcdp->chgBit(c+8729,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_because_miss));
        vcdp->chgBit(c+8737,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                             [0U]));
        vcdp->chgBit(c+8745,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_mrvq_st1
                             [0U]));
        vcdp->chgBit(c+8753,((((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                                [0U] & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_mrvq_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__recover_mrvq_state_st2)) 
                              & ((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                 << 6U) 
                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                   >> 0x1aU))) 
                                 == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                 [0U]))));
        vcdp->chgBus(c+8761,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                             [0U]),25);
        vcdp->chgBit(c+8769,((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                              [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                                      [0U]))));
        vcdp->chgBit(c+8777,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_init_ready_state_st2));
        vcdp->chgBit(c+8785,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_unqual) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)) 
                              & ((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                 << 6U) 
                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                   >> 0x1aU))) 
                                 == (0x1ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])))));
        vcdp->chgBit(c+8793,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_unqual) 
                               & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                               [0U]) & ((0x1ffffffU 
                                         & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                             << 6U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                               >> 0x1aU))) 
                                        == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                        [0U]))));
        vcdp->chgBit(c+8801,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_unqual));
        vcdp->chgBit(c+8809,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add));
        vcdp->chgBit(c+8817,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_is_mrvq));
        vcdp->chgBit(c+8825,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwbq_push_unqual) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))) & (~ 
                                                 (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_push_stall) 
                                                   | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                                  | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+8833,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwbq_push_unqual));
        vcdp->chgBit(c+8841,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_unqual) 
                              & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                 | ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU) & (~ 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                   >> 0x1bU)))))));
        vcdp->chgBit(c+8849,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_push_unqual) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+8857,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_is_dwb_in));
        vcdp->chgBit(c+8865,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_is_snp_in));
        vcdp->chgBit(c+8873,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 6U))));
        vcdp->chgBit(c+8881,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 5U))));
        vcdp->chgBit(c+8889,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_push_unqual));
        vcdp->chgQuad(c+8897,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out),54);
        vcdp->chgBit(c+8913,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__reading));
        vcdp->chgArray(c+8921,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out),153);
        vcdp->chgBit(c+8961,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__reading));
        vcdp->chgBus(c+8969,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                      >> 0x16U))),4);
        vcdp->chgBus(c+8977,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                      >> 0x12U))),4);
        vcdp->chgBus(c+8985,((0xffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                         >> 2U))),16);
        __Vtemp1401[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                                         >> 0xaU));
        __Vtemp1401[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                                         >> 0xaU));
        __Vtemp1401[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                                         >> 0xaU));
        __Vtemp1401[3U] = (0xffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                         << 0x16U) 
                                        | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                                           >> 0xaU)));
        vcdp->chgArray(c+8993,(__Vtemp1401),120);
        __Vtemp1402[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U] 
                                         >> 0xaU));
        __Vtemp1402[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                                         >> 0xaU));
        __Vtemp1402[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                                         >> 0xaU));
        __Vtemp1402[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                                         >> 0xaU));
        vcdp->chgArray(c+9025,(__Vtemp1402),128);
        vcdp->chgQuad(c+9057,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[0U]))))),42);
        vcdp->chgBit(c+9073,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty));
        vcdp->chgBit(c+9081,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__pop_qual));
        vcdp->chgBus(c+9089,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x16U) & 
                                      VL_NEGATE_I((IData)(
                                                          (1U 
                                                           & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty)))))))),4);
        vcdp->chgArray(c+9097,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out),314);
        vcdp->chgBit(c+9177,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__reading));
        vcdp->chgBus(c+9185,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->chgBus(c+9193,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgArray(c+9201,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__s0_1_c0__in),242);
        vcdp->chgBus(c+9265,((0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                              [0U])),4);
        vcdp->chgBit(c+9273,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                             [0U]));
        vcdp->chgBus(c+9281,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writeword_st1
                             [0U]),32);
        __Vtemp1403[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writedata_st1
            [0U][0U];
        __Vtemp1403[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writedata_st1
            [0U][1U];
        __Vtemp1403[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writedata_st1
            [0U][2U];
        __Vtemp1403[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writedata_st1
            [0U][3U];
        vcdp->chgArray(c+9289,(__Vtemp1403),128);
        vcdp->chgBus(c+9321,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                             [0U]),2);
        vcdp->chgBit(c+9329,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c[0]));
        vcdp->chgBit(c+9337,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c[0]));
        vcdp->chgBus(c+9345,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c[0]),16);
        vcdp->chgBus(c+9353,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c[0]),21);
        vcdp->chgArray(c+9361,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c[0]),128);
        vcdp->chgBit(c+9393,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid) 
                                    >> (0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                        [0U])))));
        vcdp->chgBit(c+9401,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty) 
                                    >> (0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                        [0U])))));
        vcdp->chgBus(c+9409,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                             [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                               [0U])]),16);
        vcdp->chgBus(c+9417,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag
                             [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                               [0U])]),21);
        __Vtemp1404[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
              [0U])][0U];
        __Vtemp1404[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
              [0U])][1U];
        __Vtemp1404[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
              [0U])][2U];
        __Vtemp1404[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
              [0U])][3U];
        vcdp->chgArray(c+9425,(__Vtemp1404),128);
        vcdp->chgBit(c+9457,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                             [0U]));
        vcdp->chgBit(c+9465,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c
                             [0U]));
        vcdp->chgBus(c+9473,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable),16);
        vcdp->chgArray(c+9481,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__data_write),128);
        vcdp->chgBit(c+9513,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__snoop_hit_no_pending));
        vcdp->chgBit(c+9521,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match));
        vcdp->chgBit(c+9529,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill));
        vcdp->chgBus(c+9537,((0x1fffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                           [0U] >> 4U))),21);
        vcdp->chgBus(c+9545,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__we),16);
        vcdp->chgBit(c+9553,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__should_write));
        vcdp->chgBit(c+9561,((((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                                [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1
                                        [0U])) & (~ 
                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                                  [0U])) 
                              & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                                 [0U]))));
        vcdp->chgBit(c+9569,(((((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                                 [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1
                                         [0U])) & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                [0U]) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                                         [0U])) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match)))));
        vcdp->chgBit(c+9577,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss));
        vcdp->chgBit(c+9585,((((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                 & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1
                                    [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                                              [0U])) 
                               & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                               [0U]) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss)))));
        vcdp->chgBit(c+9593,(((0U == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+9601,(((1U == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+9609,(((2U == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+9617,(((3U == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+9625,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable))));
        vcdp->chgArray(c+9633,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT____Vcellinp__s0_1_c0__in),167);
        vcdp->chgArray(c+9681,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__st_1e_2__in),316);
        vcdp->chgBus(c+9761,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready),16);
        vcdp->chgBus(c+9769,((0xffffU & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                                          & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                         << (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr)))),16);
        vcdp->chgBus(c+9777,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match),16);
        vcdp->chgBit(c+9785,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push));
        vcdp->chgBit(c+9793,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_pop));
        vcdp->chgBit(c+9801,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_is_mrvq))));
        vcdp->chgBit(c+9809,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__increment_head));
        vcdp->chgBit(c+9817,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready))));
        vcdp->chgBit(c+9825,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_init_ready_state_st2))));
        vcdp->chgArray(c+9833,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__cwb_queue__data_in),76);
        vcdp->chgArray(c+9857,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out),76);
        vcdp->chgBit(c+9881,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__writing));
        vcdp->chgArray(c+9889,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__dwb_queue__data_in),199);
        vcdp->chgArray(c+9945,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out),199);
        vcdp->chgBit(c+10001,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__writing));
        vcdp->chgBit(c+10009,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop));
        vcdp->chgBus(c+10017,((0x1ffffffU & (IData)(
                                                    (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                     >> 0x1dU)))),25);
        vcdp->chgBit(c+10025,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                             >> 0x1cU)))));
        vcdp->chgBus(c+10033,((0xfffffffU & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out))),28);
        vcdp->chgBit(c+10041,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop));
        vcdp->chgBus(c+10049,((0x1ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])),25);
        __Vtemp1405[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U];
        __Vtemp1405[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U];
        __Vtemp1405[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U];
        __Vtemp1405[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U];
        vcdp->chgArray(c+10057,(__Vtemp1405),128);
        vcdp->chgBit(c+10089,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop));
        vcdp->chgBit(c+10097,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request)))));
        vcdp->chgBit(c+10105,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request));
        vcdp->chgBus(c+10113,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index),2);
        vcdp->chgBit(c+10121,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_req_rw_st0));
        vcdp->chgBus(c+10129,((0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                       >> (0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 2U))))),4);
        vcdp->chgBus(c+10137,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_req_addr_st0),30);
        vcdp->chgBus(c+10145,((((0U == (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                 << 5U)))
                                 ? 0U : (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                         ((IData)(1U) 
                                          + (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index)))] 
                                         << ((IData)(0x20U) 
                                             - (0x1fU 
                                                & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 5U))))) 
                               | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                  (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))] 
                                  >> (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                               << 5U))))),32);
        vcdp->chgBit(c+10153,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_pop));
        vcdp->chgBit(c+10161,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible));
        vcdp->chgBit(c+10169,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_rw_st0));
        vcdp->chgBit(c+10177,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match))));
        vcdp->chgBit(c+10185,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_because_miss) 
                               & (((0x1ffffffU & ((
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                   << 6U) 
                                                  | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                     >> 0x1aU))) 
                                   == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                   [0U]) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                               >> 0x14U))))));
        vcdp->chgBit(c+10193,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__force_request_miss_st1e));
        vcdp->chgBit(c+10201,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__recover_mrvq_state_st2));
        vcdp->chgBit(c+10209,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_push_stall));
        vcdp->chgBit(c+10217,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwbq_push_stall));
        vcdp->chgBit(c+10225,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_push_stall));
        vcdp->chgBit(c+10233,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dram_fill_req_stall));
        vcdp->chgBit(c+10241,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__stall_bank_pipe));
        vcdp->chgBit(c+10249,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_in_pipe));
        vcdp->chgBit(c+10257,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1[0]));
        vcdp->chgBit(c+10265,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__going_to_write_st1[0]));
        vcdp->chgBit(c+10273,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual));
        vcdp->chgBit(c+10281,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop_unqual));
        vcdp->chgBit(c+10289,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop_unqual));
        vcdp->chgBit(c+10297,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop) 
                                 | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_pop)) 
                                | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop)) 
                               | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop))));
        vcdp->chgBus(c+10305,((0x1ffffffU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                              ? (0x1ffffffU 
                                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])
                                              : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                                  ? 
                                                 ((0x18fU 
                                                   >= 
                                                   (0x1ffU 
                                                    & ((IData)(0x19U) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                   ? 
                                                  (0x1ffffffU 
                                                   & (((0U 
                                                        == 
                                                        (0x1fU 
                                                         & ((IData)(0x19U) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                        ? 0U
                                                        : 
                                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                        ((IData)(1U) 
                                                         + 
                                                         (0xfU 
                                                          & (((IData)(0x19U) 
                                                              * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                             >> 5U)))] 
                                                        << 
                                                        ((IData)(0x20U) 
                                                         - 
                                                         (0x1fU 
                                                          & ((IData)(0x19U) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                      | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                         (0xfU 
                                                          & (((IData)(0x19U) 
                                                              * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                             >> 5U))] 
                                                         >> 
                                                         (0x1fU 
                                                          & ((IData)(0x19U) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                                   : 0U)
                                                  : 
                                                 ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop_unqual)
                                                   ? 
                                                  (0x1ffffffU 
                                                   & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_req_addr_st0 
                                                      >> 5U))
                                                   : 
                                                  ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop_unqual)
                                                    ? 
                                                   (0x1ffffffU 
                                                    & (IData)(
                                                              (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                               >> 0x1dU)))
                                                    : 0U)))))),25);
        vcdp->chgBus(c+10313,((3U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop_unqual)
                                      ? (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_req_addr_st0)
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                          ? (3U & (
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                    << 0x1eU) 
                                                   | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                      >> 2U)))
                                          : 0U)))),2);
        vcdp->chgBus(c+10321,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                ? ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                    << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                >> 0x15U))
                                : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop_unqual)
                                    ? (((0U == (0x1fU 
                                                & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 5U)))
                                         ? 0U : (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                                 ((IData)(1U) 
                                                  + 
                                                  (3U 
                                                   & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index)))] 
                                                 << 
                                                 ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                      << 5U))))) 
                                       | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                          (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))] 
                                          >> (0x1fU 
                                              & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                 << 5U))))
                                    : 0U))),32);
        __Vtemp1410[0U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U]
                            : 0x39U);
        __Vtemp1410[1U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U]
                            : 0U);
        __Vtemp1410[2U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U]
                            : 0U);
        __Vtemp1410[3U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U]
                            : 0U);
        vcdp->chgArray(c+10329,(__Vtemp1410),128);
        vcdp->chgQuad(c+10361,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                 ? ((VL_ULL(0x1ffffffffff80) 
                                     & (((QData)((IData)(
                                                         vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                         << 0x3eU) 
                                        | (((QData)((IData)(
                                                            vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                            [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U])) 
                                            << 0x1eU) 
                                           | (VL_ULL(0x3fffffffffffff80) 
                                              & ((QData)((IData)(
                                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                 [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])) 
                                                 >> 2U))))) 
                                    | (QData)((IData)(
                                                      (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_rw_st0) 
                                                        << 6U) 
                                                       | ((0x3cU 
                                                           & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                               [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                               << 0x1eU) 
                                                              | (0x3ffffffcU 
                                                                 & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                                    >> 2U)))) 
                                                          | (3U 
                                                             & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                 [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                                                 << 0xdU) 
                                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                                   >> 0x13U))))))))
                                 : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop_unqual)
                                     ? ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag 
                                         << 7U) | (QData)((IData)(
                                                                  (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_req_rw_st0) 
                                                                    << 6U) 
                                                                   | ((0x3cU 
                                                                       & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                                                           >> 
                                                                           (0xfU 
                                                                            & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                                               << 2U))) 
                                                                          << 2U)) 
                                                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))))))
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? ((QData)((IData)(
                                                            (0xfffffffU 
                                                             & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out)))) 
                                            << 7U) : VL_ULL(0))))),49);
        vcdp->chgBit(c+10377,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible) 
                                         & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_rw_st0))
                                         ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop_unqual) 
                                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_req_rw_st0))
                                                  ? 1U
                                                  : 0U)))));
        vcdp->chgBit(c+10385,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                               [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                               >> 1U))
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? 1U : 0U)))));
        vcdp->chgBit(c+10393,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? (1U & (IData)(
                                                          (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                           >> 0x1cU)))
                                          : 0U)))));
        vcdp->chgBit(c+10401,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1[0]));
        vcdp->chgBus(c+10409,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1[0]),25);
        vcdp->chgBus(c+10417,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1[0]),2);
        vcdp->chgBus(c+10425,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writeword_st1[0]),32);
        vcdp->chgQuad(c+10433,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__inst_meta_st1[0]),49);
        vcdp->chgArray(c+10449,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writedata_st1[0]),128);
        vcdp->chgBit(c+10481,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1[0]));
        vcdp->chgBit(c+10489,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_invalidate_st1[0]));
        vcdp->chgBit(c+10497,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_mrvq_st1[0]));
        __Vtemp1416[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp1416[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp1416[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp1416[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgBus(c+10505,((((0U == (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                                 [0U] 
                                                 << 5U)))
                                 ? 0U : (__Vtemp1416[
                                         ((IData)(1U) 
                                          + (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                             [0U]))] 
                                         << ((IData)(0x20U) 
                                             - (0x1fU 
                                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                                   [0U] 
                                                   << 5U))))) 
                               | (__Vtemp1416[(3U & 
                                               vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                               [0U])] 
                                  >> (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                               [0U] 
                                               << 5U))))),32);
        __Vtemp1417[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp1417[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp1417[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp1417[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgArray(c+10513,(__Vtemp1417),128);
        vcdp->chgBus(c+10545,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c
                              [0U]),21);
        vcdp->chgBit(c+10553,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_st1e));
        vcdp->chgBit(c+10561,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dirty_st1e));
        vcdp->chgBus(c+10569,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c
                              [0U]),16);
        vcdp->chgQuad(c+10577,((VL_ULL(0x3ffffffffff) 
                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__inst_meta_st1
                                   [0U] >> 7U))),42);
        vcdp->chgBus(c+10593,((3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__inst_meta_st1
                                            [0U]))),2);
        vcdp->chgBit(c+10601,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__inst_meta_st1
                                             [0U] >> 6U)))));
        vcdp->chgBus(c+10609,((0xfU & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__inst_meta_st1
                                               [0U] 
                                               >> 2U)))),4);
        vcdp->chgBit(c+10617,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dirty_st1e))));
        vcdp->chgBit(c+10625,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1
                              [0U]));
        vcdp->chgBit(c+10633,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_invalidate_st1
                              [0U]));
        vcdp->chgBit(c+10641,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_to_mrvq_st1e));
        vcdp->chgBit(c+10649,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_to_mrvq_st1e) 
                               | ((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                    & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1
                                       [0U])) & (~ 
                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                                                 [0U])) 
                                  & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                                  [0U]))));
        vcdp->chgBit(c+10657,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_because_miss));
        vcdp->chgBit(c+10665,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                              [0U]));
        vcdp->chgBit(c+10673,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_mrvq_st1
                              [0U]));
        vcdp->chgBit(c+10681,((((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                                 [0U] & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_mrvq_st1
                                 [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__recover_mrvq_state_st2)) 
                               & ((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 6U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x1aU))) 
                                  == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                  [0U]))));
        vcdp->chgBus(c+10689,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                              [0U]),25);
        vcdp->chgBit(c+10697,((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                               [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                                       [0U]))));
        vcdp->chgBit(c+10705,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_init_ready_state_st2));
        vcdp->chgBit(c+10713,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_unqual) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)) 
                               & ((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 6U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x1aU))) 
                                  == (0x1ffffffU & 
                                      vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])))));
        vcdp->chgBit(c+10721,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_unqual) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                                [0U]) & ((0x1ffffffU 
                                          & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 6U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x1aU))) 
                                         == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                         [0U]))));
        vcdp->chgBit(c+10729,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_unqual));
        vcdp->chgBit(c+10737,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add));
        vcdp->chgBit(c+10745,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_is_mrvq));
        vcdp->chgBit(c+10753,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwbq_push_unqual) 
                                 & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                                & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                      >> 6U))) & (~ 
                                                  (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_push_stall) 
                                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                                   | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+10761,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwbq_push_unqual));
        vcdp->chgBit(c+10769,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU)))))));
        vcdp->chgBit(c+10777,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_push_unqual) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwbq_push_stall) 
                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+10785,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_is_dwb_in));
        vcdp->chgBit(c+10793,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_is_snp_in));
        vcdp->chgBit(c+10801,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 6U))));
        vcdp->chgBit(c+10809,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 5U))));
        vcdp->chgBit(c+10817,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_push_unqual));
        vcdp->chgQuad(c+10825,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out),54);
        vcdp->chgBit(c+10841,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__reading));
        vcdp->chgArray(c+10849,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out),153);
        vcdp->chgBit(c+10889,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__reading));
        vcdp->chgBus(c+10897,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x16U))),4);
        vcdp->chgBus(c+10905,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x12U))),4);
        vcdp->chgBus(c+10913,((0xffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                          >> 2U))),16);
        __Vtemp1420[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                                         >> 0xaU));
        __Vtemp1420[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                                         >> 0xaU));
        __Vtemp1420[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                                         >> 0xaU));
        __Vtemp1420[3U] = (0xffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                         << 0x16U) 
                                        | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                                           >> 0xaU)));
        vcdp->chgArray(c+10921,(__Vtemp1420),120);
        __Vtemp1421[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U] 
                                         >> 0xaU));
        __Vtemp1421[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                                         >> 0xaU));
        __Vtemp1421[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                                         >> 0xaU));
        __Vtemp1421[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                                         >> 0xaU));
        vcdp->chgArray(c+10953,(__Vtemp1421),128);
        vcdp->chgQuad(c+10985,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[0U]))))),42);
        vcdp->chgBit(c+11001,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty));
        vcdp->chgBit(c+11009,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__pop_qual));
        vcdp->chgBus(c+11017,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                        >> 0x16U) & 
                                       VL_NEGATE_I((IData)(
                                                           (1U 
                                                            & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty)))))))),4);
        vcdp->chgArray(c+11025,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out),314);
        vcdp->chgBit(c+11105,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__reading));
        vcdp->chgBus(c+11113,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->chgBus(c+11121,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgArray(c+11129,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__s0_1_c0__in),242);
        vcdp->chgBus(c+11193,((0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                               [0U])),4);
        vcdp->chgBit(c+11201,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                              [0U]));
        vcdp->chgBus(c+11209,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writeword_st1
                              [0U]),32);
        __Vtemp1422[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writedata_st1
            [0U][0U];
        __Vtemp1422[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writedata_st1
            [0U][1U];
        __Vtemp1422[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writedata_st1
            [0U][2U];
        __Vtemp1422[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writedata_st1
            [0U][3U];
        vcdp->chgArray(c+11217,(__Vtemp1422),128);
        vcdp->chgBus(c+11249,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                              [0U]),2);
        vcdp->chgBit(c+11257,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c[0]));
        vcdp->chgBit(c+11265,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c[0]));
        vcdp->chgBus(c+11273,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c[0]),16);
        vcdp->chgBus(c+11281,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c[0]),21);
        vcdp->chgArray(c+11289,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c[0]),128);
        vcdp->chgBit(c+11321,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid) 
                                     >> (0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                         [0U])))));
        vcdp->chgBit(c+11329,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty) 
                                     >> (0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                         [0U])))));
        vcdp->chgBus(c+11337,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                [0U])]),16);
        vcdp->chgBus(c+11345,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag
                              [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                [0U])]),21);
        __Vtemp1423[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
              [0U])][0U];
        __Vtemp1423[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
              [0U])][1U];
        __Vtemp1423[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
              [0U])][2U];
        __Vtemp1423[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
              [0U])][3U];
        vcdp->chgArray(c+11353,(__Vtemp1423),128);
        vcdp->chgBit(c+11385,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                              [0U]));
        vcdp->chgBit(c+11393,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c
                              [0U]));
        vcdp->chgBus(c+11401,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable),16);
        vcdp->chgArray(c+11409,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__data_write),128);
        vcdp->chgBit(c+11441,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__snoop_hit_no_pending));
        vcdp->chgBit(c+11449,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match));
        vcdp->chgBit(c+11457,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill));
        vcdp->chgBus(c+11465,((0x1fffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                            [0U] >> 4U))),21);
        vcdp->chgBus(c+11473,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__we),16);
        vcdp->chgBit(c+11481,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__should_write));
        vcdp->chgBit(c+11489,((((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                                 [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1
                                         [0U])) & (~ 
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                                   [0U])) 
                               & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                                  [0U]))));
        vcdp->chgBit(c+11497,(((((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                                  [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1
                                          [0U])) & 
                                 vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                 [0U]) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                                          [0U])) & 
                               (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match)))));
        vcdp->chgBit(c+11505,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss));
        vcdp->chgBit(c+11513,((((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                  & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1
                                     [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                                               [0U])) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                                [0U]) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss)))));
        vcdp->chgBit(c+11521,(((0U == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+11529,(((1U == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+11537,(((2U == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+11545,(((3U == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+11553,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable))));
        vcdp->chgArray(c+11561,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT____Vcellinp__s0_1_c0__in),167);
        vcdp->chgArray(c+11609,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__st_1e_2__in),316);
        vcdp->chgBus(c+11689,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready),16);
        vcdp->chgBus(c+11697,((0xffffU & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                                           & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                          << (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr)))),16);
        vcdp->chgBus(c+11705,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match),16);
        vcdp->chgBit(c+11713,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push));
        vcdp->chgBit(c+11721,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_pop));
        vcdp->chgBit(c+11729,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_is_mrvq))));
        vcdp->chgBit(c+11737,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__increment_head));
        vcdp->chgBit(c+11745,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready))));
        vcdp->chgBit(c+11753,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_init_ready_state_st2))));
        vcdp->chgArray(c+11761,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__cwb_queue__data_in),76);
        vcdp->chgArray(c+11785,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out),76);
        vcdp->chgBit(c+11809,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__writing));
        vcdp->chgArray(c+11817,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__dwb_queue__data_in),199);
        vcdp->chgArray(c+11873,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out),199);
        vcdp->chgBit(c+11929,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__writing));
        vcdp->chgBit(c+11937,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snrq_pop));
        vcdp->chgBus(c+11945,((0x1ffffffU & (IData)(
                                                    (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                     >> 0x1dU)))),25);
        vcdp->chgBit(c+11953,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                             >> 0x1cU)))));
        vcdp->chgBus(c+11961,((0xfffffffU & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out))),28);
        vcdp->chgBit(c+11969,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfpq_pop));
        vcdp->chgBus(c+11977,((0x1ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])),25);
        __Vtemp1424[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U];
        __Vtemp1424[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U];
        __Vtemp1424[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U];
        __Vtemp1424[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U];
        vcdp->chgArray(c+11985,(__Vtemp1424),128);
        vcdp->chgBit(c+12017,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__reqq_pop));
        vcdp->chgBit(c+12025,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request)))));
        vcdp->chgBit(c+12033,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request));
        vcdp->chgBus(c+12041,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index),2);
        vcdp->chgBit(c+12049,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__reqq_req_rw_st0));
        vcdp->chgBus(c+12057,((0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                       >> (0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 2U))))),4);
        vcdp->chgBus(c+12065,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__reqq_req_addr_st0),30);
        vcdp->chgBus(c+12073,((((0U == (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                 << 5U)))
                                 ? 0U : (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                         ((IData)(1U) 
                                          + (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index)))] 
                                         << ((IData)(0x20U) 
                                             - (0x1fU 
                                                & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 5U))))) 
                               | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                  (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))] 
                                  >> (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                               << 5U))))),32);
        vcdp->chgBit(c+12081,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__mrvq_pop));
        vcdp->chgBit(c+12089,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible));
        vcdp->chgBit(c+12097,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__mrvq_rw_st0));
        vcdp->chgBit(c+12105,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match))));
        vcdp->chgBit(c+12113,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__miss_add_because_miss) 
                               & (((0x1ffffffU & ((
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                   << 6U) 
                                                  | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                     >> 0x1aU))) 
                                   == vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__addr_st1
                                   [0U]) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                               >> 0x14U))))));
        vcdp->chgBit(c+12121,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__force_request_miss_st1e));
        vcdp->chgBit(c+12129,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__recover_mrvq_state_st2));
        vcdp->chgBit(c+12137,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__mrvq_push_stall));
        vcdp->chgBit(c+12145,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwbq_push_stall));
        vcdp->chgBit(c+12153,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwbq_push_stall));
        vcdp->chgBit(c+12161,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dram_fill_req_stall));
        vcdp->chgBit(c+12169,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__stall_bank_pipe));
        vcdp->chgBit(c+12177,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_fill_in_pipe));
        vcdp->chgBit(c+12185,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_fill_st1[0]));
        vcdp->chgBit(c+12193,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__going_to_write_st1[0]));
        vcdp->chgBit(c+12201,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfpq_pop_unqual));
        vcdp->chgBit(c+12209,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__reqq_pop_unqual));
        vcdp->chgBit(c+12217,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snrq_pop_unqual));
        vcdp->chgBit(c+12225,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfpq_pop) 
                                 | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__mrvq_pop)) 
                                | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__reqq_pop)) 
                               | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snrq_pop))));
        vcdp->chgBus(c+12233,((0x1ffffffU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                              ? (0x1ffffffU 
                                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])
                                              : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                                  ? 
                                                 ((0x18fU 
                                                   >= 
                                                   (0x1ffU 
                                                    & ((IData)(0x19U) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                   ? 
                                                  (0x1ffffffU 
                                                   & (((0U 
                                                        == 
                                                        (0x1fU 
                                                         & ((IData)(0x19U) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                        ? 0U
                                                        : 
                                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                        ((IData)(1U) 
                                                         + 
                                                         (0xfU 
                                                          & (((IData)(0x19U) 
                                                              * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                             >> 5U)))] 
                                                        << 
                                                        ((IData)(0x20U) 
                                                         - 
                                                         (0x1fU 
                                                          & ((IData)(0x19U) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                      | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                         (0xfU 
                                                          & (((IData)(0x19U) 
                                                              * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                             >> 5U))] 
                                                         >> 
                                                         (0x1fU 
                                                          & ((IData)(0x19U) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                                   : 0U)
                                                  : 
                                                 ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__reqq_pop_unqual)
                                                   ? 
                                                  (0x1ffffffU 
                                                   & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__reqq_req_addr_st0 
                                                      >> 5U))
                                                   : 
                                                  ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snrq_pop_unqual)
                                                    ? 
                                                   (0x1ffffffU 
                                                    & (IData)(
                                                              (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                               >> 0x1dU)))
                                                    : 0U)))))),25);
        vcdp->chgBus(c+12241,((3U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__reqq_pop_unqual)
                                      ? (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__reqq_req_addr_st0)
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                          ? (3U & (
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                    << 0x1eU) 
                                                   | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                      >> 2U)))
                                          : 0U)))),2);
        vcdp->chgBus(c+12249,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                ? ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                    << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                >> 0x15U))
                                : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__reqq_pop_unqual)
                                    ? (((0U == (0x1fU 
                                                & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 5U)))
                                         ? 0U : (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                                 ((IData)(1U) 
                                                  + 
                                                  (3U 
                                                   & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index)))] 
                                                 << 
                                                 ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                      << 5U))))) 
                                       | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                          (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))] 
                                          >> (0x1fU 
                                              & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                 << 5U))))
                                    : 0U))),32);
        __Vtemp1429[0U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U]
                            : 0x39U);
        __Vtemp1429[1U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U]
                            : 0U);
        __Vtemp1429[2U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U]
                            : 0U);
        __Vtemp1429[3U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U]
                            : 0U);
        vcdp->chgArray(c+12257,(__Vtemp1429),128);
        vcdp->chgQuad(c+12289,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                 ? ((VL_ULL(0x1ffffffffff80) 
                                     & (((QData)((IData)(
                                                         vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                         << 0x3eU) 
                                        | (((QData)((IData)(
                                                            vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                            [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U])) 
                                            << 0x1eU) 
                                           | (VL_ULL(0x3fffffffffffff80) 
                                              & ((QData)((IData)(
                                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                 [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])) 
                                                 >> 2U))))) 
                                    | (QData)((IData)(
                                                      (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__mrvq_rw_st0) 
                                                        << 6U) 
                                                       | ((0x3cU 
                                                           & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                               [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                               << 0x1eU) 
                                                              | (0x3ffffffcU 
                                                                 & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                                    >> 2U)))) 
                                                          | (3U 
                                                             & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                 [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                                                 << 0xdU) 
                                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                                   >> 0x13U))))))))
                                 : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__reqq_pop_unqual)
                                     ? ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag 
                                         << 7U) | (QData)((IData)(
                                                                  (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__reqq_req_rw_st0) 
                                                                    << 6U) 
                                                                   | ((0x3cU 
                                                                       & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                                                           >> 
                                                                           (0xfU 
                                                                            & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                                               << 2U))) 
                                                                          << 2U)) 
                                                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))))))
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? ((QData)((IData)(
                                                            (0xfffffffU 
                                                             & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out)))) 
                                            << 7U) : VL_ULL(0))))),49);
        vcdp->chgBit(c+12305,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible) 
                                         & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__mrvq_rw_st0))
                                         ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__reqq_pop_unqual) 
                                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__reqq_req_rw_st0))
                                                  ? 1U
                                                  : 0U)))));
        vcdp->chgBit(c+12313,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                               [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                               >> 1U))
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? 1U : 0U)))));
        vcdp->chgBit(c+12321,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? (1U & (IData)(
                                                          (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                           >> 0x1cU)))
                                          : 0U)))));
        vcdp->chgBit(c+12329,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__valid_st1[0]));
        vcdp->chgBus(c+12337,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__addr_st1[0]),25);
        vcdp->chgBus(c+12345,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__wsel_st1[0]),2);
        vcdp->chgBus(c+12353,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__writeword_st1[0]),32);
        vcdp->chgQuad(c+12361,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__inst_meta_st1[0]),49);
        vcdp->chgArray(c+12377,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__writedata_st1[0]),128);
        vcdp->chgBit(c+12409,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_snp_st1[0]));
        vcdp->chgBit(c+12417,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_invalidate_st1[0]));
        vcdp->chgBit(c+12425,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_mrvq_st1[0]));
        __Vtemp1435[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp1435[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp1435[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp1435[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgBus(c+12433,((((0U == (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__wsel_st1
                                                 [0U] 
                                                 << 5U)))
                                 ? 0U : (__Vtemp1435[
                                         ((IData)(1U) 
                                          + (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__wsel_st1
                                             [0U]))] 
                                         << ((IData)(0x20U) 
                                             - (0x1fU 
                                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__wsel_st1
                                                   [0U] 
                                                   << 5U))))) 
                               | (__Vtemp1435[(3U & 
                                               vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__wsel_st1
                                               [0U])] 
                                  >> (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__wsel_st1
                                               [0U] 
                                               << 5U))))),32);
        __Vtemp1436[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp1436[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp1436[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp1436[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgArray(c+12441,(__Vtemp1436),128);
        vcdp->chgBus(c+12473,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c
                              [0U]),21);
        vcdp->chgBit(c+12481,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__miss_st1e));
        vcdp->chgBit(c+12489,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dirty_st1e));
        vcdp->chgBus(c+12497,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c
                              [0U]),16);
        vcdp->chgQuad(c+12505,((VL_ULL(0x3ffffffffff) 
                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__inst_meta_st1
                                   [0U] >> 7U))),42);
        vcdp->chgBus(c+12521,((3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__inst_meta_st1
                                            [0U]))),2);
        vcdp->chgBit(c+12529,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__inst_meta_st1
                                             [0U] >> 6U)))));
        vcdp->chgBus(c+12537,((0xfU & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__inst_meta_st1
                                               [0U] 
                                               >> 2U)))),4);
        vcdp->chgBit(c+12545,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dirty_st1e))));
        vcdp->chgBit(c+12553,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_snp_st1
                              [0U]));
        vcdp->chgBit(c+12561,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_invalidate_st1
                              [0U]));
        vcdp->chgBit(c+12569,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_to_mrvq_st1e));
        vcdp->chgBit(c+12577,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_to_mrvq_st1e) 
                               | ((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                    & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_snp_st1
                                       [0U])) & (~ 
                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_fill_st1
                                                 [0U])) 
                                  & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__valid_st1
                                  [0U]))));
        vcdp->chgBit(c+12585,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__miss_add_because_miss));
        vcdp->chgBit(c+12593,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__valid_st1
                              [0U]));
        vcdp->chgBit(c+12601,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_mrvq_st1
                              [0U]));
        vcdp->chgBit(c+12609,((((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__valid_st1
                                 [0U] & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_mrvq_st1
                                 [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__recover_mrvq_state_st2)) 
                               & ((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 6U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x1aU))) 
                                  == vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__addr_st1
                                  [0U]))));
        vcdp->chgBus(c+12617,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__addr_st1
                              [0U]),25);
        vcdp->chgBit(c+12625,((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__valid_st1
                               [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_fill_st1
                                       [0U]))));
        vcdp->chgBit(c+12633,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__mrvq_init_ready_state_st2));
        vcdp->chgBit(c+12641,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__miss_add_unqual) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfpq_pop_unqual)) 
                               & ((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 6U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x1aU))) 
                                  == (0x1ffffffU & 
                                      vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])))));
        vcdp->chgBit(c+12649,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__miss_add_unqual) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_fill_st1
                                [0U]) & ((0x1ffffffU 
                                          & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 6U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x1aU))) 
                                         == vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__addr_st1
                                         [0U]))));
        vcdp->chgBit(c+12657,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__miss_add_unqual));
        vcdp->chgBit(c+12665,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__miss_add));
        vcdp->chgBit(c+12673,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__miss_add_is_mrvq));
        vcdp->chgBit(c+12681,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwbq_push_unqual) 
                                 & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                                & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                      >> 6U))) & (~ 
                                                  (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwbq_push_stall) 
                                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                                   | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+12689,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwbq_push_unqual));
        vcdp->chgBit(c+12697,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU)))))));
        vcdp->chgBit(c+12705,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwbq_push_unqual) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwbq_push_stall) 
                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+12713,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwbq_is_dwb_in));
        vcdp->chgBit(c+12721,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwbq_is_snp_in));
        vcdp->chgBit(c+12729,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 6U))));
        vcdp->chgBit(c+12737,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 5U))));
        vcdp->chgBit(c+12745,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwbq_push_unqual));
        vcdp->chgQuad(c+12753,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out),54);
        vcdp->chgBit(c+12769,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__reading));
        vcdp->chgArray(c+12777,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out),153);
        vcdp->chgBit(c+12817,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__reading));
        vcdp->chgBus(c+12825,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x16U))),4);
        vcdp->chgBus(c+12833,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x12U))),4);
        vcdp->chgBus(c+12841,((0xffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                          >> 2U))),16);
        __Vtemp1439[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                                         >> 0xaU));
        __Vtemp1439[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                                         >> 0xaU));
        __Vtemp1439[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                                         >> 0xaU));
        __Vtemp1439[3U] = (0xffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                         << 0x16U) 
                                        | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                                           >> 0xaU)));
        vcdp->chgArray(c+12849,(__Vtemp1439),120);
        __Vtemp1440[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U] 
                                         >> 0xaU));
        __Vtemp1440[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                                         >> 0xaU));
        __Vtemp1440[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                                         >> 0xaU));
        __Vtemp1440[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                                         >> 0xaU));
        vcdp->chgArray(c+12881,(__Vtemp1440),128);
        vcdp->chgQuad(c+12913,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[0U]))))),42);
        vcdp->chgBit(c+12929,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty));
        vcdp->chgBit(c+12937,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__pop_qual));
        vcdp->chgBus(c+12945,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                        >> 0x16U) & 
                                       VL_NEGATE_I((IData)(
                                                           (1U 
                                                            & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty)))))))),4);
        vcdp->chgArray(c+12953,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out),314);
        vcdp->chgBit(c+13033,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__reading));
        vcdp->chgBus(c+13041,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->chgBus(c+13049,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgArray(c+13057,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellinp__s0_1_c0__in),242);
        vcdp->chgBus(c+13121,((0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__addr_st1
                               [0U])),4);
        vcdp->chgBit(c+13129,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_fill_st1
                              [0U]));
        vcdp->chgBus(c+13137,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__writeword_st1
                              [0U]),32);
        __Vtemp1441[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__writedata_st1
            [0U][0U];
        __Vtemp1441[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__writedata_st1
            [0U][1U];
        __Vtemp1441[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__writedata_st1
            [0U][2U];
        __Vtemp1441[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__writedata_st1
            [0U][3U];
        vcdp->chgArray(c+13145,(__Vtemp1441),128);
        vcdp->chgBus(c+13177,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__wsel_st1
                              [0U]),2);
        vcdp->chgBit(c+13185,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c[0]));
        vcdp->chgBit(c+13193,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c[0]));
        vcdp->chgBus(c+13201,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c[0]),16);
        vcdp->chgBus(c+13209,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c[0]),21);
        vcdp->chgArray(c+13217,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c[0]),128);
        vcdp->chgBit(c+13249,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid) 
                                     >> (0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__addr_st1
                                         [0U])))));
        vcdp->chgBit(c+13257,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty) 
                                     >> (0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__addr_st1
                                         [0U])))));
        vcdp->chgBus(c+13265,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__addr_st1
                                [0U])]),16);
        vcdp->chgBus(c+13273,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag
                              [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__addr_st1
                                [0U])]),21);
        __Vtemp1442[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__addr_st1
              [0U])][0U];
        __Vtemp1442[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__addr_st1
              [0U])][1U];
        __Vtemp1442[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__addr_st1
              [0U])][2U];
        __Vtemp1442[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__addr_st1
              [0U])][3U];
        vcdp->chgArray(c+13281,(__Vtemp1442),128);
        vcdp->chgBit(c+13313,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                              [0U]));
        vcdp->chgBit(c+13321,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c
                              [0U]));
        vcdp->chgBus(c+13329,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable),16);
        vcdp->chgArray(c+13337,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__data_write),128);
        vcdp->chgBit(c+13369,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__snoop_hit_no_pending));
        vcdp->chgBit(c+13377,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match));
        vcdp->chgBit(c+13385,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill));
        vcdp->chgBus(c+13393,((0x1fffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__addr_st1
                                            [0U] >> 4U))),21);
        vcdp->chgBus(c+13401,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__we),16);
        vcdp->chgBit(c+13409,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__should_write));
        vcdp->chgBit(c+13417,((((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__valid_st1
                                 [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_snp_st1
                                         [0U])) & (~ 
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                                   [0U])) 
                               & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_fill_st1
                                  [0U]))));
        vcdp->chgBit(c+13425,(((((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__valid_st1
                                  [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_snp_st1
                                          [0U])) & 
                                 vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                 [0U]) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_fill_st1
                                          [0U])) & 
                               (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match)))));
        vcdp->chgBit(c+13433,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss));
        vcdp->chgBit(c+13441,((((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                  & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_snp_st1
                                     [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__is_fill_st1
                                               [0U])) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__valid_st1
                                [0U]) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss)))));
        vcdp->chgBit(c+13449,(((0U == vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+13457,(((1U == vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+13465,(((2U == vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+13473,(((3U == vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+13481,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable))));
        vcdp->chgArray(c+13489,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT____Vcellinp__s0_1_c0__in),167);
        vcdp->chgArray(c+13537,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellinp__st_1e_2__in),316);
        vcdp->chgBus(c+13617,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready),16);
        vcdp->chgBus(c+13625,((0xffffU & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                                           & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                          << (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr)))),16);
        vcdp->chgBus(c+13633,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match),16);
        vcdp->chgBit(c+13641,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push));
        vcdp->chgBit(c+13649,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_pop));
        vcdp->chgBit(c+13657,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__miss_add) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__miss_add_is_mrvq))));
        vcdp->chgBit(c+13665,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__increment_head));
        vcdp->chgBit(c+13673,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready))));
        vcdp->chgBit(c+13681,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__mrvq_init_ready_state_st2))));
        vcdp->chgArray(c+13689,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellinp__cwb_queue__data_in),76);
        vcdp->chgArray(c+13713,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out),76);
        vcdp->chgBit(c+13737,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__writing));
        vcdp->chgArray(c+13745,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellinp__dwb_queue__data_in),199);
        vcdp->chgArray(c+13801,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out),199);
        vcdp->chgBit(c+13857,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwb_queue__DOT__writing));
        vcdp->chgBit(c+13865,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snrq_pop));
        vcdp->chgBus(c+13873,((0x1ffffffU & (IData)(
                                                    (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                     >> 0x1dU)))),25);
        vcdp->chgBit(c+13881,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                             >> 0x1cU)))));
        vcdp->chgBus(c+13889,((0xfffffffU & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out))),28);
        vcdp->chgBit(c+13897,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfpq_pop));
        vcdp->chgBus(c+13905,((0x1ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])),25);
        __Vtemp1443[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U];
        __Vtemp1443[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U];
        __Vtemp1443[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U];
        __Vtemp1443[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U];
        vcdp->chgArray(c+13913,(__Vtemp1443),128);
        vcdp->chgBit(c+13945,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__reqq_pop));
        vcdp->chgBit(c+13953,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request)))));
        vcdp->chgBit(c+13961,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request));
        vcdp->chgBus(c+13969,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index),2);
        vcdp->chgBit(c+13977,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__reqq_req_rw_st0));
        vcdp->chgBus(c+13985,((0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                       >> (0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 2U))))),4);
        vcdp->chgBus(c+13993,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__reqq_req_addr_st0),30);
        vcdp->chgBus(c+14001,((((0U == (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                 << 5U)))
                                 ? 0U : (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                         ((IData)(1U) 
                                          + (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index)))] 
                                         << ((IData)(0x20U) 
                                             - (0x1fU 
                                                & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 5U))))) 
                               | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                  (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))] 
                                  >> (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                               << 5U))))),32);
        vcdp->chgBit(c+14009,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__mrvq_pop));
        vcdp->chgBit(c+14017,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible));
        vcdp->chgBit(c+14025,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__mrvq_rw_st0));
        vcdp->chgBit(c+14033,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match))));
        vcdp->chgBit(c+14041,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__miss_add_because_miss) 
                               & (((0x1ffffffU & ((
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                   << 6U) 
                                                  | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                     >> 0x1aU))) 
                                   == vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__addr_st1
                                   [0U]) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                               >> 0x14U))))));
        vcdp->chgBit(c+14049,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__force_request_miss_st1e));
        vcdp->chgBit(c+14057,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__recover_mrvq_state_st2));
        vcdp->chgBit(c+14065,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__mrvq_push_stall));
        vcdp->chgBit(c+14073,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwbq_push_stall));
        vcdp->chgBit(c+14081,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwbq_push_stall));
        vcdp->chgBit(c+14089,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dram_fill_req_stall));
        vcdp->chgBit(c+14097,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__stall_bank_pipe));
        vcdp->chgBit(c+14105,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_fill_in_pipe));
        vcdp->chgBit(c+14113,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_fill_st1[0]));
        vcdp->chgBit(c+14121,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__going_to_write_st1[0]));
        vcdp->chgBit(c+14129,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfpq_pop_unqual));
        vcdp->chgBit(c+14137,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__reqq_pop_unqual));
        vcdp->chgBit(c+14145,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snrq_pop_unqual));
        vcdp->chgBit(c+14153,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfpq_pop) 
                                 | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__mrvq_pop)) 
                                | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__reqq_pop)) 
                               | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snrq_pop))));
        vcdp->chgBus(c+14161,((0x1ffffffU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                              ? (0x1ffffffU 
                                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])
                                              : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                                  ? 
                                                 ((0x18fU 
                                                   >= 
                                                   (0x1ffU 
                                                    & ((IData)(0x19U) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                   ? 
                                                  (0x1ffffffU 
                                                   & (((0U 
                                                        == 
                                                        (0x1fU 
                                                         & ((IData)(0x19U) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                        ? 0U
                                                        : 
                                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                        ((IData)(1U) 
                                                         + 
                                                         (0xfU 
                                                          & (((IData)(0x19U) 
                                                              * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                             >> 5U)))] 
                                                        << 
                                                        ((IData)(0x20U) 
                                                         - 
                                                         (0x1fU 
                                                          & ((IData)(0x19U) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                      | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                         (0xfU 
                                                          & (((IData)(0x19U) 
                                                              * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                             >> 5U))] 
                                                         >> 
                                                         (0x1fU 
                                                          & ((IData)(0x19U) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                                   : 0U)
                                                  : 
                                                 ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__reqq_pop_unqual)
                                                   ? 
                                                  (0x1ffffffU 
                                                   & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__reqq_req_addr_st0 
                                                      >> 5U))
                                                   : 
                                                  ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snrq_pop_unqual)
                                                    ? 
                                                   (0x1ffffffU 
                                                    & (IData)(
                                                              (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                               >> 0x1dU)))
                                                    : 0U)))))),25);
        vcdp->chgBus(c+14169,((3U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__reqq_pop_unqual)
                                      ? (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__reqq_req_addr_st0)
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                          ? (3U & (
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                    << 0x1eU) 
                                                   | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                      >> 2U)))
                                          : 0U)))),2);
        vcdp->chgBus(c+14177,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                ? ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                    << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                >> 0x15U))
                                : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__reqq_pop_unqual)
                                    ? (((0U == (0x1fU 
                                                & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 5U)))
                                         ? 0U : (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                                 ((IData)(1U) 
                                                  + 
                                                  (3U 
                                                   & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index)))] 
                                                 << 
                                                 ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                      << 5U))))) 
                                       | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                          (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))] 
                                          >> (0x1fU 
                                              & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                 << 5U))))
                                    : 0U))),32);
        __Vtemp1448[0U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U]
                            : 0x39U);
        __Vtemp1448[1U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U]
                            : 0U);
        __Vtemp1448[2U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U]
                            : 0U);
        __Vtemp1448[3U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U]
                            : 0U);
        vcdp->chgArray(c+14185,(__Vtemp1448),128);
        vcdp->chgQuad(c+14217,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                 ? ((VL_ULL(0x1ffffffffff80) 
                                     & (((QData)((IData)(
                                                         vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                         << 0x3eU) 
                                        | (((QData)((IData)(
                                                            vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                            [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U])) 
                                            << 0x1eU) 
                                           | (VL_ULL(0x3fffffffffffff80) 
                                              & ((QData)((IData)(
                                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                 [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])) 
                                                 >> 2U))))) 
                                    | (QData)((IData)(
                                                      (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__mrvq_rw_st0) 
                                                        << 6U) 
                                                       | ((0x3cU 
                                                           & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                               [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                               << 0x1eU) 
                                                              | (0x3ffffffcU 
                                                                 & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                                    >> 2U)))) 
                                                          | (3U 
                                                             & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                 [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                                                 << 0xdU) 
                                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                                   >> 0x13U))))))))
                                 : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__reqq_pop_unqual)
                                     ? ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag 
                                         << 7U) | (QData)((IData)(
                                                                  (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__reqq_req_rw_st0) 
                                                                    << 6U) 
                                                                   | ((0x3cU 
                                                                       & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                                                           >> 
                                                                           (0xfU 
                                                                            & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                                               << 2U))) 
                                                                          << 2U)) 
                                                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))))))
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? ((QData)((IData)(
                                                            (0xfffffffU 
                                                             & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out)))) 
                                            << 7U) : VL_ULL(0))))),49);
        vcdp->chgBit(c+14233,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible) 
                                         & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__mrvq_rw_st0))
                                         ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__reqq_pop_unqual) 
                                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__reqq_req_rw_st0))
                                                  ? 1U
                                                  : 0U)))));
        vcdp->chgBit(c+14241,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                               [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                               >> 1U))
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? 1U : 0U)))));
        vcdp->chgBit(c+14249,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? (1U & (IData)(
                                                          (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                           >> 0x1cU)))
                                          : 0U)))));
        vcdp->chgBit(c+14257,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__valid_st1[0]));
        vcdp->chgBus(c+14265,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__addr_st1[0]),25);
        vcdp->chgBus(c+14273,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__wsel_st1[0]),2);
        vcdp->chgBus(c+14281,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__writeword_st1[0]),32);
        vcdp->chgQuad(c+14289,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__inst_meta_st1[0]),49);
        vcdp->chgArray(c+14305,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__writedata_st1[0]),128);
        vcdp->chgBit(c+14337,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_snp_st1[0]));
        vcdp->chgBit(c+14345,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_invalidate_st1[0]));
        vcdp->chgBit(c+14353,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_mrvq_st1[0]));
        __Vtemp1454[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp1454[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp1454[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp1454[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgBus(c+14361,((((0U == (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__wsel_st1
                                                 [0U] 
                                                 << 5U)))
                                 ? 0U : (__Vtemp1454[
                                         ((IData)(1U) 
                                          + (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__wsel_st1
                                             [0U]))] 
                                         << ((IData)(0x20U) 
                                             - (0x1fU 
                                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__wsel_st1
                                                   [0U] 
                                                   << 5U))))) 
                               | (__Vtemp1454[(3U & 
                                               vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__wsel_st1
                                               [0U])] 
                                  >> (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__wsel_st1
                                               [0U] 
                                               << 5U))))),32);
        __Vtemp1455[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp1455[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp1455[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp1455[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgArray(c+14369,(__Vtemp1455),128);
        vcdp->chgBus(c+14401,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c
                              [0U]),21);
        vcdp->chgBit(c+14409,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__miss_st1e));
        vcdp->chgBit(c+14417,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dirty_st1e));
        vcdp->chgBus(c+14425,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c
                              [0U]),16);
        vcdp->chgQuad(c+14433,((VL_ULL(0x3ffffffffff) 
                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__inst_meta_st1
                                   [0U] >> 7U))),42);
        vcdp->chgBus(c+14449,((3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__inst_meta_st1
                                            [0U]))),2);
        vcdp->chgBit(c+14457,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__inst_meta_st1
                                             [0U] >> 6U)))));
        vcdp->chgBus(c+14465,((0xfU & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__inst_meta_st1
                                               [0U] 
                                               >> 2U)))),4);
        vcdp->chgBit(c+14473,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dirty_st1e))));
        vcdp->chgBit(c+14481,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_snp_st1
                              [0U]));
        vcdp->chgBit(c+14489,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_invalidate_st1
                              [0U]));
        vcdp->chgBit(c+14497,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_to_mrvq_st1e));
        vcdp->chgBit(c+14505,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_to_mrvq_st1e) 
                               | ((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                    & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_snp_st1
                                       [0U])) & (~ 
                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_fill_st1
                                                 [0U])) 
                                  & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__valid_st1
                                  [0U]))));
        vcdp->chgBit(c+14513,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__miss_add_because_miss));
        vcdp->chgBit(c+14521,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__valid_st1
                              [0U]));
        vcdp->chgBit(c+14529,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_mrvq_st1
                              [0U]));
        vcdp->chgBit(c+14537,((((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__valid_st1
                                 [0U] & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_mrvq_st1
                                 [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__recover_mrvq_state_st2)) 
                               & ((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 6U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x1aU))) 
                                  == vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__addr_st1
                                  [0U]))));
        vcdp->chgBus(c+14545,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__addr_st1
                              [0U]),25);
        vcdp->chgBit(c+14553,((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__valid_st1
                               [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_fill_st1
                                       [0U]))));
        vcdp->chgBit(c+14561,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__mrvq_init_ready_state_st2));
        vcdp->chgBit(c+14569,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__miss_add_unqual) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfpq_pop_unqual)) 
                               & ((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 6U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x1aU))) 
                                  == (0x1ffffffU & 
                                      vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])))));
        vcdp->chgBit(c+14577,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__miss_add_unqual) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_fill_st1
                                [0U]) & ((0x1ffffffU 
                                          & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 6U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x1aU))) 
                                         == vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__addr_st1
                                         [0U]))));
        vcdp->chgBit(c+14585,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__miss_add_unqual));
        vcdp->chgBit(c+14593,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__miss_add));
        vcdp->chgBit(c+14601,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__miss_add_is_mrvq));
        vcdp->chgBit(c+14609,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwbq_push_unqual) 
                                 & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                                & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                      >> 6U))) & (~ 
                                                  (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwbq_push_stall) 
                                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                                   | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+14617,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwbq_push_unqual));
        vcdp->chgBit(c+14625,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU)))))));
        vcdp->chgBit(c+14633,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwbq_push_unqual) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwbq_push_stall) 
                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+14641,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwbq_is_dwb_in));
        vcdp->chgBit(c+14649,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwbq_is_snp_in));
        vcdp->chgBit(c+14657,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 6U))));
        vcdp->chgBit(c+14665,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 5U))));
        vcdp->chgBit(c+14673,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwbq_push_unqual));
        vcdp->chgQuad(c+14681,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out),54);
        vcdp->chgBit(c+14697,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__reading));
        vcdp->chgArray(c+14705,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out),153);
        vcdp->chgBit(c+14745,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__reading));
        vcdp->chgBus(c+14753,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x16U))),4);
        vcdp->chgBus(c+14761,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x12U))),4);
        vcdp->chgBus(c+14769,((0xffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                          >> 2U))),16);
        __Vtemp1458[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                                         >> 0xaU));
        __Vtemp1458[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                                         >> 0xaU));
        __Vtemp1458[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                                         >> 0xaU));
        __Vtemp1458[3U] = (0xffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                         << 0x16U) 
                                        | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                                           >> 0xaU)));
        vcdp->chgArray(c+14777,(__Vtemp1458),120);
        __Vtemp1459[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U] 
                                         >> 0xaU));
        __Vtemp1459[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                                         >> 0xaU));
        __Vtemp1459[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                                         >> 0xaU));
        __Vtemp1459[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                                         >> 0xaU));
        vcdp->chgArray(c+14809,(__Vtemp1459),128);
        vcdp->chgQuad(c+14841,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[0U]))))),42);
        vcdp->chgBit(c+14857,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty));
        vcdp->chgBit(c+14865,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__pop_qual));
        vcdp->chgBus(c+14873,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                        >> 0x16U) & 
                                       VL_NEGATE_I((IData)(
                                                           (1U 
                                                            & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty)))))))),4);
        vcdp->chgArray(c+14881,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out),314);
        vcdp->chgBit(c+14961,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__reading));
        vcdp->chgBus(c+14969,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->chgBus(c+14977,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgArray(c+14985,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellinp__s0_1_c0__in),242);
        vcdp->chgBus(c+15049,((0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__addr_st1
                               [0U])),4);
        vcdp->chgBit(c+15057,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_fill_st1
                              [0U]));
        vcdp->chgBus(c+15065,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__writeword_st1
                              [0U]),32);
        __Vtemp1460[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__writedata_st1
            [0U][0U];
        __Vtemp1460[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__writedata_st1
            [0U][1U];
        __Vtemp1460[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__writedata_st1
            [0U][2U];
        __Vtemp1460[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__writedata_st1
            [0U][3U];
        vcdp->chgArray(c+15073,(__Vtemp1460),128);
        vcdp->chgBus(c+15105,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__wsel_st1
                              [0U]),2);
        vcdp->chgBit(c+15113,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c[0]));
        vcdp->chgBit(c+15121,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c[0]));
        vcdp->chgBus(c+15129,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c[0]),16);
        vcdp->chgBus(c+15137,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c[0]),21);
        vcdp->chgArray(c+15145,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c[0]),128);
        vcdp->chgBit(c+15177,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid) 
                                     >> (0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__addr_st1
                                         [0U])))));
        vcdp->chgBit(c+15185,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty) 
                                     >> (0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__addr_st1
                                         [0U])))));
        vcdp->chgBus(c+15193,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__addr_st1
                                [0U])]),16);
        vcdp->chgBus(c+15201,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag
                              [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__addr_st1
                                [0U])]),21);
        __Vtemp1461[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__addr_st1
              [0U])][0U];
        __Vtemp1461[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__addr_st1
              [0U])][1U];
        __Vtemp1461[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__addr_st1
              [0U])][2U];
        __Vtemp1461[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__addr_st1
              [0U])][3U];
        vcdp->chgArray(c+15209,(__Vtemp1461),128);
        vcdp->chgBit(c+15241,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                              [0U]));
        vcdp->chgBit(c+15249,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c
                              [0U]));
        vcdp->chgBus(c+15257,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable),16);
        vcdp->chgArray(c+15265,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__data_write),128);
        vcdp->chgBit(c+15297,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__snoop_hit_no_pending));
        vcdp->chgBit(c+15305,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match));
        vcdp->chgBit(c+15313,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill));
        vcdp->chgBus(c+15321,((0x1fffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__addr_st1
                                            [0U] >> 4U))),21);
        vcdp->chgBus(c+15329,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__we),16);
        vcdp->chgBit(c+15337,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__should_write));
        vcdp->chgBit(c+15345,((((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__valid_st1
                                 [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_snp_st1
                                         [0U])) & (~ 
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                                   [0U])) 
                               & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_fill_st1
                                  [0U]))));
        vcdp->chgBit(c+15353,(((((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__valid_st1
                                  [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_snp_st1
                                          [0U])) & 
                                 vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                 [0U]) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_fill_st1
                                          [0U])) & 
                               (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match)))));
        vcdp->chgBit(c+15361,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss));
        vcdp->chgBit(c+15369,((((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                  & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_snp_st1
                                     [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__is_fill_st1
                                               [0U])) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__valid_st1
                                [0U]) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss)))));
        vcdp->chgBit(c+15377,(((0U == vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+15385,(((1U == vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+15393,(((2U == vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+15401,(((3U == vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+15409,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable))));
        vcdp->chgArray(c+15417,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT____Vcellinp__s0_1_c0__in),167);
        vcdp->chgArray(c+15465,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellinp__st_1e_2__in),316);
        vcdp->chgBus(c+15545,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready),16);
        vcdp->chgBus(c+15553,((0xffffU & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                                           & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                          << (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr)))),16);
        vcdp->chgBus(c+15561,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match),16);
        vcdp->chgBit(c+15569,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push));
        vcdp->chgBit(c+15577,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_pop));
        vcdp->chgBit(c+15585,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__miss_add) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__miss_add_is_mrvq))));
        vcdp->chgBit(c+15593,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__increment_head));
        vcdp->chgBit(c+15601,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready))));
        vcdp->chgBit(c+15609,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__mrvq_init_ready_state_st2))));
        vcdp->chgArray(c+15617,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellinp__cwb_queue__data_in),76);
        vcdp->chgArray(c+15641,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out),76);
        vcdp->chgBit(c+15665,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__writing));
        vcdp->chgArray(c+15673,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellinp__dwb_queue__data_in),199);
        vcdp->chgArray(c+15729,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out),199);
        vcdp->chgBit(c+15785,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwb_queue__DOT__writing));
        vcdp->chgBit(c+15793,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snrq_pop));
        vcdp->chgBus(c+15801,((0x1ffffffU & (IData)(
                                                    (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                     >> 0x1dU)))),25);
        vcdp->chgBit(c+15809,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                             >> 0x1cU)))));
        vcdp->chgBus(c+15817,((0xfffffffU & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out))),28);
        vcdp->chgBit(c+15825,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfpq_pop));
        vcdp->chgBus(c+15833,((0x1ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])),25);
        __Vtemp1462[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U];
        __Vtemp1462[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U];
        __Vtemp1462[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U];
        __Vtemp1462[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U];
        vcdp->chgArray(c+15841,(__Vtemp1462),128);
        vcdp->chgBit(c+15873,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__reqq_pop));
        vcdp->chgBit(c+15881,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request)))));
        vcdp->chgBit(c+15889,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request));
        vcdp->chgBus(c+15897,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index),2);
        vcdp->chgBit(c+15905,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__reqq_req_rw_st0));
        vcdp->chgBus(c+15913,((0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                       >> (0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 2U))))),4);
        vcdp->chgBus(c+15921,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__reqq_req_addr_st0),30);
        vcdp->chgBus(c+15929,((((0U == (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                 << 5U)))
                                 ? 0U : (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                         ((IData)(1U) 
                                          + (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index)))] 
                                         << ((IData)(0x20U) 
                                             - (0x1fU 
                                                & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 5U))))) 
                               | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                  (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))] 
                                  >> (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                               << 5U))))),32);
        vcdp->chgBit(c+15937,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__mrvq_pop));
        vcdp->chgBit(c+15945,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible));
        vcdp->chgBit(c+15953,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__mrvq_rw_st0));
        vcdp->chgBit(c+15961,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match))));
        vcdp->chgBit(c+15969,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__miss_add_because_miss) 
                               & (((0x1ffffffU & ((
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                   << 6U) 
                                                  | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                     >> 0x1aU))) 
                                   == vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__addr_st1
                                   [0U]) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                               >> 0x14U))))));
        vcdp->chgBit(c+15977,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__force_request_miss_st1e));
        vcdp->chgBit(c+15985,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__recover_mrvq_state_st2));
        vcdp->chgBit(c+15993,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__mrvq_push_stall));
        vcdp->chgBit(c+16001,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwbq_push_stall));
        vcdp->chgBit(c+16009,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwbq_push_stall));
        vcdp->chgBit(c+16017,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dram_fill_req_stall));
        vcdp->chgBit(c+16025,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__stall_bank_pipe));
        vcdp->chgBit(c+16033,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_fill_in_pipe));
        vcdp->chgBit(c+16041,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_fill_st1[0]));
        vcdp->chgBit(c+16049,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__going_to_write_st1[0]));
        vcdp->chgBit(c+16057,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfpq_pop_unqual));
        vcdp->chgBit(c+16065,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__reqq_pop_unqual));
        vcdp->chgBit(c+16073,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snrq_pop_unqual));
        vcdp->chgBit(c+16081,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfpq_pop) 
                                 | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__mrvq_pop)) 
                                | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__reqq_pop)) 
                               | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snrq_pop))));
        vcdp->chgBus(c+16089,((0x1ffffffU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                              ? (0x1ffffffU 
                                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])
                                              : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                                  ? 
                                                 ((0x18fU 
                                                   >= 
                                                   (0x1ffU 
                                                    & ((IData)(0x19U) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                   ? 
                                                  (0x1ffffffU 
                                                   & (((0U 
                                                        == 
                                                        (0x1fU 
                                                         & ((IData)(0x19U) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                        ? 0U
                                                        : 
                                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                        ((IData)(1U) 
                                                         + 
                                                         (0xfU 
                                                          & (((IData)(0x19U) 
                                                              * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                             >> 5U)))] 
                                                        << 
                                                        ((IData)(0x20U) 
                                                         - 
                                                         (0x1fU 
                                                          & ((IData)(0x19U) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                      | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                         (0xfU 
                                                          & (((IData)(0x19U) 
                                                              * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                             >> 5U))] 
                                                         >> 
                                                         (0x1fU 
                                                          & ((IData)(0x19U) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                                   : 0U)
                                                  : 
                                                 ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__reqq_pop_unqual)
                                                   ? 
                                                  (0x1ffffffU 
                                                   & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__reqq_req_addr_st0 
                                                      >> 5U))
                                                   : 
                                                  ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snrq_pop_unqual)
                                                    ? 
                                                   (0x1ffffffU 
                                                    & (IData)(
                                                              (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                               >> 0x1dU)))
                                                    : 0U)))))),25);
        vcdp->chgBus(c+16097,((3U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__reqq_pop_unqual)
                                      ? (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__reqq_req_addr_st0)
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                          ? (3U & (
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                    << 0x1eU) 
                                                   | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                      >> 2U)))
                                          : 0U)))),2);
        vcdp->chgBus(c+16105,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                ? ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                    << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                >> 0x15U))
                                : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__reqq_pop_unqual)
                                    ? (((0U == (0x1fU 
                                                & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 5U)))
                                         ? 0U : (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                                 ((IData)(1U) 
                                                  + 
                                                  (3U 
                                                   & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index)))] 
                                                 << 
                                                 ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                      << 5U))))) 
                                       | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                          (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))] 
                                          >> (0x1fU 
                                              & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                 << 5U))))
                                    : 0U))),32);
        __Vtemp1467[0U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U]
                            : 0x39U);
        __Vtemp1467[1U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U]
                            : 0U);
        __Vtemp1467[2U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U]
                            : 0U);
        __Vtemp1467[3U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U]
                            : 0U);
        vcdp->chgArray(c+16113,(__Vtemp1467),128);
        vcdp->chgQuad(c+16145,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                 ? ((VL_ULL(0x1ffffffffff80) 
                                     & (((QData)((IData)(
                                                         vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                         << 0x3eU) 
                                        | (((QData)((IData)(
                                                            vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                            [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U])) 
                                            << 0x1eU) 
                                           | (VL_ULL(0x3fffffffffffff80) 
                                              & ((QData)((IData)(
                                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                 [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])) 
                                                 >> 2U))))) 
                                    | (QData)((IData)(
                                                      (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__mrvq_rw_st0) 
                                                        << 6U) 
                                                       | ((0x3cU 
                                                           & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                               [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                               << 0x1eU) 
                                                              | (0x3ffffffcU 
                                                                 & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                                    >> 2U)))) 
                                                          | (3U 
                                                             & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                 [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                                                 << 0xdU) 
                                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                                   >> 0x13U))))))))
                                 : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__reqq_pop_unqual)
                                     ? ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag 
                                         << 7U) | (QData)((IData)(
                                                                  (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__reqq_req_rw_st0) 
                                                                    << 6U) 
                                                                   | ((0x3cU 
                                                                       & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                                                           >> 
                                                                           (0xfU 
                                                                            & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                                               << 2U))) 
                                                                          << 2U)) 
                                                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))))))
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? ((QData)((IData)(
                                                            (0xfffffffU 
                                                             & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out)))) 
                                            << 7U) : VL_ULL(0))))),49);
        vcdp->chgBit(c+16161,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible) 
                                         & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__mrvq_rw_st0))
                                         ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__reqq_pop_unqual) 
                                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__reqq_req_rw_st0))
                                                  ? 1U
                                                  : 0U)))));
        vcdp->chgBit(c+16169,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                               [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                               >> 1U))
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? 1U : 0U)))));
        vcdp->chgBit(c+16177,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? (1U & (IData)(
                                                          (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                           >> 0x1cU)))
                                          : 0U)))));
        vcdp->chgBit(c+16185,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__valid_st1[0]));
        vcdp->chgBus(c+16193,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__addr_st1[0]),25);
        vcdp->chgBus(c+16201,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__wsel_st1[0]),2);
        vcdp->chgBus(c+16209,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__writeword_st1[0]),32);
        vcdp->chgQuad(c+16217,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__inst_meta_st1[0]),49);
        vcdp->chgArray(c+16233,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__writedata_st1[0]),128);
        vcdp->chgBit(c+16265,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_snp_st1[0]));
        vcdp->chgBit(c+16273,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_invalidate_st1[0]));
        vcdp->chgBit(c+16281,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_mrvq_st1[0]));
        __Vtemp1473[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp1473[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp1473[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp1473[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgBus(c+16289,((((0U == (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__wsel_st1
                                                 [0U] 
                                                 << 5U)))
                                 ? 0U : (__Vtemp1473[
                                         ((IData)(1U) 
                                          + (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__wsel_st1
                                             [0U]))] 
                                         << ((IData)(0x20U) 
                                             - (0x1fU 
                                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__wsel_st1
                                                   [0U] 
                                                   << 5U))))) 
                               | (__Vtemp1473[(3U & 
                                               vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__wsel_st1
                                               [0U])] 
                                  >> (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__wsel_st1
                                               [0U] 
                                               << 5U))))),32);
        __Vtemp1474[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp1474[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp1474[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp1474[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgArray(c+16297,(__Vtemp1474),128);
        vcdp->chgBus(c+16329,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c
                              [0U]),21);
        vcdp->chgBit(c+16337,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__miss_st1e));
        vcdp->chgBit(c+16345,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dirty_st1e));
        vcdp->chgBus(c+16353,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c
                              [0U]),16);
        vcdp->chgQuad(c+16361,((VL_ULL(0x3ffffffffff) 
                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__inst_meta_st1
                                   [0U] >> 7U))),42);
        vcdp->chgBus(c+16377,((3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__inst_meta_st1
                                            [0U]))),2);
        vcdp->chgBit(c+16385,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__inst_meta_st1
                                             [0U] >> 6U)))));
        vcdp->chgBus(c+16393,((0xfU & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__inst_meta_st1
                                               [0U] 
                                               >> 2U)))),4);
        vcdp->chgBit(c+16401,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dirty_st1e))));
        vcdp->chgBit(c+16409,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_snp_st1
                              [0U]));
        vcdp->chgBit(c+16417,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_invalidate_st1
                              [0U]));
        vcdp->chgBit(c+16425,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_to_mrvq_st1e));
        vcdp->chgBit(c+16433,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_to_mrvq_st1e) 
                               | ((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                    & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_snp_st1
                                       [0U])) & (~ 
                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_fill_st1
                                                 [0U])) 
                                  & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__valid_st1
                                  [0U]))));
        vcdp->chgBit(c+16441,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__miss_add_because_miss));
        vcdp->chgBit(c+16449,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__valid_st1
                              [0U]));
        vcdp->chgBit(c+16457,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_mrvq_st1
                              [0U]));
        vcdp->chgBit(c+16465,((((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__valid_st1
                                 [0U] & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_mrvq_st1
                                 [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__recover_mrvq_state_st2)) 
                               & ((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 6U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x1aU))) 
                                  == vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__addr_st1
                                  [0U]))));
        vcdp->chgBus(c+16473,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__addr_st1
                              [0U]),25);
        vcdp->chgBit(c+16481,((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__valid_st1
                               [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_fill_st1
                                       [0U]))));
        vcdp->chgBit(c+16489,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__mrvq_init_ready_state_st2));
        vcdp->chgBit(c+16497,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__miss_add_unqual) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfpq_pop_unqual)) 
                               & ((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 6U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x1aU))) 
                                  == (0x1ffffffU & 
                                      vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])))));
        vcdp->chgBit(c+16505,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__miss_add_unqual) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_fill_st1
                                [0U]) & ((0x1ffffffU 
                                          & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 6U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x1aU))) 
                                         == vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__addr_st1
                                         [0U]))));
        vcdp->chgBit(c+16513,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__miss_add_unqual));
        vcdp->chgBit(c+16521,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__miss_add));
        vcdp->chgBit(c+16529,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__miss_add_is_mrvq));
        vcdp->chgBit(c+16537,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwbq_push_unqual) 
                                 & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                                & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                      >> 6U))) & (~ 
                                                  (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwbq_push_stall) 
                                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                                   | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+16545,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwbq_push_unqual));
        vcdp->chgBit(c+16553,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU)))))));
        vcdp->chgBit(c+16561,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwbq_push_unqual) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwbq_push_stall) 
                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+16569,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwbq_is_dwb_in));
        vcdp->chgBit(c+16577,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwbq_is_snp_in));
        vcdp->chgBit(c+16585,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 6U))));
        vcdp->chgBit(c+16593,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 5U))));
        vcdp->chgBit(c+16601,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwbq_push_unqual));
        vcdp->chgQuad(c+16609,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out),54);
        vcdp->chgBit(c+16625,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__reading));
        vcdp->chgArray(c+16633,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out),153);
        vcdp->chgBit(c+16673,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__reading));
        vcdp->chgBus(c+16681,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x16U))),4);
        vcdp->chgBus(c+16689,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x12U))),4);
        vcdp->chgBus(c+16697,((0xffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                          >> 2U))),16);
        __Vtemp1477[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                                         >> 0xaU));
        __Vtemp1477[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                                         >> 0xaU));
        __Vtemp1477[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                                         >> 0xaU));
        __Vtemp1477[3U] = (0xffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                         << 0x16U) 
                                        | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                                           >> 0xaU)));
        vcdp->chgArray(c+16705,(__Vtemp1477),120);
        __Vtemp1478[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U] 
                                         >> 0xaU));
        __Vtemp1478[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                                         >> 0xaU));
        __Vtemp1478[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                                         >> 0xaU));
        __Vtemp1478[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                                         >> 0xaU));
        vcdp->chgArray(c+16737,(__Vtemp1478),128);
        vcdp->chgQuad(c+16769,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[0U]))))),42);
        vcdp->chgBit(c+16785,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty));
        vcdp->chgBit(c+16793,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__pop_qual));
        vcdp->chgBus(c+16801,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                        >> 0x16U) & 
                                       VL_NEGATE_I((IData)(
                                                           (1U 
                                                            & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty)))))))),4);
        vcdp->chgArray(c+16809,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out),314);
        vcdp->chgBit(c+16889,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__reading));
        vcdp->chgBus(c+16897,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->chgBus(c+16905,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgArray(c+16913,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellinp__s0_1_c0__in),242);
        vcdp->chgBus(c+16977,((0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__addr_st1
                               [0U])),4);
        vcdp->chgBit(c+16985,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_fill_st1
                              [0U]));
        vcdp->chgBus(c+16993,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__writeword_st1
                              [0U]),32);
        __Vtemp1479[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__writedata_st1
            [0U][0U];
        __Vtemp1479[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__writedata_st1
            [0U][1U];
        __Vtemp1479[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__writedata_st1
            [0U][2U];
        __Vtemp1479[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__writedata_st1
            [0U][3U];
        vcdp->chgArray(c+17001,(__Vtemp1479),128);
        vcdp->chgBus(c+17033,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__wsel_st1
                              [0U]),2);
        vcdp->chgBit(c+17041,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c[0]));
        vcdp->chgBit(c+17049,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c[0]));
        vcdp->chgBus(c+17057,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c[0]),16);
        vcdp->chgBus(c+17065,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c[0]),21);
        vcdp->chgArray(c+17073,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c[0]),128);
        vcdp->chgBit(c+17105,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid) 
                                     >> (0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__addr_st1
                                         [0U])))));
        vcdp->chgBit(c+17113,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty) 
                                     >> (0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__addr_st1
                                         [0U])))));
        vcdp->chgBus(c+17121,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__addr_st1
                                [0U])]),16);
        vcdp->chgBus(c+17129,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag
                              [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__addr_st1
                                [0U])]),21);
        __Vtemp1480[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__addr_st1
              [0U])][0U];
        __Vtemp1480[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__addr_st1
              [0U])][1U];
        __Vtemp1480[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__addr_st1
              [0U])][2U];
        __Vtemp1480[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__addr_st1
              [0U])][3U];
        vcdp->chgArray(c+17137,(__Vtemp1480),128);
        vcdp->chgBit(c+17169,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                              [0U]));
        vcdp->chgBit(c+17177,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c
                              [0U]));
        vcdp->chgBus(c+17185,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable),16);
        vcdp->chgArray(c+17193,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__data_write),128);
        vcdp->chgBit(c+17225,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__snoop_hit_no_pending));
        vcdp->chgBit(c+17233,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match));
        vcdp->chgBit(c+17241,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill));
        vcdp->chgBus(c+17249,((0x1fffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__addr_st1
                                            [0U] >> 4U))),21);
        vcdp->chgBus(c+17257,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__we),16);
        vcdp->chgBit(c+17265,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__should_write));
        vcdp->chgBit(c+17273,((((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__valid_st1
                                 [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_snp_st1
                                         [0U])) & (~ 
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                                   [0U])) 
                               & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_fill_st1
                                  [0U]))));
        vcdp->chgBit(c+17281,(((((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__valid_st1
                                  [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_snp_st1
                                          [0U])) & 
                                 vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                 [0U]) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_fill_st1
                                          [0U])) & 
                               (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match)))));
        vcdp->chgBit(c+17289,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss));
        vcdp->chgBit(c+17297,((((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                  & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_snp_st1
                                     [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__is_fill_st1
                                               [0U])) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__valid_st1
                                [0U]) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss)))));
        vcdp->chgBit(c+17305,(((0U == vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+17313,(((1U == vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+17321,(((2U == vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+17329,(((3U == vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+17337,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable))));
        vcdp->chgArray(c+17345,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT____Vcellinp__s0_1_c0__in),167);
        vcdp->chgArray(c+17393,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellinp__st_1e_2__in),316);
        vcdp->chgBus(c+17473,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready),16);
        vcdp->chgBus(c+17481,((0xffffU & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                                           & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                          << (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr)))),16);
        vcdp->chgBus(c+17489,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match),16);
        vcdp->chgBit(c+17497,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push));
        vcdp->chgBit(c+17505,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_pop));
        vcdp->chgBit(c+17513,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__miss_add) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__miss_add_is_mrvq))));
        vcdp->chgBit(c+17521,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__increment_head));
        vcdp->chgBit(c+17529,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready))));
        vcdp->chgBit(c+17537,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__mrvq_init_ready_state_st2))));
        vcdp->chgArray(c+17545,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellinp__cwb_queue__data_in),76);
        vcdp->chgArray(c+17569,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out),76);
        vcdp->chgBit(c+17593,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__writing));
        vcdp->chgArray(c+17601,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellinp__dwb_queue__data_in),199);
        vcdp->chgArray(c+17657,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out),199);
        vcdp->chgBit(c+17713,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwb_queue__DOT__writing));
        vcdp->chgBit(c+17721,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snrq_pop));
        vcdp->chgBus(c+17729,((0x1ffffffU & (IData)(
                                                    (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                     >> 0x1dU)))),25);
        vcdp->chgBit(c+17737,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                             >> 0x1cU)))));
        vcdp->chgBus(c+17745,((0xfffffffU & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out))),28);
        vcdp->chgBit(c+17753,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfpq_pop));
        vcdp->chgBus(c+17761,((0x1ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])),25);
        __Vtemp1481[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U];
        __Vtemp1481[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U];
        __Vtemp1481[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U];
        __Vtemp1481[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U];
        vcdp->chgArray(c+17769,(__Vtemp1481),128);
        vcdp->chgBit(c+17801,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__reqq_pop));
        vcdp->chgBit(c+17809,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request)))));
        vcdp->chgBit(c+17817,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request));
        vcdp->chgBus(c+17825,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index),2);
        vcdp->chgBit(c+17833,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__reqq_req_rw_st0));
        vcdp->chgBus(c+17841,((0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                       >> (0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 2U))))),4);
        vcdp->chgBus(c+17849,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__reqq_req_addr_st0),30);
        vcdp->chgBus(c+17857,((((0U == (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                 << 5U)))
                                 ? 0U : (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                         ((IData)(1U) 
                                          + (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index)))] 
                                         << ((IData)(0x20U) 
                                             - (0x1fU 
                                                & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 5U))))) 
                               | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                  (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))] 
                                  >> (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                               << 5U))))),32);
        vcdp->chgBit(c+17865,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__mrvq_pop));
        vcdp->chgBit(c+17873,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible));
        vcdp->chgBit(c+17881,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__mrvq_rw_st0));
        vcdp->chgBit(c+17889,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match))));
        vcdp->chgBit(c+17897,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__miss_add_because_miss) 
                               & (((0x1ffffffU & ((
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                   << 6U) 
                                                  | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                     >> 0x1aU))) 
                                   == vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__addr_st1
                                   [0U]) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                               >> 0x14U))))));
        vcdp->chgBit(c+17905,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__force_request_miss_st1e));
        vcdp->chgBit(c+17913,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__recover_mrvq_state_st2));
        vcdp->chgBit(c+17921,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__mrvq_push_stall));
        vcdp->chgBit(c+17929,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwbq_push_stall));
        vcdp->chgBit(c+17937,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwbq_push_stall));
        vcdp->chgBit(c+17945,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dram_fill_req_stall));
        vcdp->chgBit(c+17953,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__stall_bank_pipe));
        vcdp->chgBit(c+17961,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_fill_in_pipe));
        vcdp->chgBit(c+17969,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_fill_st1[0]));
        vcdp->chgBit(c+17977,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__going_to_write_st1[0]));
        vcdp->chgBit(c+17985,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfpq_pop_unqual));
        vcdp->chgBit(c+17993,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__reqq_pop_unqual));
        vcdp->chgBit(c+18001,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snrq_pop_unqual));
        vcdp->chgBit(c+18009,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfpq_pop) 
                                 | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__mrvq_pop)) 
                                | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__reqq_pop)) 
                               | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snrq_pop))));
        vcdp->chgBus(c+18017,((0x1ffffffU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                              ? (0x1ffffffU 
                                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])
                                              : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                                  ? 
                                                 ((0x18fU 
                                                   >= 
                                                   (0x1ffU 
                                                    & ((IData)(0x19U) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                   ? 
                                                  (0x1ffffffU 
                                                   & (((0U 
                                                        == 
                                                        (0x1fU 
                                                         & ((IData)(0x19U) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                        ? 0U
                                                        : 
                                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                        ((IData)(1U) 
                                                         + 
                                                         (0xfU 
                                                          & (((IData)(0x19U) 
                                                              * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                             >> 5U)))] 
                                                        << 
                                                        ((IData)(0x20U) 
                                                         - 
                                                         (0x1fU 
                                                          & ((IData)(0x19U) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                      | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                         (0xfU 
                                                          & (((IData)(0x19U) 
                                                              * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                             >> 5U))] 
                                                         >> 
                                                         (0x1fU 
                                                          & ((IData)(0x19U) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                                   : 0U)
                                                  : 
                                                 ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__reqq_pop_unqual)
                                                   ? 
                                                  (0x1ffffffU 
                                                   & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__reqq_req_addr_st0 
                                                      >> 5U))
                                                   : 
                                                  ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snrq_pop_unqual)
                                                    ? 
                                                   (0x1ffffffU 
                                                    & (IData)(
                                                              (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                               >> 0x1dU)))
                                                    : 0U)))))),25);
        vcdp->chgBus(c+18025,((3U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__reqq_pop_unqual)
                                      ? (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__reqq_req_addr_st0)
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                          ? (3U & (
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                    << 0x1eU) 
                                                   | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                      >> 2U)))
                                          : 0U)))),2);
        vcdp->chgBus(c+18033,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                ? ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                    << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                >> 0x15U))
                                : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__reqq_pop_unqual)
                                    ? (((0U == (0x1fU 
                                                & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 5U)))
                                         ? 0U : (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                                 ((IData)(1U) 
                                                  + 
                                                  (3U 
                                                   & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index)))] 
                                                 << 
                                                 ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                      << 5U))))) 
                                       | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata[
                                          (3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))] 
                                          >> (0x1fU 
                                              & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                 << 5U))))
                                    : 0U))),32);
        __Vtemp1486[0U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U]
                            : 0x39U);
        __Vtemp1486[1U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U]
                            : 0U);
        __Vtemp1486[2U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U]
                            : 0U);
        __Vtemp1486[3U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfpq_pop_unqual)
                            ? vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U]
                            : 0U);
        vcdp->chgArray(c+18041,(__Vtemp1486),128);
        vcdp->chgQuad(c+18073,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                 ? ((VL_ULL(0x1ffffffffff80) 
                                     & (((QData)((IData)(
                                                         vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                         << 0x3eU) 
                                        | (((QData)((IData)(
                                                            vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                            [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U])) 
                                            << 0x1eU) 
                                           | (VL_ULL(0x3fffffffffffff80) 
                                              & ((QData)((IData)(
                                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                 [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])) 
                                                 >> 2U))))) 
                                    | (QData)((IData)(
                                                      (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__mrvq_rw_st0) 
                                                        << 6U) 
                                                       | ((0x3cU 
                                                           & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                               [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                               << 0x1eU) 
                                                              | (0x3ffffffcU 
                                                                 & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                                    >> 2U)))) 
                                                          | (3U 
                                                             & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                 [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                                                 << 0xdU) 
                                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                                   >> 0x13U))))))))
                                 : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__reqq_pop_unqual)
                                     ? ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag 
                                         << 7U) | (QData)((IData)(
                                                                  (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__reqq_req_rw_st0) 
                                                                    << 6U) 
                                                                   | ((0x3cU 
                                                                       & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                                                           >> 
                                                                           (0xfU 
                                                                            & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                                               << 2U))) 
                                                                          << 2U)) 
                                                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index))))))
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? ((QData)((IData)(
                                                            (0xfffffffU 
                                                             & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out)))) 
                                            << 7U) : VL_ULL(0))))),49);
        vcdp->chgBit(c+18089,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible) 
                                         & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__mrvq_rw_st0))
                                         ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__reqq_pop_unqual) 
                                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__reqq_req_rw_st0))
                                                  ? 1U
                                                  : 0U)))));
        vcdp->chgBit(c+18097,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                               [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                               >> 1U))
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? 1U : 0U)))));
        vcdp->chgBit(c+18105,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? (1U & (IData)(
                                                          (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                           >> 0x1cU)))
                                          : 0U)))));
        vcdp->chgBit(c+18113,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__valid_st1[0]));
        vcdp->chgBus(c+18121,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__addr_st1[0]),25);
        vcdp->chgBus(c+18129,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__wsel_st1[0]),2);
        vcdp->chgBus(c+18137,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__writeword_st1[0]),32);
        vcdp->chgQuad(c+18145,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__inst_meta_st1[0]),49);
        vcdp->chgArray(c+18161,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__writedata_st1[0]),128);
        vcdp->chgBit(c+18193,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_snp_st1[0]));
        vcdp->chgBit(c+18201,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_invalidate_st1[0]));
        vcdp->chgBit(c+18209,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_mrvq_st1[0]));
        __Vtemp1492[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp1492[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp1492[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp1492[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgBus(c+18217,((((0U == (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__wsel_st1
                                                 [0U] 
                                                 << 5U)))
                                 ? 0U : (__Vtemp1492[
                                         ((IData)(1U) 
                                          + (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__wsel_st1
                                             [0U]))] 
                                         << ((IData)(0x20U) 
                                             - (0x1fU 
                                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__wsel_st1
                                                   [0U] 
                                                   << 5U))))) 
                               | (__Vtemp1492[(3U & 
                                               vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__wsel_st1
                                               [0U])] 
                                  >> (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__wsel_st1
                                               [0U] 
                                               << 5U))))),32);
        __Vtemp1493[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp1493[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp1493[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp1493[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgArray(c+18225,(__Vtemp1493),128);
        vcdp->chgBus(c+18257,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c
                              [0U]),21);
        vcdp->chgBit(c+18265,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__miss_st1e));
        vcdp->chgBit(c+18273,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dirty_st1e));
        vcdp->chgBus(c+18281,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c
                              [0U]),16);
        vcdp->chgQuad(c+18289,((VL_ULL(0x3ffffffffff) 
                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__inst_meta_st1
                                   [0U] >> 7U))),42);
        vcdp->chgBus(c+18305,((3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__inst_meta_st1
                                            [0U]))),2);
        vcdp->chgBit(c+18313,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__inst_meta_st1
                                             [0U] >> 6U)))));
        vcdp->chgBus(c+18321,((0xfU & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__inst_meta_st1
                                               [0U] 
                                               >> 2U)))),4);
        vcdp->chgBit(c+18329,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dirty_st1e))));
        vcdp->chgBit(c+18337,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_snp_st1
                              [0U]));
        vcdp->chgBit(c+18345,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_invalidate_st1
                              [0U]));
        vcdp->chgBit(c+18353,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_to_mrvq_st1e));
        vcdp->chgBit(c+18361,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_to_mrvq_st1e) 
                               | ((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                    & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_snp_st1
                                       [0U])) & (~ 
                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_fill_st1
                                                 [0U])) 
                                  & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__valid_st1
                                  [0U]))));
        vcdp->chgBit(c+18369,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__miss_add_because_miss));
        vcdp->chgBit(c+18377,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__valid_st1
                              [0U]));
        vcdp->chgBit(c+18385,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_mrvq_st1
                              [0U]));
        vcdp->chgBit(c+18393,((((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__valid_st1
                                 [0U] & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_mrvq_st1
                                 [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__recover_mrvq_state_st2)) 
                               & ((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 6U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x1aU))) 
                                  == vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__addr_st1
                                  [0U]))));
        vcdp->chgBus(c+18401,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__addr_st1
                              [0U]),25);
        vcdp->chgBit(c+18409,((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__valid_st1
                               [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_fill_st1
                                       [0U]))));
        vcdp->chgBit(c+18417,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__mrvq_init_ready_state_st2));
        vcdp->chgBit(c+18425,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__miss_add_unqual) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfpq_pop_unqual)) 
                               & ((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 6U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x1aU))) 
                                  == (0x1ffffffU & 
                                      vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])))));
        vcdp->chgBit(c+18433,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__miss_add_unqual) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_fill_st1
                                [0U]) & ((0x1ffffffU 
                                          & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 6U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x1aU))) 
                                         == vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__addr_st1
                                         [0U]))));
        vcdp->chgBit(c+18441,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__miss_add_unqual));
        vcdp->chgBit(c+18449,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__miss_add));
        vcdp->chgBit(c+18457,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__miss_add_is_mrvq));
        vcdp->chgBit(c+18465,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwbq_push_unqual) 
                                 & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                                & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                      >> 6U))) & (~ 
                                                  (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwbq_push_stall) 
                                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                                   | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+18473,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwbq_push_unqual));
        vcdp->chgBit(c+18481,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU)))))));
        vcdp->chgBit(c+18489,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwbq_push_unqual) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwbq_push_stall) 
                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+18497,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwbq_is_dwb_in));
        vcdp->chgBit(c+18505,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwbq_is_snp_in));
        vcdp->chgBit(c+18513,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 6U))));
        vcdp->chgBit(c+18521,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 5U))));
        vcdp->chgBit(c+18529,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwbq_push_unqual));
        vcdp->chgQuad(c+18537,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out),54);
        vcdp->chgBit(c+18553,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__reading));
        vcdp->chgArray(c+18561,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out),153);
        vcdp->chgBit(c+18601,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__reading));
        vcdp->chgBus(c+18609,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x16U))),4);
        vcdp->chgBus(c+18617,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x12U))),4);
        vcdp->chgBus(c+18625,((0xffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                          >> 2U))),16);
        __Vtemp1496[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                                         >> 0xaU));
        __Vtemp1496[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                                         >> 0xaU));
        __Vtemp1496[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                                         >> 0xaU));
        __Vtemp1496[3U] = (0xffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                         << 0x16U) 
                                        | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                                           >> 0xaU)));
        vcdp->chgArray(c+18633,(__Vtemp1496),120);
        __Vtemp1497[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U] 
                                         >> 0xaU));
        __Vtemp1497[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                                         >> 0xaU));
        __Vtemp1497[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                                         >> 0xaU));
        __Vtemp1497[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                            << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                                         >> 0xaU));
        vcdp->chgArray(c+18665,(__Vtemp1497),128);
        vcdp->chgQuad(c+18697,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[0U]))))),42);
        vcdp->chgBit(c+18713,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty));
        vcdp->chgBit(c+18721,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__pop_qual));
        vcdp->chgBus(c+18729,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                        >> 0x16U) & 
                                       VL_NEGATE_I((IData)(
                                                           (1U 
                                                            & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty)))))))),4);
        vcdp->chgArray(c+18737,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out),314);
        vcdp->chgBit(c+18817,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__reading));
        vcdp->chgBus(c+18825,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->chgBus(c+18833,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgArray(c+18841,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellinp__s0_1_c0__in),242);
        vcdp->chgBus(c+18905,((0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__addr_st1
                               [0U])),4);
        vcdp->chgBit(c+18913,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_fill_st1
                              [0U]));
        vcdp->chgBus(c+18921,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__writeword_st1
                              [0U]),32);
        __Vtemp1498[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__writedata_st1
            [0U][0U];
        __Vtemp1498[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__writedata_st1
            [0U][1U];
        __Vtemp1498[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__writedata_st1
            [0U][2U];
        __Vtemp1498[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__writedata_st1
            [0U][3U];
        vcdp->chgArray(c+18929,(__Vtemp1498),128);
        vcdp->chgBus(c+18961,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__wsel_st1
                              [0U]),2);
        vcdp->chgBit(c+18969,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c[0]));
        vcdp->chgBit(c+18977,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c[0]));
        vcdp->chgBus(c+18985,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c[0]),16);
        vcdp->chgBus(c+18993,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c[0]),21);
        vcdp->chgArray(c+19001,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c[0]),128);
        vcdp->chgBit(c+19033,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid) 
                                     >> (0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__addr_st1
                                         [0U])))));
        vcdp->chgBit(c+19041,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty) 
                                     >> (0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__addr_st1
                                         [0U])))));
        vcdp->chgBus(c+19049,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__addr_st1
                                [0U])]),16);
        vcdp->chgBus(c+19057,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag
                              [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__addr_st1
                                [0U])]),21);
        __Vtemp1499[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__addr_st1
              [0U])][0U];
        __Vtemp1499[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__addr_st1
              [0U])][1U];
        __Vtemp1499[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__addr_st1
              [0U])][2U];
        __Vtemp1499[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0xfU & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__addr_st1
              [0U])][3U];
        vcdp->chgArray(c+19065,(__Vtemp1499),128);
        vcdp->chgBit(c+19097,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                              [0U]));
        vcdp->chgBit(c+19105,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c
                              [0U]));
        vcdp->chgBus(c+19113,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable),16);
        vcdp->chgArray(c+19121,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__data_write),128);
        vcdp->chgBit(c+19153,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__snoop_hit_no_pending));
        vcdp->chgBit(c+19161,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match));
        vcdp->chgBit(c+19169,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill));
        vcdp->chgBus(c+19177,((0x1fffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__addr_st1
                                            [0U] >> 4U))),21);
        vcdp->chgBus(c+19185,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__we),16);
        vcdp->chgBit(c+19193,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__should_write));
        vcdp->chgBit(c+19201,((((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__valid_st1
                                 [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_snp_st1
                                         [0U])) & (~ 
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                                   [0U])) 
                               & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_fill_st1
                                  [0U]))));
        vcdp->chgBit(c+19209,(((((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__valid_st1
                                  [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_snp_st1
                                          [0U])) & 
                                 vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                 [0U]) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_fill_st1
                                          [0U])) & 
                               (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match)))));
        vcdp->chgBit(c+19217,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss));
        vcdp->chgBit(c+19225,((((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                  & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_snp_st1
                                     [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__is_fill_st1
                                               [0U])) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__valid_st1
                                [0U]) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss)))));
        vcdp->chgBit(c+19233,(((0U == vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+19241,(((1U == vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+19249,(((2U == vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+19257,(((3U == vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+19265,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable))));
        vcdp->chgArray(c+19273,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT____Vcellinp__s0_1_c0__in),167);
        vcdp->chgArray(c+19321,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellinp__st_1e_2__in),316);
        vcdp->chgBus(c+19401,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready),16);
        vcdp->chgBus(c+19409,((0xffffU & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                                           & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                          << (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr)))),16);
        vcdp->chgBus(c+19417,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match),16);
        vcdp->chgBit(c+19425,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push));
        vcdp->chgBit(c+19433,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_pop));
        vcdp->chgBit(c+19441,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__miss_add) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__miss_add_is_mrvq))));
        vcdp->chgBit(c+19449,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__increment_head));
        vcdp->chgBit(c+19457,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready))));
        vcdp->chgBit(c+19465,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__mrvq_init_ready_state_st2))));
        vcdp->chgArray(c+19473,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellinp__cwb_queue__data_in),76);
        vcdp->chgArray(c+19497,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out),76);
        vcdp->chgBit(c+19521,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__writing));
        vcdp->chgArray(c+19529,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellinp__dwb_queue__data_in),199);
        vcdp->chgArray(c+19585,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out),199);
        vcdp->chgBit(c+19641,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwb_queue__DOT__writing));
    }
}

void VVX_cache::traceChgThis__5(VVX_cache__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code) {
    VVX_cache* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    int c = code;
    if (0 && vcdp && c) {}  // Prevent unused
    // Variables
    WData/*127:0*/ __Vtemp1508[4];
    WData/*127:0*/ __Vtemp1509[4];
    WData/*127:0*/ __Vtemp1510[4];
    WData/*127:0*/ __Vtemp1511[4];
    WData/*127:0*/ __Vtemp1512[4];
    WData/*127:0*/ __Vtemp1513[4];
    WData/*127:0*/ __Vtemp1514[4];
    WData/*127:0*/ __Vtemp1515[4];
    WData/*127:0*/ __Vtemp1516[4];
    WData/*127:0*/ __Vtemp1517[4];
    WData/*127:0*/ __Vtemp1518[4];
    WData/*127:0*/ __Vtemp1519[4];
    WData/*127:0*/ __Vtemp1520[4];
    WData/*127:0*/ __Vtemp1521[4];
    WData/*127:0*/ __Vtemp1522[4];
    WData/*127:0*/ __Vtemp1523[4];
    WData/*127:0*/ __Vtemp1532[4];
    WData/*127:0*/ __Vtemp1533[4];
    WData/*127:0*/ __Vtemp1534[4];
    WData/*127:0*/ __Vtemp1535[4];
    WData/*127:0*/ __Vtemp1536[4];
    WData/*127:0*/ __Vtemp1537[4];
    WData/*127:0*/ __Vtemp1538[4];
    WData/*127:0*/ __Vtemp1539[4];
    WData/*127:0*/ __Vtemp1540[4];
    WData/*127:0*/ __Vtemp1541[4];
    WData/*127:0*/ __Vtemp1542[4];
    WData/*127:0*/ __Vtemp1543[4];
    WData/*127:0*/ __Vtemp1544[4];
    WData/*127:0*/ __Vtemp1545[4];
    WData/*127:0*/ __Vtemp1546[4];
    WData/*127:0*/ __Vtemp1547[4];
    WData/*127:0*/ __Vtemp1556[4];
    WData/*127:0*/ __Vtemp1557[4];
    WData/*127:0*/ __Vtemp1558[4];
    WData/*127:0*/ __Vtemp1559[4];
    WData/*127:0*/ __Vtemp1560[4];
    WData/*127:0*/ __Vtemp1561[4];
    WData/*127:0*/ __Vtemp1562[4];
    WData/*127:0*/ __Vtemp1563[4];
    WData/*127:0*/ __Vtemp1564[4];
    WData/*127:0*/ __Vtemp1565[4];
    WData/*127:0*/ __Vtemp1566[4];
    WData/*127:0*/ __Vtemp1567[4];
    WData/*127:0*/ __Vtemp1568[4];
    WData/*127:0*/ __Vtemp1569[4];
    WData/*127:0*/ __Vtemp1570[4];
    WData/*127:0*/ __Vtemp1571[4];
    WData/*127:0*/ __Vtemp1580[4];
    WData/*127:0*/ __Vtemp1581[4];
    WData/*127:0*/ __Vtemp1582[4];
    WData/*127:0*/ __Vtemp1583[4];
    WData/*127:0*/ __Vtemp1584[4];
    WData/*127:0*/ __Vtemp1585[4];
    WData/*127:0*/ __Vtemp1586[4];
    WData/*127:0*/ __Vtemp1587[4];
    WData/*127:0*/ __Vtemp1588[4];
    WData/*127:0*/ __Vtemp1589[4];
    WData/*127:0*/ __Vtemp1590[4];
    WData/*127:0*/ __Vtemp1591[4];
    WData/*127:0*/ __Vtemp1592[4];
    WData/*127:0*/ __Vtemp1593[4];
    WData/*127:0*/ __Vtemp1594[4];
    WData/*127:0*/ __Vtemp1595[4];
    WData/*127:0*/ __Vtemp1604[4];
    WData/*127:0*/ __Vtemp1605[4];
    WData/*127:0*/ __Vtemp1606[4];
    WData/*127:0*/ __Vtemp1607[4];
    WData/*127:0*/ __Vtemp1608[4];
    WData/*127:0*/ __Vtemp1609[4];
    WData/*127:0*/ __Vtemp1610[4];
    WData/*127:0*/ __Vtemp1611[4];
    WData/*127:0*/ __Vtemp1612[4];
    WData/*127:0*/ __Vtemp1613[4];
    WData/*127:0*/ __Vtemp1614[4];
    WData/*127:0*/ __Vtemp1615[4];
    WData/*127:0*/ __Vtemp1616[4];
    WData/*127:0*/ __Vtemp1617[4];
    WData/*127:0*/ __Vtemp1618[4];
    WData/*127:0*/ __Vtemp1619[4];
    WData/*127:0*/ __Vtemp1628[4];
    WData/*127:0*/ __Vtemp1629[4];
    WData/*127:0*/ __Vtemp1630[4];
    WData/*127:0*/ __Vtemp1631[4];
    WData/*127:0*/ __Vtemp1632[4];
    WData/*127:0*/ __Vtemp1633[4];
    WData/*127:0*/ __Vtemp1634[4];
    WData/*127:0*/ __Vtemp1635[4];
    WData/*127:0*/ __Vtemp1636[4];
    WData/*127:0*/ __Vtemp1637[4];
    WData/*127:0*/ __Vtemp1638[4];
    WData/*127:0*/ __Vtemp1639[4];
    WData/*127:0*/ __Vtemp1640[4];
    WData/*127:0*/ __Vtemp1641[4];
    WData/*127:0*/ __Vtemp1642[4];
    WData/*127:0*/ __Vtemp1643[4];
    WData/*127:0*/ __Vtemp1652[4];
    WData/*127:0*/ __Vtemp1653[4];
    WData/*127:0*/ __Vtemp1654[4];
    WData/*127:0*/ __Vtemp1655[4];
    WData/*127:0*/ __Vtemp1656[4];
    WData/*127:0*/ __Vtemp1657[4];
    WData/*127:0*/ __Vtemp1658[4];
    WData/*127:0*/ __Vtemp1659[4];
    WData/*127:0*/ __Vtemp1660[4];
    WData/*127:0*/ __Vtemp1661[4];
    WData/*127:0*/ __Vtemp1662[4];
    WData/*127:0*/ __Vtemp1663[4];
    WData/*127:0*/ __Vtemp1664[4];
    WData/*127:0*/ __Vtemp1665[4];
    WData/*127:0*/ __Vtemp1666[4];
    WData/*127:0*/ __Vtemp1667[4];
    WData/*127:0*/ __Vtemp1676[4];
    WData/*127:0*/ __Vtemp1677[4];
    WData/*127:0*/ __Vtemp1678[4];
    WData/*127:0*/ __Vtemp1679[4];
    WData/*127:0*/ __Vtemp1680[4];
    WData/*127:0*/ __Vtemp1681[4];
    WData/*127:0*/ __Vtemp1682[4];
    WData/*127:0*/ __Vtemp1683[4];
    WData/*127:0*/ __Vtemp1684[4];
    WData/*127:0*/ __Vtemp1685[4];
    WData/*127:0*/ __Vtemp1686[4];
    WData/*127:0*/ __Vtemp1687[4];
    WData/*127:0*/ __Vtemp1688[4];
    WData/*127:0*/ __Vtemp1689[4];
    WData/*127:0*/ __Vtemp1690[4];
    WData/*127:0*/ __Vtemp1691[4];
    WData/*127:0*/ __Vtemp1507[4];
    WData/*127:0*/ __Vtemp1531[4];
    WData/*127:0*/ __Vtemp1555[4];
    WData/*127:0*/ __Vtemp1579[4];
    WData/*127:0*/ __Vtemp1603[4];
    WData/*127:0*/ __Vtemp1627[4];
    WData/*127:0*/ __Vtemp1651[4];
    WData/*127:0*/ __Vtemp1675[4];
    // Body
    {
        vcdp->chgBit(c+19649,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+19657,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->chgBit(c+19665,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBus(c+19673,((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 6U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x1aU)))),25);
        vcdp->chgBit(c+19681,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+19689,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+19697,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->chgBit(c+19705,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBus(c+19713,((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 6U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x1aU)))),25);
        vcdp->chgBit(c+19721,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+19729,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+19737,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->chgBit(c+19745,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBus(c+19753,((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 6U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x1aU)))),25);
        vcdp->chgBit(c+19761,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+19769,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+19777,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->chgBit(c+19785,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBus(c+19793,((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 6U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x1aU)))),25);
        vcdp->chgBit(c+19801,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+19809,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+19817,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->chgBit(c+19825,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBus(c+19833,((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 6U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x1aU)))),25);
        vcdp->chgBit(c+19841,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+19849,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+19857,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->chgBit(c+19865,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBus(c+19873,((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 6U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x1aU)))),25);
        vcdp->chgBit(c+19881,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+19889,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+19897,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->chgBit(c+19905,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBus(c+19913,((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 6U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x1aU)))),25);
        vcdp->chgBit(c+19921,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+19929,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+19937,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->chgBit(c+19945,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBus(c+19953,((0x1ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 6U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x1aU)))),25);
        vcdp->chgBit(c+19961,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+19969,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBus(c+19977,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__use_addr),28);
        vcdp->chgBit(c+19985,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBus(c+19993,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__use_valid),2);
        vcdp->chgBit(c+20001,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->chgBus(c+20009,(((IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r)
                                ? vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r
                                : vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r)),28);
        vcdp->chgBit(c+20017,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+20025,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+20033,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__size_r));
        vcdp->chgBus(c+20041,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__genblk3__DOT__data[0]),28);
        vcdp->chgBus(c+20042,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__genblk3__DOT__data[1]),28);
        vcdp->chgBus(c+20057,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),28);
        vcdp->chgBus(c+20065,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),28);
        vcdp->chgBit(c+20073,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r));
        vcdp->chgBit(c+20081,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r));
        vcdp->chgBit(c+20089,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r));
        vcdp->chgBit(c+20097,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+20105,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bank_dram_fill_req_valid),8);
        vcdp->chgArray(c+20113,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bank_dram_fill_req_addr),224);
        vcdp->chgBit(c+20169,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+20177,((1U & (~ (IData)((0U 
                                                != (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bank_dram_fill_req_valid)))))));
        vcdp->chgBus(c+20185,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__size_r),4);
        vcdp->chgArray(c+20193,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__data[0]),232);
        vcdp->chgArray(c+20201,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__data[1]),232);
        vcdp->chgArray(c+20209,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__data[2]),232);
        vcdp->chgArray(c+20217,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__data[3]),232);
        vcdp->chgArray(c+20225,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__data[4]),232);
        vcdp->chgArray(c+20233,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__data[5]),232);
        vcdp->chgArray(c+20241,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__data[6]),232);
        vcdp->chgArray(c+20249,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__data[7]),232);
        vcdp->chgArray(c+20705,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),232);
        vcdp->chgArray(c+20769,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),232);
        vcdp->chgBus(c+20833,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),3);
        vcdp->chgBus(c+20841,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),3);
        vcdp->chgBus(c+20849,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),3);
        vcdp->chgBit(c+20857,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+20865,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__requests_use),8);
        vcdp->chgBit(c+20873,((0U == (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__requests_use))));
        vcdp->chgBus(c+20881,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__refill_original),8);
        vcdp->chgBit(c+20889,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+20897,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+20905,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+20913,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+20921,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgQuad(c+20929,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag),42);
        vcdp->chgBit(c+20945,((0x10U == (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBit(c+20953,((0xbU < (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+20961,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                      << 0xdU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                  >> 0x13U)))),2);
        vcdp->chgBus(c+20969,(((0x18fU >= (0x1ffU & 
                                           ((IData)(0x19U) 
                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                ? (0x1ffffffU & (((0U 
                                                   == 
                                                   (0x1fU 
                                                    & ((IData)(0x19U) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                   ? 0U
                                                   : 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                   ((IData)(1U) 
                                                    + 
                                                    (0xfU 
                                                     & (((IData)(0x19U) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U)))] 
                                                   << 
                                                   ((IData)(0x20U) 
                                                    - 
                                                    (0x1fU 
                                                     & ((IData)(0x19U) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                    (0xfU 
                                                     & (((IData)(0x19U) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U))] 
                                                    >> 
                                                    (0x1fU 
                                                     & ((IData)(0x19U) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                : 0U)),25);
        vcdp->chgBus(c+20977,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                      << 0x1eU) | (
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                   >> 2U)))),2);
        vcdp->chgBus(c+20985,(((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                            [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                            >> 0x15U))),32);
        vcdp->chgQuad(c+20993,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                    << 0x37U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U])) 
                                                  << 0x17U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])) 
                                                    >> 9U))))),42);
        vcdp->chgBus(c+21009,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                        << 0x1cU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                        >> 4U)))),4);
        vcdp->chgBit(c+21017,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                     [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                     >> 1U))));
        vcdp->chgBit(c+21025,((1U & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                               [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])));
        vcdp->chgBus(c+21033,((3U & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])),2);
        vcdp->chgQuad(c+21041,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U])) 
                                    << 0x39U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                                  << 0x19U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])) 
                                                    >> 7U))))),42);
        vcdp->chgBit(c+21057,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))));
        vcdp->chgBus(c+21065,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                        << 0x1eU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                        >> 2U)))),4);
        vcdp->chgBit(c+21073,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x14U))));
        vcdp->chgBit(c+21081,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x13U))));
        vcdp->chgBus(c+21089,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x18U)))),2);
        vcdp->chgBus(c+21097,(((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                          >> 0x18U))),32);
        vcdp->chgBus(c+21105,(((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                                          >> 0x18U))),32);
        __Vtemp1507[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                      >> 0x18U));
        __Vtemp1507[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                      >> 0x18U));
        __Vtemp1507[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                                      >> 0x18U));
        __Vtemp1507[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                                      >> 0x18U));
        vcdp->chgArray(c+21113,(__Vtemp1507),128);
        vcdp->chgBit(c+21145,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 2U))));
        vcdp->chgBit(c+21153,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 1U))));
        vcdp->chgBus(c+21161,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                           << 0xfU) 
                                          | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                             >> 0x11U)))),16);
        vcdp->chgQuad(c+21169,((VL_ULL(0x1ffffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U]))))),49);
        vcdp->chgBus(c+21185,((0x1fffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                             << 0x1dU) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                               >> 3U)))),21);
        vcdp->chgBit(c+21193,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x15U))));
        vcdp->chgBit(c+21201,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x17U))));
        vcdp->chgBit(c+21209,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x16U))));
        vcdp->chgBit(c+21217,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x18U))));
        vcdp->chgBit(c+21225,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU))));
        vcdp->chgBit(c+21233,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1bU))));
        vcdp->chgBit(c+21241,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x19U))));
        vcdp->chgBit(c+21249,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+21257,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+21265,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+21273,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBus(c+21281,(((0x1fffff0U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                              << 1U)) 
                               | (0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                           << 6U) | 
                                          (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                           >> 0x1aU))))),25);
        vcdp->chgBus(c+21289,((0xfffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                              << 0x19U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                                >> 7U)))),28);
        vcdp->chgBit(c+21297,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_dual_valid_sel));
        vcdp->chgBus(c+21305,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__size_r),5);
        vcdp->chgQuad(c+21313,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[0]),54);
        vcdp->chgQuad(c+21315,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[1]),54);
        vcdp->chgQuad(c+21317,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[2]),54);
        vcdp->chgQuad(c+21319,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[3]),54);
        vcdp->chgQuad(c+21321,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[4]),54);
        vcdp->chgQuad(c+21323,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[5]),54);
        vcdp->chgQuad(c+21325,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[6]),54);
        vcdp->chgQuad(c+21327,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[7]),54);
        vcdp->chgQuad(c+21329,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[8]),54);
        vcdp->chgQuad(c+21331,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[9]),54);
        vcdp->chgQuad(c+21333,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[10]),54);
        vcdp->chgQuad(c+21335,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[11]),54);
        vcdp->chgQuad(c+21337,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[12]),54);
        vcdp->chgQuad(c+21339,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[13]),54);
        vcdp->chgQuad(c+21341,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[14]),54);
        vcdp->chgQuad(c+21343,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[15]),54);
        vcdp->chgQuad(c+21569,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),54);
        vcdp->chgQuad(c+21585,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),54);
        vcdp->chgBus(c+21601,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+21609,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+21617,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+21625,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+21633,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__size_r),5);
        vcdp->chgArray(c+21641,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[0]),153);
        vcdp->chgArray(c+21646,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[1]),153);
        vcdp->chgArray(c+21651,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[2]),153);
        vcdp->chgArray(c+21656,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[3]),153);
        vcdp->chgArray(c+21661,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[4]),153);
        vcdp->chgArray(c+21666,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[5]),153);
        vcdp->chgArray(c+21671,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[6]),153);
        vcdp->chgArray(c+21676,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[7]),153);
        vcdp->chgArray(c+21681,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[8]),153);
        vcdp->chgArray(c+21686,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[9]),153);
        vcdp->chgArray(c+21691,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[10]),153);
        vcdp->chgArray(c+21696,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[11]),153);
        vcdp->chgArray(c+21701,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[12]),153);
        vcdp->chgArray(c+21706,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[13]),153);
        vcdp->chgArray(c+21711,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[14]),153);
        vcdp->chgArray(c+21716,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[15]),153);
        vcdp->chgArray(c+22281,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),153);
        vcdp->chgArray(c+22321,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),153);
        vcdp->chgBus(c+22361,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+22369,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+22377,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+22385,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+22393,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids),4);
        vcdp->chgBus(c+22401,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_rw),4);
        vcdp->chgBus(c+22409,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen),16);
        vcdp->chgArray(c+22417,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_addr),120);
        vcdp->chgArray(c+22449,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata),128);
        vcdp->chgBit(c+22481,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+22489,((1U & (~ (IData)((0U 
                                                != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids)))))));
        vcdp->chgBus(c+22497,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__size_r),4);
        vcdp->chgArray(c+22505,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[0]),314);
        vcdp->chgArray(c+22515,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[1]),314);
        vcdp->chgArray(c+22525,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[2]),314);
        vcdp->chgArray(c+22535,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[3]),314);
        vcdp->chgArray(c+22545,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[4]),314);
        vcdp->chgArray(c+22555,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[5]),314);
        vcdp->chgArray(c+22565,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[6]),314);
        vcdp->chgArray(c+22575,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[7]),314);
        vcdp->chgArray(c+23145,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),314);
        vcdp->chgArray(c+23225,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),314);
        vcdp->chgBus(c+23305,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),3);
        vcdp->chgBus(c+23313,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),3);
        vcdp->chgBus(c+23321,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),3);
        vcdp->chgBit(c+23329,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgArray(c+23337,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__s0_1_c0__DOT__value),242);
        __Vtemp1508[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][0U];
        __Vtemp1508[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][1U];
        __Vtemp1508[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][2U];
        __Vtemp1508[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][3U];
        vcdp->chgArray(c+23401,(__Vtemp1508),128);
        __Vtemp1509[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][0U];
        __Vtemp1509[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][1U];
        __Vtemp1509[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][2U];
        __Vtemp1509[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][3U];
        vcdp->chgArray(c+23433,(__Vtemp1509),128);
        __Vtemp1510[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][0U];
        __Vtemp1510[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][1U];
        __Vtemp1510[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][2U];
        __Vtemp1510[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][3U];
        vcdp->chgArray(c+23465,(__Vtemp1510),128);
        __Vtemp1511[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][0U];
        __Vtemp1511[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][1U];
        __Vtemp1511[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][2U];
        __Vtemp1511[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][3U];
        vcdp->chgArray(c+23497,(__Vtemp1511),128);
        __Vtemp1512[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][0U];
        __Vtemp1512[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][1U];
        __Vtemp1512[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][2U];
        __Vtemp1512[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][3U];
        vcdp->chgArray(c+23529,(__Vtemp1512),128);
        __Vtemp1513[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][0U];
        __Vtemp1513[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][1U];
        __Vtemp1513[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][2U];
        __Vtemp1513[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][3U];
        vcdp->chgArray(c+23561,(__Vtemp1513),128);
        __Vtemp1514[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][0U];
        __Vtemp1514[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][1U];
        __Vtemp1514[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][2U];
        __Vtemp1514[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][3U];
        vcdp->chgArray(c+23593,(__Vtemp1514),128);
        __Vtemp1515[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][0U];
        __Vtemp1515[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][1U];
        __Vtemp1515[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][2U];
        __Vtemp1515[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][3U];
        vcdp->chgArray(c+23625,(__Vtemp1515),128);
        __Vtemp1516[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][0U];
        __Vtemp1516[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][1U];
        __Vtemp1516[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][2U];
        __Vtemp1516[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][3U];
        vcdp->chgArray(c+23657,(__Vtemp1516),128);
        __Vtemp1517[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][0U];
        __Vtemp1517[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][1U];
        __Vtemp1517[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][2U];
        __Vtemp1517[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][3U];
        vcdp->chgArray(c+23689,(__Vtemp1517),128);
        __Vtemp1518[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][0U];
        __Vtemp1518[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][1U];
        __Vtemp1518[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][2U];
        __Vtemp1518[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][3U];
        vcdp->chgArray(c+23721,(__Vtemp1518),128);
        __Vtemp1519[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][0U];
        __Vtemp1519[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][1U];
        __Vtemp1519[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][2U];
        __Vtemp1519[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][3U];
        vcdp->chgArray(c+23753,(__Vtemp1519),128);
        __Vtemp1520[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][0U];
        __Vtemp1520[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][1U];
        __Vtemp1520[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][2U];
        __Vtemp1520[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][3U];
        vcdp->chgArray(c+23785,(__Vtemp1520),128);
        __Vtemp1521[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][0U];
        __Vtemp1521[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][1U];
        __Vtemp1521[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][2U];
        __Vtemp1521[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][3U];
        vcdp->chgArray(c+23817,(__Vtemp1521),128);
        __Vtemp1522[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][0U];
        __Vtemp1522[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][1U];
        __Vtemp1522[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][2U];
        __Vtemp1522[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][3U];
        vcdp->chgArray(c+23849,(__Vtemp1522),128);
        __Vtemp1523[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][0U];
        __Vtemp1523[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][1U];
        __Vtemp1523[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][2U];
        __Vtemp1523[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][3U];
        vcdp->chgArray(c+23881,(__Vtemp1523),128);
        vcdp->chgBus(c+23913,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[0]),21);
        vcdp->chgBus(c+23914,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[1]),21);
        vcdp->chgBus(c+23915,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[2]),21);
        vcdp->chgBus(c+23916,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[3]),21);
        vcdp->chgBus(c+23917,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[4]),21);
        vcdp->chgBus(c+23918,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[5]),21);
        vcdp->chgBus(c+23919,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[6]),21);
        vcdp->chgBus(c+23920,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[7]),21);
        vcdp->chgBus(c+23921,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[8]),21);
        vcdp->chgBus(c+23922,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[9]),21);
        vcdp->chgBus(c+23923,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[10]),21);
        vcdp->chgBus(c+23924,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[11]),21);
        vcdp->chgBus(c+23925,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[12]),21);
        vcdp->chgBus(c+23926,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[13]),21);
        vcdp->chgBus(c+23927,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[14]),21);
        vcdp->chgBus(c+23928,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[15]),21);
        vcdp->chgBus(c+24041,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0U]),16);
        vcdp->chgBus(c+24049,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [1U]),16);
        vcdp->chgBus(c+24057,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [2U]),16);
        vcdp->chgBus(c+24065,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [3U]),16);
        vcdp->chgBus(c+24073,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [4U]),16);
        vcdp->chgBus(c+24081,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [5U]),16);
        vcdp->chgBus(c+24089,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [6U]),16);
        vcdp->chgBus(c+24097,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [7U]),16);
        vcdp->chgBus(c+24105,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [8U]),16);
        vcdp->chgBus(c+24113,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [9U]),16);
        vcdp->chgBus(c+24121,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xaU]),16);
        vcdp->chgBus(c+24129,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xbU]),16);
        vcdp->chgBus(c+24137,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xcU]),16);
        vcdp->chgBus(c+24145,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xdU]),16);
        vcdp->chgBus(c+24153,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xeU]),16);
        vcdp->chgBus(c+24161,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xfU]),16);
        vcdp->chgBus(c+24169,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty),16);
        vcdp->chgBus(c+24177,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid),16);
        vcdp->chgBus(c+24185,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__i),32);
        vcdp->chgBus(c+24193,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__j),32);
        vcdp->chgArray(c+24201,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__s0_1_c0__DOT__value),167);
        vcdp->chgArray(c+24249,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value),316);
        vcdp->chgArray(c+24329,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[0]),85);
        vcdp->chgArray(c+24332,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[1]),85);
        vcdp->chgArray(c+24335,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[2]),85);
        vcdp->chgArray(c+24338,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[3]),85);
        vcdp->chgArray(c+24341,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[4]),85);
        vcdp->chgArray(c+24344,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[5]),85);
        vcdp->chgArray(c+24347,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[6]),85);
        vcdp->chgArray(c+24350,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[7]),85);
        vcdp->chgArray(c+24353,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[8]),85);
        vcdp->chgArray(c+24356,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[9]),85);
        vcdp->chgArray(c+24359,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[10]),85);
        vcdp->chgArray(c+24362,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[11]),85);
        vcdp->chgArray(c+24365,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[12]),85);
        vcdp->chgArray(c+24368,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[13]),85);
        vcdp->chgArray(c+24371,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[14]),85);
        vcdp->chgArray(c+24374,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[15]),85);
        vcdp->chgArray(c+24713,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table),400);
        vcdp->chgBus(c+24817,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_table),16);
        vcdp->chgBus(c+24825,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__ready_table),16);
        vcdp->chgBus(c+24833,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr),4);
        vcdp->chgBus(c+24841,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__head_ptr),4);
        vcdp->chgBus(c+24849,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr),4);
        vcdp->chgBus(c+24857,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size),5);
        vcdp->chgBit(c+24865,((0x10U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+24873,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__size_r),4);
        vcdp->chgArray(c+24881,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[0]),76);
        vcdp->chgArray(c+24884,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[1]),76);
        vcdp->chgArray(c+24887,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[2]),76);
        vcdp->chgArray(c+24890,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[3]),76);
        vcdp->chgArray(c+24893,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[4]),76);
        vcdp->chgArray(c+24896,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[5]),76);
        vcdp->chgArray(c+24899,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[6]),76);
        vcdp->chgArray(c+24902,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[7]),76);
        vcdp->chgArray(c+25073,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),76);
        vcdp->chgArray(c+25097,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),76);
        vcdp->chgBus(c+25121,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),3);
        vcdp->chgBus(c+25129,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),3);
        vcdp->chgBus(c+25137,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),3);
        vcdp->chgBit(c+25145,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+25153,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__size_r),3);
        vcdp->chgArray(c+25161,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[0]),199);
        vcdp->chgArray(c+25168,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[1]),199);
        vcdp->chgArray(c+25175,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[2]),199);
        vcdp->chgArray(c+25182,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[3]),199);
        vcdp->chgArray(c+25385,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),199);
        vcdp->chgArray(c+25441,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),199);
        vcdp->chgBus(c+25497,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+25505,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+25513,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+25521,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBit(c+25529,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+25537,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+25545,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+25553,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+25561,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgQuad(c+25569,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag),42);
        vcdp->chgBit(c+25585,((0x10U == (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBit(c+25593,((0xbU < (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+25601,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                      << 0xdU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                  >> 0x13U)))),2);
        vcdp->chgBus(c+25609,(((0x18fU >= (0x1ffU & 
                                           ((IData)(0x19U) 
                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                ? (0x1ffffffU & (((0U 
                                                   == 
                                                   (0x1fU 
                                                    & ((IData)(0x19U) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                   ? 0U
                                                   : 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                   ((IData)(1U) 
                                                    + 
                                                    (0xfU 
                                                     & (((IData)(0x19U) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U)))] 
                                                   << 
                                                   ((IData)(0x20U) 
                                                    - 
                                                    (0x1fU 
                                                     & ((IData)(0x19U) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                    (0xfU 
                                                     & (((IData)(0x19U) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U))] 
                                                    >> 
                                                    (0x1fU 
                                                     & ((IData)(0x19U) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                : 0U)),25);
        vcdp->chgBus(c+25617,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                      << 0x1eU) | (
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                   >> 2U)))),2);
        vcdp->chgBus(c+25625,(((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                            [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                            >> 0x15U))),32);
        vcdp->chgQuad(c+25633,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                    << 0x37U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U])) 
                                                  << 0x17U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])) 
                                                    >> 9U))))),42);
        vcdp->chgBus(c+25649,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                        << 0x1cU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                        >> 4U)))),4);
        vcdp->chgBit(c+25657,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                     [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                     >> 1U))));
        vcdp->chgBit(c+25665,((1U & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                               [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])));
        vcdp->chgBus(c+25673,((3U & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])),2);
        vcdp->chgQuad(c+25681,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U])) 
                                    << 0x39U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                                  << 0x19U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])) 
                                                    >> 7U))))),42);
        vcdp->chgBit(c+25697,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))));
        vcdp->chgBus(c+25705,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                        << 0x1eU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                        >> 2U)))),4);
        vcdp->chgBit(c+25713,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x14U))));
        vcdp->chgBit(c+25721,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x13U))));
        vcdp->chgBus(c+25729,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x18U)))),2);
        vcdp->chgBus(c+25737,(((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                          >> 0x18U))),32);
        vcdp->chgBus(c+25745,(((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                                          >> 0x18U))),32);
        __Vtemp1531[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                      >> 0x18U));
        __Vtemp1531[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                      >> 0x18U));
        __Vtemp1531[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                                      >> 0x18U));
        __Vtemp1531[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                                      >> 0x18U));
        vcdp->chgArray(c+25753,(__Vtemp1531),128);
        vcdp->chgBit(c+25785,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 2U))));
        vcdp->chgBit(c+25793,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 1U))));
        vcdp->chgBus(c+25801,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                           << 0xfU) 
                                          | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                             >> 0x11U)))),16);
        vcdp->chgQuad(c+25809,((VL_ULL(0x1ffffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U]))))),49);
        vcdp->chgBus(c+25825,((0x1fffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                             << 0x1dU) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                               >> 3U)))),21);
        vcdp->chgBit(c+25833,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x15U))));
        vcdp->chgBit(c+25841,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x17U))));
        vcdp->chgBit(c+25849,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x16U))));
        vcdp->chgBit(c+25857,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x18U))));
        vcdp->chgBit(c+25865,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU))));
        vcdp->chgBit(c+25873,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1bU))));
        vcdp->chgBit(c+25881,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x19U))));
        vcdp->chgBit(c+25889,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+25897,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+25905,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+25913,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBus(c+25921,(((0x1fffff0U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                              << 1U)) 
                               | (0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                           << 6U) | 
                                          (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                           >> 0x1aU))))),25);
        vcdp->chgBus(c+25929,((0xfffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                              << 0x19U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                                >> 7U)))),28);
        vcdp->chgBit(c+25937,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_dual_valid_sel));
        vcdp->chgBus(c+25945,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__size_r),5);
        vcdp->chgQuad(c+25953,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[0]),54);
        vcdp->chgQuad(c+25955,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[1]),54);
        vcdp->chgQuad(c+25957,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[2]),54);
        vcdp->chgQuad(c+25959,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[3]),54);
        vcdp->chgQuad(c+25961,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[4]),54);
        vcdp->chgQuad(c+25963,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[5]),54);
        vcdp->chgQuad(c+25965,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[6]),54);
        vcdp->chgQuad(c+25967,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[7]),54);
        vcdp->chgQuad(c+25969,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[8]),54);
        vcdp->chgQuad(c+25971,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[9]),54);
        vcdp->chgQuad(c+25973,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[10]),54);
        vcdp->chgQuad(c+25975,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[11]),54);
        vcdp->chgQuad(c+25977,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[12]),54);
        vcdp->chgQuad(c+25979,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[13]),54);
        vcdp->chgQuad(c+25981,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[14]),54);
        vcdp->chgQuad(c+25983,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[15]),54);
        vcdp->chgQuad(c+26209,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),54);
        vcdp->chgQuad(c+26225,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),54);
        vcdp->chgBus(c+26241,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+26249,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+26257,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+26265,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+26273,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__size_r),5);
        vcdp->chgArray(c+26281,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[0]),153);
        vcdp->chgArray(c+26286,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[1]),153);
        vcdp->chgArray(c+26291,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[2]),153);
        vcdp->chgArray(c+26296,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[3]),153);
        vcdp->chgArray(c+26301,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[4]),153);
        vcdp->chgArray(c+26306,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[5]),153);
        vcdp->chgArray(c+26311,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[6]),153);
        vcdp->chgArray(c+26316,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[7]),153);
        vcdp->chgArray(c+26321,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[8]),153);
        vcdp->chgArray(c+26326,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[9]),153);
        vcdp->chgArray(c+26331,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[10]),153);
        vcdp->chgArray(c+26336,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[11]),153);
        vcdp->chgArray(c+26341,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[12]),153);
        vcdp->chgArray(c+26346,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[13]),153);
        vcdp->chgArray(c+26351,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[14]),153);
        vcdp->chgArray(c+26356,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[15]),153);
        vcdp->chgArray(c+26921,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),153);
        vcdp->chgArray(c+26961,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),153);
        vcdp->chgBus(c+27001,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+27009,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+27017,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+27025,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+27033,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids),4);
        vcdp->chgBus(c+27041,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_rw),4);
        vcdp->chgBus(c+27049,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen),16);
        vcdp->chgArray(c+27057,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_addr),120);
        vcdp->chgArray(c+27089,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata),128);
        vcdp->chgBit(c+27121,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+27129,((1U & (~ (IData)((0U 
                                                != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids)))))));
        vcdp->chgBus(c+27137,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__size_r),4);
        vcdp->chgArray(c+27145,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[0]),314);
        vcdp->chgArray(c+27155,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[1]),314);
        vcdp->chgArray(c+27165,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[2]),314);
        vcdp->chgArray(c+27175,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[3]),314);
        vcdp->chgArray(c+27185,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[4]),314);
        vcdp->chgArray(c+27195,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[5]),314);
        vcdp->chgArray(c+27205,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[6]),314);
        vcdp->chgArray(c+27215,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[7]),314);
        vcdp->chgArray(c+27785,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),314);
        vcdp->chgArray(c+27865,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),314);
        vcdp->chgBus(c+27945,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),3);
        vcdp->chgBus(c+27953,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),3);
        vcdp->chgBus(c+27961,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),3);
        vcdp->chgBit(c+27969,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgArray(c+27977,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__s0_1_c0__DOT__value),242);
        __Vtemp1532[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][0U];
        __Vtemp1532[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][1U];
        __Vtemp1532[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][2U];
        __Vtemp1532[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][3U];
        vcdp->chgArray(c+28041,(__Vtemp1532),128);
        __Vtemp1533[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][0U];
        __Vtemp1533[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][1U];
        __Vtemp1533[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][2U];
        __Vtemp1533[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][3U];
        vcdp->chgArray(c+28073,(__Vtemp1533),128);
        __Vtemp1534[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][0U];
        __Vtemp1534[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][1U];
        __Vtemp1534[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][2U];
        __Vtemp1534[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][3U];
        vcdp->chgArray(c+28105,(__Vtemp1534),128);
        __Vtemp1535[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][0U];
        __Vtemp1535[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][1U];
        __Vtemp1535[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][2U];
        __Vtemp1535[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][3U];
        vcdp->chgArray(c+28137,(__Vtemp1535),128);
        __Vtemp1536[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][0U];
        __Vtemp1536[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][1U];
        __Vtemp1536[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][2U];
        __Vtemp1536[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][3U];
        vcdp->chgArray(c+28169,(__Vtemp1536),128);
        __Vtemp1537[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][0U];
        __Vtemp1537[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][1U];
        __Vtemp1537[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][2U];
        __Vtemp1537[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][3U];
        vcdp->chgArray(c+28201,(__Vtemp1537),128);
        __Vtemp1538[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][0U];
        __Vtemp1538[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][1U];
        __Vtemp1538[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][2U];
        __Vtemp1538[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][3U];
        vcdp->chgArray(c+28233,(__Vtemp1538),128);
        __Vtemp1539[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][0U];
        __Vtemp1539[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][1U];
        __Vtemp1539[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][2U];
        __Vtemp1539[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][3U];
        vcdp->chgArray(c+28265,(__Vtemp1539),128);
        __Vtemp1540[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][0U];
        __Vtemp1540[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][1U];
        __Vtemp1540[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][2U];
        __Vtemp1540[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][3U];
        vcdp->chgArray(c+28297,(__Vtemp1540),128);
        __Vtemp1541[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][0U];
        __Vtemp1541[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][1U];
        __Vtemp1541[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][2U];
        __Vtemp1541[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][3U];
        vcdp->chgArray(c+28329,(__Vtemp1541),128);
        __Vtemp1542[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][0U];
        __Vtemp1542[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][1U];
        __Vtemp1542[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][2U];
        __Vtemp1542[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][3U];
        vcdp->chgArray(c+28361,(__Vtemp1542),128);
        __Vtemp1543[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][0U];
        __Vtemp1543[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][1U];
        __Vtemp1543[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][2U];
        __Vtemp1543[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][3U];
        vcdp->chgArray(c+28393,(__Vtemp1543),128);
        __Vtemp1544[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][0U];
        __Vtemp1544[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][1U];
        __Vtemp1544[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][2U];
        __Vtemp1544[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][3U];
        vcdp->chgArray(c+28425,(__Vtemp1544),128);
        __Vtemp1545[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][0U];
        __Vtemp1545[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][1U];
        __Vtemp1545[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][2U];
        __Vtemp1545[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][3U];
        vcdp->chgArray(c+28457,(__Vtemp1545),128);
        __Vtemp1546[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][0U];
        __Vtemp1546[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][1U];
        __Vtemp1546[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][2U];
        __Vtemp1546[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][3U];
        vcdp->chgArray(c+28489,(__Vtemp1546),128);
        __Vtemp1547[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][0U];
        __Vtemp1547[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][1U];
        __Vtemp1547[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][2U];
        __Vtemp1547[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][3U];
        vcdp->chgArray(c+28521,(__Vtemp1547),128);
        vcdp->chgBus(c+28553,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[0]),21);
        vcdp->chgBus(c+28554,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[1]),21);
        vcdp->chgBus(c+28555,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[2]),21);
        vcdp->chgBus(c+28556,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[3]),21);
        vcdp->chgBus(c+28557,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[4]),21);
        vcdp->chgBus(c+28558,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[5]),21);
        vcdp->chgBus(c+28559,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[6]),21);
        vcdp->chgBus(c+28560,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[7]),21);
        vcdp->chgBus(c+28561,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[8]),21);
        vcdp->chgBus(c+28562,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[9]),21);
        vcdp->chgBus(c+28563,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[10]),21);
        vcdp->chgBus(c+28564,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[11]),21);
        vcdp->chgBus(c+28565,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[12]),21);
        vcdp->chgBus(c+28566,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[13]),21);
        vcdp->chgBus(c+28567,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[14]),21);
        vcdp->chgBus(c+28568,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[15]),21);
        vcdp->chgBus(c+28681,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0U]),16);
        vcdp->chgBus(c+28689,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [1U]),16);
        vcdp->chgBus(c+28697,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [2U]),16);
        vcdp->chgBus(c+28705,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [3U]),16);
        vcdp->chgBus(c+28713,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [4U]),16);
        vcdp->chgBus(c+28721,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [5U]),16);
        vcdp->chgBus(c+28729,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [6U]),16);
        vcdp->chgBus(c+28737,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [7U]),16);
        vcdp->chgBus(c+28745,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [8U]),16);
        vcdp->chgBus(c+28753,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [9U]),16);
        vcdp->chgBus(c+28761,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xaU]),16);
        vcdp->chgBus(c+28769,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xbU]),16);
        vcdp->chgBus(c+28777,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xcU]),16);
        vcdp->chgBus(c+28785,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xdU]),16);
        vcdp->chgBus(c+28793,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xeU]),16);
        vcdp->chgBus(c+28801,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xfU]),16);
        vcdp->chgBus(c+28809,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty),16);
        vcdp->chgBus(c+28817,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid),16);
        vcdp->chgBus(c+28825,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__i),32);
        vcdp->chgBus(c+28833,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__j),32);
        vcdp->chgArray(c+28841,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__s0_1_c0__DOT__value),167);
        vcdp->chgArray(c+28889,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value),316);
        vcdp->chgArray(c+28969,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[0]),85);
        vcdp->chgArray(c+28972,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[1]),85);
        vcdp->chgArray(c+28975,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[2]),85);
        vcdp->chgArray(c+28978,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[3]),85);
        vcdp->chgArray(c+28981,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[4]),85);
        vcdp->chgArray(c+28984,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[5]),85);
        vcdp->chgArray(c+28987,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[6]),85);
        vcdp->chgArray(c+28990,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[7]),85);
        vcdp->chgArray(c+28993,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[8]),85);
        vcdp->chgArray(c+28996,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[9]),85);
        vcdp->chgArray(c+28999,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[10]),85);
        vcdp->chgArray(c+29002,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[11]),85);
        vcdp->chgArray(c+29005,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[12]),85);
        vcdp->chgArray(c+29008,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[13]),85);
        vcdp->chgArray(c+29011,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[14]),85);
        vcdp->chgArray(c+29014,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[15]),85);
        vcdp->chgArray(c+29353,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table),400);
        vcdp->chgBus(c+29457,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_table),16);
        vcdp->chgBus(c+29465,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__ready_table),16);
        vcdp->chgBus(c+29473,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr),4);
        vcdp->chgBus(c+29481,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__head_ptr),4);
        vcdp->chgBus(c+29489,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr),4);
        vcdp->chgBus(c+29497,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size),5);
        vcdp->chgBit(c+29505,((0x10U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+29513,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__size_r),4);
        vcdp->chgArray(c+29521,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[0]),76);
        vcdp->chgArray(c+29524,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[1]),76);
        vcdp->chgArray(c+29527,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[2]),76);
        vcdp->chgArray(c+29530,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[3]),76);
        vcdp->chgArray(c+29533,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[4]),76);
        vcdp->chgArray(c+29536,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[5]),76);
        vcdp->chgArray(c+29539,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[6]),76);
        vcdp->chgArray(c+29542,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[7]),76);
        vcdp->chgArray(c+29713,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),76);
        vcdp->chgArray(c+29737,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),76);
        vcdp->chgBus(c+29761,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),3);
        vcdp->chgBus(c+29769,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),3);
        vcdp->chgBus(c+29777,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),3);
        vcdp->chgBit(c+29785,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+29793,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__size_r),3);
        vcdp->chgArray(c+29801,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[0]),199);
        vcdp->chgArray(c+29808,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[1]),199);
        vcdp->chgArray(c+29815,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[2]),199);
        vcdp->chgArray(c+29822,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[3]),199);
        vcdp->chgArray(c+30025,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),199);
        vcdp->chgArray(c+30081,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),199);
        vcdp->chgBus(c+30137,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+30145,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+30153,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+30161,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBit(c+30169,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+30177,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+30185,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+30193,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+30201,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgQuad(c+30209,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag),42);
        vcdp->chgBit(c+30225,((0x10U == (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBit(c+30233,((0xbU < (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+30241,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                      << 0xdU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                  >> 0x13U)))),2);
        vcdp->chgBus(c+30249,(((0x18fU >= (0x1ffU & 
                                           ((IData)(0x19U) 
                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                ? (0x1ffffffU & (((0U 
                                                   == 
                                                   (0x1fU 
                                                    & ((IData)(0x19U) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                   ? 0U
                                                   : 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                   ((IData)(1U) 
                                                    + 
                                                    (0xfU 
                                                     & (((IData)(0x19U) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U)))] 
                                                   << 
                                                   ((IData)(0x20U) 
                                                    - 
                                                    (0x1fU 
                                                     & ((IData)(0x19U) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                    (0xfU 
                                                     & (((IData)(0x19U) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U))] 
                                                    >> 
                                                    (0x1fU 
                                                     & ((IData)(0x19U) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                : 0U)),25);
        vcdp->chgBus(c+30257,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                      << 0x1eU) | (
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                   >> 2U)))),2);
        vcdp->chgBus(c+30265,(((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                            [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                            >> 0x15U))),32);
        vcdp->chgQuad(c+30273,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                    << 0x37U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U])) 
                                                  << 0x17U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])) 
                                                    >> 9U))))),42);
        vcdp->chgBus(c+30289,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                        << 0x1cU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                        >> 4U)))),4);
        vcdp->chgBit(c+30297,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                     [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                     >> 1U))));
        vcdp->chgBit(c+30305,((1U & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                               [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])));
        vcdp->chgBus(c+30313,((3U & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])),2);
        vcdp->chgQuad(c+30321,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U])) 
                                    << 0x39U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                                  << 0x19U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])) 
                                                    >> 7U))))),42);
        vcdp->chgBit(c+30337,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))));
        vcdp->chgBus(c+30345,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                        << 0x1eU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                        >> 2U)))),4);
        vcdp->chgBit(c+30353,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x14U))));
        vcdp->chgBit(c+30361,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x13U))));
        vcdp->chgBus(c+30369,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x18U)))),2);
        vcdp->chgBus(c+30377,(((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                          >> 0x18U))),32);
        vcdp->chgBus(c+30385,(((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                                          >> 0x18U))),32);
        __Vtemp1555[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                      >> 0x18U));
        __Vtemp1555[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                      >> 0x18U));
        __Vtemp1555[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                                      >> 0x18U));
        __Vtemp1555[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                                      >> 0x18U));
        vcdp->chgArray(c+30393,(__Vtemp1555),128);
        vcdp->chgBit(c+30425,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 2U))));
        vcdp->chgBit(c+30433,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 1U))));
        vcdp->chgBus(c+30441,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                           << 0xfU) 
                                          | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                             >> 0x11U)))),16);
        vcdp->chgQuad(c+30449,((VL_ULL(0x1ffffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U]))))),49);
        vcdp->chgBus(c+30465,((0x1fffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                             << 0x1dU) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                               >> 3U)))),21);
        vcdp->chgBit(c+30473,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x15U))));
        vcdp->chgBit(c+30481,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x17U))));
        vcdp->chgBit(c+30489,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x16U))));
        vcdp->chgBit(c+30497,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x18U))));
        vcdp->chgBit(c+30505,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU))));
        vcdp->chgBit(c+30513,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1bU))));
        vcdp->chgBit(c+30521,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x19U))));
        vcdp->chgBit(c+30529,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+30537,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+30545,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+30553,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBus(c+30561,(((0x1fffff0U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                              << 1U)) 
                               | (0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                           << 6U) | 
                                          (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                           >> 0x1aU))))),25);
        vcdp->chgBus(c+30569,((0xfffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                              << 0x19U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                                >> 7U)))),28);
        vcdp->chgBit(c+30577,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_dual_valid_sel));
        vcdp->chgBus(c+30585,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__size_r),5);
        vcdp->chgQuad(c+30593,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[0]),54);
        vcdp->chgQuad(c+30595,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[1]),54);
        vcdp->chgQuad(c+30597,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[2]),54);
        vcdp->chgQuad(c+30599,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[3]),54);
        vcdp->chgQuad(c+30601,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[4]),54);
        vcdp->chgQuad(c+30603,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[5]),54);
        vcdp->chgQuad(c+30605,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[6]),54);
        vcdp->chgQuad(c+30607,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[7]),54);
        vcdp->chgQuad(c+30609,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[8]),54);
        vcdp->chgQuad(c+30611,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[9]),54);
        vcdp->chgQuad(c+30613,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[10]),54);
        vcdp->chgQuad(c+30615,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[11]),54);
        vcdp->chgQuad(c+30617,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[12]),54);
        vcdp->chgQuad(c+30619,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[13]),54);
        vcdp->chgQuad(c+30621,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[14]),54);
        vcdp->chgQuad(c+30623,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[15]),54);
        vcdp->chgQuad(c+30849,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),54);
        vcdp->chgQuad(c+30865,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),54);
        vcdp->chgBus(c+30881,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+30889,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+30897,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+30905,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+30913,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__size_r),5);
        vcdp->chgArray(c+30921,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[0]),153);
        vcdp->chgArray(c+30926,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[1]),153);
        vcdp->chgArray(c+30931,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[2]),153);
        vcdp->chgArray(c+30936,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[3]),153);
        vcdp->chgArray(c+30941,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[4]),153);
        vcdp->chgArray(c+30946,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[5]),153);
        vcdp->chgArray(c+30951,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[6]),153);
        vcdp->chgArray(c+30956,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[7]),153);
        vcdp->chgArray(c+30961,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[8]),153);
        vcdp->chgArray(c+30966,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[9]),153);
        vcdp->chgArray(c+30971,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[10]),153);
        vcdp->chgArray(c+30976,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[11]),153);
        vcdp->chgArray(c+30981,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[12]),153);
        vcdp->chgArray(c+30986,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[13]),153);
        vcdp->chgArray(c+30991,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[14]),153);
        vcdp->chgArray(c+30996,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[15]),153);
        vcdp->chgArray(c+31561,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),153);
        vcdp->chgArray(c+31601,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),153);
        vcdp->chgBus(c+31641,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+31649,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+31657,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+31665,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+31673,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids),4);
        vcdp->chgBus(c+31681,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_rw),4);
        vcdp->chgBus(c+31689,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen),16);
        vcdp->chgArray(c+31697,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_addr),120);
        vcdp->chgArray(c+31729,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata),128);
        vcdp->chgBit(c+31761,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+31769,((1U & (~ (IData)((0U 
                                                != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids)))))));
        vcdp->chgBus(c+31777,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__size_r),4);
        vcdp->chgArray(c+31785,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[0]),314);
        vcdp->chgArray(c+31795,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[1]),314);
        vcdp->chgArray(c+31805,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[2]),314);
        vcdp->chgArray(c+31815,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[3]),314);
        vcdp->chgArray(c+31825,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[4]),314);
        vcdp->chgArray(c+31835,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[5]),314);
        vcdp->chgArray(c+31845,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[6]),314);
        vcdp->chgArray(c+31855,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[7]),314);
        vcdp->chgArray(c+32425,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),314);
        vcdp->chgArray(c+32505,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),314);
        vcdp->chgBus(c+32585,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),3);
        vcdp->chgBus(c+32593,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),3);
        vcdp->chgBus(c+32601,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),3);
        vcdp->chgBit(c+32609,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgArray(c+32617,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__s0_1_c0__DOT__value),242);
        __Vtemp1556[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][0U];
        __Vtemp1556[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][1U];
        __Vtemp1556[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][2U];
        __Vtemp1556[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][3U];
        vcdp->chgArray(c+32681,(__Vtemp1556),128);
        __Vtemp1557[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][0U];
        __Vtemp1557[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][1U];
        __Vtemp1557[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][2U];
        __Vtemp1557[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][3U];
        vcdp->chgArray(c+32713,(__Vtemp1557),128);
        __Vtemp1558[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][0U];
        __Vtemp1558[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][1U];
        __Vtemp1558[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][2U];
        __Vtemp1558[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][3U];
        vcdp->chgArray(c+32745,(__Vtemp1558),128);
        __Vtemp1559[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][0U];
        __Vtemp1559[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][1U];
        __Vtemp1559[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][2U];
        __Vtemp1559[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][3U];
        vcdp->chgArray(c+32777,(__Vtemp1559),128);
        __Vtemp1560[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][0U];
        __Vtemp1560[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][1U];
        __Vtemp1560[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][2U];
        __Vtemp1560[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][3U];
        vcdp->chgArray(c+32809,(__Vtemp1560),128);
        __Vtemp1561[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][0U];
        __Vtemp1561[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][1U];
        __Vtemp1561[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][2U];
        __Vtemp1561[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][3U];
        vcdp->chgArray(c+32841,(__Vtemp1561),128);
        __Vtemp1562[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][0U];
        __Vtemp1562[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][1U];
        __Vtemp1562[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][2U];
        __Vtemp1562[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][3U];
        vcdp->chgArray(c+32873,(__Vtemp1562),128);
        __Vtemp1563[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][0U];
        __Vtemp1563[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][1U];
        __Vtemp1563[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][2U];
        __Vtemp1563[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][3U];
        vcdp->chgArray(c+32905,(__Vtemp1563),128);
        __Vtemp1564[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][0U];
        __Vtemp1564[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][1U];
        __Vtemp1564[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][2U];
        __Vtemp1564[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][3U];
        vcdp->chgArray(c+32937,(__Vtemp1564),128);
        __Vtemp1565[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][0U];
        __Vtemp1565[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][1U];
        __Vtemp1565[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][2U];
        __Vtemp1565[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][3U];
        vcdp->chgArray(c+32969,(__Vtemp1565),128);
        __Vtemp1566[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][0U];
        __Vtemp1566[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][1U];
        __Vtemp1566[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][2U];
        __Vtemp1566[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][3U];
        vcdp->chgArray(c+33001,(__Vtemp1566),128);
        __Vtemp1567[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][0U];
        __Vtemp1567[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][1U];
        __Vtemp1567[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][2U];
        __Vtemp1567[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][3U];
        vcdp->chgArray(c+33033,(__Vtemp1567),128);
        __Vtemp1568[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][0U];
        __Vtemp1568[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][1U];
        __Vtemp1568[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][2U];
        __Vtemp1568[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][3U];
        vcdp->chgArray(c+33065,(__Vtemp1568),128);
        __Vtemp1569[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][0U];
        __Vtemp1569[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][1U];
        __Vtemp1569[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][2U];
        __Vtemp1569[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][3U];
        vcdp->chgArray(c+33097,(__Vtemp1569),128);
        __Vtemp1570[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][0U];
        __Vtemp1570[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][1U];
        __Vtemp1570[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][2U];
        __Vtemp1570[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][3U];
        vcdp->chgArray(c+33129,(__Vtemp1570),128);
        __Vtemp1571[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][0U];
        __Vtemp1571[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][1U];
        __Vtemp1571[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][2U];
        __Vtemp1571[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][3U];
        vcdp->chgArray(c+33161,(__Vtemp1571),128);
        vcdp->chgBus(c+33193,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[0]),21);
        vcdp->chgBus(c+33194,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[1]),21);
        vcdp->chgBus(c+33195,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[2]),21);
        vcdp->chgBus(c+33196,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[3]),21);
        vcdp->chgBus(c+33197,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[4]),21);
        vcdp->chgBus(c+33198,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[5]),21);
        vcdp->chgBus(c+33199,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[6]),21);
        vcdp->chgBus(c+33200,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[7]),21);
        vcdp->chgBus(c+33201,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[8]),21);
        vcdp->chgBus(c+33202,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[9]),21);
        vcdp->chgBus(c+33203,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[10]),21);
        vcdp->chgBus(c+33204,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[11]),21);
        vcdp->chgBus(c+33205,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[12]),21);
        vcdp->chgBus(c+33206,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[13]),21);
        vcdp->chgBus(c+33207,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[14]),21);
        vcdp->chgBus(c+33208,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[15]),21);
        vcdp->chgBus(c+33321,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0U]),16);
        vcdp->chgBus(c+33329,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [1U]),16);
        vcdp->chgBus(c+33337,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [2U]),16);
        vcdp->chgBus(c+33345,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [3U]),16);
        vcdp->chgBus(c+33353,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [4U]),16);
        vcdp->chgBus(c+33361,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [5U]),16);
        vcdp->chgBus(c+33369,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [6U]),16);
        vcdp->chgBus(c+33377,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [7U]),16);
        vcdp->chgBus(c+33385,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [8U]),16);
        vcdp->chgBus(c+33393,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [9U]),16);
        vcdp->chgBus(c+33401,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xaU]),16);
        vcdp->chgBus(c+33409,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xbU]),16);
        vcdp->chgBus(c+33417,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xcU]),16);
        vcdp->chgBus(c+33425,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xdU]),16);
        vcdp->chgBus(c+33433,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xeU]),16);
        vcdp->chgBus(c+33441,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xfU]),16);
        vcdp->chgBus(c+33449,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty),16);
        vcdp->chgBus(c+33457,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid),16);
        vcdp->chgBus(c+33465,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__i),32);
        vcdp->chgBus(c+33473,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__j),32);
        vcdp->chgArray(c+33481,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__s0_1_c0__DOT__value),167);
        vcdp->chgArray(c+33529,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value),316);
        vcdp->chgArray(c+33609,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[0]),85);
        vcdp->chgArray(c+33612,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[1]),85);
        vcdp->chgArray(c+33615,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[2]),85);
        vcdp->chgArray(c+33618,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[3]),85);
        vcdp->chgArray(c+33621,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[4]),85);
        vcdp->chgArray(c+33624,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[5]),85);
        vcdp->chgArray(c+33627,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[6]),85);
        vcdp->chgArray(c+33630,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[7]),85);
        vcdp->chgArray(c+33633,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[8]),85);
        vcdp->chgArray(c+33636,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[9]),85);
        vcdp->chgArray(c+33639,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[10]),85);
        vcdp->chgArray(c+33642,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[11]),85);
        vcdp->chgArray(c+33645,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[12]),85);
        vcdp->chgArray(c+33648,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[13]),85);
        vcdp->chgArray(c+33651,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[14]),85);
        vcdp->chgArray(c+33654,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[15]),85);
        vcdp->chgArray(c+33993,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table),400);
        vcdp->chgBus(c+34097,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_table),16);
        vcdp->chgBus(c+34105,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__ready_table),16);
        vcdp->chgBus(c+34113,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr),4);
        vcdp->chgBus(c+34121,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__head_ptr),4);
        vcdp->chgBus(c+34129,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr),4);
        vcdp->chgBus(c+34137,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size),5);
        vcdp->chgBit(c+34145,((0x10U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+34153,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__size_r),4);
        vcdp->chgArray(c+34161,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[0]),76);
        vcdp->chgArray(c+34164,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[1]),76);
        vcdp->chgArray(c+34167,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[2]),76);
        vcdp->chgArray(c+34170,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[3]),76);
        vcdp->chgArray(c+34173,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[4]),76);
        vcdp->chgArray(c+34176,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[5]),76);
        vcdp->chgArray(c+34179,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[6]),76);
        vcdp->chgArray(c+34182,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[7]),76);
        vcdp->chgArray(c+34353,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),76);
        vcdp->chgArray(c+34377,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),76);
        vcdp->chgBus(c+34401,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),3);
        vcdp->chgBus(c+34409,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),3);
        vcdp->chgBus(c+34417,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),3);
        vcdp->chgBit(c+34425,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+34433,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__size_r),3);
        vcdp->chgArray(c+34441,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[0]),199);
        vcdp->chgArray(c+34448,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[1]),199);
        vcdp->chgArray(c+34455,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[2]),199);
        vcdp->chgArray(c+34462,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[3]),199);
        vcdp->chgArray(c+34665,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),199);
        vcdp->chgArray(c+34721,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),199);
        vcdp->chgBus(c+34777,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+34785,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+34793,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+34801,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBit(c+34809,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+34817,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+34825,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+34833,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+34841,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgQuad(c+34849,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag),42);
        vcdp->chgBit(c+34865,((0x10U == (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBit(c+34873,((0xbU < (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+34881,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                      << 0xdU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                  >> 0x13U)))),2);
        vcdp->chgBus(c+34889,(((0x18fU >= (0x1ffU & 
                                           ((IData)(0x19U) 
                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                ? (0x1ffffffU & (((0U 
                                                   == 
                                                   (0x1fU 
                                                    & ((IData)(0x19U) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                   ? 0U
                                                   : 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                   ((IData)(1U) 
                                                    + 
                                                    (0xfU 
                                                     & (((IData)(0x19U) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U)))] 
                                                   << 
                                                   ((IData)(0x20U) 
                                                    - 
                                                    (0x1fU 
                                                     & ((IData)(0x19U) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                    (0xfU 
                                                     & (((IData)(0x19U) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U))] 
                                                    >> 
                                                    (0x1fU 
                                                     & ((IData)(0x19U) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                : 0U)),25);
        vcdp->chgBus(c+34897,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                      << 0x1eU) | (
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                   >> 2U)))),2);
        vcdp->chgBus(c+34905,(((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                            [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                            >> 0x15U))),32);
        vcdp->chgQuad(c+34913,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                    << 0x37U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U])) 
                                                  << 0x17U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])) 
                                                    >> 9U))))),42);
        vcdp->chgBus(c+34929,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                        << 0x1cU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                        >> 4U)))),4);
        vcdp->chgBit(c+34937,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                     [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                     >> 1U))));
        vcdp->chgBit(c+34945,((1U & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                               [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])));
        vcdp->chgBus(c+34953,((3U & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])),2);
        vcdp->chgQuad(c+34961,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U])) 
                                    << 0x39U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                                  << 0x19U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])) 
                                                    >> 7U))))),42);
        vcdp->chgBit(c+34977,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))));
        vcdp->chgBus(c+34985,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                        << 0x1eU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                        >> 2U)))),4);
        vcdp->chgBit(c+34993,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x14U))));
        vcdp->chgBit(c+35001,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x13U))));
        vcdp->chgBus(c+35009,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x18U)))),2);
        vcdp->chgBus(c+35017,(((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                          >> 0x18U))),32);
        vcdp->chgBus(c+35025,(((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                                          >> 0x18U))),32);
        __Vtemp1579[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                      >> 0x18U));
        __Vtemp1579[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                      >> 0x18U));
        __Vtemp1579[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                                      >> 0x18U));
        __Vtemp1579[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                                      >> 0x18U));
        vcdp->chgArray(c+35033,(__Vtemp1579),128);
        vcdp->chgBit(c+35065,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 2U))));
        vcdp->chgBit(c+35073,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 1U))));
        vcdp->chgBus(c+35081,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                           << 0xfU) 
                                          | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                             >> 0x11U)))),16);
        vcdp->chgQuad(c+35089,((VL_ULL(0x1ffffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U]))))),49);
        vcdp->chgBus(c+35105,((0x1fffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                             << 0x1dU) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                               >> 3U)))),21);
        vcdp->chgBit(c+35113,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x15U))));
        vcdp->chgBit(c+35121,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x17U))));
        vcdp->chgBit(c+35129,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x16U))));
        vcdp->chgBit(c+35137,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x18U))));
        vcdp->chgBit(c+35145,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU))));
        vcdp->chgBit(c+35153,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1bU))));
        vcdp->chgBit(c+35161,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x19U))));
        vcdp->chgBit(c+35169,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+35177,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+35185,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+35193,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBus(c+35201,(((0x1fffff0U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                              << 1U)) 
                               | (0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                           << 6U) | 
                                          (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                           >> 0x1aU))))),25);
        vcdp->chgBus(c+35209,((0xfffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                              << 0x19U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                                >> 7U)))),28);
        vcdp->chgBit(c+35217,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_dual_valid_sel));
        vcdp->chgBus(c+35225,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__size_r),5);
        vcdp->chgQuad(c+35233,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[0]),54);
        vcdp->chgQuad(c+35235,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[1]),54);
        vcdp->chgQuad(c+35237,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[2]),54);
        vcdp->chgQuad(c+35239,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[3]),54);
        vcdp->chgQuad(c+35241,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[4]),54);
        vcdp->chgQuad(c+35243,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[5]),54);
        vcdp->chgQuad(c+35245,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[6]),54);
        vcdp->chgQuad(c+35247,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[7]),54);
        vcdp->chgQuad(c+35249,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[8]),54);
        vcdp->chgQuad(c+35251,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[9]),54);
        vcdp->chgQuad(c+35253,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[10]),54);
        vcdp->chgQuad(c+35255,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[11]),54);
        vcdp->chgQuad(c+35257,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[12]),54);
        vcdp->chgQuad(c+35259,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[13]),54);
        vcdp->chgQuad(c+35261,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[14]),54);
        vcdp->chgQuad(c+35263,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[15]),54);
        vcdp->chgQuad(c+35489,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),54);
        vcdp->chgQuad(c+35505,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),54);
        vcdp->chgBus(c+35521,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+35529,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+35537,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+35545,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+35553,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__size_r),5);
        vcdp->chgArray(c+35561,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[0]),153);
        vcdp->chgArray(c+35566,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[1]),153);
        vcdp->chgArray(c+35571,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[2]),153);
        vcdp->chgArray(c+35576,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[3]),153);
        vcdp->chgArray(c+35581,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[4]),153);
        vcdp->chgArray(c+35586,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[5]),153);
        vcdp->chgArray(c+35591,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[6]),153);
        vcdp->chgArray(c+35596,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[7]),153);
        vcdp->chgArray(c+35601,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[8]),153);
        vcdp->chgArray(c+35606,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[9]),153);
        vcdp->chgArray(c+35611,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[10]),153);
        vcdp->chgArray(c+35616,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[11]),153);
        vcdp->chgArray(c+35621,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[12]),153);
        vcdp->chgArray(c+35626,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[13]),153);
        vcdp->chgArray(c+35631,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[14]),153);
        vcdp->chgArray(c+35636,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[15]),153);
        vcdp->chgArray(c+36201,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),153);
        vcdp->chgArray(c+36241,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),153);
        vcdp->chgBus(c+36281,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+36289,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+36297,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+36305,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+36313,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids),4);
        vcdp->chgBus(c+36321,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_rw),4);
        vcdp->chgBus(c+36329,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen),16);
        vcdp->chgArray(c+36337,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_addr),120);
        vcdp->chgArray(c+36369,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata),128);
        vcdp->chgBit(c+36401,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+36409,((1U & (~ (IData)((0U 
                                                != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids)))))));
        vcdp->chgBus(c+36417,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__size_r),4);
        vcdp->chgArray(c+36425,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[0]),314);
        vcdp->chgArray(c+36435,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[1]),314);
        vcdp->chgArray(c+36445,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[2]),314);
        vcdp->chgArray(c+36455,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[3]),314);
        vcdp->chgArray(c+36465,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[4]),314);
        vcdp->chgArray(c+36475,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[5]),314);
        vcdp->chgArray(c+36485,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[6]),314);
        vcdp->chgArray(c+36495,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[7]),314);
        vcdp->chgArray(c+37065,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),314);
        vcdp->chgArray(c+37145,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),314);
        vcdp->chgBus(c+37225,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),3);
        vcdp->chgBus(c+37233,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),3);
        vcdp->chgBus(c+37241,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),3);
        vcdp->chgBit(c+37249,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgArray(c+37257,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__s0_1_c0__DOT__value),242);
        __Vtemp1580[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][0U];
        __Vtemp1580[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][1U];
        __Vtemp1580[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][2U];
        __Vtemp1580[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][3U];
        vcdp->chgArray(c+37321,(__Vtemp1580),128);
        __Vtemp1581[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][0U];
        __Vtemp1581[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][1U];
        __Vtemp1581[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][2U];
        __Vtemp1581[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][3U];
        vcdp->chgArray(c+37353,(__Vtemp1581),128);
        __Vtemp1582[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][0U];
        __Vtemp1582[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][1U];
        __Vtemp1582[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][2U];
        __Vtemp1582[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][3U];
        vcdp->chgArray(c+37385,(__Vtemp1582),128);
        __Vtemp1583[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][0U];
        __Vtemp1583[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][1U];
        __Vtemp1583[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][2U];
        __Vtemp1583[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][3U];
        vcdp->chgArray(c+37417,(__Vtemp1583),128);
        __Vtemp1584[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][0U];
        __Vtemp1584[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][1U];
        __Vtemp1584[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][2U];
        __Vtemp1584[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][3U];
        vcdp->chgArray(c+37449,(__Vtemp1584),128);
        __Vtemp1585[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][0U];
        __Vtemp1585[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][1U];
        __Vtemp1585[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][2U];
        __Vtemp1585[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][3U];
        vcdp->chgArray(c+37481,(__Vtemp1585),128);
        __Vtemp1586[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][0U];
        __Vtemp1586[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][1U];
        __Vtemp1586[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][2U];
        __Vtemp1586[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][3U];
        vcdp->chgArray(c+37513,(__Vtemp1586),128);
        __Vtemp1587[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][0U];
        __Vtemp1587[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][1U];
        __Vtemp1587[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][2U];
        __Vtemp1587[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][3U];
        vcdp->chgArray(c+37545,(__Vtemp1587),128);
        __Vtemp1588[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][0U];
        __Vtemp1588[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][1U];
        __Vtemp1588[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][2U];
        __Vtemp1588[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][3U];
        vcdp->chgArray(c+37577,(__Vtemp1588),128);
        __Vtemp1589[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][0U];
        __Vtemp1589[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][1U];
        __Vtemp1589[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][2U];
        __Vtemp1589[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][3U];
        vcdp->chgArray(c+37609,(__Vtemp1589),128);
        __Vtemp1590[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][0U];
        __Vtemp1590[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][1U];
        __Vtemp1590[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][2U];
        __Vtemp1590[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][3U];
        vcdp->chgArray(c+37641,(__Vtemp1590),128);
        __Vtemp1591[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][0U];
        __Vtemp1591[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][1U];
        __Vtemp1591[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][2U];
        __Vtemp1591[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][3U];
        vcdp->chgArray(c+37673,(__Vtemp1591),128);
        __Vtemp1592[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][0U];
        __Vtemp1592[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][1U];
        __Vtemp1592[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][2U];
        __Vtemp1592[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][3U];
        vcdp->chgArray(c+37705,(__Vtemp1592),128);
        __Vtemp1593[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][0U];
        __Vtemp1593[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][1U];
        __Vtemp1593[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][2U];
        __Vtemp1593[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][3U];
        vcdp->chgArray(c+37737,(__Vtemp1593),128);
        __Vtemp1594[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][0U];
        __Vtemp1594[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][1U];
        __Vtemp1594[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][2U];
        __Vtemp1594[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][3U];
        vcdp->chgArray(c+37769,(__Vtemp1594),128);
        __Vtemp1595[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][0U];
        __Vtemp1595[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][1U];
        __Vtemp1595[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][2U];
        __Vtemp1595[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][3U];
        vcdp->chgArray(c+37801,(__Vtemp1595),128);
        vcdp->chgBus(c+37833,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[0]),21);
        vcdp->chgBus(c+37834,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[1]),21);
        vcdp->chgBus(c+37835,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[2]),21);
        vcdp->chgBus(c+37836,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[3]),21);
        vcdp->chgBus(c+37837,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[4]),21);
        vcdp->chgBus(c+37838,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[5]),21);
        vcdp->chgBus(c+37839,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[6]),21);
        vcdp->chgBus(c+37840,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[7]),21);
        vcdp->chgBus(c+37841,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[8]),21);
        vcdp->chgBus(c+37842,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[9]),21);
        vcdp->chgBus(c+37843,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[10]),21);
        vcdp->chgBus(c+37844,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[11]),21);
        vcdp->chgBus(c+37845,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[12]),21);
        vcdp->chgBus(c+37846,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[13]),21);
        vcdp->chgBus(c+37847,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[14]),21);
        vcdp->chgBus(c+37848,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[15]),21);
        vcdp->chgBus(c+37961,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0U]),16);
        vcdp->chgBus(c+37969,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [1U]),16);
        vcdp->chgBus(c+37977,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [2U]),16);
        vcdp->chgBus(c+37985,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [3U]),16);
        vcdp->chgBus(c+37993,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [4U]),16);
        vcdp->chgBus(c+38001,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [5U]),16);
        vcdp->chgBus(c+38009,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [6U]),16);
        vcdp->chgBus(c+38017,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [7U]),16);
        vcdp->chgBus(c+38025,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [8U]),16);
        vcdp->chgBus(c+38033,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [9U]),16);
        vcdp->chgBus(c+38041,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xaU]),16);
        vcdp->chgBus(c+38049,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xbU]),16);
        vcdp->chgBus(c+38057,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xcU]),16);
        vcdp->chgBus(c+38065,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xdU]),16);
        vcdp->chgBus(c+38073,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xeU]),16);
        vcdp->chgBus(c+38081,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xfU]),16);
        vcdp->chgBus(c+38089,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty),16);
        vcdp->chgBus(c+38097,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid),16);
        vcdp->chgBus(c+38105,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__i),32);
        vcdp->chgBus(c+38113,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__j),32);
        vcdp->chgArray(c+38121,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__s0_1_c0__DOT__value),167);
        vcdp->chgArray(c+38169,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value),316);
        vcdp->chgArray(c+38249,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[0]),85);
        vcdp->chgArray(c+38252,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[1]),85);
        vcdp->chgArray(c+38255,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[2]),85);
        vcdp->chgArray(c+38258,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[3]),85);
        vcdp->chgArray(c+38261,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[4]),85);
        vcdp->chgArray(c+38264,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[5]),85);
        vcdp->chgArray(c+38267,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[6]),85);
        vcdp->chgArray(c+38270,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[7]),85);
        vcdp->chgArray(c+38273,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[8]),85);
        vcdp->chgArray(c+38276,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[9]),85);
        vcdp->chgArray(c+38279,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[10]),85);
        vcdp->chgArray(c+38282,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[11]),85);
        vcdp->chgArray(c+38285,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[12]),85);
        vcdp->chgArray(c+38288,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[13]),85);
        vcdp->chgArray(c+38291,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[14]),85);
        vcdp->chgArray(c+38294,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[15]),85);
        vcdp->chgArray(c+38633,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table),400);
        vcdp->chgBus(c+38737,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_table),16);
        vcdp->chgBus(c+38745,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__ready_table),16);
        vcdp->chgBus(c+38753,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr),4);
        vcdp->chgBus(c+38761,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__head_ptr),4);
        vcdp->chgBus(c+38769,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr),4);
        vcdp->chgBus(c+38777,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size),5);
        vcdp->chgBit(c+38785,((0x10U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+38793,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__size_r),4);
        vcdp->chgArray(c+38801,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[0]),76);
        vcdp->chgArray(c+38804,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[1]),76);
        vcdp->chgArray(c+38807,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[2]),76);
        vcdp->chgArray(c+38810,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[3]),76);
        vcdp->chgArray(c+38813,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[4]),76);
        vcdp->chgArray(c+38816,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[5]),76);
        vcdp->chgArray(c+38819,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[6]),76);
        vcdp->chgArray(c+38822,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[7]),76);
        vcdp->chgArray(c+38993,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),76);
        vcdp->chgArray(c+39017,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),76);
        vcdp->chgBus(c+39041,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),3);
        vcdp->chgBus(c+39049,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),3);
        vcdp->chgBus(c+39057,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),3);
        vcdp->chgBit(c+39065,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+39073,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__size_r),3);
        vcdp->chgArray(c+39081,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[0]),199);
        vcdp->chgArray(c+39088,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[1]),199);
        vcdp->chgArray(c+39095,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[2]),199);
        vcdp->chgArray(c+39102,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[3]),199);
        vcdp->chgArray(c+39305,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),199);
        vcdp->chgArray(c+39361,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),199);
        vcdp->chgBus(c+39417,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+39425,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+39433,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+39441,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBit(c+39449,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+39457,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+39465,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+39473,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+39481,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgQuad(c+39489,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag),42);
        vcdp->chgBit(c+39505,((0x10U == (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBit(c+39513,((0xbU < (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+39521,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                      << 0xdU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                  >> 0x13U)))),2);
        vcdp->chgBus(c+39529,(((0x18fU >= (0x1ffU & 
                                           ((IData)(0x19U) 
                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                ? (0x1ffffffU & (((0U 
                                                   == 
                                                   (0x1fU 
                                                    & ((IData)(0x19U) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                   ? 0U
                                                   : 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                   ((IData)(1U) 
                                                    + 
                                                    (0xfU 
                                                     & (((IData)(0x19U) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U)))] 
                                                   << 
                                                   ((IData)(0x20U) 
                                                    - 
                                                    (0x1fU 
                                                     & ((IData)(0x19U) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                    (0xfU 
                                                     & (((IData)(0x19U) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U))] 
                                                    >> 
                                                    (0x1fU 
                                                     & ((IData)(0x19U) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                : 0U)),25);
        vcdp->chgBus(c+39537,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                      << 0x1eU) | (
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                   >> 2U)))),2);
        vcdp->chgBus(c+39545,(((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                            [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                            >> 0x15U))),32);
        vcdp->chgQuad(c+39553,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                    << 0x37U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U])) 
                                                  << 0x17U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])) 
                                                    >> 9U))))),42);
        vcdp->chgBus(c+39569,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                        << 0x1cU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                        >> 4U)))),4);
        vcdp->chgBit(c+39577,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                     [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                     >> 1U))));
        vcdp->chgBit(c+39585,((1U & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                               [vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])));
        vcdp->chgBus(c+39593,((3U & vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])),2);
        vcdp->chgQuad(c+39601,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U])) 
                                    << 0x39U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                                  << 0x19U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])) 
                                                    >> 7U))))),42);
        vcdp->chgBit(c+39617,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))));
        vcdp->chgBus(c+39625,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                        << 0x1eU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                        >> 2U)))),4);
        vcdp->chgBit(c+39633,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x14U))));
        vcdp->chgBit(c+39641,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x13U))));
        vcdp->chgBus(c+39649,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x18U)))),2);
        vcdp->chgBus(c+39657,(((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                          >> 0x18U))),32);
        vcdp->chgBus(c+39665,(((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                                          >> 0x18U))),32);
        __Vtemp1603[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                      >> 0x18U));
        __Vtemp1603[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                      >> 0x18U));
        __Vtemp1603[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                                      >> 0x18U));
        __Vtemp1603[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                                      >> 0x18U));
        vcdp->chgArray(c+39673,(__Vtemp1603),128);
        vcdp->chgBit(c+39705,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 2U))));
        vcdp->chgBit(c+39713,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 1U))));
        vcdp->chgBus(c+39721,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                           << 0xfU) 
                                          | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                             >> 0x11U)))),16);
        vcdp->chgQuad(c+39729,((VL_ULL(0x1ffffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U]))))),49);
        vcdp->chgBus(c+39745,((0x1fffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                             << 0x1dU) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                               >> 3U)))),21);
        vcdp->chgBit(c+39753,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x15U))));
        vcdp->chgBit(c+39761,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x17U))));
        vcdp->chgBit(c+39769,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x16U))));
        vcdp->chgBit(c+39777,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x18U))));
        vcdp->chgBit(c+39785,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU))));
        vcdp->chgBit(c+39793,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1bU))));
        vcdp->chgBit(c+39801,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x19U))));
        vcdp->chgBit(c+39809,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+39817,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+39825,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+39833,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBus(c+39841,(((0x1fffff0U & (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                              << 1U)) 
                               | (0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                           << 6U) | 
                                          (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                           >> 0x1aU))))),25);
        vcdp->chgBus(c+39849,((0xfffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                              << 0x19U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                                >> 7U)))),28);
        vcdp->chgBit(c+39857,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwbq_dual_valid_sel));
        vcdp->chgBus(c+39865,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__size_r),5);
        vcdp->chgQuad(c+39873,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[0]),54);
        vcdp->chgQuad(c+39875,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[1]),54);
        vcdp->chgQuad(c+39877,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[2]),54);
        vcdp->chgQuad(c+39879,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[3]),54);
        vcdp->chgQuad(c+39881,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[4]),54);
        vcdp->chgQuad(c+39883,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[5]),54);
        vcdp->chgQuad(c+39885,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[6]),54);
        vcdp->chgQuad(c+39887,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[7]),54);
        vcdp->chgQuad(c+39889,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[8]),54);
        vcdp->chgQuad(c+39891,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[9]),54);
        vcdp->chgQuad(c+39893,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[10]),54);
        vcdp->chgQuad(c+39895,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[11]),54);
        vcdp->chgQuad(c+39897,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[12]),54);
        vcdp->chgQuad(c+39899,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[13]),54);
        vcdp->chgQuad(c+39901,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[14]),54);
        vcdp->chgQuad(c+39903,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[15]),54);
        vcdp->chgQuad(c+40129,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),54);
        vcdp->chgQuad(c+40145,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),54);
        vcdp->chgBus(c+40161,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+40169,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+40177,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+40185,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+40193,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__size_r),5);
        vcdp->chgArray(c+40201,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[0]),153);
        vcdp->chgArray(c+40206,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[1]),153);
        vcdp->chgArray(c+40211,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[2]),153);
        vcdp->chgArray(c+40216,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[3]),153);
        vcdp->chgArray(c+40221,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[4]),153);
        vcdp->chgArray(c+40226,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[5]),153);
        vcdp->chgArray(c+40231,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[6]),153);
        vcdp->chgArray(c+40236,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[7]),153);
        vcdp->chgArray(c+40241,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[8]),153);
        vcdp->chgArray(c+40246,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[9]),153);
        vcdp->chgArray(c+40251,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[10]),153);
        vcdp->chgArray(c+40256,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[11]),153);
        vcdp->chgArray(c+40261,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[12]),153);
        vcdp->chgArray(c+40266,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[13]),153);
        vcdp->chgArray(c+40271,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[14]),153);
        vcdp->chgArray(c+40276,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[15]),153);
        vcdp->chgArray(c+40841,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),153);
        vcdp->chgArray(c+40881,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),153);
        vcdp->chgBus(c+40921,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+40929,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+40937,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+40945,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+40953,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids),4);
        vcdp->chgBus(c+40961,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_rw),4);
        vcdp->chgBus(c+40969,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen),16);
        vcdp->chgArray(c+40977,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_addr),120);
        vcdp->chgArray(c+41009,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata),128);
        vcdp->chgBit(c+41041,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+41049,((1U & (~ (IData)((0U 
                                                != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids)))))));
        vcdp->chgBus(c+41057,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__size_r),4);
        vcdp->chgArray(c+41065,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[0]),314);
        vcdp->chgArray(c+41075,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[1]),314);
        vcdp->chgArray(c+41085,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[2]),314);
        vcdp->chgArray(c+41095,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[3]),314);
        vcdp->chgArray(c+41105,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[4]),314);
        vcdp->chgArray(c+41115,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[5]),314);
        vcdp->chgArray(c+41125,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[6]),314);
        vcdp->chgArray(c+41135,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[7]),314);
        vcdp->chgArray(c+41705,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),314);
        vcdp->chgArray(c+41785,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),314);
        vcdp->chgBus(c+41865,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),3);
        vcdp->chgBus(c+41873,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),3);
        vcdp->chgBus(c+41881,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),3);
        vcdp->chgBit(c+41889,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgArray(c+41897,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__s0_1_c0__DOT__value),242);
        __Vtemp1604[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][0U];
        __Vtemp1604[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][1U];
        __Vtemp1604[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][2U];
        __Vtemp1604[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][3U];
        vcdp->chgArray(c+41961,(__Vtemp1604),128);
        __Vtemp1605[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][0U];
        __Vtemp1605[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][1U];
        __Vtemp1605[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][2U];
        __Vtemp1605[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][3U];
        vcdp->chgArray(c+41993,(__Vtemp1605),128);
        __Vtemp1606[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][0U];
        __Vtemp1606[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][1U];
        __Vtemp1606[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][2U];
        __Vtemp1606[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][3U];
        vcdp->chgArray(c+42025,(__Vtemp1606),128);
        __Vtemp1607[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][0U];
        __Vtemp1607[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][1U];
        __Vtemp1607[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][2U];
        __Vtemp1607[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][3U];
        vcdp->chgArray(c+42057,(__Vtemp1607),128);
        __Vtemp1608[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][0U];
        __Vtemp1608[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][1U];
        __Vtemp1608[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][2U];
        __Vtemp1608[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][3U];
        vcdp->chgArray(c+42089,(__Vtemp1608),128);
        __Vtemp1609[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][0U];
        __Vtemp1609[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][1U];
        __Vtemp1609[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][2U];
        __Vtemp1609[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][3U];
        vcdp->chgArray(c+42121,(__Vtemp1609),128);
        __Vtemp1610[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][0U];
        __Vtemp1610[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][1U];
        __Vtemp1610[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][2U];
        __Vtemp1610[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][3U];
        vcdp->chgArray(c+42153,(__Vtemp1610),128);
        __Vtemp1611[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][0U];
        __Vtemp1611[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][1U];
        __Vtemp1611[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][2U];
        __Vtemp1611[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][3U];
        vcdp->chgArray(c+42185,(__Vtemp1611),128);
        __Vtemp1612[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][0U];
        __Vtemp1612[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][1U];
        __Vtemp1612[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][2U];
        __Vtemp1612[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][3U];
        vcdp->chgArray(c+42217,(__Vtemp1612),128);
        __Vtemp1613[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][0U];
        __Vtemp1613[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][1U];
        __Vtemp1613[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][2U];
        __Vtemp1613[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][3U];
        vcdp->chgArray(c+42249,(__Vtemp1613),128);
        __Vtemp1614[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][0U];
        __Vtemp1614[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][1U];
        __Vtemp1614[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][2U];
        __Vtemp1614[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][3U];
        vcdp->chgArray(c+42281,(__Vtemp1614),128);
        __Vtemp1615[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][0U];
        __Vtemp1615[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][1U];
        __Vtemp1615[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][2U];
        __Vtemp1615[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][3U];
        vcdp->chgArray(c+42313,(__Vtemp1615),128);
        __Vtemp1616[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][0U];
        __Vtemp1616[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][1U];
        __Vtemp1616[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][2U];
        __Vtemp1616[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][3U];
        vcdp->chgArray(c+42345,(__Vtemp1616),128);
        __Vtemp1617[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][0U];
        __Vtemp1617[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][1U];
        __Vtemp1617[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][2U];
        __Vtemp1617[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][3U];
        vcdp->chgArray(c+42377,(__Vtemp1617),128);
        __Vtemp1618[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][0U];
        __Vtemp1618[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][1U];
        __Vtemp1618[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][2U];
        __Vtemp1618[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][3U];
        vcdp->chgArray(c+42409,(__Vtemp1618),128);
        __Vtemp1619[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][0U];
        __Vtemp1619[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][1U];
        __Vtemp1619[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][2U];
        __Vtemp1619[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][3U];
        vcdp->chgArray(c+42441,(__Vtemp1619),128);
        vcdp->chgBus(c+42473,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[0]),21);
        vcdp->chgBus(c+42474,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[1]),21);
        vcdp->chgBus(c+42475,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[2]),21);
        vcdp->chgBus(c+42476,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[3]),21);
        vcdp->chgBus(c+42477,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[4]),21);
        vcdp->chgBus(c+42478,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[5]),21);
        vcdp->chgBus(c+42479,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[6]),21);
        vcdp->chgBus(c+42480,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[7]),21);
        vcdp->chgBus(c+42481,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[8]),21);
        vcdp->chgBus(c+42482,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[9]),21);
        vcdp->chgBus(c+42483,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[10]),21);
        vcdp->chgBus(c+42484,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[11]),21);
        vcdp->chgBus(c+42485,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[12]),21);
        vcdp->chgBus(c+42486,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[13]),21);
        vcdp->chgBus(c+42487,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[14]),21);
        vcdp->chgBus(c+42488,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[15]),21);
        vcdp->chgBus(c+42601,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0U]),16);
        vcdp->chgBus(c+42609,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [1U]),16);
        vcdp->chgBus(c+42617,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [2U]),16);
        vcdp->chgBus(c+42625,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [3U]),16);
        vcdp->chgBus(c+42633,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [4U]),16);
        vcdp->chgBus(c+42641,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [5U]),16);
        vcdp->chgBus(c+42649,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [6U]),16);
        vcdp->chgBus(c+42657,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [7U]),16);
        vcdp->chgBus(c+42665,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [8U]),16);
        vcdp->chgBus(c+42673,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [9U]),16);
        vcdp->chgBus(c+42681,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xaU]),16);
        vcdp->chgBus(c+42689,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xbU]),16);
        vcdp->chgBus(c+42697,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xcU]),16);
        vcdp->chgBus(c+42705,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xdU]),16);
        vcdp->chgBus(c+42713,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xeU]),16);
        vcdp->chgBus(c+42721,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xfU]),16);
        vcdp->chgBus(c+42729,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty),16);
        vcdp->chgBus(c+42737,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid),16);
        vcdp->chgBus(c+42745,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__i),32);
        vcdp->chgBus(c+42753,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__j),32);
        vcdp->chgArray(c+42761,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__tag_data_access__DOT__s0_1_c0__DOT__value),167);
        vcdp->chgArray(c+42809,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__st_1e_2__DOT__value),316);
        vcdp->chgArray(c+42889,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[0]),85);
        vcdp->chgArray(c+42892,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[1]),85);
        vcdp->chgArray(c+42895,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[2]),85);
        vcdp->chgArray(c+42898,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[3]),85);
        vcdp->chgArray(c+42901,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[4]),85);
        vcdp->chgArray(c+42904,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[5]),85);
        vcdp->chgArray(c+42907,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[6]),85);
        vcdp->chgArray(c+42910,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[7]),85);
        vcdp->chgArray(c+42913,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[8]),85);
        vcdp->chgArray(c+42916,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[9]),85);
        vcdp->chgArray(c+42919,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[10]),85);
        vcdp->chgArray(c+42922,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[11]),85);
        vcdp->chgArray(c+42925,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[12]),85);
        vcdp->chgArray(c+42928,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[13]),85);
        vcdp->chgArray(c+42931,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[14]),85);
        vcdp->chgArray(c+42934,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[15]),85);
        vcdp->chgArray(c+43273,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table),400);
        vcdp->chgBus(c+43377,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_table),16);
        vcdp->chgBus(c+43385,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__ready_table),16);
        vcdp->chgBus(c+43393,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr),4);
        vcdp->chgBus(c+43401,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__head_ptr),4);
        vcdp->chgBus(c+43409,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr),4);
        vcdp->chgBus(c+43417,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size),5);
        vcdp->chgBit(c+43425,((0x10U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+43433,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__size_r),4);
        vcdp->chgArray(c+43441,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[0]),76);
        vcdp->chgArray(c+43444,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[1]),76);
        vcdp->chgArray(c+43447,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[2]),76);
        vcdp->chgArray(c+43450,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[3]),76);
        vcdp->chgArray(c+43453,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[4]),76);
        vcdp->chgArray(c+43456,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[5]),76);
        vcdp->chgArray(c+43459,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[6]),76);
        vcdp->chgArray(c+43462,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[7]),76);
        vcdp->chgArray(c+43633,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),76);
        vcdp->chgArray(c+43657,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),76);
        vcdp->chgBus(c+43681,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),3);
        vcdp->chgBus(c+43689,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),3);
        vcdp->chgBus(c+43697,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),3);
        vcdp->chgBit(c+43705,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+43713,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwb_queue__DOT__size_r),3);
        vcdp->chgArray(c+43721,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[0]),199);
        vcdp->chgArray(c+43728,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[1]),199);
        vcdp->chgArray(c+43735,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[2]),199);
        vcdp->chgArray(c+43742,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[3]),199);
        vcdp->chgArray(c+43945,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),199);
        vcdp->chgArray(c+44001,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),199);
        vcdp->chgBus(c+44057,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+44065,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+44073,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+44081,(vlTOPp->VX_cache__DOT__genblk5__BRA__4__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBit(c+44089,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+44097,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+44105,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+44113,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+44121,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgQuad(c+44129,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag),42);
        vcdp->chgBit(c+44145,((0x10U == (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBit(c+44153,((0xbU < (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+44161,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                      << 0xdU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                  >> 0x13U)))),2);
        vcdp->chgBus(c+44169,(((0x18fU >= (0x1ffU & 
                                           ((IData)(0x19U) 
                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                ? (0x1ffffffU & (((0U 
                                                   == 
                                                   (0x1fU 
                                                    & ((IData)(0x19U) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                   ? 0U
                                                   : 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                   ((IData)(1U) 
                                                    + 
                                                    (0xfU 
                                                     & (((IData)(0x19U) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U)))] 
                                                   << 
                                                   ((IData)(0x20U) 
                                                    - 
                                                    (0x1fU 
                                                     & ((IData)(0x19U) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                    (0xfU 
                                                     & (((IData)(0x19U) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U))] 
                                                    >> 
                                                    (0x1fU 
                                                     & ((IData)(0x19U) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                : 0U)),25);
        vcdp->chgBus(c+44177,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                      << 0x1eU) | (
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                   >> 2U)))),2);
        vcdp->chgBus(c+44185,(((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                            [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                            >> 0x15U))),32);
        vcdp->chgQuad(c+44193,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                    << 0x37U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U])) 
                                                  << 0x17U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])) 
                                                    >> 9U))))),42);
        vcdp->chgBus(c+44209,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                        << 0x1cU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                        >> 4U)))),4);
        vcdp->chgBit(c+44217,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                     [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                     >> 1U))));
        vcdp->chgBit(c+44225,((1U & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                               [vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])));
        vcdp->chgBus(c+44233,((3U & vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])),2);
        vcdp->chgQuad(c+44241,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U])) 
                                    << 0x39U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                                  << 0x19U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])) 
                                                    >> 7U))))),42);
        vcdp->chgBit(c+44257,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))));
        vcdp->chgBus(c+44265,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                        << 0x1eU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                        >> 2U)))),4);
        vcdp->chgBit(c+44273,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x14U))));
        vcdp->chgBit(c+44281,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x13U))));
        vcdp->chgBus(c+44289,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x18U)))),2);
        vcdp->chgBus(c+44297,(((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                          >> 0x18U))),32);
        vcdp->chgBus(c+44305,(((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                                          >> 0x18U))),32);
        __Vtemp1627[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                      >> 0x18U));
        __Vtemp1627[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                      >> 0x18U));
        __Vtemp1627[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                                      >> 0x18U));
        __Vtemp1627[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                                      >> 0x18U));
        vcdp->chgArray(c+44313,(__Vtemp1627),128);
        vcdp->chgBit(c+44345,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 2U))));
        vcdp->chgBit(c+44353,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 1U))));
        vcdp->chgBus(c+44361,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                           << 0xfU) 
                                          | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                             >> 0x11U)))),16);
        vcdp->chgQuad(c+44369,((VL_ULL(0x1ffffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U]))))),49);
        vcdp->chgBus(c+44385,((0x1fffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                             << 0x1dU) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                               >> 3U)))),21);
        vcdp->chgBit(c+44393,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x15U))));
        vcdp->chgBit(c+44401,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x17U))));
        vcdp->chgBit(c+44409,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x16U))));
        vcdp->chgBit(c+44417,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x18U))));
        vcdp->chgBit(c+44425,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU))));
        vcdp->chgBit(c+44433,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1bU))));
        vcdp->chgBit(c+44441,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x19U))));
        vcdp->chgBit(c+44449,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+44457,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+44465,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+44473,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBus(c+44481,(((0x1fffff0U & (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                              << 1U)) 
                               | (0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                           << 6U) | 
                                          (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                           >> 0x1aU))))),25);
        vcdp->chgBus(c+44489,((0xfffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                              << 0x19U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                                >> 7U)))),28);
        vcdp->chgBit(c+44497,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwbq_dual_valid_sel));
        vcdp->chgBus(c+44505,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__size_r),5);
        vcdp->chgQuad(c+44513,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[0]),54);
        vcdp->chgQuad(c+44515,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[1]),54);
        vcdp->chgQuad(c+44517,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[2]),54);
        vcdp->chgQuad(c+44519,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[3]),54);
        vcdp->chgQuad(c+44521,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[4]),54);
        vcdp->chgQuad(c+44523,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[5]),54);
        vcdp->chgQuad(c+44525,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[6]),54);
        vcdp->chgQuad(c+44527,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[7]),54);
        vcdp->chgQuad(c+44529,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[8]),54);
        vcdp->chgQuad(c+44531,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[9]),54);
        vcdp->chgQuad(c+44533,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[10]),54);
        vcdp->chgQuad(c+44535,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[11]),54);
        vcdp->chgQuad(c+44537,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[12]),54);
        vcdp->chgQuad(c+44539,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[13]),54);
        vcdp->chgQuad(c+44541,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[14]),54);
        vcdp->chgQuad(c+44543,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[15]),54);
        vcdp->chgQuad(c+44769,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),54);
        vcdp->chgQuad(c+44785,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),54);
        vcdp->chgBus(c+44801,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+44809,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+44817,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+44825,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+44833,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__size_r),5);
        vcdp->chgArray(c+44841,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[0]),153);
        vcdp->chgArray(c+44846,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[1]),153);
        vcdp->chgArray(c+44851,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[2]),153);
        vcdp->chgArray(c+44856,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[3]),153);
        vcdp->chgArray(c+44861,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[4]),153);
        vcdp->chgArray(c+44866,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[5]),153);
        vcdp->chgArray(c+44871,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[6]),153);
        vcdp->chgArray(c+44876,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[7]),153);
        vcdp->chgArray(c+44881,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[8]),153);
        vcdp->chgArray(c+44886,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[9]),153);
        vcdp->chgArray(c+44891,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[10]),153);
        vcdp->chgArray(c+44896,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[11]),153);
        vcdp->chgArray(c+44901,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[12]),153);
        vcdp->chgArray(c+44906,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[13]),153);
        vcdp->chgArray(c+44911,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[14]),153);
        vcdp->chgArray(c+44916,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[15]),153);
        vcdp->chgArray(c+45481,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),153);
        vcdp->chgArray(c+45521,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),153);
        vcdp->chgBus(c+45561,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+45569,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+45577,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+45585,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+45593,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids),4);
        vcdp->chgBus(c+45601,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_rw),4);
        vcdp->chgBus(c+45609,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen),16);
        vcdp->chgArray(c+45617,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_addr),120);
        vcdp->chgArray(c+45649,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata),128);
        vcdp->chgBit(c+45681,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+45689,((1U & (~ (IData)((0U 
                                                != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids)))))));
        vcdp->chgBus(c+45697,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__size_r),4);
        vcdp->chgArray(c+45705,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[0]),314);
        vcdp->chgArray(c+45715,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[1]),314);
        vcdp->chgArray(c+45725,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[2]),314);
        vcdp->chgArray(c+45735,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[3]),314);
        vcdp->chgArray(c+45745,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[4]),314);
        vcdp->chgArray(c+45755,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[5]),314);
        vcdp->chgArray(c+45765,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[6]),314);
        vcdp->chgArray(c+45775,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[7]),314);
        vcdp->chgArray(c+46345,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),314);
        vcdp->chgArray(c+46425,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),314);
        vcdp->chgBus(c+46505,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),3);
        vcdp->chgBus(c+46513,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),3);
        vcdp->chgBus(c+46521,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),3);
        vcdp->chgBit(c+46529,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgArray(c+46537,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__s0_1_c0__DOT__value),242);
        __Vtemp1628[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][0U];
        __Vtemp1628[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][1U];
        __Vtemp1628[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][2U];
        __Vtemp1628[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][3U];
        vcdp->chgArray(c+46601,(__Vtemp1628),128);
        __Vtemp1629[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][0U];
        __Vtemp1629[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][1U];
        __Vtemp1629[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][2U];
        __Vtemp1629[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][3U];
        vcdp->chgArray(c+46633,(__Vtemp1629),128);
        __Vtemp1630[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][0U];
        __Vtemp1630[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][1U];
        __Vtemp1630[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][2U];
        __Vtemp1630[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][3U];
        vcdp->chgArray(c+46665,(__Vtemp1630),128);
        __Vtemp1631[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][0U];
        __Vtemp1631[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][1U];
        __Vtemp1631[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][2U];
        __Vtemp1631[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][3U];
        vcdp->chgArray(c+46697,(__Vtemp1631),128);
        __Vtemp1632[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][0U];
        __Vtemp1632[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][1U];
        __Vtemp1632[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][2U];
        __Vtemp1632[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][3U];
        vcdp->chgArray(c+46729,(__Vtemp1632),128);
        __Vtemp1633[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][0U];
        __Vtemp1633[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][1U];
        __Vtemp1633[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][2U];
        __Vtemp1633[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][3U];
        vcdp->chgArray(c+46761,(__Vtemp1633),128);
        __Vtemp1634[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][0U];
        __Vtemp1634[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][1U];
        __Vtemp1634[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][2U];
        __Vtemp1634[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][3U];
        vcdp->chgArray(c+46793,(__Vtemp1634),128);
        __Vtemp1635[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][0U];
        __Vtemp1635[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][1U];
        __Vtemp1635[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][2U];
        __Vtemp1635[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][3U];
        vcdp->chgArray(c+46825,(__Vtemp1635),128);
        __Vtemp1636[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][0U];
        __Vtemp1636[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][1U];
        __Vtemp1636[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][2U];
        __Vtemp1636[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][3U];
        vcdp->chgArray(c+46857,(__Vtemp1636),128);
        __Vtemp1637[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][0U];
        __Vtemp1637[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][1U];
        __Vtemp1637[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][2U];
        __Vtemp1637[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][3U];
        vcdp->chgArray(c+46889,(__Vtemp1637),128);
        __Vtemp1638[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][0U];
        __Vtemp1638[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][1U];
        __Vtemp1638[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][2U];
        __Vtemp1638[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][3U];
        vcdp->chgArray(c+46921,(__Vtemp1638),128);
        __Vtemp1639[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][0U];
        __Vtemp1639[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][1U];
        __Vtemp1639[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][2U];
        __Vtemp1639[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][3U];
        vcdp->chgArray(c+46953,(__Vtemp1639),128);
        __Vtemp1640[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][0U];
        __Vtemp1640[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][1U];
        __Vtemp1640[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][2U];
        __Vtemp1640[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][3U];
        vcdp->chgArray(c+46985,(__Vtemp1640),128);
        __Vtemp1641[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][0U];
        __Vtemp1641[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][1U];
        __Vtemp1641[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][2U];
        __Vtemp1641[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][3U];
        vcdp->chgArray(c+47017,(__Vtemp1641),128);
        __Vtemp1642[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][0U];
        __Vtemp1642[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][1U];
        __Vtemp1642[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][2U];
        __Vtemp1642[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][3U];
        vcdp->chgArray(c+47049,(__Vtemp1642),128);
        __Vtemp1643[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][0U];
        __Vtemp1643[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][1U];
        __Vtemp1643[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][2U];
        __Vtemp1643[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][3U];
        vcdp->chgArray(c+47081,(__Vtemp1643),128);
        vcdp->chgBus(c+47113,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[0]),21);
        vcdp->chgBus(c+47114,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[1]),21);
        vcdp->chgBus(c+47115,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[2]),21);
        vcdp->chgBus(c+47116,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[3]),21);
        vcdp->chgBus(c+47117,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[4]),21);
        vcdp->chgBus(c+47118,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[5]),21);
        vcdp->chgBus(c+47119,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[6]),21);
        vcdp->chgBus(c+47120,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[7]),21);
        vcdp->chgBus(c+47121,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[8]),21);
        vcdp->chgBus(c+47122,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[9]),21);
        vcdp->chgBus(c+47123,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[10]),21);
        vcdp->chgBus(c+47124,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[11]),21);
        vcdp->chgBus(c+47125,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[12]),21);
        vcdp->chgBus(c+47126,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[13]),21);
        vcdp->chgBus(c+47127,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[14]),21);
        vcdp->chgBus(c+47128,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[15]),21);
        vcdp->chgBus(c+47241,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0U]),16);
        vcdp->chgBus(c+47249,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [1U]),16);
        vcdp->chgBus(c+47257,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [2U]),16);
        vcdp->chgBus(c+47265,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [3U]),16);
        vcdp->chgBus(c+47273,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [4U]),16);
        vcdp->chgBus(c+47281,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [5U]),16);
        vcdp->chgBus(c+47289,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [6U]),16);
        vcdp->chgBus(c+47297,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [7U]),16);
        vcdp->chgBus(c+47305,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [8U]),16);
        vcdp->chgBus(c+47313,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [9U]),16);
        vcdp->chgBus(c+47321,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xaU]),16);
        vcdp->chgBus(c+47329,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xbU]),16);
        vcdp->chgBus(c+47337,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xcU]),16);
        vcdp->chgBus(c+47345,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xdU]),16);
        vcdp->chgBus(c+47353,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xeU]),16);
        vcdp->chgBus(c+47361,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xfU]),16);
        vcdp->chgBus(c+47369,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty),16);
        vcdp->chgBus(c+47377,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid),16);
        vcdp->chgBus(c+47385,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__i),32);
        vcdp->chgBus(c+47393,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__j),32);
        vcdp->chgArray(c+47401,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__tag_data_access__DOT__s0_1_c0__DOT__value),167);
        vcdp->chgArray(c+47449,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__st_1e_2__DOT__value),316);
        vcdp->chgArray(c+47529,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[0]),85);
        vcdp->chgArray(c+47532,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[1]),85);
        vcdp->chgArray(c+47535,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[2]),85);
        vcdp->chgArray(c+47538,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[3]),85);
        vcdp->chgArray(c+47541,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[4]),85);
        vcdp->chgArray(c+47544,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[5]),85);
        vcdp->chgArray(c+47547,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[6]),85);
        vcdp->chgArray(c+47550,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[7]),85);
        vcdp->chgArray(c+47553,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[8]),85);
        vcdp->chgArray(c+47556,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[9]),85);
        vcdp->chgArray(c+47559,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[10]),85);
        vcdp->chgArray(c+47562,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[11]),85);
        vcdp->chgArray(c+47565,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[12]),85);
        vcdp->chgArray(c+47568,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[13]),85);
        vcdp->chgArray(c+47571,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[14]),85);
        vcdp->chgArray(c+47574,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[15]),85);
        vcdp->chgArray(c+47913,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table),400);
        vcdp->chgBus(c+48017,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_table),16);
        vcdp->chgBus(c+48025,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__ready_table),16);
        vcdp->chgBus(c+48033,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr),4);
        vcdp->chgBus(c+48041,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__head_ptr),4);
        vcdp->chgBus(c+48049,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr),4);
        vcdp->chgBus(c+48057,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size),5);
        vcdp->chgBit(c+48065,((0x10U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+48073,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__size_r),4);
        vcdp->chgArray(c+48081,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[0]),76);
        vcdp->chgArray(c+48084,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[1]),76);
        vcdp->chgArray(c+48087,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[2]),76);
        vcdp->chgArray(c+48090,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[3]),76);
        vcdp->chgArray(c+48093,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[4]),76);
        vcdp->chgArray(c+48096,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[5]),76);
        vcdp->chgArray(c+48099,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[6]),76);
        vcdp->chgArray(c+48102,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[7]),76);
        vcdp->chgArray(c+48273,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),76);
        vcdp->chgArray(c+48297,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),76);
        vcdp->chgBus(c+48321,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),3);
        vcdp->chgBus(c+48329,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),3);
        vcdp->chgBus(c+48337,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),3);
        vcdp->chgBit(c+48345,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+48353,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwb_queue__DOT__size_r),3);
        vcdp->chgArray(c+48361,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[0]),199);
        vcdp->chgArray(c+48368,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[1]),199);
        vcdp->chgArray(c+48375,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[2]),199);
        vcdp->chgArray(c+48382,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[3]),199);
        vcdp->chgArray(c+48585,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),199);
        vcdp->chgArray(c+48641,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),199);
        vcdp->chgBus(c+48697,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+48705,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+48713,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+48721,(vlTOPp->VX_cache__DOT__genblk5__BRA__5__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBit(c+48729,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+48737,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+48745,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+48753,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+48761,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgQuad(c+48769,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag),42);
        vcdp->chgBit(c+48785,((0x10U == (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBit(c+48793,((0xbU < (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+48801,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                      << 0xdU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                  >> 0x13U)))),2);
        vcdp->chgBus(c+48809,(((0x18fU >= (0x1ffU & 
                                           ((IData)(0x19U) 
                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                ? (0x1ffffffU & (((0U 
                                                   == 
                                                   (0x1fU 
                                                    & ((IData)(0x19U) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                   ? 0U
                                                   : 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                   ((IData)(1U) 
                                                    + 
                                                    (0xfU 
                                                     & (((IData)(0x19U) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U)))] 
                                                   << 
                                                   ((IData)(0x20U) 
                                                    - 
                                                    (0x1fU 
                                                     & ((IData)(0x19U) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                    (0xfU 
                                                     & (((IData)(0x19U) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U))] 
                                                    >> 
                                                    (0x1fU 
                                                     & ((IData)(0x19U) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                : 0U)),25);
        vcdp->chgBus(c+48817,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                      << 0x1eU) | (
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                   >> 2U)))),2);
        vcdp->chgBus(c+48825,(((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                            [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                            >> 0x15U))),32);
        vcdp->chgQuad(c+48833,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                    << 0x37U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U])) 
                                                  << 0x17U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])) 
                                                    >> 9U))))),42);
        vcdp->chgBus(c+48849,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                        << 0x1cU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                        >> 4U)))),4);
        vcdp->chgBit(c+48857,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                     [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                     >> 1U))));
        vcdp->chgBit(c+48865,((1U & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                               [vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])));
        vcdp->chgBus(c+48873,((3U & vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])),2);
        vcdp->chgQuad(c+48881,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U])) 
                                    << 0x39U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                                  << 0x19U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])) 
                                                    >> 7U))))),42);
        vcdp->chgBit(c+48897,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))));
        vcdp->chgBus(c+48905,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                        << 0x1eU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                        >> 2U)))),4);
        vcdp->chgBit(c+48913,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x14U))));
        vcdp->chgBit(c+48921,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x13U))));
        vcdp->chgBus(c+48929,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x18U)))),2);
        vcdp->chgBus(c+48937,(((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                          >> 0x18U))),32);
        vcdp->chgBus(c+48945,(((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                                          >> 0x18U))),32);
        __Vtemp1651[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                      >> 0x18U));
        __Vtemp1651[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                      >> 0x18U));
        __Vtemp1651[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                                      >> 0x18U));
        __Vtemp1651[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                                      >> 0x18U));
        vcdp->chgArray(c+48953,(__Vtemp1651),128);
        vcdp->chgBit(c+48985,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 2U))));
        vcdp->chgBit(c+48993,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 1U))));
        vcdp->chgBus(c+49001,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                           << 0xfU) 
                                          | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                             >> 0x11U)))),16);
        vcdp->chgQuad(c+49009,((VL_ULL(0x1ffffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U]))))),49);
        vcdp->chgBus(c+49025,((0x1fffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                             << 0x1dU) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                               >> 3U)))),21);
        vcdp->chgBit(c+49033,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x15U))));
        vcdp->chgBit(c+49041,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x17U))));
        vcdp->chgBit(c+49049,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x16U))));
        vcdp->chgBit(c+49057,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x18U))));
        vcdp->chgBit(c+49065,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU))));
        vcdp->chgBit(c+49073,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1bU))));
        vcdp->chgBit(c+49081,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x19U))));
        vcdp->chgBit(c+49089,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+49097,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+49105,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+49113,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBus(c+49121,(((0x1fffff0U & (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                              << 1U)) 
                               | (0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                           << 6U) | 
                                          (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                           >> 0x1aU))))),25);
        vcdp->chgBus(c+49129,((0xfffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                              << 0x19U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                                >> 7U)))),28);
        vcdp->chgBit(c+49137,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwbq_dual_valid_sel));
        vcdp->chgBus(c+49145,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__size_r),5);
        vcdp->chgQuad(c+49153,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[0]),54);
        vcdp->chgQuad(c+49155,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[1]),54);
        vcdp->chgQuad(c+49157,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[2]),54);
        vcdp->chgQuad(c+49159,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[3]),54);
        vcdp->chgQuad(c+49161,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[4]),54);
        vcdp->chgQuad(c+49163,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[5]),54);
        vcdp->chgQuad(c+49165,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[6]),54);
        vcdp->chgQuad(c+49167,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[7]),54);
        vcdp->chgQuad(c+49169,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[8]),54);
        vcdp->chgQuad(c+49171,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[9]),54);
        vcdp->chgQuad(c+49173,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[10]),54);
        vcdp->chgQuad(c+49175,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[11]),54);
        vcdp->chgQuad(c+49177,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[12]),54);
        vcdp->chgQuad(c+49179,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[13]),54);
        vcdp->chgQuad(c+49181,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[14]),54);
        vcdp->chgQuad(c+49183,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[15]),54);
        vcdp->chgQuad(c+49409,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),54);
        vcdp->chgQuad(c+49425,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),54);
        vcdp->chgBus(c+49441,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+49449,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+49457,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+49465,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+49473,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__size_r),5);
        vcdp->chgArray(c+49481,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[0]),153);
        vcdp->chgArray(c+49486,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[1]),153);
        vcdp->chgArray(c+49491,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[2]),153);
        vcdp->chgArray(c+49496,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[3]),153);
        vcdp->chgArray(c+49501,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[4]),153);
        vcdp->chgArray(c+49506,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[5]),153);
        vcdp->chgArray(c+49511,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[6]),153);
        vcdp->chgArray(c+49516,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[7]),153);
        vcdp->chgArray(c+49521,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[8]),153);
        vcdp->chgArray(c+49526,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[9]),153);
        vcdp->chgArray(c+49531,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[10]),153);
        vcdp->chgArray(c+49536,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[11]),153);
        vcdp->chgArray(c+49541,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[12]),153);
        vcdp->chgArray(c+49546,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[13]),153);
        vcdp->chgArray(c+49551,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[14]),153);
        vcdp->chgArray(c+49556,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[15]),153);
        vcdp->chgArray(c+50121,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),153);
        vcdp->chgArray(c+50161,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),153);
        vcdp->chgBus(c+50201,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+50209,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+50217,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+50225,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+50233,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids),4);
        vcdp->chgBus(c+50241,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_rw),4);
        vcdp->chgBus(c+50249,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen),16);
        vcdp->chgArray(c+50257,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_addr),120);
        vcdp->chgArray(c+50289,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata),128);
        vcdp->chgBit(c+50321,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+50329,((1U & (~ (IData)((0U 
                                                != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids)))))));
        vcdp->chgBus(c+50337,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__size_r),4);
        vcdp->chgArray(c+50345,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[0]),314);
        vcdp->chgArray(c+50355,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[1]),314);
        vcdp->chgArray(c+50365,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[2]),314);
        vcdp->chgArray(c+50375,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[3]),314);
        vcdp->chgArray(c+50385,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[4]),314);
        vcdp->chgArray(c+50395,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[5]),314);
        vcdp->chgArray(c+50405,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[6]),314);
        vcdp->chgArray(c+50415,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[7]),314);
        vcdp->chgArray(c+50985,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),314);
        vcdp->chgArray(c+51065,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),314);
        vcdp->chgBus(c+51145,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),3);
        vcdp->chgBus(c+51153,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),3);
        vcdp->chgBus(c+51161,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),3);
        vcdp->chgBit(c+51169,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgArray(c+51177,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__s0_1_c0__DOT__value),242);
        __Vtemp1652[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][0U];
        __Vtemp1652[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][1U];
        __Vtemp1652[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][2U];
        __Vtemp1652[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][3U];
        vcdp->chgArray(c+51241,(__Vtemp1652),128);
        __Vtemp1653[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][0U];
        __Vtemp1653[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][1U];
        __Vtemp1653[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][2U];
        __Vtemp1653[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][3U];
        vcdp->chgArray(c+51273,(__Vtemp1653),128);
        __Vtemp1654[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][0U];
        __Vtemp1654[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][1U];
        __Vtemp1654[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][2U];
        __Vtemp1654[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][3U];
        vcdp->chgArray(c+51305,(__Vtemp1654),128);
        __Vtemp1655[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][0U];
        __Vtemp1655[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][1U];
        __Vtemp1655[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][2U];
        __Vtemp1655[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][3U];
        vcdp->chgArray(c+51337,(__Vtemp1655),128);
        __Vtemp1656[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][0U];
        __Vtemp1656[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][1U];
        __Vtemp1656[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][2U];
        __Vtemp1656[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][3U];
        vcdp->chgArray(c+51369,(__Vtemp1656),128);
        __Vtemp1657[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][0U];
        __Vtemp1657[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][1U];
        __Vtemp1657[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][2U];
        __Vtemp1657[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][3U];
        vcdp->chgArray(c+51401,(__Vtemp1657),128);
        __Vtemp1658[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][0U];
        __Vtemp1658[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][1U];
        __Vtemp1658[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][2U];
        __Vtemp1658[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][3U];
        vcdp->chgArray(c+51433,(__Vtemp1658),128);
        __Vtemp1659[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][0U];
        __Vtemp1659[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][1U];
        __Vtemp1659[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][2U];
        __Vtemp1659[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][3U];
        vcdp->chgArray(c+51465,(__Vtemp1659),128);
        __Vtemp1660[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][0U];
        __Vtemp1660[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][1U];
        __Vtemp1660[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][2U];
        __Vtemp1660[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][3U];
        vcdp->chgArray(c+51497,(__Vtemp1660),128);
        __Vtemp1661[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][0U];
        __Vtemp1661[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][1U];
        __Vtemp1661[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][2U];
        __Vtemp1661[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][3U];
        vcdp->chgArray(c+51529,(__Vtemp1661),128);
        __Vtemp1662[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][0U];
        __Vtemp1662[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][1U];
        __Vtemp1662[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][2U];
        __Vtemp1662[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][3U];
        vcdp->chgArray(c+51561,(__Vtemp1662),128);
        __Vtemp1663[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][0U];
        __Vtemp1663[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][1U];
        __Vtemp1663[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][2U];
        __Vtemp1663[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][3U];
        vcdp->chgArray(c+51593,(__Vtemp1663),128);
        __Vtemp1664[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][0U];
        __Vtemp1664[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][1U];
        __Vtemp1664[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][2U];
        __Vtemp1664[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][3U];
        vcdp->chgArray(c+51625,(__Vtemp1664),128);
        __Vtemp1665[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][0U];
        __Vtemp1665[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][1U];
        __Vtemp1665[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][2U];
        __Vtemp1665[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][3U];
        vcdp->chgArray(c+51657,(__Vtemp1665),128);
        __Vtemp1666[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][0U];
        __Vtemp1666[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][1U];
        __Vtemp1666[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][2U];
        __Vtemp1666[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][3U];
        vcdp->chgArray(c+51689,(__Vtemp1666),128);
        __Vtemp1667[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][0U];
        __Vtemp1667[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][1U];
        __Vtemp1667[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][2U];
        __Vtemp1667[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][3U];
        vcdp->chgArray(c+51721,(__Vtemp1667),128);
        vcdp->chgBus(c+51753,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[0]),21);
        vcdp->chgBus(c+51754,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[1]),21);
        vcdp->chgBus(c+51755,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[2]),21);
        vcdp->chgBus(c+51756,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[3]),21);
        vcdp->chgBus(c+51757,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[4]),21);
        vcdp->chgBus(c+51758,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[5]),21);
        vcdp->chgBus(c+51759,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[6]),21);
        vcdp->chgBus(c+51760,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[7]),21);
        vcdp->chgBus(c+51761,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[8]),21);
        vcdp->chgBus(c+51762,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[9]),21);
        vcdp->chgBus(c+51763,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[10]),21);
        vcdp->chgBus(c+51764,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[11]),21);
        vcdp->chgBus(c+51765,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[12]),21);
        vcdp->chgBus(c+51766,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[13]),21);
        vcdp->chgBus(c+51767,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[14]),21);
        vcdp->chgBus(c+51768,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[15]),21);
        vcdp->chgBus(c+51881,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0U]),16);
        vcdp->chgBus(c+51889,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [1U]),16);
        vcdp->chgBus(c+51897,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [2U]),16);
        vcdp->chgBus(c+51905,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [3U]),16);
        vcdp->chgBus(c+51913,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [4U]),16);
        vcdp->chgBus(c+51921,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [5U]),16);
        vcdp->chgBus(c+51929,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [6U]),16);
        vcdp->chgBus(c+51937,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [7U]),16);
        vcdp->chgBus(c+51945,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [8U]),16);
        vcdp->chgBus(c+51953,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [9U]),16);
        vcdp->chgBus(c+51961,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xaU]),16);
        vcdp->chgBus(c+51969,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xbU]),16);
        vcdp->chgBus(c+51977,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xcU]),16);
        vcdp->chgBus(c+51985,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xdU]),16);
        vcdp->chgBus(c+51993,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xeU]),16);
        vcdp->chgBus(c+52001,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xfU]),16);
        vcdp->chgBus(c+52009,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty),16);
        vcdp->chgBus(c+52017,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid),16);
        vcdp->chgBus(c+52025,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__i),32);
        vcdp->chgBus(c+52033,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__j),32);
        vcdp->chgArray(c+52041,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__tag_data_access__DOT__s0_1_c0__DOT__value),167);
        vcdp->chgArray(c+52089,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__st_1e_2__DOT__value),316);
        vcdp->chgArray(c+52169,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[0]),85);
        vcdp->chgArray(c+52172,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[1]),85);
        vcdp->chgArray(c+52175,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[2]),85);
        vcdp->chgArray(c+52178,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[3]),85);
        vcdp->chgArray(c+52181,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[4]),85);
        vcdp->chgArray(c+52184,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[5]),85);
        vcdp->chgArray(c+52187,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[6]),85);
        vcdp->chgArray(c+52190,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[7]),85);
        vcdp->chgArray(c+52193,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[8]),85);
        vcdp->chgArray(c+52196,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[9]),85);
        vcdp->chgArray(c+52199,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[10]),85);
        vcdp->chgArray(c+52202,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[11]),85);
        vcdp->chgArray(c+52205,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[12]),85);
        vcdp->chgArray(c+52208,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[13]),85);
        vcdp->chgArray(c+52211,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[14]),85);
        vcdp->chgArray(c+52214,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[15]),85);
        vcdp->chgArray(c+52553,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table),400);
        vcdp->chgBus(c+52657,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_table),16);
        vcdp->chgBus(c+52665,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__ready_table),16);
        vcdp->chgBus(c+52673,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr),4);
        vcdp->chgBus(c+52681,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__head_ptr),4);
        vcdp->chgBus(c+52689,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr),4);
        vcdp->chgBus(c+52697,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size),5);
        vcdp->chgBit(c+52705,((0x10U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+52713,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__size_r),4);
        vcdp->chgArray(c+52721,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[0]),76);
        vcdp->chgArray(c+52724,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[1]),76);
        vcdp->chgArray(c+52727,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[2]),76);
        vcdp->chgArray(c+52730,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[3]),76);
        vcdp->chgArray(c+52733,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[4]),76);
        vcdp->chgArray(c+52736,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[5]),76);
        vcdp->chgArray(c+52739,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[6]),76);
        vcdp->chgArray(c+52742,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[7]),76);
        vcdp->chgArray(c+52913,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),76);
        vcdp->chgArray(c+52937,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),76);
        vcdp->chgBus(c+52961,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),3);
        vcdp->chgBus(c+52969,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),3);
        vcdp->chgBus(c+52977,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),3);
        vcdp->chgBit(c+52985,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+52993,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwb_queue__DOT__size_r),3);
        vcdp->chgArray(c+53001,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[0]),199);
        vcdp->chgArray(c+53008,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[1]),199);
        vcdp->chgArray(c+53015,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[2]),199);
        vcdp->chgArray(c+53022,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[3]),199);
        vcdp->chgArray(c+53225,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),199);
        vcdp->chgArray(c+53281,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),199);
        vcdp->chgBus(c+53337,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+53345,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+53353,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+53361,(vlTOPp->VX_cache__DOT__genblk5__BRA__6__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBit(c+53369,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+53377,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+53385,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+53393,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+53401,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgQuad(c+53409,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag),42);
        vcdp->chgBit(c+53425,((0x10U == (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBit(c+53433,((0xbU < (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+53441,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                      << 0xdU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                  >> 0x13U)))),2);
        vcdp->chgBus(c+53449,(((0x18fU >= (0x1ffU & 
                                           ((IData)(0x19U) 
                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                ? (0x1ffffffU & (((0U 
                                                   == 
                                                   (0x1fU 
                                                    & ((IData)(0x19U) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                   ? 0U
                                                   : 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                   ((IData)(1U) 
                                                    + 
                                                    (0xfU 
                                                     & (((IData)(0x19U) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U)))] 
                                                   << 
                                                   ((IData)(0x20U) 
                                                    - 
                                                    (0x1fU 
                                                     & ((IData)(0x19U) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                    (0xfU 
                                                     & (((IData)(0x19U) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U))] 
                                                    >> 
                                                    (0x1fU 
                                                     & ((IData)(0x19U) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                : 0U)),25);
        vcdp->chgBus(c+53457,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                      << 0x1eU) | (
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                   >> 2U)))),2);
        vcdp->chgBus(c+53465,(((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                            [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                            >> 0x15U))),32);
        vcdp->chgQuad(c+53473,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                    << 0x37U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U])) 
                                                  << 0x17U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])) 
                                                    >> 9U))))),42);
        vcdp->chgBus(c+53489,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                        << 0x1cU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                        >> 4U)))),4);
        vcdp->chgBit(c+53497,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                     [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                     >> 1U))));
        vcdp->chgBit(c+53505,((1U & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                               [vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])));
        vcdp->chgBus(c+53513,((3U & vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])),2);
        vcdp->chgQuad(c+53521,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U])) 
                                    << 0x39U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                                  << 0x19U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])) 
                                                    >> 7U))))),42);
        vcdp->chgBit(c+53537,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))));
        vcdp->chgBus(c+53545,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                        << 0x1eU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                        >> 2U)))),4);
        vcdp->chgBit(c+53553,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x14U))));
        vcdp->chgBit(c+53561,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x13U))));
        vcdp->chgBus(c+53569,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x18U)))),2);
        vcdp->chgBus(c+53577,(((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                          >> 0x18U))),32);
        vcdp->chgBus(c+53585,(((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                                          >> 0x18U))),32);
        __Vtemp1675[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                      >> 0x18U));
        __Vtemp1675[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                      >> 0x18U));
        __Vtemp1675[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                                      >> 0x18U));
        __Vtemp1675[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                            << 8U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                                      >> 0x18U));
        vcdp->chgArray(c+53593,(__Vtemp1675),128);
        vcdp->chgBit(c+53625,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 2U))));
        vcdp->chgBit(c+53633,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 1U))));
        vcdp->chgBus(c+53641,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                           << 0xfU) 
                                          | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                             >> 0x11U)))),16);
        vcdp->chgQuad(c+53649,((VL_ULL(0x1ffffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U]))))),49);
        vcdp->chgBus(c+53665,((0x1fffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                             << 0x1dU) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                               >> 3U)))),21);
        vcdp->chgBit(c+53673,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x15U))));
        vcdp->chgBit(c+53681,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x17U))));
        vcdp->chgBit(c+53689,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x16U))));
        vcdp->chgBit(c+53697,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x18U))));
        vcdp->chgBit(c+53705,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU))));
        vcdp->chgBit(c+53713,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1bU))));
        vcdp->chgBit(c+53721,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x19U))));
        vcdp->chgBit(c+53729,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+53737,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+53745,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+53753,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBus(c+53761,(((0x1fffff0U & (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                              << 1U)) 
                               | (0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                           << 6U) | 
                                          (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                           >> 0x1aU))))),25);
        vcdp->chgBus(c+53769,((0xfffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                              << 0x19U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                                >> 7U)))),28);
        vcdp->chgBit(c+53777,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwbq_dual_valid_sel));
        vcdp->chgBus(c+53785,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__size_r),5);
        vcdp->chgQuad(c+53793,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[0]),54);
        vcdp->chgQuad(c+53795,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[1]),54);
        vcdp->chgQuad(c+53797,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[2]),54);
        vcdp->chgQuad(c+53799,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[3]),54);
        vcdp->chgQuad(c+53801,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[4]),54);
        vcdp->chgQuad(c+53803,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[5]),54);
        vcdp->chgQuad(c+53805,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[6]),54);
        vcdp->chgQuad(c+53807,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[7]),54);
        vcdp->chgQuad(c+53809,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[8]),54);
        vcdp->chgQuad(c+53811,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[9]),54);
        vcdp->chgQuad(c+53813,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[10]),54);
        vcdp->chgQuad(c+53815,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[11]),54);
        vcdp->chgQuad(c+53817,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[12]),54);
        vcdp->chgQuad(c+53819,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[13]),54);
        vcdp->chgQuad(c+53821,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[14]),54);
        vcdp->chgQuad(c+53823,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[15]),54);
        vcdp->chgQuad(c+54049,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),54);
        vcdp->chgQuad(c+54065,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),54);
        vcdp->chgBus(c+54081,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+54089,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+54097,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+54105,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+54113,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__size_r),5);
        vcdp->chgArray(c+54121,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[0]),153);
        vcdp->chgArray(c+54126,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[1]),153);
        vcdp->chgArray(c+54131,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[2]),153);
        vcdp->chgArray(c+54136,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[3]),153);
        vcdp->chgArray(c+54141,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[4]),153);
        vcdp->chgArray(c+54146,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[5]),153);
        vcdp->chgArray(c+54151,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[6]),153);
        vcdp->chgArray(c+54156,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[7]),153);
        vcdp->chgArray(c+54161,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[8]),153);
        vcdp->chgArray(c+54166,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[9]),153);
        vcdp->chgArray(c+54171,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[10]),153);
        vcdp->chgArray(c+54176,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[11]),153);
        vcdp->chgArray(c+54181,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[12]),153);
        vcdp->chgArray(c+54186,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[13]),153);
        vcdp->chgArray(c+54191,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[14]),153);
        vcdp->chgArray(c+54196,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[15]),153);
        vcdp->chgArray(c+54761,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),153);
        vcdp->chgArray(c+54801,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),153);
        vcdp->chgBus(c+54841,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+54849,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+54857,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+54865,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+54873,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids),4);
        vcdp->chgBus(c+54881,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_rw),4);
        vcdp->chgBus(c+54889,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen),16);
        vcdp->chgArray(c+54897,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_addr),120);
        vcdp->chgArray(c+54929,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata),128);
        vcdp->chgBit(c+54961,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+54969,((1U & (~ (IData)((0U 
                                                != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids)))))));
        vcdp->chgBus(c+54977,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__size_r),4);
        vcdp->chgArray(c+54985,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[0]),314);
        vcdp->chgArray(c+54995,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[1]),314);
        vcdp->chgArray(c+55005,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[2]),314);
        vcdp->chgArray(c+55015,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[3]),314);
        vcdp->chgArray(c+55025,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[4]),314);
        vcdp->chgArray(c+55035,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[5]),314);
        vcdp->chgArray(c+55045,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[6]),314);
        vcdp->chgArray(c+55055,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[7]),314);
        vcdp->chgArray(c+55625,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),314);
        vcdp->chgArray(c+55705,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),314);
        vcdp->chgBus(c+55785,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),3);
        vcdp->chgBus(c+55793,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),3);
        vcdp->chgBus(c+55801,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),3);
        vcdp->chgBit(c+55809,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgArray(c+55817,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__s0_1_c0__DOT__value),242);
        __Vtemp1676[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][0U];
        __Vtemp1676[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][1U];
        __Vtemp1676[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][2U];
        __Vtemp1676[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0U][3U];
        vcdp->chgArray(c+55881,(__Vtemp1676),128);
        __Vtemp1677[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][0U];
        __Vtemp1677[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][1U];
        __Vtemp1677[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][2U];
        __Vtemp1677[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [1U][3U];
        vcdp->chgArray(c+55913,(__Vtemp1677),128);
        __Vtemp1678[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][0U];
        __Vtemp1678[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][1U];
        __Vtemp1678[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][2U];
        __Vtemp1678[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [2U][3U];
        vcdp->chgArray(c+55945,(__Vtemp1678),128);
        __Vtemp1679[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][0U];
        __Vtemp1679[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][1U];
        __Vtemp1679[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][2U];
        __Vtemp1679[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [3U][3U];
        vcdp->chgArray(c+55977,(__Vtemp1679),128);
        __Vtemp1680[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][0U];
        __Vtemp1680[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][1U];
        __Vtemp1680[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][2U];
        __Vtemp1680[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [4U][3U];
        vcdp->chgArray(c+56009,(__Vtemp1680),128);
        __Vtemp1681[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][0U];
        __Vtemp1681[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][1U];
        __Vtemp1681[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][2U];
        __Vtemp1681[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [5U][3U];
        vcdp->chgArray(c+56041,(__Vtemp1681),128);
        __Vtemp1682[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][0U];
        __Vtemp1682[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][1U];
        __Vtemp1682[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][2U];
        __Vtemp1682[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [6U][3U];
        vcdp->chgArray(c+56073,(__Vtemp1682),128);
        __Vtemp1683[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][0U];
        __Vtemp1683[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][1U];
        __Vtemp1683[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][2U];
        __Vtemp1683[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [7U][3U];
        vcdp->chgArray(c+56105,(__Vtemp1683),128);
        __Vtemp1684[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][0U];
        __Vtemp1684[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][1U];
        __Vtemp1684[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][2U];
        __Vtemp1684[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [8U][3U];
        vcdp->chgArray(c+56137,(__Vtemp1684),128);
        __Vtemp1685[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][0U];
        __Vtemp1685[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][1U];
        __Vtemp1685[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][2U];
        __Vtemp1685[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [9U][3U];
        vcdp->chgArray(c+56169,(__Vtemp1685),128);
        __Vtemp1686[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][0U];
        __Vtemp1686[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][1U];
        __Vtemp1686[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][2U];
        __Vtemp1686[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xaU][3U];
        vcdp->chgArray(c+56201,(__Vtemp1686),128);
        __Vtemp1687[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][0U];
        __Vtemp1687[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][1U];
        __Vtemp1687[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][2U];
        __Vtemp1687[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xbU][3U];
        vcdp->chgArray(c+56233,(__Vtemp1687),128);
        __Vtemp1688[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][0U];
        __Vtemp1688[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][1U];
        __Vtemp1688[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][2U];
        __Vtemp1688[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xcU][3U];
        vcdp->chgArray(c+56265,(__Vtemp1688),128);
        __Vtemp1689[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][0U];
        __Vtemp1689[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][1U];
        __Vtemp1689[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][2U];
        __Vtemp1689[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xdU][3U];
        vcdp->chgArray(c+56297,(__Vtemp1689),128);
        __Vtemp1690[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][0U];
        __Vtemp1690[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][1U];
        __Vtemp1690[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][2U];
        __Vtemp1690[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xeU][3U];
        vcdp->chgArray(c+56329,(__Vtemp1690),128);
        __Vtemp1691[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][0U];
        __Vtemp1691[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][1U];
        __Vtemp1691[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][2U];
        __Vtemp1691[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [0xfU][3U];
        vcdp->chgArray(c+56361,(__Vtemp1691),128);
        vcdp->chgBus(c+56393,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[0]),21);
        vcdp->chgBus(c+56394,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[1]),21);
        vcdp->chgBus(c+56395,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[2]),21);
        vcdp->chgBus(c+56396,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[3]),21);
        vcdp->chgBus(c+56397,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[4]),21);
        vcdp->chgBus(c+56398,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[5]),21);
        vcdp->chgBus(c+56399,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[6]),21);
        vcdp->chgBus(c+56400,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[7]),21);
        vcdp->chgBus(c+56401,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[8]),21);
        vcdp->chgBus(c+56402,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[9]),21);
        vcdp->chgBus(c+56403,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[10]),21);
        vcdp->chgBus(c+56404,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[11]),21);
        vcdp->chgBus(c+56405,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[12]),21);
        vcdp->chgBus(c+56406,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[13]),21);
        vcdp->chgBus(c+56407,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[14]),21);
        vcdp->chgBus(c+56408,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag[15]),21);
        vcdp->chgBus(c+56521,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0U]),16);
        vcdp->chgBus(c+56529,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [1U]),16);
        vcdp->chgBus(c+56537,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [2U]),16);
        vcdp->chgBus(c+56545,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [3U]),16);
        vcdp->chgBus(c+56553,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [4U]),16);
        vcdp->chgBus(c+56561,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [5U]),16);
        vcdp->chgBus(c+56569,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [6U]),16);
        vcdp->chgBus(c+56577,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [7U]),16);
        vcdp->chgBus(c+56585,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [8U]),16);
        vcdp->chgBus(c+56593,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [9U]),16);
        vcdp->chgBus(c+56601,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xaU]),16);
        vcdp->chgBus(c+56609,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xbU]),16);
        vcdp->chgBus(c+56617,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xcU]),16);
        vcdp->chgBus(c+56625,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xdU]),16);
        vcdp->chgBus(c+56633,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xeU]),16);
        vcdp->chgBus(c+56641,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [0xfU]),16);
        vcdp->chgBus(c+56649,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty),16);
        vcdp->chgBus(c+56657,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid),16);
        vcdp->chgBus(c+56665,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__i),32);
        vcdp->chgBus(c+56673,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__j),32);
        vcdp->chgArray(c+56681,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__tag_data_access__DOT__s0_1_c0__DOT__value),167);
        vcdp->chgArray(c+56729,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__st_1e_2__DOT__value),316);
        vcdp->chgArray(c+56809,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[0]),85);
        vcdp->chgArray(c+56812,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[1]),85);
        vcdp->chgArray(c+56815,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[2]),85);
        vcdp->chgArray(c+56818,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[3]),85);
        vcdp->chgArray(c+56821,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[4]),85);
        vcdp->chgArray(c+56824,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[5]),85);
        vcdp->chgArray(c+56827,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[6]),85);
        vcdp->chgArray(c+56830,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[7]),85);
        vcdp->chgArray(c+56833,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[8]),85);
        vcdp->chgArray(c+56836,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[9]),85);
        vcdp->chgArray(c+56839,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[10]),85);
        vcdp->chgArray(c+56842,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[11]),85);
        vcdp->chgArray(c+56845,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[12]),85);
        vcdp->chgArray(c+56848,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[13]),85);
        vcdp->chgArray(c+56851,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[14]),85);
        vcdp->chgArray(c+56854,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[15]),85);
        vcdp->chgArray(c+57193,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table),400);
        vcdp->chgBus(c+57297,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_table),16);
        vcdp->chgBus(c+57305,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__ready_table),16);
        vcdp->chgBus(c+57313,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr),4);
        vcdp->chgBus(c+57321,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__head_ptr),4);
        vcdp->chgBus(c+57329,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr),4);
        vcdp->chgBus(c+57337,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size),5);
        vcdp->chgBit(c+57345,((0x10U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+57353,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__size_r),4);
        vcdp->chgArray(c+57361,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[0]),76);
        vcdp->chgArray(c+57364,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[1]),76);
        vcdp->chgArray(c+57367,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[2]),76);
        vcdp->chgArray(c+57370,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[3]),76);
        vcdp->chgArray(c+57373,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[4]),76);
        vcdp->chgArray(c+57376,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[5]),76);
        vcdp->chgArray(c+57379,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[6]),76);
        vcdp->chgArray(c+57382,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[7]),76);
        vcdp->chgArray(c+57553,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),76);
        vcdp->chgArray(c+57577,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),76);
        vcdp->chgBus(c+57601,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),3);
        vcdp->chgBus(c+57609,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),3);
        vcdp->chgBus(c+57617,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),3);
        vcdp->chgBit(c+57625,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+57633,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwb_queue__DOT__size_r),3);
        vcdp->chgArray(c+57641,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[0]),199);
        vcdp->chgArray(c+57648,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[1]),199);
        vcdp->chgArray(c+57655,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[2]),199);
        vcdp->chgArray(c+57662,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[3]),199);
        vcdp->chgArray(c+57865,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),199);
        vcdp->chgArray(c+57921,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),199);
        vcdp->chgBus(c+57977,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+57985,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+57993,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+58001,(vlTOPp->VX_cache__DOT__genblk5__BRA__7__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
    }
}

void VVX_cache::traceChgThis__6(VVX_cache__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code) {
    VVX_cache* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    int c = code;
    if (0 && vcdp && c) {}  // Prevent unused
    // Body
    {
        vcdp->chgBit(c+58009,(vlTOPp->clk));
        vcdp->chgBit(c+58017,(vlTOPp->reset));
        vcdp->chgBus(c+58025,(vlTOPp->core_req_valid),4);
        vcdp->chgBus(c+58033,(vlTOPp->core_req_rw),4);
        vcdp->chgBus(c+58041,(vlTOPp->core_req_byteen),16);
        vcdp->chgArray(c+58049,(vlTOPp->core_req_addr),120);
        vcdp->chgArray(c+58081,(vlTOPp->core_req_data),128);
        vcdp->chgQuad(c+58113,(vlTOPp->core_req_tag),42);
        vcdp->chgBit(c+58129,(vlTOPp->core_req_ready));
        vcdp->chgBus(c+58137,(vlTOPp->core_rsp_valid),4);
        vcdp->chgArray(c+58145,(vlTOPp->core_rsp_data),128);
        vcdp->chgQuad(c+58177,(vlTOPp->core_rsp_tag),42);
        vcdp->chgBit(c+58193,(vlTOPp->core_rsp_ready));
        vcdp->chgBit(c+58201,(vlTOPp->dram_req_valid));
        vcdp->chgBit(c+58209,(vlTOPp->dram_req_rw));
        vcdp->chgBus(c+58217,(vlTOPp->dram_req_byteen),16);
        vcdp->chgBus(c+58225,(vlTOPp->dram_req_addr),28);
        vcdp->chgArray(c+58233,(vlTOPp->dram_req_data),128);
        vcdp->chgBus(c+58265,(vlTOPp->dram_req_tag),28);
        vcdp->chgBit(c+58273,(vlTOPp->dram_req_ready));
        vcdp->chgBit(c+58281,(vlTOPp->dram_rsp_valid));
        vcdp->chgArray(c+58289,(vlTOPp->dram_rsp_data),128);
        vcdp->chgBus(c+58321,(vlTOPp->dram_rsp_tag),28);
        vcdp->chgBit(c+58329,(vlTOPp->dram_rsp_ready));
        vcdp->chgBit(c+58337,(vlTOPp->snp_req_valid));
        vcdp->chgBus(c+58345,(vlTOPp->snp_req_addr),28);
        vcdp->chgBit(c+58353,(vlTOPp->snp_req_invalidate));
        vcdp->chgBus(c+58361,(vlTOPp->snp_req_tag),28);
        vcdp->chgBit(c+58369,(vlTOPp->snp_req_ready));
        vcdp->chgBit(c+58377,(vlTOPp->snp_rsp_valid));
        vcdp->chgBus(c+58385,(vlTOPp->snp_rsp_tag),28);
        vcdp->chgBit(c+58393,(vlTOPp->snp_rsp_ready));
        vcdp->chgBus(c+58401,(vlTOPp->snp_fwdout_valid),2);
        vcdp->chgQuad(c+58409,(vlTOPp->snp_fwdout_addr),56);
        vcdp->chgBus(c+58425,(vlTOPp->snp_fwdout_invalidate),2);
        vcdp->chgBus(c+58433,(vlTOPp->snp_fwdout_tag),2);
        vcdp->chgBus(c+58441,(vlTOPp->snp_fwdout_ready),2);
        vcdp->chgBus(c+58449,(vlTOPp->snp_fwdin_valid),2);
        vcdp->chgBus(c+58457,(vlTOPp->snp_fwdin_tag),2);
        vcdp->chgBus(c+58465,(vlTOPp->snp_fwdin_ready),2);
        vcdp->chgBit(c+58473,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_snp_req_ready) 
                                     >> (7U & vlTOPp->snp_req_addr)))));
        vcdp->chgBit(c+58481,(((IData)(vlTOPp->dram_rsp_valid) 
                               & (0U == (7U & vlTOPp->dram_rsp_tag)))));
        vcdp->chgBus(c+58489,((0x1ffffffU & (vlTOPp->dram_rsp_tag 
                                             >> 3U))),25);
        vcdp->chgBit(c+58497,(((IData)(vlTOPp->snp_req_valid) 
                               & (0U == (7U & vlTOPp->snp_req_addr)))));
        vcdp->chgBus(c+58505,((0x1ffffffU & (vlTOPp->snp_req_addr 
                                             >> 3U))),25);
        vcdp->chgBit(c+58513,(((IData)(vlTOPp->dram_rsp_valid) 
                               & (1U == (7U & vlTOPp->dram_rsp_tag)))));
        vcdp->chgBit(c+58521,(((IData)(vlTOPp->snp_req_valid) 
                               & (1U == (7U & vlTOPp->snp_req_addr)))));
        vcdp->chgBit(c+58529,(((IData)(vlTOPp->dram_rsp_valid) 
                               & (2U == (7U & vlTOPp->dram_rsp_tag)))));
        vcdp->chgBit(c+58537,(((IData)(vlTOPp->snp_req_valid) 
                               & (2U == (7U & vlTOPp->snp_req_addr)))));
        vcdp->chgBit(c+58545,(((IData)(vlTOPp->dram_rsp_valid) 
                               & (3U == (7U & vlTOPp->dram_rsp_tag)))));
        vcdp->chgBit(c+58553,(((IData)(vlTOPp->snp_req_valid) 
                               & (3U == (7U & vlTOPp->snp_req_addr)))));
        vcdp->chgBit(c+58561,(((IData)(vlTOPp->dram_rsp_valid) 
                               & (4U == (7U & vlTOPp->dram_rsp_tag)))));
        vcdp->chgBit(c+58569,(((IData)(vlTOPp->snp_req_valid) 
                               & (4U == (7U & vlTOPp->snp_req_addr)))));
        vcdp->chgBit(c+58577,(((IData)(vlTOPp->dram_rsp_valid) 
                               & (5U == (7U & vlTOPp->dram_rsp_tag)))));
        vcdp->chgBit(c+58585,(((IData)(vlTOPp->snp_req_valid) 
                               & (5U == (7U & vlTOPp->snp_req_addr)))));
        vcdp->chgBit(c+58593,(((IData)(vlTOPp->dram_rsp_valid) 
                               & (6U == (7U & vlTOPp->dram_rsp_tag)))));
        vcdp->chgBit(c+58601,(((IData)(vlTOPp->snp_req_valid) 
                               & (6U == (7U & vlTOPp->snp_req_addr)))));
        vcdp->chgBit(c+58609,(((IData)(vlTOPp->dram_rsp_valid) 
                               & (7U == (7U & vlTOPp->dram_rsp_tag)))));
        vcdp->chgBit(c+58617,(((IData)(vlTOPp->snp_req_valid) 
                               & (7U == (7U & vlTOPp->snp_req_addr)))));
        vcdp->chgBit(c+58625,(((IData)(vlTOPp->dram_req_valid) 
                               & (~ (IData)(vlTOPp->dram_req_rw)))));
        vcdp->chgBit(c+58633,((((IData)(vlTOPp->dram_req_valid) 
                                & (~ (IData)(vlTOPp->dram_req_rw))) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
    }
}
