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
        vcdp->chgBus(c+1,(vlTOPp->VX_cache__DOT____Vcellout__cache_core_req_bank_sel__per_bank_valid),16);
        vcdp->chgBus(c+9,(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready),4);
        vcdp->chgBus(c+17,(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_ready),4);
        vcdp->chgBus(c+25,(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_ready),4);
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
        vcdp->chgBus(c+161,(vlTOPp->VX_cache__DOT__cache_core_req_bank_sel__DOT__genblk2__DOT__per_bank_ready_sel),4);
        vcdp->chgBit(c+169,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dfqq_pop));
        vcdp->chgBit(c+177,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__reading));
        vcdp->chgBit(c+185,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dram_wb_req_fire));
        vcdp->chgBit(c+193,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_rsp_fire));
        vcdp->chgQuad(c+201,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__snp_req_queue__data_in),55);
        vcdp->chgBit(c+217,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__writing));
        vcdp->chgArray(c+225,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__dfp_queue__data_in),154);
        vcdp->chgBit(c+265,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__writing));
        vcdp->chgArray(c+273,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellinp__reqq_queue__data_in),314);
        vcdp->chgBit(c+353,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__writing));
        vcdp->chgBit(c+361,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__reading));
        vcdp->chgBit(c+369,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__reading));
        vcdp->chgBit(c+377,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dram_wb_req_fire));
        vcdp->chgBit(c+385,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_rsp_fire));
        vcdp->chgQuad(c+393,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__snp_req_queue__data_in),55);
        vcdp->chgBit(c+409,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__writing));
        vcdp->chgArray(c+417,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__dfp_queue__data_in),154);
        vcdp->chgBit(c+457,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__writing));
        vcdp->chgArray(c+465,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellinp__reqq_queue__data_in),314);
        vcdp->chgBit(c+545,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__writing));
        vcdp->chgBit(c+553,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__reading));
        vcdp->chgBit(c+561,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__reading));
        vcdp->chgBit(c+569,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dram_wb_req_fire));
        vcdp->chgBit(c+577,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_rsp_fire));
        vcdp->chgQuad(c+585,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__snp_req_queue__data_in),55);
        vcdp->chgBit(c+601,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__writing));
        vcdp->chgArray(c+609,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__dfp_queue__data_in),154);
        vcdp->chgBit(c+649,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__writing));
        vcdp->chgArray(c+657,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellinp__reqq_queue__data_in),314);
        vcdp->chgBit(c+737,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__writing));
        vcdp->chgBit(c+745,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__reading));
        vcdp->chgBit(c+753,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__reading));
        vcdp->chgBit(c+761,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dram_wb_req_fire));
        vcdp->chgBit(c+769,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_rsp_fire));
        vcdp->chgQuad(c+777,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__snp_req_queue__data_in),55);
        vcdp->chgBit(c+793,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__writing));
        vcdp->chgArray(c+801,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__dfp_queue__data_in),154);
        vcdp->chgBit(c+841,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__writing));
        vcdp->chgArray(c+849,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellinp__reqq_queue__data_in),314);
        vcdp->chgBit(c+929,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__writing));
        vcdp->chgBit(c+937,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__reading));
        vcdp->chgBit(c+945,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__reading));
    }
}

void VVX_cache::traceChgThis__3(VVX_cache__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code) {
    VVX_cache* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    int c = code;
    if (0 && vcdp && c) {}  // Prevent unused
    // Body
    {
        vcdp->chgBit(c+953,((((IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dfqq_pop) 
                              & (~ (IData)((0U != (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bank_dram_fill_req_valid))))) 
                             & (~ ((~ (IData)((0U != 
                                               (0xfU 
                                                & (vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[3U] 
                                                   >> 0x10U))))) 
                                   | (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r))))));
        vcdp->chgBit(c+961,(((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__curr_bank_core_req_valid)) 
                             & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+969,((1U & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)) 
                                   & (IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready)))));
        vcdp->chgBit(c+977,((((((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                 >> 7U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                              >> 6U))) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dram_wb_req_fire)) 
                              | (((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                   >> 6U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                                >> 7U))) 
                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_rsp_fire))) 
                             | (((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                  >> 7U) & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                            >> 6U)) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_rsp_fire)))));
        vcdp->chgBit(c+985,((((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__curr_bank_core_req_valid)) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                             & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+993,(((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__curr_bank_core_req_valid)) 
                             & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+1001,((1U & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)) 
                                    & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                       >> 1U)))));
        vcdp->chgBit(c+1009,((((((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                  >> 7U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                               >> 6U))) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dram_wb_req_fire)) 
                               | (((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 6U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                                 >> 7U))) 
                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_rsp_fire))) 
                              | (((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                   >> 7U) & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                             >> 6U)) 
                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_rsp_fire)))));
        vcdp->chgBit(c+1017,((((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__curr_bank_core_req_valid)) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+1025,(((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__curr_bank_core_req_valid)) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+1033,((1U & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)) 
                                    & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                       >> 2U)))));
        vcdp->chgBit(c+1041,((((((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                  >> 7U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                               >> 6U))) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dram_wb_req_fire)) 
                               | (((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 6U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                                 >> 7U))) 
                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_rsp_fire))) 
                              | (((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                   >> 7U) & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                             >> 6U)) 
                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_rsp_fire)))));
        vcdp->chgBit(c+1049,((((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__curr_bank_core_req_valid)) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+1057,(((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__curr_bank_core_req_valid)) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+1065,((1U & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)) 
                                    & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                       >> 3U)))));
        vcdp->chgBit(c+1073,((((((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                  >> 7U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                               >> 6U))) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dram_wb_req_fire)) 
                               | (((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 6U) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                                 >> 7U))) 
                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_rsp_fire))) 
                              | (((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                   >> 7U) & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                             >> 6U)) 
                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_rsp_fire)))));
        vcdp->chgBit(c+1081,((((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__curr_bank_core_req_valid)) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
    }
}

void VVX_cache::traceChgThis__4(VVX_cache__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code) {
    VVX_cache* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    int c = code;
    if (0 && vcdp && c) {}  // Prevent unused
    // Variables
    WData/*127:0*/ __Vtemp662[4];
    WData/*127:0*/ __Vtemp681[4];
    WData/*127:0*/ __Vtemp700[4];
    WData/*127:0*/ __Vtemp719[4];
    WData/*127:0*/ __Vtemp637[4];
    WData/*127:0*/ __Vtemp638[4];
    WData/*127:0*/ __Vtemp639[4];
    WData/*127:0*/ __Vtemp640[4];
    WData/*127:0*/ __Vtemp643[4];
    WData/*127:0*/ __Vtemp644[4];
    WData/*127:0*/ __Vtemp649[4];
    WData/*127:0*/ __Vtemp655[4];
    WData/*127:0*/ __Vtemp656[4];
    WData/*127:0*/ __Vtemp659[4];
    WData/*127:0*/ __Vtemp660[4];
    WData/*127:0*/ __Vtemp661[4];
    WData/*127:0*/ __Vtemp663[4];
    WData/*127:0*/ __Vtemp668[4];
    WData/*127:0*/ __Vtemp674[4];
    WData/*127:0*/ __Vtemp675[4];
    WData/*127:0*/ __Vtemp678[4];
    WData/*127:0*/ __Vtemp679[4];
    WData/*127:0*/ __Vtemp680[4];
    WData/*127:0*/ __Vtemp682[4];
    WData/*127:0*/ __Vtemp687[4];
    WData/*127:0*/ __Vtemp693[4];
    WData/*127:0*/ __Vtemp694[4];
    WData/*127:0*/ __Vtemp697[4];
    WData/*127:0*/ __Vtemp698[4];
    WData/*127:0*/ __Vtemp699[4];
    WData/*127:0*/ __Vtemp701[4];
    WData/*127:0*/ __Vtemp706[4];
    WData/*127:0*/ __Vtemp712[4];
    WData/*127:0*/ __Vtemp713[4];
    WData/*127:0*/ __Vtemp716[4];
    WData/*127:0*/ __Vtemp717[4];
    WData/*127:0*/ __Vtemp718[4];
    // Body
    {
        vcdp->chgBus(c+1089,(vlTOPp->VX_cache__DOT__per_bank_core_req_ready),4);
        vcdp->chgBus(c+1097,(vlTOPp->VX_cache__DOT__per_bank_core_rsp_valid),4);
        vcdp->chgBus(c+1105,(vlTOPp->VX_cache__DOT__per_bank_core_rsp_tid),8);
        vcdp->chgArray(c+1113,(vlTOPp->VX_cache__DOT__per_bank_core_rsp_data),128);
        vcdp->chgArray(c+1145,(vlTOPp->VX_cache__DOT__per_bank_core_rsp_tag),168);
        vcdp->chgBus(c+1193,(vlTOPp->VX_cache__DOT__per_bank_dram_fill_req_valid),4);
        vcdp->chgArray(c+1201,(vlTOPp->VX_cache__DOT__per_bank_dram_fill_req_addr),112);
        vcdp->chgBus(c+1233,(vlTOPp->VX_cache__DOT__per_bank_dram_fill_rsp_ready),4);
        vcdp->chgBus(c+1241,(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_valid),4);
        vcdp->chgQuad(c+1249,(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_byteen),64);
        vcdp->chgArray(c+1265,(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_addr),112);
        vcdp->chgArray(c+1297,(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_data),512);
        vcdp->chgBus(c+1425,(vlTOPp->VX_cache__DOT__per_bank_snp_req_ready),4);
        vcdp->chgBus(c+1433,(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_valid),4);
        vcdp->chgArray(c+1441,(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_tag),112);
        vcdp->chgBus(c+1473,((3U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U] 
                                    >> 0xaU))),2);
        vcdp->chgBus(c+1481,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[0U]),32);
        vcdp->chgQuad(c+1489,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[1U]))))),42);
        vcdp->chgBit(c+1505,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU))))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwbq_push_stall))))));
        vcdp->chgBit(c+1513,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__curr_bank_dram_wb_req_valid));
        vcdp->chgBus(c+1521,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                          << 0xaU) 
                                         | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                            >> 0x16U)))),16);
        vcdp->chgBus(c+1529,((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                             << 4U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                                               >> 0x1cU)))),26);
        __Vtemp637[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U] 
                                     >> 0x1cU));
        __Vtemp637[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                                     >> 0x1cU));
        __Vtemp637[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                                     >> 0x1cU));
        __Vtemp637[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                                     >> 0x1cU));
        vcdp->chgArray(c+1537,(__Vtemp637),128);
        vcdp->chgBit(c+1569,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__curr_bank_snp_rsp_valid));
        vcdp->chgBus(c+1577,((0xfffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U])),28);
        vcdp->chgBus(c+1585,((3U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U] 
                                    >> 0xaU))),2);
        vcdp->chgBus(c+1593,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[0U]),32);
        vcdp->chgQuad(c+1601,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[1U]))))),42);
        vcdp->chgBit(c+1617,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU))))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwbq_push_stall))))));
        vcdp->chgBit(c+1625,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__curr_bank_dram_wb_req_valid));
        vcdp->chgBus(c+1633,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                          << 0xaU) 
                                         | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                            >> 0x16U)))),16);
        vcdp->chgBus(c+1641,((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                             << 4U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                                               >> 0x1cU)))),26);
        __Vtemp638[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U] 
                                     >> 0x1cU));
        __Vtemp638[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                                     >> 0x1cU));
        __Vtemp638[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                                     >> 0x1cU));
        __Vtemp638[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                                     >> 0x1cU));
        vcdp->chgArray(c+1649,(__Vtemp638),128);
        vcdp->chgBit(c+1681,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__curr_bank_snp_rsp_valid));
        vcdp->chgBus(c+1689,((0xfffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U])),28);
        vcdp->chgBus(c+1697,((3U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U] 
                                    >> 0xaU))),2);
        vcdp->chgBus(c+1705,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[0U]),32);
        vcdp->chgQuad(c+1713,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[1U]))))),42);
        vcdp->chgBit(c+1729,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU))))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwbq_push_stall))))));
        vcdp->chgBit(c+1737,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__curr_bank_dram_wb_req_valid));
        vcdp->chgBus(c+1745,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                          << 0xaU) 
                                         | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                            >> 0x16U)))),16);
        vcdp->chgBus(c+1753,((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                             << 4U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                                               >> 0x1cU)))),26);
        __Vtemp639[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U] 
                                     >> 0x1cU));
        __Vtemp639[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                                     >> 0x1cU));
        __Vtemp639[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                                     >> 0x1cU));
        __Vtemp639[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                                     >> 0x1cU));
        vcdp->chgArray(c+1761,(__Vtemp639),128);
        vcdp->chgBit(c+1793,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__curr_bank_snp_rsp_valid));
        vcdp->chgBus(c+1801,((0xfffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U])),28);
        vcdp->chgBus(c+1809,((3U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U] 
                                    >> 0xaU))),2);
        vcdp->chgBus(c+1817,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[0U]),32);
        vcdp->chgQuad(c+1825,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[1U]))))),42);
        vcdp->chgBit(c+1841,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU))))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwbq_push_stall))))));
        vcdp->chgBit(c+1849,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__curr_bank_dram_wb_req_valid));
        vcdp->chgBus(c+1857,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                          << 0xaU) 
                                         | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                            >> 0x16U)))),16);
        vcdp->chgBus(c+1865,((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                             << 4U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                                               >> 0x1cU)))),26);
        __Vtemp640[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U] 
                                     >> 0x1cU));
        __Vtemp640[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                                     >> 0x1cU));
        __Vtemp640[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                                     >> 0x1cU));
        __Vtemp640[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                                     >> 0x1cU));
        vcdp->chgArray(c+1873,(__Vtemp640),128);
        vcdp->chgBit(c+1905,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__curr_bank_snp_rsp_valid));
        vcdp->chgBus(c+1913,((0xfffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U])),28);
        vcdp->chgBit(c+1921,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dwb_valid));
        vcdp->chgBit(c+1929,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dfqq_req));
        vcdp->chgBus(c+1937,(((0x6fU >= (0x7fU & ((IData)(0x1cU) 
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
                                                   (3U 
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
                                                   (3U 
                                                    & (((IData)(0x1cU) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_request_index)) 
                                                       >> 5U))] 
                                                   >> 
                                                   (0x1fU 
                                                    & ((IData)(0x1cU) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_request_index))))))
                               : 0U)),28);
        vcdp->chgBit(c+1945,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_has_request)))));
        vcdp->chgBit(c+1953,((0U != (IData)(vlTOPp->VX_cache__DOT__per_bank_dram_fill_req_valid))));
        vcdp->chgBus(c+1961,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dwb_bank),2);
        vcdp->chgBit(c+1969,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__update_use));
        vcdp->chgBit(c+1977,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__reading));
        vcdp->chgBit(c+1985,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__writing));
        vcdp->chgBus(c+1993,((0xfU & (vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[3U] 
                                      >> 0x10U))),4);
        __Vtemp643[0U] = vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[0U];
        __Vtemp643[1U] = vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[1U];
        __Vtemp643[2U] = vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[2U];
        __Vtemp643[3U] = (0xffffU & vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[3U]);
        vcdp->chgArray(c+2001,(__Vtemp643),112);
        vcdp->chgBus(c+2033,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bqual_bank_dram_fill_req_valid),4);
        vcdp->chgArray(c+2041,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_bank_dram_fill_req_addr),112);
        vcdp->chgBus(c+2073,(((IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bqual_bank_dram_fill_req_valid) 
                              & (~ ((IData)(1U) << (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_request_index))))),4);
        vcdp->chgBit(c+2081,((1U & ((~ (IData)((0U 
                                                != 
                                                (0xfU 
                                                 & (vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[3U] 
                                                    >> 0x10U))))) 
                                    | (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->chgBit(c+2089,(((0U != (IData)(vlTOPp->VX_cache__DOT__per_bank_dram_fill_req_valid)) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBus(c+2097,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_request_index),2);
        vcdp->chgBit(c+2105,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_has_request));
        vcdp->chgArray(c+2113,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellinp__dfqq_queue__data_in),116);
        vcdp->chgArray(c+2145,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out),116);
        vcdp->chgBit(c+2177,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__writing));
        vcdp->chgBus(c+2185,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->chgBus(c+2193,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgBus(c+2201,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__sel_dwb__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->chgBus(c+2209,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__sel_dwb__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgArray(c+2217,(vlTOPp->VX_cache__DOT____Vcellout__cache_core_rsp_merge__core_rsp_data),128);
        vcdp->chgQuad(c+2249,(((0xa7U >= (0xffU & ((IData)(0x2aU) 
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
                                                            (7U 
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
                                                           (7U 
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
                                                            (7U 
                                                             & (((IData)(0x2aU) 
                                                                 * (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__main_bank_index)) 
                                                                >> 5U))])) 
                                            >> (0x1fU 
                                                & ((IData)(0x2aU) 
                                                   * (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__main_bank_index)))))))
                                : VL_ULL(0))),42);
        vcdp->chgBus(c+2265,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__per_bank_core_rsp_pop_unqual),4);
        vcdp->chgBus(c+2273,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__main_bank_index),2);
        vcdp->chgBit(c+2281,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__grant_valid));
        vcdp->chgBus(c+2289,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->chgBus(c+2297,((((IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__requests_use) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r))) 
                              | (((IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__refill_original) 
                                  ^ (IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_valid)) 
                                 & (~ (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__refill_original))))),4);
        vcdp->chgBus(c+2305,((((IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__refill_original) 
                               ^ (IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_valid)) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__refill_original)))),4);
        vcdp->chgBus(c+2313,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgBus(c+2321,(vlTOPp->VX_cache__DOT__snp_rsp_arb__DOT__fsq_bank),2);
        vcdp->chgBit(c+2329,(vlTOPp->VX_cache__DOT__snp_rsp_arb__DOT__fsq_valid));
        vcdp->chgBus(c+2337,(vlTOPp->VX_cache__DOT__snp_rsp_arb__DOT__sel_ffsq__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->chgBus(c+2345,(vlTOPp->VX_cache__DOT__snp_rsp_arb__DOT__sel_ffsq__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgBit(c+2353,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop));
        vcdp->chgBus(c+2361,((0x3ffffffU & (IData)(
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                    >> 0x1dU)))),26);
        vcdp->chgBit(c+2369,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                            >> 0x1cU)))));
        vcdp->chgBus(c+2377,((0xfffffffU & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out))),28);
        vcdp->chgBit(c+2385,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop));
        vcdp->chgBus(c+2393,((0x3ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])),26);
        __Vtemp644[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U];
        __Vtemp644[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U];
        __Vtemp644[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U];
        __Vtemp644[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U];
        vcdp->chgArray(c+2401,(__Vtemp644),128);
        vcdp->chgBit(c+2433,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop));
        vcdp->chgBit(c+2441,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request)))));
        vcdp->chgBit(c+2449,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request));
        vcdp->chgBus(c+2457,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index),2);
        vcdp->chgBit(c+2465,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_req_rw_st0));
        vcdp->chgBus(c+2473,((0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                      >> (0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                  << 2U))))),4);
        vcdp->chgBus(c+2481,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_req_addr_st0),30);
        vcdp->chgBus(c+2489,((((0U == (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
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
        vcdp->chgBit(c+2497,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_pop));
        vcdp->chgBit(c+2505,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible));
        vcdp->chgBit(c+2513,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_rw_st0));
        vcdp->chgBit(c+2521,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match))));
        vcdp->chgBit(c+2529,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_because_miss) 
                              & (((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 7U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x19U))) 
                                  == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                  [0U]) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              >> 0x14U))))));
        vcdp->chgBit(c+2537,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__force_request_miss_st1e));
        vcdp->chgBit(c+2545,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__recover_mrvq_state_st2));
        vcdp->chgBit(c+2553,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_push_stall));
        vcdp->chgBit(c+2561,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwbq_push_stall));
        vcdp->chgBit(c+2569,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_push_stall));
        vcdp->chgBit(c+2577,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dram_fill_req_stall));
        vcdp->chgBit(c+2585,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__stall_bank_pipe));
        vcdp->chgBit(c+2593,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_in_pipe));
        vcdp->chgBit(c+2601,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1[0]));
        vcdp->chgBit(c+2609,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__going_to_write_st1[0]));
        vcdp->chgBit(c+2617,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual));
        vcdp->chgBit(c+2625,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop_unqual));
        vcdp->chgBit(c+2633,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop_unqual));
        vcdp->chgBit(c+2641,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop) 
                                | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_pop)) 
                               | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop)) 
                              | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop))));
        vcdp->chgBus(c+2649,((0x3ffffffU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                             ? (0x3ffffffU 
                                                & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])
                                             : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                                 ? 
                                                ((0x19fU 
                                                  >= 
                                                  (0x1ffU 
                                                   & ((IData)(0x1aU) 
                                                      * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                  ? 
                                                 (0x3ffffffU 
                                                  & (((0U 
                                                       == 
                                                       (0x1fU 
                                                        & ((IData)(0x1aU) 
                                                           * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                       ? 0U
                                                       : 
                                                      (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                       ((IData)(1U) 
                                                        + 
                                                        (0xfU 
                                                         & (((IData)(0x1aU) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                            >> 5U)))] 
                                                       << 
                                                       ((IData)(0x20U) 
                                                        - 
                                                        (0x1fU 
                                                         & ((IData)(0x1aU) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                     | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                        (0xfU 
                                                         & (((IData)(0x1aU) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                            >> 5U))] 
                                                        >> 
                                                        (0x1fU 
                                                         & ((IData)(0x1aU) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                                  : 0U)
                                                 : 
                                                ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop_unqual)
                                                  ? 
                                                 (0x3ffffffU 
                                                  & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_req_addr_st0 
                                                     >> 4U))
                                                  : 
                                                 ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop_unqual)
                                                   ? 
                                                  (0x3ffffffU 
                                                   & (IData)(
                                                             (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                              >> 0x1dU)))
                                                   : 0U)))))),26);
        vcdp->chgBus(c+2657,((3U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop_unqual)
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
        vcdp->chgBus(c+2665,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
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
        __Vtemp649[0U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U]
                           : 0x39U);
        __Vtemp649[1U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U]
                           : 0U);
        __Vtemp649[2U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U]
                           : 0U);
        __Vtemp649[3U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U]
                           : 0U);
        vcdp->chgArray(c+2673,(__Vtemp649),128);
        vcdp->chgQuad(c+2705,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
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
        vcdp->chgBit(c+2721,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
                               ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible) 
                                        & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_rw_st0))
                                        ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop_unqual) 
                                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_req_rw_st0))
                                                 ? 1U
                                                 : 0U)))));
        vcdp->chgBit(c+2729,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                     ? (1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                              [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                              >> 1U))
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? 1U : 0U)))));
        vcdp->chgBit(c+2737,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                     ? (1U & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? (1U & (IData)(
                                                         (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                          >> 0x1cU)))
                                         : 0U)))));
        vcdp->chgBit(c+2745,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1[0]));
        vcdp->chgBus(c+2753,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1[0]),26);
        vcdp->chgBus(c+2761,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1[0]),2);
        vcdp->chgBus(c+2769,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writeword_st1[0]),32);
        vcdp->chgQuad(c+2777,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__inst_meta_st1[0]),49);
        vcdp->chgArray(c+2793,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writedata_st1[0]),128);
        vcdp->chgBit(c+2825,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1[0]));
        vcdp->chgBit(c+2833,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_invalidate_st1[0]));
        vcdp->chgBit(c+2841,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_mrvq_st1[0]));
        __Vtemp655[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp655[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp655[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp655[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgBus(c+2849,((((0U == (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                                [0U] 
                                                << 5U)))
                                ? 0U : (__Vtemp655[
                                        ((IData)(1U) 
                                         + (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                            [0U]))] 
                                        << ((IData)(0x20U) 
                                            - (0x1fU 
                                               & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                                  [0U] 
                                                  << 5U))))) 
                              | (__Vtemp655[(3U & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                             [0U])] 
                                 >> (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                              [0U] 
                                              << 5U))))),32);
        __Vtemp656[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp656[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp656[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp656[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgArray(c+2857,(__Vtemp656),128);
        vcdp->chgBus(c+2889,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c
                             [0U]),20);
        vcdp->chgBit(c+2897,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_st1e));
        vcdp->chgBit(c+2905,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dirty_st1e));
        vcdp->chgBus(c+2913,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c
                             [0U]),16);
        vcdp->chgQuad(c+2921,((VL_ULL(0x3ffffffffff) 
                               & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__inst_meta_st1
                                  [0U] >> 7U))),42);
        vcdp->chgBus(c+2937,((3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__inst_meta_st1
                                           [0U]))),2);
        vcdp->chgBit(c+2945,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__inst_meta_st1
                                            [0U] >> 6U)))));
        vcdp->chgBus(c+2953,((0xfU & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__inst_meta_st1
                                              [0U] 
                                              >> 2U)))),4);
        vcdp->chgBit(c+2961,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dirty_st1e))));
        vcdp->chgBit(c+2969,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1
                             [0U]));
        vcdp->chgBit(c+2977,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_invalidate_st1
                             [0U]));
        vcdp->chgBit(c+2985,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_to_mrvq_st1e));
        vcdp->chgBit(c+2993,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_to_mrvq_st1e) 
                              | ((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                   & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1
                                      [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                                                [0U])) 
                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                                 [0U]))));
        vcdp->chgBit(c+3001,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_because_miss));
        vcdp->chgBit(c+3009,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                             [0U]));
        vcdp->chgBit(c+3017,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_mrvq_st1
                             [0U]));
        vcdp->chgBit(c+3025,((((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                                [0U] & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_mrvq_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__recover_mrvq_state_st2)) 
                              & ((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                 << 7U) 
                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                   >> 0x19U))) 
                                 == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                 [0U]))));
        vcdp->chgBus(c+3033,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                             [0U]),26);
        vcdp->chgBit(c+3041,((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                              [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                                      [0U]))));
        vcdp->chgBit(c+3049,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_init_ready_state_st2));
        vcdp->chgBit(c+3057,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_unqual) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)) 
                              & ((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                 << 7U) 
                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                   >> 0x19U))) 
                                 == (0x3ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])))));
        vcdp->chgBit(c+3065,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_unqual) 
                               & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                               [0U]) & ((0x3ffffffU 
                                         & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                             << 7U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                               >> 0x19U))) 
                                        == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                        [0U]))));
        vcdp->chgBit(c+3073,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_unqual));
        vcdp->chgBit(c+3081,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add));
        vcdp->chgBit(c+3089,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_is_mrvq));
        vcdp->chgBit(c+3097,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwbq_push_unqual) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))) & (~ 
                                                 (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_push_stall) 
                                                   | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                                  | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+3105,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwbq_push_unqual));
        vcdp->chgBit(c+3113,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_unqual) 
                              & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                 | ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU) & (~ 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                   >> 0x1bU)))))));
        vcdp->chgBit(c+3121,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_push_unqual) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+3129,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_is_dwb_in));
        vcdp->chgBit(c+3137,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_is_snp_in));
        vcdp->chgBit(c+3145,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 7U))));
        vcdp->chgBit(c+3153,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 6U))));
        vcdp->chgBit(c+3161,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_push_unqual));
        vcdp->chgQuad(c+3169,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out),55);
        vcdp->chgBit(c+3185,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__reading));
        vcdp->chgArray(c+3193,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out),154);
        vcdp->chgBit(c+3233,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__reading));
        vcdp->chgBus(c+3241,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                      >> 0x16U))),4);
        vcdp->chgBus(c+3249,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                      >> 0x12U))),4);
        vcdp->chgBus(c+3257,((0xffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                         >> 2U))),16);
        __Vtemp659[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                                        >> 0xaU));
        __Vtemp659[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                                        >> 0xaU));
        __Vtemp659[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                                        >> 0xaU));
        __Vtemp659[3U] = (0xffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                        << 0x16U) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                                        >> 0xaU)));
        vcdp->chgArray(c+3265,(__Vtemp659),120);
        __Vtemp660[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U] 
                                        >> 0xaU));
        __Vtemp660[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                                        >> 0xaU));
        __Vtemp660[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                                        >> 0xaU));
        __Vtemp660[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                                        >> 0xaU));
        vcdp->chgArray(c+3297,(__Vtemp660),128);
        vcdp->chgQuad(c+3329,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[0U]))))),42);
        vcdp->chgBit(c+3345,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty));
        vcdp->chgBit(c+3353,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__pop_qual));
        vcdp->chgBus(c+3361,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x16U) & 
                                      VL_NEGATE_I((IData)(
                                                          (1U 
                                                           & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty)))))))),4);
        vcdp->chgArray(c+3369,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out),314);
        vcdp->chgBit(c+3449,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__reading));
        vcdp->chgBus(c+3457,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->chgBus(c+3465,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgArray(c+3473,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__s0_1_c0__in),243);
        vcdp->chgBus(c+3537,((0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                              [0U])),6);
        vcdp->chgBit(c+3545,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                             [0U]));
        vcdp->chgBus(c+3553,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writeword_st1
                             [0U]),32);
        __Vtemp661[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writedata_st1
            [0U][0U];
        __Vtemp661[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writedata_st1
            [0U][1U];
        __Vtemp661[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writedata_st1
            [0U][2U];
        __Vtemp661[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writedata_st1
            [0U][3U];
        vcdp->chgArray(c+3561,(__Vtemp661),128);
        vcdp->chgBus(c+3593,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                             [0U]),2);
        vcdp->chgBit(c+3601,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c[0]));
        vcdp->chgBit(c+3609,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c[0]));
        vcdp->chgBus(c+3617,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c[0]),16);
        vcdp->chgBus(c+3625,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c[0]),20);
        vcdp->chgArray(c+3633,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c[0]),128);
        vcdp->chgBit(c+3665,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid 
                                            >> (0x3fU 
                                                & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                                [0U]))))));
        vcdp->chgBit(c+3673,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty 
                                            >> (0x3fU 
                                                & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                                [0U]))))));
        vcdp->chgBus(c+3681,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                             [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                               [0U])]),16);
        vcdp->chgBus(c+3689,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag
                             [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                               [0U])]),20);
        __Vtemp662[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
              [0U])][0U];
        __Vtemp662[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
              [0U])][1U];
        __Vtemp662[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
              [0U])][2U];
        __Vtemp662[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
              [0U])][3U];
        vcdp->chgArray(c+3697,(__Vtemp662),128);
        vcdp->chgBit(c+3729,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                             [0U]));
        vcdp->chgBit(c+3737,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c
                             [0U]));
        vcdp->chgBus(c+3745,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable),16);
        vcdp->chgArray(c+3753,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__data_write),128);
        vcdp->chgBit(c+3785,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__snoop_hit_no_pending));
        vcdp->chgBit(c+3793,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match));
        vcdp->chgBit(c+3801,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill));
        vcdp->chgBus(c+3809,((0xfffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                          [0U] >> 6U))),20);
        vcdp->chgBus(c+3817,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__we),16);
        vcdp->chgBit(c+3825,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__should_write));
        vcdp->chgBit(c+3833,((((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                                [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1
                                        [0U])) & (~ 
                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                                  [0U])) 
                              & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                                 [0U]))));
        vcdp->chgBit(c+3841,(((((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                                 [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1
                                         [0U])) & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                [0U]) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                                         [0U])) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match)))));
        vcdp->chgBit(c+3849,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss));
        vcdp->chgBit(c+3857,((((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                 & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1
                                    [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                                              [0U])) 
                               & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                               [0U]) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss)))));
        vcdp->chgBit(c+3865,(((0U == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+3873,(((1U == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+3881,(((2U == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+3889,(((3U == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+3897,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable))));
        vcdp->chgArray(c+3905,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT____Vcellinp__s0_1_c0__in),166);
        vcdp->chgArray(c+3953,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__st_1e_2__in),316);
        vcdp->chgBus(c+4033,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready),16);
        vcdp->chgBus(c+4041,((0xffffU & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                                          & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                         << (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr)))),16);
        vcdp->chgBus(c+4049,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match),16);
        vcdp->chgBit(c+4057,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push));
        vcdp->chgBit(c+4065,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_pop));
        vcdp->chgBit(c+4073,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_is_mrvq))));
        vcdp->chgBit(c+4081,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__increment_head));
        vcdp->chgBit(c+4089,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready))));
        vcdp->chgBit(c+4097,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_init_ready_state_st2))));
        vcdp->chgArray(c+4105,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__cwb_queue__data_in),76);
        vcdp->chgArray(c+4129,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out),76);
        vcdp->chgBit(c+4153,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__writing));
        vcdp->chgArray(c+4161,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__dwb_queue__data_in),200);
        vcdp->chgArray(c+4217,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out),200);
        vcdp->chgBit(c+4273,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__writing));
        vcdp->chgBit(c+4281,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop));
        vcdp->chgBus(c+4289,((0x3ffffffU & (IData)(
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                    >> 0x1dU)))),26);
        vcdp->chgBit(c+4297,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                            >> 0x1cU)))));
        vcdp->chgBus(c+4305,((0xfffffffU & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out))),28);
        vcdp->chgBit(c+4313,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop));
        vcdp->chgBus(c+4321,((0x3ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])),26);
        __Vtemp663[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U];
        __Vtemp663[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U];
        __Vtemp663[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U];
        __Vtemp663[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U];
        vcdp->chgArray(c+4329,(__Vtemp663),128);
        vcdp->chgBit(c+4361,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop));
        vcdp->chgBit(c+4369,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request)))));
        vcdp->chgBit(c+4377,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request));
        vcdp->chgBus(c+4385,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index),2);
        vcdp->chgBit(c+4393,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_req_rw_st0));
        vcdp->chgBus(c+4401,((0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                      >> (0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                  << 2U))))),4);
        vcdp->chgBus(c+4409,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_req_addr_st0),30);
        vcdp->chgBus(c+4417,((((0U == (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
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
        vcdp->chgBit(c+4425,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_pop));
        vcdp->chgBit(c+4433,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible));
        vcdp->chgBit(c+4441,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_rw_st0));
        vcdp->chgBit(c+4449,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match))));
        vcdp->chgBit(c+4457,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_because_miss) 
                              & (((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 7U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x19U))) 
                                  == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                  [0U]) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              >> 0x14U))))));
        vcdp->chgBit(c+4465,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__force_request_miss_st1e));
        vcdp->chgBit(c+4473,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__recover_mrvq_state_st2));
        vcdp->chgBit(c+4481,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_push_stall));
        vcdp->chgBit(c+4489,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwbq_push_stall));
        vcdp->chgBit(c+4497,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_push_stall));
        vcdp->chgBit(c+4505,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dram_fill_req_stall));
        vcdp->chgBit(c+4513,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__stall_bank_pipe));
        vcdp->chgBit(c+4521,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_in_pipe));
        vcdp->chgBit(c+4529,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1[0]));
        vcdp->chgBit(c+4537,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__going_to_write_st1[0]));
        vcdp->chgBit(c+4545,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual));
        vcdp->chgBit(c+4553,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop_unqual));
        vcdp->chgBit(c+4561,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop_unqual));
        vcdp->chgBit(c+4569,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop) 
                                | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_pop)) 
                               | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop)) 
                              | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop))));
        vcdp->chgBus(c+4577,((0x3ffffffU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                             ? (0x3ffffffU 
                                                & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])
                                             : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                                 ? 
                                                ((0x19fU 
                                                  >= 
                                                  (0x1ffU 
                                                   & ((IData)(0x1aU) 
                                                      * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                  ? 
                                                 (0x3ffffffU 
                                                  & (((0U 
                                                       == 
                                                       (0x1fU 
                                                        & ((IData)(0x1aU) 
                                                           * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                       ? 0U
                                                       : 
                                                      (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                       ((IData)(1U) 
                                                        + 
                                                        (0xfU 
                                                         & (((IData)(0x1aU) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                            >> 5U)))] 
                                                       << 
                                                       ((IData)(0x20U) 
                                                        - 
                                                        (0x1fU 
                                                         & ((IData)(0x1aU) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                     | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                        (0xfU 
                                                         & (((IData)(0x1aU) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                            >> 5U))] 
                                                        >> 
                                                        (0x1fU 
                                                         & ((IData)(0x1aU) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                                  : 0U)
                                                 : 
                                                ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop_unqual)
                                                  ? 
                                                 (0x3ffffffU 
                                                  & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_req_addr_st0 
                                                     >> 4U))
                                                  : 
                                                 ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop_unqual)
                                                   ? 
                                                  (0x3ffffffU 
                                                   & (IData)(
                                                             (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                              >> 0x1dU)))
                                                   : 0U)))))),26);
        vcdp->chgBus(c+4585,((3U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop_unqual)
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
        vcdp->chgBus(c+4593,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
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
        __Vtemp668[0U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U]
                           : 0x39U);
        __Vtemp668[1U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U]
                           : 0U);
        __Vtemp668[2U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U]
                           : 0U);
        __Vtemp668[3U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U]
                           : 0U);
        vcdp->chgArray(c+4601,(__Vtemp668),128);
        vcdp->chgQuad(c+4633,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
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
        vcdp->chgBit(c+4649,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
                               ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible) 
                                        & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_rw_st0))
                                        ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop_unqual) 
                                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_req_rw_st0))
                                                 ? 1U
                                                 : 0U)))));
        vcdp->chgBit(c+4657,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                     ? (1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                              [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                              >> 1U))
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? 1U : 0U)))));
        vcdp->chgBit(c+4665,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                     ? (1U & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? (1U & (IData)(
                                                         (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                          >> 0x1cU)))
                                         : 0U)))));
        vcdp->chgBit(c+4673,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1[0]));
        vcdp->chgBus(c+4681,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1[0]),26);
        vcdp->chgBus(c+4689,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1[0]),2);
        vcdp->chgBus(c+4697,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writeword_st1[0]),32);
        vcdp->chgQuad(c+4705,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__inst_meta_st1[0]),49);
        vcdp->chgArray(c+4721,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writedata_st1[0]),128);
        vcdp->chgBit(c+4753,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1[0]));
        vcdp->chgBit(c+4761,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_invalidate_st1[0]));
        vcdp->chgBit(c+4769,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_mrvq_st1[0]));
        __Vtemp674[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp674[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp674[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp674[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgBus(c+4777,((((0U == (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                                [0U] 
                                                << 5U)))
                                ? 0U : (__Vtemp674[
                                        ((IData)(1U) 
                                         + (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                            [0U]))] 
                                        << ((IData)(0x20U) 
                                            - (0x1fU 
                                               & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                                  [0U] 
                                                  << 5U))))) 
                              | (__Vtemp674[(3U & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                             [0U])] 
                                 >> (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                              [0U] 
                                              << 5U))))),32);
        __Vtemp675[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp675[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp675[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp675[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgArray(c+4785,(__Vtemp675),128);
        vcdp->chgBus(c+4817,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c
                             [0U]),20);
        vcdp->chgBit(c+4825,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_st1e));
        vcdp->chgBit(c+4833,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dirty_st1e));
        vcdp->chgBus(c+4841,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c
                             [0U]),16);
        vcdp->chgQuad(c+4849,((VL_ULL(0x3ffffffffff) 
                               & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__inst_meta_st1
                                  [0U] >> 7U))),42);
        vcdp->chgBus(c+4865,((3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__inst_meta_st1
                                           [0U]))),2);
        vcdp->chgBit(c+4873,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__inst_meta_st1
                                            [0U] >> 6U)))));
        vcdp->chgBus(c+4881,((0xfU & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__inst_meta_st1
                                              [0U] 
                                              >> 2U)))),4);
        vcdp->chgBit(c+4889,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dirty_st1e))));
        vcdp->chgBit(c+4897,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1
                             [0U]));
        vcdp->chgBit(c+4905,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_invalidate_st1
                             [0U]));
        vcdp->chgBit(c+4913,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_to_mrvq_st1e));
        vcdp->chgBit(c+4921,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_to_mrvq_st1e) 
                              | ((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                   & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1
                                      [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                                                [0U])) 
                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                                 [0U]))));
        vcdp->chgBit(c+4929,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_because_miss));
        vcdp->chgBit(c+4937,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                             [0U]));
        vcdp->chgBit(c+4945,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_mrvq_st1
                             [0U]));
        vcdp->chgBit(c+4953,((((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                                [0U] & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_mrvq_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__recover_mrvq_state_st2)) 
                              & ((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                 << 7U) 
                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                   >> 0x19U))) 
                                 == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                 [0U]))));
        vcdp->chgBus(c+4961,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                             [0U]),26);
        vcdp->chgBit(c+4969,((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                              [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                                      [0U]))));
        vcdp->chgBit(c+4977,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_init_ready_state_st2));
        vcdp->chgBit(c+4985,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_unqual) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)) 
                              & ((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                 << 7U) 
                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                   >> 0x19U))) 
                                 == (0x3ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])))));
        vcdp->chgBit(c+4993,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_unqual) 
                               & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                               [0U]) & ((0x3ffffffU 
                                         & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                             << 7U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                               >> 0x19U))) 
                                        == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                        [0U]))));
        vcdp->chgBit(c+5001,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_unqual));
        vcdp->chgBit(c+5009,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add));
        vcdp->chgBit(c+5017,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_is_mrvq));
        vcdp->chgBit(c+5025,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwbq_push_unqual) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))) & (~ 
                                                 (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_push_stall) 
                                                   | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                                  | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+5033,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwbq_push_unqual));
        vcdp->chgBit(c+5041,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_unqual) 
                              & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                 | ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU) & (~ 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                   >> 0x1bU)))))));
        vcdp->chgBit(c+5049,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_push_unqual) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+5057,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_is_dwb_in));
        vcdp->chgBit(c+5065,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_is_snp_in));
        vcdp->chgBit(c+5073,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 7U))));
        vcdp->chgBit(c+5081,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 6U))));
        vcdp->chgBit(c+5089,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_push_unqual));
        vcdp->chgQuad(c+5097,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out),55);
        vcdp->chgBit(c+5113,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__reading));
        vcdp->chgArray(c+5121,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out),154);
        vcdp->chgBit(c+5161,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__reading));
        vcdp->chgBus(c+5169,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                      >> 0x16U))),4);
        vcdp->chgBus(c+5177,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                      >> 0x12U))),4);
        vcdp->chgBus(c+5185,((0xffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                         >> 2U))),16);
        __Vtemp678[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                                        >> 0xaU));
        __Vtemp678[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                                        >> 0xaU));
        __Vtemp678[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                                        >> 0xaU));
        __Vtemp678[3U] = (0xffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                        << 0x16U) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                                        >> 0xaU)));
        vcdp->chgArray(c+5193,(__Vtemp678),120);
        __Vtemp679[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U] 
                                        >> 0xaU));
        __Vtemp679[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                                        >> 0xaU));
        __Vtemp679[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                                        >> 0xaU));
        __Vtemp679[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                                        >> 0xaU));
        vcdp->chgArray(c+5225,(__Vtemp679),128);
        vcdp->chgQuad(c+5257,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[0U]))))),42);
        vcdp->chgBit(c+5273,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty));
        vcdp->chgBit(c+5281,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__pop_qual));
        vcdp->chgBus(c+5289,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x16U) & 
                                      VL_NEGATE_I((IData)(
                                                          (1U 
                                                           & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty)))))))),4);
        vcdp->chgArray(c+5297,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out),314);
        vcdp->chgBit(c+5377,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__reading));
        vcdp->chgBus(c+5385,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->chgBus(c+5393,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgArray(c+5401,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__s0_1_c0__in),243);
        vcdp->chgBus(c+5465,((0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                              [0U])),6);
        vcdp->chgBit(c+5473,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                             [0U]));
        vcdp->chgBus(c+5481,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writeword_st1
                             [0U]),32);
        __Vtemp680[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writedata_st1
            [0U][0U];
        __Vtemp680[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writedata_st1
            [0U][1U];
        __Vtemp680[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writedata_st1
            [0U][2U];
        __Vtemp680[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writedata_st1
            [0U][3U];
        vcdp->chgArray(c+5489,(__Vtemp680),128);
        vcdp->chgBus(c+5521,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                             [0U]),2);
        vcdp->chgBit(c+5529,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c[0]));
        vcdp->chgBit(c+5537,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c[0]));
        vcdp->chgBus(c+5545,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c[0]),16);
        vcdp->chgBus(c+5553,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c[0]),20);
        vcdp->chgArray(c+5561,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c[0]),128);
        vcdp->chgBit(c+5593,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid 
                                            >> (0x3fU 
                                                & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                                [0U]))))));
        vcdp->chgBit(c+5601,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty 
                                            >> (0x3fU 
                                                & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                                [0U]))))));
        vcdp->chgBus(c+5609,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                             [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                               [0U])]),16);
        vcdp->chgBus(c+5617,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag
                             [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                               [0U])]),20);
        __Vtemp681[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
              [0U])][0U];
        __Vtemp681[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
              [0U])][1U];
        __Vtemp681[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
              [0U])][2U];
        __Vtemp681[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
              [0U])][3U];
        vcdp->chgArray(c+5625,(__Vtemp681),128);
        vcdp->chgBit(c+5657,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                             [0U]));
        vcdp->chgBit(c+5665,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c
                             [0U]));
        vcdp->chgBus(c+5673,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable),16);
        vcdp->chgArray(c+5681,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__data_write),128);
        vcdp->chgBit(c+5713,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__snoop_hit_no_pending));
        vcdp->chgBit(c+5721,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match));
        vcdp->chgBit(c+5729,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill));
        vcdp->chgBus(c+5737,((0xfffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                          [0U] >> 6U))),20);
        vcdp->chgBus(c+5745,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__we),16);
        vcdp->chgBit(c+5753,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__should_write));
        vcdp->chgBit(c+5761,((((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                                [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1
                                        [0U])) & (~ 
                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                                  [0U])) 
                              & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                                 [0U]))));
        vcdp->chgBit(c+5769,(((((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                                 [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1
                                         [0U])) & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                [0U]) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                                         [0U])) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match)))));
        vcdp->chgBit(c+5777,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss));
        vcdp->chgBit(c+5785,((((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                 & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1
                                    [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                                              [0U])) 
                               & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                               [0U]) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss)))));
        vcdp->chgBit(c+5793,(((0U == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+5801,(((1U == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+5809,(((2U == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+5817,(((3U == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+5825,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable))));
        vcdp->chgArray(c+5833,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT____Vcellinp__s0_1_c0__in),166);
        vcdp->chgArray(c+5881,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__st_1e_2__in),316);
        vcdp->chgBus(c+5961,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready),16);
        vcdp->chgBus(c+5969,((0xffffU & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                                          & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                         << (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr)))),16);
        vcdp->chgBus(c+5977,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match),16);
        vcdp->chgBit(c+5985,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push));
        vcdp->chgBit(c+5993,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_pop));
        vcdp->chgBit(c+6001,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_is_mrvq))));
        vcdp->chgBit(c+6009,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__increment_head));
        vcdp->chgBit(c+6017,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready))));
        vcdp->chgBit(c+6025,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_init_ready_state_st2))));
        vcdp->chgArray(c+6033,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__cwb_queue__data_in),76);
        vcdp->chgArray(c+6057,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out),76);
        vcdp->chgBit(c+6081,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__writing));
        vcdp->chgArray(c+6089,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__dwb_queue__data_in),200);
        vcdp->chgArray(c+6145,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out),200);
        vcdp->chgBit(c+6201,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__writing));
        vcdp->chgBit(c+6209,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop));
        vcdp->chgBus(c+6217,((0x3ffffffU & (IData)(
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                    >> 0x1dU)))),26);
        vcdp->chgBit(c+6225,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                            >> 0x1cU)))));
        vcdp->chgBus(c+6233,((0xfffffffU & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out))),28);
        vcdp->chgBit(c+6241,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop));
        vcdp->chgBus(c+6249,((0x3ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])),26);
        __Vtemp682[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U];
        __Vtemp682[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U];
        __Vtemp682[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U];
        __Vtemp682[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U];
        vcdp->chgArray(c+6257,(__Vtemp682),128);
        vcdp->chgBit(c+6289,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop));
        vcdp->chgBit(c+6297,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request)))));
        vcdp->chgBit(c+6305,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request));
        vcdp->chgBus(c+6313,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index),2);
        vcdp->chgBit(c+6321,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_req_rw_st0));
        vcdp->chgBus(c+6329,((0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                      >> (0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                  << 2U))))),4);
        vcdp->chgBus(c+6337,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_req_addr_st0),30);
        vcdp->chgBus(c+6345,((((0U == (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
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
        vcdp->chgBit(c+6353,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_pop));
        vcdp->chgBit(c+6361,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible));
        vcdp->chgBit(c+6369,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_rw_st0));
        vcdp->chgBit(c+6377,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match))));
        vcdp->chgBit(c+6385,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_because_miss) 
                              & (((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 7U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x19U))) 
                                  == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                  [0U]) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              >> 0x14U))))));
        vcdp->chgBit(c+6393,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__force_request_miss_st1e));
        vcdp->chgBit(c+6401,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__recover_mrvq_state_st2));
        vcdp->chgBit(c+6409,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_push_stall));
        vcdp->chgBit(c+6417,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwbq_push_stall));
        vcdp->chgBit(c+6425,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_push_stall));
        vcdp->chgBit(c+6433,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dram_fill_req_stall));
        vcdp->chgBit(c+6441,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__stall_bank_pipe));
        vcdp->chgBit(c+6449,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_in_pipe));
        vcdp->chgBit(c+6457,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1[0]));
        vcdp->chgBit(c+6465,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__going_to_write_st1[0]));
        vcdp->chgBit(c+6473,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual));
        vcdp->chgBit(c+6481,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop_unqual));
        vcdp->chgBit(c+6489,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop_unqual));
        vcdp->chgBit(c+6497,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop) 
                                | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_pop)) 
                               | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop)) 
                              | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop))));
        vcdp->chgBus(c+6505,((0x3ffffffU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                             ? (0x3ffffffU 
                                                & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])
                                             : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                                 ? 
                                                ((0x19fU 
                                                  >= 
                                                  (0x1ffU 
                                                   & ((IData)(0x1aU) 
                                                      * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                  ? 
                                                 (0x3ffffffU 
                                                  & (((0U 
                                                       == 
                                                       (0x1fU 
                                                        & ((IData)(0x1aU) 
                                                           * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                       ? 0U
                                                       : 
                                                      (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                       ((IData)(1U) 
                                                        + 
                                                        (0xfU 
                                                         & (((IData)(0x1aU) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                            >> 5U)))] 
                                                       << 
                                                       ((IData)(0x20U) 
                                                        - 
                                                        (0x1fU 
                                                         & ((IData)(0x1aU) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                     | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                        (0xfU 
                                                         & (((IData)(0x1aU) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                            >> 5U))] 
                                                        >> 
                                                        (0x1fU 
                                                         & ((IData)(0x1aU) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                                  : 0U)
                                                 : 
                                                ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop_unqual)
                                                  ? 
                                                 (0x3ffffffU 
                                                  & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_req_addr_st0 
                                                     >> 4U))
                                                  : 
                                                 ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop_unqual)
                                                   ? 
                                                  (0x3ffffffU 
                                                   & (IData)(
                                                             (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                              >> 0x1dU)))
                                                   : 0U)))))),26);
        vcdp->chgBus(c+6513,((3U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop_unqual)
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
        vcdp->chgBus(c+6521,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
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
        __Vtemp687[0U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U]
                           : 0x39U);
        __Vtemp687[1U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U]
                           : 0U);
        __Vtemp687[2U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U]
                           : 0U);
        __Vtemp687[3U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U]
                           : 0U);
        vcdp->chgArray(c+6529,(__Vtemp687),128);
        vcdp->chgQuad(c+6561,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
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
        vcdp->chgBit(c+6577,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
                               ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible) 
                                        & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_rw_st0))
                                        ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop_unqual) 
                                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_req_rw_st0))
                                                 ? 1U
                                                 : 0U)))));
        vcdp->chgBit(c+6585,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                     ? (1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                              [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                              >> 1U))
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? 1U : 0U)))));
        vcdp->chgBit(c+6593,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                     ? (1U & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? (1U & (IData)(
                                                         (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                          >> 0x1cU)))
                                         : 0U)))));
        vcdp->chgBit(c+6601,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1[0]));
        vcdp->chgBus(c+6609,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1[0]),26);
        vcdp->chgBus(c+6617,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1[0]),2);
        vcdp->chgBus(c+6625,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writeword_st1[0]),32);
        vcdp->chgQuad(c+6633,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__inst_meta_st1[0]),49);
        vcdp->chgArray(c+6649,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writedata_st1[0]),128);
        vcdp->chgBit(c+6681,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1[0]));
        vcdp->chgBit(c+6689,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_invalidate_st1[0]));
        vcdp->chgBit(c+6697,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_mrvq_st1[0]));
        __Vtemp693[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp693[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp693[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp693[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgBus(c+6705,((((0U == (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                                [0U] 
                                                << 5U)))
                                ? 0U : (__Vtemp693[
                                        ((IData)(1U) 
                                         + (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                            [0U]))] 
                                        << ((IData)(0x20U) 
                                            - (0x1fU 
                                               & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                                  [0U] 
                                                  << 5U))))) 
                              | (__Vtemp693[(3U & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                             [0U])] 
                                 >> (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                              [0U] 
                                              << 5U))))),32);
        __Vtemp694[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp694[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp694[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp694[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgArray(c+6713,(__Vtemp694),128);
        vcdp->chgBus(c+6745,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c
                             [0U]),20);
        vcdp->chgBit(c+6753,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_st1e));
        vcdp->chgBit(c+6761,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dirty_st1e));
        vcdp->chgBus(c+6769,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c
                             [0U]),16);
        vcdp->chgQuad(c+6777,((VL_ULL(0x3ffffffffff) 
                               & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__inst_meta_st1
                                  [0U] >> 7U))),42);
        vcdp->chgBus(c+6793,((3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__inst_meta_st1
                                           [0U]))),2);
        vcdp->chgBit(c+6801,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__inst_meta_st1
                                            [0U] >> 6U)))));
        vcdp->chgBus(c+6809,((0xfU & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__inst_meta_st1
                                              [0U] 
                                              >> 2U)))),4);
        vcdp->chgBit(c+6817,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dirty_st1e))));
        vcdp->chgBit(c+6825,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1
                             [0U]));
        vcdp->chgBit(c+6833,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_invalidate_st1
                             [0U]));
        vcdp->chgBit(c+6841,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_to_mrvq_st1e));
        vcdp->chgBit(c+6849,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_to_mrvq_st1e) 
                              | ((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                   & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1
                                      [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                                                [0U])) 
                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                                 [0U]))));
        vcdp->chgBit(c+6857,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_because_miss));
        vcdp->chgBit(c+6865,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                             [0U]));
        vcdp->chgBit(c+6873,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_mrvq_st1
                             [0U]));
        vcdp->chgBit(c+6881,((((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                                [0U] & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_mrvq_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__recover_mrvq_state_st2)) 
                              & ((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                 << 7U) 
                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                   >> 0x19U))) 
                                 == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                 [0U]))));
        vcdp->chgBus(c+6889,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                             [0U]),26);
        vcdp->chgBit(c+6897,((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                              [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                                      [0U]))));
        vcdp->chgBit(c+6905,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_init_ready_state_st2));
        vcdp->chgBit(c+6913,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_unqual) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)) 
                              & ((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                 << 7U) 
                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                   >> 0x19U))) 
                                 == (0x3ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])))));
        vcdp->chgBit(c+6921,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_unqual) 
                               & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                               [0U]) & ((0x3ffffffU 
                                         & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                             << 7U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                               >> 0x19U))) 
                                        == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                        [0U]))));
        vcdp->chgBit(c+6929,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_unqual));
        vcdp->chgBit(c+6937,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add));
        vcdp->chgBit(c+6945,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_is_mrvq));
        vcdp->chgBit(c+6953,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwbq_push_unqual) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))) & (~ 
                                                 (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_push_stall) 
                                                   | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                                  | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+6961,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwbq_push_unqual));
        vcdp->chgBit(c+6969,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_unqual) 
                              & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                 | ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU) & (~ 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                   >> 0x1bU)))))));
        vcdp->chgBit(c+6977,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_push_unqual) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+6985,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_is_dwb_in));
        vcdp->chgBit(c+6993,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_is_snp_in));
        vcdp->chgBit(c+7001,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 7U))));
        vcdp->chgBit(c+7009,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 6U))));
        vcdp->chgBit(c+7017,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_push_unqual));
        vcdp->chgQuad(c+7025,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out),55);
        vcdp->chgBit(c+7041,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__reading));
        vcdp->chgArray(c+7049,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out),154);
        vcdp->chgBit(c+7089,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__reading));
        vcdp->chgBus(c+7097,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                      >> 0x16U))),4);
        vcdp->chgBus(c+7105,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                      >> 0x12U))),4);
        vcdp->chgBus(c+7113,((0xffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                         >> 2U))),16);
        __Vtemp697[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                                        >> 0xaU));
        __Vtemp697[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                                        >> 0xaU));
        __Vtemp697[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                                        >> 0xaU));
        __Vtemp697[3U] = (0xffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                        << 0x16U) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                                        >> 0xaU)));
        vcdp->chgArray(c+7121,(__Vtemp697),120);
        __Vtemp698[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U] 
                                        >> 0xaU));
        __Vtemp698[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                                        >> 0xaU));
        __Vtemp698[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                                        >> 0xaU));
        __Vtemp698[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                                        >> 0xaU));
        vcdp->chgArray(c+7153,(__Vtemp698),128);
        vcdp->chgQuad(c+7185,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[0U]))))),42);
        vcdp->chgBit(c+7201,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty));
        vcdp->chgBit(c+7209,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__pop_qual));
        vcdp->chgBus(c+7217,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x16U) & 
                                      VL_NEGATE_I((IData)(
                                                          (1U 
                                                           & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty)))))))),4);
        vcdp->chgArray(c+7225,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out),314);
        vcdp->chgBit(c+7305,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__reading));
        vcdp->chgBus(c+7313,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->chgBus(c+7321,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgArray(c+7329,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__s0_1_c0__in),243);
        vcdp->chgBus(c+7393,((0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                              [0U])),6);
        vcdp->chgBit(c+7401,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                             [0U]));
        vcdp->chgBus(c+7409,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writeword_st1
                             [0U]),32);
        __Vtemp699[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writedata_st1
            [0U][0U];
        __Vtemp699[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writedata_st1
            [0U][1U];
        __Vtemp699[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writedata_st1
            [0U][2U];
        __Vtemp699[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writedata_st1
            [0U][3U];
        vcdp->chgArray(c+7417,(__Vtemp699),128);
        vcdp->chgBus(c+7449,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                             [0U]),2);
        vcdp->chgBit(c+7457,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c[0]));
        vcdp->chgBit(c+7465,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c[0]));
        vcdp->chgBus(c+7473,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c[0]),16);
        vcdp->chgBus(c+7481,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c[0]),20);
        vcdp->chgArray(c+7489,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c[0]),128);
        vcdp->chgBit(c+7521,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid 
                                            >> (0x3fU 
                                                & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                                [0U]))))));
        vcdp->chgBit(c+7529,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty 
                                            >> (0x3fU 
                                                & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                                [0U]))))));
        vcdp->chgBus(c+7537,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                             [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                               [0U])]),16);
        vcdp->chgBus(c+7545,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag
                             [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                               [0U])]),20);
        __Vtemp700[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
              [0U])][0U];
        __Vtemp700[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
              [0U])][1U];
        __Vtemp700[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
              [0U])][2U];
        __Vtemp700[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
              [0U])][3U];
        vcdp->chgArray(c+7553,(__Vtemp700),128);
        vcdp->chgBit(c+7585,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                             [0U]));
        vcdp->chgBit(c+7593,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c
                             [0U]));
        vcdp->chgBus(c+7601,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable),16);
        vcdp->chgArray(c+7609,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__data_write),128);
        vcdp->chgBit(c+7641,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__snoop_hit_no_pending));
        vcdp->chgBit(c+7649,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match));
        vcdp->chgBit(c+7657,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill));
        vcdp->chgBus(c+7665,((0xfffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                          [0U] >> 6U))),20);
        vcdp->chgBus(c+7673,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__we),16);
        vcdp->chgBit(c+7681,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__should_write));
        vcdp->chgBit(c+7689,((((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                                [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1
                                        [0U])) & (~ 
                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                                  [0U])) 
                              & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                                 [0U]))));
        vcdp->chgBit(c+7697,(((((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                                 [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1
                                         [0U])) & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                [0U]) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                                         [0U])) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match)))));
        vcdp->chgBit(c+7705,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss));
        vcdp->chgBit(c+7713,((((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                 & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1
                                    [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                                              [0U])) 
                               & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                               [0U]) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss)))));
        vcdp->chgBit(c+7721,(((0U == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+7729,(((1U == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+7737,(((2U == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+7745,(((3U == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+7753,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable))));
        vcdp->chgArray(c+7761,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT____Vcellinp__s0_1_c0__in),166);
        vcdp->chgArray(c+7809,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__st_1e_2__in),316);
        vcdp->chgBus(c+7889,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready),16);
        vcdp->chgBus(c+7897,((0xffffU & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                                          & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                         << (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr)))),16);
        vcdp->chgBus(c+7905,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match),16);
        vcdp->chgBit(c+7913,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push));
        vcdp->chgBit(c+7921,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_pop));
        vcdp->chgBit(c+7929,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_is_mrvq))));
        vcdp->chgBit(c+7937,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__increment_head));
        vcdp->chgBit(c+7945,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready))));
        vcdp->chgBit(c+7953,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_init_ready_state_st2))));
        vcdp->chgArray(c+7961,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__cwb_queue__data_in),76);
        vcdp->chgArray(c+7985,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out),76);
        vcdp->chgBit(c+8009,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__writing));
        vcdp->chgArray(c+8017,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__dwb_queue__data_in),200);
        vcdp->chgArray(c+8073,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out),200);
        vcdp->chgBit(c+8129,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__writing));
        vcdp->chgBit(c+8137,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop));
        vcdp->chgBus(c+8145,((0x3ffffffU & (IData)(
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                    >> 0x1dU)))),26);
        vcdp->chgBit(c+8153,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                            >> 0x1cU)))));
        vcdp->chgBus(c+8161,((0xfffffffU & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out))),28);
        vcdp->chgBit(c+8169,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop));
        vcdp->chgBus(c+8177,((0x3ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])),26);
        __Vtemp701[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U];
        __Vtemp701[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U];
        __Vtemp701[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U];
        __Vtemp701[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U];
        vcdp->chgArray(c+8185,(__Vtemp701),128);
        vcdp->chgBit(c+8217,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop));
        vcdp->chgBit(c+8225,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request)))));
        vcdp->chgBit(c+8233,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request));
        vcdp->chgBus(c+8241,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index),2);
        vcdp->chgBit(c+8249,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_req_rw_st0));
        vcdp->chgBus(c+8257,((0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                      >> (0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                  << 2U))))),4);
        vcdp->chgBus(c+8265,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_req_addr_st0),30);
        vcdp->chgBus(c+8273,((((0U == (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
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
        vcdp->chgBit(c+8281,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_pop));
        vcdp->chgBit(c+8289,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible));
        vcdp->chgBit(c+8297,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_rw_st0));
        vcdp->chgBit(c+8305,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match))));
        vcdp->chgBit(c+8313,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_because_miss) 
                              & (((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 7U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x19U))) 
                                  == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                  [0U]) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              >> 0x14U))))));
        vcdp->chgBit(c+8321,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__force_request_miss_st1e));
        vcdp->chgBit(c+8329,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__recover_mrvq_state_st2));
        vcdp->chgBit(c+8337,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_push_stall));
        vcdp->chgBit(c+8345,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwbq_push_stall));
        vcdp->chgBit(c+8353,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_push_stall));
        vcdp->chgBit(c+8361,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dram_fill_req_stall));
        vcdp->chgBit(c+8369,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__stall_bank_pipe));
        vcdp->chgBit(c+8377,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_in_pipe));
        vcdp->chgBit(c+8385,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1[0]));
        vcdp->chgBit(c+8393,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__going_to_write_st1[0]));
        vcdp->chgBit(c+8401,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual));
        vcdp->chgBit(c+8409,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop_unqual));
        vcdp->chgBit(c+8417,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop_unqual));
        vcdp->chgBit(c+8425,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop) 
                                | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_pop)) 
                               | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop)) 
                              | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop))));
        vcdp->chgBus(c+8433,((0x3ffffffU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                             ? (0x3ffffffU 
                                                & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])
                                             : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                                 ? 
                                                ((0x19fU 
                                                  >= 
                                                  (0x1ffU 
                                                   & ((IData)(0x1aU) 
                                                      * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                  ? 
                                                 (0x3ffffffU 
                                                  & (((0U 
                                                       == 
                                                       (0x1fU 
                                                        & ((IData)(0x1aU) 
                                                           * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                       ? 0U
                                                       : 
                                                      (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                       ((IData)(1U) 
                                                        + 
                                                        (0xfU 
                                                         & (((IData)(0x1aU) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                            >> 5U)))] 
                                                       << 
                                                       ((IData)(0x20U) 
                                                        - 
                                                        (0x1fU 
                                                         & ((IData)(0x1aU) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                     | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                        (0xfU 
                                                         & (((IData)(0x1aU) 
                                                             * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                            >> 5U))] 
                                                        >> 
                                                        (0x1fU 
                                                         & ((IData)(0x1aU) 
                                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                                  : 0U)
                                                 : 
                                                ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop_unqual)
                                                  ? 
                                                 (0x3ffffffU 
                                                  & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_req_addr_st0 
                                                     >> 4U))
                                                  : 
                                                 ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop_unqual)
                                                   ? 
                                                  (0x3ffffffU 
                                                   & (IData)(
                                                             (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                              >> 0x1dU)))
                                                   : 0U)))))),26);
        vcdp->chgBus(c+8441,((3U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop_unqual)
                                     ? (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_req_addr_st0)
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                         ? (3U & ((
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                   << 0x1eU) 
                                                  | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                     [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                     >> 2U)))
                                         : 0U)))),2);
        vcdp->chgBus(c+8449,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
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
        __Vtemp706[0U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U]
                           : 0x39U);
        __Vtemp706[1U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U]
                           : 0U);
        __Vtemp706[2U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U]
                           : 0U);
        __Vtemp706[3U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U]
                           : 0U);
        vcdp->chgArray(c+8457,(__Vtemp706),128);
        vcdp->chgQuad(c+8489,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                ? ((VL_ULL(0x1ffffffffff80) 
                                    & (((QData)((IData)(
                                                        vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                        << 0x3eU) | 
                                       (((QData)((IData)(
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
        vcdp->chgBit(c+8505,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
                               ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible) 
                                        & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_rw_st0))
                                        ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop_unqual) 
                                                 & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_req_rw_st0))
                                                 ? 1U
                                                 : 0U)))));
        vcdp->chgBit(c+8513,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                     ? (1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                              [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                              >> 1U))
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? 1U : 0U)))));
        vcdp->chgBit(c+8521,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                     ? (1U & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])
                                     : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop_unqual)
                                         ? (1U & (IData)(
                                                         (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                          >> 0x1cU)))
                                         : 0U)))));
        vcdp->chgBit(c+8529,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1[0]));
        vcdp->chgBus(c+8537,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1[0]),26);
        vcdp->chgBus(c+8545,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1[0]),2);
        vcdp->chgBus(c+8553,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writeword_st1[0]),32);
        vcdp->chgQuad(c+8561,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__inst_meta_st1[0]),49);
        vcdp->chgArray(c+8577,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writedata_st1[0]),128);
        vcdp->chgBit(c+8609,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1[0]));
        vcdp->chgBit(c+8617,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_invalidate_st1[0]));
        vcdp->chgBit(c+8625,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_mrvq_st1[0]));
        __Vtemp712[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp712[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp712[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp712[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgBus(c+8633,((((0U == (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                                [0U] 
                                                << 5U)))
                                ? 0U : (__Vtemp712[
                                        ((IData)(1U) 
                                         + (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                            [0U]))] 
                                        << ((IData)(0x20U) 
                                            - (0x1fU 
                                               & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                                  [0U] 
                                                  << 5U))))) 
                              | (__Vtemp712[(3U & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                             [0U])] 
                                 >> (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                              [0U] 
                                              << 5U))))),32);
        __Vtemp713[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp713[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp713[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp713[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->chgArray(c+8641,(__Vtemp713),128);
        vcdp->chgBus(c+8673,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c
                             [0U]),20);
        vcdp->chgBit(c+8681,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_st1e));
        vcdp->chgBit(c+8689,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dirty_st1e));
        vcdp->chgBus(c+8697,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c
                             [0U]),16);
        vcdp->chgQuad(c+8705,((VL_ULL(0x3ffffffffff) 
                               & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__inst_meta_st1
                                  [0U] >> 7U))),42);
        vcdp->chgBus(c+8721,((3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__inst_meta_st1
                                           [0U]))),2);
        vcdp->chgBit(c+8729,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__inst_meta_st1
                                            [0U] >> 6U)))));
        vcdp->chgBus(c+8737,((0xfU & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__inst_meta_st1
                                              [0U] 
                                              >> 2U)))),4);
        vcdp->chgBit(c+8745,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dirty_st1e))));
        vcdp->chgBit(c+8753,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1
                             [0U]));
        vcdp->chgBit(c+8761,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_invalidate_st1
                             [0U]));
        vcdp->chgBit(c+8769,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_to_mrvq_st1e));
        vcdp->chgBit(c+8777,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_to_mrvq_st1e) 
                              | ((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                   & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1
                                      [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                                                [0U])) 
                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                                 [0U]))));
        vcdp->chgBit(c+8785,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_because_miss));
        vcdp->chgBit(c+8793,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                             [0U]));
        vcdp->chgBit(c+8801,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_mrvq_st1
                             [0U]));
        vcdp->chgBit(c+8809,((((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                                [0U] & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_mrvq_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__recover_mrvq_state_st2)) 
                              & ((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                 << 7U) 
                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                   >> 0x19U))) 
                                 == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                 [0U]))));
        vcdp->chgBus(c+8817,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                             [0U]),26);
        vcdp->chgBit(c+8825,((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                              [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                                      [0U]))));
        vcdp->chgBit(c+8833,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_init_ready_state_st2));
        vcdp->chgBit(c+8841,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_unqual) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)) 
                              & ((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                 << 7U) 
                                                | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                   >> 0x19U))) 
                                 == (0x3ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])))));
        vcdp->chgBit(c+8849,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_unqual) 
                               & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                               [0U]) & ((0x3ffffffU 
                                         & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                             << 7U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                               >> 0x19U))) 
                                        == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                        [0U]))));
        vcdp->chgBit(c+8857,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_unqual));
        vcdp->chgBit(c+8865,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add));
        vcdp->chgBit(c+8873,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_is_mrvq));
        vcdp->chgBit(c+8881,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwbq_push_unqual) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))) & (~ 
                                                 (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_push_stall) 
                                                   | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                                  | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+8889,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwbq_push_unqual));
        vcdp->chgBit(c+8897,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_unqual) 
                              & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                 | ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU) & (~ 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                   >> 0x1bU)))))));
        vcdp->chgBit(c+8905,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_push_unqual) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwbq_push_stall) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->chgBit(c+8913,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_is_dwb_in));
        vcdp->chgBit(c+8921,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_is_snp_in));
        vcdp->chgBit(c+8929,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 7U))));
        vcdp->chgBit(c+8937,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                    >> 6U))));
        vcdp->chgBit(c+8945,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_push_unqual));
        vcdp->chgQuad(c+8953,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out),55);
        vcdp->chgBit(c+8969,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__reading));
        vcdp->chgArray(c+8977,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out),154);
        vcdp->chgBit(c+9017,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__reading));
        vcdp->chgBus(c+9025,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                      >> 0x16U))),4);
        vcdp->chgBus(c+9033,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                      >> 0x12U))),4);
        vcdp->chgBus(c+9041,((0xffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                         >> 2U))),16);
        __Vtemp716[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                                        >> 0xaU));
        __Vtemp716[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                                        >> 0xaU));
        __Vtemp716[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                                        >> 0xaU));
        __Vtemp716[3U] = (0xffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                        << 0x16U) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                                        >> 0xaU)));
        vcdp->chgArray(c+9049,(__Vtemp716),120);
        __Vtemp717[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U] 
                                        >> 0xaU));
        __Vtemp717[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                                        >> 0xaU));
        __Vtemp717[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                                        >> 0xaU));
        __Vtemp717[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                                        >> 0xaU));
        vcdp->chgArray(c+9081,(__Vtemp717),128);
        vcdp->chgQuad(c+9113,((VL_ULL(0x3ffffffffff) 
                               & (((QData)((IData)(
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U])) 
                                   << 0x20U) | (QData)((IData)(
                                                               vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[0U]))))),42);
        vcdp->chgBit(c+9129,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty));
        vcdp->chgBit(c+9137,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__pop_qual));
        vcdp->chgBus(c+9145,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x16U) & 
                                      VL_NEGATE_I((IData)(
                                                          (1U 
                                                           & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty)))))))),4);
        vcdp->chgArray(c+9153,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out),314);
        vcdp->chgBit(c+9233,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__reading));
        vcdp->chgBus(c+9241,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->chgBus(c+9249,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->chgArray(c+9257,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__s0_1_c0__in),243);
        vcdp->chgBus(c+9321,((0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                              [0U])),6);
        vcdp->chgBit(c+9329,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                             [0U]));
        vcdp->chgBus(c+9337,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writeword_st1
                             [0U]),32);
        __Vtemp718[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writedata_st1
            [0U][0U];
        __Vtemp718[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writedata_st1
            [0U][1U];
        __Vtemp718[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writedata_st1
            [0U][2U];
        __Vtemp718[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writedata_st1
            [0U][3U];
        vcdp->chgArray(c+9345,(__Vtemp718),128);
        vcdp->chgBus(c+9377,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                             [0U]),2);
        vcdp->chgBit(c+9385,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c[0]));
        vcdp->chgBit(c+9393,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c[0]));
        vcdp->chgBus(c+9401,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c[0]),16);
        vcdp->chgBus(c+9409,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c[0]),20);
        vcdp->chgArray(c+9417,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c[0]),128);
        vcdp->chgBit(c+9449,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid 
                                            >> (0x3fU 
                                                & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                                [0U]))))));
        vcdp->chgBit(c+9457,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty 
                                            >> (0x3fU 
                                                & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                                [0U]))))));
        vcdp->chgBus(c+9465,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                             [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                               [0U])]),16);
        vcdp->chgBus(c+9473,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag
                             [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                               [0U])]),20);
        __Vtemp719[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
              [0U])][0U];
        __Vtemp719[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
              [0U])][1U];
        __Vtemp719[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
              [0U])][2U];
        __Vtemp719[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
              [0U])][3U];
        vcdp->chgArray(c+9481,(__Vtemp719),128);
        vcdp->chgBit(c+9513,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                             [0U]));
        vcdp->chgBit(c+9521,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c
                             [0U]));
        vcdp->chgBus(c+9529,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable),16);
        vcdp->chgArray(c+9537,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__data_write),128);
        vcdp->chgBit(c+9569,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__snoop_hit_no_pending));
        vcdp->chgBit(c+9577,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match));
        vcdp->chgBit(c+9585,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill));
        vcdp->chgBus(c+9593,((0xfffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                          [0U] >> 6U))),20);
        vcdp->chgBus(c+9601,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__we),16);
        vcdp->chgBit(c+9609,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__should_write));
        vcdp->chgBit(c+9617,((((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                                [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1
                                        [0U])) & (~ 
                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                                  [0U])) 
                              & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                                 [0U]))));
        vcdp->chgBit(c+9625,(((((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                                 [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1
                                         [0U])) & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                [0U]) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                                         [0U])) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match)))));
        vcdp->chgBit(c+9633,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss));
        vcdp->chgBit(c+9641,((((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                 & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1
                                    [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                                              [0U])) 
                               & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                               [0U]) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss)))));
        vcdp->chgBit(c+9649,(((0U == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+9657,(((1U == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+9665,(((2U == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+9673,(((3U == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                               [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->chgBit(c+9681,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable))));
        vcdp->chgArray(c+9689,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT____Vcellinp__s0_1_c0__in),166);
        vcdp->chgArray(c+9737,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__st_1e_2__in),316);
        vcdp->chgBus(c+9817,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready),16);
        vcdp->chgBus(c+9825,((0xffffU & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                                          & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                         << (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr)))),16);
        vcdp->chgBus(c+9833,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match),16);
        vcdp->chgBit(c+9841,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push));
        vcdp->chgBit(c+9849,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_pop));
        vcdp->chgBit(c+9857,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_is_mrvq))));
        vcdp->chgBit(c+9865,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__increment_head));
        vcdp->chgBit(c+9873,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready))));
        vcdp->chgBit(c+9881,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                              & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_init_ready_state_st2))));
        vcdp->chgArray(c+9889,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__cwb_queue__data_in),76);
        vcdp->chgArray(c+9913,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out),76);
        vcdp->chgBit(c+9937,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__writing));
        vcdp->chgArray(c+9945,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__dwb_queue__data_in),200);
        vcdp->chgArray(c+10001,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out),200);
        vcdp->chgBit(c+10057,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__writing));
    }
}

void VVX_cache::traceChgThis__5(VVX_cache__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code) {
    VVX_cache* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    int c = code;
    if (0 && vcdp && c) {}  // Prevent unused
    // Variables
    WData/*127:0*/ __Vtemp727[4];
    WData/*127:0*/ __Vtemp735[4];
    WData/*127:0*/ __Vtemp743[4];
    WData/*127:0*/ __Vtemp751[4];
    // Body
    {
        vcdp->chgBit(c+10065,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+10073,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->chgBit(c+10081,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBus(c+10089,((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 7U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x19U)))),26);
        vcdp->chgBit(c+10097,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+10105,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+10113,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->chgBit(c+10121,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBus(c+10129,((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 7U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x19U)))),26);
        vcdp->chgBit(c+10137,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+10145,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+10153,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->chgBit(c+10161,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBus(c+10169,((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 7U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x19U)))),26);
        vcdp->chgBit(c+10177,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+10185,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+10193,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->chgBit(c+10201,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBus(c+10209,((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 7U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x19U)))),26);
        vcdp->chgBit(c+10217,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBit(c+10225,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->chgBus(c+10233,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__use_addr),28);
        vcdp->chgBit(c+10241,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBus(c+10249,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__use_valid),2);
        vcdp->chgBit(c+10257,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__size_r));
        vcdp->chgBus(c+10265,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__genblk2__DOT__head_r),28);
        vcdp->chgBit(c+10273,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__size_r)))));
        vcdp->chgBit(c+10281,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__size_r));
        vcdp->chgBus(c+10289,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bank_dram_fill_req_valid),4);
        vcdp->chgArray(c+10297,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bank_dram_fill_req_addr),112);
        vcdp->chgBit(c+10329,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+10337,((1U & (~ (IData)((0U 
                                                != (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bank_dram_fill_req_valid)))))));
        vcdp->chgBus(c+10345,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__size_r),3);
        vcdp->chgArray(c+10353,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__data[0]),116);
        vcdp->chgArray(c+10357,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__data[1]),116);
        vcdp->chgArray(c+10361,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__data[2]),116);
        vcdp->chgArray(c+10365,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__data[3]),116);
        vcdp->chgArray(c+10481,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),116);
        vcdp->chgArray(c+10513,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),116);
        vcdp->chgBus(c+10545,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+10553,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+10561,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+10569,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+10577,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__requests_use),4);
        vcdp->chgBit(c+10585,((0U == (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__requests_use))));
        vcdp->chgBus(c+10593,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__refill_original),4);
        vcdp->chgBit(c+10601,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+10609,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+10617,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+10625,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+10633,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgQuad(c+10641,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag),42);
        vcdp->chgBit(c+10657,((0x10U == (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBit(c+10665,((0xbU < (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+10673,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                      << 0xdU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                  >> 0x13U)))),2);
        vcdp->chgBus(c+10681,(((0x19fU >= (0x1ffU & 
                                           ((IData)(0x1aU) 
                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                ? (0x3ffffffU & (((0U 
                                                   == 
                                                   (0x1fU 
                                                    & ((IData)(0x1aU) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                   ? 0U
                                                   : 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                   ((IData)(1U) 
                                                    + 
                                                    (0xfU 
                                                     & (((IData)(0x1aU) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U)))] 
                                                   << 
                                                   ((IData)(0x20U) 
                                                    - 
                                                    (0x1fU 
                                                     & ((IData)(0x1aU) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                    (0xfU 
                                                     & (((IData)(0x1aU) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U))] 
                                                    >> 
                                                    (0x1fU 
                                                     & ((IData)(0x1aU) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                : 0U)),26);
        vcdp->chgBus(c+10689,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                      << 0x1eU) | (
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                   >> 2U)))),2);
        vcdp->chgBus(c+10697,(((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                            [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                            >> 0x15U))),32);
        vcdp->chgQuad(c+10705,((VL_ULL(0x3ffffffffff) 
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
        vcdp->chgBus(c+10721,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                        << 0x1cU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                        >> 4U)))),4);
        vcdp->chgBit(c+10729,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                     [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                     >> 1U))));
        vcdp->chgBit(c+10737,((1U & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                               [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])));
        vcdp->chgBus(c+10745,((3U & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])),2);
        vcdp->chgQuad(c+10753,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U])) 
                                    << 0x39U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                                  << 0x19U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])) 
                                                    >> 7U))))),42);
        vcdp->chgBit(c+10769,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))));
        vcdp->chgBus(c+10777,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                        << 0x1eU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                        >> 2U)))),4);
        vcdp->chgBit(c+10785,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x14U))));
        vcdp->chgBit(c+10793,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x13U))));
        vcdp->chgBus(c+10801,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x17U)))),2);
        vcdp->chgBus(c+10809,(((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                          >> 0x17U))),32);
        vcdp->chgBus(c+10817,(((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                                          >> 0x17U))),32);
        __Vtemp727[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 0x17U));
        __Vtemp727[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                     >> 0x17U));
        __Vtemp727[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                                     >> 0x17U));
        __Vtemp727[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                                     >> 0x17U));
        vcdp->chgArray(c+10825,(__Vtemp727),128);
        vcdp->chgBit(c+10857,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 2U))));
        vcdp->chgBit(c+10865,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 1U))));
        vcdp->chgBus(c+10873,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                           << 0xfU) 
                                          | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                             >> 0x11U)))),16);
        vcdp->chgQuad(c+10881,((VL_ULL(0x1ffffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U]))))),49);
        vcdp->chgBus(c+10897,((0xfffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                            << 0x1dU) 
                                           | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                              >> 3U)))),20);
        vcdp->chgBit(c+10905,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x15U))));
        vcdp->chgBit(c+10913,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x17U))));
        vcdp->chgBit(c+10921,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x16U))));
        vcdp->chgBit(c+10929,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x18U))));
        vcdp->chgBit(c+10937,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU))));
        vcdp->chgBit(c+10945,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1bU))));
        vcdp->chgBit(c+10953,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x19U))));
        vcdp->chgBit(c+10961,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+10969,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+10977,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+10985,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBus(c+10993,(((0x3ffffc0U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                              << 3U)) 
                               | (0x3fU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                            << 7U) 
                                           | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                              >> 0x19U))))),26);
        vcdp->chgBus(c+11001,((0xfffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                              << 0x19U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                                >> 7U)))),28);
        vcdp->chgBit(c+11009,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_dual_valid_sel));
        vcdp->chgBus(c+11017,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__size_r),5);
        vcdp->chgQuad(c+11025,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[0]),55);
        vcdp->chgQuad(c+11027,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[1]),55);
        vcdp->chgQuad(c+11029,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[2]),55);
        vcdp->chgQuad(c+11031,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[3]),55);
        vcdp->chgQuad(c+11033,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[4]),55);
        vcdp->chgQuad(c+11035,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[5]),55);
        vcdp->chgQuad(c+11037,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[6]),55);
        vcdp->chgQuad(c+11039,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[7]),55);
        vcdp->chgQuad(c+11041,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[8]),55);
        vcdp->chgQuad(c+11043,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[9]),55);
        vcdp->chgQuad(c+11045,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[10]),55);
        vcdp->chgQuad(c+11047,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[11]),55);
        vcdp->chgQuad(c+11049,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[12]),55);
        vcdp->chgQuad(c+11051,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[13]),55);
        vcdp->chgQuad(c+11053,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[14]),55);
        vcdp->chgQuad(c+11055,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[15]),55);
        vcdp->chgQuad(c+11281,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),55);
        vcdp->chgQuad(c+11297,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),55);
        vcdp->chgBus(c+11313,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+11321,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+11329,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+11337,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+11345,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__size_r),5);
        vcdp->chgArray(c+11353,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[0]),154);
        vcdp->chgArray(c+11358,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[1]),154);
        vcdp->chgArray(c+11363,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[2]),154);
        vcdp->chgArray(c+11368,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[3]),154);
        vcdp->chgArray(c+11373,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[4]),154);
        vcdp->chgArray(c+11378,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[5]),154);
        vcdp->chgArray(c+11383,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[6]),154);
        vcdp->chgArray(c+11388,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[7]),154);
        vcdp->chgArray(c+11393,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[8]),154);
        vcdp->chgArray(c+11398,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[9]),154);
        vcdp->chgArray(c+11403,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[10]),154);
        vcdp->chgArray(c+11408,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[11]),154);
        vcdp->chgArray(c+11413,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[12]),154);
        vcdp->chgArray(c+11418,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[13]),154);
        vcdp->chgArray(c+11423,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[14]),154);
        vcdp->chgArray(c+11428,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[15]),154);
        vcdp->chgArray(c+11993,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),154);
        vcdp->chgArray(c+12033,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),154);
        vcdp->chgBus(c+12073,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+12081,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+12089,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+12097,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+12105,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids),4);
        vcdp->chgBus(c+12113,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_rw),4);
        vcdp->chgBus(c+12121,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen),16);
        vcdp->chgArray(c+12129,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_addr),120);
        vcdp->chgArray(c+12161,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata),128);
        vcdp->chgBit(c+12193,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+12201,((1U & (~ (IData)((0U 
                                                != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids)))))));
        vcdp->chgBus(c+12209,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__size_r),3);
        vcdp->chgArray(c+12217,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[0]),314);
        vcdp->chgArray(c+12227,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[1]),314);
        vcdp->chgArray(c+12237,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[2]),314);
        vcdp->chgArray(c+12247,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[3]),314);
        vcdp->chgArray(c+12537,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),314);
        vcdp->chgArray(c+12617,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),314);
        vcdp->chgBus(c+12697,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+12705,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+12713,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+12721,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgArray(c+12729,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__s0_1_c0__DOT__value),243);
        vcdp->chgQuad(c+12793,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty),64);
        vcdp->chgQuad(c+12809,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid),64);
        vcdp->chgBus(c+12825,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__i),32);
        vcdp->chgBus(c+12833,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__j),32);
        vcdp->chgArray(c+12841,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__s0_1_c0__DOT__value),166);
        vcdp->chgArray(c+12889,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value),316);
        vcdp->chgArray(c+12969,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[0]),85);
        vcdp->chgArray(c+12972,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[1]),85);
        vcdp->chgArray(c+12975,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[2]),85);
        vcdp->chgArray(c+12978,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[3]),85);
        vcdp->chgArray(c+12981,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[4]),85);
        vcdp->chgArray(c+12984,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[5]),85);
        vcdp->chgArray(c+12987,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[6]),85);
        vcdp->chgArray(c+12990,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[7]),85);
        vcdp->chgArray(c+12993,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[8]),85);
        vcdp->chgArray(c+12996,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[9]),85);
        vcdp->chgArray(c+12999,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[10]),85);
        vcdp->chgArray(c+13002,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[11]),85);
        vcdp->chgArray(c+13005,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[12]),85);
        vcdp->chgArray(c+13008,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[13]),85);
        vcdp->chgArray(c+13011,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[14]),85);
        vcdp->chgArray(c+13014,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[15]),85);
        vcdp->chgArray(c+13353,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table),416);
        vcdp->chgBus(c+13457,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_table),16);
        vcdp->chgBus(c+13465,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__ready_table),16);
        vcdp->chgBus(c+13473,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr),4);
        vcdp->chgBus(c+13481,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__head_ptr),4);
        vcdp->chgBus(c+13489,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr),4);
        vcdp->chgBus(c+13497,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size),5);
        vcdp->chgBit(c+13505,((0x10U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+13513,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__size_r),3);
        vcdp->chgArray(c+13521,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[0]),76);
        vcdp->chgArray(c+13524,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[1]),76);
        vcdp->chgArray(c+13527,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[2]),76);
        vcdp->chgArray(c+13530,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[3]),76);
        vcdp->chgArray(c+13617,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),76);
        vcdp->chgArray(c+13641,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),76);
        vcdp->chgBus(c+13665,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+13673,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+13681,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+13689,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+13697,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__size_r),3);
        vcdp->chgArray(c+13705,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[0]),200);
        vcdp->chgArray(c+13712,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[1]),200);
        vcdp->chgArray(c+13719,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[2]),200);
        vcdp->chgArray(c+13726,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[3]),200);
        vcdp->chgArray(c+13929,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),200);
        vcdp->chgArray(c+13985,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),200);
        vcdp->chgBus(c+14041,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+14049,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+14057,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+14065,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBit(c+14073,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+14081,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+14089,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+14097,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+14105,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgQuad(c+14113,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag),42);
        vcdp->chgBit(c+14129,((0x10U == (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBit(c+14137,((0xbU < (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+14145,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                      << 0xdU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                  >> 0x13U)))),2);
        vcdp->chgBus(c+14153,(((0x19fU >= (0x1ffU & 
                                           ((IData)(0x1aU) 
                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                ? (0x3ffffffU & (((0U 
                                                   == 
                                                   (0x1fU 
                                                    & ((IData)(0x1aU) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                   ? 0U
                                                   : 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                   ((IData)(1U) 
                                                    + 
                                                    (0xfU 
                                                     & (((IData)(0x1aU) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U)))] 
                                                   << 
                                                   ((IData)(0x20U) 
                                                    - 
                                                    (0x1fU 
                                                     & ((IData)(0x1aU) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                    (0xfU 
                                                     & (((IData)(0x1aU) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U))] 
                                                    >> 
                                                    (0x1fU 
                                                     & ((IData)(0x1aU) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                : 0U)),26);
        vcdp->chgBus(c+14161,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                      << 0x1eU) | (
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                   >> 2U)))),2);
        vcdp->chgBus(c+14169,(((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                            [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                            >> 0x15U))),32);
        vcdp->chgQuad(c+14177,((VL_ULL(0x3ffffffffff) 
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
        vcdp->chgBus(c+14193,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                        << 0x1cU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                        >> 4U)))),4);
        vcdp->chgBit(c+14201,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                     [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                     >> 1U))));
        vcdp->chgBit(c+14209,((1U & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                               [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])));
        vcdp->chgBus(c+14217,((3U & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])),2);
        vcdp->chgQuad(c+14225,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U])) 
                                    << 0x39U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                                  << 0x19U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])) 
                                                    >> 7U))))),42);
        vcdp->chgBit(c+14241,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))));
        vcdp->chgBus(c+14249,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                        << 0x1eU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                        >> 2U)))),4);
        vcdp->chgBit(c+14257,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x14U))));
        vcdp->chgBit(c+14265,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x13U))));
        vcdp->chgBus(c+14273,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x17U)))),2);
        vcdp->chgBus(c+14281,(((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                          >> 0x17U))),32);
        vcdp->chgBus(c+14289,(((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                                          >> 0x17U))),32);
        __Vtemp735[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 0x17U));
        __Vtemp735[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                     >> 0x17U));
        __Vtemp735[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                                     >> 0x17U));
        __Vtemp735[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                                     >> 0x17U));
        vcdp->chgArray(c+14297,(__Vtemp735),128);
        vcdp->chgBit(c+14329,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 2U))));
        vcdp->chgBit(c+14337,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 1U))));
        vcdp->chgBus(c+14345,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                           << 0xfU) 
                                          | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                             >> 0x11U)))),16);
        vcdp->chgQuad(c+14353,((VL_ULL(0x1ffffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U]))))),49);
        vcdp->chgBus(c+14369,((0xfffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                            << 0x1dU) 
                                           | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                              >> 3U)))),20);
        vcdp->chgBit(c+14377,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x15U))));
        vcdp->chgBit(c+14385,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x17U))));
        vcdp->chgBit(c+14393,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x16U))));
        vcdp->chgBit(c+14401,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x18U))));
        vcdp->chgBit(c+14409,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU))));
        vcdp->chgBit(c+14417,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1bU))));
        vcdp->chgBit(c+14425,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x19U))));
        vcdp->chgBit(c+14433,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+14441,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+14449,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+14457,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBus(c+14465,(((0x3ffffc0U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                              << 3U)) 
                               | (0x3fU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                            << 7U) 
                                           | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                              >> 0x19U))))),26);
        vcdp->chgBus(c+14473,((0xfffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                              << 0x19U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                                >> 7U)))),28);
        vcdp->chgBit(c+14481,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_dual_valid_sel));
        vcdp->chgBus(c+14489,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__size_r),5);
        vcdp->chgQuad(c+14497,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[0]),55);
        vcdp->chgQuad(c+14499,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[1]),55);
        vcdp->chgQuad(c+14501,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[2]),55);
        vcdp->chgQuad(c+14503,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[3]),55);
        vcdp->chgQuad(c+14505,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[4]),55);
        vcdp->chgQuad(c+14507,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[5]),55);
        vcdp->chgQuad(c+14509,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[6]),55);
        vcdp->chgQuad(c+14511,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[7]),55);
        vcdp->chgQuad(c+14513,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[8]),55);
        vcdp->chgQuad(c+14515,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[9]),55);
        vcdp->chgQuad(c+14517,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[10]),55);
        vcdp->chgQuad(c+14519,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[11]),55);
        vcdp->chgQuad(c+14521,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[12]),55);
        vcdp->chgQuad(c+14523,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[13]),55);
        vcdp->chgQuad(c+14525,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[14]),55);
        vcdp->chgQuad(c+14527,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[15]),55);
        vcdp->chgQuad(c+14753,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),55);
        vcdp->chgQuad(c+14769,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),55);
        vcdp->chgBus(c+14785,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+14793,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+14801,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+14809,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+14817,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__size_r),5);
        vcdp->chgArray(c+14825,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[0]),154);
        vcdp->chgArray(c+14830,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[1]),154);
        vcdp->chgArray(c+14835,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[2]),154);
        vcdp->chgArray(c+14840,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[3]),154);
        vcdp->chgArray(c+14845,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[4]),154);
        vcdp->chgArray(c+14850,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[5]),154);
        vcdp->chgArray(c+14855,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[6]),154);
        vcdp->chgArray(c+14860,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[7]),154);
        vcdp->chgArray(c+14865,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[8]),154);
        vcdp->chgArray(c+14870,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[9]),154);
        vcdp->chgArray(c+14875,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[10]),154);
        vcdp->chgArray(c+14880,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[11]),154);
        vcdp->chgArray(c+14885,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[12]),154);
        vcdp->chgArray(c+14890,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[13]),154);
        vcdp->chgArray(c+14895,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[14]),154);
        vcdp->chgArray(c+14900,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[15]),154);
        vcdp->chgArray(c+15465,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),154);
        vcdp->chgArray(c+15505,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),154);
        vcdp->chgBus(c+15545,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+15553,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+15561,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+15569,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+15577,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids),4);
        vcdp->chgBus(c+15585,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_rw),4);
        vcdp->chgBus(c+15593,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen),16);
        vcdp->chgArray(c+15601,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_addr),120);
        vcdp->chgArray(c+15633,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata),128);
        vcdp->chgBit(c+15665,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+15673,((1U & (~ (IData)((0U 
                                                != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids)))))));
        vcdp->chgBus(c+15681,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__size_r),3);
        vcdp->chgArray(c+15689,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[0]),314);
        vcdp->chgArray(c+15699,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[1]),314);
        vcdp->chgArray(c+15709,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[2]),314);
        vcdp->chgArray(c+15719,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[3]),314);
        vcdp->chgArray(c+16009,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),314);
        vcdp->chgArray(c+16089,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),314);
        vcdp->chgBus(c+16169,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+16177,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+16185,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+16193,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgArray(c+16201,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__s0_1_c0__DOT__value),243);
        vcdp->chgQuad(c+16265,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty),64);
        vcdp->chgQuad(c+16281,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid),64);
        vcdp->chgBus(c+16297,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__i),32);
        vcdp->chgBus(c+16305,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__j),32);
        vcdp->chgArray(c+16313,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__s0_1_c0__DOT__value),166);
        vcdp->chgArray(c+16361,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value),316);
        vcdp->chgArray(c+16441,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[0]),85);
        vcdp->chgArray(c+16444,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[1]),85);
        vcdp->chgArray(c+16447,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[2]),85);
        vcdp->chgArray(c+16450,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[3]),85);
        vcdp->chgArray(c+16453,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[4]),85);
        vcdp->chgArray(c+16456,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[5]),85);
        vcdp->chgArray(c+16459,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[6]),85);
        vcdp->chgArray(c+16462,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[7]),85);
        vcdp->chgArray(c+16465,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[8]),85);
        vcdp->chgArray(c+16468,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[9]),85);
        vcdp->chgArray(c+16471,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[10]),85);
        vcdp->chgArray(c+16474,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[11]),85);
        vcdp->chgArray(c+16477,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[12]),85);
        vcdp->chgArray(c+16480,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[13]),85);
        vcdp->chgArray(c+16483,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[14]),85);
        vcdp->chgArray(c+16486,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[15]),85);
        vcdp->chgArray(c+16825,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table),416);
        vcdp->chgBus(c+16929,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_table),16);
        vcdp->chgBus(c+16937,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__ready_table),16);
        vcdp->chgBus(c+16945,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr),4);
        vcdp->chgBus(c+16953,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__head_ptr),4);
        vcdp->chgBus(c+16961,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr),4);
        vcdp->chgBus(c+16969,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size),5);
        vcdp->chgBit(c+16977,((0x10U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+16985,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__size_r),3);
        vcdp->chgArray(c+16993,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[0]),76);
        vcdp->chgArray(c+16996,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[1]),76);
        vcdp->chgArray(c+16999,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[2]),76);
        vcdp->chgArray(c+17002,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[3]),76);
        vcdp->chgArray(c+17089,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),76);
        vcdp->chgArray(c+17113,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),76);
        vcdp->chgBus(c+17137,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+17145,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+17153,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+17161,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+17169,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__size_r),3);
        vcdp->chgArray(c+17177,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[0]),200);
        vcdp->chgArray(c+17184,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[1]),200);
        vcdp->chgArray(c+17191,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[2]),200);
        vcdp->chgArray(c+17198,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[3]),200);
        vcdp->chgArray(c+17401,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),200);
        vcdp->chgArray(c+17457,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),200);
        vcdp->chgBus(c+17513,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+17521,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+17529,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+17537,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBit(c+17545,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+17553,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+17561,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+17569,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+17577,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgQuad(c+17585,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag),42);
        vcdp->chgBit(c+17601,((0x10U == (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBit(c+17609,((0xbU < (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+17617,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                      << 0xdU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                  >> 0x13U)))),2);
        vcdp->chgBus(c+17625,(((0x19fU >= (0x1ffU & 
                                           ((IData)(0x1aU) 
                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                ? (0x3ffffffU & (((0U 
                                                   == 
                                                   (0x1fU 
                                                    & ((IData)(0x1aU) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                   ? 0U
                                                   : 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                   ((IData)(1U) 
                                                    + 
                                                    (0xfU 
                                                     & (((IData)(0x1aU) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U)))] 
                                                   << 
                                                   ((IData)(0x20U) 
                                                    - 
                                                    (0x1fU 
                                                     & ((IData)(0x1aU) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                    (0xfU 
                                                     & (((IData)(0x1aU) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U))] 
                                                    >> 
                                                    (0x1fU 
                                                     & ((IData)(0x1aU) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                : 0U)),26);
        vcdp->chgBus(c+17633,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                      << 0x1eU) | (
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                   >> 2U)))),2);
        vcdp->chgBus(c+17641,(((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                            [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                            >> 0x15U))),32);
        vcdp->chgQuad(c+17649,((VL_ULL(0x3ffffffffff) 
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
        vcdp->chgBus(c+17665,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                        << 0x1cU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                        >> 4U)))),4);
        vcdp->chgBit(c+17673,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                     [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                     >> 1U))));
        vcdp->chgBit(c+17681,((1U & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                               [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])));
        vcdp->chgBus(c+17689,((3U & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])),2);
        vcdp->chgQuad(c+17697,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U])) 
                                    << 0x39U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                                  << 0x19U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])) 
                                                    >> 7U))))),42);
        vcdp->chgBit(c+17713,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))));
        vcdp->chgBus(c+17721,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                        << 0x1eU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                        >> 2U)))),4);
        vcdp->chgBit(c+17729,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x14U))));
        vcdp->chgBit(c+17737,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x13U))));
        vcdp->chgBus(c+17745,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x17U)))),2);
        vcdp->chgBus(c+17753,(((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                          >> 0x17U))),32);
        vcdp->chgBus(c+17761,(((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                                          >> 0x17U))),32);
        __Vtemp743[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 0x17U));
        __Vtemp743[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                     >> 0x17U));
        __Vtemp743[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                                     >> 0x17U));
        __Vtemp743[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                                     >> 0x17U));
        vcdp->chgArray(c+17769,(__Vtemp743),128);
        vcdp->chgBit(c+17801,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 2U))));
        vcdp->chgBit(c+17809,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 1U))));
        vcdp->chgBus(c+17817,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                           << 0xfU) 
                                          | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                             >> 0x11U)))),16);
        vcdp->chgQuad(c+17825,((VL_ULL(0x1ffffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U]))))),49);
        vcdp->chgBus(c+17841,((0xfffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                            << 0x1dU) 
                                           | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                              >> 3U)))),20);
        vcdp->chgBit(c+17849,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x15U))));
        vcdp->chgBit(c+17857,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x17U))));
        vcdp->chgBit(c+17865,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x16U))));
        vcdp->chgBit(c+17873,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x18U))));
        vcdp->chgBit(c+17881,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU))));
        vcdp->chgBit(c+17889,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1bU))));
        vcdp->chgBit(c+17897,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x19U))));
        vcdp->chgBit(c+17905,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+17913,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+17921,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+17929,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBus(c+17937,(((0x3ffffc0U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                              << 3U)) 
                               | (0x3fU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                            << 7U) 
                                           | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                              >> 0x19U))))),26);
        vcdp->chgBus(c+17945,((0xfffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                              << 0x19U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                                >> 7U)))),28);
        vcdp->chgBit(c+17953,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_dual_valid_sel));
        vcdp->chgBus(c+17961,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__size_r),5);
        vcdp->chgQuad(c+17969,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[0]),55);
        vcdp->chgQuad(c+17971,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[1]),55);
        vcdp->chgQuad(c+17973,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[2]),55);
        vcdp->chgQuad(c+17975,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[3]),55);
        vcdp->chgQuad(c+17977,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[4]),55);
        vcdp->chgQuad(c+17979,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[5]),55);
        vcdp->chgQuad(c+17981,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[6]),55);
        vcdp->chgQuad(c+17983,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[7]),55);
        vcdp->chgQuad(c+17985,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[8]),55);
        vcdp->chgQuad(c+17987,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[9]),55);
        vcdp->chgQuad(c+17989,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[10]),55);
        vcdp->chgQuad(c+17991,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[11]),55);
        vcdp->chgQuad(c+17993,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[12]),55);
        vcdp->chgQuad(c+17995,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[13]),55);
        vcdp->chgQuad(c+17997,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[14]),55);
        vcdp->chgQuad(c+17999,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[15]),55);
        vcdp->chgQuad(c+18225,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),55);
        vcdp->chgQuad(c+18241,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),55);
        vcdp->chgBus(c+18257,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+18265,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+18273,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+18281,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+18289,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__size_r),5);
        vcdp->chgArray(c+18297,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[0]),154);
        vcdp->chgArray(c+18302,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[1]),154);
        vcdp->chgArray(c+18307,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[2]),154);
        vcdp->chgArray(c+18312,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[3]),154);
        vcdp->chgArray(c+18317,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[4]),154);
        vcdp->chgArray(c+18322,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[5]),154);
        vcdp->chgArray(c+18327,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[6]),154);
        vcdp->chgArray(c+18332,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[7]),154);
        vcdp->chgArray(c+18337,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[8]),154);
        vcdp->chgArray(c+18342,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[9]),154);
        vcdp->chgArray(c+18347,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[10]),154);
        vcdp->chgArray(c+18352,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[11]),154);
        vcdp->chgArray(c+18357,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[12]),154);
        vcdp->chgArray(c+18362,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[13]),154);
        vcdp->chgArray(c+18367,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[14]),154);
        vcdp->chgArray(c+18372,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[15]),154);
        vcdp->chgArray(c+18937,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),154);
        vcdp->chgArray(c+18977,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),154);
        vcdp->chgBus(c+19017,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+19025,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+19033,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+19041,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+19049,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids),4);
        vcdp->chgBus(c+19057,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_rw),4);
        vcdp->chgBus(c+19065,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen),16);
        vcdp->chgArray(c+19073,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_addr),120);
        vcdp->chgArray(c+19105,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata),128);
        vcdp->chgBit(c+19137,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+19145,((1U & (~ (IData)((0U 
                                                != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids)))))));
        vcdp->chgBus(c+19153,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__size_r),3);
        vcdp->chgArray(c+19161,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[0]),314);
        vcdp->chgArray(c+19171,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[1]),314);
        vcdp->chgArray(c+19181,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[2]),314);
        vcdp->chgArray(c+19191,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[3]),314);
        vcdp->chgArray(c+19481,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),314);
        vcdp->chgArray(c+19561,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),314);
        vcdp->chgBus(c+19641,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+19649,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+19657,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+19665,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgArray(c+19673,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__s0_1_c0__DOT__value),243);
        vcdp->chgQuad(c+19737,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty),64);
        vcdp->chgQuad(c+19753,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid),64);
        vcdp->chgBus(c+19769,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__i),32);
        vcdp->chgBus(c+19777,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__j),32);
        vcdp->chgArray(c+19785,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__s0_1_c0__DOT__value),166);
        vcdp->chgArray(c+19833,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value),316);
        vcdp->chgArray(c+19913,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[0]),85);
        vcdp->chgArray(c+19916,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[1]),85);
        vcdp->chgArray(c+19919,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[2]),85);
        vcdp->chgArray(c+19922,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[3]),85);
        vcdp->chgArray(c+19925,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[4]),85);
        vcdp->chgArray(c+19928,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[5]),85);
        vcdp->chgArray(c+19931,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[6]),85);
        vcdp->chgArray(c+19934,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[7]),85);
        vcdp->chgArray(c+19937,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[8]),85);
        vcdp->chgArray(c+19940,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[9]),85);
        vcdp->chgArray(c+19943,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[10]),85);
        vcdp->chgArray(c+19946,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[11]),85);
        vcdp->chgArray(c+19949,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[12]),85);
        vcdp->chgArray(c+19952,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[13]),85);
        vcdp->chgArray(c+19955,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[14]),85);
        vcdp->chgArray(c+19958,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[15]),85);
        vcdp->chgArray(c+20297,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table),416);
        vcdp->chgBus(c+20401,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_table),16);
        vcdp->chgBus(c+20409,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__ready_table),16);
        vcdp->chgBus(c+20417,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr),4);
        vcdp->chgBus(c+20425,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__head_ptr),4);
        vcdp->chgBus(c+20433,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr),4);
        vcdp->chgBus(c+20441,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size),5);
        vcdp->chgBit(c+20449,((0x10U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+20457,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__size_r),3);
        vcdp->chgArray(c+20465,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[0]),76);
        vcdp->chgArray(c+20468,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[1]),76);
        vcdp->chgArray(c+20471,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[2]),76);
        vcdp->chgArray(c+20474,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[3]),76);
        vcdp->chgArray(c+20561,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),76);
        vcdp->chgArray(c+20585,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),76);
        vcdp->chgBus(c+20609,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+20617,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+20625,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+20633,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+20641,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__size_r),3);
        vcdp->chgArray(c+20649,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[0]),200);
        vcdp->chgArray(c+20656,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[1]),200);
        vcdp->chgArray(c+20663,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[2]),200);
        vcdp->chgArray(c+20670,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[3]),200);
        vcdp->chgArray(c+20873,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),200);
        vcdp->chgArray(c+20929,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),200);
        vcdp->chgBus(c+20985,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+20993,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+21001,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+21009,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBit(c+21017,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+21025,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+21033,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+21041,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+21049,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgQuad(c+21057,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag),42);
        vcdp->chgBit(c+21073,((0x10U == (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBit(c+21081,((0xbU < (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+21089,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                      << 0xdU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                  [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                  >> 0x13U)))),2);
        vcdp->chgBus(c+21097,(((0x19fU >= (0x1ffU & 
                                           ((IData)(0x1aU) 
                                            * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                ? (0x3ffffffU & (((0U 
                                                   == 
                                                   (0x1fU 
                                                    & ((IData)(0x1aU) 
                                                       * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                                   ? 0U
                                                   : 
                                                  (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                   ((IData)(1U) 
                                                    + 
                                                    (0xfU 
                                                     & (((IData)(0x1aU) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U)))] 
                                                   << 
                                                   ((IData)(0x20U) 
                                                    - 
                                                    (0x1fU 
                                                     & ((IData)(0x1aU) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)))))) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table[
                                                    (0xfU 
                                                     & (((IData)(0x1aU) 
                                                         * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr)) 
                                                        >> 5U))] 
                                                    >> 
                                                    (0x1fU 
                                                     & ((IData)(0x1aU) 
                                                        * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))))
                                : 0U)),26);
        vcdp->chgBus(c+21105,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                      << 0x1eU) | (
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                   >> 2U)))),2);
        vcdp->chgBus(c+21113,(((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                            [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                            >> 0x15U))),32);
        vcdp->chgQuad(c+21121,((VL_ULL(0x3ffffffffff) 
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
        vcdp->chgBus(c+21137,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                        << 0x1cU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                        [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                        >> 4U)))),4);
        vcdp->chgBit(c+21145,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                     [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                     >> 1U))));
        vcdp->chgBit(c+21153,((1U & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                               [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])));
        vcdp->chgBus(c+21161,((3U & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])),2);
        vcdp->chgQuad(c+21169,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U])) 
                                    << 0x39U) | (((QData)((IData)(
                                                                  vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                                  << 0x19U) 
                                                 | ((QData)((IData)(
                                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])) 
                                                    >> 7U))))),42);
        vcdp->chgBit(c+21185,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                     >> 6U))));
        vcdp->chgBus(c+21193,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                        << 0x1eU) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                        >> 2U)))),4);
        vcdp->chgBit(c+21201,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x14U))));
        vcdp->chgBit(c+21209,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x13U))));
        vcdp->chgBus(c+21217,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x17U)))),2);
        vcdp->chgBus(c+21225,(((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                          >> 0x17U))),32);
        vcdp->chgBus(c+21233,(((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                                          >> 0x17U))),32);
        __Vtemp751[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 0x17U));
        __Vtemp751[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                     >> 0x17U));
        __Vtemp751[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                                     >> 0x17U));
        __Vtemp751[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                                     >> 0x17U));
        vcdp->chgArray(c+21241,(__Vtemp751),128);
        vcdp->chgBit(c+21273,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 2U))));
        vcdp->chgBit(c+21281,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 1U))));
        vcdp->chgBus(c+21289,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                           << 0xfU) 
                                          | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                             >> 0x11U)))),16);
        vcdp->chgQuad(c+21297,((VL_ULL(0x1ffffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U]))))),49);
        vcdp->chgBus(c+21313,((0xfffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                            << 0x1dU) 
                                           | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                              >> 3U)))),20);
        vcdp->chgBit(c+21321,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x15U))));
        vcdp->chgBit(c+21329,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x17U))));
        vcdp->chgBit(c+21337,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x16U))));
        vcdp->chgBit(c+21345,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x18U))));
        vcdp->chgBit(c+21353,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1aU))));
        vcdp->chgBit(c+21361,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x1bU))));
        vcdp->chgBit(c+21369,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                     >> 0x19U))));
        vcdp->chgBit(c+21377,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+21385,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBit(c+21393,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+21401,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->chgBus(c+21409,(((0x3ffffc0U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                              << 3U)) 
                               | (0x3fU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                            << 7U) 
                                           | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                              >> 0x19U))))),26);
        vcdp->chgBus(c+21417,((0xfffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                              << 0x19U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                                >> 7U)))),28);
        vcdp->chgBit(c+21425,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_dual_valid_sel));
        vcdp->chgBus(c+21433,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__size_r),5);
        vcdp->chgQuad(c+21441,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[0]),55);
        vcdp->chgQuad(c+21443,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[1]),55);
        vcdp->chgQuad(c+21445,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[2]),55);
        vcdp->chgQuad(c+21447,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[3]),55);
        vcdp->chgQuad(c+21449,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[4]),55);
        vcdp->chgQuad(c+21451,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[5]),55);
        vcdp->chgQuad(c+21453,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[6]),55);
        vcdp->chgQuad(c+21455,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[7]),55);
        vcdp->chgQuad(c+21457,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[8]),55);
        vcdp->chgQuad(c+21459,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[9]),55);
        vcdp->chgQuad(c+21461,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[10]),55);
        vcdp->chgQuad(c+21463,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[11]),55);
        vcdp->chgQuad(c+21465,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[12]),55);
        vcdp->chgQuad(c+21467,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[13]),55);
        vcdp->chgQuad(c+21469,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[14]),55);
        vcdp->chgQuad(c+21471,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[15]),55);
        vcdp->chgQuad(c+21697,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),55);
        vcdp->chgQuad(c+21713,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),55);
        vcdp->chgBus(c+21729,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+21737,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+21745,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+21753,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+21761,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__size_r),5);
        vcdp->chgArray(c+21769,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[0]),154);
        vcdp->chgArray(c+21774,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[1]),154);
        vcdp->chgArray(c+21779,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[2]),154);
        vcdp->chgArray(c+21784,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[3]),154);
        vcdp->chgArray(c+21789,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[4]),154);
        vcdp->chgArray(c+21794,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[5]),154);
        vcdp->chgArray(c+21799,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[6]),154);
        vcdp->chgArray(c+21804,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[7]),154);
        vcdp->chgArray(c+21809,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[8]),154);
        vcdp->chgArray(c+21814,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[9]),154);
        vcdp->chgArray(c+21819,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[10]),154);
        vcdp->chgArray(c+21824,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[11]),154);
        vcdp->chgArray(c+21829,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[12]),154);
        vcdp->chgArray(c+21834,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[13]),154);
        vcdp->chgArray(c+21839,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[14]),154);
        vcdp->chgArray(c+21844,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[15]),154);
        vcdp->chgArray(c+22409,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),154);
        vcdp->chgArray(c+22449,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),154);
        vcdp->chgBus(c+22489,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->chgBus(c+22497,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->chgBus(c+22505,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->chgBit(c+22513,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+22521,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids),4);
        vcdp->chgBus(c+22529,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_rw),4);
        vcdp->chgBus(c+22537,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen),16);
        vcdp->chgArray(c+22545,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_addr),120);
        vcdp->chgArray(c+22577,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata),128);
        vcdp->chgBit(c+22609,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->chgBit(c+22617,((1U & (~ (IData)((0U 
                                                != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids)))))));
        vcdp->chgBus(c+22625,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__size_r),3);
        vcdp->chgArray(c+22633,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[0]),314);
        vcdp->chgArray(c+22643,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[1]),314);
        vcdp->chgArray(c+22653,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[2]),314);
        vcdp->chgArray(c+22663,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[3]),314);
        vcdp->chgArray(c+22953,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),314);
        vcdp->chgArray(c+23033,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),314);
        vcdp->chgBus(c+23113,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+23121,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+23129,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+23137,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgArray(c+23145,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__s0_1_c0__DOT__value),243);
        vcdp->chgQuad(c+23209,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty),64);
        vcdp->chgQuad(c+23225,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid),64);
        vcdp->chgBus(c+23241,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__i),32);
        vcdp->chgBus(c+23249,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__j),32);
        vcdp->chgArray(c+23257,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__s0_1_c0__DOT__value),166);
        vcdp->chgArray(c+23305,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value),316);
        vcdp->chgArray(c+23385,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[0]),85);
        vcdp->chgArray(c+23388,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[1]),85);
        vcdp->chgArray(c+23391,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[2]),85);
        vcdp->chgArray(c+23394,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[3]),85);
        vcdp->chgArray(c+23397,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[4]),85);
        vcdp->chgArray(c+23400,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[5]),85);
        vcdp->chgArray(c+23403,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[6]),85);
        vcdp->chgArray(c+23406,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[7]),85);
        vcdp->chgArray(c+23409,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[8]),85);
        vcdp->chgArray(c+23412,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[9]),85);
        vcdp->chgArray(c+23415,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[10]),85);
        vcdp->chgArray(c+23418,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[11]),85);
        vcdp->chgArray(c+23421,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[12]),85);
        vcdp->chgArray(c+23424,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[13]),85);
        vcdp->chgArray(c+23427,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[14]),85);
        vcdp->chgArray(c+23430,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[15]),85);
        vcdp->chgArray(c+23769,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table),416);
        vcdp->chgBus(c+23873,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_table),16);
        vcdp->chgBus(c+23881,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__ready_table),16);
        vcdp->chgBus(c+23889,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr),4);
        vcdp->chgBus(c+23897,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__head_ptr),4);
        vcdp->chgBus(c+23905,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr),4);
        vcdp->chgBus(c+23913,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size),5);
        vcdp->chgBit(c+23921,((0x10U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->chgBus(c+23929,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__size_r),3);
        vcdp->chgArray(c+23937,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[0]),76);
        vcdp->chgArray(c+23940,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[1]),76);
        vcdp->chgArray(c+23943,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[2]),76);
        vcdp->chgArray(c+23946,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[3]),76);
        vcdp->chgArray(c+24033,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),76);
        vcdp->chgArray(c+24057,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),76);
        vcdp->chgBus(c+24081,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+24089,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+24097,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+24105,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->chgBus(c+24113,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__size_r),3);
        vcdp->chgArray(c+24121,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[0]),200);
        vcdp->chgArray(c+24128,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[1]),200);
        vcdp->chgArray(c+24135,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[2]),200);
        vcdp->chgArray(c+24142,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[3]),200);
        vcdp->chgArray(c+24345,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),200);
        vcdp->chgArray(c+24401,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),200);
        vcdp->chgBus(c+24457,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->chgBus(c+24465,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->chgBus(c+24473,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->chgBit(c+24481,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
    }
}

void VVX_cache::traceChgThis__6(VVX_cache__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code) {
    VVX_cache* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    int c = code;
    if (0 && vcdp && c) {}  // Prevent unused
    // Body
    {
        vcdp->chgBit(c+24489,(vlTOPp->clk));
        vcdp->chgBit(c+24497,(vlTOPp->reset));
        vcdp->chgBus(c+24505,(vlTOPp->core_req_valid),4);
        vcdp->chgBus(c+24513,(vlTOPp->core_req_rw),4);
        vcdp->chgBus(c+24521,(vlTOPp->core_req_byteen),16);
        vcdp->chgArray(c+24529,(vlTOPp->core_req_addr),120);
        vcdp->chgArray(c+24561,(vlTOPp->core_req_data),128);
        vcdp->chgQuad(c+24593,(vlTOPp->core_req_tag),42);
        vcdp->chgBit(c+24609,(vlTOPp->core_req_ready));
        vcdp->chgBus(c+24617,(vlTOPp->core_rsp_valid),4);
        vcdp->chgArray(c+24625,(vlTOPp->core_rsp_data),128);
        vcdp->chgQuad(c+24657,(vlTOPp->core_rsp_tag),42);
        vcdp->chgBit(c+24673,(vlTOPp->core_rsp_ready));
        vcdp->chgBit(c+24681,(vlTOPp->dram_req_valid));
        vcdp->chgBit(c+24689,(vlTOPp->dram_req_rw));
        vcdp->chgBus(c+24697,(vlTOPp->dram_req_byteen),16);
        vcdp->chgBus(c+24705,(vlTOPp->dram_req_addr),28);
        vcdp->chgArray(c+24713,(vlTOPp->dram_req_data),128);
        vcdp->chgBus(c+24745,(vlTOPp->dram_req_tag),28);
        vcdp->chgBit(c+24753,(vlTOPp->dram_req_ready));
        vcdp->chgBit(c+24761,(vlTOPp->dram_rsp_valid));
        vcdp->chgArray(c+24769,(vlTOPp->dram_rsp_data),128);
        vcdp->chgBus(c+24801,(vlTOPp->dram_rsp_tag),28);
        vcdp->chgBit(c+24809,(vlTOPp->dram_rsp_ready));
        vcdp->chgBit(c+24817,(vlTOPp->snp_req_valid));
        vcdp->chgBus(c+24825,(vlTOPp->snp_req_addr),28);
        vcdp->chgBit(c+24833,(vlTOPp->snp_req_invalidate));
        vcdp->chgBus(c+24841,(vlTOPp->snp_req_tag),28);
        vcdp->chgBit(c+24849,(vlTOPp->snp_req_ready));
        vcdp->chgBit(c+24857,(vlTOPp->snp_rsp_valid));
        vcdp->chgBus(c+24865,(vlTOPp->snp_rsp_tag),28);
        vcdp->chgBit(c+24873,(vlTOPp->snp_rsp_ready));
        vcdp->chgBus(c+24881,(vlTOPp->snp_fwdout_valid),2);
        vcdp->chgQuad(c+24889,(vlTOPp->snp_fwdout_addr),56);
        vcdp->chgBus(c+24905,(vlTOPp->snp_fwdout_invalidate),2);
        vcdp->chgBus(c+24913,(vlTOPp->snp_fwdout_tag),2);
        vcdp->chgBus(c+24921,(vlTOPp->snp_fwdout_ready),2);
        vcdp->chgBus(c+24929,(vlTOPp->snp_fwdin_valid),2);
        vcdp->chgBus(c+24937,(vlTOPp->snp_fwdin_tag),2);
        vcdp->chgBus(c+24945,(vlTOPp->snp_fwdin_ready),2);
        vcdp->chgBit(c+24953,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_snp_req_ready) 
                                     >> (3U & vlTOPp->snp_req_addr)))));
        vcdp->chgBit(c+24961,(((IData)(vlTOPp->dram_rsp_valid) 
                               & (0U == (3U & vlTOPp->dram_rsp_tag)))));
        vcdp->chgBus(c+24969,((0x3ffffffU & (vlTOPp->dram_rsp_tag 
                                             >> 2U))),26);
        vcdp->chgBit(c+24977,(((IData)(vlTOPp->snp_req_valid) 
                               & (0U == (3U & vlTOPp->snp_req_addr)))));
        vcdp->chgBus(c+24985,((0x3ffffffU & (vlTOPp->snp_req_addr 
                                             >> 2U))),26);
        vcdp->chgBit(c+24993,(((IData)(vlTOPp->dram_rsp_valid) 
                               & (1U == (3U & vlTOPp->dram_rsp_tag)))));
        vcdp->chgBit(c+25001,(((IData)(vlTOPp->snp_req_valid) 
                               & (1U == (3U & vlTOPp->snp_req_addr)))));
        vcdp->chgBit(c+25009,(((IData)(vlTOPp->dram_rsp_valid) 
                               & (2U == (3U & vlTOPp->dram_rsp_tag)))));
        vcdp->chgBit(c+25017,(((IData)(vlTOPp->snp_req_valid) 
                               & (2U == (3U & vlTOPp->snp_req_addr)))));
        vcdp->chgBit(c+25025,(((IData)(vlTOPp->dram_rsp_valid) 
                               & (3U == (3U & vlTOPp->dram_rsp_tag)))));
        vcdp->chgBit(c+25033,(((IData)(vlTOPp->snp_req_valid) 
                               & (3U == (3U & vlTOPp->snp_req_addr)))));
        vcdp->chgBit(c+25041,(((IData)(vlTOPp->dram_req_valid) 
                               & (~ (IData)(vlTOPp->dram_req_rw)))));
        vcdp->chgBit(c+25049,((((IData)(vlTOPp->dram_req_valid) 
                                & (~ (IData)(vlTOPp->dram_req_rw))) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__size_r)))));
    }
}
