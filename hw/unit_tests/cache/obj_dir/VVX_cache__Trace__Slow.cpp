// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Tracing implementation internals
#include "verilated_vcd_c.h"
#include "VVX_cache__Syms.h"


//======================

void VVX_cache::trace(VerilatedVcdC* tfp, int, int) {
    tfp->spTrace()->addCallback(&VVX_cache::traceInit, &VVX_cache::traceFull, &VVX_cache::traceChg, this);
}
void VVX_cache::traceInit(VerilatedVcd* vcdp, void* userthis, uint32_t code) {
    // Callback from vcd->open()
    VVX_cache* t = (VVX_cache*)userthis;
    VVX_cache__Syms* __restrict vlSymsp = t->__VlSymsp;  // Setup global symbol table
    if (!Verilated::calcUnusedSigs()) {
        VL_FATAL_MT(__FILE__, __LINE__, __FILE__,
                        "Turning on wave traces requires Verilated::traceEverOn(true) call before time 0.");
    }
    vcdp->scopeEscape(' ');
    t->traceInitThis(vlSymsp, vcdp, code);
    vcdp->scopeEscape('.');
}
void VVX_cache::traceFull(VerilatedVcd* vcdp, void* userthis, uint32_t code) {
    // Callback from vcd->dump()
    VVX_cache* t = (VVX_cache*)userthis;
    VVX_cache__Syms* __restrict vlSymsp = t->__VlSymsp;  // Setup global symbol table
    t->traceFullThis(vlSymsp, vcdp, code);
}

//======================


void VVX_cache::traceInitThis(VVX_cache__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code) {
    VVX_cache* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    int c = code;
    if (0 && vcdp && c) {}  // Prevent unused
    vcdp->module(vlSymsp->name());  // Setup signal names
    // Body
    {
        vlTOPp->traceInitThis__1(vlSymsp, vcdp, code);
    }
}

void VVX_cache::traceFullThis(VVX_cache__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code) {
    VVX_cache* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    int c = code;
    if (0 && vcdp && c) {}  // Prevent unused
    // Body
    {
        vlTOPp->traceFullThis__1(vlSymsp, vcdp, code);
    }
    // Final
    vlTOPp->__Vm_traceActivity = 0U;
}

void VVX_cache::traceInitThis__1(VVX_cache__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code) {
    VVX_cache* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    int c = code;
    if (0 && vcdp && c) {}  // Prevent unused
    // Body
    {
        vcdp->declBit(c+24489,"clk", false,-1);
        vcdp->declBit(c+24497,"reset", false,-1);
        vcdp->declBus(c+24505,"core_req_valid", false,-1, 3,0);
        vcdp->declBus(c+24513,"core_req_rw", false,-1, 3,0);
        vcdp->declBus(c+24521,"core_req_byteen", false,-1, 15,0);
        vcdp->declArray(c+24529,"core_req_addr", false,-1, 119,0);
        vcdp->declArray(c+24561,"core_req_data", false,-1, 127,0);
        vcdp->declQuad(c+24593,"core_req_tag", false,-1, 41,0);
        vcdp->declBit(c+24609,"core_req_ready", false,-1);
        vcdp->declBus(c+24617,"core_rsp_valid", false,-1, 3,0);
        vcdp->declArray(c+24625,"core_rsp_data", false,-1, 127,0);
        vcdp->declQuad(c+24657,"core_rsp_tag", false,-1, 41,0);
        vcdp->declBit(c+24673,"core_rsp_ready", false,-1);
        vcdp->declBit(c+24681,"dram_req_valid", false,-1);
        vcdp->declBit(c+24689,"dram_req_rw", false,-1);
        vcdp->declBus(c+24697,"dram_req_byteen", false,-1, 15,0);
        vcdp->declBus(c+24705,"dram_req_addr", false,-1, 27,0);
        vcdp->declArray(c+24713,"dram_req_data", false,-1, 127,0);
        vcdp->declBus(c+24745,"dram_req_tag", false,-1, 27,0);
        vcdp->declBit(c+24753,"dram_req_ready", false,-1);
        vcdp->declBit(c+24761,"dram_rsp_valid", false,-1);
        vcdp->declArray(c+24769,"dram_rsp_data", false,-1, 127,0);
        vcdp->declBus(c+24801,"dram_rsp_tag", false,-1, 27,0);
        vcdp->declBit(c+24809,"dram_rsp_ready", false,-1);
        vcdp->declBit(c+24817,"snp_req_valid", false,-1);
        vcdp->declBus(c+24825,"snp_req_addr", false,-1, 27,0);
        vcdp->declBit(c+24833,"snp_req_invalidate", false,-1);
        vcdp->declBus(c+24841,"snp_req_tag", false,-1, 27,0);
        vcdp->declBit(c+24849,"snp_req_ready", false,-1);
        vcdp->declBit(c+24857,"snp_rsp_valid", false,-1);
        vcdp->declBus(c+24865,"snp_rsp_tag", false,-1, 27,0);
        vcdp->declBit(c+24873,"snp_rsp_ready", false,-1);
        vcdp->declBus(c+24881,"snp_fwdout_valid", false,-1, 1,0);
        vcdp->declQuad(c+24889,"snp_fwdout_addr", false,-1, 55,0);
        vcdp->declBus(c+24905,"snp_fwdout_invalidate", false,-1, 1,0);
        vcdp->declBus(c+24913,"snp_fwdout_tag", false,-1, 1,0);
        vcdp->declBus(c+24921,"snp_fwdout_ready", false,-1, 1,0);
        vcdp->declBus(c+24929,"snp_fwdin_valid", false,-1, 1,0);
        vcdp->declBus(c+24937,"snp_fwdin_tag", false,-1, 1,0);
        vcdp->declBus(c+24945,"snp_fwdin_ready", false,-1, 1,0);
        vcdp->declBus(c+25057,"VX_cache CACHE_ID", false,-1, 31,0);
        vcdp->declBus(c+25065,"VX_cache CACHE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache NUM_REQUESTS", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache STAGE_1_CYCLES", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache CREQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache MRVQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache DFPQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache SNRQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache CWBQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache DWBQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache DFQQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache WRITE_ENABLE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache DRAM_ENABLE", false,-1, 31,0);
        vcdp->declBus(c+25057,"VX_cache SNOOP_FORWARDING", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache PRFQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25057,"VX_cache PRFQ_STRIDE", false,-1, 31,0);
        vcdp->declBus(c+25097,"VX_cache CORE_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBus(c+25105,"VX_cache CORE_TAG_ID_BITS", false,-1, 31,0);
        vcdp->declBus(c+25113,"VX_cache DRAM_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBus(c+25121,"VX_cache NUM_SNP_REQUESTS", false,-1, 31,0);
        vcdp->declBus(c+25113,"VX_cache SNP_REQ_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache SNP_FWD_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache reset", false,-1);
        vcdp->declBus(c+24505,"VX_cache core_req_valid", false,-1, 3,0);
        vcdp->declBus(c+24513,"VX_cache core_req_rw", false,-1, 3,0);
        vcdp->declBus(c+24521,"VX_cache core_req_byteen", false,-1, 15,0);
        vcdp->declArray(c+24529,"VX_cache core_req_addr", false,-1, 119,0);
        vcdp->declArray(c+24561,"VX_cache core_req_data", false,-1, 127,0);
        vcdp->declQuad(c+24593,"VX_cache core_req_tag", false,-1, 41,0);
        vcdp->declBit(c+24609,"VX_cache core_req_ready", false,-1);
        vcdp->declBus(c+24617,"VX_cache core_rsp_valid", false,-1, 3,0);
        vcdp->declArray(c+24625,"VX_cache core_rsp_data", false,-1, 127,0);
        vcdp->declQuad(c+24657,"VX_cache core_rsp_tag", false,-1, 41,0);
        vcdp->declBit(c+24673,"VX_cache core_rsp_ready", false,-1);
        vcdp->declBit(c+24681,"VX_cache dram_req_valid", false,-1);
        vcdp->declBit(c+24689,"VX_cache dram_req_rw", false,-1);
        vcdp->declBus(c+24697,"VX_cache dram_req_byteen", false,-1, 15,0);
        vcdp->declBus(c+24705,"VX_cache dram_req_addr", false,-1, 27,0);
        vcdp->declArray(c+24713,"VX_cache dram_req_data", false,-1, 127,0);
        vcdp->declBus(c+24745,"VX_cache dram_req_tag", false,-1, 27,0);
        vcdp->declBit(c+24753,"VX_cache dram_req_ready", false,-1);
        vcdp->declBit(c+24761,"VX_cache dram_rsp_valid", false,-1);
        vcdp->declArray(c+24769,"VX_cache dram_rsp_data", false,-1, 127,0);
        vcdp->declBus(c+24801,"VX_cache dram_rsp_tag", false,-1, 27,0);
        vcdp->declBit(c+24809,"VX_cache dram_rsp_ready", false,-1);
        vcdp->declBit(c+24817,"VX_cache snp_req_valid", false,-1);
        vcdp->declBus(c+24825,"VX_cache snp_req_addr", false,-1, 27,0);
        vcdp->declBit(c+24833,"VX_cache snp_req_invalidate", false,-1);
        vcdp->declBus(c+24841,"VX_cache snp_req_tag", false,-1, 27,0);
        vcdp->declBit(c+24849,"VX_cache snp_req_ready", false,-1);
        vcdp->declBit(c+24857,"VX_cache snp_rsp_valid", false,-1);
        vcdp->declBus(c+24865,"VX_cache snp_rsp_tag", false,-1, 27,0);
        vcdp->declBit(c+24873,"VX_cache snp_rsp_ready", false,-1);
        vcdp->declBus(c+24881,"VX_cache snp_fwdout_valid", false,-1, 1,0);
        vcdp->declQuad(c+24889,"VX_cache snp_fwdout_addr", false,-1, 55,0);
        vcdp->declBus(c+24905,"VX_cache snp_fwdout_invalidate", false,-1, 1,0);
        vcdp->declBus(c+24913,"VX_cache snp_fwdout_tag", false,-1, 1,0);
        vcdp->declBus(c+24921,"VX_cache snp_fwdout_ready", false,-1, 1,0);
        vcdp->declBus(c+24929,"VX_cache snp_fwdin_valid", false,-1, 1,0);
        vcdp->declBus(c+24937,"VX_cache snp_fwdin_tag", false,-1, 1,0);
        vcdp->declBus(c+24945,"VX_cache snp_fwdin_ready", false,-1, 1,0);
        vcdp->declBus(c+1,"VX_cache per_bank_valid", false,-1, 15,0);
        vcdp->declBus(c+1089,"VX_cache per_bank_core_req_ready", false,-1, 3,0);
        vcdp->declBus(c+1097,"VX_cache per_bank_core_rsp_valid", false,-1, 3,0);
        vcdp->declBus(c+1105,"VX_cache per_bank_core_rsp_tid", false,-1, 7,0);
        vcdp->declArray(c+1113,"VX_cache per_bank_core_rsp_data", false,-1, 127,0);
        vcdp->declArray(c+1145,"VX_cache per_bank_core_rsp_tag", false,-1, 167,0);
        vcdp->declBus(c+9,"VX_cache per_bank_core_rsp_ready", false,-1, 3,0);
        vcdp->declBus(c+1193,"VX_cache per_bank_dram_fill_req_valid", false,-1, 3,0);
        vcdp->declArray(c+1201,"VX_cache per_bank_dram_fill_req_addr", false,-1, 111,0);
        vcdp->declBit(c+10065,"VX_cache dram_fill_req_ready", false,-1);
        vcdp->declBus(c+1233,"VX_cache per_bank_dram_fill_rsp_ready", false,-1, 3,0);
        vcdp->declBus(c+17,"VX_cache per_bank_dram_wb_req_ready", false,-1, 3,0);
        vcdp->declBus(c+1241,"VX_cache per_bank_dram_wb_req_valid", false,-1, 3,0);
        vcdp->declQuad(c+1249,"VX_cache per_bank_dram_wb_req_byteen", false,-1, 63,0);
        vcdp->declArray(c+1265,"VX_cache per_bank_dram_wb_req_addr", false,-1, 111,0);
        vcdp->declArray(c+1297,"VX_cache per_bank_dram_wb_req_data", false,-1, 511,0);
        vcdp->declBus(c+1425,"VX_cache per_bank_snp_req_ready", false,-1, 3,0);
        vcdp->declBus(c+1433,"VX_cache per_bank_snp_rsp_valid", false,-1, 3,0);
        vcdp->declArray(c+1441,"VX_cache per_bank_snp_rsp_tag", false,-1, 111,0);
        vcdp->declBus(c+25,"VX_cache per_bank_snp_rsp_ready", false,-1, 3,0);
        vcdp->declBit(c+24817,"VX_cache snp_req_valid_qual", false,-1);
        vcdp->declBus(c+24825,"VX_cache snp_req_addr_qual", false,-1, 27,0);
        vcdp->declBit(c+24833,"VX_cache snp_req_invalidate_qual", false,-1);
        vcdp->declBus(c+24841,"VX_cache snp_req_tag_qual", false,-1, 27,0);
        vcdp->declBit(c+24953,"VX_cache snp_req_ready_qual", false,-1);
        vcdp->declBus(c+33,"VX_cache genblk5[0] curr_bank_core_req_valid", false,-1, 3,0);
        vcdp->declBus(c+24513,"VX_cache genblk5[0] curr_bank_core_req_rw", false,-1, 3,0);
        vcdp->declBus(c+24521,"VX_cache genblk5[0] curr_bank_core_req_byteen", false,-1, 15,0);
        vcdp->declArray(c+24529,"VX_cache genblk5[0] curr_bank_core_req_addr", false,-1, 119,0);
        vcdp->declQuad(c+24593,"VX_cache genblk5[0] curr_bank_core_req_tag", false,-1, 41,0);
        vcdp->declArray(c+24561,"VX_cache genblk5[0] curr_bank_core_req_data", false,-1, 127,0);
        vcdp->declBit(c+10073,"VX_cache genblk5[0] curr_bank_core_rsp_valid", false,-1);
        vcdp->declBus(c+1473,"VX_cache genblk5[0] curr_bank_core_rsp_tid", false,-1, 1,0);
        vcdp->declBus(c+1481,"VX_cache genblk5[0] curr_bank_core_rsp_data", false,-1, 31,0);
        vcdp->declQuad(c+1489,"VX_cache genblk5[0] curr_bank_core_rsp_tag", false,-1, 41,0);
        vcdp->declBit(c+41,"VX_cache genblk5[0] curr_bank_core_rsp_ready", false,-1);
        vcdp->declBit(c+24961,"VX_cache genblk5[0] curr_bank_dram_fill_rsp_valid", false,-1);
        vcdp->declArray(c+24769,"VX_cache genblk5[0] curr_bank_dram_fill_rsp_data", false,-1, 127,0);
        vcdp->declBus(c+24969,"VX_cache genblk5[0] curr_bank_dram_fill_rsp_addr", false,-1, 25,0);
        vcdp->declBit(c+10081,"VX_cache genblk5[0] curr_bank_dram_fill_rsp_ready", false,-1);
        vcdp->declBit(c+1505,"VX_cache genblk5[0] curr_bank_dram_fill_req_valid", false,-1);
        vcdp->declBus(c+10089,"VX_cache genblk5[0] curr_bank_dram_fill_req_addr", false,-1, 25,0);
        vcdp->declBit(c+10065,"VX_cache genblk5[0] curr_bank_dram_fill_req_ready", false,-1);
        vcdp->declBit(c+1513,"VX_cache genblk5[0] curr_bank_dram_wb_req_valid", false,-1);
        vcdp->declBus(c+1521,"VX_cache genblk5[0] curr_bank_dram_wb_req_byteen", false,-1, 15,0);
        vcdp->declBus(c+1529,"VX_cache genblk5[0] curr_bank_dram_wb_req_addr", false,-1, 25,0);
        vcdp->declArray(c+1537,"VX_cache genblk5[0] curr_bank_dram_wb_req_data", false,-1, 127,0);
        vcdp->declBit(c+49,"VX_cache genblk5[0] curr_bank_dram_wb_req_ready", false,-1);
        vcdp->declBit(c+24977,"VX_cache genblk5[0] curr_bank_snp_req_valid", false,-1);
        vcdp->declBus(c+24985,"VX_cache genblk5[0] curr_bank_snp_req_addr", false,-1, 25,0);
        vcdp->declBit(c+24833,"VX_cache genblk5[0] curr_bank_snp_req_invalidate", false,-1);
        vcdp->declBus(c+24841,"VX_cache genblk5[0] curr_bank_snp_req_tag", false,-1, 27,0);
        vcdp->declBit(c+10097,"VX_cache genblk5[0] curr_bank_snp_req_ready", false,-1);
        vcdp->declBit(c+1569,"VX_cache genblk5[0] curr_bank_snp_rsp_valid", false,-1);
        vcdp->declBus(c+1577,"VX_cache genblk5[0] curr_bank_snp_rsp_tag", false,-1, 27,0);
        vcdp->declBit(c+57,"VX_cache genblk5[0] curr_bank_snp_rsp_ready", false,-1);
        vcdp->declBit(c+10105,"VX_cache genblk5[0] curr_bank_core_req_ready", false,-1);
        vcdp->declBus(c+65,"VX_cache genblk5[1] curr_bank_core_req_valid", false,-1, 3,0);
        vcdp->declBus(c+24513,"VX_cache genblk5[1] curr_bank_core_req_rw", false,-1, 3,0);
        vcdp->declBus(c+24521,"VX_cache genblk5[1] curr_bank_core_req_byteen", false,-1, 15,0);
        vcdp->declArray(c+24529,"VX_cache genblk5[1] curr_bank_core_req_addr", false,-1, 119,0);
        vcdp->declQuad(c+24593,"VX_cache genblk5[1] curr_bank_core_req_tag", false,-1, 41,0);
        vcdp->declArray(c+24561,"VX_cache genblk5[1] curr_bank_core_req_data", false,-1, 127,0);
        vcdp->declBit(c+10113,"VX_cache genblk5[1] curr_bank_core_rsp_valid", false,-1);
        vcdp->declBus(c+1585,"VX_cache genblk5[1] curr_bank_core_rsp_tid", false,-1, 1,0);
        vcdp->declBus(c+1593,"VX_cache genblk5[1] curr_bank_core_rsp_data", false,-1, 31,0);
        vcdp->declQuad(c+1601,"VX_cache genblk5[1] curr_bank_core_rsp_tag", false,-1, 41,0);
        vcdp->declBit(c+73,"VX_cache genblk5[1] curr_bank_core_rsp_ready", false,-1);
        vcdp->declBit(c+24993,"VX_cache genblk5[1] curr_bank_dram_fill_rsp_valid", false,-1);
        vcdp->declArray(c+24769,"VX_cache genblk5[1] curr_bank_dram_fill_rsp_data", false,-1, 127,0);
        vcdp->declBus(c+24969,"VX_cache genblk5[1] curr_bank_dram_fill_rsp_addr", false,-1, 25,0);
        vcdp->declBit(c+10121,"VX_cache genblk5[1] curr_bank_dram_fill_rsp_ready", false,-1);
        vcdp->declBit(c+1617,"VX_cache genblk5[1] curr_bank_dram_fill_req_valid", false,-1);
        vcdp->declBus(c+10129,"VX_cache genblk5[1] curr_bank_dram_fill_req_addr", false,-1, 25,0);
        vcdp->declBit(c+10065,"VX_cache genblk5[1] curr_bank_dram_fill_req_ready", false,-1);
        vcdp->declBit(c+1625,"VX_cache genblk5[1] curr_bank_dram_wb_req_valid", false,-1);
        vcdp->declBus(c+1633,"VX_cache genblk5[1] curr_bank_dram_wb_req_byteen", false,-1, 15,0);
        vcdp->declBus(c+1641,"VX_cache genblk5[1] curr_bank_dram_wb_req_addr", false,-1, 25,0);
        vcdp->declArray(c+1649,"VX_cache genblk5[1] curr_bank_dram_wb_req_data", false,-1, 127,0);
        vcdp->declBit(c+81,"VX_cache genblk5[1] curr_bank_dram_wb_req_ready", false,-1);
        vcdp->declBit(c+25001,"VX_cache genblk5[1] curr_bank_snp_req_valid", false,-1);
        vcdp->declBus(c+24985,"VX_cache genblk5[1] curr_bank_snp_req_addr", false,-1, 25,0);
        vcdp->declBit(c+24833,"VX_cache genblk5[1] curr_bank_snp_req_invalidate", false,-1);
        vcdp->declBus(c+24841,"VX_cache genblk5[1] curr_bank_snp_req_tag", false,-1, 27,0);
        vcdp->declBit(c+10137,"VX_cache genblk5[1] curr_bank_snp_req_ready", false,-1);
        vcdp->declBit(c+1681,"VX_cache genblk5[1] curr_bank_snp_rsp_valid", false,-1);
        vcdp->declBus(c+1689,"VX_cache genblk5[1] curr_bank_snp_rsp_tag", false,-1, 27,0);
        vcdp->declBit(c+89,"VX_cache genblk5[1] curr_bank_snp_rsp_ready", false,-1);
        vcdp->declBit(c+10145,"VX_cache genblk5[1] curr_bank_core_req_ready", false,-1);
        vcdp->declBus(c+97,"VX_cache genblk5[2] curr_bank_core_req_valid", false,-1, 3,0);
        vcdp->declBus(c+24513,"VX_cache genblk5[2] curr_bank_core_req_rw", false,-1, 3,0);
        vcdp->declBus(c+24521,"VX_cache genblk5[2] curr_bank_core_req_byteen", false,-1, 15,0);
        vcdp->declArray(c+24529,"VX_cache genblk5[2] curr_bank_core_req_addr", false,-1, 119,0);
        vcdp->declQuad(c+24593,"VX_cache genblk5[2] curr_bank_core_req_tag", false,-1, 41,0);
        vcdp->declArray(c+24561,"VX_cache genblk5[2] curr_bank_core_req_data", false,-1, 127,0);
        vcdp->declBit(c+10153,"VX_cache genblk5[2] curr_bank_core_rsp_valid", false,-1);
        vcdp->declBus(c+1697,"VX_cache genblk5[2] curr_bank_core_rsp_tid", false,-1, 1,0);
        vcdp->declBus(c+1705,"VX_cache genblk5[2] curr_bank_core_rsp_data", false,-1, 31,0);
        vcdp->declQuad(c+1713,"VX_cache genblk5[2] curr_bank_core_rsp_tag", false,-1, 41,0);
        vcdp->declBit(c+105,"VX_cache genblk5[2] curr_bank_core_rsp_ready", false,-1);
        vcdp->declBit(c+25009,"VX_cache genblk5[2] curr_bank_dram_fill_rsp_valid", false,-1);
        vcdp->declArray(c+24769,"VX_cache genblk5[2] curr_bank_dram_fill_rsp_data", false,-1, 127,0);
        vcdp->declBus(c+24969,"VX_cache genblk5[2] curr_bank_dram_fill_rsp_addr", false,-1, 25,0);
        vcdp->declBit(c+10161,"VX_cache genblk5[2] curr_bank_dram_fill_rsp_ready", false,-1);
        vcdp->declBit(c+1729,"VX_cache genblk5[2] curr_bank_dram_fill_req_valid", false,-1);
        vcdp->declBus(c+10169,"VX_cache genblk5[2] curr_bank_dram_fill_req_addr", false,-1, 25,0);
        vcdp->declBit(c+10065,"VX_cache genblk5[2] curr_bank_dram_fill_req_ready", false,-1);
        vcdp->declBit(c+1737,"VX_cache genblk5[2] curr_bank_dram_wb_req_valid", false,-1);
        vcdp->declBus(c+1745,"VX_cache genblk5[2] curr_bank_dram_wb_req_byteen", false,-1, 15,0);
        vcdp->declBus(c+1753,"VX_cache genblk5[2] curr_bank_dram_wb_req_addr", false,-1, 25,0);
        vcdp->declArray(c+1761,"VX_cache genblk5[2] curr_bank_dram_wb_req_data", false,-1, 127,0);
        vcdp->declBit(c+113,"VX_cache genblk5[2] curr_bank_dram_wb_req_ready", false,-1);
        vcdp->declBit(c+25017,"VX_cache genblk5[2] curr_bank_snp_req_valid", false,-1);
        vcdp->declBus(c+24985,"VX_cache genblk5[2] curr_bank_snp_req_addr", false,-1, 25,0);
        vcdp->declBit(c+24833,"VX_cache genblk5[2] curr_bank_snp_req_invalidate", false,-1);
        vcdp->declBus(c+24841,"VX_cache genblk5[2] curr_bank_snp_req_tag", false,-1, 27,0);
        vcdp->declBit(c+10177,"VX_cache genblk5[2] curr_bank_snp_req_ready", false,-1);
        vcdp->declBit(c+1793,"VX_cache genblk5[2] curr_bank_snp_rsp_valid", false,-1);
        vcdp->declBus(c+1801,"VX_cache genblk5[2] curr_bank_snp_rsp_tag", false,-1, 27,0);
        vcdp->declBit(c+121,"VX_cache genblk5[2] curr_bank_snp_rsp_ready", false,-1);
        vcdp->declBit(c+10185,"VX_cache genblk5[2] curr_bank_core_req_ready", false,-1);
        vcdp->declBus(c+129,"VX_cache genblk5[3] curr_bank_core_req_valid", false,-1, 3,0);
        vcdp->declBus(c+24513,"VX_cache genblk5[3] curr_bank_core_req_rw", false,-1, 3,0);
        vcdp->declBus(c+24521,"VX_cache genblk5[3] curr_bank_core_req_byteen", false,-1, 15,0);
        vcdp->declArray(c+24529,"VX_cache genblk5[3] curr_bank_core_req_addr", false,-1, 119,0);
        vcdp->declQuad(c+24593,"VX_cache genblk5[3] curr_bank_core_req_tag", false,-1, 41,0);
        vcdp->declArray(c+24561,"VX_cache genblk5[3] curr_bank_core_req_data", false,-1, 127,0);
        vcdp->declBit(c+10193,"VX_cache genblk5[3] curr_bank_core_rsp_valid", false,-1);
        vcdp->declBus(c+1809,"VX_cache genblk5[3] curr_bank_core_rsp_tid", false,-1, 1,0);
        vcdp->declBus(c+1817,"VX_cache genblk5[3] curr_bank_core_rsp_data", false,-1, 31,0);
        vcdp->declQuad(c+1825,"VX_cache genblk5[3] curr_bank_core_rsp_tag", false,-1, 41,0);
        vcdp->declBit(c+137,"VX_cache genblk5[3] curr_bank_core_rsp_ready", false,-1);
        vcdp->declBit(c+25025,"VX_cache genblk5[3] curr_bank_dram_fill_rsp_valid", false,-1);
        vcdp->declArray(c+24769,"VX_cache genblk5[3] curr_bank_dram_fill_rsp_data", false,-1, 127,0);
        vcdp->declBus(c+24969,"VX_cache genblk5[3] curr_bank_dram_fill_rsp_addr", false,-1, 25,0);
        vcdp->declBit(c+10201,"VX_cache genblk5[3] curr_bank_dram_fill_rsp_ready", false,-1);
        vcdp->declBit(c+1841,"VX_cache genblk5[3] curr_bank_dram_fill_req_valid", false,-1);
        vcdp->declBus(c+10209,"VX_cache genblk5[3] curr_bank_dram_fill_req_addr", false,-1, 25,0);
        vcdp->declBit(c+10065,"VX_cache genblk5[3] curr_bank_dram_fill_req_ready", false,-1);
        vcdp->declBit(c+1849,"VX_cache genblk5[3] curr_bank_dram_wb_req_valid", false,-1);
        vcdp->declBus(c+1857,"VX_cache genblk5[3] curr_bank_dram_wb_req_byteen", false,-1, 15,0);
        vcdp->declBus(c+1865,"VX_cache genblk5[3] curr_bank_dram_wb_req_addr", false,-1, 25,0);
        vcdp->declArray(c+1873,"VX_cache genblk5[3] curr_bank_dram_wb_req_data", false,-1, 127,0);
        vcdp->declBit(c+145,"VX_cache genblk5[3] curr_bank_dram_wb_req_ready", false,-1);
        vcdp->declBit(c+25033,"VX_cache genblk5[3] curr_bank_snp_req_valid", false,-1);
        vcdp->declBus(c+24985,"VX_cache genblk5[3] curr_bank_snp_req_addr", false,-1, 25,0);
        vcdp->declBit(c+24833,"VX_cache genblk5[3] curr_bank_snp_req_invalidate", false,-1);
        vcdp->declBus(c+24841,"VX_cache genblk5[3] curr_bank_snp_req_tag", false,-1, 27,0);
        vcdp->declBit(c+10217,"VX_cache genblk5[3] curr_bank_snp_req_ready", false,-1);
        vcdp->declBit(c+1905,"VX_cache genblk5[3] curr_bank_snp_rsp_valid", false,-1);
        vcdp->declBus(c+1913,"VX_cache genblk5[3] curr_bank_snp_rsp_tag", false,-1, 27,0);
        vcdp->declBit(c+153,"VX_cache genblk5[3] curr_bank_snp_rsp_ready", false,-1);
        vcdp->declBit(c+10225,"VX_cache genblk5[3] curr_bank_core_req_ready", false,-1);
        vcdp->declBus(c+25073,"VX_cache cache_core_req_bank_sel BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache cache_core_req_bank_sel WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache cache_core_req_bank_sel NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache cache_core_req_bank_sel NUM_REQUESTS", false,-1, 31,0);
        vcdp->declBus(c+24505,"VX_cache cache_core_req_bank_sel core_req_valid", false,-1, 3,0);
        vcdp->declArray(c+24529,"VX_cache cache_core_req_bank_sel core_req_addr", false,-1, 119,0);
        vcdp->declBus(c+1089,"VX_cache cache_core_req_bank_sel per_bank_ready", false,-1, 3,0);
        vcdp->declBus(c+1,"VX_cache cache_core_req_bank_sel per_bank_valid", false,-1, 15,0);
        vcdp->declBit(c+24609,"VX_cache cache_core_req_bank_sel core_req_ready", false,-1);
        vcdp->declBus(c+25129,"VX_cache cache_core_req_bank_sel i", false,-1, 31,0);
        vcdp->declBus(c+161,"VX_cache cache_core_req_bank_sel genblk2 per_bank_ready_sel", false,-1, 3,0);
        vcdp->declBus(c+25073,"VX_cache cache_dram_req_arb BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache cache_dram_req_arb NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache cache_dram_req_arb WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache cache_dram_req_arb DFQQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache cache_dram_req_arb PRFQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25057,"VX_cache cache_dram_req_arb PRFQ_STRIDE", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache cache_dram_req_arb clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache cache_dram_req_arb reset", false,-1);
        vcdp->declBus(c+1193,"VX_cache cache_dram_req_arb per_bank_dram_fill_req_valid", false,-1, 3,0);
        vcdp->declArray(c+1201,"VX_cache cache_dram_req_arb per_bank_dram_fill_req_addr", false,-1, 111,0);
        vcdp->declBit(c+10065,"VX_cache cache_dram_req_arb dram_fill_req_ready", false,-1);
        vcdp->declBus(c+1241,"VX_cache cache_dram_req_arb per_bank_dram_wb_req_valid", false,-1, 3,0);
        vcdp->declQuad(c+1249,"VX_cache cache_dram_req_arb per_bank_dram_wb_req_byteen", false,-1, 63,0);
        vcdp->declArray(c+1265,"VX_cache cache_dram_req_arb per_bank_dram_wb_req_addr", false,-1, 111,0);
        vcdp->declArray(c+1297,"VX_cache cache_dram_req_arb per_bank_dram_wb_req_data", false,-1, 511,0);
        vcdp->declBus(c+17,"VX_cache cache_dram_req_arb per_bank_dram_wb_req_ready", false,-1, 3,0);
        vcdp->declBit(c+24681,"VX_cache cache_dram_req_arb dram_req_valid", false,-1);
        vcdp->declBit(c+24689,"VX_cache cache_dram_req_arb dram_req_rw", false,-1);
        vcdp->declBus(c+24697,"VX_cache cache_dram_req_arb dram_req_byteen", false,-1, 15,0);
        vcdp->declBus(c+24705,"VX_cache cache_dram_req_arb dram_req_addr", false,-1, 27,0);
        vcdp->declArray(c+24713,"VX_cache cache_dram_req_arb dram_req_data", false,-1, 127,0);
        vcdp->declBit(c+24753,"VX_cache cache_dram_req_arb dram_req_ready", false,-1);
        vcdp->declBit(c+25137,"VX_cache cache_dram_req_arb pref_pop", false,-1);
        vcdp->declBit(c+25137,"VX_cache cache_dram_req_arb pref_valid", false,-1);
        vcdp->declBus(c+10233,"VX_cache cache_dram_req_arb pref_addr", false,-1, 27,0);
        vcdp->declBit(c+1921,"VX_cache cache_dram_req_arb dwb_valid", false,-1);
        vcdp->declBit(c+1929,"VX_cache cache_dram_req_arb dfqq_req", false,-1);
        vcdp->declBus(c+1937,"VX_cache cache_dram_req_arb dfqq_req_addr", false,-1, 27,0);
        vcdp->declBit(c+1945,"VX_cache cache_dram_req_arb dfqq_empty", false,-1);
        vcdp->declBit(c+169,"VX_cache cache_dram_req_arb dfqq_pop", false,-1);
        vcdp->declBit(c+1953,"VX_cache cache_dram_req_arb dfqq_push", false,-1);
        vcdp->declBit(c+10241,"VX_cache cache_dram_req_arb dfqq_full", false,-1);
        vcdp->declBus(c+1961,"VX_cache cache_dram_req_arb dwb_bank", false,-1, 1,0);
        vcdp->declBus(c+25073,"VX_cache cache_dram_req_arb prfqq BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache cache_dram_req_arb prfqq WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache cache_dram_req_arb prfqq PRFQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25057,"VX_cache cache_dram_req_arb prfqq PRFQ_STRIDE", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache cache_dram_req_arb prfqq clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache cache_dram_req_arb prfqq reset", false,-1);
        vcdp->declBit(c+25041,"VX_cache cache_dram_req_arb prfqq dram_req", false,-1);
        vcdp->declBus(c+24705,"VX_cache cache_dram_req_arb prfqq dram_req_addr", false,-1, 27,0);
        vcdp->declBit(c+25137,"VX_cache cache_dram_req_arb prfqq pref_pop", false,-1);
        vcdp->declBit(c+25137,"VX_cache cache_dram_req_arb prfqq pref_valid", false,-1);
        vcdp->declBus(c+10233,"VX_cache cache_dram_req_arb prfqq pref_addr", false,-1, 27,0);
        vcdp->declBus(c+10249,"VX_cache cache_dram_req_arb prfqq use_valid", false,-1, 1,0);
        vcdp->declBus(c+10233,"VX_cache cache_dram_req_arb prfqq use_addr", false,-1, 27,0);
        vcdp->declBit(c+10257,"VX_cache cache_dram_req_arb prfqq current_valid", false,-1);
        vcdp->declBus(c+10265,"VX_cache cache_dram_req_arb prfqq current_addr", false,-1, 27,0);
        vcdp->declBit(c+10257,"VX_cache cache_dram_req_arb prfqq current_full", false,-1);
        vcdp->declBit(c+10273,"VX_cache cache_dram_req_arb prfqq current_empty", false,-1);
        vcdp->declBit(c+1969,"VX_cache cache_dram_req_arb prfqq update_use", false,-1);
        vcdp->declBus(c+25113,"VX_cache cache_dram_req_arb prfqq pfq_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache cache_dram_req_arb prfqq pfq_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache cache_dram_req_arb prfqq pfq_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache cache_dram_req_arb prfqq pfq_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache cache_dram_req_arb prfqq pfq_queue reset", false,-1);
        vcdp->declBit(c+25049,"VX_cache cache_dram_req_arb prfqq pfq_queue push", false,-1);
        vcdp->declBit(c+1969,"VX_cache cache_dram_req_arb prfqq pfq_queue pop", false,-1);
        vcdp->declBus(c+24705,"VX_cache cache_dram_req_arb prfqq pfq_queue data_in", false,-1, 27,0);
        vcdp->declBus(c+10265,"VX_cache cache_dram_req_arb prfqq pfq_queue data_out", false,-1, 27,0);
        vcdp->declBit(c+10273,"VX_cache cache_dram_req_arb prfqq pfq_queue empty", false,-1);
        vcdp->declBit(c+10257,"VX_cache cache_dram_req_arb prfqq pfq_queue full", false,-1);
        vcdp->declBus(c+10281,"VX_cache cache_dram_req_arb prfqq pfq_queue size", false,-1, 0,0);
        vcdp->declBus(c+10281,"VX_cache cache_dram_req_arb prfqq pfq_queue size_r", false,-1, 0,0);
        vcdp->declBit(c+1977,"VX_cache cache_dram_req_arb prfqq pfq_queue reading", false,-1);
        vcdp->declBit(c+1985,"VX_cache cache_dram_req_arb prfqq pfq_queue writing", false,-1);
        vcdp->declBus(c+10265,"VX_cache cache_dram_req_arb prfqq pfq_queue genblk2 head_r", false,-1, 27,0);
        vcdp->declBus(c+25073,"VX_cache cache_dram_req_arb dram_fill_arb BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache cache_dram_req_arb dram_fill_arb NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache cache_dram_req_arb dram_fill_arb DFQQ_SIZE", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache cache_dram_req_arb dram_fill_arb clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache cache_dram_req_arb dram_fill_arb reset", false,-1);
        vcdp->declBit(c+1953,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_push", false,-1);
        vcdp->declBus(c+1193,"VX_cache cache_dram_req_arb dram_fill_arb per_bank_dram_fill_req_valid", false,-1, 3,0);
        vcdp->declArray(c+1201,"VX_cache cache_dram_req_arb dram_fill_arb per_bank_dram_fill_req_addr", false,-1, 111,0);
        vcdp->declBit(c+169,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_pop", false,-1);
        vcdp->declBit(c+1929,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_req", false,-1);
        vcdp->declBus(c+1937,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_req_addr", false,-1, 27,0);
        vcdp->declBit(c+1945,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_empty", false,-1);
        vcdp->declBit(c+10241,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_full", false,-1);
        vcdp->declBus(c+10289,"VX_cache cache_dram_req_arb dram_fill_arb use_per_bank_dram_fill_req_valid", false,-1, 3,0);
        vcdp->declArray(c+10297,"VX_cache cache_dram_req_arb dram_fill_arb use_per_bank_dram_fill_req_addr", false,-1, 111,0);
        vcdp->declBus(c+1993,"VX_cache cache_dram_req_arb dram_fill_arb out_per_bank_dram_fill_req_valid", false,-1, 3,0);
        vcdp->declArray(c+2001,"VX_cache cache_dram_req_arb dram_fill_arb out_per_bank_dram_fill_req_addr", false,-1, 111,0);
        vcdp->declBus(c+2033,"VX_cache cache_dram_req_arb dram_fill_arb use_per_bqual_bank_dram_fill_req_valid", false,-1, 3,0);
        vcdp->declArray(c+2041,"VX_cache cache_dram_req_arb dram_fill_arb qual_bank_dram_fill_req_addr", false,-1, 111,0);
        vcdp->declBus(c+2073,"VX_cache cache_dram_req_arb dram_fill_arb updated_bank_dram_fill_req_valid", false,-1, 3,0);
        vcdp->declBit(c+10329,"VX_cache cache_dram_req_arb dram_fill_arb o_empty", false,-1);
        vcdp->declBit(c+10337,"VX_cache cache_dram_req_arb dram_fill_arb use_empty", false,-1);
        vcdp->declBit(c+2081,"VX_cache cache_dram_req_arb dram_fill_arb out_empty", false,-1);
        vcdp->declBit(c+2089,"VX_cache cache_dram_req_arb dram_fill_arb push_qual", false,-1);
        vcdp->declBit(c+953,"VX_cache cache_dram_req_arb dram_fill_arb pop_qual", false,-1);
        vcdp->declBus(c+2097,"VX_cache cache_dram_req_arb dram_fill_arb qual_request_index", false,-1, 1,0);
        vcdp->declBit(c+2105,"VX_cache cache_dram_req_arb dram_fill_arb qual_has_request", false,-1);
        vcdp->declBus(c+25145,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue reset", false,-1);
        vcdp->declBit(c+2089,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue push", false,-1);
        vcdp->declBit(c+953,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue pop", false,-1);
        vcdp->declArray(c+2113,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue data_in", false,-1, 115,0);
        vcdp->declArray(c+2145,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue data_out", false,-1, 115,0);
        vcdp->declBit(c+10329,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue empty", false,-1);
        vcdp->declBit(c+10241,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue full", false,-1);
        vcdp->declBus(c+10345,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue size", false,-1, 2,0);
        vcdp->declBus(c+10345,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue size_r", false,-1, 2,0);
        vcdp->declBit(c+177,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue reading", false,-1);
        vcdp->declBit(c+2177,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue writing", false,-1);
        {int i; for (i=0; i<4; i++) {
                vcdp->declArray(c+10353+i*4,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue genblk3 data", true,(i+0), 115,0);}}
        vcdp->declArray(c+10481,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue genblk3 genblk2 head_r", false,-1, 115,0);
        vcdp->declArray(c+10513,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue genblk3 genblk2 curr_r", false,-1, 115,0);
        vcdp->declBus(c+10545,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue genblk3 genblk2 wr_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+10553,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue genblk3 genblk2 rd_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+10561,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 1,0);
        vcdp->declBit(c+10329,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+10241,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+10569,"VX_cache cache_dram_req_arb dram_fill_arb dfqq_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25081,"VX_cache cache_dram_req_arb dram_fill_arb sel_bank N", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache cache_dram_req_arb dram_fill_arb sel_bank clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache cache_dram_req_arb dram_fill_arb sel_bank reset", false,-1);
        vcdp->declBus(c+2033,"VX_cache cache_dram_req_arb dram_fill_arb sel_bank requests", false,-1, 3,0);
        vcdp->declBus(c+2097,"VX_cache cache_dram_req_arb dram_fill_arb sel_bank grant_index", false,-1, 1,0);
        vcdp->declBus(c+2185,"VX_cache cache_dram_req_arb dram_fill_arb sel_bank grant_onehot", false,-1, 3,0);
        vcdp->declBit(c+2105,"VX_cache cache_dram_req_arb dram_fill_arb sel_bank grant_valid", false,-1);
        vcdp->declBus(c+2185,"VX_cache cache_dram_req_arb dram_fill_arb sel_bank genblk2 grant_onehot_r", false,-1, 3,0);
        vcdp->declBus(c+25081,"VX_cache cache_dram_req_arb dram_fill_arb sel_bank genblk2 priority_encoder N", false,-1, 31,0);
        vcdp->declBus(c+2033,"VX_cache cache_dram_req_arb dram_fill_arb sel_bank genblk2 priority_encoder data_in", false,-1, 3,0);
        vcdp->declBus(c+2097,"VX_cache cache_dram_req_arb dram_fill_arb sel_bank genblk2 priority_encoder data_out", false,-1, 1,0);
        vcdp->declBit(c+2105,"VX_cache cache_dram_req_arb dram_fill_arb sel_bank genblk2 priority_encoder valid_out", false,-1);
        vcdp->declBus(c+2193,"VX_cache cache_dram_req_arb dram_fill_arb sel_bank genblk2 priority_encoder i", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache cache_dram_req_arb sel_dwb N", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache cache_dram_req_arb sel_dwb clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache cache_dram_req_arb sel_dwb reset", false,-1);
        vcdp->declBus(c+1241,"VX_cache cache_dram_req_arb sel_dwb requests", false,-1, 3,0);
        vcdp->declBus(c+1961,"VX_cache cache_dram_req_arb sel_dwb grant_index", false,-1, 1,0);
        vcdp->declBus(c+2201,"VX_cache cache_dram_req_arb sel_dwb grant_onehot", false,-1, 3,0);
        vcdp->declBit(c+1921,"VX_cache cache_dram_req_arb sel_dwb grant_valid", false,-1);
        vcdp->declBus(c+2201,"VX_cache cache_dram_req_arb sel_dwb genblk2 grant_onehot_r", false,-1, 3,0);
        vcdp->declBus(c+25081,"VX_cache cache_dram_req_arb sel_dwb genblk2 priority_encoder N", false,-1, 31,0);
        vcdp->declBus(c+1241,"VX_cache cache_dram_req_arb sel_dwb genblk2 priority_encoder data_in", false,-1, 3,0);
        vcdp->declBus(c+1961,"VX_cache cache_dram_req_arb sel_dwb genblk2 priority_encoder data_out", false,-1, 1,0);
        vcdp->declBit(c+1921,"VX_cache cache_dram_req_arb sel_dwb genblk2 priority_encoder valid_out", false,-1);
        vcdp->declBus(c+2209,"VX_cache cache_dram_req_arb sel_dwb genblk2 priority_encoder i", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache cache_core_rsp_merge NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache cache_core_rsp_merge WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache cache_core_rsp_merge NUM_REQUESTS", false,-1, 31,0);
        vcdp->declBus(c+25097,"VX_cache cache_core_rsp_merge CORE_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBus(c+25105,"VX_cache cache_core_rsp_merge CORE_TAG_ID_BITS", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache cache_core_rsp_merge clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache cache_core_rsp_merge reset", false,-1);
        vcdp->declBus(c+1105,"VX_cache cache_core_rsp_merge per_bank_core_rsp_tid", false,-1, 7,0);
        vcdp->declBus(c+1097,"VX_cache cache_core_rsp_merge per_bank_core_rsp_valid", false,-1, 3,0);
        vcdp->declArray(c+1113,"VX_cache cache_core_rsp_merge per_bank_core_rsp_data", false,-1, 127,0);
        vcdp->declArray(c+1145,"VX_cache cache_core_rsp_merge per_bank_core_rsp_tag", false,-1, 167,0);
        vcdp->declBus(c+9,"VX_cache cache_core_rsp_merge per_bank_core_rsp_ready", false,-1, 3,0);
        vcdp->declBus(c+24617,"VX_cache cache_core_rsp_merge core_rsp_valid", false,-1, 3,0);
        vcdp->declArray(c+2217,"VX_cache cache_core_rsp_merge core_rsp_data", false,-1, 127,0);
        vcdp->declQuad(c+2249,"VX_cache cache_core_rsp_merge core_rsp_tag", false,-1, 41,0);
        vcdp->declBit(c+24673,"VX_cache cache_core_rsp_merge core_rsp_ready", false,-1);
        vcdp->declBus(c+2265,"VX_cache cache_core_rsp_merge main_bank_index", false,-1, 1,0);
        vcdp->declBus(c+2273,"VX_cache cache_core_rsp_merge per_bank_core_rsp_pop_unqual", false,-1, 3,0);
        vcdp->declBus(c+25129,"VX_cache cache_core_rsp_merge i", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache cache_core_rsp_merge sel_bank N", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache cache_core_rsp_merge sel_bank clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache cache_core_rsp_merge sel_bank reset", false,-1);
        vcdp->declBus(c+1097,"VX_cache cache_core_rsp_merge sel_bank requests", false,-1, 3,0);
        vcdp->declBus(c+2265,"VX_cache cache_core_rsp_merge sel_bank grant_index", false,-1, 1,0);
        vcdp->declBus(c+2281,"VX_cache cache_core_rsp_merge sel_bank grant_onehot", false,-1, 3,0);
        vcdp->declBit(c+2289,"VX_cache cache_core_rsp_merge sel_bank grant_valid", false,-1);
        vcdp->declBus(c+10577,"VX_cache cache_core_rsp_merge sel_bank genblk2 requests_use", false,-1, 3,0);
        vcdp->declBus(c+2297,"VX_cache cache_core_rsp_merge sel_bank genblk2 update_value", false,-1, 3,0);
        vcdp->declBus(c+2305,"VX_cache cache_core_rsp_merge sel_bank genblk2 late_value", false,-1, 3,0);
        vcdp->declBit(c+10585,"VX_cache cache_core_rsp_merge sel_bank genblk2 refill", false,-1);
        vcdp->declBus(c+1097,"VX_cache cache_core_rsp_merge sel_bank genblk2 refill_value", false,-1, 3,0);
        vcdp->declBus(c+10593,"VX_cache cache_core_rsp_merge sel_bank genblk2 refill_original", false,-1, 3,0);
        vcdp->declBus(c+2281,"VX_cache cache_core_rsp_merge sel_bank genblk2 grant_onehot_r", false,-1, 3,0);
        vcdp->declBus(c+25081,"VX_cache cache_core_rsp_merge sel_bank genblk2 priority_encoder N", false,-1, 31,0);
        vcdp->declBus(c+10577,"VX_cache cache_core_rsp_merge sel_bank genblk2 priority_encoder data_in", false,-1, 3,0);
        vcdp->declBus(c+2265,"VX_cache cache_core_rsp_merge sel_bank genblk2 priority_encoder data_out", false,-1, 1,0);
        vcdp->declBit(c+2289,"VX_cache cache_core_rsp_merge sel_bank genblk2 priority_encoder valid_out", false,-1);
        vcdp->declBus(c+2313,"VX_cache cache_core_rsp_merge sel_bank genblk2 priority_encoder i", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache snp_rsp_arb NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache snp_rsp_arb BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25113,"VX_cache snp_rsp_arb SNP_REQ_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache snp_rsp_arb clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache snp_rsp_arb reset", false,-1);
        vcdp->declBus(c+1433,"VX_cache snp_rsp_arb per_bank_snp_rsp_valid", false,-1, 3,0);
        vcdp->declArray(c+1441,"VX_cache snp_rsp_arb per_bank_snp_rsp_tag", false,-1, 111,0);
        vcdp->declBus(c+25,"VX_cache snp_rsp_arb per_bank_snp_rsp_ready", false,-1, 3,0);
        vcdp->declBit(c+24857,"VX_cache snp_rsp_arb snp_rsp_valid", false,-1);
        vcdp->declBus(c+24865,"VX_cache snp_rsp_arb snp_rsp_tag", false,-1, 27,0);
        vcdp->declBit(c+24873,"VX_cache snp_rsp_arb snp_rsp_ready", false,-1);
        vcdp->declBus(c+2321,"VX_cache snp_rsp_arb fsq_bank", false,-1, 1,0);
        vcdp->declBit(c+2329,"VX_cache snp_rsp_arb fsq_valid", false,-1);
        vcdp->declBus(c+25081,"VX_cache snp_rsp_arb sel_ffsq N", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache snp_rsp_arb sel_ffsq clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache snp_rsp_arb sel_ffsq reset", false,-1);
        vcdp->declBus(c+1433,"VX_cache snp_rsp_arb sel_ffsq requests", false,-1, 3,0);
        vcdp->declBus(c+2321,"VX_cache snp_rsp_arb sel_ffsq grant_index", false,-1, 1,0);
        vcdp->declBus(c+2337,"VX_cache snp_rsp_arb sel_ffsq grant_onehot", false,-1, 3,0);
        vcdp->declBit(c+2329,"VX_cache snp_rsp_arb sel_ffsq grant_valid", false,-1);
        vcdp->declBus(c+2337,"VX_cache snp_rsp_arb sel_ffsq genblk2 grant_onehot_r", false,-1, 3,0);
        vcdp->declBus(c+25081,"VX_cache snp_rsp_arb sel_ffsq genblk2 priority_encoder N", false,-1, 31,0);
        vcdp->declBus(c+1433,"VX_cache snp_rsp_arb sel_ffsq genblk2 priority_encoder data_in", false,-1, 3,0);
        vcdp->declBus(c+2321,"VX_cache snp_rsp_arb sel_ffsq genblk2 priority_encoder data_out", false,-1, 1,0);
        vcdp->declBit(c+2329,"VX_cache snp_rsp_arb sel_ffsq genblk2 priority_encoder valid_out", false,-1);
        vcdp->declBus(c+2345,"VX_cache snp_rsp_arb sel_ffsq genblk2 priority_encoder i", false,-1, 31,0);
        vcdp->declBus(c+25057,"VX_cache genblk5[0] bank CACHE_ID", false,-1, 31,0);
        vcdp->declBus(c+25153,"VX_cache genblk5[0] bank BANK_ID", false,-1, 31,0);
        vcdp->declBus(c+25065,"VX_cache genblk5[0] bank CACHE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[0] bank BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank NUM_REQUESTS", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[0] bank STAGE_1_CYCLES", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank CREQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[0] bank MRVQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[0] bank DFPQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[0] bank SNRQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank CWBQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank DWBQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank DFQQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[0] bank WRITE_ENABLE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[0] bank DRAM_ENABLE", false,-1, 31,0);
        vcdp->declBus(c+25057,"VX_cache genblk5[0] bank SNOOP_FORWARDING", false,-1, 31,0);
        vcdp->declBus(c+25097,"VX_cache genblk5[0] bank CORE_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBus(c+25105,"VX_cache genblk5[0] bank CORE_TAG_ID_BITS", false,-1, 31,0);
        vcdp->declBus(c+25113,"VX_cache genblk5[0] bank SNP_REQ_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[0] bank clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[0] bank reset", false,-1);
        vcdp->declBus(c+33,"VX_cache genblk5[0] bank core_req_valid", false,-1, 3,0);
        vcdp->declBus(c+24513,"VX_cache genblk5[0] bank core_req_rw", false,-1, 3,0);
        vcdp->declBus(c+24521,"VX_cache genblk5[0] bank core_req_byteen", false,-1, 15,0);
        vcdp->declArray(c+24529,"VX_cache genblk5[0] bank core_req_addr", false,-1, 119,0);
        vcdp->declArray(c+24561,"VX_cache genblk5[0] bank core_req_data", false,-1, 127,0);
        vcdp->declQuad(c+24593,"VX_cache genblk5[0] bank core_req_tag", false,-1, 41,0);
        vcdp->declBit(c+10105,"VX_cache genblk5[0] bank core_req_ready", false,-1);
        vcdp->declBit(c+10073,"VX_cache genblk5[0] bank core_rsp_valid", false,-1);
        vcdp->declBus(c+1473,"VX_cache genblk5[0] bank core_rsp_tid", false,-1, 1,0);
        vcdp->declBus(c+1481,"VX_cache genblk5[0] bank core_rsp_data", false,-1, 31,0);
        vcdp->declQuad(c+1489,"VX_cache genblk5[0] bank core_rsp_tag", false,-1, 41,0);
        vcdp->declBit(c+41,"VX_cache genblk5[0] bank core_rsp_ready", false,-1);
        vcdp->declBit(c+1505,"VX_cache genblk5[0] bank dram_fill_req_valid", false,-1);
        vcdp->declBus(c+10089,"VX_cache genblk5[0] bank dram_fill_req_addr", false,-1, 25,0);
        vcdp->declBit(c+10065,"VX_cache genblk5[0] bank dram_fill_req_ready", false,-1);
        vcdp->declBit(c+24961,"VX_cache genblk5[0] bank dram_fill_rsp_valid", false,-1);
        vcdp->declArray(c+24769,"VX_cache genblk5[0] bank dram_fill_rsp_data", false,-1, 127,0);
        vcdp->declBus(c+24969,"VX_cache genblk5[0] bank dram_fill_rsp_addr", false,-1, 25,0);
        vcdp->declBit(c+10081,"VX_cache genblk5[0] bank dram_fill_rsp_ready", false,-1);
        vcdp->declBit(c+1513,"VX_cache genblk5[0] bank dram_wb_req_valid", false,-1);
        vcdp->declBus(c+1521,"VX_cache genblk5[0] bank dram_wb_req_byteen", false,-1, 15,0);
        vcdp->declBus(c+1529,"VX_cache genblk5[0] bank dram_wb_req_addr", false,-1, 25,0);
        vcdp->declArray(c+1537,"VX_cache genblk5[0] bank dram_wb_req_data", false,-1, 127,0);
        vcdp->declBit(c+49,"VX_cache genblk5[0] bank dram_wb_req_ready", false,-1);
        vcdp->declBit(c+24977,"VX_cache genblk5[0] bank snp_req_valid", false,-1);
        vcdp->declBus(c+24985,"VX_cache genblk5[0] bank snp_req_addr", false,-1, 25,0);
        vcdp->declBit(c+24833,"VX_cache genblk5[0] bank snp_req_invalidate", false,-1);
        vcdp->declBus(c+24841,"VX_cache genblk5[0] bank snp_req_tag", false,-1, 27,0);
        vcdp->declBit(c+10097,"VX_cache genblk5[0] bank snp_req_ready", false,-1);
        vcdp->declBit(c+1569,"VX_cache genblk5[0] bank snp_rsp_valid", false,-1);
        vcdp->declBus(c+1577,"VX_cache genblk5[0] bank snp_rsp_tag", false,-1, 27,0);
        vcdp->declBit(c+57,"VX_cache genblk5[0] bank snp_rsp_ready", false,-1);
        vcdp->declBit(c+2353,"VX_cache genblk5[0] bank snrq_pop", false,-1);
        vcdp->declBit(c+10601,"VX_cache genblk5[0] bank snrq_empty", false,-1);
        vcdp->declBit(c+10609,"VX_cache genblk5[0] bank snrq_full", false,-1);
        vcdp->declBus(c+2361,"VX_cache genblk5[0] bank snrq_addr_st0", false,-1, 25,0);
        vcdp->declBit(c+2369,"VX_cache genblk5[0] bank snrq_invalidate_st0", false,-1);
        vcdp->declBus(c+2377,"VX_cache genblk5[0] bank snrq_tag_st0", false,-1, 27,0);
        vcdp->declBit(c+2385,"VX_cache genblk5[0] bank dfpq_pop", false,-1);
        vcdp->declBit(c+10617,"VX_cache genblk5[0] bank dfpq_empty", false,-1);
        vcdp->declBit(c+10625,"VX_cache genblk5[0] bank dfpq_full", false,-1);
        vcdp->declBus(c+2393,"VX_cache genblk5[0] bank dfpq_addr_st0", false,-1, 25,0);
        vcdp->declArray(c+2401,"VX_cache genblk5[0] bank dfpq_filldata_st0", false,-1, 127,0);
        vcdp->declBit(c+2433,"VX_cache genblk5[0] bank reqq_pop", false,-1);
        vcdp->declBit(c+961,"VX_cache genblk5[0] bank reqq_push", false,-1);
        vcdp->declBit(c+2441,"VX_cache genblk5[0] bank reqq_empty", false,-1);
        vcdp->declBit(c+10633,"VX_cache genblk5[0] bank reqq_full", false,-1);
        vcdp->declBit(c+2449,"VX_cache genblk5[0] bank reqq_req_st0", false,-1);
        vcdp->declBus(c+2457,"VX_cache genblk5[0] bank reqq_req_tid_st0", false,-1, 1,0);
        vcdp->declBit(c+2465,"VX_cache genblk5[0] bank reqq_req_rw_st0", false,-1);
        vcdp->declBus(c+2473,"VX_cache genblk5[0] bank reqq_req_byteen_st0", false,-1, 3,0);
        vcdp->declBus(c+2481,"VX_cache genblk5[0] bank reqq_req_addr_st0", false,-1, 29,0);
        vcdp->declBus(c+2489,"VX_cache genblk5[0] bank reqq_req_writeword_st0", false,-1, 31,0);
        vcdp->declQuad(c+10641,"VX_cache genblk5[0] bank reqq_req_tag_st0", false,-1, 41,0);
        vcdp->declBit(c+2497,"VX_cache genblk5[0] bank mrvq_pop", false,-1);
        vcdp->declBit(c+10657,"VX_cache genblk5[0] bank mrvq_full", false,-1);
        vcdp->declBit(c+10665,"VX_cache genblk5[0] bank mrvq_stop", false,-1);
        vcdp->declBit(c+2505,"VX_cache genblk5[0] bank mrvq_valid_st0", false,-1);
        vcdp->declBus(c+10673,"VX_cache genblk5[0] bank mrvq_tid_st0", false,-1, 1,0);
        vcdp->declBus(c+10681,"VX_cache genblk5[0] bank mrvq_addr_st0", false,-1, 25,0);
        vcdp->declBus(c+10689,"VX_cache genblk5[0] bank mrvq_wsel_st0", false,-1, 1,0);
        vcdp->declBus(c+10697,"VX_cache genblk5[0] bank mrvq_writeword_st0", false,-1, 31,0);
        vcdp->declQuad(c+10705,"VX_cache genblk5[0] bank mrvq_tag_st0", false,-1, 41,0);
        vcdp->declBit(c+2513,"VX_cache genblk5[0] bank mrvq_rw_st0", false,-1);
        vcdp->declBus(c+10721,"VX_cache genblk5[0] bank mrvq_byteen_st0", false,-1, 3,0);
        vcdp->declBit(c+10729,"VX_cache genblk5[0] bank mrvq_is_snp_st0", false,-1);
        vcdp->declBit(c+10737,"VX_cache genblk5[0] bank mrvq_snp_invalidate_st0", false,-1);
        vcdp->declBit(c+2521,"VX_cache genblk5[0] bank mrvq_pending_hazard_st1e", false,-1);
        vcdp->declBit(c+2529,"VX_cache genblk5[0] bank st2_pending_hazard_st1e", false,-1);
        vcdp->declBit(c+2537,"VX_cache genblk5[0] bank force_request_miss_st1e", false,-1);
        vcdp->declBus(c+10745,"VX_cache genblk5[0] bank miss_add_tid", false,-1, 1,0);
        vcdp->declQuad(c+10753,"VX_cache genblk5[0] bank miss_add_tag", false,-1, 41,0);
        vcdp->declBit(c+10769,"VX_cache genblk5[0] bank miss_add_rw", false,-1);
        vcdp->declBus(c+10777,"VX_cache genblk5[0] bank miss_add_byteen", false,-1, 3,0);
        vcdp->declBus(c+10089,"VX_cache genblk5[0] bank addr_st2", false,-1, 25,0);
        vcdp->declBit(c+10785,"VX_cache genblk5[0] bank is_fill_st2", false,-1);
        vcdp->declBit(c+2545,"VX_cache genblk5[0] bank recover_mrvq_state_st2", false,-1);
        vcdp->declBit(c+2553,"VX_cache genblk5[0] bank mrvq_push_stall", false,-1);
        vcdp->declBit(c+2561,"VX_cache genblk5[0] bank cwbq_push_stall", false,-1);
        vcdp->declBit(c+2569,"VX_cache genblk5[0] bank dwbq_push_stall", false,-1);
        vcdp->declBit(c+2577,"VX_cache genblk5[0] bank dram_fill_req_stall", false,-1);
        vcdp->declBit(c+2585,"VX_cache genblk5[0] bank stall_bank_pipe", false,-1);
        vcdp->declBit(c+2593,"VX_cache genblk5[0] bank is_fill_in_pipe", false,-1);
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+2601+i*1,"VX_cache genblk5[0] bank is_fill_st1", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+2609+i*1,"VX_cache genblk5[0] bank going_to_write_st1", true,(i+0));}}
        vcdp->declBus(c+25161,"VX_cache genblk5[0] bank j", false,-1, 31,0);
        vcdp->declBit(c+2505,"VX_cache genblk5[0] bank mrvq_pop_unqual", false,-1);
        vcdp->declBit(c+2617,"VX_cache genblk5[0] bank dfpq_pop_unqual", false,-1);
        vcdp->declBit(c+2625,"VX_cache genblk5[0] bank reqq_pop_unqual", false,-1);
        vcdp->declBit(c+2633,"VX_cache genblk5[0] bank snrq_pop_unqual", false,-1);
        vcdp->declBit(c+2617,"VX_cache genblk5[0] bank qual_is_fill_st0", false,-1);
        vcdp->declBit(c+2641,"VX_cache genblk5[0] bank qual_valid_st0", false,-1);
        vcdp->declBus(c+2649,"VX_cache genblk5[0] bank qual_addr_st0", false,-1, 25,0);
        vcdp->declBus(c+2657,"VX_cache genblk5[0] bank qual_wsel_st0", false,-1, 1,0);
        vcdp->declBit(c+2505,"VX_cache genblk5[0] bank qual_is_mrvq_st0", false,-1);
        vcdp->declBus(c+2665,"VX_cache genblk5[0] bank qual_writeword_st0", false,-1, 31,0);
        vcdp->declArray(c+2673,"VX_cache genblk5[0] bank qual_writedata_st0", false,-1, 127,0);
        vcdp->declQuad(c+2705,"VX_cache genblk5[0] bank qual_inst_meta_st0", false,-1, 48,0);
        vcdp->declBit(c+2721,"VX_cache genblk5[0] bank qual_going_to_write_st0", false,-1);
        vcdp->declBit(c+2729,"VX_cache genblk5[0] bank qual_is_snp_st0", false,-1);
        vcdp->declBit(c+2737,"VX_cache genblk5[0] bank qual_snp_invalidate_st0", false,-1);
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+2745+i*1,"VX_cache genblk5[0] bank valid_st1", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+2753+i*1,"VX_cache genblk5[0] bank addr_st1", true,(i+0), 25,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+2761+i*1,"VX_cache genblk5[0] bank wsel_st1", true,(i+0), 1,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+2769+i*1,"VX_cache genblk5[0] bank writeword_st1", true,(i+0), 31,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declQuad(c+2777+i*2,"VX_cache genblk5[0] bank inst_meta_st1", true,(i+0), 48,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declArray(c+2793+i*4,"VX_cache genblk5[0] bank writedata_st1", true,(i+0), 127,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+2825+i*1,"VX_cache genblk5[0] bank is_snp_st1", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+2833+i*1,"VX_cache genblk5[0] bank snp_invalidate_st1", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+2841+i*1,"VX_cache genblk5[0] bank is_mrvq_st1", true,(i+0));}}
        vcdp->declBus(c+2849,"VX_cache genblk5[0] bank readword_st1e", false,-1, 31,0);
        vcdp->declArray(c+2857,"VX_cache genblk5[0] bank readdata_st1e", false,-1, 127,0);
        vcdp->declBus(c+2889,"VX_cache genblk5[0] bank readtag_st1e", false,-1, 19,0);
        vcdp->declBit(c+2897,"VX_cache genblk5[0] bank miss_st1e", false,-1);
        vcdp->declBit(c+2905,"VX_cache genblk5[0] bank dirty_st1e", false,-1);
        vcdp->declBus(c+2913,"VX_cache genblk5[0] bank dirtyb_st1e", false,-1, 15,0);
        vcdp->declQuad(c+2921,"VX_cache genblk5[0] bank tag_st1e", false,-1, 41,0);
        vcdp->declBus(c+2937,"VX_cache genblk5[0] bank tid_st1e", false,-1, 1,0);
        vcdp->declBit(c+2945,"VX_cache genblk5[0] bank mem_rw_st1e", false,-1);
        vcdp->declBus(c+2953,"VX_cache genblk5[0] bank mem_byteen_st1e", false,-1, 3,0);
        vcdp->declBit(c+2961,"VX_cache genblk5[0] bank fill_saw_dirty_st1e", false,-1);
        vcdp->declBit(c+2969,"VX_cache genblk5[0] bank is_snp_st1e", false,-1);
        vcdp->declBit(c+2977,"VX_cache genblk5[0] bank snp_invalidate_st1e", false,-1);
        vcdp->declBit(c+2985,"VX_cache genblk5[0] bank snp_to_mrvq_st1e", false,-1);
        vcdp->declBit(c+2993,"VX_cache genblk5[0] bank mrvq_init_ready_state_st1e", false,-1);
        vcdp->declBit(c+3001,"VX_cache genblk5[0] bank miss_add_because_miss", false,-1);
        vcdp->declBit(c+3009,"VX_cache genblk5[0] bank valid_st1e", false,-1);
        vcdp->declBit(c+3017,"VX_cache genblk5[0] bank is_mrvq_st1e", false,-1);
        vcdp->declBit(c+3025,"VX_cache genblk5[0] bank mrvq_recover_ready_state_st1e", false,-1);
        vcdp->declBus(c+3033,"VX_cache genblk5[0] bank addr_st1e", false,-1, 25,0);
        vcdp->declBit(c+3041,"VX_cache genblk5[0] bank qual_valid_st1e_2", false,-1);
        vcdp->declBit(c+3017,"VX_cache genblk5[0] bank is_mrvq_st1e_st2", false,-1);
        vcdp->declBit(c+10793,"VX_cache genblk5[0] bank valid_st2", false,-1);
        vcdp->declBus(c+10801,"VX_cache genblk5[0] bank wsel_st2", false,-1, 1,0);
        vcdp->declBus(c+10809,"VX_cache genblk5[0] bank writeword_st2", false,-1, 31,0);
        vcdp->declBus(c+10817,"VX_cache genblk5[0] bank readword_st2", false,-1, 31,0);
        vcdp->declArray(c+10825,"VX_cache genblk5[0] bank readdata_st2", false,-1, 127,0);
        vcdp->declBit(c+10857,"VX_cache genblk5[0] bank miss_st2", false,-1);
        vcdp->declBit(c+10865,"VX_cache genblk5[0] bank dirty_st2", false,-1);
        vcdp->declBus(c+10873,"VX_cache genblk5[0] bank dirtyb_st2", false,-1, 15,0);
        vcdp->declQuad(c+10881,"VX_cache genblk5[0] bank inst_meta_st2", false,-1, 48,0);
        vcdp->declBus(c+10897,"VX_cache genblk5[0] bank readtag_st2", false,-1, 19,0);
        vcdp->declBit(c+10905,"VX_cache genblk5[0] bank fill_saw_dirty_st2", false,-1);
        vcdp->declBit(c+10913,"VX_cache genblk5[0] bank is_snp_st2", false,-1);
        vcdp->declBit(c+10921,"VX_cache genblk5[0] bank snp_invalidate_st2", false,-1);
        vcdp->declBit(c+10929,"VX_cache genblk5[0] bank snp_to_mrvq_st2", false,-1);
        vcdp->declBit(c+10937,"VX_cache genblk5[0] bank is_mrvq_st2", false,-1);
        vcdp->declBit(c+3049,"VX_cache genblk5[0] bank mrvq_init_ready_state_st2", false,-1);
        vcdp->declBit(c+10945,"VX_cache genblk5[0] bank mrvq_recover_ready_state_st2", false,-1);
        vcdp->declBit(c+10953,"VX_cache genblk5[0] bank mrvq_init_ready_state_unqual_st2", false,-1);
        vcdp->declBit(c+3057,"VX_cache genblk5[0] bank mrvq_init_ready_state_hazard_st0_st1", false,-1);
        vcdp->declBit(c+3065,"VX_cache genblk5[0] bank mrvq_init_ready_state_hazard_st1e_st1", false,-1);
        vcdp->declBit(c+10929,"VX_cache genblk5[0] bank miss_add_because_pending", false,-1);
        vcdp->declBit(c+3073,"VX_cache genblk5[0] bank miss_add_unqual", false,-1);
        vcdp->declBit(c+3081,"VX_cache genblk5[0] bank miss_add", false,-1);
        vcdp->declBus(c+10089,"VX_cache genblk5[0] bank miss_add_addr", false,-1, 25,0);
        vcdp->declBus(c+10801,"VX_cache genblk5[0] bank miss_add_wsel", false,-1, 1,0);
        vcdp->declBus(c+10809,"VX_cache genblk5[0] bank miss_add_data", false,-1, 31,0);
        vcdp->declBit(c+10913,"VX_cache genblk5[0] bank miss_add_is_snp", false,-1);
        vcdp->declBit(c+10921,"VX_cache genblk5[0] bank miss_add_snp_invalidate", false,-1);
        vcdp->declBit(c+3089,"VX_cache genblk5[0] bank miss_add_is_mrvq", false,-1);
        vcdp->declBit(c+3097,"VX_cache genblk5[0] bank cwbq_push", false,-1);
        vcdp->declBit(c+969,"VX_cache genblk5[0] bank cwbq_pop", false,-1);
        vcdp->declBit(c+10961,"VX_cache genblk5[0] bank cwbq_empty", false,-1);
        vcdp->declBit(c+10969,"VX_cache genblk5[0] bank cwbq_full", false,-1);
        vcdp->declBit(c+3105,"VX_cache genblk5[0] bank cwbq_push_unqual", false,-1);
        vcdp->declBus(c+10817,"VX_cache genblk5[0] bank cwbq_data", false,-1, 31,0);
        vcdp->declBus(c+10745,"VX_cache genblk5[0] bank cwbq_tid", false,-1, 1,0);
        vcdp->declQuad(c+10753,"VX_cache genblk5[0] bank cwbq_tag", false,-1, 41,0);
        vcdp->declBit(c+3073,"VX_cache genblk5[0] bank dram_fill_req_fast", false,-1);
        vcdp->declBit(c+3113,"VX_cache genblk5[0] bank dram_fill_req_unqual", false,-1);
        vcdp->declBit(c+3121,"VX_cache genblk5[0] bank dwbq_push", false,-1);
        vcdp->declBit(c+977,"VX_cache genblk5[0] bank dwbq_pop", false,-1);
        vcdp->declBit(c+10977,"VX_cache genblk5[0] bank dwbq_empty", false,-1);
        vcdp->declBit(c+10985,"VX_cache genblk5[0] bank dwbq_full", false,-1);
        vcdp->declBit(c+3129,"VX_cache genblk5[0] bank dwbq_is_dwb_in", false,-1);
        vcdp->declBit(c+3137,"VX_cache genblk5[0] bank dwbq_is_snp_in", false,-1);
        vcdp->declBit(c+3145,"VX_cache genblk5[0] bank dwbq_is_dwb_out", false,-1);
        vcdp->declBit(c+3153,"VX_cache genblk5[0] bank dwbq_is_snp_out", false,-1);
        vcdp->declBit(c+3161,"VX_cache genblk5[0] bank dwbq_push_unqual", false,-1);
        vcdp->declBus(c+10993,"VX_cache genblk5[0] bank dwbq_req_addr", false,-1, 25,0);
        vcdp->declBus(c+11001,"VX_cache genblk5[0] bank snrq_tag_st2", false,-1, 27,0);
        vcdp->declBit(c+185,"VX_cache genblk5[0] bank dram_wb_req_fire", false,-1);
        vcdp->declBit(c+193,"VX_cache genblk5[0] bank snp_rsp_fire", false,-1);
        vcdp->declBit(c+11009,"VX_cache genblk5[0] bank dwbq_dual_valid_sel", false,-1);
        vcdp->declBus(c+25169,"VX_cache genblk5[0] bank snp_req_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[0] bank snp_req_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[0] bank snp_req_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[0] bank snp_req_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[0] bank snp_req_queue reset", false,-1);
        vcdp->declBit(c+24977,"VX_cache genblk5[0] bank snp_req_queue push", false,-1);
        vcdp->declBit(c+2353,"VX_cache genblk5[0] bank snp_req_queue pop", false,-1);
        vcdp->declQuad(c+201,"VX_cache genblk5[0] bank snp_req_queue data_in", false,-1, 54,0);
        vcdp->declQuad(c+3169,"VX_cache genblk5[0] bank snp_req_queue data_out", false,-1, 54,0);
        vcdp->declBit(c+10601,"VX_cache genblk5[0] bank snp_req_queue empty", false,-1);
        vcdp->declBit(c+10609,"VX_cache genblk5[0] bank snp_req_queue full", false,-1);
        vcdp->declBus(c+11017,"VX_cache genblk5[0] bank snp_req_queue size", false,-1, 4,0);
        vcdp->declBus(c+11017,"VX_cache genblk5[0] bank snp_req_queue size_r", false,-1, 4,0);
        vcdp->declBit(c+3185,"VX_cache genblk5[0] bank snp_req_queue reading", false,-1);
        vcdp->declBit(c+217,"VX_cache genblk5[0] bank snp_req_queue writing", false,-1);
        {int i; for (i=0; i<16; i++) {
                vcdp->declQuad(c+11025+i*2,"VX_cache genblk5[0] bank snp_req_queue genblk3 data", true,(i+0), 54,0);}}
        vcdp->declQuad(c+11281,"VX_cache genblk5[0] bank snp_req_queue genblk3 genblk2 head_r", false,-1, 54,0);
        vcdp->declQuad(c+11297,"VX_cache genblk5[0] bank snp_req_queue genblk3 genblk2 curr_r", false,-1, 54,0);
        vcdp->declBus(c+11313,"VX_cache genblk5[0] bank snp_req_queue genblk3 genblk2 wr_ptr_r", false,-1, 3,0);
        vcdp->declBus(c+11321,"VX_cache genblk5[0] bank snp_req_queue genblk3 genblk2 rd_ptr_r", false,-1, 3,0);
        vcdp->declBus(c+11329,"VX_cache genblk5[0] bank snp_req_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 3,0);
        vcdp->declBit(c+10601,"VX_cache genblk5[0] bank snp_req_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+10609,"VX_cache genblk5[0] bank snp_req_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+11337,"VX_cache genblk5[0] bank snp_req_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25177,"VX_cache genblk5[0] bank dfp_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[0] bank dfp_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[0] bank dfp_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[0] bank dfp_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[0] bank dfp_queue reset", false,-1);
        vcdp->declBit(c+24961,"VX_cache genblk5[0] bank dfp_queue push", false,-1);
        vcdp->declBit(c+2385,"VX_cache genblk5[0] bank dfp_queue pop", false,-1);
        vcdp->declArray(c+225,"VX_cache genblk5[0] bank dfp_queue data_in", false,-1, 153,0);
        vcdp->declArray(c+3193,"VX_cache genblk5[0] bank dfp_queue data_out", false,-1, 153,0);
        vcdp->declBit(c+10617,"VX_cache genblk5[0] bank dfp_queue empty", false,-1);
        vcdp->declBit(c+10625,"VX_cache genblk5[0] bank dfp_queue full", false,-1);
        vcdp->declBus(c+11345,"VX_cache genblk5[0] bank dfp_queue size", false,-1, 4,0);
        vcdp->declBus(c+11345,"VX_cache genblk5[0] bank dfp_queue size_r", false,-1, 4,0);
        vcdp->declBit(c+3233,"VX_cache genblk5[0] bank dfp_queue reading", false,-1);
        vcdp->declBit(c+265,"VX_cache genblk5[0] bank dfp_queue writing", false,-1);
        {int i; for (i=0; i<16; i++) {
                vcdp->declArray(c+11353+i*5,"VX_cache genblk5[0] bank dfp_queue genblk3 data", true,(i+0), 153,0);}}
        vcdp->declArray(c+11993,"VX_cache genblk5[0] bank dfp_queue genblk3 genblk2 head_r", false,-1, 153,0);
        vcdp->declArray(c+12033,"VX_cache genblk5[0] bank dfp_queue genblk3 genblk2 curr_r", false,-1, 153,0);
        vcdp->declBus(c+12073,"VX_cache genblk5[0] bank dfp_queue genblk3 genblk2 wr_ptr_r", false,-1, 3,0);
        vcdp->declBus(c+12081,"VX_cache genblk5[0] bank dfp_queue genblk3 genblk2 rd_ptr_r", false,-1, 3,0);
        vcdp->declBus(c+12089,"VX_cache genblk5[0] bank dfp_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 3,0);
        vcdp->declBit(c+10617,"VX_cache genblk5[0] bank dfp_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+10625,"VX_cache genblk5[0] bank dfp_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+12097,"VX_cache genblk5[0] bank dfp_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank core_req_arb WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank core_req_arb NUM_REQUESTS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank core_req_arb CREQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25097,"VX_cache genblk5[0] bank core_req_arb CORE_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBus(c+25105,"VX_cache genblk5[0] bank core_req_arb CORE_TAG_ID_BITS", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[0] bank core_req_arb clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[0] bank core_req_arb reset", false,-1);
        vcdp->declBit(c+961,"VX_cache genblk5[0] bank core_req_arb reqq_push", false,-1);
        vcdp->declBus(c+33,"VX_cache genblk5[0] bank core_req_arb bank_valids", false,-1, 3,0);
        vcdp->declBus(c+24513,"VX_cache genblk5[0] bank core_req_arb bank_rw", false,-1, 3,0);
        vcdp->declBus(c+24521,"VX_cache genblk5[0] bank core_req_arb bank_byteen", false,-1, 15,0);
        vcdp->declArray(c+24561,"VX_cache genblk5[0] bank core_req_arb bank_writedata", false,-1, 127,0);
        vcdp->declArray(c+24529,"VX_cache genblk5[0] bank core_req_arb bank_addr", false,-1, 119,0);
        vcdp->declQuad(c+24593,"VX_cache genblk5[0] bank core_req_arb bank_tag", false,-1, 41,0);
        vcdp->declBit(c+2433,"VX_cache genblk5[0] bank core_req_arb reqq_pop", false,-1);
        vcdp->declBit(c+2449,"VX_cache genblk5[0] bank core_req_arb reqq_req_st0", false,-1);
        vcdp->declBus(c+2457,"VX_cache genblk5[0] bank core_req_arb reqq_req_tid_st0", false,-1, 1,0);
        vcdp->declBit(c+2465,"VX_cache genblk5[0] bank core_req_arb reqq_req_rw_st0", false,-1);
        vcdp->declBus(c+2473,"VX_cache genblk5[0] bank core_req_arb reqq_req_byteen_st0", false,-1, 3,0);
        vcdp->declBus(c+2481,"VX_cache genblk5[0] bank core_req_arb reqq_req_addr_st0", false,-1, 29,0);
        vcdp->declBus(c+2489,"VX_cache genblk5[0] bank core_req_arb reqq_req_writedata_st0", false,-1, 31,0);
        vcdp->declQuad(c+10641,"VX_cache genblk5[0] bank core_req_arb reqq_req_tag_st0", false,-1, 41,0);
        vcdp->declBit(c+2441,"VX_cache genblk5[0] bank core_req_arb reqq_empty", false,-1);
        vcdp->declBit(c+10633,"VX_cache genblk5[0] bank core_req_arb reqq_full", false,-1);
        vcdp->declBus(c+3241,"VX_cache genblk5[0] bank core_req_arb out_per_valids", false,-1, 3,0);
        vcdp->declBus(c+3249,"VX_cache genblk5[0] bank core_req_arb out_per_rw", false,-1, 3,0);
        vcdp->declBus(c+3257,"VX_cache genblk5[0] bank core_req_arb out_per_byteen", false,-1, 15,0);
        vcdp->declArray(c+3265,"VX_cache genblk5[0] bank core_req_arb out_per_addr", false,-1, 119,0);
        vcdp->declArray(c+3297,"VX_cache genblk5[0] bank core_req_arb out_per_writedata", false,-1, 127,0);
        vcdp->declQuad(c+3329,"VX_cache genblk5[0] bank core_req_arb out_per_tag", false,-1, 41,0);
        vcdp->declBus(c+12105,"VX_cache genblk5[0] bank core_req_arb use_per_valids", false,-1, 3,0);
        vcdp->declBus(c+12113,"VX_cache genblk5[0] bank core_req_arb use_per_rw", false,-1, 3,0);
        vcdp->declBus(c+12121,"VX_cache genblk5[0] bank core_req_arb use_per_byteen", false,-1, 15,0);
        vcdp->declArray(c+12129,"VX_cache genblk5[0] bank core_req_arb use_per_addr", false,-1, 119,0);
        vcdp->declArray(c+12161,"VX_cache genblk5[0] bank core_req_arb use_per_writedata", false,-1, 127,0);
        vcdp->declQuad(c+10641,"VX_cache genblk5[0] bank core_req_arb use_per_tag", false,-1, 41,0);
        vcdp->declBus(c+12105,"VX_cache genblk5[0] bank core_req_arb qual_valids", false,-1, 3,0);
        vcdp->declBus(c+12113,"VX_cache genblk5[0] bank core_req_arb qual_rw", false,-1, 3,0);
        vcdp->declBus(c+12121,"VX_cache genblk5[0] bank core_req_arb qual_byteen", false,-1, 15,0);
        vcdp->declArray(c+12129,"VX_cache genblk5[0] bank core_req_arb qual_addr", false,-1, 119,0);
        vcdp->declArray(c+12161,"VX_cache genblk5[0] bank core_req_arb qual_writedata", false,-1, 127,0);
        vcdp->declQuad(c+10641,"VX_cache genblk5[0] bank core_req_arb qual_tag", false,-1, 41,0);
        vcdp->declBit(c+12193,"VX_cache genblk5[0] bank core_req_arb o_empty", false,-1);
        vcdp->declBit(c+12201,"VX_cache genblk5[0] bank core_req_arb use_empty", false,-1);
        vcdp->declBit(c+3345,"VX_cache genblk5[0] bank core_req_arb out_empty", false,-1);
        vcdp->declBit(c+985,"VX_cache genblk5[0] bank core_req_arb push_qual", false,-1);
        vcdp->declBit(c+3353,"VX_cache genblk5[0] bank core_req_arb pop_qual", false,-1);
        vcdp->declBus(c+3361,"VX_cache genblk5[0] bank core_req_arb real_out_per_valids", false,-1, 3,0);
        vcdp->declBus(c+2457,"VX_cache genblk5[0] bank core_req_arb qual_request_index", false,-1, 1,0);
        vcdp->declBit(c+2449,"VX_cache genblk5[0] bank core_req_arb qual_has_request", false,-1);
        vcdp->declBus(c+25185,"VX_cache genblk5[0] bank core_req_arb reqq_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank core_req_arb reqq_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[0] bank core_req_arb reqq_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[0] bank core_req_arb reqq_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[0] bank core_req_arb reqq_queue reset", false,-1);
        vcdp->declBit(c+985,"VX_cache genblk5[0] bank core_req_arb reqq_queue push", false,-1);
        vcdp->declBit(c+3353,"VX_cache genblk5[0] bank core_req_arb reqq_queue pop", false,-1);
        vcdp->declArray(c+273,"VX_cache genblk5[0] bank core_req_arb reqq_queue data_in", false,-1, 313,0);
        vcdp->declArray(c+3369,"VX_cache genblk5[0] bank core_req_arb reqq_queue data_out", false,-1, 313,0);
        vcdp->declBit(c+12193,"VX_cache genblk5[0] bank core_req_arb reqq_queue empty", false,-1);
        vcdp->declBit(c+10633,"VX_cache genblk5[0] bank core_req_arb reqq_queue full", false,-1);
        vcdp->declBus(c+12209,"VX_cache genblk5[0] bank core_req_arb reqq_queue size", false,-1, 2,0);
        vcdp->declBus(c+12209,"VX_cache genblk5[0] bank core_req_arb reqq_queue size_r", false,-1, 2,0);
        vcdp->declBit(c+3449,"VX_cache genblk5[0] bank core_req_arb reqq_queue reading", false,-1);
        vcdp->declBit(c+353,"VX_cache genblk5[0] bank core_req_arb reqq_queue writing", false,-1);
        {int i; for (i=0; i<4; i++) {
                vcdp->declArray(c+12217+i*10,"VX_cache genblk5[0] bank core_req_arb reqq_queue genblk3 data", true,(i+0), 313,0);}}
        vcdp->declArray(c+12537,"VX_cache genblk5[0] bank core_req_arb reqq_queue genblk3 genblk2 head_r", false,-1, 313,0);
        vcdp->declArray(c+12617,"VX_cache genblk5[0] bank core_req_arb reqq_queue genblk3 genblk2 curr_r", false,-1, 313,0);
        vcdp->declBus(c+12697,"VX_cache genblk5[0] bank core_req_arb reqq_queue genblk3 genblk2 wr_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+12705,"VX_cache genblk5[0] bank core_req_arb reqq_queue genblk3 genblk2 rd_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+12713,"VX_cache genblk5[0] bank core_req_arb reqq_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 1,0);
        vcdp->declBit(c+12193,"VX_cache genblk5[0] bank core_req_arb reqq_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+10633,"VX_cache genblk5[0] bank core_req_arb reqq_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+12721,"VX_cache genblk5[0] bank core_req_arb reqq_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank core_req_arb sel_bank N", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[0] bank core_req_arb sel_bank clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[0] bank core_req_arb sel_bank reset", false,-1);
        vcdp->declBus(c+12105,"VX_cache genblk5[0] bank core_req_arb sel_bank requests", false,-1, 3,0);
        vcdp->declBus(c+2457,"VX_cache genblk5[0] bank core_req_arb sel_bank grant_index", false,-1, 1,0);
        vcdp->declBus(c+3457,"VX_cache genblk5[0] bank core_req_arb sel_bank grant_onehot", false,-1, 3,0);
        vcdp->declBit(c+2449,"VX_cache genblk5[0] bank core_req_arb sel_bank grant_valid", false,-1);
        vcdp->declBus(c+3457,"VX_cache genblk5[0] bank core_req_arb sel_bank genblk2 grant_onehot_r", false,-1, 3,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank core_req_arb sel_bank genblk2 priority_encoder N", false,-1, 31,0);
        vcdp->declBus(c+12105,"VX_cache genblk5[0] bank core_req_arb sel_bank genblk2 priority_encoder data_in", false,-1, 3,0);
        vcdp->declBus(c+2457,"VX_cache genblk5[0] bank core_req_arb sel_bank genblk2 priority_encoder data_out", false,-1, 1,0);
        vcdp->declBit(c+2449,"VX_cache genblk5[0] bank core_req_arb sel_bank genblk2 priority_encoder valid_out", false,-1);
        vcdp->declBus(c+3465,"VX_cache genblk5[0] bank core_req_arb sel_bank genblk2 priority_encoder i", false,-1, 31,0);
        vcdp->declBus(c+25193,"VX_cache genblk5[0] bank s0_1_c0 N", false,-1, 31,0);
        vcdp->declBus(c+25057,"VX_cache genblk5[0] bank s0_1_c0 PASSTHRU", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[0] bank s0_1_c0 clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[0] bank s0_1_c0 reset", false,-1);
        vcdp->declBit(c+2585,"VX_cache genblk5[0] bank s0_1_c0 stall", false,-1);
        vcdp->declBit(c+25137,"VX_cache genblk5[0] bank s0_1_c0 flush", false,-1);
        vcdp->declArray(c+3473,"VX_cache genblk5[0] bank s0_1_c0 in", false,-1, 242,0);
        vcdp->declArray(c+12729,"VX_cache genblk5[0] bank s0_1_c0 out", false,-1, 242,0);
        vcdp->declArray(c+12729,"VX_cache genblk5[0] bank s0_1_c0 value", false,-1, 242,0);
        vcdp->declBus(c+25065,"VX_cache genblk5[0] bank tag_data_access CACHE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[0] bank tag_data_access BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank tag_data_access NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank tag_data_access WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[0] bank tag_data_access STAGE_1_CYCLES", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[0] bank tag_data_access WRITE_ENABLE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[0] bank tag_data_access DRAM_ENABLE", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[0] bank tag_data_access clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[0] bank tag_data_access reset", false,-1);
        vcdp->declBit(c+2585,"VX_cache genblk5[0] bank tag_data_access stall", false,-1);
        vcdp->declBit(c+2969,"VX_cache genblk5[0] bank tag_data_access is_snp_st1e", false,-1);
        vcdp->declBit(c+2977,"VX_cache genblk5[0] bank tag_data_access snp_invalidate_st1e", false,-1);
        vcdp->declBit(c+2585,"VX_cache genblk5[0] bank tag_data_access stall_bank_pipe", false,-1);
        vcdp->declBit(c+2537,"VX_cache genblk5[0] bank tag_data_access force_request_miss_st1e", false,-1);
        vcdp->declBus(c+3537,"VX_cache genblk5[0] bank tag_data_access readaddr_st10", false,-1, 5,0);
        vcdp->declBus(c+3033,"VX_cache genblk5[0] bank tag_data_access writeaddr_st1e", false,-1, 25,0);
        vcdp->declBit(c+3009,"VX_cache genblk5[0] bank tag_data_access valid_req_st1e", false,-1);
        vcdp->declBit(c+3545,"VX_cache genblk5[0] bank tag_data_access writefill_st1e", false,-1);
        vcdp->declBus(c+3553,"VX_cache genblk5[0] bank tag_data_access writeword_st1e", false,-1, 31,0);
        vcdp->declArray(c+3561,"VX_cache genblk5[0] bank tag_data_access writedata_st1e", false,-1, 127,0);
        vcdp->declBit(c+2945,"VX_cache genblk5[0] bank tag_data_access mem_rw_st1e", false,-1);
        vcdp->declBus(c+2953,"VX_cache genblk5[0] bank tag_data_access mem_byteen_st1e", false,-1, 3,0);
        vcdp->declBus(c+3593,"VX_cache genblk5[0] bank tag_data_access wordsel_st1e", false,-1, 1,0);
        vcdp->declBus(c+2849,"VX_cache genblk5[0] bank tag_data_access readword_st1e", false,-1, 31,0);
        vcdp->declArray(c+2857,"VX_cache genblk5[0] bank tag_data_access readdata_st1e", false,-1, 127,0);
        vcdp->declBus(c+2889,"VX_cache genblk5[0] bank tag_data_access readtag_st1e", false,-1, 19,0);
        vcdp->declBit(c+2897,"VX_cache genblk5[0] bank tag_data_access miss_st1e", false,-1);
        vcdp->declBit(c+2905,"VX_cache genblk5[0] bank tag_data_access dirty_st1e", false,-1);
        vcdp->declBus(c+2913,"VX_cache genblk5[0] bank tag_data_access dirtyb_st1e", false,-1, 15,0);
        vcdp->declBit(c+2961,"VX_cache genblk5[0] bank tag_data_access fill_saw_dirty_st1e", false,-1);
        vcdp->declBit(c+2985,"VX_cache genblk5[0] bank tag_data_access snp_to_mrvq_st1e", false,-1);
        vcdp->declBit(c+2993,"VX_cache genblk5[0] bank tag_data_access mrvq_init_ready_state_st1e", false,-1);
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+3601+i*1,"VX_cache genblk5[0] bank tag_data_access read_valid_st1c", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+3609+i*1,"VX_cache genblk5[0] bank tag_data_access read_dirty_st1c", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+3617+i*1,"VX_cache genblk5[0] bank tag_data_access read_dirtyb_st1c", true,(i+0), 15,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+3625+i*1,"VX_cache genblk5[0] bank tag_data_access read_tag_st1c", true,(i+0), 19,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declArray(c+3633+i*4,"VX_cache genblk5[0] bank tag_data_access read_data_st1c", true,(i+0), 127,0);}}
        vcdp->declBit(c+3665,"VX_cache genblk5[0] bank tag_data_access qual_read_valid_st1", false,-1);
        vcdp->declBit(c+3673,"VX_cache genblk5[0] bank tag_data_access qual_read_dirty_st1", false,-1);
        vcdp->declBus(c+3681,"VX_cache genblk5[0] bank tag_data_access qual_read_dirtyb_st1", false,-1, 15,0);
        vcdp->declBus(c+3689,"VX_cache genblk5[0] bank tag_data_access qual_read_tag_st1", false,-1, 19,0);
        vcdp->declArray(c+3697,"VX_cache genblk5[0] bank tag_data_access qual_read_data_st1", false,-1, 127,0);
        vcdp->declBit(c+3729,"VX_cache genblk5[0] bank tag_data_access use_read_valid_st1e", false,-1);
        vcdp->declBit(c+3737,"VX_cache genblk5[0] bank tag_data_access use_read_dirty_st1e", false,-1);
        vcdp->declBus(c+2913,"VX_cache genblk5[0] bank tag_data_access use_read_dirtyb_st1e", false,-1, 15,0);
        vcdp->declBus(c+2889,"VX_cache genblk5[0] bank tag_data_access use_read_tag_st1e", false,-1, 19,0);
        vcdp->declArray(c+2857,"VX_cache genblk5[0] bank tag_data_access use_read_data_st1e", false,-1, 127,0);
        vcdp->declBus(c+3745,"VX_cache genblk5[0] bank tag_data_access use_write_enable", false,-1, 15,0);
        vcdp->declArray(c+3753,"VX_cache genblk5[0] bank tag_data_access use_write_data", false,-1, 127,0);
        vcdp->declBit(c+2897,"VX_cache genblk5[0] bank tag_data_access fill_sent", false,-1);
        vcdp->declBit(c+3785,"VX_cache genblk5[0] bank tag_data_access invalidate_line", false,-1);
        vcdp->declBit(c+3793,"VX_cache genblk5[0] bank tag_data_access tags_match", false,-1);
        vcdp->declBit(c+3801,"VX_cache genblk5[0] bank tag_data_access real_writefill", false,-1);
        vcdp->declBus(c+3809,"VX_cache genblk5[0] bank tag_data_access writetag_st1e", false,-1, 19,0);
        vcdp->declBus(c+3537,"VX_cache genblk5[0] bank tag_data_access writeladdr_st1e", false,-1, 5,0);
        vcdp->declBus(c+3817,"VX_cache genblk5[0] bank tag_data_access we", false,-1, 15,0);
        vcdp->declArray(c+3753,"VX_cache genblk5[0] bank tag_data_access data_write", false,-1, 127,0);
        vcdp->declBit(c+3825,"VX_cache genblk5[0] bank tag_data_access should_write", false,-1);
        vcdp->declBit(c+3785,"VX_cache genblk5[0] bank tag_data_access snoop_hit_no_pending", false,-1);
        vcdp->declBit(c+3833,"VX_cache genblk5[0] bank tag_data_access req_invalid", false,-1);
        vcdp->declBit(c+3841,"VX_cache genblk5[0] bank tag_data_access req_miss", false,-1);
        vcdp->declBit(c+3849,"VX_cache genblk5[0] bank tag_data_access real_miss", false,-1);
        vcdp->declBit(c+3857,"VX_cache genblk5[0] bank tag_data_access force_core_miss", false,-1);
        vcdp->declBit(c+3865,"VX_cache genblk5[0] bank tag_data_access genblk4[0] normal_write", false,-1);
        vcdp->declBit(c+3873,"VX_cache genblk5[0] bank tag_data_access genblk4[1] normal_write", false,-1);
        vcdp->declBit(c+3881,"VX_cache genblk5[0] bank tag_data_access genblk4[2] normal_write", false,-1);
        vcdp->declBit(c+3889,"VX_cache genblk5[0] bank tag_data_access genblk4[3] normal_write", false,-1);
        vcdp->declBus(c+25065,"VX_cache genblk5[0] bank tag_data_access tag_data_structure CACHE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[0] bank tag_data_access tag_data_structure BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank tag_data_access tag_data_structure NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank tag_data_access tag_data_structure WORD_SIZE", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[0] bank tag_data_access tag_data_structure clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[0] bank tag_data_access tag_data_structure reset", false,-1);
        vcdp->declBit(c+2585,"VX_cache genblk5[0] bank tag_data_access tag_data_structure stall_bank_pipe", false,-1);
        vcdp->declBus(c+3537,"VX_cache genblk5[0] bank tag_data_access tag_data_structure read_addr", false,-1, 5,0);
        vcdp->declBit(c+3665,"VX_cache genblk5[0] bank tag_data_access tag_data_structure read_valid", false,-1);
        vcdp->declBit(c+3673,"VX_cache genblk5[0] bank tag_data_access tag_data_structure read_dirty", false,-1);
        vcdp->declBus(c+3681,"VX_cache genblk5[0] bank tag_data_access tag_data_structure read_dirtyb", false,-1, 15,0);
        vcdp->declBus(c+3689,"VX_cache genblk5[0] bank tag_data_access tag_data_structure read_tag", false,-1, 19,0);
        vcdp->declArray(c+3697,"VX_cache genblk5[0] bank tag_data_access tag_data_structure read_data", false,-1, 127,0);
        vcdp->declBit(c+3785,"VX_cache genblk5[0] bank tag_data_access tag_data_structure invalidate", false,-1);
        vcdp->declBus(c+3745,"VX_cache genblk5[0] bank tag_data_access tag_data_structure write_enable", false,-1, 15,0);
        vcdp->declBit(c+3801,"VX_cache genblk5[0] bank tag_data_access tag_data_structure write_fill", false,-1);
        vcdp->declBus(c+3537,"VX_cache genblk5[0] bank tag_data_access tag_data_structure write_addr", false,-1, 5,0);
        vcdp->declBus(c+3809,"VX_cache genblk5[0] bank tag_data_access tag_data_structure tag_index", false,-1, 19,0);
        vcdp->declArray(c+3753,"VX_cache genblk5[0] bank tag_data_access tag_data_structure write_data", false,-1, 127,0);
        vcdp->declBit(c+2897,"VX_cache genblk5[0] bank tag_data_access tag_data_structure fill_sent", false,-1);
        vcdp->declQuad(c+12793,"VX_cache genblk5[0] bank tag_data_access tag_data_structure dirty", false,-1, 63,0);
        vcdp->declQuad(c+12809,"VX_cache genblk5[0] bank tag_data_access tag_data_structure valid", false,-1, 63,0);
        vcdp->declBit(c+3897,"VX_cache genblk5[0] bank tag_data_access tag_data_structure do_write", false,-1);
        vcdp->declBus(c+12825,"VX_cache genblk5[0] bank tag_data_access tag_data_structure i", false,-1, 31,0);
        vcdp->declBus(c+12833,"VX_cache genblk5[0] bank tag_data_access tag_data_structure j", false,-1, 31,0);
        vcdp->declBus(c+25201,"VX_cache genblk5[0] bank tag_data_access s0_1_c0 N", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[0] bank tag_data_access s0_1_c0 PASSTHRU", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[0] bank tag_data_access s0_1_c0 clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[0] bank tag_data_access s0_1_c0 reset", false,-1);
        vcdp->declBit(c+2585,"VX_cache genblk5[0] bank tag_data_access s0_1_c0 stall", false,-1);
        vcdp->declBit(c+25137,"VX_cache genblk5[0] bank tag_data_access s0_1_c0 flush", false,-1);
        vcdp->declArray(c+3905,"VX_cache genblk5[0] bank tag_data_access s0_1_c0 in", false,-1, 165,0);
        vcdp->declArray(c+3905,"VX_cache genblk5[0] bank tag_data_access s0_1_c0 out", false,-1, 165,0);
        vcdp->declArray(c+12841,"VX_cache genblk5[0] bank tag_data_access s0_1_c0 value", false,-1, 165,0);
        vcdp->declBus(c+25209,"VX_cache genblk5[0] bank st_1e_2 N", false,-1, 31,0);
        vcdp->declBus(c+25057,"VX_cache genblk5[0] bank st_1e_2 PASSTHRU", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[0] bank st_1e_2 clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[0] bank st_1e_2 reset", false,-1);
        vcdp->declBit(c+2585,"VX_cache genblk5[0] bank st_1e_2 stall", false,-1);
        vcdp->declBit(c+25137,"VX_cache genblk5[0] bank st_1e_2 flush", false,-1);
        vcdp->declArray(c+3953,"VX_cache genblk5[0] bank st_1e_2 in", false,-1, 315,0);
        vcdp->declArray(c+12889,"VX_cache genblk5[0] bank st_1e_2 out", false,-1, 315,0);
        vcdp->declArray(c+12889,"VX_cache genblk5[0] bank st_1e_2 value", false,-1, 315,0);
        vcdp->declBus(c+25057,"VX_cache genblk5[0] bank cache_miss_resrv CACHE_ID", false,-1, 31,0);
        vcdp->declBus(c+25153,"VX_cache genblk5[0] bank cache_miss_resrv BANK_ID", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[0] bank cache_miss_resrv BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank cache_miss_resrv NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank cache_miss_resrv WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank cache_miss_resrv NUM_REQUESTS", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[0] bank cache_miss_resrv MRVQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25097,"VX_cache genblk5[0] bank cache_miss_resrv CORE_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBus(c+25113,"VX_cache genblk5[0] bank cache_miss_resrv SNP_REQ_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[0] bank cache_miss_resrv clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[0] bank cache_miss_resrv reset", false,-1);
        vcdp->declBit(c+3081,"VX_cache genblk5[0] bank cache_miss_resrv miss_add", false,-1);
        vcdp->declBit(c+3089,"VX_cache genblk5[0] bank cache_miss_resrv is_mrvq", false,-1);
        vcdp->declBus(c+10089,"VX_cache genblk5[0] bank cache_miss_resrv miss_add_addr", false,-1, 25,0);
        vcdp->declBus(c+10801,"VX_cache genblk5[0] bank cache_miss_resrv miss_add_wsel", false,-1, 1,0);
        vcdp->declBus(c+10809,"VX_cache genblk5[0] bank cache_miss_resrv miss_add_data", false,-1, 31,0);
        vcdp->declBus(c+10745,"VX_cache genblk5[0] bank cache_miss_resrv miss_add_tid", false,-1, 1,0);
        vcdp->declQuad(c+10753,"VX_cache genblk5[0] bank cache_miss_resrv miss_add_tag", false,-1, 41,0);
        vcdp->declBit(c+10769,"VX_cache genblk5[0] bank cache_miss_resrv miss_add_rw", false,-1);
        vcdp->declBus(c+10777,"VX_cache genblk5[0] bank cache_miss_resrv miss_add_byteen", false,-1, 3,0);
        vcdp->declBit(c+3049,"VX_cache genblk5[0] bank cache_miss_resrv mrvq_init_ready_state", false,-1);
        vcdp->declBit(c+10913,"VX_cache genblk5[0] bank cache_miss_resrv miss_add_is_snp", false,-1);
        vcdp->declBit(c+10921,"VX_cache genblk5[0] bank cache_miss_resrv miss_add_snp_invalidate", false,-1);
        vcdp->declBit(c+10657,"VX_cache genblk5[0] bank cache_miss_resrv miss_resrv_full", false,-1);
        vcdp->declBit(c+10665,"VX_cache genblk5[0] bank cache_miss_resrv miss_resrv_stop", false,-1);
        vcdp->declBit(c+3545,"VX_cache genblk5[0] bank cache_miss_resrv is_fill_st1", false,-1);
        vcdp->declBus(c+3033,"VX_cache genblk5[0] bank cache_miss_resrv fill_addr_st1", false,-1, 25,0);
        vcdp->declBit(c+2521,"VX_cache genblk5[0] bank cache_miss_resrv pending_hazard", false,-1);
        vcdp->declBit(c+2497,"VX_cache genblk5[0] bank cache_miss_resrv miss_resrv_pop", false,-1);
        vcdp->declBit(c+2505,"VX_cache genblk5[0] bank cache_miss_resrv miss_resrv_valid_st0", false,-1);
        vcdp->declBus(c+10681,"VX_cache genblk5[0] bank cache_miss_resrv miss_resrv_addr_st0", false,-1, 25,0);
        vcdp->declBus(c+10689,"VX_cache genblk5[0] bank cache_miss_resrv miss_resrv_wsel_st0", false,-1, 1,0);
        vcdp->declBus(c+10697,"VX_cache genblk5[0] bank cache_miss_resrv miss_resrv_data_st0", false,-1, 31,0);
        vcdp->declBus(c+10673,"VX_cache genblk5[0] bank cache_miss_resrv miss_resrv_tid_st0", false,-1, 1,0);
        vcdp->declQuad(c+10705,"VX_cache genblk5[0] bank cache_miss_resrv miss_resrv_tag_st0", false,-1, 41,0);
        vcdp->declBit(c+2513,"VX_cache genblk5[0] bank cache_miss_resrv miss_resrv_rw_st0", false,-1);
        vcdp->declBus(c+10721,"VX_cache genblk5[0] bank cache_miss_resrv miss_resrv_byteen_st0", false,-1, 3,0);
        vcdp->declBit(c+10729,"VX_cache genblk5[0] bank cache_miss_resrv miss_resrv_is_snp_st0", false,-1);
        vcdp->declBit(c+10737,"VX_cache genblk5[0] bank cache_miss_resrv miss_resrv_snp_invalidate_st0", false,-1);
        {int i; for (i=0; i<16; i++) {
                vcdp->declArray(c+12969+i*3,"VX_cache genblk5[0] bank cache_miss_resrv metadata_table", true,(i+0), 84,0);}}
        vcdp->declArray(c+13353,"VX_cache genblk5[0] bank cache_miss_resrv addr_table", false,-1, 415,0);
        vcdp->declBus(c+13457,"VX_cache genblk5[0] bank cache_miss_resrv valid_table", false,-1, 15,0);
        vcdp->declBus(c+13465,"VX_cache genblk5[0] bank cache_miss_resrv ready_table", false,-1, 15,0);
        vcdp->declBus(c+13473,"VX_cache genblk5[0] bank cache_miss_resrv schedule_ptr", false,-1, 3,0);
        vcdp->declBus(c+13481,"VX_cache genblk5[0] bank cache_miss_resrv head_ptr", false,-1, 3,0);
        vcdp->declBus(c+13489,"VX_cache genblk5[0] bank cache_miss_resrv tail_ptr", false,-1, 3,0);
        vcdp->declBus(c+13497,"VX_cache genblk5[0] bank cache_miss_resrv size", false,-1, 4,0);
        vcdp->declBit(c+13505,"VX_cache genblk5[0] bank cache_miss_resrv enqueue_possible", false,-1);
        vcdp->declBus(c+13489,"VX_cache genblk5[0] bank cache_miss_resrv enqueue_index", false,-1, 3,0);
        vcdp->declBus(c+4033,"VX_cache genblk5[0] bank cache_miss_resrv make_ready", false,-1, 15,0);
        vcdp->declBus(c+4041,"VX_cache genblk5[0] bank cache_miss_resrv make_ready_push", false,-1, 15,0);
        vcdp->declBus(c+4049,"VX_cache genblk5[0] bank cache_miss_resrv valid_address_match", false,-1, 15,0);
        vcdp->declBit(c+2505,"VX_cache genblk5[0] bank cache_miss_resrv dequeue_possible", false,-1);
        vcdp->declBus(c+13473,"VX_cache genblk5[0] bank cache_miss_resrv dequeue_index", false,-1, 3,0);
        vcdp->declBit(c+4057,"VX_cache genblk5[0] bank cache_miss_resrv mrvq_push", false,-1);
        vcdp->declBit(c+4065,"VX_cache genblk5[0] bank cache_miss_resrv mrvq_pop", false,-1);
        vcdp->declBit(c+4073,"VX_cache genblk5[0] bank cache_miss_resrv recover_state", false,-1);
        vcdp->declBit(c+4081,"VX_cache genblk5[0] bank cache_miss_resrv increment_head", false,-1);
        vcdp->declBit(c+4089,"VX_cache genblk5[0] bank cache_miss_resrv update_ready", false,-1);
        vcdp->declBit(c+4097,"VX_cache genblk5[0] bank cache_miss_resrv qual_mrvq_init", false,-1);
        vcdp->declBus(c+25217,"VX_cache genblk5[0] bank cwb_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank cwb_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[0] bank cwb_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[0] bank cwb_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[0] bank cwb_queue reset", false,-1);
        vcdp->declBit(c+3097,"VX_cache genblk5[0] bank cwb_queue push", false,-1);
        vcdp->declBit(c+969,"VX_cache genblk5[0] bank cwb_queue pop", false,-1);
        vcdp->declArray(c+4105,"VX_cache genblk5[0] bank cwb_queue data_in", false,-1, 75,0);
        vcdp->declArray(c+4129,"VX_cache genblk5[0] bank cwb_queue data_out", false,-1, 75,0);
        vcdp->declBit(c+10961,"VX_cache genblk5[0] bank cwb_queue empty", false,-1);
        vcdp->declBit(c+10969,"VX_cache genblk5[0] bank cwb_queue full", false,-1);
        vcdp->declBus(c+13513,"VX_cache genblk5[0] bank cwb_queue size", false,-1, 2,0);
        vcdp->declBus(c+13513,"VX_cache genblk5[0] bank cwb_queue size_r", false,-1, 2,0);
        vcdp->declBit(c+361,"VX_cache genblk5[0] bank cwb_queue reading", false,-1);
        vcdp->declBit(c+4153,"VX_cache genblk5[0] bank cwb_queue writing", false,-1);
        {int i; for (i=0; i<4; i++) {
                vcdp->declArray(c+13521+i*3,"VX_cache genblk5[0] bank cwb_queue genblk3 data", true,(i+0), 75,0);}}
        vcdp->declArray(c+13617,"VX_cache genblk5[0] bank cwb_queue genblk3 genblk2 head_r", false,-1, 75,0);
        vcdp->declArray(c+13641,"VX_cache genblk5[0] bank cwb_queue genblk3 genblk2 curr_r", false,-1, 75,0);
        vcdp->declBus(c+13665,"VX_cache genblk5[0] bank cwb_queue genblk3 genblk2 wr_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+13673,"VX_cache genblk5[0] bank cwb_queue genblk3 genblk2 rd_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+13681,"VX_cache genblk5[0] bank cwb_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 1,0);
        vcdp->declBit(c+10961,"VX_cache genblk5[0] bank cwb_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+10969,"VX_cache genblk5[0] bank cwb_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+13689,"VX_cache genblk5[0] bank cwb_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25225,"VX_cache genblk5[0] bank dwb_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[0] bank dwb_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[0] bank dwb_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[0] bank dwb_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[0] bank dwb_queue reset", false,-1);
        vcdp->declBit(c+3121,"VX_cache genblk5[0] bank dwb_queue push", false,-1);
        vcdp->declBit(c+977,"VX_cache genblk5[0] bank dwb_queue pop", false,-1);
        vcdp->declArray(c+4161,"VX_cache genblk5[0] bank dwb_queue data_in", false,-1, 199,0);
        vcdp->declArray(c+4217,"VX_cache genblk5[0] bank dwb_queue data_out", false,-1, 199,0);
        vcdp->declBit(c+10977,"VX_cache genblk5[0] bank dwb_queue empty", false,-1);
        vcdp->declBit(c+10985,"VX_cache genblk5[0] bank dwb_queue full", false,-1);
        vcdp->declBus(c+13697,"VX_cache genblk5[0] bank dwb_queue size", false,-1, 2,0);
        vcdp->declBus(c+13697,"VX_cache genblk5[0] bank dwb_queue size_r", false,-1, 2,0);
        vcdp->declBit(c+369,"VX_cache genblk5[0] bank dwb_queue reading", false,-1);
        vcdp->declBit(c+4273,"VX_cache genblk5[0] bank dwb_queue writing", false,-1);
        {int i; for (i=0; i<4; i++) {
                vcdp->declArray(c+13705+i*7,"VX_cache genblk5[0] bank dwb_queue genblk3 data", true,(i+0), 199,0);}}
        vcdp->declArray(c+13929,"VX_cache genblk5[0] bank dwb_queue genblk3 genblk2 head_r", false,-1, 199,0);
        vcdp->declArray(c+13985,"VX_cache genblk5[0] bank dwb_queue genblk3 genblk2 curr_r", false,-1, 199,0);
        vcdp->declBus(c+14041,"VX_cache genblk5[0] bank dwb_queue genblk3 genblk2 wr_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+14049,"VX_cache genblk5[0] bank dwb_queue genblk3 genblk2 rd_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+14057,"VX_cache genblk5[0] bank dwb_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 1,0);
        vcdp->declBit(c+10977,"VX_cache genblk5[0] bank dwb_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+10985,"VX_cache genblk5[0] bank dwb_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+14065,"VX_cache genblk5[0] bank dwb_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25057,"VX_cache genblk5[1] bank CACHE_ID", false,-1, 31,0);
        vcdp->declBus(c+25161,"VX_cache genblk5[1] bank BANK_ID", false,-1, 31,0);
        vcdp->declBus(c+25065,"VX_cache genblk5[1] bank CACHE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[1] bank BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank NUM_REQUESTS", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[1] bank STAGE_1_CYCLES", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank CREQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[1] bank MRVQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[1] bank DFPQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[1] bank SNRQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank CWBQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank DWBQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank DFQQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[1] bank WRITE_ENABLE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[1] bank DRAM_ENABLE", false,-1, 31,0);
        vcdp->declBus(c+25057,"VX_cache genblk5[1] bank SNOOP_FORWARDING", false,-1, 31,0);
        vcdp->declBus(c+25097,"VX_cache genblk5[1] bank CORE_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBus(c+25105,"VX_cache genblk5[1] bank CORE_TAG_ID_BITS", false,-1, 31,0);
        vcdp->declBus(c+25113,"VX_cache genblk5[1] bank SNP_REQ_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[1] bank clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[1] bank reset", false,-1);
        vcdp->declBus(c+65,"VX_cache genblk5[1] bank core_req_valid", false,-1, 3,0);
        vcdp->declBus(c+24513,"VX_cache genblk5[1] bank core_req_rw", false,-1, 3,0);
        vcdp->declBus(c+24521,"VX_cache genblk5[1] bank core_req_byteen", false,-1, 15,0);
        vcdp->declArray(c+24529,"VX_cache genblk5[1] bank core_req_addr", false,-1, 119,0);
        vcdp->declArray(c+24561,"VX_cache genblk5[1] bank core_req_data", false,-1, 127,0);
        vcdp->declQuad(c+24593,"VX_cache genblk5[1] bank core_req_tag", false,-1, 41,0);
        vcdp->declBit(c+10145,"VX_cache genblk5[1] bank core_req_ready", false,-1);
        vcdp->declBit(c+10113,"VX_cache genblk5[1] bank core_rsp_valid", false,-1);
        vcdp->declBus(c+1585,"VX_cache genblk5[1] bank core_rsp_tid", false,-1, 1,0);
        vcdp->declBus(c+1593,"VX_cache genblk5[1] bank core_rsp_data", false,-1, 31,0);
        vcdp->declQuad(c+1601,"VX_cache genblk5[1] bank core_rsp_tag", false,-1, 41,0);
        vcdp->declBit(c+73,"VX_cache genblk5[1] bank core_rsp_ready", false,-1);
        vcdp->declBit(c+1617,"VX_cache genblk5[1] bank dram_fill_req_valid", false,-1);
        vcdp->declBus(c+10129,"VX_cache genblk5[1] bank dram_fill_req_addr", false,-1, 25,0);
        vcdp->declBit(c+10065,"VX_cache genblk5[1] bank dram_fill_req_ready", false,-1);
        vcdp->declBit(c+24993,"VX_cache genblk5[1] bank dram_fill_rsp_valid", false,-1);
        vcdp->declArray(c+24769,"VX_cache genblk5[1] bank dram_fill_rsp_data", false,-1, 127,0);
        vcdp->declBus(c+24969,"VX_cache genblk5[1] bank dram_fill_rsp_addr", false,-1, 25,0);
        vcdp->declBit(c+10121,"VX_cache genblk5[1] bank dram_fill_rsp_ready", false,-1);
        vcdp->declBit(c+1625,"VX_cache genblk5[1] bank dram_wb_req_valid", false,-1);
        vcdp->declBus(c+1633,"VX_cache genblk5[1] bank dram_wb_req_byteen", false,-1, 15,0);
        vcdp->declBus(c+1641,"VX_cache genblk5[1] bank dram_wb_req_addr", false,-1, 25,0);
        vcdp->declArray(c+1649,"VX_cache genblk5[1] bank dram_wb_req_data", false,-1, 127,0);
        vcdp->declBit(c+81,"VX_cache genblk5[1] bank dram_wb_req_ready", false,-1);
        vcdp->declBit(c+25001,"VX_cache genblk5[1] bank snp_req_valid", false,-1);
        vcdp->declBus(c+24985,"VX_cache genblk5[1] bank snp_req_addr", false,-1, 25,0);
        vcdp->declBit(c+24833,"VX_cache genblk5[1] bank snp_req_invalidate", false,-1);
        vcdp->declBus(c+24841,"VX_cache genblk5[1] bank snp_req_tag", false,-1, 27,0);
        vcdp->declBit(c+10137,"VX_cache genblk5[1] bank snp_req_ready", false,-1);
        vcdp->declBit(c+1681,"VX_cache genblk5[1] bank snp_rsp_valid", false,-1);
        vcdp->declBus(c+1689,"VX_cache genblk5[1] bank snp_rsp_tag", false,-1, 27,0);
        vcdp->declBit(c+89,"VX_cache genblk5[1] bank snp_rsp_ready", false,-1);
        vcdp->declBit(c+4281,"VX_cache genblk5[1] bank snrq_pop", false,-1);
        vcdp->declBit(c+14073,"VX_cache genblk5[1] bank snrq_empty", false,-1);
        vcdp->declBit(c+14081,"VX_cache genblk5[1] bank snrq_full", false,-1);
        vcdp->declBus(c+4289,"VX_cache genblk5[1] bank snrq_addr_st0", false,-1, 25,0);
        vcdp->declBit(c+4297,"VX_cache genblk5[1] bank snrq_invalidate_st0", false,-1);
        vcdp->declBus(c+4305,"VX_cache genblk5[1] bank snrq_tag_st0", false,-1, 27,0);
        vcdp->declBit(c+4313,"VX_cache genblk5[1] bank dfpq_pop", false,-1);
        vcdp->declBit(c+14089,"VX_cache genblk5[1] bank dfpq_empty", false,-1);
        vcdp->declBit(c+14097,"VX_cache genblk5[1] bank dfpq_full", false,-1);
        vcdp->declBus(c+4321,"VX_cache genblk5[1] bank dfpq_addr_st0", false,-1, 25,0);
        vcdp->declArray(c+4329,"VX_cache genblk5[1] bank dfpq_filldata_st0", false,-1, 127,0);
        vcdp->declBit(c+4361,"VX_cache genblk5[1] bank reqq_pop", false,-1);
        vcdp->declBit(c+993,"VX_cache genblk5[1] bank reqq_push", false,-1);
        vcdp->declBit(c+4369,"VX_cache genblk5[1] bank reqq_empty", false,-1);
        vcdp->declBit(c+14105,"VX_cache genblk5[1] bank reqq_full", false,-1);
        vcdp->declBit(c+4377,"VX_cache genblk5[1] bank reqq_req_st0", false,-1);
        vcdp->declBus(c+4385,"VX_cache genblk5[1] bank reqq_req_tid_st0", false,-1, 1,0);
        vcdp->declBit(c+4393,"VX_cache genblk5[1] bank reqq_req_rw_st0", false,-1);
        vcdp->declBus(c+4401,"VX_cache genblk5[1] bank reqq_req_byteen_st0", false,-1, 3,0);
        vcdp->declBus(c+4409,"VX_cache genblk5[1] bank reqq_req_addr_st0", false,-1, 29,0);
        vcdp->declBus(c+4417,"VX_cache genblk5[1] bank reqq_req_writeword_st0", false,-1, 31,0);
        vcdp->declQuad(c+14113,"VX_cache genblk5[1] bank reqq_req_tag_st0", false,-1, 41,0);
        vcdp->declBit(c+4425,"VX_cache genblk5[1] bank mrvq_pop", false,-1);
        vcdp->declBit(c+14129,"VX_cache genblk5[1] bank mrvq_full", false,-1);
        vcdp->declBit(c+14137,"VX_cache genblk5[1] bank mrvq_stop", false,-1);
        vcdp->declBit(c+4433,"VX_cache genblk5[1] bank mrvq_valid_st0", false,-1);
        vcdp->declBus(c+14145,"VX_cache genblk5[1] bank mrvq_tid_st0", false,-1, 1,0);
        vcdp->declBus(c+14153,"VX_cache genblk5[1] bank mrvq_addr_st0", false,-1, 25,0);
        vcdp->declBus(c+14161,"VX_cache genblk5[1] bank mrvq_wsel_st0", false,-1, 1,0);
        vcdp->declBus(c+14169,"VX_cache genblk5[1] bank mrvq_writeword_st0", false,-1, 31,0);
        vcdp->declQuad(c+14177,"VX_cache genblk5[1] bank mrvq_tag_st0", false,-1, 41,0);
        vcdp->declBit(c+4441,"VX_cache genblk5[1] bank mrvq_rw_st0", false,-1);
        vcdp->declBus(c+14193,"VX_cache genblk5[1] bank mrvq_byteen_st0", false,-1, 3,0);
        vcdp->declBit(c+14201,"VX_cache genblk5[1] bank mrvq_is_snp_st0", false,-1);
        vcdp->declBit(c+14209,"VX_cache genblk5[1] bank mrvq_snp_invalidate_st0", false,-1);
        vcdp->declBit(c+4449,"VX_cache genblk5[1] bank mrvq_pending_hazard_st1e", false,-1);
        vcdp->declBit(c+4457,"VX_cache genblk5[1] bank st2_pending_hazard_st1e", false,-1);
        vcdp->declBit(c+4465,"VX_cache genblk5[1] bank force_request_miss_st1e", false,-1);
        vcdp->declBus(c+14217,"VX_cache genblk5[1] bank miss_add_tid", false,-1, 1,0);
        vcdp->declQuad(c+14225,"VX_cache genblk5[1] bank miss_add_tag", false,-1, 41,0);
        vcdp->declBit(c+14241,"VX_cache genblk5[1] bank miss_add_rw", false,-1);
        vcdp->declBus(c+14249,"VX_cache genblk5[1] bank miss_add_byteen", false,-1, 3,0);
        vcdp->declBus(c+10129,"VX_cache genblk5[1] bank addr_st2", false,-1, 25,0);
        vcdp->declBit(c+14257,"VX_cache genblk5[1] bank is_fill_st2", false,-1);
        vcdp->declBit(c+4473,"VX_cache genblk5[1] bank recover_mrvq_state_st2", false,-1);
        vcdp->declBit(c+4481,"VX_cache genblk5[1] bank mrvq_push_stall", false,-1);
        vcdp->declBit(c+4489,"VX_cache genblk5[1] bank cwbq_push_stall", false,-1);
        vcdp->declBit(c+4497,"VX_cache genblk5[1] bank dwbq_push_stall", false,-1);
        vcdp->declBit(c+4505,"VX_cache genblk5[1] bank dram_fill_req_stall", false,-1);
        vcdp->declBit(c+4513,"VX_cache genblk5[1] bank stall_bank_pipe", false,-1);
        vcdp->declBit(c+4521,"VX_cache genblk5[1] bank is_fill_in_pipe", false,-1);
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+4529+i*1,"VX_cache genblk5[1] bank is_fill_st1", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+4537+i*1,"VX_cache genblk5[1] bank going_to_write_st1", true,(i+0));}}
        vcdp->declBus(c+25161,"VX_cache genblk5[1] bank j", false,-1, 31,0);
        vcdp->declBit(c+4433,"VX_cache genblk5[1] bank mrvq_pop_unqual", false,-1);
        vcdp->declBit(c+4545,"VX_cache genblk5[1] bank dfpq_pop_unqual", false,-1);
        vcdp->declBit(c+4553,"VX_cache genblk5[1] bank reqq_pop_unqual", false,-1);
        vcdp->declBit(c+4561,"VX_cache genblk5[1] bank snrq_pop_unqual", false,-1);
        vcdp->declBit(c+4545,"VX_cache genblk5[1] bank qual_is_fill_st0", false,-1);
        vcdp->declBit(c+4569,"VX_cache genblk5[1] bank qual_valid_st0", false,-1);
        vcdp->declBus(c+4577,"VX_cache genblk5[1] bank qual_addr_st0", false,-1, 25,0);
        vcdp->declBus(c+4585,"VX_cache genblk5[1] bank qual_wsel_st0", false,-1, 1,0);
        vcdp->declBit(c+4433,"VX_cache genblk5[1] bank qual_is_mrvq_st0", false,-1);
        vcdp->declBus(c+4593,"VX_cache genblk5[1] bank qual_writeword_st0", false,-1, 31,0);
        vcdp->declArray(c+4601,"VX_cache genblk5[1] bank qual_writedata_st0", false,-1, 127,0);
        vcdp->declQuad(c+4633,"VX_cache genblk5[1] bank qual_inst_meta_st0", false,-1, 48,0);
        vcdp->declBit(c+4649,"VX_cache genblk5[1] bank qual_going_to_write_st0", false,-1);
        vcdp->declBit(c+4657,"VX_cache genblk5[1] bank qual_is_snp_st0", false,-1);
        vcdp->declBit(c+4665,"VX_cache genblk5[1] bank qual_snp_invalidate_st0", false,-1);
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+4673+i*1,"VX_cache genblk5[1] bank valid_st1", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+4681+i*1,"VX_cache genblk5[1] bank addr_st1", true,(i+0), 25,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+4689+i*1,"VX_cache genblk5[1] bank wsel_st1", true,(i+0), 1,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+4697+i*1,"VX_cache genblk5[1] bank writeword_st1", true,(i+0), 31,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declQuad(c+4705+i*2,"VX_cache genblk5[1] bank inst_meta_st1", true,(i+0), 48,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declArray(c+4721+i*4,"VX_cache genblk5[1] bank writedata_st1", true,(i+0), 127,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+4753+i*1,"VX_cache genblk5[1] bank is_snp_st1", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+4761+i*1,"VX_cache genblk5[1] bank snp_invalidate_st1", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+4769+i*1,"VX_cache genblk5[1] bank is_mrvq_st1", true,(i+0));}}
        vcdp->declBus(c+4777,"VX_cache genblk5[1] bank readword_st1e", false,-1, 31,0);
        vcdp->declArray(c+4785,"VX_cache genblk5[1] bank readdata_st1e", false,-1, 127,0);
        vcdp->declBus(c+4817,"VX_cache genblk5[1] bank readtag_st1e", false,-1, 19,0);
        vcdp->declBit(c+4825,"VX_cache genblk5[1] bank miss_st1e", false,-1);
        vcdp->declBit(c+4833,"VX_cache genblk5[1] bank dirty_st1e", false,-1);
        vcdp->declBus(c+4841,"VX_cache genblk5[1] bank dirtyb_st1e", false,-1, 15,0);
        vcdp->declQuad(c+4849,"VX_cache genblk5[1] bank tag_st1e", false,-1, 41,0);
        vcdp->declBus(c+4865,"VX_cache genblk5[1] bank tid_st1e", false,-1, 1,0);
        vcdp->declBit(c+4873,"VX_cache genblk5[1] bank mem_rw_st1e", false,-1);
        vcdp->declBus(c+4881,"VX_cache genblk5[1] bank mem_byteen_st1e", false,-1, 3,0);
        vcdp->declBit(c+4889,"VX_cache genblk5[1] bank fill_saw_dirty_st1e", false,-1);
        vcdp->declBit(c+4897,"VX_cache genblk5[1] bank is_snp_st1e", false,-1);
        vcdp->declBit(c+4905,"VX_cache genblk5[1] bank snp_invalidate_st1e", false,-1);
        vcdp->declBit(c+4913,"VX_cache genblk5[1] bank snp_to_mrvq_st1e", false,-1);
        vcdp->declBit(c+4921,"VX_cache genblk5[1] bank mrvq_init_ready_state_st1e", false,-1);
        vcdp->declBit(c+4929,"VX_cache genblk5[1] bank miss_add_because_miss", false,-1);
        vcdp->declBit(c+4937,"VX_cache genblk5[1] bank valid_st1e", false,-1);
        vcdp->declBit(c+4945,"VX_cache genblk5[1] bank is_mrvq_st1e", false,-1);
        vcdp->declBit(c+4953,"VX_cache genblk5[1] bank mrvq_recover_ready_state_st1e", false,-1);
        vcdp->declBus(c+4961,"VX_cache genblk5[1] bank addr_st1e", false,-1, 25,0);
        vcdp->declBit(c+4969,"VX_cache genblk5[1] bank qual_valid_st1e_2", false,-1);
        vcdp->declBit(c+4945,"VX_cache genblk5[1] bank is_mrvq_st1e_st2", false,-1);
        vcdp->declBit(c+14265,"VX_cache genblk5[1] bank valid_st2", false,-1);
        vcdp->declBus(c+14273,"VX_cache genblk5[1] bank wsel_st2", false,-1, 1,0);
        vcdp->declBus(c+14281,"VX_cache genblk5[1] bank writeword_st2", false,-1, 31,0);
        vcdp->declBus(c+14289,"VX_cache genblk5[1] bank readword_st2", false,-1, 31,0);
        vcdp->declArray(c+14297,"VX_cache genblk5[1] bank readdata_st2", false,-1, 127,0);
        vcdp->declBit(c+14329,"VX_cache genblk5[1] bank miss_st2", false,-1);
        vcdp->declBit(c+14337,"VX_cache genblk5[1] bank dirty_st2", false,-1);
        vcdp->declBus(c+14345,"VX_cache genblk5[1] bank dirtyb_st2", false,-1, 15,0);
        vcdp->declQuad(c+14353,"VX_cache genblk5[1] bank inst_meta_st2", false,-1, 48,0);
        vcdp->declBus(c+14369,"VX_cache genblk5[1] bank readtag_st2", false,-1, 19,0);
        vcdp->declBit(c+14377,"VX_cache genblk5[1] bank fill_saw_dirty_st2", false,-1);
        vcdp->declBit(c+14385,"VX_cache genblk5[1] bank is_snp_st2", false,-1);
        vcdp->declBit(c+14393,"VX_cache genblk5[1] bank snp_invalidate_st2", false,-1);
        vcdp->declBit(c+14401,"VX_cache genblk5[1] bank snp_to_mrvq_st2", false,-1);
        vcdp->declBit(c+14409,"VX_cache genblk5[1] bank is_mrvq_st2", false,-1);
        vcdp->declBit(c+4977,"VX_cache genblk5[1] bank mrvq_init_ready_state_st2", false,-1);
        vcdp->declBit(c+14417,"VX_cache genblk5[1] bank mrvq_recover_ready_state_st2", false,-1);
        vcdp->declBit(c+14425,"VX_cache genblk5[1] bank mrvq_init_ready_state_unqual_st2", false,-1);
        vcdp->declBit(c+4985,"VX_cache genblk5[1] bank mrvq_init_ready_state_hazard_st0_st1", false,-1);
        vcdp->declBit(c+4993,"VX_cache genblk5[1] bank mrvq_init_ready_state_hazard_st1e_st1", false,-1);
        vcdp->declBit(c+14401,"VX_cache genblk5[1] bank miss_add_because_pending", false,-1);
        vcdp->declBit(c+5001,"VX_cache genblk5[1] bank miss_add_unqual", false,-1);
        vcdp->declBit(c+5009,"VX_cache genblk5[1] bank miss_add", false,-1);
        vcdp->declBus(c+10129,"VX_cache genblk5[1] bank miss_add_addr", false,-1, 25,0);
        vcdp->declBus(c+14273,"VX_cache genblk5[1] bank miss_add_wsel", false,-1, 1,0);
        vcdp->declBus(c+14281,"VX_cache genblk5[1] bank miss_add_data", false,-1, 31,0);
        vcdp->declBit(c+14385,"VX_cache genblk5[1] bank miss_add_is_snp", false,-1);
        vcdp->declBit(c+14393,"VX_cache genblk5[1] bank miss_add_snp_invalidate", false,-1);
        vcdp->declBit(c+5017,"VX_cache genblk5[1] bank miss_add_is_mrvq", false,-1);
        vcdp->declBit(c+5025,"VX_cache genblk5[1] bank cwbq_push", false,-1);
        vcdp->declBit(c+1001,"VX_cache genblk5[1] bank cwbq_pop", false,-1);
        vcdp->declBit(c+14433,"VX_cache genblk5[1] bank cwbq_empty", false,-1);
        vcdp->declBit(c+14441,"VX_cache genblk5[1] bank cwbq_full", false,-1);
        vcdp->declBit(c+5033,"VX_cache genblk5[1] bank cwbq_push_unqual", false,-1);
        vcdp->declBus(c+14289,"VX_cache genblk5[1] bank cwbq_data", false,-1, 31,0);
        vcdp->declBus(c+14217,"VX_cache genblk5[1] bank cwbq_tid", false,-1, 1,0);
        vcdp->declQuad(c+14225,"VX_cache genblk5[1] bank cwbq_tag", false,-1, 41,0);
        vcdp->declBit(c+5001,"VX_cache genblk5[1] bank dram_fill_req_fast", false,-1);
        vcdp->declBit(c+5041,"VX_cache genblk5[1] bank dram_fill_req_unqual", false,-1);
        vcdp->declBit(c+5049,"VX_cache genblk5[1] bank dwbq_push", false,-1);
        vcdp->declBit(c+1009,"VX_cache genblk5[1] bank dwbq_pop", false,-1);
        vcdp->declBit(c+14449,"VX_cache genblk5[1] bank dwbq_empty", false,-1);
        vcdp->declBit(c+14457,"VX_cache genblk5[1] bank dwbq_full", false,-1);
        vcdp->declBit(c+5057,"VX_cache genblk5[1] bank dwbq_is_dwb_in", false,-1);
        vcdp->declBit(c+5065,"VX_cache genblk5[1] bank dwbq_is_snp_in", false,-1);
        vcdp->declBit(c+5073,"VX_cache genblk5[1] bank dwbq_is_dwb_out", false,-1);
        vcdp->declBit(c+5081,"VX_cache genblk5[1] bank dwbq_is_snp_out", false,-1);
        vcdp->declBit(c+5089,"VX_cache genblk5[1] bank dwbq_push_unqual", false,-1);
        vcdp->declBus(c+14465,"VX_cache genblk5[1] bank dwbq_req_addr", false,-1, 25,0);
        vcdp->declBus(c+14473,"VX_cache genblk5[1] bank snrq_tag_st2", false,-1, 27,0);
        vcdp->declBit(c+377,"VX_cache genblk5[1] bank dram_wb_req_fire", false,-1);
        vcdp->declBit(c+385,"VX_cache genblk5[1] bank snp_rsp_fire", false,-1);
        vcdp->declBit(c+14481,"VX_cache genblk5[1] bank dwbq_dual_valid_sel", false,-1);
        vcdp->declBus(c+25169,"VX_cache genblk5[1] bank snp_req_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[1] bank snp_req_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[1] bank snp_req_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[1] bank snp_req_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[1] bank snp_req_queue reset", false,-1);
        vcdp->declBit(c+25001,"VX_cache genblk5[1] bank snp_req_queue push", false,-1);
        vcdp->declBit(c+4281,"VX_cache genblk5[1] bank snp_req_queue pop", false,-1);
        vcdp->declQuad(c+393,"VX_cache genblk5[1] bank snp_req_queue data_in", false,-1, 54,0);
        vcdp->declQuad(c+5097,"VX_cache genblk5[1] bank snp_req_queue data_out", false,-1, 54,0);
        vcdp->declBit(c+14073,"VX_cache genblk5[1] bank snp_req_queue empty", false,-1);
        vcdp->declBit(c+14081,"VX_cache genblk5[1] bank snp_req_queue full", false,-1);
        vcdp->declBus(c+14489,"VX_cache genblk5[1] bank snp_req_queue size", false,-1, 4,0);
        vcdp->declBus(c+14489,"VX_cache genblk5[1] bank snp_req_queue size_r", false,-1, 4,0);
        vcdp->declBit(c+5113,"VX_cache genblk5[1] bank snp_req_queue reading", false,-1);
        vcdp->declBit(c+409,"VX_cache genblk5[1] bank snp_req_queue writing", false,-1);
        {int i; for (i=0; i<16; i++) {
                vcdp->declQuad(c+14497+i*2,"VX_cache genblk5[1] bank snp_req_queue genblk3 data", true,(i+0), 54,0);}}
        vcdp->declQuad(c+14753,"VX_cache genblk5[1] bank snp_req_queue genblk3 genblk2 head_r", false,-1, 54,0);
        vcdp->declQuad(c+14769,"VX_cache genblk5[1] bank snp_req_queue genblk3 genblk2 curr_r", false,-1, 54,0);
        vcdp->declBus(c+14785,"VX_cache genblk5[1] bank snp_req_queue genblk3 genblk2 wr_ptr_r", false,-1, 3,0);
        vcdp->declBus(c+14793,"VX_cache genblk5[1] bank snp_req_queue genblk3 genblk2 rd_ptr_r", false,-1, 3,0);
        vcdp->declBus(c+14801,"VX_cache genblk5[1] bank snp_req_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 3,0);
        vcdp->declBit(c+14073,"VX_cache genblk5[1] bank snp_req_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+14081,"VX_cache genblk5[1] bank snp_req_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+14809,"VX_cache genblk5[1] bank snp_req_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25177,"VX_cache genblk5[1] bank dfp_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[1] bank dfp_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[1] bank dfp_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[1] bank dfp_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[1] bank dfp_queue reset", false,-1);
        vcdp->declBit(c+24993,"VX_cache genblk5[1] bank dfp_queue push", false,-1);
        vcdp->declBit(c+4313,"VX_cache genblk5[1] bank dfp_queue pop", false,-1);
        vcdp->declArray(c+417,"VX_cache genblk5[1] bank dfp_queue data_in", false,-1, 153,0);
        vcdp->declArray(c+5121,"VX_cache genblk5[1] bank dfp_queue data_out", false,-1, 153,0);
        vcdp->declBit(c+14089,"VX_cache genblk5[1] bank dfp_queue empty", false,-1);
        vcdp->declBit(c+14097,"VX_cache genblk5[1] bank dfp_queue full", false,-1);
        vcdp->declBus(c+14817,"VX_cache genblk5[1] bank dfp_queue size", false,-1, 4,0);
        vcdp->declBus(c+14817,"VX_cache genblk5[1] bank dfp_queue size_r", false,-1, 4,0);
        vcdp->declBit(c+5161,"VX_cache genblk5[1] bank dfp_queue reading", false,-1);
        vcdp->declBit(c+457,"VX_cache genblk5[1] bank dfp_queue writing", false,-1);
        {int i; for (i=0; i<16; i++) {
                vcdp->declArray(c+14825+i*5,"VX_cache genblk5[1] bank dfp_queue genblk3 data", true,(i+0), 153,0);}}
        vcdp->declArray(c+15465,"VX_cache genblk5[1] bank dfp_queue genblk3 genblk2 head_r", false,-1, 153,0);
        vcdp->declArray(c+15505,"VX_cache genblk5[1] bank dfp_queue genblk3 genblk2 curr_r", false,-1, 153,0);
        vcdp->declBus(c+15545,"VX_cache genblk5[1] bank dfp_queue genblk3 genblk2 wr_ptr_r", false,-1, 3,0);
        vcdp->declBus(c+15553,"VX_cache genblk5[1] bank dfp_queue genblk3 genblk2 rd_ptr_r", false,-1, 3,0);
        vcdp->declBus(c+15561,"VX_cache genblk5[1] bank dfp_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 3,0);
        vcdp->declBit(c+14089,"VX_cache genblk5[1] bank dfp_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+14097,"VX_cache genblk5[1] bank dfp_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+15569,"VX_cache genblk5[1] bank dfp_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank core_req_arb WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank core_req_arb NUM_REQUESTS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank core_req_arb CREQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25097,"VX_cache genblk5[1] bank core_req_arb CORE_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBus(c+25105,"VX_cache genblk5[1] bank core_req_arb CORE_TAG_ID_BITS", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[1] bank core_req_arb clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[1] bank core_req_arb reset", false,-1);
        vcdp->declBit(c+993,"VX_cache genblk5[1] bank core_req_arb reqq_push", false,-1);
        vcdp->declBus(c+65,"VX_cache genblk5[1] bank core_req_arb bank_valids", false,-1, 3,0);
        vcdp->declBus(c+24513,"VX_cache genblk5[1] bank core_req_arb bank_rw", false,-1, 3,0);
        vcdp->declBus(c+24521,"VX_cache genblk5[1] bank core_req_arb bank_byteen", false,-1, 15,0);
        vcdp->declArray(c+24561,"VX_cache genblk5[1] bank core_req_arb bank_writedata", false,-1, 127,0);
        vcdp->declArray(c+24529,"VX_cache genblk5[1] bank core_req_arb bank_addr", false,-1, 119,0);
        vcdp->declQuad(c+24593,"VX_cache genblk5[1] bank core_req_arb bank_tag", false,-1, 41,0);
        vcdp->declBit(c+4361,"VX_cache genblk5[1] bank core_req_arb reqq_pop", false,-1);
        vcdp->declBit(c+4377,"VX_cache genblk5[1] bank core_req_arb reqq_req_st0", false,-1);
        vcdp->declBus(c+4385,"VX_cache genblk5[1] bank core_req_arb reqq_req_tid_st0", false,-1, 1,0);
        vcdp->declBit(c+4393,"VX_cache genblk5[1] bank core_req_arb reqq_req_rw_st0", false,-1);
        vcdp->declBus(c+4401,"VX_cache genblk5[1] bank core_req_arb reqq_req_byteen_st0", false,-1, 3,0);
        vcdp->declBus(c+4409,"VX_cache genblk5[1] bank core_req_arb reqq_req_addr_st0", false,-1, 29,0);
        vcdp->declBus(c+4417,"VX_cache genblk5[1] bank core_req_arb reqq_req_writedata_st0", false,-1, 31,0);
        vcdp->declQuad(c+14113,"VX_cache genblk5[1] bank core_req_arb reqq_req_tag_st0", false,-1, 41,0);
        vcdp->declBit(c+4369,"VX_cache genblk5[1] bank core_req_arb reqq_empty", false,-1);
        vcdp->declBit(c+14105,"VX_cache genblk5[1] bank core_req_arb reqq_full", false,-1);
        vcdp->declBus(c+5169,"VX_cache genblk5[1] bank core_req_arb out_per_valids", false,-1, 3,0);
        vcdp->declBus(c+5177,"VX_cache genblk5[1] bank core_req_arb out_per_rw", false,-1, 3,0);
        vcdp->declBus(c+5185,"VX_cache genblk5[1] bank core_req_arb out_per_byteen", false,-1, 15,0);
        vcdp->declArray(c+5193,"VX_cache genblk5[1] bank core_req_arb out_per_addr", false,-1, 119,0);
        vcdp->declArray(c+5225,"VX_cache genblk5[1] bank core_req_arb out_per_writedata", false,-1, 127,0);
        vcdp->declQuad(c+5257,"VX_cache genblk5[1] bank core_req_arb out_per_tag", false,-1, 41,0);
        vcdp->declBus(c+15577,"VX_cache genblk5[1] bank core_req_arb use_per_valids", false,-1, 3,0);
        vcdp->declBus(c+15585,"VX_cache genblk5[1] bank core_req_arb use_per_rw", false,-1, 3,0);
        vcdp->declBus(c+15593,"VX_cache genblk5[1] bank core_req_arb use_per_byteen", false,-1, 15,0);
        vcdp->declArray(c+15601,"VX_cache genblk5[1] bank core_req_arb use_per_addr", false,-1, 119,0);
        vcdp->declArray(c+15633,"VX_cache genblk5[1] bank core_req_arb use_per_writedata", false,-1, 127,0);
        vcdp->declQuad(c+14113,"VX_cache genblk5[1] bank core_req_arb use_per_tag", false,-1, 41,0);
        vcdp->declBus(c+15577,"VX_cache genblk5[1] bank core_req_arb qual_valids", false,-1, 3,0);
        vcdp->declBus(c+15585,"VX_cache genblk5[1] bank core_req_arb qual_rw", false,-1, 3,0);
        vcdp->declBus(c+15593,"VX_cache genblk5[1] bank core_req_arb qual_byteen", false,-1, 15,0);
        vcdp->declArray(c+15601,"VX_cache genblk5[1] bank core_req_arb qual_addr", false,-1, 119,0);
        vcdp->declArray(c+15633,"VX_cache genblk5[1] bank core_req_arb qual_writedata", false,-1, 127,0);
        vcdp->declQuad(c+14113,"VX_cache genblk5[1] bank core_req_arb qual_tag", false,-1, 41,0);
        vcdp->declBit(c+15665,"VX_cache genblk5[1] bank core_req_arb o_empty", false,-1);
        vcdp->declBit(c+15673,"VX_cache genblk5[1] bank core_req_arb use_empty", false,-1);
        vcdp->declBit(c+5273,"VX_cache genblk5[1] bank core_req_arb out_empty", false,-1);
        vcdp->declBit(c+1017,"VX_cache genblk5[1] bank core_req_arb push_qual", false,-1);
        vcdp->declBit(c+5281,"VX_cache genblk5[1] bank core_req_arb pop_qual", false,-1);
        vcdp->declBus(c+5289,"VX_cache genblk5[1] bank core_req_arb real_out_per_valids", false,-1, 3,0);
        vcdp->declBus(c+4385,"VX_cache genblk5[1] bank core_req_arb qual_request_index", false,-1, 1,0);
        vcdp->declBit(c+4377,"VX_cache genblk5[1] bank core_req_arb qual_has_request", false,-1);
        vcdp->declBus(c+25185,"VX_cache genblk5[1] bank core_req_arb reqq_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank core_req_arb reqq_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[1] bank core_req_arb reqq_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[1] bank core_req_arb reqq_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[1] bank core_req_arb reqq_queue reset", false,-1);
        vcdp->declBit(c+1017,"VX_cache genblk5[1] bank core_req_arb reqq_queue push", false,-1);
        vcdp->declBit(c+5281,"VX_cache genblk5[1] bank core_req_arb reqq_queue pop", false,-1);
        vcdp->declArray(c+465,"VX_cache genblk5[1] bank core_req_arb reqq_queue data_in", false,-1, 313,0);
        vcdp->declArray(c+5297,"VX_cache genblk5[1] bank core_req_arb reqq_queue data_out", false,-1, 313,0);
        vcdp->declBit(c+15665,"VX_cache genblk5[1] bank core_req_arb reqq_queue empty", false,-1);
        vcdp->declBit(c+14105,"VX_cache genblk5[1] bank core_req_arb reqq_queue full", false,-1);
        vcdp->declBus(c+15681,"VX_cache genblk5[1] bank core_req_arb reqq_queue size", false,-1, 2,0);
        vcdp->declBus(c+15681,"VX_cache genblk5[1] bank core_req_arb reqq_queue size_r", false,-1, 2,0);
        vcdp->declBit(c+5377,"VX_cache genblk5[1] bank core_req_arb reqq_queue reading", false,-1);
        vcdp->declBit(c+545,"VX_cache genblk5[1] bank core_req_arb reqq_queue writing", false,-1);
        {int i; for (i=0; i<4; i++) {
                vcdp->declArray(c+15689+i*10,"VX_cache genblk5[1] bank core_req_arb reqq_queue genblk3 data", true,(i+0), 313,0);}}
        vcdp->declArray(c+16009,"VX_cache genblk5[1] bank core_req_arb reqq_queue genblk3 genblk2 head_r", false,-1, 313,0);
        vcdp->declArray(c+16089,"VX_cache genblk5[1] bank core_req_arb reqq_queue genblk3 genblk2 curr_r", false,-1, 313,0);
        vcdp->declBus(c+16169,"VX_cache genblk5[1] bank core_req_arb reqq_queue genblk3 genblk2 wr_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+16177,"VX_cache genblk5[1] bank core_req_arb reqq_queue genblk3 genblk2 rd_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+16185,"VX_cache genblk5[1] bank core_req_arb reqq_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 1,0);
        vcdp->declBit(c+15665,"VX_cache genblk5[1] bank core_req_arb reqq_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+14105,"VX_cache genblk5[1] bank core_req_arb reqq_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+16193,"VX_cache genblk5[1] bank core_req_arb reqq_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank core_req_arb sel_bank N", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[1] bank core_req_arb sel_bank clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[1] bank core_req_arb sel_bank reset", false,-1);
        vcdp->declBus(c+15577,"VX_cache genblk5[1] bank core_req_arb sel_bank requests", false,-1, 3,0);
        vcdp->declBus(c+4385,"VX_cache genblk5[1] bank core_req_arb sel_bank grant_index", false,-1, 1,0);
        vcdp->declBus(c+5385,"VX_cache genblk5[1] bank core_req_arb sel_bank grant_onehot", false,-1, 3,0);
        vcdp->declBit(c+4377,"VX_cache genblk5[1] bank core_req_arb sel_bank grant_valid", false,-1);
        vcdp->declBus(c+5385,"VX_cache genblk5[1] bank core_req_arb sel_bank genblk2 grant_onehot_r", false,-1, 3,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank core_req_arb sel_bank genblk2 priority_encoder N", false,-1, 31,0);
        vcdp->declBus(c+15577,"VX_cache genblk5[1] bank core_req_arb sel_bank genblk2 priority_encoder data_in", false,-1, 3,0);
        vcdp->declBus(c+4385,"VX_cache genblk5[1] bank core_req_arb sel_bank genblk2 priority_encoder data_out", false,-1, 1,0);
        vcdp->declBit(c+4377,"VX_cache genblk5[1] bank core_req_arb sel_bank genblk2 priority_encoder valid_out", false,-1);
        vcdp->declBus(c+5393,"VX_cache genblk5[1] bank core_req_arb sel_bank genblk2 priority_encoder i", false,-1, 31,0);
        vcdp->declBus(c+25193,"VX_cache genblk5[1] bank s0_1_c0 N", false,-1, 31,0);
        vcdp->declBus(c+25057,"VX_cache genblk5[1] bank s0_1_c0 PASSTHRU", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[1] bank s0_1_c0 clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[1] bank s0_1_c0 reset", false,-1);
        vcdp->declBit(c+4513,"VX_cache genblk5[1] bank s0_1_c0 stall", false,-1);
        vcdp->declBit(c+25137,"VX_cache genblk5[1] bank s0_1_c0 flush", false,-1);
        vcdp->declArray(c+5401,"VX_cache genblk5[1] bank s0_1_c0 in", false,-1, 242,0);
        vcdp->declArray(c+16201,"VX_cache genblk5[1] bank s0_1_c0 out", false,-1, 242,0);
        vcdp->declArray(c+16201,"VX_cache genblk5[1] bank s0_1_c0 value", false,-1, 242,0);
        vcdp->declBus(c+25065,"VX_cache genblk5[1] bank tag_data_access CACHE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[1] bank tag_data_access BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank tag_data_access NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank tag_data_access WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[1] bank tag_data_access STAGE_1_CYCLES", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[1] bank tag_data_access WRITE_ENABLE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[1] bank tag_data_access DRAM_ENABLE", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[1] bank tag_data_access clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[1] bank tag_data_access reset", false,-1);
        vcdp->declBit(c+4513,"VX_cache genblk5[1] bank tag_data_access stall", false,-1);
        vcdp->declBit(c+4897,"VX_cache genblk5[1] bank tag_data_access is_snp_st1e", false,-1);
        vcdp->declBit(c+4905,"VX_cache genblk5[1] bank tag_data_access snp_invalidate_st1e", false,-1);
        vcdp->declBit(c+4513,"VX_cache genblk5[1] bank tag_data_access stall_bank_pipe", false,-1);
        vcdp->declBit(c+4465,"VX_cache genblk5[1] bank tag_data_access force_request_miss_st1e", false,-1);
        vcdp->declBus(c+5465,"VX_cache genblk5[1] bank tag_data_access readaddr_st10", false,-1, 5,0);
        vcdp->declBus(c+4961,"VX_cache genblk5[1] bank tag_data_access writeaddr_st1e", false,-1, 25,0);
        vcdp->declBit(c+4937,"VX_cache genblk5[1] bank tag_data_access valid_req_st1e", false,-1);
        vcdp->declBit(c+5473,"VX_cache genblk5[1] bank tag_data_access writefill_st1e", false,-1);
        vcdp->declBus(c+5481,"VX_cache genblk5[1] bank tag_data_access writeword_st1e", false,-1, 31,0);
        vcdp->declArray(c+5489,"VX_cache genblk5[1] bank tag_data_access writedata_st1e", false,-1, 127,0);
        vcdp->declBit(c+4873,"VX_cache genblk5[1] bank tag_data_access mem_rw_st1e", false,-1);
        vcdp->declBus(c+4881,"VX_cache genblk5[1] bank tag_data_access mem_byteen_st1e", false,-1, 3,0);
        vcdp->declBus(c+5521,"VX_cache genblk5[1] bank tag_data_access wordsel_st1e", false,-1, 1,0);
        vcdp->declBus(c+4777,"VX_cache genblk5[1] bank tag_data_access readword_st1e", false,-1, 31,0);
        vcdp->declArray(c+4785,"VX_cache genblk5[1] bank tag_data_access readdata_st1e", false,-1, 127,0);
        vcdp->declBus(c+4817,"VX_cache genblk5[1] bank tag_data_access readtag_st1e", false,-1, 19,0);
        vcdp->declBit(c+4825,"VX_cache genblk5[1] bank tag_data_access miss_st1e", false,-1);
        vcdp->declBit(c+4833,"VX_cache genblk5[1] bank tag_data_access dirty_st1e", false,-1);
        vcdp->declBus(c+4841,"VX_cache genblk5[1] bank tag_data_access dirtyb_st1e", false,-1, 15,0);
        vcdp->declBit(c+4889,"VX_cache genblk5[1] bank tag_data_access fill_saw_dirty_st1e", false,-1);
        vcdp->declBit(c+4913,"VX_cache genblk5[1] bank tag_data_access snp_to_mrvq_st1e", false,-1);
        vcdp->declBit(c+4921,"VX_cache genblk5[1] bank tag_data_access mrvq_init_ready_state_st1e", false,-1);
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+5529+i*1,"VX_cache genblk5[1] bank tag_data_access read_valid_st1c", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+5537+i*1,"VX_cache genblk5[1] bank tag_data_access read_dirty_st1c", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+5545+i*1,"VX_cache genblk5[1] bank tag_data_access read_dirtyb_st1c", true,(i+0), 15,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+5553+i*1,"VX_cache genblk5[1] bank tag_data_access read_tag_st1c", true,(i+0), 19,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declArray(c+5561+i*4,"VX_cache genblk5[1] bank tag_data_access read_data_st1c", true,(i+0), 127,0);}}
        vcdp->declBit(c+5593,"VX_cache genblk5[1] bank tag_data_access qual_read_valid_st1", false,-1);
        vcdp->declBit(c+5601,"VX_cache genblk5[1] bank tag_data_access qual_read_dirty_st1", false,-1);
        vcdp->declBus(c+5609,"VX_cache genblk5[1] bank tag_data_access qual_read_dirtyb_st1", false,-1, 15,0);
        vcdp->declBus(c+5617,"VX_cache genblk5[1] bank tag_data_access qual_read_tag_st1", false,-1, 19,0);
        vcdp->declArray(c+5625,"VX_cache genblk5[1] bank tag_data_access qual_read_data_st1", false,-1, 127,0);
        vcdp->declBit(c+5657,"VX_cache genblk5[1] bank tag_data_access use_read_valid_st1e", false,-1);
        vcdp->declBit(c+5665,"VX_cache genblk5[1] bank tag_data_access use_read_dirty_st1e", false,-1);
        vcdp->declBus(c+4841,"VX_cache genblk5[1] bank tag_data_access use_read_dirtyb_st1e", false,-1, 15,0);
        vcdp->declBus(c+4817,"VX_cache genblk5[1] bank tag_data_access use_read_tag_st1e", false,-1, 19,0);
        vcdp->declArray(c+4785,"VX_cache genblk5[1] bank tag_data_access use_read_data_st1e", false,-1, 127,0);
        vcdp->declBus(c+5673,"VX_cache genblk5[1] bank tag_data_access use_write_enable", false,-1, 15,0);
        vcdp->declArray(c+5681,"VX_cache genblk5[1] bank tag_data_access use_write_data", false,-1, 127,0);
        vcdp->declBit(c+4825,"VX_cache genblk5[1] bank tag_data_access fill_sent", false,-1);
        vcdp->declBit(c+5713,"VX_cache genblk5[1] bank tag_data_access invalidate_line", false,-1);
        vcdp->declBit(c+5721,"VX_cache genblk5[1] bank tag_data_access tags_match", false,-1);
        vcdp->declBit(c+5729,"VX_cache genblk5[1] bank tag_data_access real_writefill", false,-1);
        vcdp->declBus(c+5737,"VX_cache genblk5[1] bank tag_data_access writetag_st1e", false,-1, 19,0);
        vcdp->declBus(c+5465,"VX_cache genblk5[1] bank tag_data_access writeladdr_st1e", false,-1, 5,0);
        vcdp->declBus(c+5745,"VX_cache genblk5[1] bank tag_data_access we", false,-1, 15,0);
        vcdp->declArray(c+5681,"VX_cache genblk5[1] bank tag_data_access data_write", false,-1, 127,0);
        vcdp->declBit(c+5753,"VX_cache genblk5[1] bank tag_data_access should_write", false,-1);
        vcdp->declBit(c+5713,"VX_cache genblk5[1] bank tag_data_access snoop_hit_no_pending", false,-1);
        vcdp->declBit(c+5761,"VX_cache genblk5[1] bank tag_data_access req_invalid", false,-1);
        vcdp->declBit(c+5769,"VX_cache genblk5[1] bank tag_data_access req_miss", false,-1);
        vcdp->declBit(c+5777,"VX_cache genblk5[1] bank tag_data_access real_miss", false,-1);
        vcdp->declBit(c+5785,"VX_cache genblk5[1] bank tag_data_access force_core_miss", false,-1);
        vcdp->declBit(c+5793,"VX_cache genblk5[1] bank tag_data_access genblk4[0] normal_write", false,-1);
        vcdp->declBit(c+5801,"VX_cache genblk5[1] bank tag_data_access genblk4[1] normal_write", false,-1);
        vcdp->declBit(c+5809,"VX_cache genblk5[1] bank tag_data_access genblk4[2] normal_write", false,-1);
        vcdp->declBit(c+5817,"VX_cache genblk5[1] bank tag_data_access genblk4[3] normal_write", false,-1);
        vcdp->declBus(c+25065,"VX_cache genblk5[1] bank tag_data_access tag_data_structure CACHE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[1] bank tag_data_access tag_data_structure BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank tag_data_access tag_data_structure NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank tag_data_access tag_data_structure WORD_SIZE", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[1] bank tag_data_access tag_data_structure clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[1] bank tag_data_access tag_data_structure reset", false,-1);
        vcdp->declBit(c+4513,"VX_cache genblk5[1] bank tag_data_access tag_data_structure stall_bank_pipe", false,-1);
        vcdp->declBus(c+5465,"VX_cache genblk5[1] bank tag_data_access tag_data_structure read_addr", false,-1, 5,0);
        vcdp->declBit(c+5593,"VX_cache genblk5[1] bank tag_data_access tag_data_structure read_valid", false,-1);
        vcdp->declBit(c+5601,"VX_cache genblk5[1] bank tag_data_access tag_data_structure read_dirty", false,-1);
        vcdp->declBus(c+5609,"VX_cache genblk5[1] bank tag_data_access tag_data_structure read_dirtyb", false,-1, 15,0);
        vcdp->declBus(c+5617,"VX_cache genblk5[1] bank tag_data_access tag_data_structure read_tag", false,-1, 19,0);
        vcdp->declArray(c+5625,"VX_cache genblk5[1] bank tag_data_access tag_data_structure read_data", false,-1, 127,0);
        vcdp->declBit(c+5713,"VX_cache genblk5[1] bank tag_data_access tag_data_structure invalidate", false,-1);
        vcdp->declBus(c+5673,"VX_cache genblk5[1] bank tag_data_access tag_data_structure write_enable", false,-1, 15,0);
        vcdp->declBit(c+5729,"VX_cache genblk5[1] bank tag_data_access tag_data_structure write_fill", false,-1);
        vcdp->declBus(c+5465,"VX_cache genblk5[1] bank tag_data_access tag_data_structure write_addr", false,-1, 5,0);
        vcdp->declBus(c+5737,"VX_cache genblk5[1] bank tag_data_access tag_data_structure tag_index", false,-1, 19,0);
        vcdp->declArray(c+5681,"VX_cache genblk5[1] bank tag_data_access tag_data_structure write_data", false,-1, 127,0);
        vcdp->declBit(c+4825,"VX_cache genblk5[1] bank tag_data_access tag_data_structure fill_sent", false,-1);
        vcdp->declQuad(c+16265,"VX_cache genblk5[1] bank tag_data_access tag_data_structure dirty", false,-1, 63,0);
        vcdp->declQuad(c+16281,"VX_cache genblk5[1] bank tag_data_access tag_data_structure valid", false,-1, 63,0);
        vcdp->declBit(c+5825,"VX_cache genblk5[1] bank tag_data_access tag_data_structure do_write", false,-1);
        vcdp->declBus(c+16297,"VX_cache genblk5[1] bank tag_data_access tag_data_structure i", false,-1, 31,0);
        vcdp->declBus(c+16305,"VX_cache genblk5[1] bank tag_data_access tag_data_structure j", false,-1, 31,0);
        vcdp->declBus(c+25201,"VX_cache genblk5[1] bank tag_data_access s0_1_c0 N", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[1] bank tag_data_access s0_1_c0 PASSTHRU", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[1] bank tag_data_access s0_1_c0 clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[1] bank tag_data_access s0_1_c0 reset", false,-1);
        vcdp->declBit(c+4513,"VX_cache genblk5[1] bank tag_data_access s0_1_c0 stall", false,-1);
        vcdp->declBit(c+25137,"VX_cache genblk5[1] bank tag_data_access s0_1_c0 flush", false,-1);
        vcdp->declArray(c+5833,"VX_cache genblk5[1] bank tag_data_access s0_1_c0 in", false,-1, 165,0);
        vcdp->declArray(c+5833,"VX_cache genblk5[1] bank tag_data_access s0_1_c0 out", false,-1, 165,0);
        vcdp->declArray(c+16313,"VX_cache genblk5[1] bank tag_data_access s0_1_c0 value", false,-1, 165,0);
        vcdp->declBus(c+25209,"VX_cache genblk5[1] bank st_1e_2 N", false,-1, 31,0);
        vcdp->declBus(c+25057,"VX_cache genblk5[1] bank st_1e_2 PASSTHRU", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[1] bank st_1e_2 clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[1] bank st_1e_2 reset", false,-1);
        vcdp->declBit(c+4513,"VX_cache genblk5[1] bank st_1e_2 stall", false,-1);
        vcdp->declBit(c+25137,"VX_cache genblk5[1] bank st_1e_2 flush", false,-1);
        vcdp->declArray(c+5881,"VX_cache genblk5[1] bank st_1e_2 in", false,-1, 315,0);
        vcdp->declArray(c+16361,"VX_cache genblk5[1] bank st_1e_2 out", false,-1, 315,0);
        vcdp->declArray(c+16361,"VX_cache genblk5[1] bank st_1e_2 value", false,-1, 315,0);
        vcdp->declBus(c+25057,"VX_cache genblk5[1] bank cache_miss_resrv CACHE_ID", false,-1, 31,0);
        vcdp->declBus(c+25161,"VX_cache genblk5[1] bank cache_miss_resrv BANK_ID", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[1] bank cache_miss_resrv BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank cache_miss_resrv NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank cache_miss_resrv WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank cache_miss_resrv NUM_REQUESTS", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[1] bank cache_miss_resrv MRVQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25097,"VX_cache genblk5[1] bank cache_miss_resrv CORE_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBus(c+25113,"VX_cache genblk5[1] bank cache_miss_resrv SNP_REQ_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[1] bank cache_miss_resrv clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[1] bank cache_miss_resrv reset", false,-1);
        vcdp->declBit(c+5009,"VX_cache genblk5[1] bank cache_miss_resrv miss_add", false,-1);
        vcdp->declBit(c+5017,"VX_cache genblk5[1] bank cache_miss_resrv is_mrvq", false,-1);
        vcdp->declBus(c+10129,"VX_cache genblk5[1] bank cache_miss_resrv miss_add_addr", false,-1, 25,0);
        vcdp->declBus(c+14273,"VX_cache genblk5[1] bank cache_miss_resrv miss_add_wsel", false,-1, 1,0);
        vcdp->declBus(c+14281,"VX_cache genblk5[1] bank cache_miss_resrv miss_add_data", false,-1, 31,0);
        vcdp->declBus(c+14217,"VX_cache genblk5[1] bank cache_miss_resrv miss_add_tid", false,-1, 1,0);
        vcdp->declQuad(c+14225,"VX_cache genblk5[1] bank cache_miss_resrv miss_add_tag", false,-1, 41,0);
        vcdp->declBit(c+14241,"VX_cache genblk5[1] bank cache_miss_resrv miss_add_rw", false,-1);
        vcdp->declBus(c+14249,"VX_cache genblk5[1] bank cache_miss_resrv miss_add_byteen", false,-1, 3,0);
        vcdp->declBit(c+4977,"VX_cache genblk5[1] bank cache_miss_resrv mrvq_init_ready_state", false,-1);
        vcdp->declBit(c+14385,"VX_cache genblk5[1] bank cache_miss_resrv miss_add_is_snp", false,-1);
        vcdp->declBit(c+14393,"VX_cache genblk5[1] bank cache_miss_resrv miss_add_snp_invalidate", false,-1);
        vcdp->declBit(c+14129,"VX_cache genblk5[1] bank cache_miss_resrv miss_resrv_full", false,-1);
        vcdp->declBit(c+14137,"VX_cache genblk5[1] bank cache_miss_resrv miss_resrv_stop", false,-1);
        vcdp->declBit(c+5473,"VX_cache genblk5[1] bank cache_miss_resrv is_fill_st1", false,-1);
        vcdp->declBus(c+4961,"VX_cache genblk5[1] bank cache_miss_resrv fill_addr_st1", false,-1, 25,0);
        vcdp->declBit(c+4449,"VX_cache genblk5[1] bank cache_miss_resrv pending_hazard", false,-1);
        vcdp->declBit(c+4425,"VX_cache genblk5[1] bank cache_miss_resrv miss_resrv_pop", false,-1);
        vcdp->declBit(c+4433,"VX_cache genblk5[1] bank cache_miss_resrv miss_resrv_valid_st0", false,-1);
        vcdp->declBus(c+14153,"VX_cache genblk5[1] bank cache_miss_resrv miss_resrv_addr_st0", false,-1, 25,0);
        vcdp->declBus(c+14161,"VX_cache genblk5[1] bank cache_miss_resrv miss_resrv_wsel_st0", false,-1, 1,0);
        vcdp->declBus(c+14169,"VX_cache genblk5[1] bank cache_miss_resrv miss_resrv_data_st0", false,-1, 31,0);
        vcdp->declBus(c+14145,"VX_cache genblk5[1] bank cache_miss_resrv miss_resrv_tid_st0", false,-1, 1,0);
        vcdp->declQuad(c+14177,"VX_cache genblk5[1] bank cache_miss_resrv miss_resrv_tag_st0", false,-1, 41,0);
        vcdp->declBit(c+4441,"VX_cache genblk5[1] bank cache_miss_resrv miss_resrv_rw_st0", false,-1);
        vcdp->declBus(c+14193,"VX_cache genblk5[1] bank cache_miss_resrv miss_resrv_byteen_st0", false,-1, 3,0);
        vcdp->declBit(c+14201,"VX_cache genblk5[1] bank cache_miss_resrv miss_resrv_is_snp_st0", false,-1);
        vcdp->declBit(c+14209,"VX_cache genblk5[1] bank cache_miss_resrv miss_resrv_snp_invalidate_st0", false,-1);
        {int i; for (i=0; i<16; i++) {
                vcdp->declArray(c+16441+i*3,"VX_cache genblk5[1] bank cache_miss_resrv metadata_table", true,(i+0), 84,0);}}
        vcdp->declArray(c+16825,"VX_cache genblk5[1] bank cache_miss_resrv addr_table", false,-1, 415,0);
        vcdp->declBus(c+16929,"VX_cache genblk5[1] bank cache_miss_resrv valid_table", false,-1, 15,0);
        vcdp->declBus(c+16937,"VX_cache genblk5[1] bank cache_miss_resrv ready_table", false,-1, 15,0);
        vcdp->declBus(c+16945,"VX_cache genblk5[1] bank cache_miss_resrv schedule_ptr", false,-1, 3,0);
        vcdp->declBus(c+16953,"VX_cache genblk5[1] bank cache_miss_resrv head_ptr", false,-1, 3,0);
        vcdp->declBus(c+16961,"VX_cache genblk5[1] bank cache_miss_resrv tail_ptr", false,-1, 3,0);
        vcdp->declBus(c+16969,"VX_cache genblk5[1] bank cache_miss_resrv size", false,-1, 4,0);
        vcdp->declBit(c+16977,"VX_cache genblk5[1] bank cache_miss_resrv enqueue_possible", false,-1);
        vcdp->declBus(c+16961,"VX_cache genblk5[1] bank cache_miss_resrv enqueue_index", false,-1, 3,0);
        vcdp->declBus(c+5961,"VX_cache genblk5[1] bank cache_miss_resrv make_ready", false,-1, 15,0);
        vcdp->declBus(c+5969,"VX_cache genblk5[1] bank cache_miss_resrv make_ready_push", false,-1, 15,0);
        vcdp->declBus(c+5977,"VX_cache genblk5[1] bank cache_miss_resrv valid_address_match", false,-1, 15,0);
        vcdp->declBit(c+4433,"VX_cache genblk5[1] bank cache_miss_resrv dequeue_possible", false,-1);
        vcdp->declBus(c+16945,"VX_cache genblk5[1] bank cache_miss_resrv dequeue_index", false,-1, 3,0);
        vcdp->declBit(c+5985,"VX_cache genblk5[1] bank cache_miss_resrv mrvq_push", false,-1);
        vcdp->declBit(c+5993,"VX_cache genblk5[1] bank cache_miss_resrv mrvq_pop", false,-1);
        vcdp->declBit(c+6001,"VX_cache genblk5[1] bank cache_miss_resrv recover_state", false,-1);
        vcdp->declBit(c+6009,"VX_cache genblk5[1] bank cache_miss_resrv increment_head", false,-1);
        vcdp->declBit(c+6017,"VX_cache genblk5[1] bank cache_miss_resrv update_ready", false,-1);
        vcdp->declBit(c+6025,"VX_cache genblk5[1] bank cache_miss_resrv qual_mrvq_init", false,-1);
        vcdp->declBus(c+25217,"VX_cache genblk5[1] bank cwb_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank cwb_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[1] bank cwb_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[1] bank cwb_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[1] bank cwb_queue reset", false,-1);
        vcdp->declBit(c+5025,"VX_cache genblk5[1] bank cwb_queue push", false,-1);
        vcdp->declBit(c+1001,"VX_cache genblk5[1] bank cwb_queue pop", false,-1);
        vcdp->declArray(c+6033,"VX_cache genblk5[1] bank cwb_queue data_in", false,-1, 75,0);
        vcdp->declArray(c+6057,"VX_cache genblk5[1] bank cwb_queue data_out", false,-1, 75,0);
        vcdp->declBit(c+14433,"VX_cache genblk5[1] bank cwb_queue empty", false,-1);
        vcdp->declBit(c+14441,"VX_cache genblk5[1] bank cwb_queue full", false,-1);
        vcdp->declBus(c+16985,"VX_cache genblk5[1] bank cwb_queue size", false,-1, 2,0);
        vcdp->declBus(c+16985,"VX_cache genblk5[1] bank cwb_queue size_r", false,-1, 2,0);
        vcdp->declBit(c+553,"VX_cache genblk5[1] bank cwb_queue reading", false,-1);
        vcdp->declBit(c+6081,"VX_cache genblk5[1] bank cwb_queue writing", false,-1);
        {int i; for (i=0; i<4; i++) {
                vcdp->declArray(c+16993+i*3,"VX_cache genblk5[1] bank cwb_queue genblk3 data", true,(i+0), 75,0);}}
        vcdp->declArray(c+17089,"VX_cache genblk5[1] bank cwb_queue genblk3 genblk2 head_r", false,-1, 75,0);
        vcdp->declArray(c+17113,"VX_cache genblk5[1] bank cwb_queue genblk3 genblk2 curr_r", false,-1, 75,0);
        vcdp->declBus(c+17137,"VX_cache genblk5[1] bank cwb_queue genblk3 genblk2 wr_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+17145,"VX_cache genblk5[1] bank cwb_queue genblk3 genblk2 rd_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+17153,"VX_cache genblk5[1] bank cwb_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 1,0);
        vcdp->declBit(c+14433,"VX_cache genblk5[1] bank cwb_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+14441,"VX_cache genblk5[1] bank cwb_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+17161,"VX_cache genblk5[1] bank cwb_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25225,"VX_cache genblk5[1] bank dwb_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[1] bank dwb_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[1] bank dwb_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[1] bank dwb_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[1] bank dwb_queue reset", false,-1);
        vcdp->declBit(c+5049,"VX_cache genblk5[1] bank dwb_queue push", false,-1);
        vcdp->declBit(c+1009,"VX_cache genblk5[1] bank dwb_queue pop", false,-1);
        vcdp->declArray(c+6089,"VX_cache genblk5[1] bank dwb_queue data_in", false,-1, 199,0);
        vcdp->declArray(c+6145,"VX_cache genblk5[1] bank dwb_queue data_out", false,-1, 199,0);
        vcdp->declBit(c+14449,"VX_cache genblk5[1] bank dwb_queue empty", false,-1);
        vcdp->declBit(c+14457,"VX_cache genblk5[1] bank dwb_queue full", false,-1);
        vcdp->declBus(c+17169,"VX_cache genblk5[1] bank dwb_queue size", false,-1, 2,0);
        vcdp->declBus(c+17169,"VX_cache genblk5[1] bank dwb_queue size_r", false,-1, 2,0);
        vcdp->declBit(c+561,"VX_cache genblk5[1] bank dwb_queue reading", false,-1);
        vcdp->declBit(c+6201,"VX_cache genblk5[1] bank dwb_queue writing", false,-1);
        {int i; for (i=0; i<4; i++) {
                vcdp->declArray(c+17177+i*7,"VX_cache genblk5[1] bank dwb_queue genblk3 data", true,(i+0), 199,0);}}
        vcdp->declArray(c+17401,"VX_cache genblk5[1] bank dwb_queue genblk3 genblk2 head_r", false,-1, 199,0);
        vcdp->declArray(c+17457,"VX_cache genblk5[1] bank dwb_queue genblk3 genblk2 curr_r", false,-1, 199,0);
        vcdp->declBus(c+17513,"VX_cache genblk5[1] bank dwb_queue genblk3 genblk2 wr_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+17521,"VX_cache genblk5[1] bank dwb_queue genblk3 genblk2 rd_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+17529,"VX_cache genblk5[1] bank dwb_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 1,0);
        vcdp->declBit(c+14449,"VX_cache genblk5[1] bank dwb_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+14457,"VX_cache genblk5[1] bank dwb_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+17537,"VX_cache genblk5[1] bank dwb_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25057,"VX_cache genblk5[2] bank CACHE_ID", false,-1, 31,0);
        vcdp->declBus(c+25233,"VX_cache genblk5[2] bank BANK_ID", false,-1, 31,0);
        vcdp->declBus(c+25065,"VX_cache genblk5[2] bank CACHE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[2] bank BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank NUM_REQUESTS", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[2] bank STAGE_1_CYCLES", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank CREQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[2] bank MRVQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[2] bank DFPQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[2] bank SNRQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank CWBQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank DWBQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank DFQQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[2] bank WRITE_ENABLE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[2] bank DRAM_ENABLE", false,-1, 31,0);
        vcdp->declBus(c+25057,"VX_cache genblk5[2] bank SNOOP_FORWARDING", false,-1, 31,0);
        vcdp->declBus(c+25097,"VX_cache genblk5[2] bank CORE_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBus(c+25105,"VX_cache genblk5[2] bank CORE_TAG_ID_BITS", false,-1, 31,0);
        vcdp->declBus(c+25113,"VX_cache genblk5[2] bank SNP_REQ_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[2] bank clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[2] bank reset", false,-1);
        vcdp->declBus(c+97,"VX_cache genblk5[2] bank core_req_valid", false,-1, 3,0);
        vcdp->declBus(c+24513,"VX_cache genblk5[2] bank core_req_rw", false,-1, 3,0);
        vcdp->declBus(c+24521,"VX_cache genblk5[2] bank core_req_byteen", false,-1, 15,0);
        vcdp->declArray(c+24529,"VX_cache genblk5[2] bank core_req_addr", false,-1, 119,0);
        vcdp->declArray(c+24561,"VX_cache genblk5[2] bank core_req_data", false,-1, 127,0);
        vcdp->declQuad(c+24593,"VX_cache genblk5[2] bank core_req_tag", false,-1, 41,0);
        vcdp->declBit(c+10185,"VX_cache genblk5[2] bank core_req_ready", false,-1);
        vcdp->declBit(c+10153,"VX_cache genblk5[2] bank core_rsp_valid", false,-1);
        vcdp->declBus(c+1697,"VX_cache genblk5[2] bank core_rsp_tid", false,-1, 1,0);
        vcdp->declBus(c+1705,"VX_cache genblk5[2] bank core_rsp_data", false,-1, 31,0);
        vcdp->declQuad(c+1713,"VX_cache genblk5[2] bank core_rsp_tag", false,-1, 41,0);
        vcdp->declBit(c+105,"VX_cache genblk5[2] bank core_rsp_ready", false,-1);
        vcdp->declBit(c+1729,"VX_cache genblk5[2] bank dram_fill_req_valid", false,-1);
        vcdp->declBus(c+10169,"VX_cache genblk5[2] bank dram_fill_req_addr", false,-1, 25,0);
        vcdp->declBit(c+10065,"VX_cache genblk5[2] bank dram_fill_req_ready", false,-1);
        vcdp->declBit(c+25009,"VX_cache genblk5[2] bank dram_fill_rsp_valid", false,-1);
        vcdp->declArray(c+24769,"VX_cache genblk5[2] bank dram_fill_rsp_data", false,-1, 127,0);
        vcdp->declBus(c+24969,"VX_cache genblk5[2] bank dram_fill_rsp_addr", false,-1, 25,0);
        vcdp->declBit(c+10161,"VX_cache genblk5[2] bank dram_fill_rsp_ready", false,-1);
        vcdp->declBit(c+1737,"VX_cache genblk5[2] bank dram_wb_req_valid", false,-1);
        vcdp->declBus(c+1745,"VX_cache genblk5[2] bank dram_wb_req_byteen", false,-1, 15,0);
        vcdp->declBus(c+1753,"VX_cache genblk5[2] bank dram_wb_req_addr", false,-1, 25,0);
        vcdp->declArray(c+1761,"VX_cache genblk5[2] bank dram_wb_req_data", false,-1, 127,0);
        vcdp->declBit(c+113,"VX_cache genblk5[2] bank dram_wb_req_ready", false,-1);
        vcdp->declBit(c+25017,"VX_cache genblk5[2] bank snp_req_valid", false,-1);
        vcdp->declBus(c+24985,"VX_cache genblk5[2] bank snp_req_addr", false,-1, 25,0);
        vcdp->declBit(c+24833,"VX_cache genblk5[2] bank snp_req_invalidate", false,-1);
        vcdp->declBus(c+24841,"VX_cache genblk5[2] bank snp_req_tag", false,-1, 27,0);
        vcdp->declBit(c+10177,"VX_cache genblk5[2] bank snp_req_ready", false,-1);
        vcdp->declBit(c+1793,"VX_cache genblk5[2] bank snp_rsp_valid", false,-1);
        vcdp->declBus(c+1801,"VX_cache genblk5[2] bank snp_rsp_tag", false,-1, 27,0);
        vcdp->declBit(c+121,"VX_cache genblk5[2] bank snp_rsp_ready", false,-1);
        vcdp->declBit(c+6209,"VX_cache genblk5[2] bank snrq_pop", false,-1);
        vcdp->declBit(c+17545,"VX_cache genblk5[2] bank snrq_empty", false,-1);
        vcdp->declBit(c+17553,"VX_cache genblk5[2] bank snrq_full", false,-1);
        vcdp->declBus(c+6217,"VX_cache genblk5[2] bank snrq_addr_st0", false,-1, 25,0);
        vcdp->declBit(c+6225,"VX_cache genblk5[2] bank snrq_invalidate_st0", false,-1);
        vcdp->declBus(c+6233,"VX_cache genblk5[2] bank snrq_tag_st0", false,-1, 27,0);
        vcdp->declBit(c+6241,"VX_cache genblk5[2] bank dfpq_pop", false,-1);
        vcdp->declBit(c+17561,"VX_cache genblk5[2] bank dfpq_empty", false,-1);
        vcdp->declBit(c+17569,"VX_cache genblk5[2] bank dfpq_full", false,-1);
        vcdp->declBus(c+6249,"VX_cache genblk5[2] bank dfpq_addr_st0", false,-1, 25,0);
        vcdp->declArray(c+6257,"VX_cache genblk5[2] bank dfpq_filldata_st0", false,-1, 127,0);
        vcdp->declBit(c+6289,"VX_cache genblk5[2] bank reqq_pop", false,-1);
        vcdp->declBit(c+1025,"VX_cache genblk5[2] bank reqq_push", false,-1);
        vcdp->declBit(c+6297,"VX_cache genblk5[2] bank reqq_empty", false,-1);
        vcdp->declBit(c+17577,"VX_cache genblk5[2] bank reqq_full", false,-1);
        vcdp->declBit(c+6305,"VX_cache genblk5[2] bank reqq_req_st0", false,-1);
        vcdp->declBus(c+6313,"VX_cache genblk5[2] bank reqq_req_tid_st0", false,-1, 1,0);
        vcdp->declBit(c+6321,"VX_cache genblk5[2] bank reqq_req_rw_st0", false,-1);
        vcdp->declBus(c+6329,"VX_cache genblk5[2] bank reqq_req_byteen_st0", false,-1, 3,0);
        vcdp->declBus(c+6337,"VX_cache genblk5[2] bank reqq_req_addr_st0", false,-1, 29,0);
        vcdp->declBus(c+6345,"VX_cache genblk5[2] bank reqq_req_writeword_st0", false,-1, 31,0);
        vcdp->declQuad(c+17585,"VX_cache genblk5[2] bank reqq_req_tag_st0", false,-1, 41,0);
        vcdp->declBit(c+6353,"VX_cache genblk5[2] bank mrvq_pop", false,-1);
        vcdp->declBit(c+17601,"VX_cache genblk5[2] bank mrvq_full", false,-1);
        vcdp->declBit(c+17609,"VX_cache genblk5[2] bank mrvq_stop", false,-1);
        vcdp->declBit(c+6361,"VX_cache genblk5[2] bank mrvq_valid_st0", false,-1);
        vcdp->declBus(c+17617,"VX_cache genblk5[2] bank mrvq_tid_st0", false,-1, 1,0);
        vcdp->declBus(c+17625,"VX_cache genblk5[2] bank mrvq_addr_st0", false,-1, 25,0);
        vcdp->declBus(c+17633,"VX_cache genblk5[2] bank mrvq_wsel_st0", false,-1, 1,0);
        vcdp->declBus(c+17641,"VX_cache genblk5[2] bank mrvq_writeword_st0", false,-1, 31,0);
        vcdp->declQuad(c+17649,"VX_cache genblk5[2] bank mrvq_tag_st0", false,-1, 41,0);
        vcdp->declBit(c+6369,"VX_cache genblk5[2] bank mrvq_rw_st0", false,-1);
        vcdp->declBus(c+17665,"VX_cache genblk5[2] bank mrvq_byteen_st0", false,-1, 3,0);
        vcdp->declBit(c+17673,"VX_cache genblk5[2] bank mrvq_is_snp_st0", false,-1);
        vcdp->declBit(c+17681,"VX_cache genblk5[2] bank mrvq_snp_invalidate_st0", false,-1);
        vcdp->declBit(c+6377,"VX_cache genblk5[2] bank mrvq_pending_hazard_st1e", false,-1);
        vcdp->declBit(c+6385,"VX_cache genblk5[2] bank st2_pending_hazard_st1e", false,-1);
        vcdp->declBit(c+6393,"VX_cache genblk5[2] bank force_request_miss_st1e", false,-1);
        vcdp->declBus(c+17689,"VX_cache genblk5[2] bank miss_add_tid", false,-1, 1,0);
        vcdp->declQuad(c+17697,"VX_cache genblk5[2] bank miss_add_tag", false,-1, 41,0);
        vcdp->declBit(c+17713,"VX_cache genblk5[2] bank miss_add_rw", false,-1);
        vcdp->declBus(c+17721,"VX_cache genblk5[2] bank miss_add_byteen", false,-1, 3,0);
        vcdp->declBus(c+10169,"VX_cache genblk5[2] bank addr_st2", false,-1, 25,0);
        vcdp->declBit(c+17729,"VX_cache genblk5[2] bank is_fill_st2", false,-1);
        vcdp->declBit(c+6401,"VX_cache genblk5[2] bank recover_mrvq_state_st2", false,-1);
        vcdp->declBit(c+6409,"VX_cache genblk5[2] bank mrvq_push_stall", false,-1);
        vcdp->declBit(c+6417,"VX_cache genblk5[2] bank cwbq_push_stall", false,-1);
        vcdp->declBit(c+6425,"VX_cache genblk5[2] bank dwbq_push_stall", false,-1);
        vcdp->declBit(c+6433,"VX_cache genblk5[2] bank dram_fill_req_stall", false,-1);
        vcdp->declBit(c+6441,"VX_cache genblk5[2] bank stall_bank_pipe", false,-1);
        vcdp->declBit(c+6449,"VX_cache genblk5[2] bank is_fill_in_pipe", false,-1);
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+6457+i*1,"VX_cache genblk5[2] bank is_fill_st1", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+6465+i*1,"VX_cache genblk5[2] bank going_to_write_st1", true,(i+0));}}
        vcdp->declBus(c+25161,"VX_cache genblk5[2] bank j", false,-1, 31,0);
        vcdp->declBit(c+6361,"VX_cache genblk5[2] bank mrvq_pop_unqual", false,-1);
        vcdp->declBit(c+6473,"VX_cache genblk5[2] bank dfpq_pop_unqual", false,-1);
        vcdp->declBit(c+6481,"VX_cache genblk5[2] bank reqq_pop_unqual", false,-1);
        vcdp->declBit(c+6489,"VX_cache genblk5[2] bank snrq_pop_unqual", false,-1);
        vcdp->declBit(c+6473,"VX_cache genblk5[2] bank qual_is_fill_st0", false,-1);
        vcdp->declBit(c+6497,"VX_cache genblk5[2] bank qual_valid_st0", false,-1);
        vcdp->declBus(c+6505,"VX_cache genblk5[2] bank qual_addr_st0", false,-1, 25,0);
        vcdp->declBus(c+6513,"VX_cache genblk5[2] bank qual_wsel_st0", false,-1, 1,0);
        vcdp->declBit(c+6361,"VX_cache genblk5[2] bank qual_is_mrvq_st0", false,-1);
        vcdp->declBus(c+6521,"VX_cache genblk5[2] bank qual_writeword_st0", false,-1, 31,0);
        vcdp->declArray(c+6529,"VX_cache genblk5[2] bank qual_writedata_st0", false,-1, 127,0);
        vcdp->declQuad(c+6561,"VX_cache genblk5[2] bank qual_inst_meta_st0", false,-1, 48,0);
        vcdp->declBit(c+6577,"VX_cache genblk5[2] bank qual_going_to_write_st0", false,-1);
        vcdp->declBit(c+6585,"VX_cache genblk5[2] bank qual_is_snp_st0", false,-1);
        vcdp->declBit(c+6593,"VX_cache genblk5[2] bank qual_snp_invalidate_st0", false,-1);
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+6601+i*1,"VX_cache genblk5[2] bank valid_st1", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+6609+i*1,"VX_cache genblk5[2] bank addr_st1", true,(i+0), 25,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+6617+i*1,"VX_cache genblk5[2] bank wsel_st1", true,(i+0), 1,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+6625+i*1,"VX_cache genblk5[2] bank writeword_st1", true,(i+0), 31,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declQuad(c+6633+i*2,"VX_cache genblk5[2] bank inst_meta_st1", true,(i+0), 48,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declArray(c+6649+i*4,"VX_cache genblk5[2] bank writedata_st1", true,(i+0), 127,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+6681+i*1,"VX_cache genblk5[2] bank is_snp_st1", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+6689+i*1,"VX_cache genblk5[2] bank snp_invalidate_st1", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+6697+i*1,"VX_cache genblk5[2] bank is_mrvq_st1", true,(i+0));}}
        vcdp->declBus(c+6705,"VX_cache genblk5[2] bank readword_st1e", false,-1, 31,0);
        vcdp->declArray(c+6713,"VX_cache genblk5[2] bank readdata_st1e", false,-1, 127,0);
        vcdp->declBus(c+6745,"VX_cache genblk5[2] bank readtag_st1e", false,-1, 19,0);
        vcdp->declBit(c+6753,"VX_cache genblk5[2] bank miss_st1e", false,-1);
        vcdp->declBit(c+6761,"VX_cache genblk5[2] bank dirty_st1e", false,-1);
        vcdp->declBus(c+6769,"VX_cache genblk5[2] bank dirtyb_st1e", false,-1, 15,0);
        vcdp->declQuad(c+6777,"VX_cache genblk5[2] bank tag_st1e", false,-1, 41,0);
        vcdp->declBus(c+6793,"VX_cache genblk5[2] bank tid_st1e", false,-1, 1,0);
        vcdp->declBit(c+6801,"VX_cache genblk5[2] bank mem_rw_st1e", false,-1);
        vcdp->declBus(c+6809,"VX_cache genblk5[2] bank mem_byteen_st1e", false,-1, 3,0);
        vcdp->declBit(c+6817,"VX_cache genblk5[2] bank fill_saw_dirty_st1e", false,-1);
        vcdp->declBit(c+6825,"VX_cache genblk5[2] bank is_snp_st1e", false,-1);
        vcdp->declBit(c+6833,"VX_cache genblk5[2] bank snp_invalidate_st1e", false,-1);
        vcdp->declBit(c+6841,"VX_cache genblk5[2] bank snp_to_mrvq_st1e", false,-1);
        vcdp->declBit(c+6849,"VX_cache genblk5[2] bank mrvq_init_ready_state_st1e", false,-1);
        vcdp->declBit(c+6857,"VX_cache genblk5[2] bank miss_add_because_miss", false,-1);
        vcdp->declBit(c+6865,"VX_cache genblk5[2] bank valid_st1e", false,-1);
        vcdp->declBit(c+6873,"VX_cache genblk5[2] bank is_mrvq_st1e", false,-1);
        vcdp->declBit(c+6881,"VX_cache genblk5[2] bank mrvq_recover_ready_state_st1e", false,-1);
        vcdp->declBus(c+6889,"VX_cache genblk5[2] bank addr_st1e", false,-1, 25,0);
        vcdp->declBit(c+6897,"VX_cache genblk5[2] bank qual_valid_st1e_2", false,-1);
        vcdp->declBit(c+6873,"VX_cache genblk5[2] bank is_mrvq_st1e_st2", false,-1);
        vcdp->declBit(c+17737,"VX_cache genblk5[2] bank valid_st2", false,-1);
        vcdp->declBus(c+17745,"VX_cache genblk5[2] bank wsel_st2", false,-1, 1,0);
        vcdp->declBus(c+17753,"VX_cache genblk5[2] bank writeword_st2", false,-1, 31,0);
        vcdp->declBus(c+17761,"VX_cache genblk5[2] bank readword_st2", false,-1, 31,0);
        vcdp->declArray(c+17769,"VX_cache genblk5[2] bank readdata_st2", false,-1, 127,0);
        vcdp->declBit(c+17801,"VX_cache genblk5[2] bank miss_st2", false,-1);
        vcdp->declBit(c+17809,"VX_cache genblk5[2] bank dirty_st2", false,-1);
        vcdp->declBus(c+17817,"VX_cache genblk5[2] bank dirtyb_st2", false,-1, 15,0);
        vcdp->declQuad(c+17825,"VX_cache genblk5[2] bank inst_meta_st2", false,-1, 48,0);
        vcdp->declBus(c+17841,"VX_cache genblk5[2] bank readtag_st2", false,-1, 19,0);
        vcdp->declBit(c+17849,"VX_cache genblk5[2] bank fill_saw_dirty_st2", false,-1);
        vcdp->declBit(c+17857,"VX_cache genblk5[2] bank is_snp_st2", false,-1);
        vcdp->declBit(c+17865,"VX_cache genblk5[2] bank snp_invalidate_st2", false,-1);
        vcdp->declBit(c+17873,"VX_cache genblk5[2] bank snp_to_mrvq_st2", false,-1);
        vcdp->declBit(c+17881,"VX_cache genblk5[2] bank is_mrvq_st2", false,-1);
        vcdp->declBit(c+6905,"VX_cache genblk5[2] bank mrvq_init_ready_state_st2", false,-1);
        vcdp->declBit(c+17889,"VX_cache genblk5[2] bank mrvq_recover_ready_state_st2", false,-1);
        vcdp->declBit(c+17897,"VX_cache genblk5[2] bank mrvq_init_ready_state_unqual_st2", false,-1);
        vcdp->declBit(c+6913,"VX_cache genblk5[2] bank mrvq_init_ready_state_hazard_st0_st1", false,-1);
        vcdp->declBit(c+6921,"VX_cache genblk5[2] bank mrvq_init_ready_state_hazard_st1e_st1", false,-1);
        vcdp->declBit(c+17873,"VX_cache genblk5[2] bank miss_add_because_pending", false,-1);
        vcdp->declBit(c+6929,"VX_cache genblk5[2] bank miss_add_unqual", false,-1);
        vcdp->declBit(c+6937,"VX_cache genblk5[2] bank miss_add", false,-1);
        vcdp->declBus(c+10169,"VX_cache genblk5[2] bank miss_add_addr", false,-1, 25,0);
        vcdp->declBus(c+17745,"VX_cache genblk5[2] bank miss_add_wsel", false,-1, 1,0);
        vcdp->declBus(c+17753,"VX_cache genblk5[2] bank miss_add_data", false,-1, 31,0);
        vcdp->declBit(c+17857,"VX_cache genblk5[2] bank miss_add_is_snp", false,-1);
        vcdp->declBit(c+17865,"VX_cache genblk5[2] bank miss_add_snp_invalidate", false,-1);
        vcdp->declBit(c+6945,"VX_cache genblk5[2] bank miss_add_is_mrvq", false,-1);
        vcdp->declBit(c+6953,"VX_cache genblk5[2] bank cwbq_push", false,-1);
        vcdp->declBit(c+1033,"VX_cache genblk5[2] bank cwbq_pop", false,-1);
        vcdp->declBit(c+17905,"VX_cache genblk5[2] bank cwbq_empty", false,-1);
        vcdp->declBit(c+17913,"VX_cache genblk5[2] bank cwbq_full", false,-1);
        vcdp->declBit(c+6961,"VX_cache genblk5[2] bank cwbq_push_unqual", false,-1);
        vcdp->declBus(c+17761,"VX_cache genblk5[2] bank cwbq_data", false,-1, 31,0);
        vcdp->declBus(c+17689,"VX_cache genblk5[2] bank cwbq_tid", false,-1, 1,0);
        vcdp->declQuad(c+17697,"VX_cache genblk5[2] bank cwbq_tag", false,-1, 41,0);
        vcdp->declBit(c+6929,"VX_cache genblk5[2] bank dram_fill_req_fast", false,-1);
        vcdp->declBit(c+6969,"VX_cache genblk5[2] bank dram_fill_req_unqual", false,-1);
        vcdp->declBit(c+6977,"VX_cache genblk5[2] bank dwbq_push", false,-1);
        vcdp->declBit(c+1041,"VX_cache genblk5[2] bank dwbq_pop", false,-1);
        vcdp->declBit(c+17921,"VX_cache genblk5[2] bank dwbq_empty", false,-1);
        vcdp->declBit(c+17929,"VX_cache genblk5[2] bank dwbq_full", false,-1);
        vcdp->declBit(c+6985,"VX_cache genblk5[2] bank dwbq_is_dwb_in", false,-1);
        vcdp->declBit(c+6993,"VX_cache genblk5[2] bank dwbq_is_snp_in", false,-1);
        vcdp->declBit(c+7001,"VX_cache genblk5[2] bank dwbq_is_dwb_out", false,-1);
        vcdp->declBit(c+7009,"VX_cache genblk5[2] bank dwbq_is_snp_out", false,-1);
        vcdp->declBit(c+7017,"VX_cache genblk5[2] bank dwbq_push_unqual", false,-1);
        vcdp->declBus(c+17937,"VX_cache genblk5[2] bank dwbq_req_addr", false,-1, 25,0);
        vcdp->declBus(c+17945,"VX_cache genblk5[2] bank snrq_tag_st2", false,-1, 27,0);
        vcdp->declBit(c+569,"VX_cache genblk5[2] bank dram_wb_req_fire", false,-1);
        vcdp->declBit(c+577,"VX_cache genblk5[2] bank snp_rsp_fire", false,-1);
        vcdp->declBit(c+17953,"VX_cache genblk5[2] bank dwbq_dual_valid_sel", false,-1);
        vcdp->declBus(c+25169,"VX_cache genblk5[2] bank snp_req_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[2] bank snp_req_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[2] bank snp_req_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[2] bank snp_req_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[2] bank snp_req_queue reset", false,-1);
        vcdp->declBit(c+25017,"VX_cache genblk5[2] bank snp_req_queue push", false,-1);
        vcdp->declBit(c+6209,"VX_cache genblk5[2] bank snp_req_queue pop", false,-1);
        vcdp->declQuad(c+585,"VX_cache genblk5[2] bank snp_req_queue data_in", false,-1, 54,0);
        vcdp->declQuad(c+7025,"VX_cache genblk5[2] bank snp_req_queue data_out", false,-1, 54,0);
        vcdp->declBit(c+17545,"VX_cache genblk5[2] bank snp_req_queue empty", false,-1);
        vcdp->declBit(c+17553,"VX_cache genblk5[2] bank snp_req_queue full", false,-1);
        vcdp->declBus(c+17961,"VX_cache genblk5[2] bank snp_req_queue size", false,-1, 4,0);
        vcdp->declBus(c+17961,"VX_cache genblk5[2] bank snp_req_queue size_r", false,-1, 4,0);
        vcdp->declBit(c+7041,"VX_cache genblk5[2] bank snp_req_queue reading", false,-1);
        vcdp->declBit(c+601,"VX_cache genblk5[2] bank snp_req_queue writing", false,-1);
        {int i; for (i=0; i<16; i++) {
                vcdp->declQuad(c+17969+i*2,"VX_cache genblk5[2] bank snp_req_queue genblk3 data", true,(i+0), 54,0);}}
        vcdp->declQuad(c+18225,"VX_cache genblk5[2] bank snp_req_queue genblk3 genblk2 head_r", false,-1, 54,0);
        vcdp->declQuad(c+18241,"VX_cache genblk5[2] bank snp_req_queue genblk3 genblk2 curr_r", false,-1, 54,0);
        vcdp->declBus(c+18257,"VX_cache genblk5[2] bank snp_req_queue genblk3 genblk2 wr_ptr_r", false,-1, 3,0);
        vcdp->declBus(c+18265,"VX_cache genblk5[2] bank snp_req_queue genblk3 genblk2 rd_ptr_r", false,-1, 3,0);
        vcdp->declBus(c+18273,"VX_cache genblk5[2] bank snp_req_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 3,0);
        vcdp->declBit(c+17545,"VX_cache genblk5[2] bank snp_req_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+17553,"VX_cache genblk5[2] bank snp_req_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+18281,"VX_cache genblk5[2] bank snp_req_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25177,"VX_cache genblk5[2] bank dfp_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[2] bank dfp_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[2] bank dfp_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[2] bank dfp_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[2] bank dfp_queue reset", false,-1);
        vcdp->declBit(c+25009,"VX_cache genblk5[2] bank dfp_queue push", false,-1);
        vcdp->declBit(c+6241,"VX_cache genblk5[2] bank dfp_queue pop", false,-1);
        vcdp->declArray(c+609,"VX_cache genblk5[2] bank dfp_queue data_in", false,-1, 153,0);
        vcdp->declArray(c+7049,"VX_cache genblk5[2] bank dfp_queue data_out", false,-1, 153,0);
        vcdp->declBit(c+17561,"VX_cache genblk5[2] bank dfp_queue empty", false,-1);
        vcdp->declBit(c+17569,"VX_cache genblk5[2] bank dfp_queue full", false,-1);
        vcdp->declBus(c+18289,"VX_cache genblk5[2] bank dfp_queue size", false,-1, 4,0);
        vcdp->declBus(c+18289,"VX_cache genblk5[2] bank dfp_queue size_r", false,-1, 4,0);
        vcdp->declBit(c+7089,"VX_cache genblk5[2] bank dfp_queue reading", false,-1);
        vcdp->declBit(c+649,"VX_cache genblk5[2] bank dfp_queue writing", false,-1);
        {int i; for (i=0; i<16; i++) {
                vcdp->declArray(c+18297+i*5,"VX_cache genblk5[2] bank dfp_queue genblk3 data", true,(i+0), 153,0);}}
        vcdp->declArray(c+18937,"VX_cache genblk5[2] bank dfp_queue genblk3 genblk2 head_r", false,-1, 153,0);
        vcdp->declArray(c+18977,"VX_cache genblk5[2] bank dfp_queue genblk3 genblk2 curr_r", false,-1, 153,0);
        vcdp->declBus(c+19017,"VX_cache genblk5[2] bank dfp_queue genblk3 genblk2 wr_ptr_r", false,-1, 3,0);
        vcdp->declBus(c+19025,"VX_cache genblk5[2] bank dfp_queue genblk3 genblk2 rd_ptr_r", false,-1, 3,0);
        vcdp->declBus(c+19033,"VX_cache genblk5[2] bank dfp_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 3,0);
        vcdp->declBit(c+17561,"VX_cache genblk5[2] bank dfp_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+17569,"VX_cache genblk5[2] bank dfp_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+19041,"VX_cache genblk5[2] bank dfp_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank core_req_arb WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank core_req_arb NUM_REQUESTS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank core_req_arb CREQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25097,"VX_cache genblk5[2] bank core_req_arb CORE_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBus(c+25105,"VX_cache genblk5[2] bank core_req_arb CORE_TAG_ID_BITS", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[2] bank core_req_arb clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[2] bank core_req_arb reset", false,-1);
        vcdp->declBit(c+1025,"VX_cache genblk5[2] bank core_req_arb reqq_push", false,-1);
        vcdp->declBus(c+97,"VX_cache genblk5[2] bank core_req_arb bank_valids", false,-1, 3,0);
        vcdp->declBus(c+24513,"VX_cache genblk5[2] bank core_req_arb bank_rw", false,-1, 3,0);
        vcdp->declBus(c+24521,"VX_cache genblk5[2] bank core_req_arb bank_byteen", false,-1, 15,0);
        vcdp->declArray(c+24561,"VX_cache genblk5[2] bank core_req_arb bank_writedata", false,-1, 127,0);
        vcdp->declArray(c+24529,"VX_cache genblk5[2] bank core_req_arb bank_addr", false,-1, 119,0);
        vcdp->declQuad(c+24593,"VX_cache genblk5[2] bank core_req_arb bank_tag", false,-1, 41,0);
        vcdp->declBit(c+6289,"VX_cache genblk5[2] bank core_req_arb reqq_pop", false,-1);
        vcdp->declBit(c+6305,"VX_cache genblk5[2] bank core_req_arb reqq_req_st0", false,-1);
        vcdp->declBus(c+6313,"VX_cache genblk5[2] bank core_req_arb reqq_req_tid_st0", false,-1, 1,0);
        vcdp->declBit(c+6321,"VX_cache genblk5[2] bank core_req_arb reqq_req_rw_st0", false,-1);
        vcdp->declBus(c+6329,"VX_cache genblk5[2] bank core_req_arb reqq_req_byteen_st0", false,-1, 3,0);
        vcdp->declBus(c+6337,"VX_cache genblk5[2] bank core_req_arb reqq_req_addr_st0", false,-1, 29,0);
        vcdp->declBus(c+6345,"VX_cache genblk5[2] bank core_req_arb reqq_req_writedata_st0", false,-1, 31,0);
        vcdp->declQuad(c+17585,"VX_cache genblk5[2] bank core_req_arb reqq_req_tag_st0", false,-1, 41,0);
        vcdp->declBit(c+6297,"VX_cache genblk5[2] bank core_req_arb reqq_empty", false,-1);
        vcdp->declBit(c+17577,"VX_cache genblk5[2] bank core_req_arb reqq_full", false,-1);
        vcdp->declBus(c+7097,"VX_cache genblk5[2] bank core_req_arb out_per_valids", false,-1, 3,0);
        vcdp->declBus(c+7105,"VX_cache genblk5[2] bank core_req_arb out_per_rw", false,-1, 3,0);
        vcdp->declBus(c+7113,"VX_cache genblk5[2] bank core_req_arb out_per_byteen", false,-1, 15,0);
        vcdp->declArray(c+7121,"VX_cache genblk5[2] bank core_req_arb out_per_addr", false,-1, 119,0);
        vcdp->declArray(c+7153,"VX_cache genblk5[2] bank core_req_arb out_per_writedata", false,-1, 127,0);
        vcdp->declQuad(c+7185,"VX_cache genblk5[2] bank core_req_arb out_per_tag", false,-1, 41,0);
        vcdp->declBus(c+19049,"VX_cache genblk5[2] bank core_req_arb use_per_valids", false,-1, 3,0);
        vcdp->declBus(c+19057,"VX_cache genblk5[2] bank core_req_arb use_per_rw", false,-1, 3,0);
        vcdp->declBus(c+19065,"VX_cache genblk5[2] bank core_req_arb use_per_byteen", false,-1, 15,0);
        vcdp->declArray(c+19073,"VX_cache genblk5[2] bank core_req_arb use_per_addr", false,-1, 119,0);
        vcdp->declArray(c+19105,"VX_cache genblk5[2] bank core_req_arb use_per_writedata", false,-1, 127,0);
        vcdp->declQuad(c+17585,"VX_cache genblk5[2] bank core_req_arb use_per_tag", false,-1, 41,0);
        vcdp->declBus(c+19049,"VX_cache genblk5[2] bank core_req_arb qual_valids", false,-1, 3,0);
        vcdp->declBus(c+19057,"VX_cache genblk5[2] bank core_req_arb qual_rw", false,-1, 3,0);
        vcdp->declBus(c+19065,"VX_cache genblk5[2] bank core_req_arb qual_byteen", false,-1, 15,0);
        vcdp->declArray(c+19073,"VX_cache genblk5[2] bank core_req_arb qual_addr", false,-1, 119,0);
        vcdp->declArray(c+19105,"VX_cache genblk5[2] bank core_req_arb qual_writedata", false,-1, 127,0);
        vcdp->declQuad(c+17585,"VX_cache genblk5[2] bank core_req_arb qual_tag", false,-1, 41,0);
        vcdp->declBit(c+19137,"VX_cache genblk5[2] bank core_req_arb o_empty", false,-1);
        vcdp->declBit(c+19145,"VX_cache genblk5[2] bank core_req_arb use_empty", false,-1);
        vcdp->declBit(c+7201,"VX_cache genblk5[2] bank core_req_arb out_empty", false,-1);
        vcdp->declBit(c+1049,"VX_cache genblk5[2] bank core_req_arb push_qual", false,-1);
        vcdp->declBit(c+7209,"VX_cache genblk5[2] bank core_req_arb pop_qual", false,-1);
        vcdp->declBus(c+7217,"VX_cache genblk5[2] bank core_req_arb real_out_per_valids", false,-1, 3,0);
        vcdp->declBus(c+6313,"VX_cache genblk5[2] bank core_req_arb qual_request_index", false,-1, 1,0);
        vcdp->declBit(c+6305,"VX_cache genblk5[2] bank core_req_arb qual_has_request", false,-1);
        vcdp->declBus(c+25185,"VX_cache genblk5[2] bank core_req_arb reqq_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank core_req_arb reqq_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[2] bank core_req_arb reqq_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[2] bank core_req_arb reqq_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[2] bank core_req_arb reqq_queue reset", false,-1);
        vcdp->declBit(c+1049,"VX_cache genblk5[2] bank core_req_arb reqq_queue push", false,-1);
        vcdp->declBit(c+7209,"VX_cache genblk5[2] bank core_req_arb reqq_queue pop", false,-1);
        vcdp->declArray(c+657,"VX_cache genblk5[2] bank core_req_arb reqq_queue data_in", false,-1, 313,0);
        vcdp->declArray(c+7225,"VX_cache genblk5[2] bank core_req_arb reqq_queue data_out", false,-1, 313,0);
        vcdp->declBit(c+19137,"VX_cache genblk5[2] bank core_req_arb reqq_queue empty", false,-1);
        vcdp->declBit(c+17577,"VX_cache genblk5[2] bank core_req_arb reqq_queue full", false,-1);
        vcdp->declBus(c+19153,"VX_cache genblk5[2] bank core_req_arb reqq_queue size", false,-1, 2,0);
        vcdp->declBus(c+19153,"VX_cache genblk5[2] bank core_req_arb reqq_queue size_r", false,-1, 2,0);
        vcdp->declBit(c+7305,"VX_cache genblk5[2] bank core_req_arb reqq_queue reading", false,-1);
        vcdp->declBit(c+737,"VX_cache genblk5[2] bank core_req_arb reqq_queue writing", false,-1);
        {int i; for (i=0; i<4; i++) {
                vcdp->declArray(c+19161+i*10,"VX_cache genblk5[2] bank core_req_arb reqq_queue genblk3 data", true,(i+0), 313,0);}}
        vcdp->declArray(c+19481,"VX_cache genblk5[2] bank core_req_arb reqq_queue genblk3 genblk2 head_r", false,-1, 313,0);
        vcdp->declArray(c+19561,"VX_cache genblk5[2] bank core_req_arb reqq_queue genblk3 genblk2 curr_r", false,-1, 313,0);
        vcdp->declBus(c+19641,"VX_cache genblk5[2] bank core_req_arb reqq_queue genblk3 genblk2 wr_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+19649,"VX_cache genblk5[2] bank core_req_arb reqq_queue genblk3 genblk2 rd_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+19657,"VX_cache genblk5[2] bank core_req_arb reqq_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 1,0);
        vcdp->declBit(c+19137,"VX_cache genblk5[2] bank core_req_arb reqq_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+17577,"VX_cache genblk5[2] bank core_req_arb reqq_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+19665,"VX_cache genblk5[2] bank core_req_arb reqq_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank core_req_arb sel_bank N", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[2] bank core_req_arb sel_bank clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[2] bank core_req_arb sel_bank reset", false,-1);
        vcdp->declBus(c+19049,"VX_cache genblk5[2] bank core_req_arb sel_bank requests", false,-1, 3,0);
        vcdp->declBus(c+6313,"VX_cache genblk5[2] bank core_req_arb sel_bank grant_index", false,-1, 1,0);
        vcdp->declBus(c+7313,"VX_cache genblk5[2] bank core_req_arb sel_bank grant_onehot", false,-1, 3,0);
        vcdp->declBit(c+6305,"VX_cache genblk5[2] bank core_req_arb sel_bank grant_valid", false,-1);
        vcdp->declBus(c+7313,"VX_cache genblk5[2] bank core_req_arb sel_bank genblk2 grant_onehot_r", false,-1, 3,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank core_req_arb sel_bank genblk2 priority_encoder N", false,-1, 31,0);
        vcdp->declBus(c+19049,"VX_cache genblk5[2] bank core_req_arb sel_bank genblk2 priority_encoder data_in", false,-1, 3,0);
        vcdp->declBus(c+6313,"VX_cache genblk5[2] bank core_req_arb sel_bank genblk2 priority_encoder data_out", false,-1, 1,0);
        vcdp->declBit(c+6305,"VX_cache genblk5[2] bank core_req_arb sel_bank genblk2 priority_encoder valid_out", false,-1);
        vcdp->declBus(c+7321,"VX_cache genblk5[2] bank core_req_arb sel_bank genblk2 priority_encoder i", false,-1, 31,0);
        vcdp->declBus(c+25193,"VX_cache genblk5[2] bank s0_1_c0 N", false,-1, 31,0);
        vcdp->declBus(c+25057,"VX_cache genblk5[2] bank s0_1_c0 PASSTHRU", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[2] bank s0_1_c0 clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[2] bank s0_1_c0 reset", false,-1);
        vcdp->declBit(c+6441,"VX_cache genblk5[2] bank s0_1_c0 stall", false,-1);
        vcdp->declBit(c+25137,"VX_cache genblk5[2] bank s0_1_c0 flush", false,-1);
        vcdp->declArray(c+7329,"VX_cache genblk5[2] bank s0_1_c0 in", false,-1, 242,0);
        vcdp->declArray(c+19673,"VX_cache genblk5[2] bank s0_1_c0 out", false,-1, 242,0);
        vcdp->declArray(c+19673,"VX_cache genblk5[2] bank s0_1_c0 value", false,-1, 242,0);
        vcdp->declBus(c+25065,"VX_cache genblk5[2] bank tag_data_access CACHE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[2] bank tag_data_access BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank tag_data_access NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank tag_data_access WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[2] bank tag_data_access STAGE_1_CYCLES", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[2] bank tag_data_access WRITE_ENABLE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[2] bank tag_data_access DRAM_ENABLE", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[2] bank tag_data_access clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[2] bank tag_data_access reset", false,-1);
        vcdp->declBit(c+6441,"VX_cache genblk5[2] bank tag_data_access stall", false,-1);
        vcdp->declBit(c+6825,"VX_cache genblk5[2] bank tag_data_access is_snp_st1e", false,-1);
        vcdp->declBit(c+6833,"VX_cache genblk5[2] bank tag_data_access snp_invalidate_st1e", false,-1);
        vcdp->declBit(c+6441,"VX_cache genblk5[2] bank tag_data_access stall_bank_pipe", false,-1);
        vcdp->declBit(c+6393,"VX_cache genblk5[2] bank tag_data_access force_request_miss_st1e", false,-1);
        vcdp->declBus(c+7393,"VX_cache genblk5[2] bank tag_data_access readaddr_st10", false,-1, 5,0);
        vcdp->declBus(c+6889,"VX_cache genblk5[2] bank tag_data_access writeaddr_st1e", false,-1, 25,0);
        vcdp->declBit(c+6865,"VX_cache genblk5[2] bank tag_data_access valid_req_st1e", false,-1);
        vcdp->declBit(c+7401,"VX_cache genblk5[2] bank tag_data_access writefill_st1e", false,-1);
        vcdp->declBus(c+7409,"VX_cache genblk5[2] bank tag_data_access writeword_st1e", false,-1, 31,0);
        vcdp->declArray(c+7417,"VX_cache genblk5[2] bank tag_data_access writedata_st1e", false,-1, 127,0);
        vcdp->declBit(c+6801,"VX_cache genblk5[2] bank tag_data_access mem_rw_st1e", false,-1);
        vcdp->declBus(c+6809,"VX_cache genblk5[2] bank tag_data_access mem_byteen_st1e", false,-1, 3,0);
        vcdp->declBus(c+7449,"VX_cache genblk5[2] bank tag_data_access wordsel_st1e", false,-1, 1,0);
        vcdp->declBus(c+6705,"VX_cache genblk5[2] bank tag_data_access readword_st1e", false,-1, 31,0);
        vcdp->declArray(c+6713,"VX_cache genblk5[2] bank tag_data_access readdata_st1e", false,-1, 127,0);
        vcdp->declBus(c+6745,"VX_cache genblk5[2] bank tag_data_access readtag_st1e", false,-1, 19,0);
        vcdp->declBit(c+6753,"VX_cache genblk5[2] bank tag_data_access miss_st1e", false,-1);
        vcdp->declBit(c+6761,"VX_cache genblk5[2] bank tag_data_access dirty_st1e", false,-1);
        vcdp->declBus(c+6769,"VX_cache genblk5[2] bank tag_data_access dirtyb_st1e", false,-1, 15,0);
        vcdp->declBit(c+6817,"VX_cache genblk5[2] bank tag_data_access fill_saw_dirty_st1e", false,-1);
        vcdp->declBit(c+6841,"VX_cache genblk5[2] bank tag_data_access snp_to_mrvq_st1e", false,-1);
        vcdp->declBit(c+6849,"VX_cache genblk5[2] bank tag_data_access mrvq_init_ready_state_st1e", false,-1);
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+7457+i*1,"VX_cache genblk5[2] bank tag_data_access read_valid_st1c", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+7465+i*1,"VX_cache genblk5[2] bank tag_data_access read_dirty_st1c", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+7473+i*1,"VX_cache genblk5[2] bank tag_data_access read_dirtyb_st1c", true,(i+0), 15,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+7481+i*1,"VX_cache genblk5[2] bank tag_data_access read_tag_st1c", true,(i+0), 19,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declArray(c+7489+i*4,"VX_cache genblk5[2] bank tag_data_access read_data_st1c", true,(i+0), 127,0);}}
        vcdp->declBit(c+7521,"VX_cache genblk5[2] bank tag_data_access qual_read_valid_st1", false,-1);
        vcdp->declBit(c+7529,"VX_cache genblk5[2] bank tag_data_access qual_read_dirty_st1", false,-1);
        vcdp->declBus(c+7537,"VX_cache genblk5[2] bank tag_data_access qual_read_dirtyb_st1", false,-1, 15,0);
        vcdp->declBus(c+7545,"VX_cache genblk5[2] bank tag_data_access qual_read_tag_st1", false,-1, 19,0);
        vcdp->declArray(c+7553,"VX_cache genblk5[2] bank tag_data_access qual_read_data_st1", false,-1, 127,0);
        vcdp->declBit(c+7585,"VX_cache genblk5[2] bank tag_data_access use_read_valid_st1e", false,-1);
        vcdp->declBit(c+7593,"VX_cache genblk5[2] bank tag_data_access use_read_dirty_st1e", false,-1);
        vcdp->declBus(c+6769,"VX_cache genblk5[2] bank tag_data_access use_read_dirtyb_st1e", false,-1, 15,0);
        vcdp->declBus(c+6745,"VX_cache genblk5[2] bank tag_data_access use_read_tag_st1e", false,-1, 19,0);
        vcdp->declArray(c+6713,"VX_cache genblk5[2] bank tag_data_access use_read_data_st1e", false,-1, 127,0);
        vcdp->declBus(c+7601,"VX_cache genblk5[2] bank tag_data_access use_write_enable", false,-1, 15,0);
        vcdp->declArray(c+7609,"VX_cache genblk5[2] bank tag_data_access use_write_data", false,-1, 127,0);
        vcdp->declBit(c+6753,"VX_cache genblk5[2] bank tag_data_access fill_sent", false,-1);
        vcdp->declBit(c+7641,"VX_cache genblk5[2] bank tag_data_access invalidate_line", false,-1);
        vcdp->declBit(c+7649,"VX_cache genblk5[2] bank tag_data_access tags_match", false,-1);
        vcdp->declBit(c+7657,"VX_cache genblk5[2] bank tag_data_access real_writefill", false,-1);
        vcdp->declBus(c+7665,"VX_cache genblk5[2] bank tag_data_access writetag_st1e", false,-1, 19,0);
        vcdp->declBus(c+7393,"VX_cache genblk5[2] bank tag_data_access writeladdr_st1e", false,-1, 5,0);
        vcdp->declBus(c+7673,"VX_cache genblk5[2] bank tag_data_access we", false,-1, 15,0);
        vcdp->declArray(c+7609,"VX_cache genblk5[2] bank tag_data_access data_write", false,-1, 127,0);
        vcdp->declBit(c+7681,"VX_cache genblk5[2] bank tag_data_access should_write", false,-1);
        vcdp->declBit(c+7641,"VX_cache genblk5[2] bank tag_data_access snoop_hit_no_pending", false,-1);
        vcdp->declBit(c+7689,"VX_cache genblk5[2] bank tag_data_access req_invalid", false,-1);
        vcdp->declBit(c+7697,"VX_cache genblk5[2] bank tag_data_access req_miss", false,-1);
        vcdp->declBit(c+7705,"VX_cache genblk5[2] bank tag_data_access real_miss", false,-1);
        vcdp->declBit(c+7713,"VX_cache genblk5[2] bank tag_data_access force_core_miss", false,-1);
        vcdp->declBit(c+7721,"VX_cache genblk5[2] bank tag_data_access genblk4[0] normal_write", false,-1);
        vcdp->declBit(c+7729,"VX_cache genblk5[2] bank tag_data_access genblk4[1] normal_write", false,-1);
        vcdp->declBit(c+7737,"VX_cache genblk5[2] bank tag_data_access genblk4[2] normal_write", false,-1);
        vcdp->declBit(c+7745,"VX_cache genblk5[2] bank tag_data_access genblk4[3] normal_write", false,-1);
        vcdp->declBus(c+25065,"VX_cache genblk5[2] bank tag_data_access tag_data_structure CACHE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[2] bank tag_data_access tag_data_structure BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank tag_data_access tag_data_structure NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank tag_data_access tag_data_structure WORD_SIZE", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[2] bank tag_data_access tag_data_structure clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[2] bank tag_data_access tag_data_structure reset", false,-1);
        vcdp->declBit(c+6441,"VX_cache genblk5[2] bank tag_data_access tag_data_structure stall_bank_pipe", false,-1);
        vcdp->declBus(c+7393,"VX_cache genblk5[2] bank tag_data_access tag_data_structure read_addr", false,-1, 5,0);
        vcdp->declBit(c+7521,"VX_cache genblk5[2] bank tag_data_access tag_data_structure read_valid", false,-1);
        vcdp->declBit(c+7529,"VX_cache genblk5[2] bank tag_data_access tag_data_structure read_dirty", false,-1);
        vcdp->declBus(c+7537,"VX_cache genblk5[2] bank tag_data_access tag_data_structure read_dirtyb", false,-1, 15,0);
        vcdp->declBus(c+7545,"VX_cache genblk5[2] bank tag_data_access tag_data_structure read_tag", false,-1, 19,0);
        vcdp->declArray(c+7553,"VX_cache genblk5[2] bank tag_data_access tag_data_structure read_data", false,-1, 127,0);
        vcdp->declBit(c+7641,"VX_cache genblk5[2] bank tag_data_access tag_data_structure invalidate", false,-1);
        vcdp->declBus(c+7601,"VX_cache genblk5[2] bank tag_data_access tag_data_structure write_enable", false,-1, 15,0);
        vcdp->declBit(c+7657,"VX_cache genblk5[2] bank tag_data_access tag_data_structure write_fill", false,-1);
        vcdp->declBus(c+7393,"VX_cache genblk5[2] bank tag_data_access tag_data_structure write_addr", false,-1, 5,0);
        vcdp->declBus(c+7665,"VX_cache genblk5[2] bank tag_data_access tag_data_structure tag_index", false,-1, 19,0);
        vcdp->declArray(c+7609,"VX_cache genblk5[2] bank tag_data_access tag_data_structure write_data", false,-1, 127,0);
        vcdp->declBit(c+6753,"VX_cache genblk5[2] bank tag_data_access tag_data_structure fill_sent", false,-1);
        vcdp->declQuad(c+19737,"VX_cache genblk5[2] bank tag_data_access tag_data_structure dirty", false,-1, 63,0);
        vcdp->declQuad(c+19753,"VX_cache genblk5[2] bank tag_data_access tag_data_structure valid", false,-1, 63,0);
        vcdp->declBit(c+7753,"VX_cache genblk5[2] bank tag_data_access tag_data_structure do_write", false,-1);
        vcdp->declBus(c+19769,"VX_cache genblk5[2] bank tag_data_access tag_data_structure i", false,-1, 31,0);
        vcdp->declBus(c+19777,"VX_cache genblk5[2] bank tag_data_access tag_data_structure j", false,-1, 31,0);
        vcdp->declBus(c+25201,"VX_cache genblk5[2] bank tag_data_access s0_1_c0 N", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[2] bank tag_data_access s0_1_c0 PASSTHRU", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[2] bank tag_data_access s0_1_c0 clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[2] bank tag_data_access s0_1_c0 reset", false,-1);
        vcdp->declBit(c+6441,"VX_cache genblk5[2] bank tag_data_access s0_1_c0 stall", false,-1);
        vcdp->declBit(c+25137,"VX_cache genblk5[2] bank tag_data_access s0_1_c0 flush", false,-1);
        vcdp->declArray(c+7761,"VX_cache genblk5[2] bank tag_data_access s0_1_c0 in", false,-1, 165,0);
        vcdp->declArray(c+7761,"VX_cache genblk5[2] bank tag_data_access s0_1_c0 out", false,-1, 165,0);
        vcdp->declArray(c+19785,"VX_cache genblk5[2] bank tag_data_access s0_1_c0 value", false,-1, 165,0);
        vcdp->declBus(c+25209,"VX_cache genblk5[2] bank st_1e_2 N", false,-1, 31,0);
        vcdp->declBus(c+25057,"VX_cache genblk5[2] bank st_1e_2 PASSTHRU", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[2] bank st_1e_2 clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[2] bank st_1e_2 reset", false,-1);
        vcdp->declBit(c+6441,"VX_cache genblk5[2] bank st_1e_2 stall", false,-1);
        vcdp->declBit(c+25137,"VX_cache genblk5[2] bank st_1e_2 flush", false,-1);
        vcdp->declArray(c+7809,"VX_cache genblk5[2] bank st_1e_2 in", false,-1, 315,0);
        vcdp->declArray(c+19833,"VX_cache genblk5[2] bank st_1e_2 out", false,-1, 315,0);
        vcdp->declArray(c+19833,"VX_cache genblk5[2] bank st_1e_2 value", false,-1, 315,0);
        vcdp->declBus(c+25057,"VX_cache genblk5[2] bank cache_miss_resrv CACHE_ID", false,-1, 31,0);
        vcdp->declBus(c+25233,"VX_cache genblk5[2] bank cache_miss_resrv BANK_ID", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[2] bank cache_miss_resrv BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank cache_miss_resrv NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank cache_miss_resrv WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank cache_miss_resrv NUM_REQUESTS", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[2] bank cache_miss_resrv MRVQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25097,"VX_cache genblk5[2] bank cache_miss_resrv CORE_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBus(c+25113,"VX_cache genblk5[2] bank cache_miss_resrv SNP_REQ_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[2] bank cache_miss_resrv clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[2] bank cache_miss_resrv reset", false,-1);
        vcdp->declBit(c+6937,"VX_cache genblk5[2] bank cache_miss_resrv miss_add", false,-1);
        vcdp->declBit(c+6945,"VX_cache genblk5[2] bank cache_miss_resrv is_mrvq", false,-1);
        vcdp->declBus(c+10169,"VX_cache genblk5[2] bank cache_miss_resrv miss_add_addr", false,-1, 25,0);
        vcdp->declBus(c+17745,"VX_cache genblk5[2] bank cache_miss_resrv miss_add_wsel", false,-1, 1,0);
        vcdp->declBus(c+17753,"VX_cache genblk5[2] bank cache_miss_resrv miss_add_data", false,-1, 31,0);
        vcdp->declBus(c+17689,"VX_cache genblk5[2] bank cache_miss_resrv miss_add_tid", false,-1, 1,0);
        vcdp->declQuad(c+17697,"VX_cache genblk5[2] bank cache_miss_resrv miss_add_tag", false,-1, 41,0);
        vcdp->declBit(c+17713,"VX_cache genblk5[2] bank cache_miss_resrv miss_add_rw", false,-1);
        vcdp->declBus(c+17721,"VX_cache genblk5[2] bank cache_miss_resrv miss_add_byteen", false,-1, 3,0);
        vcdp->declBit(c+6905,"VX_cache genblk5[2] bank cache_miss_resrv mrvq_init_ready_state", false,-1);
        vcdp->declBit(c+17857,"VX_cache genblk5[2] bank cache_miss_resrv miss_add_is_snp", false,-1);
        vcdp->declBit(c+17865,"VX_cache genblk5[2] bank cache_miss_resrv miss_add_snp_invalidate", false,-1);
        vcdp->declBit(c+17601,"VX_cache genblk5[2] bank cache_miss_resrv miss_resrv_full", false,-1);
        vcdp->declBit(c+17609,"VX_cache genblk5[2] bank cache_miss_resrv miss_resrv_stop", false,-1);
        vcdp->declBit(c+7401,"VX_cache genblk5[2] bank cache_miss_resrv is_fill_st1", false,-1);
        vcdp->declBus(c+6889,"VX_cache genblk5[2] bank cache_miss_resrv fill_addr_st1", false,-1, 25,0);
        vcdp->declBit(c+6377,"VX_cache genblk5[2] bank cache_miss_resrv pending_hazard", false,-1);
        vcdp->declBit(c+6353,"VX_cache genblk5[2] bank cache_miss_resrv miss_resrv_pop", false,-1);
        vcdp->declBit(c+6361,"VX_cache genblk5[2] bank cache_miss_resrv miss_resrv_valid_st0", false,-1);
        vcdp->declBus(c+17625,"VX_cache genblk5[2] bank cache_miss_resrv miss_resrv_addr_st0", false,-1, 25,0);
        vcdp->declBus(c+17633,"VX_cache genblk5[2] bank cache_miss_resrv miss_resrv_wsel_st0", false,-1, 1,0);
        vcdp->declBus(c+17641,"VX_cache genblk5[2] bank cache_miss_resrv miss_resrv_data_st0", false,-1, 31,0);
        vcdp->declBus(c+17617,"VX_cache genblk5[2] bank cache_miss_resrv miss_resrv_tid_st0", false,-1, 1,0);
        vcdp->declQuad(c+17649,"VX_cache genblk5[2] bank cache_miss_resrv miss_resrv_tag_st0", false,-1, 41,0);
        vcdp->declBit(c+6369,"VX_cache genblk5[2] bank cache_miss_resrv miss_resrv_rw_st0", false,-1);
        vcdp->declBus(c+17665,"VX_cache genblk5[2] bank cache_miss_resrv miss_resrv_byteen_st0", false,-1, 3,0);
        vcdp->declBit(c+17673,"VX_cache genblk5[2] bank cache_miss_resrv miss_resrv_is_snp_st0", false,-1);
        vcdp->declBit(c+17681,"VX_cache genblk5[2] bank cache_miss_resrv miss_resrv_snp_invalidate_st0", false,-1);
        {int i; for (i=0; i<16; i++) {
                vcdp->declArray(c+19913+i*3,"VX_cache genblk5[2] bank cache_miss_resrv metadata_table", true,(i+0), 84,0);}}
        vcdp->declArray(c+20297,"VX_cache genblk5[2] bank cache_miss_resrv addr_table", false,-1, 415,0);
        vcdp->declBus(c+20401,"VX_cache genblk5[2] bank cache_miss_resrv valid_table", false,-1, 15,0);
        vcdp->declBus(c+20409,"VX_cache genblk5[2] bank cache_miss_resrv ready_table", false,-1, 15,0);
        vcdp->declBus(c+20417,"VX_cache genblk5[2] bank cache_miss_resrv schedule_ptr", false,-1, 3,0);
        vcdp->declBus(c+20425,"VX_cache genblk5[2] bank cache_miss_resrv head_ptr", false,-1, 3,0);
        vcdp->declBus(c+20433,"VX_cache genblk5[2] bank cache_miss_resrv tail_ptr", false,-1, 3,0);
        vcdp->declBus(c+20441,"VX_cache genblk5[2] bank cache_miss_resrv size", false,-1, 4,0);
        vcdp->declBit(c+20449,"VX_cache genblk5[2] bank cache_miss_resrv enqueue_possible", false,-1);
        vcdp->declBus(c+20433,"VX_cache genblk5[2] bank cache_miss_resrv enqueue_index", false,-1, 3,0);
        vcdp->declBus(c+7889,"VX_cache genblk5[2] bank cache_miss_resrv make_ready", false,-1, 15,0);
        vcdp->declBus(c+7897,"VX_cache genblk5[2] bank cache_miss_resrv make_ready_push", false,-1, 15,0);
        vcdp->declBus(c+7905,"VX_cache genblk5[2] bank cache_miss_resrv valid_address_match", false,-1, 15,0);
        vcdp->declBit(c+6361,"VX_cache genblk5[2] bank cache_miss_resrv dequeue_possible", false,-1);
        vcdp->declBus(c+20417,"VX_cache genblk5[2] bank cache_miss_resrv dequeue_index", false,-1, 3,0);
        vcdp->declBit(c+7913,"VX_cache genblk5[2] bank cache_miss_resrv mrvq_push", false,-1);
        vcdp->declBit(c+7921,"VX_cache genblk5[2] bank cache_miss_resrv mrvq_pop", false,-1);
        vcdp->declBit(c+7929,"VX_cache genblk5[2] bank cache_miss_resrv recover_state", false,-1);
        vcdp->declBit(c+7937,"VX_cache genblk5[2] bank cache_miss_resrv increment_head", false,-1);
        vcdp->declBit(c+7945,"VX_cache genblk5[2] bank cache_miss_resrv update_ready", false,-1);
        vcdp->declBit(c+7953,"VX_cache genblk5[2] bank cache_miss_resrv qual_mrvq_init", false,-1);
        vcdp->declBus(c+25217,"VX_cache genblk5[2] bank cwb_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank cwb_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[2] bank cwb_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[2] bank cwb_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[2] bank cwb_queue reset", false,-1);
        vcdp->declBit(c+6953,"VX_cache genblk5[2] bank cwb_queue push", false,-1);
        vcdp->declBit(c+1033,"VX_cache genblk5[2] bank cwb_queue pop", false,-1);
        vcdp->declArray(c+7961,"VX_cache genblk5[2] bank cwb_queue data_in", false,-1, 75,0);
        vcdp->declArray(c+7985,"VX_cache genblk5[2] bank cwb_queue data_out", false,-1, 75,0);
        vcdp->declBit(c+17905,"VX_cache genblk5[2] bank cwb_queue empty", false,-1);
        vcdp->declBit(c+17913,"VX_cache genblk5[2] bank cwb_queue full", false,-1);
        vcdp->declBus(c+20457,"VX_cache genblk5[2] bank cwb_queue size", false,-1, 2,0);
        vcdp->declBus(c+20457,"VX_cache genblk5[2] bank cwb_queue size_r", false,-1, 2,0);
        vcdp->declBit(c+745,"VX_cache genblk5[2] bank cwb_queue reading", false,-1);
        vcdp->declBit(c+8009,"VX_cache genblk5[2] bank cwb_queue writing", false,-1);
        {int i; for (i=0; i<4; i++) {
                vcdp->declArray(c+20465+i*3,"VX_cache genblk5[2] bank cwb_queue genblk3 data", true,(i+0), 75,0);}}
        vcdp->declArray(c+20561,"VX_cache genblk5[2] bank cwb_queue genblk3 genblk2 head_r", false,-1, 75,0);
        vcdp->declArray(c+20585,"VX_cache genblk5[2] bank cwb_queue genblk3 genblk2 curr_r", false,-1, 75,0);
        vcdp->declBus(c+20609,"VX_cache genblk5[2] bank cwb_queue genblk3 genblk2 wr_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+20617,"VX_cache genblk5[2] bank cwb_queue genblk3 genblk2 rd_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+20625,"VX_cache genblk5[2] bank cwb_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 1,0);
        vcdp->declBit(c+17905,"VX_cache genblk5[2] bank cwb_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+17913,"VX_cache genblk5[2] bank cwb_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+20633,"VX_cache genblk5[2] bank cwb_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25225,"VX_cache genblk5[2] bank dwb_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[2] bank dwb_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[2] bank dwb_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[2] bank dwb_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[2] bank dwb_queue reset", false,-1);
        vcdp->declBit(c+6977,"VX_cache genblk5[2] bank dwb_queue push", false,-1);
        vcdp->declBit(c+1041,"VX_cache genblk5[2] bank dwb_queue pop", false,-1);
        vcdp->declArray(c+8017,"VX_cache genblk5[2] bank dwb_queue data_in", false,-1, 199,0);
        vcdp->declArray(c+8073,"VX_cache genblk5[2] bank dwb_queue data_out", false,-1, 199,0);
        vcdp->declBit(c+17921,"VX_cache genblk5[2] bank dwb_queue empty", false,-1);
        vcdp->declBit(c+17929,"VX_cache genblk5[2] bank dwb_queue full", false,-1);
        vcdp->declBus(c+20641,"VX_cache genblk5[2] bank dwb_queue size", false,-1, 2,0);
        vcdp->declBus(c+20641,"VX_cache genblk5[2] bank dwb_queue size_r", false,-1, 2,0);
        vcdp->declBit(c+753,"VX_cache genblk5[2] bank dwb_queue reading", false,-1);
        vcdp->declBit(c+8129,"VX_cache genblk5[2] bank dwb_queue writing", false,-1);
        {int i; for (i=0; i<4; i++) {
                vcdp->declArray(c+20649+i*7,"VX_cache genblk5[2] bank dwb_queue genblk3 data", true,(i+0), 199,0);}}
        vcdp->declArray(c+20873,"VX_cache genblk5[2] bank dwb_queue genblk3 genblk2 head_r", false,-1, 199,0);
        vcdp->declArray(c+20929,"VX_cache genblk5[2] bank dwb_queue genblk3 genblk2 curr_r", false,-1, 199,0);
        vcdp->declBus(c+20985,"VX_cache genblk5[2] bank dwb_queue genblk3 genblk2 wr_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+20993,"VX_cache genblk5[2] bank dwb_queue genblk3 genblk2 rd_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+21001,"VX_cache genblk5[2] bank dwb_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 1,0);
        vcdp->declBit(c+17921,"VX_cache genblk5[2] bank dwb_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+17929,"VX_cache genblk5[2] bank dwb_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+21009,"VX_cache genblk5[2] bank dwb_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25057,"VX_cache genblk5[3] bank CACHE_ID", false,-1, 31,0);
        vcdp->declBus(c+25241,"VX_cache genblk5[3] bank BANK_ID", false,-1, 31,0);
        vcdp->declBus(c+25065,"VX_cache genblk5[3] bank CACHE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[3] bank BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank NUM_REQUESTS", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[3] bank STAGE_1_CYCLES", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank CREQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[3] bank MRVQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[3] bank DFPQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[3] bank SNRQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank CWBQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank DWBQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank DFQQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[3] bank WRITE_ENABLE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[3] bank DRAM_ENABLE", false,-1, 31,0);
        vcdp->declBus(c+25057,"VX_cache genblk5[3] bank SNOOP_FORWARDING", false,-1, 31,0);
        vcdp->declBus(c+25097,"VX_cache genblk5[3] bank CORE_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBus(c+25105,"VX_cache genblk5[3] bank CORE_TAG_ID_BITS", false,-1, 31,0);
        vcdp->declBus(c+25113,"VX_cache genblk5[3] bank SNP_REQ_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[3] bank clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[3] bank reset", false,-1);
        vcdp->declBus(c+129,"VX_cache genblk5[3] bank core_req_valid", false,-1, 3,0);
        vcdp->declBus(c+24513,"VX_cache genblk5[3] bank core_req_rw", false,-1, 3,0);
        vcdp->declBus(c+24521,"VX_cache genblk5[3] bank core_req_byteen", false,-1, 15,0);
        vcdp->declArray(c+24529,"VX_cache genblk5[3] bank core_req_addr", false,-1, 119,0);
        vcdp->declArray(c+24561,"VX_cache genblk5[3] bank core_req_data", false,-1, 127,0);
        vcdp->declQuad(c+24593,"VX_cache genblk5[3] bank core_req_tag", false,-1, 41,0);
        vcdp->declBit(c+10225,"VX_cache genblk5[3] bank core_req_ready", false,-1);
        vcdp->declBit(c+10193,"VX_cache genblk5[3] bank core_rsp_valid", false,-1);
        vcdp->declBus(c+1809,"VX_cache genblk5[3] bank core_rsp_tid", false,-1, 1,0);
        vcdp->declBus(c+1817,"VX_cache genblk5[3] bank core_rsp_data", false,-1, 31,0);
        vcdp->declQuad(c+1825,"VX_cache genblk5[3] bank core_rsp_tag", false,-1, 41,0);
        vcdp->declBit(c+137,"VX_cache genblk5[3] bank core_rsp_ready", false,-1);
        vcdp->declBit(c+1841,"VX_cache genblk5[3] bank dram_fill_req_valid", false,-1);
        vcdp->declBus(c+10209,"VX_cache genblk5[3] bank dram_fill_req_addr", false,-1, 25,0);
        vcdp->declBit(c+10065,"VX_cache genblk5[3] bank dram_fill_req_ready", false,-1);
        vcdp->declBit(c+25025,"VX_cache genblk5[3] bank dram_fill_rsp_valid", false,-1);
        vcdp->declArray(c+24769,"VX_cache genblk5[3] bank dram_fill_rsp_data", false,-1, 127,0);
        vcdp->declBus(c+24969,"VX_cache genblk5[3] bank dram_fill_rsp_addr", false,-1, 25,0);
        vcdp->declBit(c+10201,"VX_cache genblk5[3] bank dram_fill_rsp_ready", false,-1);
        vcdp->declBit(c+1849,"VX_cache genblk5[3] bank dram_wb_req_valid", false,-1);
        vcdp->declBus(c+1857,"VX_cache genblk5[3] bank dram_wb_req_byteen", false,-1, 15,0);
        vcdp->declBus(c+1865,"VX_cache genblk5[3] bank dram_wb_req_addr", false,-1, 25,0);
        vcdp->declArray(c+1873,"VX_cache genblk5[3] bank dram_wb_req_data", false,-1, 127,0);
        vcdp->declBit(c+145,"VX_cache genblk5[3] bank dram_wb_req_ready", false,-1);
        vcdp->declBit(c+25033,"VX_cache genblk5[3] bank snp_req_valid", false,-1);
        vcdp->declBus(c+24985,"VX_cache genblk5[3] bank snp_req_addr", false,-1, 25,0);
        vcdp->declBit(c+24833,"VX_cache genblk5[3] bank snp_req_invalidate", false,-1);
        vcdp->declBus(c+24841,"VX_cache genblk5[3] bank snp_req_tag", false,-1, 27,0);
        vcdp->declBit(c+10217,"VX_cache genblk5[3] bank snp_req_ready", false,-1);
        vcdp->declBit(c+1905,"VX_cache genblk5[3] bank snp_rsp_valid", false,-1);
        vcdp->declBus(c+1913,"VX_cache genblk5[3] bank snp_rsp_tag", false,-1, 27,0);
        vcdp->declBit(c+153,"VX_cache genblk5[3] bank snp_rsp_ready", false,-1);
        vcdp->declBit(c+8137,"VX_cache genblk5[3] bank snrq_pop", false,-1);
        vcdp->declBit(c+21017,"VX_cache genblk5[3] bank snrq_empty", false,-1);
        vcdp->declBit(c+21025,"VX_cache genblk5[3] bank snrq_full", false,-1);
        vcdp->declBus(c+8145,"VX_cache genblk5[3] bank snrq_addr_st0", false,-1, 25,0);
        vcdp->declBit(c+8153,"VX_cache genblk5[3] bank snrq_invalidate_st0", false,-1);
        vcdp->declBus(c+8161,"VX_cache genblk5[3] bank snrq_tag_st0", false,-1, 27,0);
        vcdp->declBit(c+8169,"VX_cache genblk5[3] bank dfpq_pop", false,-1);
        vcdp->declBit(c+21033,"VX_cache genblk5[3] bank dfpq_empty", false,-1);
        vcdp->declBit(c+21041,"VX_cache genblk5[3] bank dfpq_full", false,-1);
        vcdp->declBus(c+8177,"VX_cache genblk5[3] bank dfpq_addr_st0", false,-1, 25,0);
        vcdp->declArray(c+8185,"VX_cache genblk5[3] bank dfpq_filldata_st0", false,-1, 127,0);
        vcdp->declBit(c+8217,"VX_cache genblk5[3] bank reqq_pop", false,-1);
        vcdp->declBit(c+1057,"VX_cache genblk5[3] bank reqq_push", false,-1);
        vcdp->declBit(c+8225,"VX_cache genblk5[3] bank reqq_empty", false,-1);
        vcdp->declBit(c+21049,"VX_cache genblk5[3] bank reqq_full", false,-1);
        vcdp->declBit(c+8233,"VX_cache genblk5[3] bank reqq_req_st0", false,-1);
        vcdp->declBus(c+8241,"VX_cache genblk5[3] bank reqq_req_tid_st0", false,-1, 1,0);
        vcdp->declBit(c+8249,"VX_cache genblk5[3] bank reqq_req_rw_st0", false,-1);
        vcdp->declBus(c+8257,"VX_cache genblk5[3] bank reqq_req_byteen_st0", false,-1, 3,0);
        vcdp->declBus(c+8265,"VX_cache genblk5[3] bank reqq_req_addr_st0", false,-1, 29,0);
        vcdp->declBus(c+8273,"VX_cache genblk5[3] bank reqq_req_writeword_st0", false,-1, 31,0);
        vcdp->declQuad(c+21057,"VX_cache genblk5[3] bank reqq_req_tag_st0", false,-1, 41,0);
        vcdp->declBit(c+8281,"VX_cache genblk5[3] bank mrvq_pop", false,-1);
        vcdp->declBit(c+21073,"VX_cache genblk5[3] bank mrvq_full", false,-1);
        vcdp->declBit(c+21081,"VX_cache genblk5[3] bank mrvq_stop", false,-1);
        vcdp->declBit(c+8289,"VX_cache genblk5[3] bank mrvq_valid_st0", false,-1);
        vcdp->declBus(c+21089,"VX_cache genblk5[3] bank mrvq_tid_st0", false,-1, 1,0);
        vcdp->declBus(c+21097,"VX_cache genblk5[3] bank mrvq_addr_st0", false,-1, 25,0);
        vcdp->declBus(c+21105,"VX_cache genblk5[3] bank mrvq_wsel_st0", false,-1, 1,0);
        vcdp->declBus(c+21113,"VX_cache genblk5[3] bank mrvq_writeword_st0", false,-1, 31,0);
        vcdp->declQuad(c+21121,"VX_cache genblk5[3] bank mrvq_tag_st0", false,-1, 41,0);
        vcdp->declBit(c+8297,"VX_cache genblk5[3] bank mrvq_rw_st0", false,-1);
        vcdp->declBus(c+21137,"VX_cache genblk5[3] bank mrvq_byteen_st0", false,-1, 3,0);
        vcdp->declBit(c+21145,"VX_cache genblk5[3] bank mrvq_is_snp_st0", false,-1);
        vcdp->declBit(c+21153,"VX_cache genblk5[3] bank mrvq_snp_invalidate_st0", false,-1);
        vcdp->declBit(c+8305,"VX_cache genblk5[3] bank mrvq_pending_hazard_st1e", false,-1);
        vcdp->declBit(c+8313,"VX_cache genblk5[3] bank st2_pending_hazard_st1e", false,-1);
        vcdp->declBit(c+8321,"VX_cache genblk5[3] bank force_request_miss_st1e", false,-1);
        vcdp->declBus(c+21161,"VX_cache genblk5[3] bank miss_add_tid", false,-1, 1,0);
        vcdp->declQuad(c+21169,"VX_cache genblk5[3] bank miss_add_tag", false,-1, 41,0);
        vcdp->declBit(c+21185,"VX_cache genblk5[3] bank miss_add_rw", false,-1);
        vcdp->declBus(c+21193,"VX_cache genblk5[3] bank miss_add_byteen", false,-1, 3,0);
        vcdp->declBus(c+10209,"VX_cache genblk5[3] bank addr_st2", false,-1, 25,0);
        vcdp->declBit(c+21201,"VX_cache genblk5[3] bank is_fill_st2", false,-1);
        vcdp->declBit(c+8329,"VX_cache genblk5[3] bank recover_mrvq_state_st2", false,-1);
        vcdp->declBit(c+8337,"VX_cache genblk5[3] bank mrvq_push_stall", false,-1);
        vcdp->declBit(c+8345,"VX_cache genblk5[3] bank cwbq_push_stall", false,-1);
        vcdp->declBit(c+8353,"VX_cache genblk5[3] bank dwbq_push_stall", false,-1);
        vcdp->declBit(c+8361,"VX_cache genblk5[3] bank dram_fill_req_stall", false,-1);
        vcdp->declBit(c+8369,"VX_cache genblk5[3] bank stall_bank_pipe", false,-1);
        vcdp->declBit(c+8377,"VX_cache genblk5[3] bank is_fill_in_pipe", false,-1);
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+8385+i*1,"VX_cache genblk5[3] bank is_fill_st1", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+8393+i*1,"VX_cache genblk5[3] bank going_to_write_st1", true,(i+0));}}
        vcdp->declBus(c+25161,"VX_cache genblk5[3] bank j", false,-1, 31,0);
        vcdp->declBit(c+8289,"VX_cache genblk5[3] bank mrvq_pop_unqual", false,-1);
        vcdp->declBit(c+8401,"VX_cache genblk5[3] bank dfpq_pop_unqual", false,-1);
        vcdp->declBit(c+8409,"VX_cache genblk5[3] bank reqq_pop_unqual", false,-1);
        vcdp->declBit(c+8417,"VX_cache genblk5[3] bank snrq_pop_unqual", false,-1);
        vcdp->declBit(c+8401,"VX_cache genblk5[3] bank qual_is_fill_st0", false,-1);
        vcdp->declBit(c+8425,"VX_cache genblk5[3] bank qual_valid_st0", false,-1);
        vcdp->declBus(c+8433,"VX_cache genblk5[3] bank qual_addr_st0", false,-1, 25,0);
        vcdp->declBus(c+8441,"VX_cache genblk5[3] bank qual_wsel_st0", false,-1, 1,0);
        vcdp->declBit(c+8289,"VX_cache genblk5[3] bank qual_is_mrvq_st0", false,-1);
        vcdp->declBus(c+8449,"VX_cache genblk5[3] bank qual_writeword_st0", false,-1, 31,0);
        vcdp->declArray(c+8457,"VX_cache genblk5[3] bank qual_writedata_st0", false,-1, 127,0);
        vcdp->declQuad(c+8489,"VX_cache genblk5[3] bank qual_inst_meta_st0", false,-1, 48,0);
        vcdp->declBit(c+8505,"VX_cache genblk5[3] bank qual_going_to_write_st0", false,-1);
        vcdp->declBit(c+8513,"VX_cache genblk5[3] bank qual_is_snp_st0", false,-1);
        vcdp->declBit(c+8521,"VX_cache genblk5[3] bank qual_snp_invalidate_st0", false,-1);
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+8529+i*1,"VX_cache genblk5[3] bank valid_st1", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+8537+i*1,"VX_cache genblk5[3] bank addr_st1", true,(i+0), 25,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+8545+i*1,"VX_cache genblk5[3] bank wsel_st1", true,(i+0), 1,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+8553+i*1,"VX_cache genblk5[3] bank writeword_st1", true,(i+0), 31,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declQuad(c+8561+i*2,"VX_cache genblk5[3] bank inst_meta_st1", true,(i+0), 48,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declArray(c+8577+i*4,"VX_cache genblk5[3] bank writedata_st1", true,(i+0), 127,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+8609+i*1,"VX_cache genblk5[3] bank is_snp_st1", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+8617+i*1,"VX_cache genblk5[3] bank snp_invalidate_st1", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+8625+i*1,"VX_cache genblk5[3] bank is_mrvq_st1", true,(i+0));}}
        vcdp->declBus(c+8633,"VX_cache genblk5[3] bank readword_st1e", false,-1, 31,0);
        vcdp->declArray(c+8641,"VX_cache genblk5[3] bank readdata_st1e", false,-1, 127,0);
        vcdp->declBus(c+8673,"VX_cache genblk5[3] bank readtag_st1e", false,-1, 19,0);
        vcdp->declBit(c+8681,"VX_cache genblk5[3] bank miss_st1e", false,-1);
        vcdp->declBit(c+8689,"VX_cache genblk5[3] bank dirty_st1e", false,-1);
        vcdp->declBus(c+8697,"VX_cache genblk5[3] bank dirtyb_st1e", false,-1, 15,0);
        vcdp->declQuad(c+8705,"VX_cache genblk5[3] bank tag_st1e", false,-1, 41,0);
        vcdp->declBus(c+8721,"VX_cache genblk5[3] bank tid_st1e", false,-1, 1,0);
        vcdp->declBit(c+8729,"VX_cache genblk5[3] bank mem_rw_st1e", false,-1);
        vcdp->declBus(c+8737,"VX_cache genblk5[3] bank mem_byteen_st1e", false,-1, 3,0);
        vcdp->declBit(c+8745,"VX_cache genblk5[3] bank fill_saw_dirty_st1e", false,-1);
        vcdp->declBit(c+8753,"VX_cache genblk5[3] bank is_snp_st1e", false,-1);
        vcdp->declBit(c+8761,"VX_cache genblk5[3] bank snp_invalidate_st1e", false,-1);
        vcdp->declBit(c+8769,"VX_cache genblk5[3] bank snp_to_mrvq_st1e", false,-1);
        vcdp->declBit(c+8777,"VX_cache genblk5[3] bank mrvq_init_ready_state_st1e", false,-1);
        vcdp->declBit(c+8785,"VX_cache genblk5[3] bank miss_add_because_miss", false,-1);
        vcdp->declBit(c+8793,"VX_cache genblk5[3] bank valid_st1e", false,-1);
        vcdp->declBit(c+8801,"VX_cache genblk5[3] bank is_mrvq_st1e", false,-1);
        vcdp->declBit(c+8809,"VX_cache genblk5[3] bank mrvq_recover_ready_state_st1e", false,-1);
        vcdp->declBus(c+8817,"VX_cache genblk5[3] bank addr_st1e", false,-1, 25,0);
        vcdp->declBit(c+8825,"VX_cache genblk5[3] bank qual_valid_st1e_2", false,-1);
        vcdp->declBit(c+8801,"VX_cache genblk5[3] bank is_mrvq_st1e_st2", false,-1);
        vcdp->declBit(c+21209,"VX_cache genblk5[3] bank valid_st2", false,-1);
        vcdp->declBus(c+21217,"VX_cache genblk5[3] bank wsel_st2", false,-1, 1,0);
        vcdp->declBus(c+21225,"VX_cache genblk5[3] bank writeword_st2", false,-1, 31,0);
        vcdp->declBus(c+21233,"VX_cache genblk5[3] bank readword_st2", false,-1, 31,0);
        vcdp->declArray(c+21241,"VX_cache genblk5[3] bank readdata_st2", false,-1, 127,0);
        vcdp->declBit(c+21273,"VX_cache genblk5[3] bank miss_st2", false,-1);
        vcdp->declBit(c+21281,"VX_cache genblk5[3] bank dirty_st2", false,-1);
        vcdp->declBus(c+21289,"VX_cache genblk5[3] bank dirtyb_st2", false,-1, 15,0);
        vcdp->declQuad(c+21297,"VX_cache genblk5[3] bank inst_meta_st2", false,-1, 48,0);
        vcdp->declBus(c+21313,"VX_cache genblk5[3] bank readtag_st2", false,-1, 19,0);
        vcdp->declBit(c+21321,"VX_cache genblk5[3] bank fill_saw_dirty_st2", false,-1);
        vcdp->declBit(c+21329,"VX_cache genblk5[3] bank is_snp_st2", false,-1);
        vcdp->declBit(c+21337,"VX_cache genblk5[3] bank snp_invalidate_st2", false,-1);
        vcdp->declBit(c+21345,"VX_cache genblk5[3] bank snp_to_mrvq_st2", false,-1);
        vcdp->declBit(c+21353,"VX_cache genblk5[3] bank is_mrvq_st2", false,-1);
        vcdp->declBit(c+8833,"VX_cache genblk5[3] bank mrvq_init_ready_state_st2", false,-1);
        vcdp->declBit(c+21361,"VX_cache genblk5[3] bank mrvq_recover_ready_state_st2", false,-1);
        vcdp->declBit(c+21369,"VX_cache genblk5[3] bank mrvq_init_ready_state_unqual_st2", false,-1);
        vcdp->declBit(c+8841,"VX_cache genblk5[3] bank mrvq_init_ready_state_hazard_st0_st1", false,-1);
        vcdp->declBit(c+8849,"VX_cache genblk5[3] bank mrvq_init_ready_state_hazard_st1e_st1", false,-1);
        vcdp->declBit(c+21345,"VX_cache genblk5[3] bank miss_add_because_pending", false,-1);
        vcdp->declBit(c+8857,"VX_cache genblk5[3] bank miss_add_unqual", false,-1);
        vcdp->declBit(c+8865,"VX_cache genblk5[3] bank miss_add", false,-1);
        vcdp->declBus(c+10209,"VX_cache genblk5[3] bank miss_add_addr", false,-1, 25,0);
        vcdp->declBus(c+21217,"VX_cache genblk5[3] bank miss_add_wsel", false,-1, 1,0);
        vcdp->declBus(c+21225,"VX_cache genblk5[3] bank miss_add_data", false,-1, 31,0);
        vcdp->declBit(c+21329,"VX_cache genblk5[3] bank miss_add_is_snp", false,-1);
        vcdp->declBit(c+21337,"VX_cache genblk5[3] bank miss_add_snp_invalidate", false,-1);
        vcdp->declBit(c+8873,"VX_cache genblk5[3] bank miss_add_is_mrvq", false,-1);
        vcdp->declBit(c+8881,"VX_cache genblk5[3] bank cwbq_push", false,-1);
        vcdp->declBit(c+1065,"VX_cache genblk5[3] bank cwbq_pop", false,-1);
        vcdp->declBit(c+21377,"VX_cache genblk5[3] bank cwbq_empty", false,-1);
        vcdp->declBit(c+21385,"VX_cache genblk5[3] bank cwbq_full", false,-1);
        vcdp->declBit(c+8889,"VX_cache genblk5[3] bank cwbq_push_unqual", false,-1);
        vcdp->declBus(c+21233,"VX_cache genblk5[3] bank cwbq_data", false,-1, 31,0);
        vcdp->declBus(c+21161,"VX_cache genblk5[3] bank cwbq_tid", false,-1, 1,0);
        vcdp->declQuad(c+21169,"VX_cache genblk5[3] bank cwbq_tag", false,-1, 41,0);
        vcdp->declBit(c+8857,"VX_cache genblk5[3] bank dram_fill_req_fast", false,-1);
        vcdp->declBit(c+8897,"VX_cache genblk5[3] bank dram_fill_req_unqual", false,-1);
        vcdp->declBit(c+8905,"VX_cache genblk5[3] bank dwbq_push", false,-1);
        vcdp->declBit(c+1073,"VX_cache genblk5[3] bank dwbq_pop", false,-1);
        vcdp->declBit(c+21393,"VX_cache genblk5[3] bank dwbq_empty", false,-1);
        vcdp->declBit(c+21401,"VX_cache genblk5[3] bank dwbq_full", false,-1);
        vcdp->declBit(c+8913,"VX_cache genblk5[3] bank dwbq_is_dwb_in", false,-1);
        vcdp->declBit(c+8921,"VX_cache genblk5[3] bank dwbq_is_snp_in", false,-1);
        vcdp->declBit(c+8929,"VX_cache genblk5[3] bank dwbq_is_dwb_out", false,-1);
        vcdp->declBit(c+8937,"VX_cache genblk5[3] bank dwbq_is_snp_out", false,-1);
        vcdp->declBit(c+8945,"VX_cache genblk5[3] bank dwbq_push_unqual", false,-1);
        vcdp->declBus(c+21409,"VX_cache genblk5[3] bank dwbq_req_addr", false,-1, 25,0);
        vcdp->declBus(c+21417,"VX_cache genblk5[3] bank snrq_tag_st2", false,-1, 27,0);
        vcdp->declBit(c+761,"VX_cache genblk5[3] bank dram_wb_req_fire", false,-1);
        vcdp->declBit(c+769,"VX_cache genblk5[3] bank snp_rsp_fire", false,-1);
        vcdp->declBit(c+21425,"VX_cache genblk5[3] bank dwbq_dual_valid_sel", false,-1);
        vcdp->declBus(c+25169,"VX_cache genblk5[3] bank snp_req_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[3] bank snp_req_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[3] bank snp_req_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[3] bank snp_req_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[3] bank snp_req_queue reset", false,-1);
        vcdp->declBit(c+25033,"VX_cache genblk5[3] bank snp_req_queue push", false,-1);
        vcdp->declBit(c+8137,"VX_cache genblk5[3] bank snp_req_queue pop", false,-1);
        vcdp->declQuad(c+777,"VX_cache genblk5[3] bank snp_req_queue data_in", false,-1, 54,0);
        vcdp->declQuad(c+8953,"VX_cache genblk5[3] bank snp_req_queue data_out", false,-1, 54,0);
        vcdp->declBit(c+21017,"VX_cache genblk5[3] bank snp_req_queue empty", false,-1);
        vcdp->declBit(c+21025,"VX_cache genblk5[3] bank snp_req_queue full", false,-1);
        vcdp->declBus(c+21433,"VX_cache genblk5[3] bank snp_req_queue size", false,-1, 4,0);
        vcdp->declBus(c+21433,"VX_cache genblk5[3] bank snp_req_queue size_r", false,-1, 4,0);
        vcdp->declBit(c+8969,"VX_cache genblk5[3] bank snp_req_queue reading", false,-1);
        vcdp->declBit(c+793,"VX_cache genblk5[3] bank snp_req_queue writing", false,-1);
        {int i; for (i=0; i<16; i++) {
                vcdp->declQuad(c+21441+i*2,"VX_cache genblk5[3] bank snp_req_queue genblk3 data", true,(i+0), 54,0);}}
        vcdp->declQuad(c+21697,"VX_cache genblk5[3] bank snp_req_queue genblk3 genblk2 head_r", false,-1, 54,0);
        vcdp->declQuad(c+21713,"VX_cache genblk5[3] bank snp_req_queue genblk3 genblk2 curr_r", false,-1, 54,0);
        vcdp->declBus(c+21729,"VX_cache genblk5[3] bank snp_req_queue genblk3 genblk2 wr_ptr_r", false,-1, 3,0);
        vcdp->declBus(c+21737,"VX_cache genblk5[3] bank snp_req_queue genblk3 genblk2 rd_ptr_r", false,-1, 3,0);
        vcdp->declBus(c+21745,"VX_cache genblk5[3] bank snp_req_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 3,0);
        vcdp->declBit(c+21017,"VX_cache genblk5[3] bank snp_req_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+21025,"VX_cache genblk5[3] bank snp_req_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+21753,"VX_cache genblk5[3] bank snp_req_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25177,"VX_cache genblk5[3] bank dfp_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[3] bank dfp_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[3] bank dfp_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[3] bank dfp_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[3] bank dfp_queue reset", false,-1);
        vcdp->declBit(c+25025,"VX_cache genblk5[3] bank dfp_queue push", false,-1);
        vcdp->declBit(c+8169,"VX_cache genblk5[3] bank dfp_queue pop", false,-1);
        vcdp->declArray(c+801,"VX_cache genblk5[3] bank dfp_queue data_in", false,-1, 153,0);
        vcdp->declArray(c+8977,"VX_cache genblk5[3] bank dfp_queue data_out", false,-1, 153,0);
        vcdp->declBit(c+21033,"VX_cache genblk5[3] bank dfp_queue empty", false,-1);
        vcdp->declBit(c+21041,"VX_cache genblk5[3] bank dfp_queue full", false,-1);
        vcdp->declBus(c+21761,"VX_cache genblk5[3] bank dfp_queue size", false,-1, 4,0);
        vcdp->declBus(c+21761,"VX_cache genblk5[3] bank dfp_queue size_r", false,-1, 4,0);
        vcdp->declBit(c+9017,"VX_cache genblk5[3] bank dfp_queue reading", false,-1);
        vcdp->declBit(c+841,"VX_cache genblk5[3] bank dfp_queue writing", false,-1);
        {int i; for (i=0; i<16; i++) {
                vcdp->declArray(c+21769+i*5,"VX_cache genblk5[3] bank dfp_queue genblk3 data", true,(i+0), 153,0);}}
        vcdp->declArray(c+22409,"VX_cache genblk5[3] bank dfp_queue genblk3 genblk2 head_r", false,-1, 153,0);
        vcdp->declArray(c+22449,"VX_cache genblk5[3] bank dfp_queue genblk3 genblk2 curr_r", false,-1, 153,0);
        vcdp->declBus(c+22489,"VX_cache genblk5[3] bank dfp_queue genblk3 genblk2 wr_ptr_r", false,-1, 3,0);
        vcdp->declBus(c+22497,"VX_cache genblk5[3] bank dfp_queue genblk3 genblk2 rd_ptr_r", false,-1, 3,0);
        vcdp->declBus(c+22505,"VX_cache genblk5[3] bank dfp_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 3,0);
        vcdp->declBit(c+21033,"VX_cache genblk5[3] bank dfp_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+21041,"VX_cache genblk5[3] bank dfp_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+22513,"VX_cache genblk5[3] bank dfp_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank core_req_arb WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank core_req_arb NUM_REQUESTS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank core_req_arb CREQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25097,"VX_cache genblk5[3] bank core_req_arb CORE_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBus(c+25105,"VX_cache genblk5[3] bank core_req_arb CORE_TAG_ID_BITS", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[3] bank core_req_arb clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[3] bank core_req_arb reset", false,-1);
        vcdp->declBit(c+1057,"VX_cache genblk5[3] bank core_req_arb reqq_push", false,-1);
        vcdp->declBus(c+129,"VX_cache genblk5[3] bank core_req_arb bank_valids", false,-1, 3,0);
        vcdp->declBus(c+24513,"VX_cache genblk5[3] bank core_req_arb bank_rw", false,-1, 3,0);
        vcdp->declBus(c+24521,"VX_cache genblk5[3] bank core_req_arb bank_byteen", false,-1, 15,0);
        vcdp->declArray(c+24561,"VX_cache genblk5[3] bank core_req_arb bank_writedata", false,-1, 127,0);
        vcdp->declArray(c+24529,"VX_cache genblk5[3] bank core_req_arb bank_addr", false,-1, 119,0);
        vcdp->declQuad(c+24593,"VX_cache genblk5[3] bank core_req_arb bank_tag", false,-1, 41,0);
        vcdp->declBit(c+8217,"VX_cache genblk5[3] bank core_req_arb reqq_pop", false,-1);
        vcdp->declBit(c+8233,"VX_cache genblk5[3] bank core_req_arb reqq_req_st0", false,-1);
        vcdp->declBus(c+8241,"VX_cache genblk5[3] bank core_req_arb reqq_req_tid_st0", false,-1, 1,0);
        vcdp->declBit(c+8249,"VX_cache genblk5[3] bank core_req_arb reqq_req_rw_st0", false,-1);
        vcdp->declBus(c+8257,"VX_cache genblk5[3] bank core_req_arb reqq_req_byteen_st0", false,-1, 3,0);
        vcdp->declBus(c+8265,"VX_cache genblk5[3] bank core_req_arb reqq_req_addr_st0", false,-1, 29,0);
        vcdp->declBus(c+8273,"VX_cache genblk5[3] bank core_req_arb reqq_req_writedata_st0", false,-1, 31,0);
        vcdp->declQuad(c+21057,"VX_cache genblk5[3] bank core_req_arb reqq_req_tag_st0", false,-1, 41,0);
        vcdp->declBit(c+8225,"VX_cache genblk5[3] bank core_req_arb reqq_empty", false,-1);
        vcdp->declBit(c+21049,"VX_cache genblk5[3] bank core_req_arb reqq_full", false,-1);
        vcdp->declBus(c+9025,"VX_cache genblk5[3] bank core_req_arb out_per_valids", false,-1, 3,0);
        vcdp->declBus(c+9033,"VX_cache genblk5[3] bank core_req_arb out_per_rw", false,-1, 3,0);
        vcdp->declBus(c+9041,"VX_cache genblk5[3] bank core_req_arb out_per_byteen", false,-1, 15,0);
        vcdp->declArray(c+9049,"VX_cache genblk5[3] bank core_req_arb out_per_addr", false,-1, 119,0);
        vcdp->declArray(c+9081,"VX_cache genblk5[3] bank core_req_arb out_per_writedata", false,-1, 127,0);
        vcdp->declQuad(c+9113,"VX_cache genblk5[3] bank core_req_arb out_per_tag", false,-1, 41,0);
        vcdp->declBus(c+22521,"VX_cache genblk5[3] bank core_req_arb use_per_valids", false,-1, 3,0);
        vcdp->declBus(c+22529,"VX_cache genblk5[3] bank core_req_arb use_per_rw", false,-1, 3,0);
        vcdp->declBus(c+22537,"VX_cache genblk5[3] bank core_req_arb use_per_byteen", false,-1, 15,0);
        vcdp->declArray(c+22545,"VX_cache genblk5[3] bank core_req_arb use_per_addr", false,-1, 119,0);
        vcdp->declArray(c+22577,"VX_cache genblk5[3] bank core_req_arb use_per_writedata", false,-1, 127,0);
        vcdp->declQuad(c+21057,"VX_cache genblk5[3] bank core_req_arb use_per_tag", false,-1, 41,0);
        vcdp->declBus(c+22521,"VX_cache genblk5[3] bank core_req_arb qual_valids", false,-1, 3,0);
        vcdp->declBus(c+22529,"VX_cache genblk5[3] bank core_req_arb qual_rw", false,-1, 3,0);
        vcdp->declBus(c+22537,"VX_cache genblk5[3] bank core_req_arb qual_byteen", false,-1, 15,0);
        vcdp->declArray(c+22545,"VX_cache genblk5[3] bank core_req_arb qual_addr", false,-1, 119,0);
        vcdp->declArray(c+22577,"VX_cache genblk5[3] bank core_req_arb qual_writedata", false,-1, 127,0);
        vcdp->declQuad(c+21057,"VX_cache genblk5[3] bank core_req_arb qual_tag", false,-1, 41,0);
        vcdp->declBit(c+22609,"VX_cache genblk5[3] bank core_req_arb o_empty", false,-1);
        vcdp->declBit(c+22617,"VX_cache genblk5[3] bank core_req_arb use_empty", false,-1);
        vcdp->declBit(c+9129,"VX_cache genblk5[3] bank core_req_arb out_empty", false,-1);
        vcdp->declBit(c+1081,"VX_cache genblk5[3] bank core_req_arb push_qual", false,-1);
        vcdp->declBit(c+9137,"VX_cache genblk5[3] bank core_req_arb pop_qual", false,-1);
        vcdp->declBus(c+9145,"VX_cache genblk5[3] bank core_req_arb real_out_per_valids", false,-1, 3,0);
        vcdp->declBus(c+8241,"VX_cache genblk5[3] bank core_req_arb qual_request_index", false,-1, 1,0);
        vcdp->declBit(c+8233,"VX_cache genblk5[3] bank core_req_arb qual_has_request", false,-1);
        vcdp->declBus(c+25185,"VX_cache genblk5[3] bank core_req_arb reqq_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank core_req_arb reqq_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[3] bank core_req_arb reqq_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[3] bank core_req_arb reqq_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[3] bank core_req_arb reqq_queue reset", false,-1);
        vcdp->declBit(c+1081,"VX_cache genblk5[3] bank core_req_arb reqq_queue push", false,-1);
        vcdp->declBit(c+9137,"VX_cache genblk5[3] bank core_req_arb reqq_queue pop", false,-1);
        vcdp->declArray(c+849,"VX_cache genblk5[3] bank core_req_arb reqq_queue data_in", false,-1, 313,0);
        vcdp->declArray(c+9153,"VX_cache genblk5[3] bank core_req_arb reqq_queue data_out", false,-1, 313,0);
        vcdp->declBit(c+22609,"VX_cache genblk5[3] bank core_req_arb reqq_queue empty", false,-1);
        vcdp->declBit(c+21049,"VX_cache genblk5[3] bank core_req_arb reqq_queue full", false,-1);
        vcdp->declBus(c+22625,"VX_cache genblk5[3] bank core_req_arb reqq_queue size", false,-1, 2,0);
        vcdp->declBus(c+22625,"VX_cache genblk5[3] bank core_req_arb reqq_queue size_r", false,-1, 2,0);
        vcdp->declBit(c+9233,"VX_cache genblk5[3] bank core_req_arb reqq_queue reading", false,-1);
        vcdp->declBit(c+929,"VX_cache genblk5[3] bank core_req_arb reqq_queue writing", false,-1);
        {int i; for (i=0; i<4; i++) {
                vcdp->declArray(c+22633+i*10,"VX_cache genblk5[3] bank core_req_arb reqq_queue genblk3 data", true,(i+0), 313,0);}}
        vcdp->declArray(c+22953,"VX_cache genblk5[3] bank core_req_arb reqq_queue genblk3 genblk2 head_r", false,-1, 313,0);
        vcdp->declArray(c+23033,"VX_cache genblk5[3] bank core_req_arb reqq_queue genblk3 genblk2 curr_r", false,-1, 313,0);
        vcdp->declBus(c+23113,"VX_cache genblk5[3] bank core_req_arb reqq_queue genblk3 genblk2 wr_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+23121,"VX_cache genblk5[3] bank core_req_arb reqq_queue genblk3 genblk2 rd_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+23129,"VX_cache genblk5[3] bank core_req_arb reqq_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 1,0);
        vcdp->declBit(c+22609,"VX_cache genblk5[3] bank core_req_arb reqq_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+21049,"VX_cache genblk5[3] bank core_req_arb reqq_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+23137,"VX_cache genblk5[3] bank core_req_arb reqq_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank core_req_arb sel_bank N", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[3] bank core_req_arb sel_bank clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[3] bank core_req_arb sel_bank reset", false,-1);
        vcdp->declBus(c+22521,"VX_cache genblk5[3] bank core_req_arb sel_bank requests", false,-1, 3,0);
        vcdp->declBus(c+8241,"VX_cache genblk5[3] bank core_req_arb sel_bank grant_index", false,-1, 1,0);
        vcdp->declBus(c+9241,"VX_cache genblk5[3] bank core_req_arb sel_bank grant_onehot", false,-1, 3,0);
        vcdp->declBit(c+8233,"VX_cache genblk5[3] bank core_req_arb sel_bank grant_valid", false,-1);
        vcdp->declBus(c+9241,"VX_cache genblk5[3] bank core_req_arb sel_bank genblk2 grant_onehot_r", false,-1, 3,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank core_req_arb sel_bank genblk2 priority_encoder N", false,-1, 31,0);
        vcdp->declBus(c+22521,"VX_cache genblk5[3] bank core_req_arb sel_bank genblk2 priority_encoder data_in", false,-1, 3,0);
        vcdp->declBus(c+8241,"VX_cache genblk5[3] bank core_req_arb sel_bank genblk2 priority_encoder data_out", false,-1, 1,0);
        vcdp->declBit(c+8233,"VX_cache genblk5[3] bank core_req_arb sel_bank genblk2 priority_encoder valid_out", false,-1);
        vcdp->declBus(c+9249,"VX_cache genblk5[3] bank core_req_arb sel_bank genblk2 priority_encoder i", false,-1, 31,0);
        vcdp->declBus(c+25193,"VX_cache genblk5[3] bank s0_1_c0 N", false,-1, 31,0);
        vcdp->declBus(c+25057,"VX_cache genblk5[3] bank s0_1_c0 PASSTHRU", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[3] bank s0_1_c0 clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[3] bank s0_1_c0 reset", false,-1);
        vcdp->declBit(c+8369,"VX_cache genblk5[3] bank s0_1_c0 stall", false,-1);
        vcdp->declBit(c+25137,"VX_cache genblk5[3] bank s0_1_c0 flush", false,-1);
        vcdp->declArray(c+9257,"VX_cache genblk5[3] bank s0_1_c0 in", false,-1, 242,0);
        vcdp->declArray(c+23145,"VX_cache genblk5[3] bank s0_1_c0 out", false,-1, 242,0);
        vcdp->declArray(c+23145,"VX_cache genblk5[3] bank s0_1_c0 value", false,-1, 242,0);
        vcdp->declBus(c+25065,"VX_cache genblk5[3] bank tag_data_access CACHE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[3] bank tag_data_access BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank tag_data_access NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank tag_data_access WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[3] bank tag_data_access STAGE_1_CYCLES", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[3] bank tag_data_access WRITE_ENABLE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[3] bank tag_data_access DRAM_ENABLE", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[3] bank tag_data_access clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[3] bank tag_data_access reset", false,-1);
        vcdp->declBit(c+8369,"VX_cache genblk5[3] bank tag_data_access stall", false,-1);
        vcdp->declBit(c+8753,"VX_cache genblk5[3] bank tag_data_access is_snp_st1e", false,-1);
        vcdp->declBit(c+8761,"VX_cache genblk5[3] bank tag_data_access snp_invalidate_st1e", false,-1);
        vcdp->declBit(c+8369,"VX_cache genblk5[3] bank tag_data_access stall_bank_pipe", false,-1);
        vcdp->declBit(c+8321,"VX_cache genblk5[3] bank tag_data_access force_request_miss_st1e", false,-1);
        vcdp->declBus(c+9321,"VX_cache genblk5[3] bank tag_data_access readaddr_st10", false,-1, 5,0);
        vcdp->declBus(c+8817,"VX_cache genblk5[3] bank tag_data_access writeaddr_st1e", false,-1, 25,0);
        vcdp->declBit(c+8793,"VX_cache genblk5[3] bank tag_data_access valid_req_st1e", false,-1);
        vcdp->declBit(c+9329,"VX_cache genblk5[3] bank tag_data_access writefill_st1e", false,-1);
        vcdp->declBus(c+9337,"VX_cache genblk5[3] bank tag_data_access writeword_st1e", false,-1, 31,0);
        vcdp->declArray(c+9345,"VX_cache genblk5[3] bank tag_data_access writedata_st1e", false,-1, 127,0);
        vcdp->declBit(c+8729,"VX_cache genblk5[3] bank tag_data_access mem_rw_st1e", false,-1);
        vcdp->declBus(c+8737,"VX_cache genblk5[3] bank tag_data_access mem_byteen_st1e", false,-1, 3,0);
        vcdp->declBus(c+9377,"VX_cache genblk5[3] bank tag_data_access wordsel_st1e", false,-1, 1,0);
        vcdp->declBus(c+8633,"VX_cache genblk5[3] bank tag_data_access readword_st1e", false,-1, 31,0);
        vcdp->declArray(c+8641,"VX_cache genblk5[3] bank tag_data_access readdata_st1e", false,-1, 127,0);
        vcdp->declBus(c+8673,"VX_cache genblk5[3] bank tag_data_access readtag_st1e", false,-1, 19,0);
        vcdp->declBit(c+8681,"VX_cache genblk5[3] bank tag_data_access miss_st1e", false,-1);
        vcdp->declBit(c+8689,"VX_cache genblk5[3] bank tag_data_access dirty_st1e", false,-1);
        vcdp->declBus(c+8697,"VX_cache genblk5[3] bank tag_data_access dirtyb_st1e", false,-1, 15,0);
        vcdp->declBit(c+8745,"VX_cache genblk5[3] bank tag_data_access fill_saw_dirty_st1e", false,-1);
        vcdp->declBit(c+8769,"VX_cache genblk5[3] bank tag_data_access snp_to_mrvq_st1e", false,-1);
        vcdp->declBit(c+8777,"VX_cache genblk5[3] bank tag_data_access mrvq_init_ready_state_st1e", false,-1);
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+9385+i*1,"VX_cache genblk5[3] bank tag_data_access read_valid_st1c", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBit(c+9393+i*1,"VX_cache genblk5[3] bank tag_data_access read_dirty_st1c", true,(i+0));}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+9401+i*1,"VX_cache genblk5[3] bank tag_data_access read_dirtyb_st1c", true,(i+0), 15,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declBus(c+9409+i*1,"VX_cache genblk5[3] bank tag_data_access read_tag_st1c", true,(i+0), 19,0);}}
        {int i; for (i=0; i<1; i++) {
                vcdp->declArray(c+9417+i*4,"VX_cache genblk5[3] bank tag_data_access read_data_st1c", true,(i+0), 127,0);}}
        vcdp->declBit(c+9449,"VX_cache genblk5[3] bank tag_data_access qual_read_valid_st1", false,-1);
        vcdp->declBit(c+9457,"VX_cache genblk5[3] bank tag_data_access qual_read_dirty_st1", false,-1);
        vcdp->declBus(c+9465,"VX_cache genblk5[3] bank tag_data_access qual_read_dirtyb_st1", false,-1, 15,0);
        vcdp->declBus(c+9473,"VX_cache genblk5[3] bank tag_data_access qual_read_tag_st1", false,-1, 19,0);
        vcdp->declArray(c+9481,"VX_cache genblk5[3] bank tag_data_access qual_read_data_st1", false,-1, 127,0);
        vcdp->declBit(c+9513,"VX_cache genblk5[3] bank tag_data_access use_read_valid_st1e", false,-1);
        vcdp->declBit(c+9521,"VX_cache genblk5[3] bank tag_data_access use_read_dirty_st1e", false,-1);
        vcdp->declBus(c+8697,"VX_cache genblk5[3] bank tag_data_access use_read_dirtyb_st1e", false,-1, 15,0);
        vcdp->declBus(c+8673,"VX_cache genblk5[3] bank tag_data_access use_read_tag_st1e", false,-1, 19,0);
        vcdp->declArray(c+8641,"VX_cache genblk5[3] bank tag_data_access use_read_data_st1e", false,-1, 127,0);
        vcdp->declBus(c+9529,"VX_cache genblk5[3] bank tag_data_access use_write_enable", false,-1, 15,0);
        vcdp->declArray(c+9537,"VX_cache genblk5[3] bank tag_data_access use_write_data", false,-1, 127,0);
        vcdp->declBit(c+8681,"VX_cache genblk5[3] bank tag_data_access fill_sent", false,-1);
        vcdp->declBit(c+9569,"VX_cache genblk5[3] bank tag_data_access invalidate_line", false,-1);
        vcdp->declBit(c+9577,"VX_cache genblk5[3] bank tag_data_access tags_match", false,-1);
        vcdp->declBit(c+9585,"VX_cache genblk5[3] bank tag_data_access real_writefill", false,-1);
        vcdp->declBus(c+9593,"VX_cache genblk5[3] bank tag_data_access writetag_st1e", false,-1, 19,0);
        vcdp->declBus(c+9321,"VX_cache genblk5[3] bank tag_data_access writeladdr_st1e", false,-1, 5,0);
        vcdp->declBus(c+9601,"VX_cache genblk5[3] bank tag_data_access we", false,-1, 15,0);
        vcdp->declArray(c+9537,"VX_cache genblk5[3] bank tag_data_access data_write", false,-1, 127,0);
        vcdp->declBit(c+9609,"VX_cache genblk5[3] bank tag_data_access should_write", false,-1);
        vcdp->declBit(c+9569,"VX_cache genblk5[3] bank tag_data_access snoop_hit_no_pending", false,-1);
        vcdp->declBit(c+9617,"VX_cache genblk5[3] bank tag_data_access req_invalid", false,-1);
        vcdp->declBit(c+9625,"VX_cache genblk5[3] bank tag_data_access req_miss", false,-1);
        vcdp->declBit(c+9633,"VX_cache genblk5[3] bank tag_data_access real_miss", false,-1);
        vcdp->declBit(c+9641,"VX_cache genblk5[3] bank tag_data_access force_core_miss", false,-1);
        vcdp->declBit(c+9649,"VX_cache genblk5[3] bank tag_data_access genblk4[0] normal_write", false,-1);
        vcdp->declBit(c+9657,"VX_cache genblk5[3] bank tag_data_access genblk4[1] normal_write", false,-1);
        vcdp->declBit(c+9665,"VX_cache genblk5[3] bank tag_data_access genblk4[2] normal_write", false,-1);
        vcdp->declBit(c+9673,"VX_cache genblk5[3] bank tag_data_access genblk4[3] normal_write", false,-1);
        vcdp->declBus(c+25065,"VX_cache genblk5[3] bank tag_data_access tag_data_structure CACHE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[3] bank tag_data_access tag_data_structure BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank tag_data_access tag_data_structure NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank tag_data_access tag_data_structure WORD_SIZE", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[3] bank tag_data_access tag_data_structure clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[3] bank tag_data_access tag_data_structure reset", false,-1);
        vcdp->declBit(c+8369,"VX_cache genblk5[3] bank tag_data_access tag_data_structure stall_bank_pipe", false,-1);
        vcdp->declBus(c+9321,"VX_cache genblk5[3] bank tag_data_access tag_data_structure read_addr", false,-1, 5,0);
        vcdp->declBit(c+9449,"VX_cache genblk5[3] bank tag_data_access tag_data_structure read_valid", false,-1);
        vcdp->declBit(c+9457,"VX_cache genblk5[3] bank tag_data_access tag_data_structure read_dirty", false,-1);
        vcdp->declBus(c+9465,"VX_cache genblk5[3] bank tag_data_access tag_data_structure read_dirtyb", false,-1, 15,0);
        vcdp->declBus(c+9473,"VX_cache genblk5[3] bank tag_data_access tag_data_structure read_tag", false,-1, 19,0);
        vcdp->declArray(c+9481,"VX_cache genblk5[3] bank tag_data_access tag_data_structure read_data", false,-1, 127,0);
        vcdp->declBit(c+9569,"VX_cache genblk5[3] bank tag_data_access tag_data_structure invalidate", false,-1);
        vcdp->declBus(c+9529,"VX_cache genblk5[3] bank tag_data_access tag_data_structure write_enable", false,-1, 15,0);
        vcdp->declBit(c+9585,"VX_cache genblk5[3] bank tag_data_access tag_data_structure write_fill", false,-1);
        vcdp->declBus(c+9321,"VX_cache genblk5[3] bank tag_data_access tag_data_structure write_addr", false,-1, 5,0);
        vcdp->declBus(c+9593,"VX_cache genblk5[3] bank tag_data_access tag_data_structure tag_index", false,-1, 19,0);
        vcdp->declArray(c+9537,"VX_cache genblk5[3] bank tag_data_access tag_data_structure write_data", false,-1, 127,0);
        vcdp->declBit(c+8681,"VX_cache genblk5[3] bank tag_data_access tag_data_structure fill_sent", false,-1);
        vcdp->declQuad(c+23209,"VX_cache genblk5[3] bank tag_data_access tag_data_structure dirty", false,-1, 63,0);
        vcdp->declQuad(c+23225,"VX_cache genblk5[3] bank tag_data_access tag_data_structure valid", false,-1, 63,0);
        vcdp->declBit(c+9681,"VX_cache genblk5[3] bank tag_data_access tag_data_structure do_write", false,-1);
        vcdp->declBus(c+23241,"VX_cache genblk5[3] bank tag_data_access tag_data_structure i", false,-1, 31,0);
        vcdp->declBus(c+23249,"VX_cache genblk5[3] bank tag_data_access tag_data_structure j", false,-1, 31,0);
        vcdp->declBus(c+25201,"VX_cache genblk5[3] bank tag_data_access s0_1_c0 N", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[3] bank tag_data_access s0_1_c0 PASSTHRU", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[3] bank tag_data_access s0_1_c0 clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[3] bank tag_data_access s0_1_c0 reset", false,-1);
        vcdp->declBit(c+8369,"VX_cache genblk5[3] bank tag_data_access s0_1_c0 stall", false,-1);
        vcdp->declBit(c+25137,"VX_cache genblk5[3] bank tag_data_access s0_1_c0 flush", false,-1);
        vcdp->declArray(c+9689,"VX_cache genblk5[3] bank tag_data_access s0_1_c0 in", false,-1, 165,0);
        vcdp->declArray(c+9689,"VX_cache genblk5[3] bank tag_data_access s0_1_c0 out", false,-1, 165,0);
        vcdp->declArray(c+23257,"VX_cache genblk5[3] bank tag_data_access s0_1_c0 value", false,-1, 165,0);
        vcdp->declBus(c+25209,"VX_cache genblk5[3] bank st_1e_2 N", false,-1, 31,0);
        vcdp->declBus(c+25057,"VX_cache genblk5[3] bank st_1e_2 PASSTHRU", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[3] bank st_1e_2 clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[3] bank st_1e_2 reset", false,-1);
        vcdp->declBit(c+8369,"VX_cache genblk5[3] bank st_1e_2 stall", false,-1);
        vcdp->declBit(c+25137,"VX_cache genblk5[3] bank st_1e_2 flush", false,-1);
        vcdp->declArray(c+9737,"VX_cache genblk5[3] bank st_1e_2 in", false,-1, 315,0);
        vcdp->declArray(c+23305,"VX_cache genblk5[3] bank st_1e_2 out", false,-1, 315,0);
        vcdp->declArray(c+23305,"VX_cache genblk5[3] bank st_1e_2 value", false,-1, 315,0);
        vcdp->declBus(c+25057,"VX_cache genblk5[3] bank cache_miss_resrv CACHE_ID", false,-1, 31,0);
        vcdp->declBus(c+25241,"VX_cache genblk5[3] bank cache_miss_resrv BANK_ID", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[3] bank cache_miss_resrv BANK_LINE_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank cache_miss_resrv NUM_BANKS", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank cache_miss_resrv WORD_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank cache_miss_resrv NUM_REQUESTS", false,-1, 31,0);
        vcdp->declBus(c+25073,"VX_cache genblk5[3] bank cache_miss_resrv MRVQ_SIZE", false,-1, 31,0);
        vcdp->declBus(c+25097,"VX_cache genblk5[3] bank cache_miss_resrv CORE_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBus(c+25113,"VX_cache genblk5[3] bank cache_miss_resrv SNP_REQ_TAG_WIDTH", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[3] bank cache_miss_resrv clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[3] bank cache_miss_resrv reset", false,-1);
        vcdp->declBit(c+8865,"VX_cache genblk5[3] bank cache_miss_resrv miss_add", false,-1);
        vcdp->declBit(c+8873,"VX_cache genblk5[3] bank cache_miss_resrv is_mrvq", false,-1);
        vcdp->declBus(c+10209,"VX_cache genblk5[3] bank cache_miss_resrv miss_add_addr", false,-1, 25,0);
        vcdp->declBus(c+21217,"VX_cache genblk5[3] bank cache_miss_resrv miss_add_wsel", false,-1, 1,0);
        vcdp->declBus(c+21225,"VX_cache genblk5[3] bank cache_miss_resrv miss_add_data", false,-1, 31,0);
        vcdp->declBus(c+21161,"VX_cache genblk5[3] bank cache_miss_resrv miss_add_tid", false,-1, 1,0);
        vcdp->declQuad(c+21169,"VX_cache genblk5[3] bank cache_miss_resrv miss_add_tag", false,-1, 41,0);
        vcdp->declBit(c+21185,"VX_cache genblk5[3] bank cache_miss_resrv miss_add_rw", false,-1);
        vcdp->declBus(c+21193,"VX_cache genblk5[3] bank cache_miss_resrv miss_add_byteen", false,-1, 3,0);
        vcdp->declBit(c+8833,"VX_cache genblk5[3] bank cache_miss_resrv mrvq_init_ready_state", false,-1);
        vcdp->declBit(c+21329,"VX_cache genblk5[3] bank cache_miss_resrv miss_add_is_snp", false,-1);
        vcdp->declBit(c+21337,"VX_cache genblk5[3] bank cache_miss_resrv miss_add_snp_invalidate", false,-1);
        vcdp->declBit(c+21073,"VX_cache genblk5[3] bank cache_miss_resrv miss_resrv_full", false,-1);
        vcdp->declBit(c+21081,"VX_cache genblk5[3] bank cache_miss_resrv miss_resrv_stop", false,-1);
        vcdp->declBit(c+9329,"VX_cache genblk5[3] bank cache_miss_resrv is_fill_st1", false,-1);
        vcdp->declBus(c+8817,"VX_cache genblk5[3] bank cache_miss_resrv fill_addr_st1", false,-1, 25,0);
        vcdp->declBit(c+8305,"VX_cache genblk5[3] bank cache_miss_resrv pending_hazard", false,-1);
        vcdp->declBit(c+8281,"VX_cache genblk5[3] bank cache_miss_resrv miss_resrv_pop", false,-1);
        vcdp->declBit(c+8289,"VX_cache genblk5[3] bank cache_miss_resrv miss_resrv_valid_st0", false,-1);
        vcdp->declBus(c+21097,"VX_cache genblk5[3] bank cache_miss_resrv miss_resrv_addr_st0", false,-1, 25,0);
        vcdp->declBus(c+21105,"VX_cache genblk5[3] bank cache_miss_resrv miss_resrv_wsel_st0", false,-1, 1,0);
        vcdp->declBus(c+21113,"VX_cache genblk5[3] bank cache_miss_resrv miss_resrv_data_st0", false,-1, 31,0);
        vcdp->declBus(c+21089,"VX_cache genblk5[3] bank cache_miss_resrv miss_resrv_tid_st0", false,-1, 1,0);
        vcdp->declQuad(c+21121,"VX_cache genblk5[3] bank cache_miss_resrv miss_resrv_tag_st0", false,-1, 41,0);
        vcdp->declBit(c+8297,"VX_cache genblk5[3] bank cache_miss_resrv miss_resrv_rw_st0", false,-1);
        vcdp->declBus(c+21137,"VX_cache genblk5[3] bank cache_miss_resrv miss_resrv_byteen_st0", false,-1, 3,0);
        vcdp->declBit(c+21145,"VX_cache genblk5[3] bank cache_miss_resrv miss_resrv_is_snp_st0", false,-1);
        vcdp->declBit(c+21153,"VX_cache genblk5[3] bank cache_miss_resrv miss_resrv_snp_invalidate_st0", false,-1);
        {int i; for (i=0; i<16; i++) {
                vcdp->declArray(c+23385+i*3,"VX_cache genblk5[3] bank cache_miss_resrv metadata_table", true,(i+0), 84,0);}}
        vcdp->declArray(c+23769,"VX_cache genblk5[3] bank cache_miss_resrv addr_table", false,-1, 415,0);
        vcdp->declBus(c+23873,"VX_cache genblk5[3] bank cache_miss_resrv valid_table", false,-1, 15,0);
        vcdp->declBus(c+23881,"VX_cache genblk5[3] bank cache_miss_resrv ready_table", false,-1, 15,0);
        vcdp->declBus(c+23889,"VX_cache genblk5[3] bank cache_miss_resrv schedule_ptr", false,-1, 3,0);
        vcdp->declBus(c+23897,"VX_cache genblk5[3] bank cache_miss_resrv head_ptr", false,-1, 3,0);
        vcdp->declBus(c+23905,"VX_cache genblk5[3] bank cache_miss_resrv tail_ptr", false,-1, 3,0);
        vcdp->declBus(c+23913,"VX_cache genblk5[3] bank cache_miss_resrv size", false,-1, 4,0);
        vcdp->declBit(c+23921,"VX_cache genblk5[3] bank cache_miss_resrv enqueue_possible", false,-1);
        vcdp->declBus(c+23905,"VX_cache genblk5[3] bank cache_miss_resrv enqueue_index", false,-1, 3,0);
        vcdp->declBus(c+9817,"VX_cache genblk5[3] bank cache_miss_resrv make_ready", false,-1, 15,0);
        vcdp->declBus(c+9825,"VX_cache genblk5[3] bank cache_miss_resrv make_ready_push", false,-1, 15,0);
        vcdp->declBus(c+9833,"VX_cache genblk5[3] bank cache_miss_resrv valid_address_match", false,-1, 15,0);
        vcdp->declBit(c+8289,"VX_cache genblk5[3] bank cache_miss_resrv dequeue_possible", false,-1);
        vcdp->declBus(c+23889,"VX_cache genblk5[3] bank cache_miss_resrv dequeue_index", false,-1, 3,0);
        vcdp->declBit(c+9841,"VX_cache genblk5[3] bank cache_miss_resrv mrvq_push", false,-1);
        vcdp->declBit(c+9849,"VX_cache genblk5[3] bank cache_miss_resrv mrvq_pop", false,-1);
        vcdp->declBit(c+9857,"VX_cache genblk5[3] bank cache_miss_resrv recover_state", false,-1);
        vcdp->declBit(c+9865,"VX_cache genblk5[3] bank cache_miss_resrv increment_head", false,-1);
        vcdp->declBit(c+9873,"VX_cache genblk5[3] bank cache_miss_resrv update_ready", false,-1);
        vcdp->declBit(c+9881,"VX_cache genblk5[3] bank cache_miss_resrv qual_mrvq_init", false,-1);
        vcdp->declBus(c+25217,"VX_cache genblk5[3] bank cwb_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank cwb_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[3] bank cwb_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[3] bank cwb_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[3] bank cwb_queue reset", false,-1);
        vcdp->declBit(c+8881,"VX_cache genblk5[3] bank cwb_queue push", false,-1);
        vcdp->declBit(c+1065,"VX_cache genblk5[3] bank cwb_queue pop", false,-1);
        vcdp->declArray(c+9889,"VX_cache genblk5[3] bank cwb_queue data_in", false,-1, 75,0);
        vcdp->declArray(c+9913,"VX_cache genblk5[3] bank cwb_queue data_out", false,-1, 75,0);
        vcdp->declBit(c+21377,"VX_cache genblk5[3] bank cwb_queue empty", false,-1);
        vcdp->declBit(c+21385,"VX_cache genblk5[3] bank cwb_queue full", false,-1);
        vcdp->declBus(c+23929,"VX_cache genblk5[3] bank cwb_queue size", false,-1, 2,0);
        vcdp->declBus(c+23929,"VX_cache genblk5[3] bank cwb_queue size_r", false,-1, 2,0);
        vcdp->declBit(c+937,"VX_cache genblk5[3] bank cwb_queue reading", false,-1);
        vcdp->declBit(c+9937,"VX_cache genblk5[3] bank cwb_queue writing", false,-1);
        {int i; for (i=0; i<4; i++) {
                vcdp->declArray(c+23937+i*3,"VX_cache genblk5[3] bank cwb_queue genblk3 data", true,(i+0), 75,0);}}
        vcdp->declArray(c+24033,"VX_cache genblk5[3] bank cwb_queue genblk3 genblk2 head_r", false,-1, 75,0);
        vcdp->declArray(c+24057,"VX_cache genblk5[3] bank cwb_queue genblk3 genblk2 curr_r", false,-1, 75,0);
        vcdp->declBus(c+24081,"VX_cache genblk5[3] bank cwb_queue genblk3 genblk2 wr_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+24089,"VX_cache genblk5[3] bank cwb_queue genblk3 genblk2 rd_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+24097,"VX_cache genblk5[3] bank cwb_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 1,0);
        vcdp->declBit(c+21377,"VX_cache genblk5[3] bank cwb_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+21385,"VX_cache genblk5[3] bank cwb_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+24105,"VX_cache genblk5[3] bank cwb_queue genblk3 genblk2 bypass_r", false,-1);
        vcdp->declBus(c+25225,"VX_cache genblk5[3] bank dwb_queue DATAW", false,-1, 31,0);
        vcdp->declBus(c+25081,"VX_cache genblk5[3] bank dwb_queue SIZE", false,-1, 31,0);
        vcdp->declBus(c+25089,"VX_cache genblk5[3] bank dwb_queue BUFFERED_OUTPUT", false,-1, 31,0);
        vcdp->declBit(c+24489,"VX_cache genblk5[3] bank dwb_queue clk", false,-1);
        vcdp->declBit(c+24497,"VX_cache genblk5[3] bank dwb_queue reset", false,-1);
        vcdp->declBit(c+8905,"VX_cache genblk5[3] bank dwb_queue push", false,-1);
        vcdp->declBit(c+1073,"VX_cache genblk5[3] bank dwb_queue pop", false,-1);
        vcdp->declArray(c+9945,"VX_cache genblk5[3] bank dwb_queue data_in", false,-1, 199,0);
        vcdp->declArray(c+10001,"VX_cache genblk5[3] bank dwb_queue data_out", false,-1, 199,0);
        vcdp->declBit(c+21393,"VX_cache genblk5[3] bank dwb_queue empty", false,-1);
        vcdp->declBit(c+21401,"VX_cache genblk5[3] bank dwb_queue full", false,-1);
        vcdp->declBus(c+24113,"VX_cache genblk5[3] bank dwb_queue size", false,-1, 2,0);
        vcdp->declBus(c+24113,"VX_cache genblk5[3] bank dwb_queue size_r", false,-1, 2,0);
        vcdp->declBit(c+945,"VX_cache genblk5[3] bank dwb_queue reading", false,-1);
        vcdp->declBit(c+10057,"VX_cache genblk5[3] bank dwb_queue writing", false,-1);
        {int i; for (i=0; i<4; i++) {
                vcdp->declArray(c+24121+i*7,"VX_cache genblk5[3] bank dwb_queue genblk3 data", true,(i+0), 199,0);}}
        vcdp->declArray(c+24345,"VX_cache genblk5[3] bank dwb_queue genblk3 genblk2 head_r", false,-1, 199,0);
        vcdp->declArray(c+24401,"VX_cache genblk5[3] bank dwb_queue genblk3 genblk2 curr_r", false,-1, 199,0);
        vcdp->declBus(c+24457,"VX_cache genblk5[3] bank dwb_queue genblk3 genblk2 wr_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+24465,"VX_cache genblk5[3] bank dwb_queue genblk3 genblk2 rd_ptr_r", false,-1, 1,0);
        vcdp->declBus(c+24473,"VX_cache genblk5[3] bank dwb_queue genblk3 genblk2 rd_ptr_next_r", false,-1, 1,0);
        vcdp->declBit(c+21393,"VX_cache genblk5[3] bank dwb_queue genblk3 genblk2 empty_r", false,-1);
        vcdp->declBit(c+21401,"VX_cache genblk5[3] bank dwb_queue genblk3 genblk2 full_r", false,-1);
        vcdp->declBit(c+24481,"VX_cache genblk5[3] bank dwb_queue genblk3 genblk2 bypass_r", false,-1);
    }
}

void VVX_cache::traceFullThis__1(VVX_cache__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code) {
    VVX_cache* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    int c = code;
    if (0 && vcdp && c) {}  // Prevent unused
    // Variables
    WData/*127:0*/ __Vtemp547[4];
    WData/*127:0*/ __Vtemp566[4];
    WData/*127:0*/ __Vtemp585[4];
    WData/*127:0*/ __Vtemp604[4];
    WData/*127:0*/ __Vtemp522[4];
    WData/*127:0*/ __Vtemp523[4];
    WData/*127:0*/ __Vtemp524[4];
    WData/*127:0*/ __Vtemp525[4];
    WData/*127:0*/ __Vtemp528[4];
    WData/*127:0*/ __Vtemp529[4];
    WData/*127:0*/ __Vtemp534[4];
    WData/*127:0*/ __Vtemp540[4];
    WData/*127:0*/ __Vtemp541[4];
    WData/*127:0*/ __Vtemp544[4];
    WData/*127:0*/ __Vtemp545[4];
    WData/*127:0*/ __Vtemp546[4];
    WData/*127:0*/ __Vtemp548[4];
    WData/*127:0*/ __Vtemp553[4];
    WData/*127:0*/ __Vtemp559[4];
    WData/*127:0*/ __Vtemp560[4];
    WData/*127:0*/ __Vtemp563[4];
    WData/*127:0*/ __Vtemp564[4];
    WData/*127:0*/ __Vtemp565[4];
    WData/*127:0*/ __Vtemp567[4];
    WData/*127:0*/ __Vtemp572[4];
    WData/*127:0*/ __Vtemp578[4];
    WData/*127:0*/ __Vtemp579[4];
    WData/*127:0*/ __Vtemp582[4];
    WData/*127:0*/ __Vtemp583[4];
    WData/*127:0*/ __Vtemp584[4];
    WData/*127:0*/ __Vtemp586[4];
    WData/*127:0*/ __Vtemp591[4];
    WData/*127:0*/ __Vtemp597[4];
    WData/*127:0*/ __Vtemp598[4];
    WData/*127:0*/ __Vtemp601[4];
    WData/*127:0*/ __Vtemp602[4];
    WData/*127:0*/ __Vtemp603[4];
    WData/*127:0*/ __Vtemp612[4];
    WData/*127:0*/ __Vtemp620[4];
    WData/*127:0*/ __Vtemp628[4];
    WData/*127:0*/ __Vtemp636[4];
    // Body
    {
        vcdp->fullBus(c+1,(vlTOPp->VX_cache__DOT____Vcellout__cache_core_req_bank_sel__per_bank_valid),16);
        vcdp->fullBus(c+9,(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready),4);
        vcdp->fullBus(c+17,(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_ready),4);
        vcdp->fullBus(c+25,(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_ready),4);
        vcdp->fullBus(c+33,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__curr_bank_core_req_valid),4);
        vcdp->fullBit(c+41,((1U & (IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready))));
        vcdp->fullBit(c+49,((1U & (IData)(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_ready))));
        vcdp->fullBit(c+57,((1U & (IData)(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_ready))));
        vcdp->fullBus(c+65,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__curr_bank_core_req_valid),4);
        vcdp->fullBit(c+73,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                   >> 1U))));
        vcdp->fullBit(c+81,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_ready) 
                                   >> 1U))));
        vcdp->fullBit(c+89,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_ready) 
                                   >> 1U))));
        vcdp->fullBus(c+97,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__curr_bank_core_req_valid),4);
        vcdp->fullBit(c+105,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                    >> 2U))));
        vcdp->fullBit(c+113,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_ready) 
                                    >> 2U))));
        vcdp->fullBit(c+121,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_ready) 
                                    >> 2U))));
        vcdp->fullBus(c+129,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__curr_bank_core_req_valid),4);
        vcdp->fullBit(c+137,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                    >> 3U))));
        vcdp->fullBit(c+145,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_ready) 
                                    >> 3U))));
        vcdp->fullBit(c+153,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_ready) 
                                    >> 3U))));
        vcdp->fullBus(c+161,(vlTOPp->VX_cache__DOT__cache_core_req_bank_sel__DOT__genblk2__DOT__per_bank_ready_sel),4);
        vcdp->fullBit(c+169,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dfqq_pop));
        vcdp->fullBit(c+177,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__reading));
        vcdp->fullBit(c+185,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dram_wb_req_fire));
        vcdp->fullBit(c+193,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_rsp_fire));
        vcdp->fullQuad(c+201,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__snp_req_queue__data_in),55);
        vcdp->fullBit(c+217,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__writing));
        vcdp->fullArray(c+225,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__dfp_queue__data_in),154);
        vcdp->fullBit(c+265,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__writing));
        vcdp->fullArray(c+273,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellinp__reqq_queue__data_in),314);
        vcdp->fullBit(c+353,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__writing));
        vcdp->fullBit(c+361,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__reading));
        vcdp->fullBit(c+369,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__reading));
        vcdp->fullBit(c+377,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dram_wb_req_fire));
        vcdp->fullBit(c+385,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_rsp_fire));
        vcdp->fullQuad(c+393,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__snp_req_queue__data_in),55);
        vcdp->fullBit(c+409,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__writing));
        vcdp->fullArray(c+417,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__dfp_queue__data_in),154);
        vcdp->fullBit(c+457,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__writing));
        vcdp->fullArray(c+465,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellinp__reqq_queue__data_in),314);
        vcdp->fullBit(c+545,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__writing));
        vcdp->fullBit(c+553,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__reading));
        vcdp->fullBit(c+561,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__reading));
        vcdp->fullBit(c+569,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dram_wb_req_fire));
        vcdp->fullBit(c+577,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_rsp_fire));
        vcdp->fullQuad(c+585,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__snp_req_queue__data_in),55);
        vcdp->fullBit(c+601,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__writing));
        vcdp->fullArray(c+609,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__dfp_queue__data_in),154);
        vcdp->fullBit(c+649,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__writing));
        vcdp->fullArray(c+657,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellinp__reqq_queue__data_in),314);
        vcdp->fullBit(c+737,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__writing));
        vcdp->fullBit(c+745,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__reading));
        vcdp->fullBit(c+753,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__reading));
        vcdp->fullBit(c+761,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dram_wb_req_fire));
        vcdp->fullBit(c+769,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_rsp_fire));
        vcdp->fullQuad(c+777,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__snp_req_queue__data_in),55);
        vcdp->fullBit(c+793,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__writing));
        vcdp->fullArray(c+801,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__dfp_queue__data_in),154);
        vcdp->fullBit(c+841,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__writing));
        vcdp->fullArray(c+849,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellinp__reqq_queue__data_in),314);
        vcdp->fullBit(c+929,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__writing));
        vcdp->fullBit(c+937,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__reading));
        vcdp->fullBit(c+945,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__reading));
        vcdp->fullBit(c+953,((((IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dfqq_pop) 
                               & (~ (IData)((0U != (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bank_dram_fill_req_valid))))) 
                              & (~ ((~ (IData)((0U 
                                                != 
                                                (0xfU 
                                                 & (vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[3U] 
                                                    >> 0x10U))))) 
                                    | (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r))))));
        vcdp->fullBit(c+961,(((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__curr_bank_core_req_valid)) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBit(c+969,((1U & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)) 
                                    & (IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready)))));
        vcdp->fullBit(c+977,((((((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
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
        vcdp->fullBit(c+985,((((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__curr_bank_core_req_valid)) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBit(c+993,(((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__curr_bank_core_req_valid)) 
                              & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBit(c+1001,((1U & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)) 
                                     & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                        >> 1U)))));
        vcdp->fullBit(c+1009,((((((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
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
        vcdp->fullBit(c+1017,((((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__curr_bank_core_req_valid)) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBit(c+1025,(((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__curr_bank_core_req_valid)) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBit(c+1033,((1U & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)) 
                                     & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                        >> 2U)))));
        vcdp->fullBit(c+1041,((((((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
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
        vcdp->fullBit(c+1049,((((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__curr_bank_core_req_valid)) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBit(c+1057,(((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__curr_bank_core_req_valid)) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBit(c+1065,((1U & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)) 
                                     & ((IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_ready) 
                                        >> 3U)))));
        vcdp->fullBit(c+1073,((((((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
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
        vcdp->fullBit(c+1081,((((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__curr_bank_core_req_valid)) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBus(c+1089,(vlTOPp->VX_cache__DOT__per_bank_core_req_ready),4);
        vcdp->fullBus(c+1097,(vlTOPp->VX_cache__DOT__per_bank_core_rsp_valid),4);
        vcdp->fullBus(c+1105,(vlTOPp->VX_cache__DOT__per_bank_core_rsp_tid),8);
        vcdp->fullArray(c+1113,(vlTOPp->VX_cache__DOT__per_bank_core_rsp_data),128);
        vcdp->fullArray(c+1145,(vlTOPp->VX_cache__DOT__per_bank_core_rsp_tag),168);
        vcdp->fullBus(c+1193,(vlTOPp->VX_cache__DOT__per_bank_dram_fill_req_valid),4);
        vcdp->fullArray(c+1201,(vlTOPp->VX_cache__DOT__per_bank_dram_fill_req_addr),112);
        vcdp->fullBus(c+1233,(vlTOPp->VX_cache__DOT__per_bank_dram_fill_rsp_ready),4);
        vcdp->fullBus(c+1241,(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_valid),4);
        vcdp->fullQuad(c+1249,(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_byteen),64);
        vcdp->fullArray(c+1265,(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_addr),112);
        vcdp->fullArray(c+1297,(vlTOPp->VX_cache__DOT__per_bank_dram_wb_req_data),512);
        vcdp->fullBus(c+1425,(vlTOPp->VX_cache__DOT__per_bank_snp_req_ready),4);
        vcdp->fullBus(c+1433,(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_valid),4);
        vcdp->fullArray(c+1441,(vlTOPp->VX_cache__DOT__per_bank_snp_rsp_tag),112);
        vcdp->fullBus(c+1473,((3U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U] 
                                     >> 0xaU))),2);
        vcdp->fullBus(c+1481,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[0U]),32);
        vcdp->fullQuad(c+1489,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[1U]))))),42);
        vcdp->fullBit(c+1505,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_unqual) 
                                & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                   | ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                       >> 0x1aU) & 
                                      (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                          >> 0x1bU))))) 
                               & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_push_stall) 
                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwbq_push_stall))))));
        vcdp->fullBit(c+1513,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__curr_bank_dram_wb_req_valid));
        vcdp->fullBus(c+1521,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                           << 0xaU) 
                                          | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                             >> 0x16U)))),16);
        vcdp->fullBus(c+1529,((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                              << 4U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                                                >> 0x1cU)))),26);
        __Vtemp522[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U] 
                                     >> 0x1cU));
        __Vtemp522[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                                     >> 0x1cU));
        __Vtemp522[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                                     >> 0x1cU));
        __Vtemp522[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                                     >> 0x1cU));
        vcdp->fullArray(c+1537,(__Vtemp522),128);
        vcdp->fullBit(c+1569,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__curr_bank_snp_rsp_valid));
        vcdp->fullBus(c+1577,((0xfffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U])),28);
        vcdp->fullBus(c+1585,((3U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U] 
                                     >> 0xaU))),2);
        vcdp->fullBus(c+1593,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[0U]),32);
        vcdp->fullQuad(c+1601,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[1U]))))),42);
        vcdp->fullBit(c+1617,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_unqual) 
                                & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                   | ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                       >> 0x1aU) & 
                                      (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                          >> 0x1bU))))) 
                               & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_push_stall) 
                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwbq_push_stall))))));
        vcdp->fullBit(c+1625,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__curr_bank_dram_wb_req_valid));
        vcdp->fullBus(c+1633,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                           << 0xaU) 
                                          | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                             >> 0x16U)))),16);
        vcdp->fullBus(c+1641,((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                              << 4U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                                                >> 0x1cU)))),26);
        __Vtemp523[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U] 
                                     >> 0x1cU));
        __Vtemp523[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                                     >> 0x1cU));
        __Vtemp523[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                                     >> 0x1cU));
        __Vtemp523[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                                     >> 0x1cU));
        vcdp->fullArray(c+1649,(__Vtemp523),128);
        vcdp->fullBit(c+1681,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__curr_bank_snp_rsp_valid));
        vcdp->fullBus(c+1689,((0xfffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U])),28);
        vcdp->fullBus(c+1697,((3U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U] 
                                     >> 0xaU))),2);
        vcdp->fullBus(c+1705,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[0U]),32);
        vcdp->fullQuad(c+1713,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[1U]))))),42);
        vcdp->fullBit(c+1729,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_unqual) 
                                & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                   | ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                       >> 0x1aU) & 
                                      (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                          >> 0x1bU))))) 
                               & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_push_stall) 
                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwbq_push_stall))))));
        vcdp->fullBit(c+1737,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__curr_bank_dram_wb_req_valid));
        vcdp->fullBus(c+1745,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                           << 0xaU) 
                                          | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                             >> 0x16U)))),16);
        vcdp->fullBus(c+1753,((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                              << 4U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                                                >> 0x1cU)))),26);
        __Vtemp524[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U] 
                                     >> 0x1cU));
        __Vtemp524[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                                     >> 0x1cU));
        __Vtemp524[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                                     >> 0x1cU));
        __Vtemp524[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                                     >> 0x1cU));
        vcdp->fullArray(c+1761,(__Vtemp524),128);
        vcdp->fullBit(c+1793,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__curr_bank_snp_rsp_valid));
        vcdp->fullBus(c+1801,((0xfffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U])),28);
        vcdp->fullBus(c+1809,((3U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U] 
                                     >> 0xaU))),2);
        vcdp->fullBus(c+1817,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[0U]),32);
        vcdp->fullQuad(c+1825,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[2U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out[1U]))))),42);
        vcdp->fullBit(c+1841,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_unqual) 
                                & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                   | ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                       >> 0x1aU) & 
                                      (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                          >> 0x1bU))))) 
                               & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_push_stall) 
                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwbq_push_stall))))));
        vcdp->fullBit(c+1849,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__curr_bank_dram_wb_req_valid));
        vcdp->fullBus(c+1857,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                           << 0xaU) 
                                          | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                             >> 0x16U)))),16);
        vcdp->fullBus(c+1865,((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[5U] 
                                              << 4U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                                                >> 0x1cU)))),26);
        __Vtemp525[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U] 
                                     >> 0x1cU));
        __Vtemp525[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[1U] 
                                     >> 0x1cU));
        __Vtemp525[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[2U] 
                                     >> 0x1cU));
        __Vtemp525[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[4U] 
                           << 4U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[3U] 
                                     >> 0x1cU));
        vcdp->fullArray(c+1873,(__Vtemp525),128);
        vcdp->fullBit(c+1905,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__curr_bank_snp_rsp_valid));
        vcdp->fullBus(c+1913,((0xfffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[0U])),28);
        vcdp->fullBit(c+1921,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dwb_valid));
        vcdp->fullBit(c+1929,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dfqq_req));
        vcdp->fullBus(c+1937,(((0x6fU >= (0x7fU & ((IData)(0x1cU) 
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
        vcdp->fullBit(c+1945,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_has_request)))));
        vcdp->fullBit(c+1953,((0U != (IData)(vlTOPp->VX_cache__DOT__per_bank_dram_fill_req_valid))));
        vcdp->fullBus(c+1961,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dwb_bank),2);
        vcdp->fullBit(c+1969,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__update_use));
        vcdp->fullBit(c+1977,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__reading));
        vcdp->fullBit(c+1985,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__writing));
        vcdp->fullBus(c+1993,((0xfU & (vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[3U] 
                                       >> 0x10U))),4);
        __Vtemp528[0U] = vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[0U];
        __Vtemp528[1U] = vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[1U];
        __Vtemp528[2U] = vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[2U];
        __Vtemp528[3U] = (0xffffU & vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[3U]);
        vcdp->fullArray(c+2001,(__Vtemp528),112);
        vcdp->fullBus(c+2033,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bqual_bank_dram_fill_req_valid),4);
        vcdp->fullArray(c+2041,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_bank_dram_fill_req_addr),112);
        vcdp->fullBus(c+2073,(((IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bqual_bank_dram_fill_req_valid) 
                               & (~ ((IData)(1U) << (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_request_index))))),4);
        vcdp->fullBit(c+2081,((1U & ((~ (IData)((0U 
                                                 != 
                                                 (0xfU 
                                                  & (vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out[3U] 
                                                     >> 0x10U))))) 
                                     | (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->fullBit(c+2089,(((0U != (IData)(vlTOPp->VX_cache__DOT__per_bank_dram_fill_req_valid)) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBus(c+2097,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_request_index),2);
        vcdp->fullBit(c+2105,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__qual_has_request));
        vcdp->fullArray(c+2113,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellinp__dfqq_queue__data_in),116);
        vcdp->fullArray(c+2145,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT____Vcellout__dfqq_queue__data_out),116);
        vcdp->fullBit(c+2177,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__writing));
        vcdp->fullBus(c+2185,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->fullBus(c+2193,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->fullBus(c+2201,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__sel_dwb__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->fullBus(c+2209,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__sel_dwb__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->fullArray(c+2217,(vlTOPp->VX_cache__DOT____Vcellout__cache_core_rsp_merge__core_rsp_data),128);
        vcdp->fullQuad(c+2249,(((0xa7U >= (0xffU & 
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
                                           << ((0U 
                                                == 
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
        vcdp->fullBus(c+2265,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__main_bank_index),2);
        vcdp->fullBus(c+2273,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__per_bank_core_rsp_pop_unqual),4);
        vcdp->fullBus(c+2281,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->fullBit(c+2289,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__grant_valid));
        vcdp->fullBus(c+2297,((((IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__requests_use) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r))) 
                               | (((IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__refill_original) 
                                   ^ (IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_valid)) 
                                  & (~ (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__refill_original))))),4);
        vcdp->fullBus(c+2305,((((IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__refill_original) 
                                ^ (IData)(vlTOPp->VX_cache__DOT__per_bank_core_rsp_valid)) 
                               & (~ (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__refill_original)))),4);
        vcdp->fullBus(c+2313,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->fullBus(c+2321,(vlTOPp->VX_cache__DOT__snp_rsp_arb__DOT__fsq_bank),2);
        vcdp->fullBit(c+2329,(vlTOPp->VX_cache__DOT__snp_rsp_arb__DOT__fsq_valid));
        vcdp->fullBus(c+2337,(vlTOPp->VX_cache__DOT__snp_rsp_arb__DOT__sel_ffsq__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->fullBus(c+2345,(vlTOPp->VX_cache__DOT__snp_rsp_arb__DOT__sel_ffsq__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->fullBit(c+2353,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop));
        vcdp->fullBus(c+2361,((0x3ffffffU & (IData)(
                                                    (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                     >> 0x1dU)))),26);
        vcdp->fullBit(c+2369,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                             >> 0x1cU)))));
        vcdp->fullBus(c+2377,((0xfffffffU & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out))),28);
        vcdp->fullBit(c+2385,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop));
        vcdp->fullBus(c+2393,((0x3ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])),26);
        __Vtemp529[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U];
        __Vtemp529[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U];
        __Vtemp529[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U];
        __Vtemp529[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U];
        vcdp->fullArray(c+2401,(__Vtemp529),128);
        vcdp->fullBit(c+2433,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop));
        vcdp->fullBit(c+2441,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request)))));
        vcdp->fullBit(c+2449,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request));
        vcdp->fullBus(c+2457,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index),2);
        vcdp->fullBit(c+2465,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_req_rw_st0));
        vcdp->fullBus(c+2473,((0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                       >> (0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 2U))))),4);
        vcdp->fullBus(c+2481,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_req_addr_st0),30);
        vcdp->fullBus(c+2489,((((0U == (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
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
        vcdp->fullBit(c+2497,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_pop));
        vcdp->fullBit(c+2505,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible));
        vcdp->fullBit(c+2513,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_rw_st0));
        vcdp->fullBit(c+2521,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match))));
        vcdp->fullBit(c+2529,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_because_miss) 
                               & (((0x3ffffffU & ((
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                   << 7U) 
                                                  | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                     >> 0x19U))) 
                                   == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                   [0U]) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                               >> 0x14U))))));
        vcdp->fullBit(c+2537,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__force_request_miss_st1e));
        vcdp->fullBit(c+2545,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__recover_mrvq_state_st2));
        vcdp->fullBit(c+2553,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_push_stall));
        vcdp->fullBit(c+2561,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwbq_push_stall));
        vcdp->fullBit(c+2569,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_push_stall));
        vcdp->fullBit(c+2577,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dram_fill_req_stall));
        vcdp->fullBit(c+2585,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__stall_bank_pipe));
        vcdp->fullBit(c+2593,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_in_pipe));
        vcdp->fullBit(c+2601,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1[0]));
        vcdp->fullBit(c+2609,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__going_to_write_st1[0]));
        vcdp->fullBit(c+2617,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual));
        vcdp->fullBit(c+2625,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop_unqual));
        vcdp->fullBit(c+2633,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop_unqual));
        vcdp->fullBit(c+2641,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop) 
                                 | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_pop)) 
                                | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop)) 
                               | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop))));
        vcdp->fullBus(c+2649,((0x3ffffffU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
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
        vcdp->fullBus(c+2657,((3U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop_unqual)
                                      ? (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_req_addr_st0)
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                          ? (3U & (
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                    << 0x1eU) 
                                                   | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                      >> 2U)))
                                          : 0U)))),2);
        vcdp->fullBus(c+2665,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
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
        __Vtemp534[0U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U]
                           : 0x39U);
        __Vtemp534[1U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U]
                           : 0U);
        __Vtemp534[2U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U]
                           : 0U);
        __Vtemp534[3U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U]
                           : 0U);
        vcdp->fullArray(c+2673,(__Vtemp534),128);
        vcdp->fullQuad(c+2705,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                 ? ((VL_ULL(0x1ffffffffff80) 
                                     & (((QData)((IData)(
                                                         vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                         << 0x3eU) 
                                        | (((QData)((IData)(
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
        vcdp->fullBit(c+2721,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible) 
                                         & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_rw_st0))
                                         ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_pop_unqual) 
                                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__reqq_req_rw_st0))
                                                  ? 1U
                                                  : 0U)))));
        vcdp->fullBit(c+2729,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                               [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                               >> 1U))
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? 1U : 0U)))));
        vcdp->fullBit(c+2737,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? (1U & (IData)(
                                                          (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                           >> 0x1cU)))
                                          : 0U)))));
        vcdp->fullBit(c+2745,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1[0]));
        vcdp->fullBus(c+2753,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1[0]),26);
        vcdp->fullBus(c+2761,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1[0]),2);
        vcdp->fullBus(c+2769,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writeword_st1[0]),32);
        vcdp->fullQuad(c+2777,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__inst_meta_st1[0]),49);
        vcdp->fullArray(c+2793,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writedata_st1[0]),128);
        vcdp->fullBit(c+2825,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1[0]));
        vcdp->fullBit(c+2833,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_invalidate_st1[0]));
        vcdp->fullBit(c+2841,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_mrvq_st1[0]));
        __Vtemp540[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp540[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp540[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp540[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->fullBus(c+2849,((((0U == (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                                 [0U] 
                                                 << 5U)))
                                 ? 0U : (__Vtemp540[
                                         ((IData)(1U) 
                                          + (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                             [0U]))] 
                                         << ((IData)(0x20U) 
                                             - (0x1fU 
                                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                                   [0U] 
                                                   << 5U))))) 
                               | (__Vtemp540[(3U & 
                                              vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                              [0U])] 
                                  >> (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                               [0U] 
                                               << 5U))))),32);
        __Vtemp541[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp541[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp541[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp541[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->fullArray(c+2857,(__Vtemp541),128);
        vcdp->fullBus(c+2889,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c
                              [0U]),20);
        vcdp->fullBit(c+2897,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_st1e));
        vcdp->fullBit(c+2905,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dirty_st1e));
        vcdp->fullBus(c+2913,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c
                              [0U]),16);
        vcdp->fullQuad(c+2921,((VL_ULL(0x3ffffffffff) 
                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__inst_meta_st1
                                   [0U] >> 7U))),42);
        vcdp->fullBus(c+2937,((3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__inst_meta_st1
                                            [0U]))),2);
        vcdp->fullBit(c+2945,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__inst_meta_st1
                                             [0U] >> 6U)))));
        vcdp->fullBus(c+2953,((0xfU & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__inst_meta_st1
                                               [0U] 
                                               >> 2U)))),4);
        vcdp->fullBit(c+2961,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dirty_st1e))));
        vcdp->fullBit(c+2969,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1
                              [0U]));
        vcdp->fullBit(c+2977,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_invalidate_st1
                              [0U]));
        vcdp->fullBit(c+2985,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_to_mrvq_st1e));
        vcdp->fullBit(c+2993,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_to_mrvq_st1e) 
                               | ((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                    & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1
                                       [0U])) & (~ 
                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                                                 [0U])) 
                                  & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                                  [0U]))));
        vcdp->fullBit(c+3001,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_because_miss));
        vcdp->fullBit(c+3009,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                              [0U]));
        vcdp->fullBit(c+3017,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_mrvq_st1
                              [0U]));
        vcdp->fullBit(c+3025,((((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                                 [0U] & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_mrvq_st1
                                 [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__recover_mrvq_state_st2)) 
                               & ((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 7U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x19U))) 
                                  == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                  [0U]))));
        vcdp->fullBus(c+3033,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                              [0U]),26);
        vcdp->fullBit(c+3041,((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                               [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                                       [0U]))));
        vcdp->fullBit(c+3049,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_init_ready_state_st2));
        vcdp->fullBit(c+3057,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_unqual) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfpq_pop_unqual)) 
                               & ((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 7U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x19U))) 
                                  == (0x3ffffffU & 
                                      vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])))));
        vcdp->fullBit(c+3065,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_unqual) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                                [0U]) & ((0x3ffffffU 
                                          & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 7U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x19U))) 
                                         == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                         [0U]))));
        vcdp->fullBit(c+3073,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_unqual));
        vcdp->fullBit(c+3081,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add));
        vcdp->fullBit(c+3089,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_is_mrvq));
        vcdp->fullBit(c+3097,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwbq_push_unqual) 
                                 & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                                & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                      >> 6U))) & (~ 
                                                  (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_push_stall) 
                                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                                   | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->fullBit(c+3105,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwbq_push_unqual));
        vcdp->fullBit(c+3113,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU)))))));
        vcdp->fullBit(c+3121,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_push_unqual) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwbq_push_stall) 
                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->fullBit(c+3129,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_is_dwb_in));
        vcdp->fullBit(c+3137,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_is_snp_in));
        vcdp->fullBit(c+3145,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 7U))));
        vcdp->fullBit(c+3153,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 6U))));
        vcdp->fullBit(c+3161,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_push_unqual));
        vcdp->fullQuad(c+3169,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out),55);
        vcdp->fullBit(c+3185,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__reading));
        vcdp->fullArray(c+3193,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out),154);
        vcdp->fullBit(c+3233,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__reading));
        vcdp->fullBus(c+3241,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x16U))),4);
        vcdp->fullBus(c+3249,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x12U))),4);
        vcdp->fullBus(c+3257,((0xffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                          >> 2U))),16);
        __Vtemp544[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                                        >> 0xaU));
        __Vtemp544[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                                        >> 0xaU));
        __Vtemp544[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                                        >> 0xaU));
        __Vtemp544[3U] = (0xffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                        << 0x16U) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                                        >> 0xaU)));
        vcdp->fullArray(c+3265,(__Vtemp544),120);
        __Vtemp545[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U] 
                                        >> 0xaU));
        __Vtemp545[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                                        >> 0xaU));
        __Vtemp545[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                                        >> 0xaU));
        __Vtemp545[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                                        >> 0xaU));
        vcdp->fullArray(c+3297,(__Vtemp545),128);
        vcdp->fullQuad(c+3329,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[0U]))))),42);
        vcdp->fullBit(c+3345,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty));
        vcdp->fullBit(c+3353,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__pop_qual));
        vcdp->fullBus(c+3361,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                        >> 0x16U) & 
                                       VL_NEGATE_I((IData)(
                                                           (1U 
                                                            & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty)))))))),4);
        vcdp->fullArray(c+3369,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out),314);
        vcdp->fullBit(c+3449,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__reading));
        vcdp->fullBus(c+3457,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->fullBus(c+3465,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->fullArray(c+3473,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__s0_1_c0__in),243);
        vcdp->fullBus(c+3537,((0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                               [0U])),6);
        vcdp->fullBit(c+3545,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                              [0U]));
        vcdp->fullBus(c+3553,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writeword_st1
                              [0U]),32);
        __Vtemp546[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writedata_st1
            [0U][0U];
        __Vtemp546[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writedata_st1
            [0U][1U];
        __Vtemp546[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writedata_st1
            [0U][2U];
        __Vtemp546[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__writedata_st1
            [0U][3U];
        vcdp->fullArray(c+3561,(__Vtemp546),128);
        vcdp->fullBus(c+3593,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                              [0U]),2);
        vcdp->fullBit(c+3601,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c[0]));
        vcdp->fullBit(c+3609,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c[0]));
        vcdp->fullBus(c+3617,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c[0]),16);
        vcdp->fullBus(c+3625,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c[0]),20);
        vcdp->fullArray(c+3633,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c[0]),128);
        vcdp->fullBit(c+3665,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid 
                                             >> (0x3fU 
                                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                                 [0U]))))));
        vcdp->fullBit(c+3673,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty 
                                             >> (0x3fU 
                                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                                 [0U]))))));
        vcdp->fullBus(c+3681,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                [0U])]),16);
        vcdp->fullBus(c+3689,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag
                              [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                [0U])]),20);
        __Vtemp547[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
              [0U])][0U];
        __Vtemp547[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
              [0U])][1U];
        __Vtemp547[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
              [0U])][2U];
        __Vtemp547[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
              [0U])][3U];
        vcdp->fullArray(c+3697,(__Vtemp547),128);
        vcdp->fullBit(c+3729,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                              [0U]));
        vcdp->fullBit(c+3737,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c
                              [0U]));
        vcdp->fullBus(c+3745,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable),16);
        vcdp->fullArray(c+3753,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__data_write),128);
        vcdp->fullBit(c+3785,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__snoop_hit_no_pending));
        vcdp->fullBit(c+3793,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match));
        vcdp->fullBit(c+3801,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill));
        vcdp->fullBus(c+3809,((0xfffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__addr_st1
                                           [0U] >> 6U))),20);
        vcdp->fullBus(c+3817,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__we),16);
        vcdp->fullBit(c+3825,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__should_write));
        vcdp->fullBit(c+3833,((((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                                 [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1
                                         [0U])) & (~ 
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                                   [0U])) 
                               & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                                  [0U]))));
        vcdp->fullBit(c+3841,(((((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                                  [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1
                                          [0U])) & 
                                 vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                 [0U]) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                                          [0U])) & 
                               (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match)))));
        vcdp->fullBit(c+3849,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss));
        vcdp->fullBit(c+3857,((((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                  & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_snp_st1
                                     [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__is_fill_st1
                                               [0U])) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__valid_st1
                                [0U]) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss)))));
        vcdp->fullBit(c+3865,(((0U == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->fullBit(c+3873,(((1U == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->fullBit(c+3881,(((2U == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->fullBit(c+3889,(((3U == vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->fullBit(c+3897,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable))));
        vcdp->fullArray(c+3905,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT____Vcellinp__s0_1_c0__in),166);
        vcdp->fullArray(c+3953,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__st_1e_2__in),316);
        vcdp->fullBus(c+4033,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready),16);
        vcdp->fullBus(c+4041,((0xffffU & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                                           & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                          << (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr)))),16);
        vcdp->fullBus(c+4049,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match),16);
        vcdp->fullBit(c+4057,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push));
        vcdp->fullBit(c+4065,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_pop));
        vcdp->fullBit(c+4073,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__miss_add_is_mrvq))));
        vcdp->fullBit(c+4081,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__increment_head));
        vcdp->fullBit(c+4089,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready))));
        vcdp->fullBit(c+4097,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__mrvq_init_ready_state_st2))));
        vcdp->fullArray(c+4105,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__cwb_queue__data_in),76);
        vcdp->fullArray(c+4129,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out),76);
        vcdp->fullBit(c+4153,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__writing));
        vcdp->fullArray(c+4161,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellinp__dwb_queue__data_in),200);
        vcdp->fullArray(c+4217,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out),200);
        vcdp->fullBit(c+4273,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__writing));
        vcdp->fullBit(c+4281,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop));
        vcdp->fullBus(c+4289,((0x3ffffffU & (IData)(
                                                    (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                     >> 0x1dU)))),26);
        vcdp->fullBit(c+4297,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                             >> 0x1cU)))));
        vcdp->fullBus(c+4305,((0xfffffffU & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out))),28);
        vcdp->fullBit(c+4313,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop));
        vcdp->fullBus(c+4321,((0x3ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])),26);
        __Vtemp548[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U];
        __Vtemp548[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U];
        __Vtemp548[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U];
        __Vtemp548[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U];
        vcdp->fullArray(c+4329,(__Vtemp548),128);
        vcdp->fullBit(c+4361,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop));
        vcdp->fullBit(c+4369,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request)))));
        vcdp->fullBit(c+4377,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request));
        vcdp->fullBus(c+4385,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index),2);
        vcdp->fullBit(c+4393,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_req_rw_st0));
        vcdp->fullBus(c+4401,((0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                       >> (0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 2U))))),4);
        vcdp->fullBus(c+4409,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_req_addr_st0),30);
        vcdp->fullBus(c+4417,((((0U == (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
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
        vcdp->fullBit(c+4425,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_pop));
        vcdp->fullBit(c+4433,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible));
        vcdp->fullBit(c+4441,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_rw_st0));
        vcdp->fullBit(c+4449,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match))));
        vcdp->fullBit(c+4457,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_because_miss) 
                               & (((0x3ffffffU & ((
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                   << 7U) 
                                                  | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                     >> 0x19U))) 
                                   == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                   [0U]) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                               >> 0x14U))))));
        vcdp->fullBit(c+4465,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__force_request_miss_st1e));
        vcdp->fullBit(c+4473,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__recover_mrvq_state_st2));
        vcdp->fullBit(c+4481,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_push_stall));
        vcdp->fullBit(c+4489,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwbq_push_stall));
        vcdp->fullBit(c+4497,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_push_stall));
        vcdp->fullBit(c+4505,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dram_fill_req_stall));
        vcdp->fullBit(c+4513,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__stall_bank_pipe));
        vcdp->fullBit(c+4521,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_in_pipe));
        vcdp->fullBit(c+4529,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1[0]));
        vcdp->fullBit(c+4537,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__going_to_write_st1[0]));
        vcdp->fullBit(c+4545,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual));
        vcdp->fullBit(c+4553,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop_unqual));
        vcdp->fullBit(c+4561,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop_unqual));
        vcdp->fullBit(c+4569,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop) 
                                 | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_pop)) 
                                | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop)) 
                               | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop))));
        vcdp->fullBus(c+4577,((0x3ffffffU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
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
        vcdp->fullBus(c+4585,((3U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop_unqual)
                                      ? (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_req_addr_st0)
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                          ? (3U & (
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                    << 0x1eU) 
                                                   | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                      >> 2U)))
                                          : 0U)))),2);
        vcdp->fullBus(c+4593,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
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
        __Vtemp553[0U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U]
                           : 0x39U);
        __Vtemp553[1U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U]
                           : 0U);
        __Vtemp553[2U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U]
                           : 0U);
        __Vtemp553[3U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U]
                           : 0U);
        vcdp->fullArray(c+4601,(__Vtemp553),128);
        vcdp->fullQuad(c+4633,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                 ? ((VL_ULL(0x1ffffffffff80) 
                                     & (((QData)((IData)(
                                                         vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                         << 0x3eU) 
                                        | (((QData)((IData)(
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
        vcdp->fullBit(c+4649,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible) 
                                         & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_rw_st0))
                                         ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_pop_unqual) 
                                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__reqq_req_rw_st0))
                                                  ? 1U
                                                  : 0U)))));
        vcdp->fullBit(c+4657,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                               [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                               >> 1U))
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? 1U : 0U)))));
        vcdp->fullBit(c+4665,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? (1U & (IData)(
                                                          (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                           >> 0x1cU)))
                                          : 0U)))));
        vcdp->fullBit(c+4673,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1[0]));
        vcdp->fullBus(c+4681,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1[0]),26);
        vcdp->fullBus(c+4689,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1[0]),2);
        vcdp->fullBus(c+4697,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writeword_st1[0]),32);
        vcdp->fullQuad(c+4705,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__inst_meta_st1[0]),49);
        vcdp->fullArray(c+4721,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writedata_st1[0]),128);
        vcdp->fullBit(c+4753,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1[0]));
        vcdp->fullBit(c+4761,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_invalidate_st1[0]));
        vcdp->fullBit(c+4769,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_mrvq_st1[0]));
        __Vtemp559[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp559[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp559[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp559[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->fullBus(c+4777,((((0U == (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                                 [0U] 
                                                 << 5U)))
                                 ? 0U : (__Vtemp559[
                                         ((IData)(1U) 
                                          + (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                             [0U]))] 
                                         << ((IData)(0x20U) 
                                             - (0x1fU 
                                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                                   [0U] 
                                                   << 5U))))) 
                               | (__Vtemp559[(3U & 
                                              vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                              [0U])] 
                                  >> (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                               [0U] 
                                               << 5U))))),32);
        __Vtemp560[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp560[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp560[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp560[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->fullArray(c+4785,(__Vtemp560),128);
        vcdp->fullBus(c+4817,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c
                              [0U]),20);
        vcdp->fullBit(c+4825,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_st1e));
        vcdp->fullBit(c+4833,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dirty_st1e));
        vcdp->fullBus(c+4841,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c
                              [0U]),16);
        vcdp->fullQuad(c+4849,((VL_ULL(0x3ffffffffff) 
                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__inst_meta_st1
                                   [0U] >> 7U))),42);
        vcdp->fullBus(c+4865,((3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__inst_meta_st1
                                            [0U]))),2);
        vcdp->fullBit(c+4873,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__inst_meta_st1
                                             [0U] >> 6U)))));
        vcdp->fullBus(c+4881,((0xfU & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__inst_meta_st1
                                               [0U] 
                                               >> 2U)))),4);
        vcdp->fullBit(c+4889,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dirty_st1e))));
        vcdp->fullBit(c+4897,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1
                              [0U]));
        vcdp->fullBit(c+4905,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_invalidate_st1
                              [0U]));
        vcdp->fullBit(c+4913,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_to_mrvq_st1e));
        vcdp->fullBit(c+4921,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_to_mrvq_st1e) 
                               | ((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                    & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1
                                       [0U])) & (~ 
                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                                                 [0U])) 
                                  & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                                  [0U]))));
        vcdp->fullBit(c+4929,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_because_miss));
        vcdp->fullBit(c+4937,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                              [0U]));
        vcdp->fullBit(c+4945,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_mrvq_st1
                              [0U]));
        vcdp->fullBit(c+4953,((((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                                 [0U] & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_mrvq_st1
                                 [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__recover_mrvq_state_st2)) 
                               & ((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 7U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x19U))) 
                                  == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                  [0U]))));
        vcdp->fullBus(c+4961,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                              [0U]),26);
        vcdp->fullBit(c+4969,((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                               [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                                       [0U]))));
        vcdp->fullBit(c+4977,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_init_ready_state_st2));
        vcdp->fullBit(c+4985,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_unqual) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfpq_pop_unqual)) 
                               & ((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 7U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x19U))) 
                                  == (0x3ffffffU & 
                                      vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])))));
        vcdp->fullBit(c+4993,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_unqual) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                                [0U]) & ((0x3ffffffU 
                                          & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 7U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x19U))) 
                                         == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                         [0U]))));
        vcdp->fullBit(c+5001,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_unqual));
        vcdp->fullBit(c+5009,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add));
        vcdp->fullBit(c+5017,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_is_mrvq));
        vcdp->fullBit(c+5025,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwbq_push_unqual) 
                                 & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                                & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                      >> 6U))) & (~ 
                                                  (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_push_stall) 
                                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                                   | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->fullBit(c+5033,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwbq_push_unqual));
        vcdp->fullBit(c+5041,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU)))))));
        vcdp->fullBit(c+5049,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_push_unqual) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwbq_push_stall) 
                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->fullBit(c+5057,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_is_dwb_in));
        vcdp->fullBit(c+5065,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_is_snp_in));
        vcdp->fullBit(c+5073,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 7U))));
        vcdp->fullBit(c+5081,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 6U))));
        vcdp->fullBit(c+5089,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_push_unqual));
        vcdp->fullQuad(c+5097,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out),55);
        vcdp->fullBit(c+5113,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__reading));
        vcdp->fullArray(c+5121,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out),154);
        vcdp->fullBit(c+5161,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__reading));
        vcdp->fullBus(c+5169,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x16U))),4);
        vcdp->fullBus(c+5177,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x12U))),4);
        vcdp->fullBus(c+5185,((0xffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                          >> 2U))),16);
        __Vtemp563[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                                        >> 0xaU));
        __Vtemp563[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                                        >> 0xaU));
        __Vtemp563[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                                        >> 0xaU));
        __Vtemp563[3U] = (0xffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                        << 0x16U) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                                        >> 0xaU)));
        vcdp->fullArray(c+5193,(__Vtemp563),120);
        __Vtemp564[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U] 
                                        >> 0xaU));
        __Vtemp564[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                                        >> 0xaU));
        __Vtemp564[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                                        >> 0xaU));
        __Vtemp564[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                                        >> 0xaU));
        vcdp->fullArray(c+5225,(__Vtemp564),128);
        vcdp->fullQuad(c+5257,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[0U]))))),42);
        vcdp->fullBit(c+5273,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty));
        vcdp->fullBit(c+5281,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__pop_qual));
        vcdp->fullBus(c+5289,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                        >> 0x16U) & 
                                       VL_NEGATE_I((IData)(
                                                           (1U 
                                                            & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty)))))))),4);
        vcdp->fullArray(c+5297,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out),314);
        vcdp->fullBit(c+5377,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__reading));
        vcdp->fullBus(c+5385,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->fullBus(c+5393,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->fullArray(c+5401,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__s0_1_c0__in),243);
        vcdp->fullBus(c+5465,((0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                               [0U])),6);
        vcdp->fullBit(c+5473,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                              [0U]));
        vcdp->fullBus(c+5481,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writeword_st1
                              [0U]),32);
        __Vtemp565[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writedata_st1
            [0U][0U];
        __Vtemp565[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writedata_st1
            [0U][1U];
        __Vtemp565[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writedata_st1
            [0U][2U];
        __Vtemp565[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__writedata_st1
            [0U][3U];
        vcdp->fullArray(c+5489,(__Vtemp565),128);
        vcdp->fullBus(c+5521,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                              [0U]),2);
        vcdp->fullBit(c+5529,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c[0]));
        vcdp->fullBit(c+5537,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c[0]));
        vcdp->fullBus(c+5545,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c[0]),16);
        vcdp->fullBus(c+5553,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c[0]),20);
        vcdp->fullArray(c+5561,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c[0]),128);
        vcdp->fullBit(c+5593,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid 
                                             >> (0x3fU 
                                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                                 [0U]))))));
        vcdp->fullBit(c+5601,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty 
                                             >> (0x3fU 
                                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                                 [0U]))))));
        vcdp->fullBus(c+5609,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                [0U])]),16);
        vcdp->fullBus(c+5617,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag
                              [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                [0U])]),20);
        __Vtemp566[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
              [0U])][0U];
        __Vtemp566[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
              [0U])][1U];
        __Vtemp566[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
              [0U])][2U];
        __Vtemp566[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
              [0U])][3U];
        vcdp->fullArray(c+5625,(__Vtemp566),128);
        vcdp->fullBit(c+5657,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                              [0U]));
        vcdp->fullBit(c+5665,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c
                              [0U]));
        vcdp->fullBus(c+5673,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable),16);
        vcdp->fullArray(c+5681,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__data_write),128);
        vcdp->fullBit(c+5713,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__snoop_hit_no_pending));
        vcdp->fullBit(c+5721,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match));
        vcdp->fullBit(c+5729,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill));
        vcdp->fullBus(c+5737,((0xfffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__addr_st1
                                           [0U] >> 6U))),20);
        vcdp->fullBus(c+5745,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__we),16);
        vcdp->fullBit(c+5753,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__should_write));
        vcdp->fullBit(c+5761,((((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                                 [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1
                                         [0U])) & (~ 
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                                   [0U])) 
                               & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                                  [0U]))));
        vcdp->fullBit(c+5769,(((((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                                  [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1
                                          [0U])) & 
                                 vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                 [0U]) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                                          [0U])) & 
                               (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match)))));
        vcdp->fullBit(c+5777,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss));
        vcdp->fullBit(c+5785,((((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                  & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_snp_st1
                                     [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__is_fill_st1
                                               [0U])) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__valid_st1
                                [0U]) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss)))));
        vcdp->fullBit(c+5793,(((0U == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->fullBit(c+5801,(((1U == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->fullBit(c+5809,(((2U == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->fullBit(c+5817,(((3U == vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->fullBit(c+5825,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable))));
        vcdp->fullArray(c+5833,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT____Vcellinp__s0_1_c0__in),166);
        vcdp->fullArray(c+5881,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__st_1e_2__in),316);
        vcdp->fullBus(c+5961,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready),16);
        vcdp->fullBus(c+5969,((0xffffU & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                                           & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                          << (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr)))),16);
        vcdp->fullBus(c+5977,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match),16);
        vcdp->fullBit(c+5985,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push));
        vcdp->fullBit(c+5993,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_pop));
        vcdp->fullBit(c+6001,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__miss_add_is_mrvq))));
        vcdp->fullBit(c+6009,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__increment_head));
        vcdp->fullBit(c+6017,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready))));
        vcdp->fullBit(c+6025,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__mrvq_init_ready_state_st2))));
        vcdp->fullArray(c+6033,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__cwb_queue__data_in),76);
        vcdp->fullArray(c+6057,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out),76);
        vcdp->fullBit(c+6081,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__writing));
        vcdp->fullArray(c+6089,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellinp__dwb_queue__data_in),200);
        vcdp->fullArray(c+6145,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out),200);
        vcdp->fullBit(c+6201,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__writing));
        vcdp->fullBit(c+6209,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop));
        vcdp->fullBus(c+6217,((0x3ffffffU & (IData)(
                                                    (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                     >> 0x1dU)))),26);
        vcdp->fullBit(c+6225,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                             >> 0x1cU)))));
        vcdp->fullBus(c+6233,((0xfffffffU & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out))),28);
        vcdp->fullBit(c+6241,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop));
        vcdp->fullBus(c+6249,((0x3ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])),26);
        __Vtemp567[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U];
        __Vtemp567[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U];
        __Vtemp567[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U];
        __Vtemp567[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U];
        vcdp->fullArray(c+6257,(__Vtemp567),128);
        vcdp->fullBit(c+6289,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop));
        vcdp->fullBit(c+6297,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request)))));
        vcdp->fullBit(c+6305,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request));
        vcdp->fullBus(c+6313,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index),2);
        vcdp->fullBit(c+6321,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_req_rw_st0));
        vcdp->fullBus(c+6329,((0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                       >> (0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 2U))))),4);
        vcdp->fullBus(c+6337,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_req_addr_st0),30);
        vcdp->fullBus(c+6345,((((0U == (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
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
        vcdp->fullBit(c+6353,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_pop));
        vcdp->fullBit(c+6361,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible));
        vcdp->fullBit(c+6369,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_rw_st0));
        vcdp->fullBit(c+6377,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match))));
        vcdp->fullBit(c+6385,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_because_miss) 
                               & (((0x3ffffffU & ((
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                   << 7U) 
                                                  | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                     >> 0x19U))) 
                                   == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                   [0U]) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                               >> 0x14U))))));
        vcdp->fullBit(c+6393,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__force_request_miss_st1e));
        vcdp->fullBit(c+6401,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__recover_mrvq_state_st2));
        vcdp->fullBit(c+6409,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_push_stall));
        vcdp->fullBit(c+6417,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwbq_push_stall));
        vcdp->fullBit(c+6425,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_push_stall));
        vcdp->fullBit(c+6433,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dram_fill_req_stall));
        vcdp->fullBit(c+6441,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__stall_bank_pipe));
        vcdp->fullBit(c+6449,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_in_pipe));
        vcdp->fullBit(c+6457,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1[0]));
        vcdp->fullBit(c+6465,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__going_to_write_st1[0]));
        vcdp->fullBit(c+6473,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual));
        vcdp->fullBit(c+6481,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop_unqual));
        vcdp->fullBit(c+6489,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop_unqual));
        vcdp->fullBit(c+6497,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop) 
                                 | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_pop)) 
                                | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop)) 
                               | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop))));
        vcdp->fullBus(c+6505,((0x3ffffffU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
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
        vcdp->fullBus(c+6513,((3U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop_unqual)
                                      ? (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_req_addr_st0)
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                          ? (3U & (
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                    [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                    << 0x1eU) 
                                                   | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                                      >> 2U)))
                                          : 0U)))),2);
        vcdp->fullBus(c+6521,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
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
        __Vtemp572[0U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U]
                           : 0x39U);
        __Vtemp572[1U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U]
                           : 0U);
        __Vtemp572[2U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U]
                           : 0U);
        __Vtemp572[3U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U]
                           : 0U);
        vcdp->fullArray(c+6529,(__Vtemp572),128);
        vcdp->fullQuad(c+6561,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                 ? ((VL_ULL(0x1ffffffffff80) 
                                     & (((QData)((IData)(
                                                         vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U])) 
                                         << 0x3eU) 
                                        | (((QData)((IData)(
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
        vcdp->fullBit(c+6577,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible) 
                                         & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_rw_st0))
                                         ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_pop_unqual) 
                                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__reqq_req_rw_st0))
                                                  ? 1U
                                                  : 0U)))));
        vcdp->fullBit(c+6585,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                               [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                               >> 1U))
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? 1U : 0U)))));
        vcdp->fullBit(c+6593,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? (1U & (IData)(
                                                          (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                           >> 0x1cU)))
                                          : 0U)))));
        vcdp->fullBit(c+6601,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1[0]));
        vcdp->fullBus(c+6609,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1[0]),26);
        vcdp->fullBus(c+6617,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1[0]),2);
        vcdp->fullBus(c+6625,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writeword_st1[0]),32);
        vcdp->fullQuad(c+6633,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__inst_meta_st1[0]),49);
        vcdp->fullArray(c+6649,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writedata_st1[0]),128);
        vcdp->fullBit(c+6681,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1[0]));
        vcdp->fullBit(c+6689,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_invalidate_st1[0]));
        vcdp->fullBit(c+6697,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_mrvq_st1[0]));
        __Vtemp578[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp578[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp578[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp578[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->fullBus(c+6705,((((0U == (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                                 [0U] 
                                                 << 5U)))
                                 ? 0U : (__Vtemp578[
                                         ((IData)(1U) 
                                          + (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                             [0U]))] 
                                         << ((IData)(0x20U) 
                                             - (0x1fU 
                                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                                   [0U] 
                                                   << 5U))))) 
                               | (__Vtemp578[(3U & 
                                              vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                              [0U])] 
                                  >> (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                               [0U] 
                                               << 5U))))),32);
        __Vtemp579[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp579[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp579[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp579[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->fullArray(c+6713,(__Vtemp579),128);
        vcdp->fullBus(c+6745,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c
                              [0U]),20);
        vcdp->fullBit(c+6753,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_st1e));
        vcdp->fullBit(c+6761,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dirty_st1e));
        vcdp->fullBus(c+6769,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c
                              [0U]),16);
        vcdp->fullQuad(c+6777,((VL_ULL(0x3ffffffffff) 
                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__inst_meta_st1
                                   [0U] >> 7U))),42);
        vcdp->fullBus(c+6793,((3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__inst_meta_st1
                                            [0U]))),2);
        vcdp->fullBit(c+6801,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__inst_meta_st1
                                             [0U] >> 6U)))));
        vcdp->fullBus(c+6809,((0xfU & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__inst_meta_st1
                                               [0U] 
                                               >> 2U)))),4);
        vcdp->fullBit(c+6817,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dirty_st1e))));
        vcdp->fullBit(c+6825,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1
                              [0U]));
        vcdp->fullBit(c+6833,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_invalidate_st1
                              [0U]));
        vcdp->fullBit(c+6841,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_to_mrvq_st1e));
        vcdp->fullBit(c+6849,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_to_mrvq_st1e) 
                               | ((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                    & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1
                                       [0U])) & (~ 
                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                                                 [0U])) 
                                  & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                                  [0U]))));
        vcdp->fullBit(c+6857,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_because_miss));
        vcdp->fullBit(c+6865,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                              [0U]));
        vcdp->fullBit(c+6873,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_mrvq_st1
                              [0U]));
        vcdp->fullBit(c+6881,((((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                                 [0U] & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_mrvq_st1
                                 [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__recover_mrvq_state_st2)) 
                               & ((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 7U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x19U))) 
                                  == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                  [0U]))));
        vcdp->fullBus(c+6889,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                              [0U]),26);
        vcdp->fullBit(c+6897,((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                               [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                                       [0U]))));
        vcdp->fullBit(c+6905,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_init_ready_state_st2));
        vcdp->fullBit(c+6913,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_unqual) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfpq_pop_unqual)) 
                               & ((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 7U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x19U))) 
                                  == (0x3ffffffU & 
                                      vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])))));
        vcdp->fullBit(c+6921,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_unqual) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                                [0U]) & ((0x3ffffffU 
                                          & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 7U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x19U))) 
                                         == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                         [0U]))));
        vcdp->fullBit(c+6929,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_unqual));
        vcdp->fullBit(c+6937,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add));
        vcdp->fullBit(c+6945,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_is_mrvq));
        vcdp->fullBit(c+6953,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwbq_push_unqual) 
                                 & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                                & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                      >> 6U))) & (~ 
                                                  (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_push_stall) 
                                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                                   | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->fullBit(c+6961,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwbq_push_unqual));
        vcdp->fullBit(c+6969,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU)))))));
        vcdp->fullBit(c+6977,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_push_unqual) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwbq_push_stall) 
                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->fullBit(c+6985,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_is_dwb_in));
        vcdp->fullBit(c+6993,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_is_snp_in));
        vcdp->fullBit(c+7001,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 7U))));
        vcdp->fullBit(c+7009,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 6U))));
        vcdp->fullBit(c+7017,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_push_unqual));
        vcdp->fullQuad(c+7025,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out),55);
        vcdp->fullBit(c+7041,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__reading));
        vcdp->fullArray(c+7049,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out),154);
        vcdp->fullBit(c+7089,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__reading));
        vcdp->fullBus(c+7097,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x16U))),4);
        vcdp->fullBus(c+7105,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x12U))),4);
        vcdp->fullBus(c+7113,((0xffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                          >> 2U))),16);
        __Vtemp582[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                                        >> 0xaU));
        __Vtemp582[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                                        >> 0xaU));
        __Vtemp582[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                                        >> 0xaU));
        __Vtemp582[3U] = (0xffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                        << 0x16U) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                                        >> 0xaU)));
        vcdp->fullArray(c+7121,(__Vtemp582),120);
        __Vtemp583[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U] 
                                        >> 0xaU));
        __Vtemp583[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                                        >> 0xaU));
        __Vtemp583[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                                        >> 0xaU));
        __Vtemp583[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                                        >> 0xaU));
        vcdp->fullArray(c+7153,(__Vtemp583),128);
        vcdp->fullQuad(c+7185,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[0U]))))),42);
        vcdp->fullBit(c+7201,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty));
        vcdp->fullBit(c+7209,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__pop_qual));
        vcdp->fullBus(c+7217,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                        >> 0x16U) & 
                                       VL_NEGATE_I((IData)(
                                                           (1U 
                                                            & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty)))))))),4);
        vcdp->fullArray(c+7225,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out),314);
        vcdp->fullBit(c+7305,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__reading));
        vcdp->fullBus(c+7313,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->fullBus(c+7321,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->fullArray(c+7329,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__s0_1_c0__in),243);
        vcdp->fullBus(c+7393,((0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                               [0U])),6);
        vcdp->fullBit(c+7401,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                              [0U]));
        vcdp->fullBus(c+7409,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writeword_st1
                              [0U]),32);
        __Vtemp584[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writedata_st1
            [0U][0U];
        __Vtemp584[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writedata_st1
            [0U][1U];
        __Vtemp584[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writedata_st1
            [0U][2U];
        __Vtemp584[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__writedata_st1
            [0U][3U];
        vcdp->fullArray(c+7417,(__Vtemp584),128);
        vcdp->fullBus(c+7449,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                              [0U]),2);
        vcdp->fullBit(c+7457,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c[0]));
        vcdp->fullBit(c+7465,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c[0]));
        vcdp->fullBus(c+7473,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c[0]),16);
        vcdp->fullBus(c+7481,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c[0]),20);
        vcdp->fullArray(c+7489,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c[0]),128);
        vcdp->fullBit(c+7521,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid 
                                             >> (0x3fU 
                                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                                 [0U]))))));
        vcdp->fullBit(c+7529,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty 
                                             >> (0x3fU 
                                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                                 [0U]))))));
        vcdp->fullBus(c+7537,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                [0U])]),16);
        vcdp->fullBus(c+7545,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag
                              [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                [0U])]),20);
        __Vtemp585[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
              [0U])][0U];
        __Vtemp585[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
              [0U])][1U];
        __Vtemp585[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
              [0U])][2U];
        __Vtemp585[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
              [0U])][3U];
        vcdp->fullArray(c+7553,(__Vtemp585),128);
        vcdp->fullBit(c+7585,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                              [0U]));
        vcdp->fullBit(c+7593,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c
                              [0U]));
        vcdp->fullBus(c+7601,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable),16);
        vcdp->fullArray(c+7609,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__data_write),128);
        vcdp->fullBit(c+7641,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__snoop_hit_no_pending));
        vcdp->fullBit(c+7649,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match));
        vcdp->fullBit(c+7657,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill));
        vcdp->fullBus(c+7665,((0xfffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__addr_st1
                                           [0U] >> 6U))),20);
        vcdp->fullBus(c+7673,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__we),16);
        vcdp->fullBit(c+7681,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__should_write));
        vcdp->fullBit(c+7689,((((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                                 [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1
                                         [0U])) & (~ 
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                                   [0U])) 
                               & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                                  [0U]))));
        vcdp->fullBit(c+7697,(((((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                                  [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1
                                          [0U])) & 
                                 vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                 [0U]) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                                          [0U])) & 
                               (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match)))));
        vcdp->fullBit(c+7705,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss));
        vcdp->fullBit(c+7713,((((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                  & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_snp_st1
                                     [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__is_fill_st1
                                               [0U])) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__valid_st1
                                [0U]) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss)))));
        vcdp->fullBit(c+7721,(((0U == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->fullBit(c+7729,(((1U == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->fullBit(c+7737,(((2U == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->fullBit(c+7745,(((3U == vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->fullBit(c+7753,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable))));
        vcdp->fullArray(c+7761,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT____Vcellinp__s0_1_c0__in),166);
        vcdp->fullArray(c+7809,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__st_1e_2__in),316);
        vcdp->fullBus(c+7889,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready),16);
        vcdp->fullBus(c+7897,((0xffffU & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                                           & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                          << (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr)))),16);
        vcdp->fullBus(c+7905,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match),16);
        vcdp->fullBit(c+7913,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push));
        vcdp->fullBit(c+7921,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_pop));
        vcdp->fullBit(c+7929,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__miss_add_is_mrvq))));
        vcdp->fullBit(c+7937,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__increment_head));
        vcdp->fullBit(c+7945,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready))));
        vcdp->fullBit(c+7953,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__mrvq_init_ready_state_st2))));
        vcdp->fullArray(c+7961,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__cwb_queue__data_in),76);
        vcdp->fullArray(c+7985,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out),76);
        vcdp->fullBit(c+8009,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__writing));
        vcdp->fullArray(c+8017,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellinp__dwb_queue__data_in),200);
        vcdp->fullArray(c+8073,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out),200);
        vcdp->fullBit(c+8129,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__writing));
        vcdp->fullBit(c+8137,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop));
        vcdp->fullBus(c+8145,((0x3ffffffU & (IData)(
                                                    (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                     >> 0x1dU)))),26);
        vcdp->fullBit(c+8153,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                             >> 0x1cU)))));
        vcdp->fullBus(c+8161,((0xfffffffU & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out))),28);
        vcdp->fullBit(c+8169,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop));
        vcdp->fullBus(c+8177,((0x3ffffffU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])),26);
        __Vtemp586[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U];
        __Vtemp586[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U];
        __Vtemp586[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U];
        __Vtemp586[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U];
        vcdp->fullArray(c+8185,(__Vtemp586),128);
        vcdp->fullBit(c+8217,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop));
        vcdp->fullBit(c+8225,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request)))));
        vcdp->fullBit(c+8233,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_has_request));
        vcdp->fullBus(c+8241,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index),2);
        vcdp->fullBit(c+8249,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_req_rw_st0));
        vcdp->fullBus(c+8257,((0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen) 
                                       >> (0xfU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
                                                   << 2U))))),4);
        vcdp->fullBus(c+8265,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_req_addr_st0),30);
        vcdp->fullBus(c+8273,((((0U == (0x1fU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__qual_request_index) 
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
        vcdp->fullBit(c+8281,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_pop));
        vcdp->fullBit(c+8289,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible));
        vcdp->fullBit(c+8297,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_rw_st0));
        vcdp->fullBit(c+8305,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match))));
        vcdp->fullBit(c+8313,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_because_miss) 
                               & (((0x3ffffffU & ((
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                   << 7U) 
                                                  | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                     >> 0x19U))) 
                                   == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                   [0U]) & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                               >> 0x14U))))));
        vcdp->fullBit(c+8321,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__force_request_miss_st1e));
        vcdp->fullBit(c+8329,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__recover_mrvq_state_st2));
        vcdp->fullBit(c+8337,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_push_stall));
        vcdp->fullBit(c+8345,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwbq_push_stall));
        vcdp->fullBit(c+8353,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_push_stall));
        vcdp->fullBit(c+8361,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dram_fill_req_stall));
        vcdp->fullBit(c+8369,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__stall_bank_pipe));
        vcdp->fullBit(c+8377,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_in_pipe));
        vcdp->fullBit(c+8385,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1[0]));
        vcdp->fullBit(c+8393,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__going_to_write_st1[0]));
        vcdp->fullBit(c+8401,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual));
        vcdp->fullBit(c+8409,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop_unqual));
        vcdp->fullBit(c+8417,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop_unqual));
        vcdp->fullBit(c+8425,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop) 
                                 | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_pop)) 
                                | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop)) 
                               | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop))));
        vcdp->fullBus(c+8433,((0x3ffffffU & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
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
        vcdp->fullBus(c+8441,((3U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop_unqual)
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
        vcdp->fullBus(c+8449,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
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
        __Vtemp591[0U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[0U]
                           : 0x39U);
        __Vtemp591[1U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[1U]
                           : 0U);
        __Vtemp591[2U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[2U]
                           : 0U);
        __Vtemp591[3U] = ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
                           ? vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[3U]
                           : 0U);
        vcdp->fullArray(c+8457,(__Vtemp591),128);
        vcdp->fullQuad(c+8489,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
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
        vcdp->fullBit(c+8505,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)
                                ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible) 
                                         & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_rw_st0))
                                         ? 1U : (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_pop_unqual) 
                                                  & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__reqq_req_rw_st0))
                                                  ? 1U
                                                  : 0U)))));
        vcdp->fullBit(c+8513,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                               [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                               >> 1U))
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? 1U : 0U)))));
        vcdp->fullBit(c+8521,((1U & ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__dequeue_possible)
                                      ? (1U & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])
                                      : ((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snrq_pop_unqual)
                                          ? (1U & (IData)(
                                                          (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out 
                                                           >> 0x1cU)))
                                          : 0U)))));
        vcdp->fullBit(c+8529,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1[0]));
        vcdp->fullBus(c+8537,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1[0]),26);
        vcdp->fullBus(c+8545,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1[0]),2);
        vcdp->fullBus(c+8553,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writeword_st1[0]),32);
        vcdp->fullQuad(c+8561,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__inst_meta_st1[0]),49);
        vcdp->fullArray(c+8577,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writedata_st1[0]),128);
        vcdp->fullBit(c+8609,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1[0]));
        vcdp->fullBit(c+8617,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_invalidate_st1[0]));
        vcdp->fullBit(c+8625,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_mrvq_st1[0]));
        __Vtemp597[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp597[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp597[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp597[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->fullBus(c+8633,((((0U == (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                                 [0U] 
                                                 << 5U)))
                                 ? 0U : (__Vtemp597[
                                         ((IData)(1U) 
                                          + (3U & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                             [0U]))] 
                                         << ((IData)(0x20U) 
                                             - (0x1fU 
                                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                                   [0U] 
                                                   << 5U))))) 
                               | (__Vtemp597[(3U & 
                                              vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                              [0U])] 
                                  >> (0x1fU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                               [0U] 
                                               << 5U))))),32);
        __Vtemp598[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][0U];
        __Vtemp598[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][1U];
        __Vtemp598[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][2U];
        __Vtemp598[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c
            [0U][3U];
        vcdp->fullArray(c+8641,(__Vtemp598),128);
        vcdp->fullBus(c+8673,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c
                              [0U]),20);
        vcdp->fullBit(c+8681,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_st1e));
        vcdp->fullBit(c+8689,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dirty_st1e));
        vcdp->fullBus(c+8697,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c
                              [0U]),16);
        vcdp->fullQuad(c+8705,((VL_ULL(0x3ffffffffff) 
                                & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__inst_meta_st1
                                   [0U] >> 7U))),42);
        vcdp->fullBus(c+8721,((3U & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__inst_meta_st1
                                            [0U]))),2);
        vcdp->fullBit(c+8729,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__inst_meta_st1
                                             [0U] >> 6U)))));
        vcdp->fullBus(c+8737,((0xfU & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__inst_meta_st1
                                               [0U] 
                                               >> 2U)))),4);
        vcdp->fullBit(c+8745,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dirty_st1e))));
        vcdp->fullBit(c+8753,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1
                              [0U]));
        vcdp->fullBit(c+8761,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_invalidate_st1
                              [0U]));
        vcdp->fullBit(c+8769,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_to_mrvq_st1e));
        vcdp->fullBit(c+8777,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_to_mrvq_st1e) 
                               | ((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                    & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1
                                       [0U])) & (~ 
                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                                                 [0U])) 
                                  & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                                  [0U]))));
        vcdp->fullBit(c+8785,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_because_miss));
        vcdp->fullBit(c+8793,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                              [0U]));
        vcdp->fullBit(c+8801,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_mrvq_st1
                              [0U]));
        vcdp->fullBit(c+8809,((((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                                 [0U] & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_mrvq_st1
                                 [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__recover_mrvq_state_st2)) 
                               & ((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 7U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x19U))) 
                                  == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                  [0U]))));
        vcdp->fullBus(c+8817,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                              [0U]),26);
        vcdp->fullBit(c+8825,((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                               [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                                       [0U]))));
        vcdp->fullBit(c+8833,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_init_ready_state_st2));
        vcdp->fullBit(c+8841,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_unqual) 
                                & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfpq_pop_unqual)) 
                               & ((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                  << 7U) 
                                                 | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                    >> 0x19U))) 
                                  == (0x3ffffffU & 
                                      vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out[4U])))));
        vcdp->fullBit(c+8849,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_unqual) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                                [0U]) & ((0x3ffffffU 
                                          & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                              << 7U) 
                                             | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                >> 0x19U))) 
                                         == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                         [0U]))));
        vcdp->fullBit(c+8857,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_unqual));
        vcdp->fullBit(c+8865,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add));
        vcdp->fullBit(c+8873,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_is_mrvq));
        vcdp->fullBit(c+8881,(((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwbq_push_unqual) 
                                 & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                                & (~ (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                      >> 6U))) & (~ 
                                                  (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_push_stall) 
                                                    | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                                   | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->fullBit(c+8889,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwbq_push_unqual));
        vcdp->fullBit(c+8897,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_unqual) 
                               & ((~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                  | ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU) & (~ 
                                                   (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                                    >> 0x1bU)))))));
        vcdp->fullBit(c+8905,((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_push_unqual) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r))) 
                               & (~ (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwbq_push_stall) 
                                      | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_push_stall)) 
                                     | (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dram_fill_req_stall))))));
        vcdp->fullBit(c+8913,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_is_dwb_in));
        vcdp->fullBit(c+8921,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_is_snp_in));
        vcdp->fullBit(c+8929,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 7U))));
        vcdp->fullBit(c+8937,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out[6U] 
                                     >> 6U))));
        vcdp->fullBit(c+8945,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_push_unqual));
        vcdp->fullQuad(c+8953,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__snp_req_queue__data_out),55);
        vcdp->fullBit(c+8969,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__reading));
        vcdp->fullArray(c+8977,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dfp_queue__data_out),154);
        vcdp->fullBit(c+9017,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__reading));
        vcdp->fullBus(c+9025,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x16U))),4);
        vcdp->fullBus(c+9033,((0xfU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                       >> 0x12U))),4);
        vcdp->fullBus(c+9041,((0xffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                          >> 2U))),16);
        __Vtemp601[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                                        >> 0xaU));
        __Vtemp601[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[6U] 
                                        >> 0xaU));
        __Vtemp601[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[7U] 
                                        >> 0xaU));
        __Vtemp601[3U] = (0xffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                        << 0x16U) | 
                                       (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[8U] 
                                        >> 0xaU)));
        vcdp->fullArray(c+9049,(__Vtemp601),120);
        __Vtemp602[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U] 
                                        >> 0xaU));
        __Vtemp602[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[2U] 
                                        >> 0xaU));
        __Vtemp602[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[3U] 
                                        >> 0xaU));
        __Vtemp602[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[5U] 
                           << 0x16U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[4U] 
                                        >> 0xaU));
        vcdp->fullArray(c+9081,(__Vtemp602),128);
        vcdp->fullQuad(c+9113,((VL_ULL(0x3ffffffffff) 
                                & (((QData)((IData)(
                                                    vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[1U])) 
                                    << 0x20U) | (QData)((IData)(
                                                                vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[0U]))))),42);
        vcdp->fullBit(c+9129,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty));
        vcdp->fullBit(c+9137,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__pop_qual));
        vcdp->fullBus(c+9145,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out[9U] 
                                        >> 0x16U) & 
                                       VL_NEGATE_I((IData)(
                                                           (1U 
                                                            & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__out_empty)))))))),4);
        vcdp->fullArray(c+9153,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT____Vcellout__reqq_queue__data_out),314);
        vcdp->fullBit(c+9233,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__reading));
        vcdp->fullBus(c+9241,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__grant_onehot_r),4);
        vcdp->fullBus(c+9249,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__sel_bank__DOT__genblk2__DOT__priority_encoder__DOT__i),32);
        vcdp->fullArray(c+9257,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__s0_1_c0__in),243);
        vcdp->fullBus(c+9321,((0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                               [0U])),6);
        vcdp->fullBit(c+9329,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                              [0U]));
        vcdp->fullBus(c+9337,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writeword_st1
                              [0U]),32);
        __Vtemp603[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writedata_st1
            [0U][0U];
        __Vtemp603[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writedata_st1
            [0U][1U];
        __Vtemp603[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writedata_st1
            [0U][2U];
        __Vtemp603[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__writedata_st1
            [0U][3U];
        vcdp->fullArray(c+9345,(__Vtemp603),128);
        vcdp->fullBus(c+9377,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                              [0U]),2);
        vcdp->fullBit(c+9385,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c[0]));
        vcdp->fullBit(c+9393,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c[0]));
        vcdp->fullBus(c+9401,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirtyb_st1c[0]),16);
        vcdp->fullBus(c+9409,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_tag_st1c[0]),20);
        vcdp->fullArray(c+9417,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_data_st1c[0]),128);
        vcdp->fullBit(c+9449,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid 
                                             >> (0x3fU 
                                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                                 [0U]))))));
        vcdp->fullBit(c+9457,((1U & (IData)((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty 
                                             >> (0x3fU 
                                                 & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                                 [0U]))))));
        vcdp->fullBus(c+9465,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirtyb
                              [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                [0U])]),16);
        vcdp->fullBus(c+9473,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__tag
                              [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                [0U])]),20);
        __Vtemp604[0U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
              [0U])][0U];
        __Vtemp604[1U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
              [0U])][1U];
        __Vtemp604[2U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
              [0U])][2U];
        __Vtemp604[3U] = vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__data
            [(0x3fU & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
              [0U])][3U];
        vcdp->fullArray(c+9481,(__Vtemp604),128);
        vcdp->fullBit(c+9513,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                              [0U]));
        vcdp->fullBit(c+9521,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_dirty_st1c
                              [0U]));
        vcdp->fullBus(c+9529,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable),16);
        vcdp->fullArray(c+9537,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__data_write),128);
        vcdp->fullBit(c+9569,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__snoop_hit_no_pending));
        vcdp->fullBit(c+9577,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match));
        vcdp->fullBit(c+9585,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__real_writefill));
        vcdp->fullBus(c+9593,((0xfffffU & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__addr_st1
                                           [0U] >> 6U))),20);
        vcdp->fullBus(c+9601,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__we),16);
        vcdp->fullBit(c+9609,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__should_write));
        vcdp->fullBit(c+9617,((((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                                 [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1
                                         [0U])) & (~ 
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                                   [0U])) 
                               & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                                  [0U]))));
        vcdp->fullBit(c+9625,(((((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                                  [0U] & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1
                                          [0U])) & 
                                 vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__read_valid_st1c
                                 [0U]) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                                          [0U])) & 
                               (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tags_match)))));
        vcdp->fullBit(c+9633,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss));
        vcdp->fullBit(c+9641,((((((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__force_request_miss_st1e) 
                                  & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_snp_st1
                                     [0U])) & (~ vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__is_fill_st1
                                               [0U])) 
                                & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__valid_st1
                                [0U]) & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__real_miss)))));
        vcdp->fullBit(c+9649,(((0U == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->fullBit(c+9657,(((1U == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->fullBit(c+9665,(((2U == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->fullBit(c+9673,(((3U == vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__wsel_st1
                                [0U]) & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__should_write))));
        vcdp->fullBit(c+9681,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__use_write_enable))));
        vcdp->fullArray(c+9689,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT____Vcellinp__s0_1_c0__in),166);
        vcdp->fullArray(c+9737,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__st_1e_2__in),316);
        vcdp->fullBus(c+9817,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready),16);
        vcdp->fullBus(c+9825,((0xffffU & (((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                                           & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_init_ready_state_st2)) 
                                          << (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr)))),16);
        vcdp->fullBus(c+9833,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_address_match),16);
        vcdp->fullBit(c+9841,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push));
        vcdp->fullBit(c+9849,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_pop));
        vcdp->fullBit(c+9857,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__miss_add_is_mrvq))));
        vcdp->fullBit(c+9865,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__increment_head));
        vcdp->fullBit(c+9873,((0U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__make_ready))));
        vcdp->fullBit(c+9881,(((IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__mrvq_push) 
                               & (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__mrvq_init_ready_state_st2))));
        vcdp->fullArray(c+9889,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__cwb_queue__data_in),76);
        vcdp->fullArray(c+9913,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__cwb_queue__data_out),76);
        vcdp->fullBit(c+9937,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__writing));
        vcdp->fullArray(c+9945,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellinp__dwb_queue__data_in),200);
        vcdp->fullArray(c+10001,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT____Vcellout__dwb_queue__data_out),200);
        vcdp->fullBit(c+10057,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__writing));
        vcdp->fullBit(c+10065,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBit(c+10073,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->fullBit(c+10081,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBus(c+10089,((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                               << 7U) 
                                              | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                 >> 0x19U)))),26);
        vcdp->fullBit(c+10097,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBit(c+10105,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBit(c+10113,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->fullBit(c+10121,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBus(c+10129,((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                               << 7U) 
                                              | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                 >> 0x19U)))),26);
        vcdp->fullBit(c+10137,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBit(c+10145,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBit(c+10153,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->fullBit(c+10161,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBus(c+10169,((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                               << 7U) 
                                              | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                 >> 0x19U)))),26);
        vcdp->fullBit(c+10177,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBit(c+10185,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBit(c+10193,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r)))));
        vcdp->fullBit(c+10201,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBus(c+10209,((0x3ffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                               << 7U) 
                                              | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                 >> 0x19U)))),26);
        vcdp->fullBit(c+10217,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBit(c+10225,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r)))));
        vcdp->fullBus(c+10233,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__use_addr),28);
        vcdp->fullBit(c+10241,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullBus(c+10249,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__use_valid),2);
        vcdp->fullBit(c+10257,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__size_r));
        vcdp->fullBus(c+10265,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__genblk2__DOT__head_r),28);
        vcdp->fullBit(c+10273,((1U & (~ (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__size_r)))));
        vcdp->fullBit(c+10281,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__size_r));
        vcdp->fullBus(c+10289,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bank_dram_fill_req_valid),4);
        vcdp->fullArray(c+10297,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bank_dram_fill_req_addr),112);
        vcdp->fullBit(c+10329,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+10337,((1U & (~ (IData)((0U 
                                                 != (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__use_per_bank_dram_fill_req_valid)))))));
        vcdp->fullBus(c+10345,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__size_r),3);
        vcdp->fullArray(c+10353,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__data[0]),116);
        vcdp->fullArray(c+10357,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__data[1]),116);
        vcdp->fullArray(c+10361,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__data[2]),116);
        vcdp->fullArray(c+10365,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__data[3]),116);
        vcdp->fullArray(c+10481,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),116);
        vcdp->fullArray(c+10513,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),116);
        vcdp->fullBus(c+10545,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->fullBus(c+10553,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->fullBus(c+10561,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->fullBit(c+10569,(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__dram_fill_arb__DOT__dfqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullBus(c+10577,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__requests_use),4);
        vcdp->fullBit(c+10585,((0U == (IData)(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__requests_use))));
        vcdp->fullBus(c+10593,(vlTOPp->VX_cache__DOT__cache_core_rsp_merge__DOT__sel_bank__DOT__genblk2__DOT__refill_original),4);
        vcdp->fullBit(c+10601,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+10609,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullBit(c+10617,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+10625,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullBit(c+10633,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullQuad(c+10641,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag),42);
        vcdp->fullBit(c+10657,((0x10U == (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->fullBit(c+10665,((0xbU < (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->fullBus(c+10673,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                       [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                       << 0xdU) | (
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                   >> 0x13U)))),2);
        vcdp->fullBus(c+10681,(((0x19fU >= (0x1ffU 
                                            & ((IData)(0x1aU) 
                                               * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                 ? (0x3ffffffU & ((
                                                   (0U 
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
        vcdp->fullBus(c+10689,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                       [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                       << 0x1eU) | 
                                      (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                       [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                       >> 2U)))),2);
        vcdp->fullBus(c+10697,(((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                 [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                 << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                             [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                             >> 0x15U))),32);
        vcdp->fullQuad(c+10705,((VL_ULL(0x3ffffffffff) 
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
        vcdp->fullBus(c+10721,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                         << 0x1cU) 
                                        | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                           [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                           >> 4U)))),4);
        vcdp->fullBit(c+10729,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                      >> 1U))));
        vcdp->fullBit(c+10737,((1U & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                [vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])));
        vcdp->fullBus(c+10745,((3U & vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])),2);
        vcdp->fullQuad(c+10753,((VL_ULL(0x3ffffffffff) 
                                 & (((QData)((IData)(
                                                     vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U])) 
                                     << 0x39U) | (((QData)((IData)(
                                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                                   << 0x19U) 
                                                  | ((QData)((IData)(
                                                                     vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])) 
                                                     >> 7U))))),42);
        vcdp->fullBit(c+10769,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                      >> 6U))));
        vcdp->fullBus(c+10777,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                         << 0x1eU) 
                                        | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                           >> 2U)))),4);
        vcdp->fullBit(c+10785,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x14U))));
        vcdp->fullBit(c+10793,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x13U))));
        vcdp->fullBus(c+10801,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                       << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                 >> 0x17U)))),2);
        vcdp->fullBus(c+10809,(((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                 << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                           >> 0x17U))),32);
        vcdp->fullBus(c+10817,(((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                 << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                                           >> 0x17U))),32);
        __Vtemp612[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 0x17U));
        __Vtemp612[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                     >> 0x17U));
        __Vtemp612[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                                     >> 0x17U));
        __Vtemp612[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                                     >> 0x17U));
        vcdp->fullArray(c+10825,(__Vtemp612),128);
        vcdp->fullBit(c+10857,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                      >> 2U))));
        vcdp->fullBit(c+10865,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                      >> 1U))));
        vcdp->fullBus(c+10873,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                            << 0xfU) 
                                           | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                              >> 0x11U)))),16);
        vcdp->fullQuad(c+10881,((VL_ULL(0x1ffffffffffff) 
                                 & (((QData)((IData)(
                                                     vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                     << 0x20U) | (QData)((IData)(
                                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U]))))),49);
        vcdp->fullBus(c+10897,((0xfffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                             << 0x1dU) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                               >> 3U)))),20);
        vcdp->fullBit(c+10905,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x15U))));
        vcdp->fullBit(c+10913,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x17U))));
        vcdp->fullBit(c+10921,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x16U))));
        vcdp->fullBit(c+10929,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x18U))));
        vcdp->fullBit(c+10937,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU))));
        vcdp->fullBit(c+10945,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1bU))));
        vcdp->fullBit(c+10953,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x19U))));
        vcdp->fullBit(c+10961,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+10969,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullBit(c+10977,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+10985,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullBus(c+10993,(((0x3ffffc0U & (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                               << 3U)) 
                                | (0x3fU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                             << 7U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                               >> 0x19U))))),26);
        vcdp->fullBus(c+11001,((0xfffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                               << 0x19U) 
                                              | (vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                                 >> 7U)))),28);
        vcdp->fullBit(c+11009,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwbq_dual_valid_sel));
        vcdp->fullBus(c+11017,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__size_r),5);
        vcdp->fullQuad(c+11025,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[0]),55);
        vcdp->fullQuad(c+11027,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[1]),55);
        vcdp->fullQuad(c+11029,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[2]),55);
        vcdp->fullQuad(c+11031,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[3]),55);
        vcdp->fullQuad(c+11033,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[4]),55);
        vcdp->fullQuad(c+11035,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[5]),55);
        vcdp->fullQuad(c+11037,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[6]),55);
        vcdp->fullQuad(c+11039,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[7]),55);
        vcdp->fullQuad(c+11041,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[8]),55);
        vcdp->fullQuad(c+11043,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[9]),55);
        vcdp->fullQuad(c+11045,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[10]),55);
        vcdp->fullQuad(c+11047,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[11]),55);
        vcdp->fullQuad(c+11049,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[12]),55);
        vcdp->fullQuad(c+11051,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[13]),55);
        vcdp->fullQuad(c+11053,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[14]),55);
        vcdp->fullQuad(c+11055,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[15]),55);
        vcdp->fullQuad(c+11281,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),55);
        vcdp->fullQuad(c+11297,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),55);
        vcdp->fullBus(c+11313,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->fullBus(c+11321,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->fullBus(c+11329,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->fullBit(c+11337,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullBus(c+11345,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__size_r),5);
        vcdp->fullArray(c+11353,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[0]),154);
        vcdp->fullArray(c+11358,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[1]),154);
        vcdp->fullArray(c+11363,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[2]),154);
        vcdp->fullArray(c+11368,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[3]),154);
        vcdp->fullArray(c+11373,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[4]),154);
        vcdp->fullArray(c+11378,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[5]),154);
        vcdp->fullArray(c+11383,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[6]),154);
        vcdp->fullArray(c+11388,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[7]),154);
        vcdp->fullArray(c+11393,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[8]),154);
        vcdp->fullArray(c+11398,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[9]),154);
        vcdp->fullArray(c+11403,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[10]),154);
        vcdp->fullArray(c+11408,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[11]),154);
        vcdp->fullArray(c+11413,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[12]),154);
        vcdp->fullArray(c+11418,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[13]),154);
        vcdp->fullArray(c+11423,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[14]),154);
        vcdp->fullArray(c+11428,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[15]),154);
        vcdp->fullArray(c+11993,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),154);
        vcdp->fullArray(c+12033,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),154);
        vcdp->fullBus(c+12073,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->fullBus(c+12081,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->fullBus(c+12089,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->fullBit(c+12097,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullBus(c+12105,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids),4);
        vcdp->fullBus(c+12113,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_rw),4);
        vcdp->fullBus(c+12121,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen),16);
        vcdp->fullArray(c+12129,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_addr),120);
        vcdp->fullArray(c+12161,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata),128);
        vcdp->fullBit(c+12193,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+12201,((1U & (~ (IData)((0U 
                                                 != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids)))))));
        vcdp->fullBus(c+12209,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__size_r),3);
        vcdp->fullArray(c+12217,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[0]),314);
        vcdp->fullArray(c+12227,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[1]),314);
        vcdp->fullArray(c+12237,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[2]),314);
        vcdp->fullArray(c+12247,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[3]),314);
        vcdp->fullArray(c+12537,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),314);
        vcdp->fullArray(c+12617,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),314);
        vcdp->fullBus(c+12697,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->fullBus(c+12705,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->fullBus(c+12713,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->fullBit(c+12721,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullArray(c+12729,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__s0_1_c0__DOT__value),243);
        vcdp->fullQuad(c+12793,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty),64);
        vcdp->fullQuad(c+12809,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid),64);
        vcdp->fullBus(c+12825,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__i),32);
        vcdp->fullBus(c+12833,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__j),32);
        vcdp->fullArray(c+12841,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__tag_data_access__DOT__s0_1_c0__DOT__value),166);
        vcdp->fullArray(c+12889,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__st_1e_2__DOT__value),316);
        vcdp->fullArray(c+12969,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[0]),85);
        vcdp->fullArray(c+12972,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[1]),85);
        vcdp->fullArray(c+12975,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[2]),85);
        vcdp->fullArray(c+12978,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[3]),85);
        vcdp->fullArray(c+12981,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[4]),85);
        vcdp->fullArray(c+12984,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[5]),85);
        vcdp->fullArray(c+12987,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[6]),85);
        vcdp->fullArray(c+12990,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[7]),85);
        vcdp->fullArray(c+12993,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[8]),85);
        vcdp->fullArray(c+12996,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[9]),85);
        vcdp->fullArray(c+12999,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[10]),85);
        vcdp->fullArray(c+13002,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[11]),85);
        vcdp->fullArray(c+13005,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[12]),85);
        vcdp->fullArray(c+13008,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[13]),85);
        vcdp->fullArray(c+13011,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[14]),85);
        vcdp->fullArray(c+13014,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[15]),85);
        vcdp->fullArray(c+13353,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table),416);
        vcdp->fullBus(c+13457,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_table),16);
        vcdp->fullBus(c+13465,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__ready_table),16);
        vcdp->fullBus(c+13473,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr),4);
        vcdp->fullBus(c+13481,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__head_ptr),4);
        vcdp->fullBus(c+13489,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr),4);
        vcdp->fullBus(c+13497,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size),5);
        vcdp->fullBit(c+13505,((0x10U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->fullBus(c+13513,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__size_r),3);
        vcdp->fullArray(c+13521,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[0]),76);
        vcdp->fullArray(c+13524,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[1]),76);
        vcdp->fullArray(c+13527,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[2]),76);
        vcdp->fullArray(c+13530,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[3]),76);
        vcdp->fullArray(c+13617,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),76);
        vcdp->fullArray(c+13641,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),76);
        vcdp->fullBus(c+13665,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->fullBus(c+13673,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->fullBus(c+13681,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->fullBit(c+13689,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullBus(c+13697,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__size_r),3);
        vcdp->fullArray(c+13705,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[0]),200);
        vcdp->fullArray(c+13712,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[1]),200);
        vcdp->fullArray(c+13719,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[2]),200);
        vcdp->fullArray(c+13726,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[3]),200);
        vcdp->fullArray(c+13929,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),200);
        vcdp->fullArray(c+13985,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),200);
        vcdp->fullBus(c+14041,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->fullBus(c+14049,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->fullBus(c+14057,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->fullBit(c+14065,(vlTOPp->VX_cache__DOT__genblk5__BRA__0__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullBit(c+14073,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+14081,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullBit(c+14089,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+14097,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullBit(c+14105,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullQuad(c+14113,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag),42);
        vcdp->fullBit(c+14129,((0x10U == (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->fullBit(c+14137,((0xbU < (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->fullBus(c+14145,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                       [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                       << 0xdU) | (
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                   >> 0x13U)))),2);
        vcdp->fullBus(c+14153,(((0x19fU >= (0x1ffU 
                                            & ((IData)(0x1aU) 
                                               * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                 ? (0x3ffffffU & ((
                                                   (0U 
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
        vcdp->fullBus(c+14161,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                       [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                       << 0x1eU) | 
                                      (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                       [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                       >> 2U)))),2);
        vcdp->fullBus(c+14169,(((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                 [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                 << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                             [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                             >> 0x15U))),32);
        vcdp->fullQuad(c+14177,((VL_ULL(0x3ffffffffff) 
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
        vcdp->fullBus(c+14193,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                         << 0x1cU) 
                                        | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                           [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                           >> 4U)))),4);
        vcdp->fullBit(c+14201,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                      >> 1U))));
        vcdp->fullBit(c+14209,((1U & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                [vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])));
        vcdp->fullBus(c+14217,((3U & vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])),2);
        vcdp->fullQuad(c+14225,((VL_ULL(0x3ffffffffff) 
                                 & (((QData)((IData)(
                                                     vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U])) 
                                     << 0x39U) | (((QData)((IData)(
                                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                                   << 0x19U) 
                                                  | ((QData)((IData)(
                                                                     vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])) 
                                                     >> 7U))))),42);
        vcdp->fullBit(c+14241,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                      >> 6U))));
        vcdp->fullBus(c+14249,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                         << 0x1eU) 
                                        | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                           >> 2U)))),4);
        vcdp->fullBit(c+14257,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x14U))));
        vcdp->fullBit(c+14265,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x13U))));
        vcdp->fullBus(c+14273,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                       << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                 >> 0x17U)))),2);
        vcdp->fullBus(c+14281,(((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                 << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                           >> 0x17U))),32);
        vcdp->fullBus(c+14289,(((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                 << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                                           >> 0x17U))),32);
        __Vtemp620[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 0x17U));
        __Vtemp620[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                     >> 0x17U));
        __Vtemp620[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                                     >> 0x17U));
        __Vtemp620[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                                     >> 0x17U));
        vcdp->fullArray(c+14297,(__Vtemp620),128);
        vcdp->fullBit(c+14329,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                      >> 2U))));
        vcdp->fullBit(c+14337,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                      >> 1U))));
        vcdp->fullBus(c+14345,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                            << 0xfU) 
                                           | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                              >> 0x11U)))),16);
        vcdp->fullQuad(c+14353,((VL_ULL(0x1ffffffffffff) 
                                 & (((QData)((IData)(
                                                     vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                     << 0x20U) | (QData)((IData)(
                                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U]))))),49);
        vcdp->fullBus(c+14369,((0xfffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                             << 0x1dU) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                               >> 3U)))),20);
        vcdp->fullBit(c+14377,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x15U))));
        vcdp->fullBit(c+14385,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x17U))));
        vcdp->fullBit(c+14393,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x16U))));
        vcdp->fullBit(c+14401,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x18U))));
        vcdp->fullBit(c+14409,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU))));
        vcdp->fullBit(c+14417,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1bU))));
        vcdp->fullBit(c+14425,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x19U))));
        vcdp->fullBit(c+14433,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+14441,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullBit(c+14449,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+14457,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullBus(c+14465,(((0x3ffffc0U & (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                               << 3U)) 
                                | (0x3fU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                             << 7U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                               >> 0x19U))))),26);
        vcdp->fullBus(c+14473,((0xfffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                               << 0x19U) 
                                              | (vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                                 >> 7U)))),28);
        vcdp->fullBit(c+14481,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwbq_dual_valid_sel));
        vcdp->fullBus(c+14489,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__size_r),5);
        vcdp->fullQuad(c+14497,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[0]),55);
        vcdp->fullQuad(c+14499,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[1]),55);
        vcdp->fullQuad(c+14501,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[2]),55);
        vcdp->fullQuad(c+14503,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[3]),55);
        vcdp->fullQuad(c+14505,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[4]),55);
        vcdp->fullQuad(c+14507,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[5]),55);
        vcdp->fullQuad(c+14509,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[6]),55);
        vcdp->fullQuad(c+14511,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[7]),55);
        vcdp->fullQuad(c+14513,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[8]),55);
        vcdp->fullQuad(c+14515,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[9]),55);
        vcdp->fullQuad(c+14517,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[10]),55);
        vcdp->fullQuad(c+14519,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[11]),55);
        vcdp->fullQuad(c+14521,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[12]),55);
        vcdp->fullQuad(c+14523,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[13]),55);
        vcdp->fullQuad(c+14525,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[14]),55);
        vcdp->fullQuad(c+14527,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[15]),55);
        vcdp->fullQuad(c+14753,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),55);
        vcdp->fullQuad(c+14769,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),55);
        vcdp->fullBus(c+14785,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->fullBus(c+14793,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->fullBus(c+14801,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->fullBit(c+14809,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullBus(c+14817,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__size_r),5);
        vcdp->fullArray(c+14825,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[0]),154);
        vcdp->fullArray(c+14830,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[1]),154);
        vcdp->fullArray(c+14835,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[2]),154);
        vcdp->fullArray(c+14840,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[3]),154);
        vcdp->fullArray(c+14845,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[4]),154);
        vcdp->fullArray(c+14850,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[5]),154);
        vcdp->fullArray(c+14855,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[6]),154);
        vcdp->fullArray(c+14860,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[7]),154);
        vcdp->fullArray(c+14865,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[8]),154);
        vcdp->fullArray(c+14870,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[9]),154);
        vcdp->fullArray(c+14875,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[10]),154);
        vcdp->fullArray(c+14880,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[11]),154);
        vcdp->fullArray(c+14885,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[12]),154);
        vcdp->fullArray(c+14890,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[13]),154);
        vcdp->fullArray(c+14895,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[14]),154);
        vcdp->fullArray(c+14900,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[15]),154);
        vcdp->fullArray(c+15465,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),154);
        vcdp->fullArray(c+15505,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),154);
        vcdp->fullBus(c+15545,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->fullBus(c+15553,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->fullBus(c+15561,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->fullBit(c+15569,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullBus(c+15577,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids),4);
        vcdp->fullBus(c+15585,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_rw),4);
        vcdp->fullBus(c+15593,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen),16);
        vcdp->fullArray(c+15601,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_addr),120);
        vcdp->fullArray(c+15633,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata),128);
        vcdp->fullBit(c+15665,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+15673,((1U & (~ (IData)((0U 
                                                 != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids)))))));
        vcdp->fullBus(c+15681,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__size_r),3);
        vcdp->fullArray(c+15689,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[0]),314);
        vcdp->fullArray(c+15699,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[1]),314);
        vcdp->fullArray(c+15709,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[2]),314);
        vcdp->fullArray(c+15719,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[3]),314);
        vcdp->fullArray(c+16009,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),314);
        vcdp->fullArray(c+16089,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),314);
        vcdp->fullBus(c+16169,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->fullBus(c+16177,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->fullBus(c+16185,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->fullBit(c+16193,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullArray(c+16201,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__s0_1_c0__DOT__value),243);
        vcdp->fullQuad(c+16265,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty),64);
        vcdp->fullQuad(c+16281,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid),64);
        vcdp->fullBus(c+16297,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__i),32);
        vcdp->fullBus(c+16305,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__j),32);
        vcdp->fullArray(c+16313,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__tag_data_access__DOT__s0_1_c0__DOT__value),166);
        vcdp->fullArray(c+16361,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__st_1e_2__DOT__value),316);
        vcdp->fullArray(c+16441,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[0]),85);
        vcdp->fullArray(c+16444,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[1]),85);
        vcdp->fullArray(c+16447,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[2]),85);
        vcdp->fullArray(c+16450,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[3]),85);
        vcdp->fullArray(c+16453,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[4]),85);
        vcdp->fullArray(c+16456,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[5]),85);
        vcdp->fullArray(c+16459,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[6]),85);
        vcdp->fullArray(c+16462,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[7]),85);
        vcdp->fullArray(c+16465,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[8]),85);
        vcdp->fullArray(c+16468,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[9]),85);
        vcdp->fullArray(c+16471,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[10]),85);
        vcdp->fullArray(c+16474,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[11]),85);
        vcdp->fullArray(c+16477,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[12]),85);
        vcdp->fullArray(c+16480,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[13]),85);
        vcdp->fullArray(c+16483,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[14]),85);
        vcdp->fullArray(c+16486,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[15]),85);
        vcdp->fullArray(c+16825,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table),416);
        vcdp->fullBus(c+16929,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_table),16);
        vcdp->fullBus(c+16937,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__ready_table),16);
        vcdp->fullBus(c+16945,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr),4);
        vcdp->fullBus(c+16953,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__head_ptr),4);
        vcdp->fullBus(c+16961,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr),4);
        vcdp->fullBus(c+16969,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size),5);
        vcdp->fullBit(c+16977,((0x10U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->fullBus(c+16985,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__size_r),3);
        vcdp->fullArray(c+16993,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[0]),76);
        vcdp->fullArray(c+16996,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[1]),76);
        vcdp->fullArray(c+16999,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[2]),76);
        vcdp->fullArray(c+17002,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[3]),76);
        vcdp->fullArray(c+17089,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),76);
        vcdp->fullArray(c+17113,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),76);
        vcdp->fullBus(c+17137,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->fullBus(c+17145,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->fullBus(c+17153,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->fullBit(c+17161,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullBus(c+17169,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__size_r),3);
        vcdp->fullArray(c+17177,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[0]),200);
        vcdp->fullArray(c+17184,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[1]),200);
        vcdp->fullArray(c+17191,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[2]),200);
        vcdp->fullArray(c+17198,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[3]),200);
        vcdp->fullArray(c+17401,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),200);
        vcdp->fullArray(c+17457,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),200);
        vcdp->fullBus(c+17513,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->fullBus(c+17521,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->fullBus(c+17529,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->fullBit(c+17537,(vlTOPp->VX_cache__DOT__genblk5__BRA__1__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullBit(c+17545,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+17553,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullBit(c+17561,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+17569,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullBit(c+17577,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullQuad(c+17585,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag),42);
        vcdp->fullBit(c+17601,((0x10U == (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->fullBit(c+17609,((0xbU < (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->fullBus(c+17617,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                       [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                       << 0xdU) | (
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                   >> 0x13U)))),2);
        vcdp->fullBus(c+17625,(((0x19fU >= (0x1ffU 
                                            & ((IData)(0x1aU) 
                                               * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                 ? (0x3ffffffU & ((
                                                   (0U 
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
        vcdp->fullBus(c+17633,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                       [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                       << 0x1eU) | 
                                      (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                       [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                       >> 2U)))),2);
        vcdp->fullBus(c+17641,(((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                 [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                 << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                             [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                             >> 0x15U))),32);
        vcdp->fullQuad(c+17649,((VL_ULL(0x3ffffffffff) 
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
        vcdp->fullBus(c+17665,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                         << 0x1cU) 
                                        | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                           [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                           >> 4U)))),4);
        vcdp->fullBit(c+17673,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                      >> 1U))));
        vcdp->fullBit(c+17681,((1U & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                [vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])));
        vcdp->fullBus(c+17689,((3U & vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])),2);
        vcdp->fullQuad(c+17697,((VL_ULL(0x3ffffffffff) 
                                 & (((QData)((IData)(
                                                     vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U])) 
                                     << 0x39U) | (((QData)((IData)(
                                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                                   << 0x19U) 
                                                  | ((QData)((IData)(
                                                                     vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])) 
                                                     >> 7U))))),42);
        vcdp->fullBit(c+17713,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                      >> 6U))));
        vcdp->fullBus(c+17721,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                         << 0x1eU) 
                                        | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                           >> 2U)))),4);
        vcdp->fullBit(c+17729,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x14U))));
        vcdp->fullBit(c+17737,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x13U))));
        vcdp->fullBus(c+17745,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                       << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                 >> 0x17U)))),2);
        vcdp->fullBus(c+17753,(((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                 << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                           >> 0x17U))),32);
        vcdp->fullBus(c+17761,(((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                 << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                                           >> 0x17U))),32);
        __Vtemp628[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 0x17U));
        __Vtemp628[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                     >> 0x17U));
        __Vtemp628[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                                     >> 0x17U));
        __Vtemp628[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                                     >> 0x17U));
        vcdp->fullArray(c+17769,(__Vtemp628),128);
        vcdp->fullBit(c+17801,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                      >> 2U))));
        vcdp->fullBit(c+17809,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                      >> 1U))));
        vcdp->fullBus(c+17817,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                            << 0xfU) 
                                           | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                              >> 0x11U)))),16);
        vcdp->fullQuad(c+17825,((VL_ULL(0x1ffffffffffff) 
                                 & (((QData)((IData)(
                                                     vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                     << 0x20U) | (QData)((IData)(
                                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U]))))),49);
        vcdp->fullBus(c+17841,((0xfffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                             << 0x1dU) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                               >> 3U)))),20);
        vcdp->fullBit(c+17849,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x15U))));
        vcdp->fullBit(c+17857,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x17U))));
        vcdp->fullBit(c+17865,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x16U))));
        vcdp->fullBit(c+17873,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x18U))));
        vcdp->fullBit(c+17881,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU))));
        vcdp->fullBit(c+17889,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1bU))));
        vcdp->fullBit(c+17897,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x19U))));
        vcdp->fullBit(c+17905,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+17913,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullBit(c+17921,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+17929,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullBus(c+17937,(((0x3ffffc0U & (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                               << 3U)) 
                                | (0x3fU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                             << 7U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                               >> 0x19U))))),26);
        vcdp->fullBus(c+17945,((0xfffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                               << 0x19U) 
                                              | (vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                                 >> 7U)))),28);
        vcdp->fullBit(c+17953,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwbq_dual_valid_sel));
        vcdp->fullBus(c+17961,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__size_r),5);
        vcdp->fullQuad(c+17969,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[0]),55);
        vcdp->fullQuad(c+17971,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[1]),55);
        vcdp->fullQuad(c+17973,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[2]),55);
        vcdp->fullQuad(c+17975,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[3]),55);
        vcdp->fullQuad(c+17977,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[4]),55);
        vcdp->fullQuad(c+17979,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[5]),55);
        vcdp->fullQuad(c+17981,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[6]),55);
        vcdp->fullQuad(c+17983,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[7]),55);
        vcdp->fullQuad(c+17985,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[8]),55);
        vcdp->fullQuad(c+17987,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[9]),55);
        vcdp->fullQuad(c+17989,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[10]),55);
        vcdp->fullQuad(c+17991,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[11]),55);
        vcdp->fullQuad(c+17993,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[12]),55);
        vcdp->fullQuad(c+17995,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[13]),55);
        vcdp->fullQuad(c+17997,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[14]),55);
        vcdp->fullQuad(c+17999,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[15]),55);
        vcdp->fullQuad(c+18225,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),55);
        vcdp->fullQuad(c+18241,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),55);
        vcdp->fullBus(c+18257,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->fullBus(c+18265,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->fullBus(c+18273,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->fullBit(c+18281,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullBus(c+18289,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__size_r),5);
        vcdp->fullArray(c+18297,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[0]),154);
        vcdp->fullArray(c+18302,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[1]),154);
        vcdp->fullArray(c+18307,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[2]),154);
        vcdp->fullArray(c+18312,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[3]),154);
        vcdp->fullArray(c+18317,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[4]),154);
        vcdp->fullArray(c+18322,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[5]),154);
        vcdp->fullArray(c+18327,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[6]),154);
        vcdp->fullArray(c+18332,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[7]),154);
        vcdp->fullArray(c+18337,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[8]),154);
        vcdp->fullArray(c+18342,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[9]),154);
        vcdp->fullArray(c+18347,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[10]),154);
        vcdp->fullArray(c+18352,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[11]),154);
        vcdp->fullArray(c+18357,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[12]),154);
        vcdp->fullArray(c+18362,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[13]),154);
        vcdp->fullArray(c+18367,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[14]),154);
        vcdp->fullArray(c+18372,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[15]),154);
        vcdp->fullArray(c+18937,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),154);
        vcdp->fullArray(c+18977,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),154);
        vcdp->fullBus(c+19017,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->fullBus(c+19025,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->fullBus(c+19033,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->fullBit(c+19041,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullBus(c+19049,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids),4);
        vcdp->fullBus(c+19057,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_rw),4);
        vcdp->fullBus(c+19065,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen),16);
        vcdp->fullArray(c+19073,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_addr),120);
        vcdp->fullArray(c+19105,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata),128);
        vcdp->fullBit(c+19137,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+19145,((1U & (~ (IData)((0U 
                                                 != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids)))))));
        vcdp->fullBus(c+19153,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__size_r),3);
        vcdp->fullArray(c+19161,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[0]),314);
        vcdp->fullArray(c+19171,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[1]),314);
        vcdp->fullArray(c+19181,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[2]),314);
        vcdp->fullArray(c+19191,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[3]),314);
        vcdp->fullArray(c+19481,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),314);
        vcdp->fullArray(c+19561,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),314);
        vcdp->fullBus(c+19641,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->fullBus(c+19649,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->fullBus(c+19657,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->fullBit(c+19665,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullArray(c+19673,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__s0_1_c0__DOT__value),243);
        vcdp->fullQuad(c+19737,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty),64);
        vcdp->fullQuad(c+19753,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid),64);
        vcdp->fullBus(c+19769,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__i),32);
        vcdp->fullBus(c+19777,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__j),32);
        vcdp->fullArray(c+19785,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__tag_data_access__DOT__s0_1_c0__DOT__value),166);
        vcdp->fullArray(c+19833,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__st_1e_2__DOT__value),316);
        vcdp->fullArray(c+19913,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[0]),85);
        vcdp->fullArray(c+19916,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[1]),85);
        vcdp->fullArray(c+19919,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[2]),85);
        vcdp->fullArray(c+19922,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[3]),85);
        vcdp->fullArray(c+19925,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[4]),85);
        vcdp->fullArray(c+19928,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[5]),85);
        vcdp->fullArray(c+19931,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[6]),85);
        vcdp->fullArray(c+19934,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[7]),85);
        vcdp->fullArray(c+19937,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[8]),85);
        vcdp->fullArray(c+19940,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[9]),85);
        vcdp->fullArray(c+19943,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[10]),85);
        vcdp->fullArray(c+19946,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[11]),85);
        vcdp->fullArray(c+19949,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[12]),85);
        vcdp->fullArray(c+19952,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[13]),85);
        vcdp->fullArray(c+19955,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[14]),85);
        vcdp->fullArray(c+19958,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[15]),85);
        vcdp->fullArray(c+20297,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table),416);
        vcdp->fullBus(c+20401,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_table),16);
        vcdp->fullBus(c+20409,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__ready_table),16);
        vcdp->fullBus(c+20417,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr),4);
        vcdp->fullBus(c+20425,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__head_ptr),4);
        vcdp->fullBus(c+20433,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr),4);
        vcdp->fullBus(c+20441,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size),5);
        vcdp->fullBit(c+20449,((0x10U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->fullBus(c+20457,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__size_r),3);
        vcdp->fullArray(c+20465,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[0]),76);
        vcdp->fullArray(c+20468,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[1]),76);
        vcdp->fullArray(c+20471,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[2]),76);
        vcdp->fullArray(c+20474,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[3]),76);
        vcdp->fullArray(c+20561,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),76);
        vcdp->fullArray(c+20585,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),76);
        vcdp->fullBus(c+20609,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->fullBus(c+20617,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->fullBus(c+20625,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->fullBit(c+20633,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullBus(c+20641,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__size_r),3);
        vcdp->fullArray(c+20649,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[0]),200);
        vcdp->fullArray(c+20656,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[1]),200);
        vcdp->fullArray(c+20663,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[2]),200);
        vcdp->fullArray(c+20670,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[3]),200);
        vcdp->fullArray(c+20873,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),200);
        vcdp->fullArray(c+20929,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),200);
        vcdp->fullBus(c+20985,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->fullBus(c+20993,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->fullBus(c+21001,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->fullBit(c+21009,(vlTOPp->VX_cache__DOT__genblk5__BRA__2__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullBit(c+21017,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+21025,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullBit(c+21033,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+21041,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullBit(c+21049,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullQuad(c+21057,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_tag),42);
        vcdp->fullBit(c+21073,((0x10U == (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->fullBit(c+21081,((0xbU < (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->fullBus(c+21089,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                       [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                       << 0xdU) | (
                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                                   [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                                   >> 0x13U)))),2);
        vcdp->fullBus(c+21097,(((0x19fU >= (0x1ffU 
                                            & ((IData)(0x1aU) 
                                               * (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr))))
                                 ? (0x3ffffffU & ((
                                                   (0U 
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
        vcdp->fullBus(c+21105,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                       [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                       << 0x1eU) | 
                                      (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                       [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                       >> 2U)))),2);
        vcdp->fullBus(c+21113,(((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                 [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][2U] 
                                 << 0xbU) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                             [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                             >> 0x15U))),32);
        vcdp->fullQuad(c+21121,((VL_ULL(0x3ffffffffff) 
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
        vcdp->fullBus(c+21137,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                         [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][1U] 
                                         << 0x1cU) 
                                        | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                           [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                           >> 4U)))),4);
        vcdp->fullBit(c+21145,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                      [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U] 
                                      >> 1U))));
        vcdp->fullBit(c+21153,((1U & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table
                                [vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr][0U])));
        vcdp->fullBus(c+21161,((3U & vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])),2);
        vcdp->fullQuad(c+21169,((VL_ULL(0x3ffffffffff) 
                                 & (((QData)((IData)(
                                                     vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U])) 
                                     << 0x39U) | (((QData)((IData)(
                                                                   vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                                   << 0x19U) 
                                                  | ((QData)((IData)(
                                                                     vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U])) 
                                                     >> 7U))))),42);
        vcdp->fullBit(c+21185,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                      >> 6U))));
        vcdp->fullBus(c+21193,((0xfU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                         << 0x1eU) 
                                        | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                           >> 2U)))),4);
        vcdp->fullBit(c+21201,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x14U))));
        vcdp->fullBit(c+21209,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x13U))));
        vcdp->fullBus(c+21217,((3U & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                       << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                                 >> 0x17U)))),2);
        vcdp->fullBus(c+21225,(((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                 << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                           >> 0x17U))),32);
        vcdp->fullBus(c+21233,(((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[7U] 
                                 << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                                           >> 0x17U))),32);
        __Vtemp636[0U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                     >> 0x17U));
        __Vtemp636[1U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                     >> 0x17U));
        __Vtemp636[2U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[4U] 
                                     >> 0x17U));
        __Vtemp636[3U] = ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[6U] 
                           << 9U) | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[5U] 
                                     >> 0x17U));
        vcdp->fullArray(c+21241,(__Vtemp636),128);
        vcdp->fullBit(c+21273,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                      >> 2U))));
        vcdp->fullBit(c+21281,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                      >> 1U))));
        vcdp->fullBus(c+21289,((0xffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                            << 0xfU) 
                                           | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                              >> 0x11U)))),16);
        vcdp->fullQuad(c+21297,((VL_ULL(0x1ffffffffffff) 
                                 & (((QData)((IData)(
                                                     vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U])) 
                                     << 0x20U) | (QData)((IData)(
                                                                 vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U]))))),49);
        vcdp->fullBus(c+21313,((0xfffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[3U] 
                                             << 0x1dU) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                               >> 3U)))),20);
        vcdp->fullBit(c+21321,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x15U))));
        vcdp->fullBit(c+21329,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x17U))));
        vcdp->fullBit(c+21337,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x16U))));
        vcdp->fullBit(c+21345,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x18U))));
        vcdp->fullBit(c+21353,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1aU))));
        vcdp->fullBit(c+21361,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x1bU))));
        vcdp->fullBit(c+21369,((1U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                      >> 0x19U))));
        vcdp->fullBit(c+21377,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+21385,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullBit(c+21393,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+21401,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__full_r));
        vcdp->fullBus(c+21409,(((0x3ffffc0U & (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[2U] 
                                               << 3U)) 
                                | (0x3fU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[9U] 
                                             << 7U) 
                                            | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[8U] 
                                               >> 0x19U))))),26);
        vcdp->fullBus(c+21417,((0xfffffffU & ((vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[1U] 
                                               << 0x19U) 
                                              | (vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value[0U] 
                                                 >> 7U)))),28);
        vcdp->fullBit(c+21425,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwbq_dual_valid_sel));
        vcdp->fullBus(c+21433,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__size_r),5);
        vcdp->fullQuad(c+21441,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[0]),55);
        vcdp->fullQuad(c+21443,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[1]),55);
        vcdp->fullQuad(c+21445,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[2]),55);
        vcdp->fullQuad(c+21447,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[3]),55);
        vcdp->fullQuad(c+21449,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[4]),55);
        vcdp->fullQuad(c+21451,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[5]),55);
        vcdp->fullQuad(c+21453,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[6]),55);
        vcdp->fullQuad(c+21455,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[7]),55);
        vcdp->fullQuad(c+21457,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[8]),55);
        vcdp->fullQuad(c+21459,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[9]),55);
        vcdp->fullQuad(c+21461,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[10]),55);
        vcdp->fullQuad(c+21463,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[11]),55);
        vcdp->fullQuad(c+21465,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[12]),55);
        vcdp->fullQuad(c+21467,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[13]),55);
        vcdp->fullQuad(c+21469,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[14]),55);
        vcdp->fullQuad(c+21471,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__data[15]),55);
        vcdp->fullQuad(c+21697,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),55);
        vcdp->fullQuad(c+21713,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),55);
        vcdp->fullBus(c+21729,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->fullBus(c+21737,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->fullBus(c+21745,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->fullBit(c+21753,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__snp_req_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullBus(c+21761,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__size_r),5);
        vcdp->fullArray(c+21769,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[0]),154);
        vcdp->fullArray(c+21774,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[1]),154);
        vcdp->fullArray(c+21779,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[2]),154);
        vcdp->fullArray(c+21784,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[3]),154);
        vcdp->fullArray(c+21789,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[4]),154);
        vcdp->fullArray(c+21794,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[5]),154);
        vcdp->fullArray(c+21799,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[6]),154);
        vcdp->fullArray(c+21804,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[7]),154);
        vcdp->fullArray(c+21809,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[8]),154);
        vcdp->fullArray(c+21814,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[9]),154);
        vcdp->fullArray(c+21819,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[10]),154);
        vcdp->fullArray(c+21824,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[11]),154);
        vcdp->fullArray(c+21829,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[12]),154);
        vcdp->fullArray(c+21834,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[13]),154);
        vcdp->fullArray(c+21839,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[14]),154);
        vcdp->fullArray(c+21844,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__data[15]),154);
        vcdp->fullArray(c+22409,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),154);
        vcdp->fullArray(c+22449,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),154);
        vcdp->fullBus(c+22489,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),4);
        vcdp->fullBus(c+22497,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),4);
        vcdp->fullBus(c+22505,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),4);
        vcdp->fullBit(c+22513,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dfp_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullBus(c+22521,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids),4);
        vcdp->fullBus(c+22529,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_rw),4);
        vcdp->fullBus(c+22537,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_byteen),16);
        vcdp->fullArray(c+22545,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_addr),120);
        vcdp->fullArray(c+22577,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_writedata),128);
        vcdp->fullBit(c+22609,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__empty_r));
        vcdp->fullBit(c+22617,((1U & (~ (IData)((0U 
                                                 != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__use_per_valids)))))));
        vcdp->fullBus(c+22625,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__size_r),3);
        vcdp->fullArray(c+22633,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[0]),314);
        vcdp->fullArray(c+22643,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[1]),314);
        vcdp->fullArray(c+22653,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[2]),314);
        vcdp->fullArray(c+22663,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__data[3]),314);
        vcdp->fullArray(c+22953,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),314);
        vcdp->fullArray(c+23033,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),314);
        vcdp->fullBus(c+23113,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->fullBus(c+23121,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->fullBus(c+23129,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->fullBit(c+23137,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__core_req_arb__DOT__reqq_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullArray(c+23145,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__s0_1_c0__DOT__value),243);
        vcdp->fullQuad(c+23209,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__dirty),64);
        vcdp->fullQuad(c+23225,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__valid),64);
        vcdp->fullBus(c+23241,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__i),32);
        vcdp->fullBus(c+23249,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__tag_data_structure__DOT__j),32);
        vcdp->fullArray(c+23257,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__tag_data_access__DOT__s0_1_c0__DOT__value),166);
        vcdp->fullArray(c+23305,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__st_1e_2__DOT__value),316);
        vcdp->fullArray(c+23385,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[0]),85);
        vcdp->fullArray(c+23388,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[1]),85);
        vcdp->fullArray(c+23391,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[2]),85);
        vcdp->fullArray(c+23394,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[3]),85);
        vcdp->fullArray(c+23397,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[4]),85);
        vcdp->fullArray(c+23400,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[5]),85);
        vcdp->fullArray(c+23403,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[6]),85);
        vcdp->fullArray(c+23406,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[7]),85);
        vcdp->fullArray(c+23409,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[8]),85);
        vcdp->fullArray(c+23412,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[9]),85);
        vcdp->fullArray(c+23415,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[10]),85);
        vcdp->fullArray(c+23418,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[11]),85);
        vcdp->fullArray(c+23421,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[12]),85);
        vcdp->fullArray(c+23424,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[13]),85);
        vcdp->fullArray(c+23427,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[14]),85);
        vcdp->fullArray(c+23430,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__metadata_table[15]),85);
        vcdp->fullArray(c+23769,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__addr_table),416);
        vcdp->fullBus(c+23873,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__valid_table),16);
        vcdp->fullBus(c+23881,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__ready_table),16);
        vcdp->fullBus(c+23889,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__schedule_ptr),4);
        vcdp->fullBus(c+23897,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__head_ptr),4);
        vcdp->fullBus(c+23905,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__tail_ptr),4);
        vcdp->fullBus(c+23913,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size),5);
        vcdp->fullBit(c+23921,((0x10U != (IData)(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cache_miss_resrv__DOT__size))));
        vcdp->fullBus(c+23929,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__size_r),3);
        vcdp->fullArray(c+23937,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[0]),76);
        vcdp->fullArray(c+23940,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[1]),76);
        vcdp->fullArray(c+23943,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[2]),76);
        vcdp->fullArray(c+23946,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__data[3]),76);
        vcdp->fullArray(c+24033,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),76);
        vcdp->fullArray(c+24057,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),76);
        vcdp->fullBus(c+24081,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->fullBus(c+24089,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->fullBus(c+24097,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->fullBit(c+24105,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__cwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullBus(c+24113,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__size_r),3);
        vcdp->fullArray(c+24121,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[0]),200);
        vcdp->fullArray(c+24128,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[1]),200);
        vcdp->fullArray(c+24135,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[2]),200);
        vcdp->fullArray(c+24142,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__data[3]),200);
        vcdp->fullArray(c+24345,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__head_r),200);
        vcdp->fullArray(c+24401,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__curr_r),200);
        vcdp->fullBus(c+24457,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__wr_ptr_r),2);
        vcdp->fullBus(c+24465,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_r),2);
        vcdp->fullBus(c+24473,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__rd_ptr_next_r),2);
        vcdp->fullBit(c+24481,(vlTOPp->VX_cache__DOT__genblk5__BRA__3__KET____DOT__bank__DOT__dwb_queue__DOT__genblk3__DOT__genblk2__DOT__bypass_r));
        vcdp->fullBit(c+24489,(vlTOPp->clk));
        vcdp->fullBit(c+24497,(vlTOPp->reset));
        vcdp->fullBus(c+24505,(vlTOPp->core_req_valid),4);
        vcdp->fullBus(c+24513,(vlTOPp->core_req_rw),4);
        vcdp->fullBus(c+24521,(vlTOPp->core_req_byteen),16);
        vcdp->fullArray(c+24529,(vlTOPp->core_req_addr),120);
        vcdp->fullArray(c+24561,(vlTOPp->core_req_data),128);
        vcdp->fullQuad(c+24593,(vlTOPp->core_req_tag),42);
        vcdp->fullBit(c+24609,(vlTOPp->core_req_ready));
        vcdp->fullBus(c+24617,(vlTOPp->core_rsp_valid),4);
        vcdp->fullArray(c+24625,(vlTOPp->core_rsp_data),128);
        vcdp->fullQuad(c+24657,(vlTOPp->core_rsp_tag),42);
        vcdp->fullBit(c+24673,(vlTOPp->core_rsp_ready));
        vcdp->fullBit(c+24681,(vlTOPp->dram_req_valid));
        vcdp->fullBit(c+24689,(vlTOPp->dram_req_rw));
        vcdp->fullBus(c+24697,(vlTOPp->dram_req_byteen),16);
        vcdp->fullBus(c+24705,(vlTOPp->dram_req_addr),28);
        vcdp->fullArray(c+24713,(vlTOPp->dram_req_data),128);
        vcdp->fullBus(c+24745,(vlTOPp->dram_req_tag),28);
        vcdp->fullBit(c+24753,(vlTOPp->dram_req_ready));
        vcdp->fullBit(c+24761,(vlTOPp->dram_rsp_valid));
        vcdp->fullArray(c+24769,(vlTOPp->dram_rsp_data),128);
        vcdp->fullBus(c+24801,(vlTOPp->dram_rsp_tag),28);
        vcdp->fullBit(c+24809,(vlTOPp->dram_rsp_ready));
        vcdp->fullBit(c+24817,(vlTOPp->snp_req_valid));
        vcdp->fullBus(c+24825,(vlTOPp->snp_req_addr),28);
        vcdp->fullBit(c+24833,(vlTOPp->snp_req_invalidate));
        vcdp->fullBus(c+24841,(vlTOPp->snp_req_tag),28);
        vcdp->fullBit(c+24849,(vlTOPp->snp_req_ready));
        vcdp->fullBit(c+24857,(vlTOPp->snp_rsp_valid));
        vcdp->fullBus(c+24865,(vlTOPp->snp_rsp_tag),28);
        vcdp->fullBit(c+24873,(vlTOPp->snp_rsp_ready));
        vcdp->fullBus(c+24881,(vlTOPp->snp_fwdout_valid),2);
        vcdp->fullQuad(c+24889,(vlTOPp->snp_fwdout_addr),56);
        vcdp->fullBus(c+24905,(vlTOPp->snp_fwdout_invalidate),2);
        vcdp->fullBus(c+24913,(vlTOPp->snp_fwdout_tag),2);
        vcdp->fullBus(c+24921,(vlTOPp->snp_fwdout_ready),2);
        vcdp->fullBus(c+24929,(vlTOPp->snp_fwdin_valid),2);
        vcdp->fullBus(c+24937,(vlTOPp->snp_fwdin_tag),2);
        vcdp->fullBus(c+24945,(vlTOPp->snp_fwdin_ready),2);
        vcdp->fullBit(c+24953,((1U & ((IData)(vlTOPp->VX_cache__DOT__per_bank_snp_req_ready) 
                                      >> (3U & vlTOPp->snp_req_addr)))));
        vcdp->fullBit(c+24961,(((IData)(vlTOPp->dram_rsp_valid) 
                                & (0U == (3U & vlTOPp->dram_rsp_tag)))));
        vcdp->fullBus(c+24969,((0x3ffffffU & (vlTOPp->dram_rsp_tag 
                                              >> 2U))),26);
        vcdp->fullBit(c+24977,(((IData)(vlTOPp->snp_req_valid) 
                                & (0U == (3U & vlTOPp->snp_req_addr)))));
        vcdp->fullBus(c+24985,((0x3ffffffU & (vlTOPp->snp_req_addr 
                                              >> 2U))),26);
        vcdp->fullBit(c+24993,(((IData)(vlTOPp->dram_rsp_valid) 
                                & (1U == (3U & vlTOPp->dram_rsp_tag)))));
        vcdp->fullBit(c+25001,(((IData)(vlTOPp->snp_req_valid) 
                                & (1U == (3U & vlTOPp->snp_req_addr)))));
        vcdp->fullBit(c+25009,(((IData)(vlTOPp->dram_rsp_valid) 
                                & (2U == (3U & vlTOPp->dram_rsp_tag)))));
        vcdp->fullBit(c+25017,(((IData)(vlTOPp->snp_req_valid) 
                                & (2U == (3U & vlTOPp->snp_req_addr)))));
        vcdp->fullBit(c+25025,(((IData)(vlTOPp->dram_rsp_valid) 
                                & (3U == (3U & vlTOPp->dram_rsp_tag)))));
        vcdp->fullBit(c+25033,(((IData)(vlTOPp->snp_req_valid) 
                                & (3U == (3U & vlTOPp->snp_req_addr)))));
        vcdp->fullBit(c+25041,(((IData)(vlTOPp->dram_req_valid) 
                                & (~ (IData)(vlTOPp->dram_req_rw)))));
        vcdp->fullBit(c+25049,((((IData)(vlTOPp->dram_req_valid) 
                                 & (~ (IData)(vlTOPp->dram_req_rw))) 
                                & (~ (IData)(vlTOPp->VX_cache__DOT__cache_dram_req_arb__DOT__prfqq__DOT__pfq_queue__DOT__size_r)))));
        vcdp->fullBus(c+25057,(0U),32);
        vcdp->fullBus(c+25065,(0x1000U),32);
        vcdp->fullBus(c+25073,(0x10U),32);
        vcdp->fullBus(c+25081,(4U),32);
        vcdp->fullBus(c+25089,(1U),32);
        vcdp->fullBus(c+25097,(0x2aU),32);
        vcdp->fullBus(c+25105,(8U),32);
        vcdp->fullBus(c+25113,(0x1cU),32);
        vcdp->fullBus(c+25121,(2U),32);
        vcdp->fullBus(c+25129,(4U),32);
        vcdp->fullBit(c+25137,(0U));
        vcdp->fullBus(c+25145,(0x74U),32);
        vcdp->fullBus(c+25153,(0U),32);
        vcdp->fullBus(c+25161,(1U),32);
        vcdp->fullBus(c+25169,(0x37U),32);
        vcdp->fullBus(c+25177,(0x9aU),32);
        vcdp->fullBus(c+25185,(0x13aU),32);
        vcdp->fullBus(c+25193,(0xf3U),32);
        vcdp->fullBus(c+25201,(0xa6U),32);
        vcdp->fullBus(c+25209,(0x13cU),32);
        vcdp->fullBus(c+25217,(0x4cU),32);
        vcdp->fullBus(c+25225,(0xc8U),32);
        vcdp->fullBus(c+25233,(2U),32);
        vcdp->fullBus(c+25241,(3U),32);
    }
}
