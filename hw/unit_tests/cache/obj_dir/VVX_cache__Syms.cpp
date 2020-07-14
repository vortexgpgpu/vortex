// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table implementation internals

#include "VVX_cache__Syms.h"
#include "VVX_cache.h"



// FUNCTIONS
VVX_cache__Syms::VVX_cache__Syms(VVX_cache* topp, const char* namep)
    // Setup locals
    : __Vm_namep(namep)
    , __Vm_activity(false)
    , __Vm_didInit(false)
    // Setup submodule names
{
    // Pointer to top level
    TOPp = topp;
    // Setup each module's pointers to their submodules
    // Setup each module's pointer back to symbol table (for public functions)
    TOPp->__Vconfigure(this, true);
    // Setup scopes
    __Vscope_VX_cache__cache_dram_req_arb__dram_fill_arb__dfqq_queue.configure(this, name(), "VX_cache.cache_dram_req_arb.dram_fill_arb.dfqq_queue", "dfqq_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__cache_dram_req_arb__dram_fill_arb__dfqq_queue__genblk3__genblk2.configure(this, name(), "VX_cache.cache_dram_req_arb.dram_fill_arb.dfqq_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__0__KET____bank__core_req_arb__reqq_queue.configure(this, name(), "VX_cache.genblk5[0].bank.core_req_arb.reqq_queue", "reqq_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__0__KET____bank__core_req_arb__reqq_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[0].bank.core_req_arb.reqq_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__0__KET____bank__cwb_queue.configure(this, name(), "VX_cache.genblk5[0].bank.cwb_queue", "cwb_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__0__KET____bank__cwb_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[0].bank.cwb_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__0__KET____bank__dfp_queue.configure(this, name(), "VX_cache.genblk5[0].bank.dfp_queue", "dfp_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__0__KET____bank__dfp_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[0].bank.dfp_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__0__KET____bank__dwb_queue.configure(this, name(), "VX_cache.genblk5[0].bank.dwb_queue", "dwb_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__0__KET____bank__dwb_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[0].bank.dwb_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__0__KET____bank__snp_req_queue.configure(this, name(), "VX_cache.genblk5[0].bank.snp_req_queue", "snp_req_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__0__KET____bank__snp_req_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[0].bank.snp_req_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__1__KET____bank__core_req_arb__reqq_queue.configure(this, name(), "VX_cache.genblk5[1].bank.core_req_arb.reqq_queue", "reqq_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__1__KET____bank__core_req_arb__reqq_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[1].bank.core_req_arb.reqq_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__1__KET____bank__cwb_queue.configure(this, name(), "VX_cache.genblk5[1].bank.cwb_queue", "cwb_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__1__KET____bank__cwb_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[1].bank.cwb_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__1__KET____bank__dfp_queue.configure(this, name(), "VX_cache.genblk5[1].bank.dfp_queue", "dfp_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__1__KET____bank__dfp_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[1].bank.dfp_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__1__KET____bank__dwb_queue.configure(this, name(), "VX_cache.genblk5[1].bank.dwb_queue", "dwb_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__1__KET____bank__dwb_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[1].bank.dwb_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__1__KET____bank__snp_req_queue.configure(this, name(), "VX_cache.genblk5[1].bank.snp_req_queue", "snp_req_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__1__KET____bank__snp_req_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[1].bank.snp_req_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__2__KET____bank__core_req_arb__reqq_queue.configure(this, name(), "VX_cache.genblk5[2].bank.core_req_arb.reqq_queue", "reqq_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__2__KET____bank__core_req_arb__reqq_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[2].bank.core_req_arb.reqq_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__2__KET____bank__cwb_queue.configure(this, name(), "VX_cache.genblk5[2].bank.cwb_queue", "cwb_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__2__KET____bank__cwb_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[2].bank.cwb_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__2__KET____bank__dfp_queue.configure(this, name(), "VX_cache.genblk5[2].bank.dfp_queue", "dfp_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__2__KET____bank__dfp_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[2].bank.dfp_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__2__KET____bank__dwb_queue.configure(this, name(), "VX_cache.genblk5[2].bank.dwb_queue", "dwb_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__2__KET____bank__dwb_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[2].bank.dwb_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__2__KET____bank__snp_req_queue.configure(this, name(), "VX_cache.genblk5[2].bank.snp_req_queue", "snp_req_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__2__KET____bank__snp_req_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[2].bank.snp_req_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__3__KET____bank__core_req_arb__reqq_queue.configure(this, name(), "VX_cache.genblk5[3].bank.core_req_arb.reqq_queue", "reqq_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__3__KET____bank__core_req_arb__reqq_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[3].bank.core_req_arb.reqq_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__3__KET____bank__cwb_queue.configure(this, name(), "VX_cache.genblk5[3].bank.cwb_queue", "cwb_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__3__KET____bank__cwb_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[3].bank.cwb_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__3__KET____bank__dfp_queue.configure(this, name(), "VX_cache.genblk5[3].bank.dfp_queue", "dfp_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__3__KET____bank__dfp_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[3].bank.dfp_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__3__KET____bank__dwb_queue.configure(this, name(), "VX_cache.genblk5[3].bank.dwb_queue", "dwb_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__3__KET____bank__dwb_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[3].bank.dwb_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__3__KET____bank__snp_req_queue.configure(this, name(), "VX_cache.genblk5[3].bank.snp_req_queue", "snp_req_queue", VerilatedScope::SCOPE_OTHER);
    __Vscope_VX_cache__genblk5__BRA__3__KET____bank__snp_req_queue__genblk3__genblk2.configure(this, name(), "VX_cache.genblk5[3].bank.snp_req_queue.genblk3.genblk2", "genblk2", VerilatedScope::SCOPE_OTHER);
}
