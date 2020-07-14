// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Symbol table internal header
//
// Internal details; most calling programs do not need this header,
// unless using verilator public meta comments.

#ifndef _VVX_CACHE__SYMS_H_
#define _VVX_CACHE__SYMS_H_  // guard

#include "verilated_heavy.h"

// INCLUDE MODULE CLASSES
#include "VVX_cache.h"

// SYMS CLASS
class VVX_cache__Syms : public VerilatedSyms {
  public:
    
    // LOCAL STATE
    const char* __Vm_namep;
    bool __Vm_activity;  ///< Used by trace routines to determine change occurred
    bool __Vm_didInit;
    
    // SUBCELL STATE
    VVX_cache*                     TOPp;
    
    // SCOPE NAMES
    VerilatedScope __Vscope_VX_cache__cache_dram_req_arb__dram_fill_arb__dfqq_queue;
    VerilatedScope __Vscope_VX_cache__cache_dram_req_arb__dram_fill_arb__dfqq_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__0__KET____bank__core_req_arb__reqq_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__0__KET____bank__core_req_arb__reqq_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__0__KET____bank__cwb_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__0__KET____bank__cwb_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__0__KET____bank__dfp_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__0__KET____bank__dfp_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__0__KET____bank__dwb_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__0__KET____bank__dwb_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__0__KET____bank__snp_req_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__0__KET____bank__snp_req_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__1__KET____bank__core_req_arb__reqq_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__1__KET____bank__core_req_arb__reqq_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__1__KET____bank__cwb_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__1__KET____bank__cwb_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__1__KET____bank__dfp_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__1__KET____bank__dfp_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__1__KET____bank__dwb_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__1__KET____bank__dwb_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__1__KET____bank__snp_req_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__1__KET____bank__snp_req_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__2__KET____bank__core_req_arb__reqq_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__2__KET____bank__core_req_arb__reqq_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__2__KET____bank__cwb_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__2__KET____bank__cwb_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__2__KET____bank__dfp_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__2__KET____bank__dfp_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__2__KET____bank__dwb_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__2__KET____bank__dwb_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__2__KET____bank__snp_req_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__2__KET____bank__snp_req_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__3__KET____bank__core_req_arb__reqq_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__3__KET____bank__core_req_arb__reqq_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__3__KET____bank__cwb_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__3__KET____bank__cwb_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__3__KET____bank__dfp_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__3__KET____bank__dfp_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__3__KET____bank__dwb_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__3__KET____bank__dwb_queue__genblk3__genblk2;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__3__KET____bank__snp_req_queue;
    VerilatedScope __Vscope_VX_cache__genblk5__BRA__3__KET____bank__snp_req_queue__genblk3__genblk2;
    
    // CREATORS
    VVX_cache__Syms(VVX_cache* topp, const char* namep);
    ~VVX_cache__Syms() {}
    
    // METHODS
    inline const char* name() { return __Vm_namep; }
    inline bool getClearActivity() { bool r=__Vm_activity; __Vm_activity=false; return r; }
    
} VL_ATTR_ALIGNED(VL_CACHE_LINE_BYTES);

#endif  // guard
