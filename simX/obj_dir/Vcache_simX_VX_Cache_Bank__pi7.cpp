// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Design implementation internals
// See Vcache_simX.h for the primary calling header

#include "Vcache_simX_VX_Cache_Bank__pi7.h"
#include "Vcache_simX__Syms.h"


//--------------------
// STATIC VARIABLES


//--------------------

VL_CTOR_IMP(Vcache_simX_VX_Cache_Bank__pi7) {
    // Reset internal values
    // Reset structure values
    _ctor_var_reset();
}

void Vcache_simX_VX_Cache_Bank__pi7::__Vconfigure(Vcache_simX__Syms* vlSymsp, bool first) {
    if (0 && first) {}  // Prevent unused
    this->__VlSymsp = vlSymsp;
}

Vcache_simX_VX_Cache_Bank__pi7::~Vcache_simX_VX_Cache_Bank__pi7() {
}

//--------------------
// Internal Methods

void Vcache_simX_VX_Cache_Bank__pi7::_settle__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__1(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            Vcache_simX_VX_Cache_Bank__pi7::_settle__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__1\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    WData/*31:0*/ __Vtemp1[4];
    WData/*31:0*/ __Vtemp2[4];
    WData/*31:0*/ __Vtemp3[4];
    WData/*31:0*/ __Vtemp4[4];
    WData/*31:0*/ __Vtemp5[4];
    WData/*31:0*/ __Vtemp6[4];
    WData/*31:0*/ __Vtemp7[4];
    // Body
    this->__PVT__way_to_update = vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__global_way_to_evict;
    this->__PVT__write_from_mem = ((2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state)) 
                                   & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__use_valid_in));
    this->__PVT__access = ((0U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state)) 
                           & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__use_valid_in));
    this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))];
    this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))];
    __Vtemp1[0U] = 0U;
    __Vtemp1[1U] = 0U;
    __Vtemp1[2U] = 0U;
    __Vtemp1[3U] = 0U;
    __Vtemp2[0U] = 0U;
    __Vtemp2[1U] = 0U;
    __Vtemp2[2U] = 0U;
    __Vtemp2[3U] = 0U;
    __Vtemp3[0U] = 0U;
    __Vtemp3[1U] = 0U;
    __Vtemp3[2U] = 0U;
    __Vtemp3[3U] = 0U;
    __Vtemp4[0U] = 0U;
    __Vtemp4[1U] = 0U;
    __Vtemp4[2U] = 0U;
    __Vtemp4[3U] = 0U;
    __Vtemp5[0U] = 0U;
    __Vtemp5[1U] = 0U;
    __Vtemp5[2U] = 0U;
    __Vtemp5[3U] = 0U;
    __Vtemp6[0U] = 0U;
    __Vtemp6[1U] = 0U;
    __Vtemp6[2U] = 0U;
    __Vtemp6[3U] = 0U;
    __Vtemp7[0U] = 0U;
    __Vtemp7[1U] = 0U;
    __Vtemp7[2U] = 0U;
    __Vtemp7[3U] = 0U;
    this->__PVT__use_write_data = ((0U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write))
                                    ? ((1U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                        ? (0xff00U 
                                           & (__Vtemp1[
                                              (3U & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank))] 
                                              << 8U))
                                        : ((2U == (3U 
                                                   & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                            ? (0xff0000U 
                                               & (__Vtemp2[
                                                  (3U 
                                                   & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank))] 
                                                  << 0x10U))
                                            : ((3U 
                                                == 
                                                (3U 
                                                 & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                                ? (0xff000000U 
                                                   & (__Vtemp3[
                                                      (3U 
                                                       & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank))] 
                                                      << 0x18U))
                                                : __Vtemp4[
                                               (3U 
                                                & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank))])))
                                    : ((1U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write))
                                        ? ((2U == (3U 
                                                   & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                            ? (0xffff0000U 
                                               & (__Vtemp5[
                                                  (3U 
                                                   & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank))] 
                                                  << 0x10U))
                                            : __Vtemp6[
                                           (3U & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank))])
                                        : __Vtemp7[
                                       (3U & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank))]));
    this->__PVT__sb_mask = ((0U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                             ? 1U : ((1U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                      ? 2U : ((2U == 
                                               (3U 
                                                & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                               ? 4U
                                               : 8U)));
    this->__PVT__data_structures__DOT__data_use_per_way[0U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))][0U];
    this->__PVT__data_structures__DOT__data_use_per_way[1U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))][1U];
    this->__PVT__data_structures__DOT__data_use_per_way[2U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))][2U];
    this->__PVT__data_structures__DOT__data_use_per_way[3U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))][3U];
    this->__PVT__data_structures__DOT__data_use_per_way[4U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))][0U];
    this->__PVT__data_structures__DOT__data_use_per_way[5U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))][1U];
    this->__PVT__data_structures__DOT__data_use_per_way[6U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))][2U];
    this->__PVT__data_structures__DOT__data_use_per_way[7U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))][3U];
    this->__PVT__data_structures__DOT__valid_use_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)) 
           | this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid
           [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                      >> 6U))]);
    this->__PVT__data_structures__DOT__valid_use_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)) 
           | (this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid
              [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                         >> 6U))] << 1U));
    this->__PVT__data_structures__DOT__tag_use_per_way 
        = ((VL_ULL(0x3ffffe00000) & this->__PVT__data_structures__DOT__tag_use_per_way) 
           | (IData)((IData)(this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag
                             [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                        >> 6U))])));
    this->__PVT__data_structures__DOT__tag_use_per_way 
        = ((VL_ULL(0x1fffff) & this->__PVT__data_structures__DOT__tag_use_per_way) 
           | ((QData)((IData)(this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag
                              [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                         >> 6U))])) 
              << 0x15U));
    this->__PVT__data_structures__DOT__dirty_use_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__dirty_use_per_way)) 
           | (IData)(this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use));
    this->__PVT__data_structures__DOT__dirty_use_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__dirty_use_per_way)) 
           | ((IData)(this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use) 
              << 1U));
    this->__PVT__data_write[0U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[0U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[1U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[1U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[2U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[2U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[3U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[3U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_structures__DOT__invalid_found = 0U;
    if ((1U & (~ ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                  >> 1U)))) {
        this->__PVT__data_structures__DOT__invalid_found = 1U;
    }
    if ((1U & (~ (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)))) {
        this->__PVT__data_structures__DOT__invalid_found = 1U;
    }
    this->__PVT__data_structures__DOT__invalid_index = 0U;
    if ((1U & (~ ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                  >> 1U)))) {
        this->__PVT__data_structures__DOT__invalid_index = 1U;
    }
    if ((1U & (~ (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)))) {
        this->__PVT__data_structures__DOT__invalid_index = 0U;
    }
    this->__PVT__data_structures__DOT__hit_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way)) 
           | (((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
               & ((0x1fffffU & (IData)(this->__PVT__data_structures__DOT__tag_use_per_way)) 
                  == (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                   >> 0xbU)))) ? 1U
               : 0U));
    this->__PVT__data_structures__DOT__hit_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way)) 
           | (((((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                 >> 1U) & ((0x1fffffU & (IData)((this->__PVT__data_structures__DOT__tag_use_per_way 
                                                 >> 0x15U))) 
                           == (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                            >> 0xbU))))
                ? 1U : 0U) << 1U));
    this->__PVT__data_structures__DOT__data_write_per_way[0U] 
        = this->__PVT__data_write[0U];
    this->__PVT__data_structures__DOT__data_write_per_way[1U] 
        = this->__PVT__data_write[1U];
    this->__PVT__data_structures__DOT__data_write_per_way[2U] 
        = this->__PVT__data_write[2U];
    this->__PVT__data_structures__DOT__data_write_per_way[3U] 
        = this->__PVT__data_write[3U];
    this->__PVT__data_structures__DOT__data_write_per_way[4U] 
        = this->__PVT__data_write[0U];
    this->__PVT__data_structures__DOT__data_write_per_way[5U] 
        = this->__PVT__data_write[1U];
    this->__PVT__data_structures__DOT__data_write_per_way[6U] 
        = this->__PVT__data_write[2U];
    this->__PVT__data_structures__DOT__data_write_per_way[7U] 
        = this->__PVT__data_write[3U];
    this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 0U;
    if ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 1U;
    }
    if ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 1U;
    }
    this->__PVT__data_structures__DOT__way_index = 0U;
    if ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__way_index = 1U;
    }
    if ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__way_index = 0U;
    }
    this->__PVT__data_structures__DOT__way_use_Qual 
        = ((0U != (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state))
            ? (IData)(this->__PVT__way_to_update) : (IData)(this->__PVT__data_structures__DOT__way_index));
    this->__PVT__data_structures__DOT__write_from_mem_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)) 
           | ((IData)(this->__PVT__write_from_mem) 
              & (~ (IData)(this->__PVT__data_structures__DOT__way_use_Qual))));
    this->__PVT__data_structures__DOT__write_from_mem_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)) 
           | (((IData)(this->__PVT__write_from_mem) 
               & (IData)(this->__PVT__data_structures__DOT__way_use_Qual)) 
              << 1U));
    this->__Vcellout__data_structures__data_use[0U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(1U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                     << 2U))] >> (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                           << 7U))));
    this->__Vcellout__data_structures__data_use[1U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(2U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(1U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__Vcellout__data_structures__data_use[2U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(3U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(2U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__Vcellout__data_structures__data_use[3U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(4U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(3U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__PVT__valid_use = (1U & ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                                    >> (IData)(this->__PVT__data_structures__DOT__way_use_Qual)));
    this->__PVT__tag_use = ((0x29U >= (0x3fU & ((IData)(0x15U) 
                                                * (IData)(this->__PVT__data_structures__DOT__way_use_Qual))))
                             ? (0x1fffffU & (IData)(
                                                    (this->__PVT__data_structures__DOT__tag_use_per_way 
                                                     >> 
                                                     (0x3fU 
                                                      & ((IData)(0x15U) 
                                                         * (IData)(this->__PVT__data_structures__DOT__way_use_Qual))))))
                             : 0U);
    this->__PVT__data_unQual = (((0U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr)) 
                                 | (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_read)))
                                 ? this->__Vcellout__data_structures__data_use[
                                (3U & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                       >> 4U))] : (
                                                   (1U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                                    ? 
                                                   (this->__Vcellout__data_structures__data_use[
                                                    (3U 
                                                     & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                                        >> 4U))] 
                                                    >> 8U)
                                                    : 
                                                   ((2U 
                                                     == 
                                                     (3U 
                                                      & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                                     ? 
                                                    (this->__Vcellout__data_structures__data_use[
                                                     (3U 
                                                      & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                                         >> 4U))] 
                                                     >> 0x10U)
                                                     : 
                                                    (this->__Vcellout__data_structures__data_use[
                                                     (3U 
                                                      & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                                         >> 4U))] 
                                                     >> 0x18U))));
    this->__PVT__miss = (((this->__PVT__tag_use != 
                           (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                         >> 0xbU))) 
                          & (IData)(this->__PVT__valid_use)) 
                         & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__use_valid_in));
    this->__PVT__genblk1__BRA__0__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (0U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__1__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (1U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__2__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (2U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__3__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (3U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__we = ((0xfff0U & (IData)(this->__PVT__we)) 
                       | ((IData)(this->__PVT__write_from_mem)
                           ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                      & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                      ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                                 & (0U 
                                                    == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                 ? (IData)(this->__PVT__sb_mask)
                                                 : 
                                                (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                                  & (1U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? 
                                                 ((0U 
                                                   == 
                                                   (3U 
                                                    & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                                   ? 3U
                                                   : 0xcU)
                                                  : 0U)))));
    this->__PVT__we = ((0xff0fU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 4U));
    this->__PVT__we = ((0xf0ffU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 8U));
    this->__PVT__we = ((0xfffU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 0xcU));
    this->__PVT__data_structures__DOT__we_per_way = 
        ((0xffff0000U & this->__PVT__data_structures__DOT__we_per_way) 
         | ((IData)(this->__PVT__data_structures__DOT__way_use_Qual)
             ? 0U : (0xffffU & (IData)(this->__PVT__we))));
    this->__PVT__data_structures__DOT__we_per_way = 
        ((0xffffU & this->__PVT__data_structures__DOT__we_per_way) 
         | (((IData)(this->__PVT__data_structures__DOT__way_use_Qual)
              ? (0xffffU & (IData)(this->__PVT__we))
              : 0U) << 0x10U));
}

VL_INLINE_OPT void Vcache_simX_VX_Cache_Bank__pi7::_sequent__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__5(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            Vcache_simX_VX_Cache_Bank__pi7::_sequent__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__5\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    this->__PVT__way_to_update = vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__global_way_to_evict;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 = 0U;
    if (vlTOPp->reset) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__ini_ind = 0x20U;
    }
    if ((1U & (~ (IData)(vlTOPp->reset)))) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__f = 4U;
    }
    if (vlTOPp->reset) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__ini_ind = 0x20U;
    }
    if ((1U & (~ (IData)(vlTOPp->reset)))) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__f = 4U;
    }
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 = 0U;
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v0 = 1U;
    } else {
        if ((1U & (((~ (IData)(this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use)) 
                    & (0U != (0xffffU & (this->__PVT__data_structures__DOT__we_per_way 
                                         >> 0x10U)))) 
                   | ((IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way) 
                      >> 1U)))) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 
                = ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))
                    ? 0U : (1U & (0U != (0xffffU & 
                                         (this->__PVT__data_structures__DOT__we_per_way 
                                          >> 0x10U)))));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v0 = 1U;
    } else {
        if ((1U & (((~ (IData)(this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use)) 
                    & (0U != (0xffffU & this->__PVT__data_structures__DOT__we_per_way))) 
                   | (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)))) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 
                = ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))
                    ? 0U : (1U & (0U != (0xffffU & this->__PVT__data_structures__DOT__we_per_way))));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v0 = 1U;
    } else {
        if ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))) {
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v0 = 1U;
    } else {
        if ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))) {
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v0 = 1U;
    } else {
        if ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 
                = (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                >> 0xbU));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v0 = 1U;
    } else {
        if ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 
                = (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                >> 0xbU));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v0 = 1U;
    } else {
        if ((0x10000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[4U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 = 0U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x20000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 = 8U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x40000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 = 0x10U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x80000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 = 0x18U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x100000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[5U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 = 0x20U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x200000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 = 0x28U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x400000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 = 0x30U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x800000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 = 0x38U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x1000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[6U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 = 0x40U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x2000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 = 0x48U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x4000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 = 0x50U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x8000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 = 0x58U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x10000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[7U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 = 0x60U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x20000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 
                = (0xffU & (this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                            >> 8U));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 = 0x68U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x40000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 
                = (0xffU & (this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                            >> 0x10U));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 = 0x70U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x80000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 
                = (0xffU & (this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                            >> 0x18U));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 = 0x78U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v0 = 1U;
    } else {
        if ((1U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[0U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 = 0U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((2U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[0U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 = 8U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((4U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[0U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 = 0x10U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((8U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[0U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 = 0x18U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x10U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[1U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 = 0x20U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x20U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 = 0x28U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x40U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 = 0x30U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x80U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 = 0x38U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x100U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[2U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 = 0x40U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x200U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 = 0x48U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x400U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 = 0x50U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x800U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 = 0x58U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x1000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[3U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 = 0x60U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x2000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 = 0x68U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x4000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 = 0x70U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x8000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 = 0x78U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32] 
            = this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32] 
            = this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32] = 1U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32] = 1U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32] 
            = this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32] 
            = this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[1U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[1U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[1U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[1U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[2U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[2U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[2U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[2U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[3U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[3U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[3U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[3U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[4U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[4U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[4U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[4U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[5U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[5U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[5U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[5U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[6U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[6U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[6U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[6U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[7U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[7U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[7U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[7U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[8U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[8U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[8U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[8U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[9U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[9U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[9U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[9U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xaU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xaU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xaU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xaU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xbU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xbU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xbU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xbU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xcU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xcU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xcU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xcU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xdU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xdU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xdU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xdU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xeU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xeU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xeU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xeU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xfU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xfU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xfU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xfU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x10U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x10U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x10U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x10U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x11U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x11U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x11U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x11U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x12U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x12U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x12U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x12U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x13U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x13U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x13U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x13U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x14U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x14U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x14U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x14U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x15U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x15U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x15U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x15U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x16U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x16U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x16U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x16U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x17U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x17U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x17U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x17U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x18U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x18U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x18U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x18U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x19U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x19U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x19U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x19U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1aU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1aU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1aU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1aU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1bU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1bU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1bU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1bU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1cU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1cU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1cU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1cU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1dU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1dU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1dU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1dU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1eU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1eU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1eU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1eU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1fU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1fU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1fU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1fU][3U] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[1U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[1U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[1U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[1U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[2U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[2U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[2U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[2U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[3U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[3U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[3U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[3U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[4U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[4U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[4U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[4U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[5U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[5U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[5U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[5U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[6U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[6U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[6U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[6U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[7U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[7U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[7U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[7U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[8U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[8U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[8U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[8U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[9U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[9U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[9U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[9U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xaU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xaU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xaU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xaU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xbU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xbU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xbU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xbU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xcU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xcU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xcU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xcU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xdU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xdU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xdU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xdU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xeU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xeU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xeU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xeU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xfU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xfU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xfU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xfU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x10U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x10U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x10U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x10U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x11U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x11U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x11U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x11U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x12U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x12U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x12U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x12U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x13U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x13U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x13U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x13U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x14U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x14U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x14U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x14U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x15U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x15U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x15U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x15U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x16U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x16U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x16U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x16U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x17U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x17U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x17U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x17U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x18U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x18U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x18U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x18U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x19U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x19U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x19U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x19U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1aU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1aU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1aU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1aU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1bU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1bU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1bU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1bU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1cU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1cU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1cU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1cU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1dU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1dU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1dU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1dU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1eU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1eU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1eU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1eU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1fU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1fU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1fU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1fU][3U] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47);
    }
}

VL_INLINE_OPT void Vcache_simX_VX_Cache_Bank__pi7::_combo__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__9(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            Vcache_simX_VX_Cache_Bank__pi7::_combo__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__9\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    WData/*31:0*/ __Vtemp75[4];
    WData/*31:0*/ __Vtemp76[4];
    WData/*31:0*/ __Vtemp77[4];
    WData/*31:0*/ __Vtemp78[4];
    WData/*31:0*/ __Vtemp79[4];
    WData/*31:0*/ __Vtemp80[4];
    WData/*31:0*/ __Vtemp81[4];
    // Body
    this->__PVT__write_from_mem = ((2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state)) 
                                   & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__use_valid_in));
    this->__PVT__access = ((0U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state)) 
                           & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__use_valid_in));
    this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))];
    this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))];
    __Vtemp75[0U] = 0U;
    __Vtemp75[1U] = 0U;
    __Vtemp75[2U] = 0U;
    __Vtemp75[3U] = 0U;
    __Vtemp76[0U] = 0U;
    __Vtemp76[1U] = 0U;
    __Vtemp76[2U] = 0U;
    __Vtemp76[3U] = 0U;
    __Vtemp77[0U] = 0U;
    __Vtemp77[1U] = 0U;
    __Vtemp77[2U] = 0U;
    __Vtemp77[3U] = 0U;
    __Vtemp78[0U] = 0U;
    __Vtemp78[1U] = 0U;
    __Vtemp78[2U] = 0U;
    __Vtemp78[3U] = 0U;
    __Vtemp79[0U] = 0U;
    __Vtemp79[1U] = 0U;
    __Vtemp79[2U] = 0U;
    __Vtemp79[3U] = 0U;
    __Vtemp80[0U] = 0U;
    __Vtemp80[1U] = 0U;
    __Vtemp80[2U] = 0U;
    __Vtemp80[3U] = 0U;
    __Vtemp81[0U] = 0U;
    __Vtemp81[1U] = 0U;
    __Vtemp81[2U] = 0U;
    __Vtemp81[3U] = 0U;
    this->__PVT__use_write_data = ((0U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write))
                                    ? ((1U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                        ? (0xff00U 
                                           & (__Vtemp75[
                                              (3U & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank))] 
                                              << 8U))
                                        : ((2U == (3U 
                                                   & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                            ? (0xff0000U 
                                               & (__Vtemp76[
                                                  (3U 
                                                   & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank))] 
                                                  << 0x10U))
                                            : ((3U 
                                                == 
                                                (3U 
                                                 & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                                ? (0xff000000U 
                                                   & (__Vtemp77[
                                                      (3U 
                                                       & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank))] 
                                                      << 0x18U))
                                                : __Vtemp78[
                                               (3U 
                                                & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank))])))
                                    : ((1U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write))
                                        ? ((2U == (3U 
                                                   & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                            ? (0xffff0000U 
                                               & (__Vtemp79[
                                                  (3U 
                                                   & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank))] 
                                                  << 0x10U))
                                            : __Vtemp80[
                                           (3U & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank))])
                                        : __Vtemp81[
                                       (3U & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank))]));
    this->__PVT__sb_mask = ((0U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                             ? 1U : ((1U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                      ? 2U : ((2U == 
                                               (3U 
                                                & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                               ? 4U
                                               : 8U)));
    this->__PVT__data_structures__DOT__data_use_per_way[0U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))][0U];
    this->__PVT__data_structures__DOT__data_use_per_way[1U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))][1U];
    this->__PVT__data_structures__DOT__data_use_per_way[2U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))][2U];
    this->__PVT__data_structures__DOT__data_use_per_way[3U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))][3U];
    this->__PVT__data_structures__DOT__data_use_per_way[4U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))][0U];
    this->__PVT__data_structures__DOT__data_use_per_way[5U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))][1U];
    this->__PVT__data_structures__DOT__data_use_per_way[6U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))][2U];
    this->__PVT__data_structures__DOT__data_use_per_way[7U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                   >> 6U))][3U];
    this->__PVT__data_structures__DOT__valid_use_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)) 
           | this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid
           [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                      >> 6U))]);
    this->__PVT__data_structures__DOT__valid_use_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)) 
           | (this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid
              [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                         >> 6U))] << 1U));
    this->__PVT__data_structures__DOT__tag_use_per_way 
        = ((VL_ULL(0x3ffffe00000) & this->__PVT__data_structures__DOT__tag_use_per_way) 
           | (IData)((IData)(this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag
                             [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                        >> 6U))])));
    this->__PVT__data_structures__DOT__tag_use_per_way 
        = ((VL_ULL(0x1fffff) & this->__PVT__data_structures__DOT__tag_use_per_way) 
           | ((QData)((IData)(this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag
                              [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                         >> 6U))])) 
              << 0x15U));
    this->__PVT__data_structures__DOT__dirty_use_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__dirty_use_per_way)) 
           | (IData)(this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use));
    this->__PVT__data_structures__DOT__dirty_use_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__dirty_use_per_way)) 
           | ((IData)(this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use) 
              << 1U));
    this->__PVT__data_write[0U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[0U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[1U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[1U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[2U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[2U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[3U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[3U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_structures__DOT__invalid_found = 0U;
    if ((1U & (~ ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                  >> 1U)))) {
        this->__PVT__data_structures__DOT__invalid_found = 1U;
    }
    if ((1U & (~ (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)))) {
        this->__PVT__data_structures__DOT__invalid_found = 1U;
    }
    this->__PVT__data_structures__DOT__invalid_index = 0U;
    if ((1U & (~ ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                  >> 1U)))) {
        this->__PVT__data_structures__DOT__invalid_index = 1U;
    }
    if ((1U & (~ (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)))) {
        this->__PVT__data_structures__DOT__invalid_index = 0U;
    }
    this->__PVT__data_structures__DOT__hit_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way)) 
           | (((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
               & ((0x1fffffU & (IData)(this->__PVT__data_structures__DOT__tag_use_per_way)) 
                  == (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                   >> 0xbU)))) ? 1U
               : 0U));
    this->__PVT__data_structures__DOT__hit_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way)) 
           | (((((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                 >> 1U) & ((0x1fffffU & (IData)((this->__PVT__data_structures__DOT__tag_use_per_way 
                                                 >> 0x15U))) 
                           == (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                            >> 0xbU))))
                ? 1U : 0U) << 1U));
    this->__PVT__data_structures__DOT__data_write_per_way[0U] 
        = this->__PVT__data_write[0U];
    this->__PVT__data_structures__DOT__data_write_per_way[1U] 
        = this->__PVT__data_write[1U];
    this->__PVT__data_structures__DOT__data_write_per_way[2U] 
        = this->__PVT__data_write[2U];
    this->__PVT__data_structures__DOT__data_write_per_way[3U] 
        = this->__PVT__data_write[3U];
    this->__PVT__data_structures__DOT__data_write_per_way[4U] 
        = this->__PVT__data_write[0U];
    this->__PVT__data_structures__DOT__data_write_per_way[5U] 
        = this->__PVT__data_write[1U];
    this->__PVT__data_structures__DOT__data_write_per_way[6U] 
        = this->__PVT__data_write[2U];
    this->__PVT__data_structures__DOT__data_write_per_way[7U] 
        = this->__PVT__data_write[3U];
    this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 0U;
    if ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 1U;
    }
    if ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 1U;
    }
    this->__PVT__data_structures__DOT__way_index = 0U;
    if ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__way_index = 1U;
    }
    if ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__way_index = 0U;
    }
    this->__PVT__data_structures__DOT__way_use_Qual 
        = ((0U != (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state))
            ? (IData)(this->__PVT__way_to_update) : (IData)(this->__PVT__data_structures__DOT__way_index));
    this->__PVT__data_structures__DOT__write_from_mem_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)) 
           | ((IData)(this->__PVT__write_from_mem) 
              & (~ (IData)(this->__PVT__data_structures__DOT__way_use_Qual))));
    this->__PVT__data_structures__DOT__write_from_mem_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)) 
           | (((IData)(this->__PVT__write_from_mem) 
               & (IData)(this->__PVT__data_structures__DOT__way_use_Qual)) 
              << 1U));
    this->__Vcellout__data_structures__data_use[0U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(1U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                     << 2U))] >> (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                           << 7U))));
    this->__Vcellout__data_structures__data_use[1U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(2U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(1U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__Vcellout__data_structures__data_use[2U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(3U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(2U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__Vcellout__data_structures__data_use[3U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(4U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(3U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__PVT__valid_use = (1U & ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                                    >> (IData)(this->__PVT__data_structures__DOT__way_use_Qual)));
    this->__PVT__tag_use = ((0x29U >= (0x3fU & ((IData)(0x15U) 
                                                * (IData)(this->__PVT__data_structures__DOT__way_use_Qual))))
                             ? (0x1fffffU & (IData)(
                                                    (this->__PVT__data_structures__DOT__tag_use_per_way 
                                                     >> 
                                                     (0x3fU 
                                                      & ((IData)(0x15U) 
                                                         * (IData)(this->__PVT__data_structures__DOT__way_use_Qual))))))
                             : 0U);
    this->__PVT__data_unQual = (((0U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr)) 
                                 | (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_read)))
                                 ? this->__Vcellout__data_structures__data_use[
                                (3U & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                       >> 4U))] : (
                                                   (1U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                                    ? 
                                                   (this->__Vcellout__data_structures__data_use[
                                                    (3U 
                                                     & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                                        >> 4U))] 
                                                    >> 8U)
                                                    : 
                                                   ((2U 
                                                     == 
                                                     (3U 
                                                      & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                                     ? 
                                                    (this->__Vcellout__data_structures__data_use[
                                                     (3U 
                                                      & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                                         >> 4U))] 
                                                     >> 0x10U)
                                                     : 
                                                    (this->__Vcellout__data_structures__data_use[
                                                     (3U 
                                                      & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                                         >> 4U))] 
                                                     >> 0x18U))));
    this->__PVT__miss = (((this->__PVT__tag_use != 
                           (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                         >> 0xbU))) 
                          & (IData)(this->__PVT__valid_use)) 
                         & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__use_valid_in));
    this->__PVT__genblk1__BRA__0__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (0U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__1__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (1U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__2__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (2U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__3__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (3U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__we = ((0xfff0U & (IData)(this->__PVT__we)) 
                       | ((IData)(this->__PVT__write_from_mem)
                           ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                      & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                      ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                                 & (0U 
                                                    == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                 ? (IData)(this->__PVT__sb_mask)
                                                 : 
                                                (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                                  & (1U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? 
                                                 ((0U 
                                                   == 
                                                   (3U 
                                                    & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                                   ? 3U
                                                   : 0xcU)
                                                  : 0U)))));
    this->__PVT__we = ((0xff0fU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 4U));
    this->__PVT__we = ((0xf0ffU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 8U));
    this->__PVT__we = ((0xfffU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 0xcU));
    this->__PVT__data_structures__DOT__we_per_way = 
        ((0xffff0000U & this->__PVT__data_structures__DOT__we_per_way) 
         | ((IData)(this->__PVT__data_structures__DOT__way_use_Qual)
             ? 0U : (0xffffU & (IData)(this->__PVT__we))));
    this->__PVT__data_structures__DOT__we_per_way = 
        ((0xffffU & this->__PVT__data_structures__DOT__we_per_way) 
         | (((IData)(this->__PVT__data_structures__DOT__way_use_Qual)
              ? (0xffffU & (IData)(this->__PVT__we))
              : 0U) << 0x10U));
}

void Vcache_simX_VX_Cache_Bank__pi7::_settle__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure__2(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            Vcache_simX_VX_Cache_Bank__pi7::_settle__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure__2\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    WData/*31:0*/ __Vtemp85[4];
    WData/*31:0*/ __Vtemp86[4];
    WData/*31:0*/ __Vtemp87[4];
    WData/*31:0*/ __Vtemp88[4];
    WData/*31:0*/ __Vtemp89[4];
    WData/*31:0*/ __Vtemp90[4];
    WData/*31:0*/ __Vtemp91[4];
    // Body
    this->__PVT__way_to_update = vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__global_way_to_evict;
    this->__PVT__write_from_mem = ((2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state)) 
                                   & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__use_valid_in));
    this->__PVT__access = ((0U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state)) 
                           & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__use_valid_in));
    this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))];
    this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))];
    __Vtemp85[0U] = 0U;
    __Vtemp85[1U] = 0U;
    __Vtemp85[2U] = 0U;
    __Vtemp85[3U] = 0U;
    __Vtemp86[0U] = 0U;
    __Vtemp86[1U] = 0U;
    __Vtemp86[2U] = 0U;
    __Vtemp86[3U] = 0U;
    __Vtemp87[0U] = 0U;
    __Vtemp87[1U] = 0U;
    __Vtemp87[2U] = 0U;
    __Vtemp87[3U] = 0U;
    __Vtemp88[0U] = 0U;
    __Vtemp88[1U] = 0U;
    __Vtemp88[2U] = 0U;
    __Vtemp88[3U] = 0U;
    __Vtemp89[0U] = 0U;
    __Vtemp89[1U] = 0U;
    __Vtemp89[2U] = 0U;
    __Vtemp89[3U] = 0U;
    __Vtemp90[0U] = 0U;
    __Vtemp90[1U] = 0U;
    __Vtemp90[2U] = 0U;
    __Vtemp90[3U] = 0U;
    __Vtemp91[0U] = 0U;
    __Vtemp91[1U] = 0U;
    __Vtemp91[2U] = 0U;
    __Vtemp91[3U] = 0U;
    this->__PVT__use_write_data = ((0U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write))
                                    ? ((1U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                        ? (0xff00U 
                                           & (__Vtemp85[
                                              (3U & 
                                               ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                >> 2U))] 
                                              << 8U))
                                        : ((2U == (3U 
                                                   & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                            ? (0xff0000U 
                                               & (__Vtemp86[
                                                  (3U 
                                                   & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                      >> 2U))] 
                                                  << 0x10U))
                                            : ((3U 
                                                == 
                                                (3U 
                                                 & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                                ? (0xff000000U 
                                                   & (__Vtemp87[
                                                      (3U 
                                                       & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                          >> 2U))] 
                                                      << 0x18U))
                                                : __Vtemp88[
                                               (3U 
                                                & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                   >> 2U))])))
                                    : ((1U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write))
                                        ? ((2U == (3U 
                                                   & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                            ? (0xffff0000U 
                                               & (__Vtemp89[
                                                  (3U 
                                                   & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                      >> 2U))] 
                                                  << 0x10U))
                                            : __Vtemp90[
                                           (3U & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                  >> 2U))])
                                        : __Vtemp91[
                                       (3U & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                              >> 2U))]));
    this->__PVT__sb_mask = ((0U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                             ? 1U : ((1U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                      ? 2U : ((2U == 
                                               (3U 
                                                & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                               ? 4U
                                               : 8U)));
    this->__PVT__data_structures__DOT__data_use_per_way[0U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))][0U];
    this->__PVT__data_structures__DOT__data_use_per_way[1U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))][1U];
    this->__PVT__data_structures__DOT__data_use_per_way[2U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))][2U];
    this->__PVT__data_structures__DOT__data_use_per_way[3U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))][3U];
    this->__PVT__data_structures__DOT__data_use_per_way[4U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))][0U];
    this->__PVT__data_structures__DOT__data_use_per_way[5U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))][1U];
    this->__PVT__data_structures__DOT__data_use_per_way[6U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))][2U];
    this->__PVT__data_structures__DOT__data_use_per_way[7U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))][3U];
    this->__PVT__data_structures__DOT__valid_use_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)) 
           | this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid
           [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                      >> 6U))]);
    this->__PVT__data_structures__DOT__valid_use_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)) 
           | (this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid
              [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                         >> 6U))] << 1U));
    this->__PVT__data_structures__DOT__tag_use_per_way 
        = ((VL_ULL(0x3ffffe00000) & this->__PVT__data_structures__DOT__tag_use_per_way) 
           | (IData)((IData)(this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag
                             [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                        >> 6U))])));
    this->__PVT__data_structures__DOT__tag_use_per_way 
        = ((VL_ULL(0x1fffff) & this->__PVT__data_structures__DOT__tag_use_per_way) 
           | ((QData)((IData)(this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag
                              [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                         >> 6U))])) 
              << 0x15U));
    this->__PVT__data_structures__DOT__dirty_use_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__dirty_use_per_way)) 
           | (IData)(this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use));
    this->__PVT__data_structures__DOT__dirty_use_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__dirty_use_per_way)) 
           | ((IData)(this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use) 
              << 1U));
    this->__PVT__data_write[0U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[4U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[1U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[5U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[2U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[6U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[3U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[7U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_structures__DOT__invalid_found = 0U;
    if ((1U & (~ ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                  >> 1U)))) {
        this->__PVT__data_structures__DOT__invalid_found = 1U;
    }
    if ((1U & (~ (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)))) {
        this->__PVT__data_structures__DOT__invalid_found = 1U;
    }
    this->__PVT__data_structures__DOT__invalid_index = 0U;
    if ((1U & (~ ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                  >> 1U)))) {
        this->__PVT__data_structures__DOT__invalid_index = 1U;
    }
    if ((1U & (~ (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)))) {
        this->__PVT__data_structures__DOT__invalid_index = 0U;
    }
    this->__PVT__data_structures__DOT__hit_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way)) 
           | (((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
               & ((0x1fffffU & (IData)(this->__PVT__data_structures__DOT__tag_use_per_way)) 
                  == (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                   >> 0xbU)))) ? 1U
               : 0U));
    this->__PVT__data_structures__DOT__hit_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way)) 
           | (((((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                 >> 1U) & ((0x1fffffU & (IData)((this->__PVT__data_structures__DOT__tag_use_per_way 
                                                 >> 0x15U))) 
                           == (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                            >> 0xbU))))
                ? 1U : 0U) << 1U));
    this->__PVT__data_structures__DOT__data_write_per_way[0U] 
        = this->__PVT__data_write[0U];
    this->__PVT__data_structures__DOT__data_write_per_way[1U] 
        = this->__PVT__data_write[1U];
    this->__PVT__data_structures__DOT__data_write_per_way[2U] 
        = this->__PVT__data_write[2U];
    this->__PVT__data_structures__DOT__data_write_per_way[3U] 
        = this->__PVT__data_write[3U];
    this->__PVT__data_structures__DOT__data_write_per_way[4U] 
        = this->__PVT__data_write[0U];
    this->__PVT__data_structures__DOT__data_write_per_way[5U] 
        = this->__PVT__data_write[1U];
    this->__PVT__data_structures__DOT__data_write_per_way[6U] 
        = this->__PVT__data_write[2U];
    this->__PVT__data_structures__DOT__data_write_per_way[7U] 
        = this->__PVT__data_write[3U];
    this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 0U;
    if ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 1U;
    }
    if ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 1U;
    }
    this->__PVT__data_structures__DOT__way_index = 0U;
    if ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__way_index = 1U;
    }
    if ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__way_index = 0U;
    }
    this->__PVT__data_structures__DOT__way_use_Qual 
        = ((0U != (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state))
            ? (IData)(this->__PVT__way_to_update) : (IData)(this->__PVT__data_structures__DOT__way_index));
    this->__PVT__data_structures__DOT__write_from_mem_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)) 
           | ((IData)(this->__PVT__write_from_mem) 
              & (~ (IData)(this->__PVT__data_structures__DOT__way_use_Qual))));
    this->__PVT__data_structures__DOT__write_from_mem_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)) 
           | (((IData)(this->__PVT__write_from_mem) 
               & (IData)(this->__PVT__data_structures__DOT__way_use_Qual)) 
              << 1U));
    this->__Vcellout__data_structures__data_use[0U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(1U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                     << 2U))] >> (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                           << 7U))));
    this->__Vcellout__data_structures__data_use[1U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(2U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(1U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__Vcellout__data_structures__data_use[2U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(3U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(2U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__Vcellout__data_structures__data_use[3U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(4U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(3U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__PVT__valid_use = (1U & ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                                    >> (IData)(this->__PVT__data_structures__DOT__way_use_Qual)));
    this->__PVT__tag_use = ((0x29U >= (0x3fU & ((IData)(0x15U) 
                                                * (IData)(this->__PVT__data_structures__DOT__way_use_Qual))))
                             ? (0x1fffffU & (IData)(
                                                    (this->__PVT__data_structures__DOT__tag_use_per_way 
                                                     >> 
                                                     (0x3fU 
                                                      & ((IData)(0x15U) 
                                                         * (IData)(this->__PVT__data_structures__DOT__way_use_Qual))))))
                             : 0U);
    this->__PVT__data_unQual = (((0U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr)) 
                                 | (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_read)))
                                 ? this->__Vcellout__data_structures__data_use[
                                (3U & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                       >> 4U))] : (
                                                   (1U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                                    ? 
                                                   (this->__Vcellout__data_structures__data_use[
                                                    (3U 
                                                     & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                                        >> 4U))] 
                                                    >> 8U)
                                                    : 
                                                   ((2U 
                                                     == 
                                                     (3U 
                                                      & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                                     ? 
                                                    (this->__Vcellout__data_structures__data_use[
                                                     (3U 
                                                      & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                                         >> 4U))] 
                                                     >> 0x10U)
                                                     : 
                                                    (this->__Vcellout__data_structures__data_use[
                                                     (3U 
                                                      & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                                         >> 4U))] 
                                                     >> 0x18U))));
    this->__PVT__miss = (((this->__PVT__tag_use != 
                           (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                         >> 0xbU))) 
                          & (IData)(this->__PVT__valid_use)) 
                         & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__use_valid_in));
    this->__PVT__genblk1__BRA__0__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (0U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__1__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (1U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__2__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (2U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__3__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (3U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__we = ((0xfff0U & (IData)(this->__PVT__we)) 
                       | ((IData)(this->__PVT__write_from_mem)
                           ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                      & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                      ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                                 & (0U 
                                                    == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                 ? (IData)(this->__PVT__sb_mask)
                                                 : 
                                                (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                                  & (1U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? 
                                                 ((0U 
                                                   == 
                                                   (3U 
                                                    & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                                   ? 3U
                                                   : 0xcU)
                                                  : 0U)))));
    this->__PVT__we = ((0xff0fU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 4U));
    this->__PVT__we = ((0xf0ffU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 8U));
    this->__PVT__we = ((0xfffU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 0xcU));
    this->__PVT__data_structures__DOT__we_per_way = 
        ((0xffff0000U & this->__PVT__data_structures__DOT__we_per_way) 
         | ((IData)(this->__PVT__data_structures__DOT__way_use_Qual)
             ? 0U : (0xffffU & (IData)(this->__PVT__we))));
    this->__PVT__data_structures__DOT__we_per_way = 
        ((0xffffU & this->__PVT__data_structures__DOT__we_per_way) 
         | (((IData)(this->__PVT__data_structures__DOT__way_use_Qual)
              ? (0xffffU & (IData)(this->__PVT__we))
              : 0U) << 0x10U));
}

VL_INLINE_OPT void Vcache_simX_VX_Cache_Bank__pi7::_sequent__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure__6(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            Vcache_simX_VX_Cache_Bank__pi7::_sequent__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure__6\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    this->__PVT__way_to_update = vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__global_way_to_evict;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 = 0U;
    if (vlTOPp->reset) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__ini_ind = 0x20U;
    }
    if ((1U & (~ (IData)(vlTOPp->reset)))) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__f = 4U;
    }
    if (vlTOPp->reset) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__ini_ind = 0x20U;
    }
    if ((1U & (~ (IData)(vlTOPp->reset)))) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__f = 4U;
    }
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 = 0U;
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v0 = 1U;
    } else {
        if ((1U & (((~ (IData)(this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use)) 
                    & (0U != (0xffffU & (this->__PVT__data_structures__DOT__we_per_way 
                                         >> 0x10U)))) 
                   | ((IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way) 
                      >> 1U)))) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 
                = ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))
                    ? 0U : (1U & (0U != (0xffffU & 
                                         (this->__PVT__data_structures__DOT__we_per_way 
                                          >> 0x10U)))));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v0 = 1U;
    } else {
        if ((1U & (((~ (IData)(this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use)) 
                    & (0U != (0xffffU & this->__PVT__data_structures__DOT__we_per_way))) 
                   | (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)))) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 
                = ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))
                    ? 0U : (1U & (0U != (0xffffU & this->__PVT__data_structures__DOT__we_per_way))));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v0 = 1U;
    } else {
        if ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))) {
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v0 = 1U;
    } else {
        if ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))) {
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v0 = 1U;
    } else {
        if ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 
                = (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                >> 0xbU));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v0 = 1U;
    } else {
        if ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 
                = (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                >> 0xbU));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v0 = 1U;
    } else {
        if ((0x10000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[4U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 = 0U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x20000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 = 8U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x40000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 = 0x10U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x80000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 = 0x18U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x100000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[5U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 = 0x20U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x200000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 = 0x28U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x400000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 = 0x30U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x800000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 = 0x38U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x1000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[6U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 = 0x40U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x2000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 = 0x48U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x4000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 = 0x50U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x8000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 = 0x58U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x10000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[7U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 = 0x60U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x20000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 
                = (0xffU & (this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                            >> 8U));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 = 0x68U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x40000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 
                = (0xffU & (this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                            >> 0x10U));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 = 0x70U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x80000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 
                = (0xffU & (this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                            >> 0x18U));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 = 0x78U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v0 = 1U;
    } else {
        if ((1U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[0U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 = 0U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((2U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[0U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 = 8U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((4U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[0U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 = 0x10U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((8U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[0U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 = 0x18U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x10U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[1U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 = 0x20U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x20U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 = 0x28U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x40U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 = 0x30U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x80U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 = 0x38U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x100U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[2U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 = 0x40U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x200U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 = 0x48U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x400U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 = 0x50U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x800U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 = 0x58U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x1000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[3U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 = 0x60U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x2000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 = 0x68U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x4000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 = 0x70U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x8000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 = 0x78U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32] 
            = this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32] 
            = this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32] = 1U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32] = 1U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32] 
            = this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32] 
            = this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[1U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[1U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[1U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[1U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[2U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[2U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[2U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[2U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[3U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[3U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[3U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[3U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[4U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[4U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[4U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[4U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[5U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[5U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[5U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[5U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[6U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[6U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[6U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[6U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[7U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[7U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[7U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[7U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[8U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[8U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[8U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[8U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[9U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[9U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[9U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[9U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xaU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xaU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xaU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xaU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xbU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xbU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xbU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xbU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xcU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xcU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xcU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xcU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xdU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xdU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xdU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xdU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xeU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xeU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xeU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xeU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xfU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xfU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xfU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xfU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x10U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x10U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x10U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x10U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x11U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x11U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x11U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x11U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x12U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x12U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x12U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x12U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x13U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x13U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x13U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x13U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x14U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x14U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x14U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x14U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x15U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x15U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x15U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x15U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x16U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x16U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x16U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x16U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x17U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x17U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x17U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x17U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x18U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x18U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x18U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x18U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x19U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x19U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x19U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x19U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1aU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1aU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1aU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1aU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1bU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1bU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1bU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1bU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1cU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1cU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1cU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1cU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1dU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1dU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1dU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1dU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1eU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1eU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1eU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1eU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1fU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1fU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1fU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1fU][3U] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[1U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[1U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[1U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[1U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[2U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[2U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[2U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[2U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[3U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[3U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[3U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[3U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[4U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[4U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[4U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[4U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[5U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[5U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[5U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[5U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[6U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[6U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[6U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[6U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[7U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[7U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[7U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[7U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[8U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[8U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[8U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[8U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[9U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[9U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[9U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[9U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xaU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xaU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xaU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xaU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xbU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xbU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xbU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xbU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xcU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xcU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xcU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xcU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xdU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xdU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xdU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xdU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xeU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xeU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xeU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xeU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xfU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xfU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xfU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xfU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x10U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x10U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x10U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x10U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x11U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x11U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x11U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x11U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x12U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x12U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x12U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x12U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x13U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x13U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x13U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x13U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x14U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x14U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x14U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x14U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x15U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x15U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x15U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x15U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x16U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x16U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x16U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x16U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x17U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x17U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x17U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x17U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x18U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x18U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x18U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x18U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x19U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x19U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x19U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x19U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1aU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1aU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1aU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1aU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1bU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1bU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1bU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1bU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1cU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1cU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1cU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1cU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1dU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1dU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1dU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1dU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1eU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1eU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1eU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1eU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1fU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1fU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1fU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1fU][3U] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47);
    }
}

VL_INLINE_OPT void Vcache_simX_VX_Cache_Bank__pi7::_combo__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure__10(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            Vcache_simX_VX_Cache_Bank__pi7::_combo__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure__10\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    WData/*31:0*/ __Vtemp159[4];
    WData/*31:0*/ __Vtemp160[4];
    WData/*31:0*/ __Vtemp161[4];
    WData/*31:0*/ __Vtemp162[4];
    WData/*31:0*/ __Vtemp163[4];
    WData/*31:0*/ __Vtemp164[4];
    WData/*31:0*/ __Vtemp165[4];
    // Body
    this->__PVT__write_from_mem = ((2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state)) 
                                   & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__use_valid_in));
    this->__PVT__access = ((0U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state)) 
                           & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__use_valid_in));
    this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))];
    this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))];
    __Vtemp159[0U] = 0U;
    __Vtemp159[1U] = 0U;
    __Vtemp159[2U] = 0U;
    __Vtemp159[3U] = 0U;
    __Vtemp160[0U] = 0U;
    __Vtemp160[1U] = 0U;
    __Vtemp160[2U] = 0U;
    __Vtemp160[3U] = 0U;
    __Vtemp161[0U] = 0U;
    __Vtemp161[1U] = 0U;
    __Vtemp161[2U] = 0U;
    __Vtemp161[3U] = 0U;
    __Vtemp162[0U] = 0U;
    __Vtemp162[1U] = 0U;
    __Vtemp162[2U] = 0U;
    __Vtemp162[3U] = 0U;
    __Vtemp163[0U] = 0U;
    __Vtemp163[1U] = 0U;
    __Vtemp163[2U] = 0U;
    __Vtemp163[3U] = 0U;
    __Vtemp164[0U] = 0U;
    __Vtemp164[1U] = 0U;
    __Vtemp164[2U] = 0U;
    __Vtemp164[3U] = 0U;
    __Vtemp165[0U] = 0U;
    __Vtemp165[1U] = 0U;
    __Vtemp165[2U] = 0U;
    __Vtemp165[3U] = 0U;
    this->__PVT__use_write_data = ((0U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write))
                                    ? ((1U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                        ? (0xff00U 
                                           & (__Vtemp159[
                                              (3U & 
                                               ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                >> 2U))] 
                                              << 8U))
                                        : ((2U == (3U 
                                                   & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                            ? (0xff0000U 
                                               & (__Vtemp160[
                                                  (3U 
                                                   & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                      >> 2U))] 
                                                  << 0x10U))
                                            : ((3U 
                                                == 
                                                (3U 
                                                 & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                                ? (0xff000000U 
                                                   & (__Vtemp161[
                                                      (3U 
                                                       & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                          >> 2U))] 
                                                      << 0x18U))
                                                : __Vtemp162[
                                               (3U 
                                                & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                   >> 2U))])))
                                    : ((1U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write))
                                        ? ((2U == (3U 
                                                   & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                            ? (0xffff0000U 
                                               & (__Vtemp163[
                                                  (3U 
                                                   & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                      >> 2U))] 
                                                  << 0x10U))
                                            : __Vtemp164[
                                           (3U & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                  >> 2U))])
                                        : __Vtemp165[
                                       (3U & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                              >> 2U))]));
    this->__PVT__sb_mask = ((0U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                             ? 1U : ((1U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                      ? 2U : ((2U == 
                                               (3U 
                                                & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                               ? 4U
                                               : 8U)));
    this->__PVT__data_structures__DOT__data_use_per_way[0U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))][0U];
    this->__PVT__data_structures__DOT__data_use_per_way[1U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))][1U];
    this->__PVT__data_structures__DOT__data_use_per_way[2U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))][2U];
    this->__PVT__data_structures__DOT__data_use_per_way[3U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))][3U];
    this->__PVT__data_structures__DOT__data_use_per_way[4U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))][0U];
    this->__PVT__data_structures__DOT__data_use_per_way[5U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))][1U];
    this->__PVT__data_structures__DOT__data_use_per_way[6U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))][2U];
    this->__PVT__data_structures__DOT__data_use_per_way[7U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                   >> 6U))][3U];
    this->__PVT__data_structures__DOT__valid_use_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)) 
           | this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid
           [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                      >> 6U))]);
    this->__PVT__data_structures__DOT__valid_use_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)) 
           | (this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid
              [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                         >> 6U))] << 1U));
    this->__PVT__data_structures__DOT__tag_use_per_way 
        = ((VL_ULL(0x3ffffe00000) & this->__PVT__data_structures__DOT__tag_use_per_way) 
           | (IData)((IData)(this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag
                             [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                        >> 6U))])));
    this->__PVT__data_structures__DOT__tag_use_per_way 
        = ((VL_ULL(0x1fffff) & this->__PVT__data_structures__DOT__tag_use_per_way) 
           | ((QData)((IData)(this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag
                              [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                         >> 6U))])) 
              << 0x15U));
    this->__PVT__data_structures__DOT__dirty_use_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__dirty_use_per_way)) 
           | (IData)(this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use));
    this->__PVT__data_structures__DOT__dirty_use_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__dirty_use_per_way)) 
           | ((IData)(this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use) 
              << 1U));
    this->__PVT__data_write[0U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[4U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[1U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[5U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[2U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[6U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[3U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[7U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_structures__DOT__invalid_found = 0U;
    if ((1U & (~ ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                  >> 1U)))) {
        this->__PVT__data_structures__DOT__invalid_found = 1U;
    }
    if ((1U & (~ (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)))) {
        this->__PVT__data_structures__DOT__invalid_found = 1U;
    }
    this->__PVT__data_structures__DOT__invalid_index = 0U;
    if ((1U & (~ ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                  >> 1U)))) {
        this->__PVT__data_structures__DOT__invalid_index = 1U;
    }
    if ((1U & (~ (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)))) {
        this->__PVT__data_structures__DOT__invalid_index = 0U;
    }
    this->__PVT__data_structures__DOT__hit_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way)) 
           | (((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
               & ((0x1fffffU & (IData)(this->__PVT__data_structures__DOT__tag_use_per_way)) 
                  == (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                   >> 0xbU)))) ? 1U
               : 0U));
    this->__PVT__data_structures__DOT__hit_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way)) 
           | (((((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                 >> 1U) & ((0x1fffffU & (IData)((this->__PVT__data_structures__DOT__tag_use_per_way 
                                                 >> 0x15U))) 
                           == (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                            >> 0xbU))))
                ? 1U : 0U) << 1U));
    this->__PVT__data_structures__DOT__data_write_per_way[0U] 
        = this->__PVT__data_write[0U];
    this->__PVT__data_structures__DOT__data_write_per_way[1U] 
        = this->__PVT__data_write[1U];
    this->__PVT__data_structures__DOT__data_write_per_way[2U] 
        = this->__PVT__data_write[2U];
    this->__PVT__data_structures__DOT__data_write_per_way[3U] 
        = this->__PVT__data_write[3U];
    this->__PVT__data_structures__DOT__data_write_per_way[4U] 
        = this->__PVT__data_write[0U];
    this->__PVT__data_structures__DOT__data_write_per_way[5U] 
        = this->__PVT__data_write[1U];
    this->__PVT__data_structures__DOT__data_write_per_way[6U] 
        = this->__PVT__data_write[2U];
    this->__PVT__data_structures__DOT__data_write_per_way[7U] 
        = this->__PVT__data_write[3U];
    this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 0U;
    if ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 1U;
    }
    if ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 1U;
    }
    this->__PVT__data_structures__DOT__way_index = 0U;
    if ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__way_index = 1U;
    }
    if ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__way_index = 0U;
    }
    this->__PVT__data_structures__DOT__way_use_Qual 
        = ((0U != (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state))
            ? (IData)(this->__PVT__way_to_update) : (IData)(this->__PVT__data_structures__DOT__way_index));
    this->__PVT__data_structures__DOT__write_from_mem_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)) 
           | ((IData)(this->__PVT__write_from_mem) 
              & (~ (IData)(this->__PVT__data_structures__DOT__way_use_Qual))));
    this->__PVT__data_structures__DOT__write_from_mem_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)) 
           | (((IData)(this->__PVT__write_from_mem) 
               & (IData)(this->__PVT__data_structures__DOT__way_use_Qual)) 
              << 1U));
    this->__Vcellout__data_structures__data_use[0U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(1U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                     << 2U))] >> (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                           << 7U))));
    this->__Vcellout__data_structures__data_use[1U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(2U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(1U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__Vcellout__data_structures__data_use[2U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(3U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(2U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__Vcellout__data_structures__data_use[3U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(4U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(3U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__PVT__valid_use = (1U & ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                                    >> (IData)(this->__PVT__data_structures__DOT__way_use_Qual)));
    this->__PVT__tag_use = ((0x29U >= (0x3fU & ((IData)(0x15U) 
                                                * (IData)(this->__PVT__data_structures__DOT__way_use_Qual))))
                             ? (0x1fffffU & (IData)(
                                                    (this->__PVT__data_structures__DOT__tag_use_per_way 
                                                     >> 
                                                     (0x3fU 
                                                      & ((IData)(0x15U) 
                                                         * (IData)(this->__PVT__data_structures__DOT__way_use_Qual))))))
                             : 0U);
    this->__PVT__data_unQual = (((0U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr)) 
                                 | (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_read)))
                                 ? this->__Vcellout__data_structures__data_use[
                                (3U & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                       >> 4U))] : (
                                                   (1U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                                    ? 
                                                   (this->__Vcellout__data_structures__data_use[
                                                    (3U 
                                                     & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                                        >> 4U))] 
                                                    >> 8U)
                                                    : 
                                                   ((2U 
                                                     == 
                                                     (3U 
                                                      & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                                     ? 
                                                    (this->__Vcellout__data_structures__data_use[
                                                     (3U 
                                                      & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                                         >> 4U))] 
                                                     >> 0x10U)
                                                     : 
                                                    (this->__Vcellout__data_structures__data_use[
                                                     (3U 
                                                      & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                                         >> 4U))] 
                                                     >> 0x18U))));
    this->__PVT__miss = (((this->__PVT__tag_use != 
                           (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                         >> 0xbU))) 
                          & (IData)(this->__PVT__valid_use)) 
                         & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__use_valid_in));
    this->__PVT__genblk1__BRA__0__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (0U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__1__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (1U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__2__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (2U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__3__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (3U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__we = ((0xfff0U & (IData)(this->__PVT__we)) 
                       | ((IData)(this->__PVT__write_from_mem)
                           ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                      & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                      ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                                 & (0U 
                                                    == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                 ? (IData)(this->__PVT__sb_mask)
                                                 : 
                                                (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                                  & (1U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? 
                                                 ((0U 
                                                   == 
                                                   (3U 
                                                    & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                                   ? 3U
                                                   : 0xcU)
                                                  : 0U)))));
    this->__PVT__we = ((0xff0fU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 4U));
    this->__PVT__we = ((0xf0ffU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 8U));
    this->__PVT__we = ((0xfffU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 0xcU));
    this->__PVT__data_structures__DOT__we_per_way = 
        ((0xffff0000U & this->__PVT__data_structures__DOT__we_per_way) 
         | ((IData)(this->__PVT__data_structures__DOT__way_use_Qual)
             ? 0U : (0xffffU & (IData)(this->__PVT__we))));
    this->__PVT__data_structures__DOT__we_per_way = 
        ((0xffffU & this->__PVT__data_structures__DOT__we_per_way) 
         | (((IData)(this->__PVT__data_structures__DOT__way_use_Qual)
              ? (0xffffU & (IData)(this->__PVT__we))
              : 0U) << 0x10U));
}

void Vcache_simX_VX_Cache_Bank__pi7::_settle__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure__3(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            Vcache_simX_VX_Cache_Bank__pi7::_settle__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure__3\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    WData/*31:0*/ __Vtemp169[4];
    WData/*31:0*/ __Vtemp170[4];
    WData/*31:0*/ __Vtemp171[4];
    WData/*31:0*/ __Vtemp172[4];
    WData/*31:0*/ __Vtemp173[4];
    WData/*31:0*/ __Vtemp174[4];
    WData/*31:0*/ __Vtemp175[4];
    // Body
    this->__PVT__way_to_update = vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__global_way_to_evict;
    this->__PVT__write_from_mem = ((2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state)) 
                                   & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__use_valid_in));
    this->__PVT__access = ((0U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state)) 
                           & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__use_valid_in));
    this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))];
    this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))];
    __Vtemp169[0U] = 0U;
    __Vtemp169[1U] = 0U;
    __Vtemp169[2U] = 0U;
    __Vtemp169[3U] = 0U;
    __Vtemp170[0U] = 0U;
    __Vtemp170[1U] = 0U;
    __Vtemp170[2U] = 0U;
    __Vtemp170[3U] = 0U;
    __Vtemp171[0U] = 0U;
    __Vtemp171[1U] = 0U;
    __Vtemp171[2U] = 0U;
    __Vtemp171[3U] = 0U;
    __Vtemp172[0U] = 0U;
    __Vtemp172[1U] = 0U;
    __Vtemp172[2U] = 0U;
    __Vtemp172[3U] = 0U;
    __Vtemp173[0U] = 0U;
    __Vtemp173[1U] = 0U;
    __Vtemp173[2U] = 0U;
    __Vtemp173[3U] = 0U;
    __Vtemp174[0U] = 0U;
    __Vtemp174[1U] = 0U;
    __Vtemp174[2U] = 0U;
    __Vtemp174[3U] = 0U;
    __Vtemp175[0U] = 0U;
    __Vtemp175[1U] = 0U;
    __Vtemp175[2U] = 0U;
    __Vtemp175[3U] = 0U;
    this->__PVT__use_write_data = ((0U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write))
                                    ? ((1U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                        ? (0xff00U 
                                           & (__Vtemp169[
                                              (3U & 
                                               ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                >> 4U))] 
                                              << 8U))
                                        : ((2U == (3U 
                                                   & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                            ? (0xff0000U 
                                               & (__Vtemp170[
                                                  (3U 
                                                   & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                      >> 4U))] 
                                                  << 0x10U))
                                            : ((3U 
                                                == 
                                                (3U 
                                                 & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                                ? (0xff000000U 
                                                   & (__Vtemp171[
                                                      (3U 
                                                       & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                          >> 4U))] 
                                                      << 0x18U))
                                                : __Vtemp172[
                                               (3U 
                                                & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                   >> 4U))])))
                                    : ((1U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write))
                                        ? ((2U == (3U 
                                                   & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                            ? (0xffff0000U 
                                               & (__Vtemp173[
                                                  (3U 
                                                   & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                      >> 4U))] 
                                                  << 0x10U))
                                            : __Vtemp174[
                                           (3U & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                  >> 4U))])
                                        : __Vtemp175[
                                       (3U & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                              >> 4U))]));
    this->__PVT__sb_mask = ((0U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                             ? 1U : ((1U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                      ? 2U : ((2U == 
                                               (3U 
                                                & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                               ? 4U
                                               : 8U)));
    this->__PVT__data_structures__DOT__data_use_per_way[0U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))][0U];
    this->__PVT__data_structures__DOT__data_use_per_way[1U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))][1U];
    this->__PVT__data_structures__DOT__data_use_per_way[2U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))][2U];
    this->__PVT__data_structures__DOT__data_use_per_way[3U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))][3U];
    this->__PVT__data_structures__DOT__data_use_per_way[4U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))][0U];
    this->__PVT__data_structures__DOT__data_use_per_way[5U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))][1U];
    this->__PVT__data_structures__DOT__data_use_per_way[6U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))][2U];
    this->__PVT__data_structures__DOT__data_use_per_way[7U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))][3U];
    this->__PVT__data_structures__DOT__valid_use_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)) 
           | this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid
           [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                      >> 6U))]);
    this->__PVT__data_structures__DOT__valid_use_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)) 
           | (this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid
              [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                         >> 6U))] << 1U));
    this->__PVT__data_structures__DOT__tag_use_per_way 
        = ((VL_ULL(0x3ffffe00000) & this->__PVT__data_structures__DOT__tag_use_per_way) 
           | (IData)((IData)(this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag
                             [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                        >> 6U))])));
    this->__PVT__data_structures__DOT__tag_use_per_way 
        = ((VL_ULL(0x1fffff) & this->__PVT__data_structures__DOT__tag_use_per_way) 
           | ((QData)((IData)(this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag
                              [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                         >> 6U))])) 
              << 0x15U));
    this->__PVT__data_structures__DOT__dirty_use_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__dirty_use_per_way)) 
           | (IData)(this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use));
    this->__PVT__data_structures__DOT__dirty_use_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__dirty_use_per_way)) 
           | ((IData)(this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use) 
              << 1U));
    this->__PVT__data_write[0U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[8U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[1U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[9U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[2U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[0xaU]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[3U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[0xbU]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_structures__DOT__invalid_found = 0U;
    if ((1U & (~ ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                  >> 1U)))) {
        this->__PVT__data_structures__DOT__invalid_found = 1U;
    }
    if ((1U & (~ (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)))) {
        this->__PVT__data_structures__DOT__invalid_found = 1U;
    }
    this->__PVT__data_structures__DOT__invalid_index = 0U;
    if ((1U & (~ ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                  >> 1U)))) {
        this->__PVT__data_structures__DOT__invalid_index = 1U;
    }
    if ((1U & (~ (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)))) {
        this->__PVT__data_structures__DOT__invalid_index = 0U;
    }
    this->__PVT__data_structures__DOT__hit_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way)) 
           | (((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
               & ((0x1fffffU & (IData)(this->__PVT__data_structures__DOT__tag_use_per_way)) 
                  == (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                   >> 0xbU)))) ? 1U
               : 0U));
    this->__PVT__data_structures__DOT__hit_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way)) 
           | (((((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                 >> 1U) & ((0x1fffffU & (IData)((this->__PVT__data_structures__DOT__tag_use_per_way 
                                                 >> 0x15U))) 
                           == (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                            >> 0xbU))))
                ? 1U : 0U) << 1U));
    this->__PVT__data_structures__DOT__data_write_per_way[0U] 
        = this->__PVT__data_write[0U];
    this->__PVT__data_structures__DOT__data_write_per_way[1U] 
        = this->__PVT__data_write[1U];
    this->__PVT__data_structures__DOT__data_write_per_way[2U] 
        = this->__PVT__data_write[2U];
    this->__PVT__data_structures__DOT__data_write_per_way[3U] 
        = this->__PVT__data_write[3U];
    this->__PVT__data_structures__DOT__data_write_per_way[4U] 
        = this->__PVT__data_write[0U];
    this->__PVT__data_structures__DOT__data_write_per_way[5U] 
        = this->__PVT__data_write[1U];
    this->__PVT__data_structures__DOT__data_write_per_way[6U] 
        = this->__PVT__data_write[2U];
    this->__PVT__data_structures__DOT__data_write_per_way[7U] 
        = this->__PVT__data_write[3U];
    this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 0U;
    if ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 1U;
    }
    if ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 1U;
    }
    this->__PVT__data_structures__DOT__way_index = 0U;
    if ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__way_index = 1U;
    }
    if ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__way_index = 0U;
    }
    this->__PVT__data_structures__DOT__way_use_Qual 
        = ((0U != (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state))
            ? (IData)(this->__PVT__way_to_update) : (IData)(this->__PVT__data_structures__DOT__way_index));
    this->__PVT__data_structures__DOT__write_from_mem_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)) 
           | ((IData)(this->__PVT__write_from_mem) 
              & (~ (IData)(this->__PVT__data_structures__DOT__way_use_Qual))));
    this->__PVT__data_structures__DOT__write_from_mem_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)) 
           | (((IData)(this->__PVT__write_from_mem) 
               & (IData)(this->__PVT__data_structures__DOT__way_use_Qual)) 
              << 1U));
    this->__Vcellout__data_structures__data_use[0U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(1U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                     << 2U))] >> (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                           << 7U))));
    this->__Vcellout__data_structures__data_use[1U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(2U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(1U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__Vcellout__data_structures__data_use[2U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(3U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(2U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__Vcellout__data_structures__data_use[3U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(4U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(3U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__PVT__valid_use = (1U & ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                                    >> (IData)(this->__PVT__data_structures__DOT__way_use_Qual)));
    this->__PVT__tag_use = ((0x29U >= (0x3fU & ((IData)(0x15U) 
                                                * (IData)(this->__PVT__data_structures__DOT__way_use_Qual))))
                             ? (0x1fffffU & (IData)(
                                                    (this->__PVT__data_structures__DOT__tag_use_per_way 
                                                     >> 
                                                     (0x3fU 
                                                      & ((IData)(0x15U) 
                                                         * (IData)(this->__PVT__data_structures__DOT__way_use_Qual))))))
                             : 0U);
    this->__PVT__data_unQual = (((0U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr)) 
                                 | (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_read)))
                                 ? this->__Vcellout__data_structures__data_use[
                                (3U & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                       >> 4U))] : (
                                                   (1U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                                    ? 
                                                   (this->__Vcellout__data_structures__data_use[
                                                    (3U 
                                                     & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                                        >> 4U))] 
                                                    >> 8U)
                                                    : 
                                                   ((2U 
                                                     == 
                                                     (3U 
                                                      & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                                     ? 
                                                    (this->__Vcellout__data_structures__data_use[
                                                     (3U 
                                                      & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                                         >> 4U))] 
                                                     >> 0x10U)
                                                     : 
                                                    (this->__Vcellout__data_structures__data_use[
                                                     (3U 
                                                      & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                                         >> 4U))] 
                                                     >> 0x18U))));
    this->__PVT__miss = (((this->__PVT__tag_use != 
                           (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                         >> 0xbU))) 
                          & (IData)(this->__PVT__valid_use)) 
                         & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__use_valid_in));
    this->__PVT__genblk1__BRA__0__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (0U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__1__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (1U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__2__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (2U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__3__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (3U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__we = ((0xfff0U & (IData)(this->__PVT__we)) 
                       | ((IData)(this->__PVT__write_from_mem)
                           ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                      & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                      ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                                 & (0U 
                                                    == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                 ? (IData)(this->__PVT__sb_mask)
                                                 : 
                                                (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                                  & (1U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? 
                                                 ((0U 
                                                   == 
                                                   (3U 
                                                    & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                                   ? 3U
                                                   : 0xcU)
                                                  : 0U)))));
    this->__PVT__we = ((0xff0fU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 4U));
    this->__PVT__we = ((0xf0ffU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 8U));
    this->__PVT__we = ((0xfffU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 0xcU));
    this->__PVT__data_structures__DOT__we_per_way = 
        ((0xffff0000U & this->__PVT__data_structures__DOT__we_per_way) 
         | ((IData)(this->__PVT__data_structures__DOT__way_use_Qual)
             ? 0U : (0xffffU & (IData)(this->__PVT__we))));
    this->__PVT__data_structures__DOT__we_per_way = 
        ((0xffffU & this->__PVT__data_structures__DOT__we_per_way) 
         | (((IData)(this->__PVT__data_structures__DOT__way_use_Qual)
              ? (0xffffU & (IData)(this->__PVT__we))
              : 0U) << 0x10U));
}

VL_INLINE_OPT void Vcache_simX_VX_Cache_Bank__pi7::_sequent__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure__7(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            Vcache_simX_VX_Cache_Bank__pi7::_sequent__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure__7\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    this->__PVT__way_to_update = vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__global_way_to_evict;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 = 0U;
    if (vlTOPp->reset) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__ini_ind = 0x20U;
    }
    if ((1U & (~ (IData)(vlTOPp->reset)))) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__f = 4U;
    }
    if (vlTOPp->reset) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__ini_ind = 0x20U;
    }
    if ((1U & (~ (IData)(vlTOPp->reset)))) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__f = 4U;
    }
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 = 0U;
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v0 = 1U;
    } else {
        if ((1U & (((~ (IData)(this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use)) 
                    & (0U != (0xffffU & (this->__PVT__data_structures__DOT__we_per_way 
                                         >> 0x10U)))) 
                   | ((IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way) 
                      >> 1U)))) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 
                = ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))
                    ? 0U : (1U & (0U != (0xffffU & 
                                         (this->__PVT__data_structures__DOT__we_per_way 
                                          >> 0x10U)))));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v0 = 1U;
    } else {
        if ((1U & (((~ (IData)(this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use)) 
                    & (0U != (0xffffU & this->__PVT__data_structures__DOT__we_per_way))) 
                   | (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)))) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 
                = ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))
                    ? 0U : (1U & (0U != (0xffffU & this->__PVT__data_structures__DOT__we_per_way))));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v0 = 1U;
    } else {
        if ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))) {
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v0 = 1U;
    } else {
        if ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))) {
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v0 = 1U;
    } else {
        if ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 
                = (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                >> 0xbU));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v0 = 1U;
    } else {
        if ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 
                = (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                >> 0xbU));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v0 = 1U;
    } else {
        if ((0x10000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[4U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 = 0U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x20000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 = 8U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x40000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 = 0x10U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x80000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 = 0x18U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x100000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[5U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 = 0x20U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x200000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 = 0x28U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x400000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 = 0x30U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x800000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 = 0x38U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x1000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[6U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 = 0x40U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x2000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 = 0x48U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x4000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 = 0x50U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x8000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 = 0x58U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x10000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[7U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 = 0x60U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x20000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 
                = (0xffU & (this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                            >> 8U));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 = 0x68U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x40000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 
                = (0xffU & (this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                            >> 0x10U));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 = 0x70U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x80000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 
                = (0xffU & (this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                            >> 0x18U));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 = 0x78U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v0 = 1U;
    } else {
        if ((1U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[0U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 = 0U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((2U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[0U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 = 8U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((4U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[0U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 = 0x10U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((8U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[0U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 = 0x18U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x10U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[1U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 = 0x20U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x20U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 = 0x28U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x40U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 = 0x30U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x80U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 = 0x38U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x100U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[2U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 = 0x40U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x200U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 = 0x48U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x400U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 = 0x50U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x800U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 = 0x58U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x1000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[3U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 = 0x60U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x2000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 = 0x68U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x4000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 = 0x70U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x8000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 = 0x78U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32] 
            = this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32] 
            = this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32] = 1U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32] = 1U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32] 
            = this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32] 
            = this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[1U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[1U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[1U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[1U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[2U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[2U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[2U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[2U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[3U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[3U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[3U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[3U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[4U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[4U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[4U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[4U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[5U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[5U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[5U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[5U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[6U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[6U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[6U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[6U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[7U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[7U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[7U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[7U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[8U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[8U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[8U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[8U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[9U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[9U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[9U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[9U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xaU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xaU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xaU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xaU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xbU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xbU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xbU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xbU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xcU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xcU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xcU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xcU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xdU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xdU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xdU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xdU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xeU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xeU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xeU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xeU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xfU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xfU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xfU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xfU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x10U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x10U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x10U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x10U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x11U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x11U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x11U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x11U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x12U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x12U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x12U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x12U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x13U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x13U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x13U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x13U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x14U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x14U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x14U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x14U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x15U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x15U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x15U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x15U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x16U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x16U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x16U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x16U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x17U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x17U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x17U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x17U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x18U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x18U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x18U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x18U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x19U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x19U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x19U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x19U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1aU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1aU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1aU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1aU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1bU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1bU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1bU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1bU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1cU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1cU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1cU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1cU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1dU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1dU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1dU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1dU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1eU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1eU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1eU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1eU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1fU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1fU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1fU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1fU][3U] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[1U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[1U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[1U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[1U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[2U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[2U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[2U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[2U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[3U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[3U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[3U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[3U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[4U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[4U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[4U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[4U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[5U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[5U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[5U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[5U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[6U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[6U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[6U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[6U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[7U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[7U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[7U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[7U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[8U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[8U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[8U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[8U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[9U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[9U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[9U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[9U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xaU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xaU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xaU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xaU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xbU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xbU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xbU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xbU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xcU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xcU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xcU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xcU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xdU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xdU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xdU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xdU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xeU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xeU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xeU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xeU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xfU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xfU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xfU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xfU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x10U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x10U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x10U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x10U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x11U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x11U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x11U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x11U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x12U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x12U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x12U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x12U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x13U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x13U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x13U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x13U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x14U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x14U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x14U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x14U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x15U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x15U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x15U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x15U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x16U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x16U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x16U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x16U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x17U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x17U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x17U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x17U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x18U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x18U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x18U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x18U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x19U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x19U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x19U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x19U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1aU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1aU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1aU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1aU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1bU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1bU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1bU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1bU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1cU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1cU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1cU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1cU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1dU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1dU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1dU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1dU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1eU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1eU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1eU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1eU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1fU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1fU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1fU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1fU][3U] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47);
    }
}

VL_INLINE_OPT void Vcache_simX_VX_Cache_Bank__pi7::_combo__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure__11(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            Vcache_simX_VX_Cache_Bank__pi7::_combo__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure__11\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    WData/*31:0*/ __Vtemp243[4];
    WData/*31:0*/ __Vtemp244[4];
    WData/*31:0*/ __Vtemp245[4];
    WData/*31:0*/ __Vtemp246[4];
    WData/*31:0*/ __Vtemp247[4];
    WData/*31:0*/ __Vtemp248[4];
    WData/*31:0*/ __Vtemp249[4];
    // Body
    this->__PVT__write_from_mem = ((2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state)) 
                                   & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__use_valid_in));
    this->__PVT__access = ((0U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state)) 
                           & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__use_valid_in));
    this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))];
    this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))];
    __Vtemp243[0U] = 0U;
    __Vtemp243[1U] = 0U;
    __Vtemp243[2U] = 0U;
    __Vtemp243[3U] = 0U;
    __Vtemp244[0U] = 0U;
    __Vtemp244[1U] = 0U;
    __Vtemp244[2U] = 0U;
    __Vtemp244[3U] = 0U;
    __Vtemp245[0U] = 0U;
    __Vtemp245[1U] = 0U;
    __Vtemp245[2U] = 0U;
    __Vtemp245[3U] = 0U;
    __Vtemp246[0U] = 0U;
    __Vtemp246[1U] = 0U;
    __Vtemp246[2U] = 0U;
    __Vtemp246[3U] = 0U;
    __Vtemp247[0U] = 0U;
    __Vtemp247[1U] = 0U;
    __Vtemp247[2U] = 0U;
    __Vtemp247[3U] = 0U;
    __Vtemp248[0U] = 0U;
    __Vtemp248[1U] = 0U;
    __Vtemp248[2U] = 0U;
    __Vtemp248[3U] = 0U;
    __Vtemp249[0U] = 0U;
    __Vtemp249[1U] = 0U;
    __Vtemp249[2U] = 0U;
    __Vtemp249[3U] = 0U;
    this->__PVT__use_write_data = ((0U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write))
                                    ? ((1U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                        ? (0xff00U 
                                           & (__Vtemp243[
                                              (3U & 
                                               ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                >> 4U))] 
                                              << 8U))
                                        : ((2U == (3U 
                                                   & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                            ? (0xff0000U 
                                               & (__Vtemp244[
                                                  (3U 
                                                   & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                      >> 4U))] 
                                                  << 0x10U))
                                            : ((3U 
                                                == 
                                                (3U 
                                                 & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                                ? (0xff000000U 
                                                   & (__Vtemp245[
                                                      (3U 
                                                       & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                          >> 4U))] 
                                                      << 0x18U))
                                                : __Vtemp246[
                                               (3U 
                                                & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                   >> 4U))])))
                                    : ((1U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write))
                                        ? ((2U == (3U 
                                                   & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                            ? (0xffff0000U 
                                               & (__Vtemp247[
                                                  (3U 
                                                   & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                      >> 4U))] 
                                                  << 0x10U))
                                            : __Vtemp248[
                                           (3U & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                  >> 4U))])
                                        : __Vtemp249[
                                       (3U & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                              >> 4U))]));
    this->__PVT__sb_mask = ((0U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                             ? 1U : ((1U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                      ? 2U : ((2U == 
                                               (3U 
                                                & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                               ? 4U
                                               : 8U)));
    this->__PVT__data_structures__DOT__data_use_per_way[0U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))][0U];
    this->__PVT__data_structures__DOT__data_use_per_way[1U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))][1U];
    this->__PVT__data_structures__DOT__data_use_per_way[2U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))][2U];
    this->__PVT__data_structures__DOT__data_use_per_way[3U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))][3U];
    this->__PVT__data_structures__DOT__data_use_per_way[4U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))][0U];
    this->__PVT__data_structures__DOT__data_use_per_way[5U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))][1U];
    this->__PVT__data_structures__DOT__data_use_per_way[6U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))][2U];
    this->__PVT__data_structures__DOT__data_use_per_way[7U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                   >> 6U))][3U];
    this->__PVT__data_structures__DOT__valid_use_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)) 
           | this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid
           [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                      >> 6U))]);
    this->__PVT__data_structures__DOT__valid_use_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)) 
           | (this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid
              [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                         >> 6U))] << 1U));
    this->__PVT__data_structures__DOT__tag_use_per_way 
        = ((VL_ULL(0x3ffffe00000) & this->__PVT__data_structures__DOT__tag_use_per_way) 
           | (IData)((IData)(this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag
                             [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                        >> 6U))])));
    this->__PVT__data_structures__DOT__tag_use_per_way 
        = ((VL_ULL(0x1fffff) & this->__PVT__data_structures__DOT__tag_use_per_way) 
           | ((QData)((IData)(this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag
                              [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                         >> 6U))])) 
              << 0x15U));
    this->__PVT__data_structures__DOT__dirty_use_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__dirty_use_per_way)) 
           | (IData)(this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use));
    this->__PVT__data_structures__DOT__dirty_use_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__dirty_use_per_way)) 
           | ((IData)(this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use) 
              << 1U));
    this->__PVT__data_write[0U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[8U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[1U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[9U]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[2U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[0xaU]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[3U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[0xbU]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_structures__DOT__invalid_found = 0U;
    if ((1U & (~ ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                  >> 1U)))) {
        this->__PVT__data_structures__DOT__invalid_found = 1U;
    }
    if ((1U & (~ (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)))) {
        this->__PVT__data_structures__DOT__invalid_found = 1U;
    }
    this->__PVT__data_structures__DOT__invalid_index = 0U;
    if ((1U & (~ ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                  >> 1U)))) {
        this->__PVT__data_structures__DOT__invalid_index = 1U;
    }
    if ((1U & (~ (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)))) {
        this->__PVT__data_structures__DOT__invalid_index = 0U;
    }
    this->__PVT__data_structures__DOT__hit_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way)) 
           | (((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
               & ((0x1fffffU & (IData)(this->__PVT__data_structures__DOT__tag_use_per_way)) 
                  == (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                   >> 0xbU)))) ? 1U
               : 0U));
    this->__PVT__data_structures__DOT__hit_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way)) 
           | (((((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                 >> 1U) & ((0x1fffffU & (IData)((this->__PVT__data_structures__DOT__tag_use_per_way 
                                                 >> 0x15U))) 
                           == (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                            >> 0xbU))))
                ? 1U : 0U) << 1U));
    this->__PVT__data_structures__DOT__data_write_per_way[0U] 
        = this->__PVT__data_write[0U];
    this->__PVT__data_structures__DOT__data_write_per_way[1U] 
        = this->__PVT__data_write[1U];
    this->__PVT__data_structures__DOT__data_write_per_way[2U] 
        = this->__PVT__data_write[2U];
    this->__PVT__data_structures__DOT__data_write_per_way[3U] 
        = this->__PVT__data_write[3U];
    this->__PVT__data_structures__DOT__data_write_per_way[4U] 
        = this->__PVT__data_write[0U];
    this->__PVT__data_structures__DOT__data_write_per_way[5U] 
        = this->__PVT__data_write[1U];
    this->__PVT__data_structures__DOT__data_write_per_way[6U] 
        = this->__PVT__data_write[2U];
    this->__PVT__data_structures__DOT__data_write_per_way[7U] 
        = this->__PVT__data_write[3U];
    this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 0U;
    if ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 1U;
    }
    if ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 1U;
    }
    this->__PVT__data_structures__DOT__way_index = 0U;
    if ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__way_index = 1U;
    }
    if ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__way_index = 0U;
    }
    this->__PVT__data_structures__DOT__way_use_Qual 
        = ((0U != (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state))
            ? (IData)(this->__PVT__way_to_update) : (IData)(this->__PVT__data_structures__DOT__way_index));
    this->__PVT__data_structures__DOT__write_from_mem_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)) 
           | ((IData)(this->__PVT__write_from_mem) 
              & (~ (IData)(this->__PVT__data_structures__DOT__way_use_Qual))));
    this->__PVT__data_structures__DOT__write_from_mem_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)) 
           | (((IData)(this->__PVT__write_from_mem) 
               & (IData)(this->__PVT__data_structures__DOT__way_use_Qual)) 
              << 1U));
    this->__Vcellout__data_structures__data_use[0U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(1U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                     << 2U))] >> (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                           << 7U))));
    this->__Vcellout__data_structures__data_use[1U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(2U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(1U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__Vcellout__data_structures__data_use[2U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(3U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(2U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__Vcellout__data_structures__data_use[3U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(4U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(3U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__PVT__valid_use = (1U & ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                                    >> (IData)(this->__PVT__data_structures__DOT__way_use_Qual)));
    this->__PVT__tag_use = ((0x29U >= (0x3fU & ((IData)(0x15U) 
                                                * (IData)(this->__PVT__data_structures__DOT__way_use_Qual))))
                             ? (0x1fffffU & (IData)(
                                                    (this->__PVT__data_structures__DOT__tag_use_per_way 
                                                     >> 
                                                     (0x3fU 
                                                      & ((IData)(0x15U) 
                                                         * (IData)(this->__PVT__data_structures__DOT__way_use_Qual))))))
                             : 0U);
    this->__PVT__data_unQual = (((0U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr)) 
                                 | (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_read)))
                                 ? this->__Vcellout__data_structures__data_use[
                                (3U & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                       >> 4U))] : (
                                                   (1U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                                    ? 
                                                   (this->__Vcellout__data_structures__data_use[
                                                    (3U 
                                                     & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                                        >> 4U))] 
                                                    >> 8U)
                                                    : 
                                                   ((2U 
                                                     == 
                                                     (3U 
                                                      & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                                     ? 
                                                    (this->__Vcellout__data_structures__data_use[
                                                     (3U 
                                                      & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                                         >> 4U))] 
                                                     >> 0x10U)
                                                     : 
                                                    (this->__Vcellout__data_structures__data_use[
                                                     (3U 
                                                      & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                                         >> 4U))] 
                                                     >> 0x18U))));
    this->__PVT__miss = (((this->__PVT__tag_use != 
                           (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                         >> 0xbU))) 
                          & (IData)(this->__PVT__valid_use)) 
                         & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__use_valid_in));
    this->__PVT__genblk1__BRA__0__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (0U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__1__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (1U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__2__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (2U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__3__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (3U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__we = ((0xfff0U & (IData)(this->__PVT__we)) 
                       | ((IData)(this->__PVT__write_from_mem)
                           ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                      & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                      ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                                 & (0U 
                                                    == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                 ? (IData)(this->__PVT__sb_mask)
                                                 : 
                                                (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                                  & (1U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? 
                                                 ((0U 
                                                   == 
                                                   (3U 
                                                    & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                                   ? 3U
                                                   : 0xcU)
                                                  : 0U)))));
    this->__PVT__we = ((0xff0fU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 4U));
    this->__PVT__we = ((0xf0ffU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 8U));
    this->__PVT__we = ((0xfffU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 0xcU));
    this->__PVT__data_structures__DOT__we_per_way = 
        ((0xffff0000U & this->__PVT__data_structures__DOT__we_per_way) 
         | ((IData)(this->__PVT__data_structures__DOT__way_use_Qual)
             ? 0U : (0xffffU & (IData)(this->__PVT__we))));
    this->__PVT__data_structures__DOT__we_per_way = 
        ((0xffffU & this->__PVT__data_structures__DOT__we_per_way) 
         | (((IData)(this->__PVT__data_structures__DOT__way_use_Qual)
              ? (0xffffU & (IData)(this->__PVT__we))
              : 0U) << 0x10U));
}

void Vcache_simX_VX_Cache_Bank__pi7::_settle__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure__4(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            Vcache_simX_VX_Cache_Bank__pi7::_settle__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure__4\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    WData/*31:0*/ __Vtemp253[4];
    WData/*31:0*/ __Vtemp254[4];
    WData/*31:0*/ __Vtemp255[4];
    WData/*31:0*/ __Vtemp256[4];
    WData/*31:0*/ __Vtemp257[4];
    WData/*31:0*/ __Vtemp258[4];
    WData/*31:0*/ __Vtemp259[4];
    // Body
    this->__PVT__way_to_update = vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__global_way_to_evict;
    this->__PVT__write_from_mem = ((2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state)) 
                                   & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__use_valid_in));
    this->__PVT__access = ((0U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state)) 
                           & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__use_valid_in));
    this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))];
    this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))];
    __Vtemp253[0U] = 0U;
    __Vtemp253[1U] = 0U;
    __Vtemp253[2U] = 0U;
    __Vtemp253[3U] = 0U;
    __Vtemp254[0U] = 0U;
    __Vtemp254[1U] = 0U;
    __Vtemp254[2U] = 0U;
    __Vtemp254[3U] = 0U;
    __Vtemp255[0U] = 0U;
    __Vtemp255[1U] = 0U;
    __Vtemp255[2U] = 0U;
    __Vtemp255[3U] = 0U;
    __Vtemp256[0U] = 0U;
    __Vtemp256[1U] = 0U;
    __Vtemp256[2U] = 0U;
    __Vtemp256[3U] = 0U;
    __Vtemp257[0U] = 0U;
    __Vtemp257[1U] = 0U;
    __Vtemp257[2U] = 0U;
    __Vtemp257[3U] = 0U;
    __Vtemp258[0U] = 0U;
    __Vtemp258[1U] = 0U;
    __Vtemp258[2U] = 0U;
    __Vtemp258[3U] = 0U;
    __Vtemp259[0U] = 0U;
    __Vtemp259[1U] = 0U;
    __Vtemp259[2U] = 0U;
    __Vtemp259[3U] = 0U;
    this->__PVT__use_write_data = ((0U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write))
                                    ? ((1U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                        ? (0xff00U 
                                           & (__Vtemp253[
                                              (3U & 
                                               ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                >> 6U))] 
                                              << 8U))
                                        : ((2U == (3U 
                                                   & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                            ? (0xff0000U 
                                               & (__Vtemp254[
                                                  (3U 
                                                   & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                      >> 6U))] 
                                                  << 0x10U))
                                            : ((3U 
                                                == 
                                                (3U 
                                                 & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                                ? (0xff000000U 
                                                   & (__Vtemp255[
                                                      (3U 
                                                       & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                          >> 6U))] 
                                                      << 0x18U))
                                                : __Vtemp256[
                                               (3U 
                                                & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                   >> 6U))])))
                                    : ((1U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write))
                                        ? ((2U == (3U 
                                                   & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                            ? (0xffff0000U 
                                               & (__Vtemp257[
                                                  (3U 
                                                   & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                      >> 6U))] 
                                                  << 0x10U))
                                            : __Vtemp258[
                                           (3U & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                  >> 6U))])
                                        : __Vtemp259[
                                       (3U & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                              >> 6U))]));
    this->__PVT__sb_mask = ((0U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                             ? 1U : ((1U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                      ? 2U : ((2U == 
                                               (3U 
                                                & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                               ? 4U
                                               : 8U)));
    this->__PVT__data_structures__DOT__data_use_per_way[0U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))][0U];
    this->__PVT__data_structures__DOT__data_use_per_way[1U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))][1U];
    this->__PVT__data_structures__DOT__data_use_per_way[2U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))][2U];
    this->__PVT__data_structures__DOT__data_use_per_way[3U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))][3U];
    this->__PVT__data_structures__DOT__data_use_per_way[4U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))][0U];
    this->__PVT__data_structures__DOT__data_use_per_way[5U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))][1U];
    this->__PVT__data_structures__DOT__data_use_per_way[6U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))][2U];
    this->__PVT__data_structures__DOT__data_use_per_way[7U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))][3U];
    this->__PVT__data_structures__DOT__valid_use_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)) 
           | this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid
           [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                      >> 6U))]);
    this->__PVT__data_structures__DOT__valid_use_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)) 
           | (this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid
              [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                         >> 6U))] << 1U));
    this->__PVT__data_structures__DOT__tag_use_per_way 
        = ((VL_ULL(0x3ffffe00000) & this->__PVT__data_structures__DOT__tag_use_per_way) 
           | (IData)((IData)(this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag
                             [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                        >> 6U))])));
    this->__PVT__data_structures__DOT__tag_use_per_way 
        = ((VL_ULL(0x1fffff) & this->__PVT__data_structures__DOT__tag_use_per_way) 
           | ((QData)((IData)(this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag
                              [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                         >> 6U))])) 
              << 0x15U));
    this->__PVT__data_structures__DOT__dirty_use_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__dirty_use_per_way)) 
           | (IData)(this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use));
    this->__PVT__data_structures__DOT__dirty_use_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__dirty_use_per_way)) 
           | ((IData)(this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use) 
              << 1U));
    this->__PVT__data_write[0U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[0xcU]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[1U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[0xdU]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[2U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[0xeU]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[3U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[0xfU]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_structures__DOT__invalid_found = 0U;
    if ((1U & (~ ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                  >> 1U)))) {
        this->__PVT__data_structures__DOT__invalid_found = 1U;
    }
    if ((1U & (~ (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)))) {
        this->__PVT__data_structures__DOT__invalid_found = 1U;
    }
    this->__PVT__data_structures__DOT__invalid_index = 0U;
    if ((1U & (~ ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                  >> 1U)))) {
        this->__PVT__data_structures__DOT__invalid_index = 1U;
    }
    if ((1U & (~ (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)))) {
        this->__PVT__data_structures__DOT__invalid_index = 0U;
    }
    this->__PVT__data_structures__DOT__hit_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way)) 
           | (((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
               & ((0x1fffffU & (IData)(this->__PVT__data_structures__DOT__tag_use_per_way)) 
                  == (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                   >> 0xbU)))) ? 1U
               : 0U));
    this->__PVT__data_structures__DOT__hit_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way)) 
           | (((((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                 >> 1U) & ((0x1fffffU & (IData)((this->__PVT__data_structures__DOT__tag_use_per_way 
                                                 >> 0x15U))) 
                           == (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                            >> 0xbU))))
                ? 1U : 0U) << 1U));
    this->__PVT__data_structures__DOT__data_write_per_way[0U] 
        = this->__PVT__data_write[0U];
    this->__PVT__data_structures__DOT__data_write_per_way[1U] 
        = this->__PVT__data_write[1U];
    this->__PVT__data_structures__DOT__data_write_per_way[2U] 
        = this->__PVT__data_write[2U];
    this->__PVT__data_structures__DOT__data_write_per_way[3U] 
        = this->__PVT__data_write[3U];
    this->__PVT__data_structures__DOT__data_write_per_way[4U] 
        = this->__PVT__data_write[0U];
    this->__PVT__data_structures__DOT__data_write_per_way[5U] 
        = this->__PVT__data_write[1U];
    this->__PVT__data_structures__DOT__data_write_per_way[6U] 
        = this->__PVT__data_write[2U];
    this->__PVT__data_structures__DOT__data_write_per_way[7U] 
        = this->__PVT__data_write[3U];
    this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 0U;
    if ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 1U;
    }
    if ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 1U;
    }
    this->__PVT__data_structures__DOT__way_index = 0U;
    if ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__way_index = 1U;
    }
    if ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__way_index = 0U;
    }
    this->__PVT__data_structures__DOT__way_use_Qual 
        = ((0U != (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state))
            ? (IData)(this->__PVT__way_to_update) : (IData)(this->__PVT__data_structures__DOT__way_index));
    this->__PVT__data_structures__DOT__write_from_mem_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)) 
           | ((IData)(this->__PVT__write_from_mem) 
              & (~ (IData)(this->__PVT__data_structures__DOT__way_use_Qual))));
    this->__PVT__data_structures__DOT__write_from_mem_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)) 
           | (((IData)(this->__PVT__write_from_mem) 
               & (IData)(this->__PVT__data_structures__DOT__way_use_Qual)) 
              << 1U));
    this->__Vcellout__data_structures__data_use[0U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(1U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                     << 2U))] >> (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                           << 7U))));
    this->__Vcellout__data_structures__data_use[1U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(2U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(1U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__Vcellout__data_structures__data_use[2U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(3U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(2U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__Vcellout__data_structures__data_use[3U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(4U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(3U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__PVT__valid_use = (1U & ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                                    >> (IData)(this->__PVT__data_structures__DOT__way_use_Qual)));
    this->__PVT__tag_use = ((0x29U >= (0x3fU & ((IData)(0x15U) 
                                                * (IData)(this->__PVT__data_structures__DOT__way_use_Qual))))
                             ? (0x1fffffU & (IData)(
                                                    (this->__PVT__data_structures__DOT__tag_use_per_way 
                                                     >> 
                                                     (0x3fU 
                                                      & ((IData)(0x15U) 
                                                         * (IData)(this->__PVT__data_structures__DOT__way_use_Qual))))))
                             : 0U);
    this->__PVT__data_unQual = (((0U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr)) 
                                 | (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_read)))
                                 ? this->__Vcellout__data_structures__data_use[
                                (3U & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                       >> 4U))] : (
                                                   (1U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                                    ? 
                                                   (this->__Vcellout__data_structures__data_use[
                                                    (3U 
                                                     & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                                        >> 4U))] 
                                                    >> 8U)
                                                    : 
                                                   ((2U 
                                                     == 
                                                     (3U 
                                                      & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                                     ? 
                                                    (this->__Vcellout__data_structures__data_use[
                                                     (3U 
                                                      & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                                         >> 4U))] 
                                                     >> 0x10U)
                                                     : 
                                                    (this->__Vcellout__data_structures__data_use[
                                                     (3U 
                                                      & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                                         >> 4U))] 
                                                     >> 0x18U))));
    this->__PVT__miss = (((this->__PVT__tag_use != 
                           (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                         >> 0xbU))) 
                          & (IData)(this->__PVT__valid_use)) 
                         & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__use_valid_in));
    this->__PVT__genblk1__BRA__0__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (0U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__1__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (1U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__2__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (2U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__3__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (3U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__we = ((0xfff0U & (IData)(this->__PVT__we)) 
                       | ((IData)(this->__PVT__write_from_mem)
                           ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                      & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                      ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                                 & (0U 
                                                    == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                 ? (IData)(this->__PVT__sb_mask)
                                                 : 
                                                (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                                  & (1U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? 
                                                 ((0U 
                                                   == 
                                                   (3U 
                                                    & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                                   ? 3U
                                                   : 0xcU)
                                                  : 0U)))));
    this->__PVT__we = ((0xff0fU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 4U));
    this->__PVT__we = ((0xf0ffU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 8U));
    this->__PVT__we = ((0xfffU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 0xcU));
    this->__PVT__data_structures__DOT__we_per_way = 
        ((0xffff0000U & this->__PVT__data_structures__DOT__we_per_way) 
         | ((IData)(this->__PVT__data_structures__DOT__way_use_Qual)
             ? 0U : (0xffffU & (IData)(this->__PVT__we))));
    this->__PVT__data_structures__DOT__we_per_way = 
        ((0xffffU & this->__PVT__data_structures__DOT__we_per_way) 
         | (((IData)(this->__PVT__data_structures__DOT__way_use_Qual)
              ? (0xffffU & (IData)(this->__PVT__we))
              : 0U) << 0x10U));
}

VL_INLINE_OPT void Vcache_simX_VX_Cache_Bank__pi7::_sequent__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure__8(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            Vcache_simX_VX_Cache_Bank__pi7::_sequent__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure__8\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Body
    this->__PVT__way_to_update = vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__global_way_to_evict;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 = 0U;
    if (vlTOPp->reset) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__ini_ind = 0x20U;
    }
    if ((1U & (~ (IData)(vlTOPp->reset)))) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__f = 4U;
    }
    if (vlTOPp->reset) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__ini_ind = 0x20U;
    }
    if ((1U & (~ (IData)(vlTOPp->reset)))) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__f = 4U;
    }
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v0 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 = 0U;
    this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 = 0U;
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v0 = 1U;
    } else {
        if ((1U & (((~ (IData)(this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use)) 
                    & (0U != (0xffffU & (this->__PVT__data_structures__DOT__we_per_way 
                                         >> 0x10U)))) 
                   | ((IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way) 
                      >> 1U)))) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 
                = ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))
                    ? 0U : (1U & (0U != (0xffffU & 
                                         (this->__PVT__data_structures__DOT__we_per_way 
                                          >> 0x10U)))));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v0 = 1U;
    } else {
        if ((1U & (((~ (IData)(this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use)) 
                    & (0U != (0xffffU & this->__PVT__data_structures__DOT__we_per_way))) 
                   | (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)))) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 
                = ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))
                    ? 0U : (1U & (0U != (0xffffU & this->__PVT__data_structures__DOT__we_per_way))));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v0 = 1U;
    } else {
        if ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))) {
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v0 = 1U;
    } else {
        if ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))) {
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v0 = 1U;
    } else {
        if ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 
                = (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                >> 0xbU));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v0 = 1U;
    } else {
        if ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way))) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 
                = (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                >> 0xbU));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 = 1U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v0 = 1U;
    } else {
        if ((0x10000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[4U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 = 0U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x20000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 = 8U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x40000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 = 0x10U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x80000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 = 0x18U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x100000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[5U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 = 0x20U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x200000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 = 0x28U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x400000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 = 0x30U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x800000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[5U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 = 0x38U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x1000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[6U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 = 0x40U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x2000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 = 0x48U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x4000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 = 0x50U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x8000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[6U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 = 0x58U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x10000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[7U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 = 0x60U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x20000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 
                = (0xffU & (this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                            >> 8U));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 = 0x68U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x40000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 
                = (0xffU & (this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                            >> 0x10U));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 = 0x70U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x80000000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 
                = (0xffU & (this->__PVT__data_structures__DOT__data_write_per_way[7U] 
                            >> 0x18U));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 = 0x78U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (vlTOPp->reset) {
        this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v0 = 1U;
    } else {
        if ((1U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[0U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 = 0U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((2U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[0U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 = 8U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((4U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[0U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 = 0x10U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((8U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[0U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 = 0x18U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x10U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[1U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 = 0x20U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x20U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 = 0x28U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x40U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 = 0x30U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x80U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[1U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 = 0x38U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x100U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[2U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 = 0x40U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x200U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 = 0x48U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x400U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 = 0x50U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x800U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[2U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 = 0x58U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x1000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 
                = (0xffU & this->__PVT__data_structures__DOT__data_write_per_way[3U]);
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 = 0x60U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x2000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                             << 0x18U) | (this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                                          >> 8U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 = 0x68U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x4000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                             << 0x10U) | (this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                                          >> 0x10U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 = 0x70U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
        if ((0x8000U & this->__PVT__data_structures__DOT__we_per_way)) {
            this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 
                = (0xffU & ((this->__PVT__data_structures__DOT__data_write_per_way[4U] 
                             << 8U) | (this->__PVT__data_structures__DOT__data_write_per_way[3U] 
                                       >> 0x18U)));
            this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 = 1U;
            this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 = 0x78U;
            this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 
                = (0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                            >> 6U));
        }
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32] 
            = this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32] 
            = this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32] = 1U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32] = 1U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32] 
            = this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[4U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[5U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[6U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[7U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[8U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[9U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xaU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xbU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xcU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xdU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xeU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0xfU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x10U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x11U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x12U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x13U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x14U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x15U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x16U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x17U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x18U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x19U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1aU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1bU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1cU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1dU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1eU] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[0x1fU] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32] 
            = this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[1U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[1U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[1U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[1U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[2U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[2U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[2U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[2U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[3U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[3U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[3U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[3U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[4U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[4U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[4U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[4U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[5U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[5U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[5U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[5U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[6U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[6U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[6U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[6U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[7U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[7U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[7U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[7U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[8U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[8U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[8U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[8U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[9U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[9U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[9U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[9U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xaU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xaU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xaU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xaU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xbU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xbU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xbU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xbU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xcU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xcU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xcU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xcU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xdU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xdU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xdU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xdU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xeU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xeU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xeU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xeU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xfU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xfU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xfU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0xfU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x10U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x10U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x10U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x10U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x11U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x11U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x11U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x11U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x12U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x12U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x12U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x12U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x13U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x13U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x13U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x13U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x14U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x14U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x14U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x14U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x15U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x15U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x15U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x15U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x16U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x16U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x16U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x16U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x17U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x17U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x17U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x17U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x18U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x18U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x18U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x18U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x19U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x19U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x19U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x19U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1aU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1aU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1aU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1aU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1bU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1bU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1bU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1bU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1cU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1cU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1cU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1cU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1dU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1dU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1dU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1dU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1eU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1eU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1eU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1eU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1fU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1fU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1fU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[0x1fU][3U] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47), 
                          this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47], this->__Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v0) {
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[1U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[1U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[1U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[1U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[2U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[2U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[2U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[2U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[3U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[3U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[3U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[3U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[4U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[4U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[4U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[4U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[5U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[5U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[5U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[5U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[6U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[6U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[6U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[6U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[7U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[7U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[7U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[7U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[8U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[8U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[8U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[8U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[9U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[9U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[9U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[9U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xaU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xaU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xaU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xaU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xbU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xbU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xbU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xbU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xcU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xcU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xcU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xcU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xdU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xdU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xdU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xdU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xeU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xeU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xeU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xeU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xfU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xfU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xfU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0xfU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x10U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x10U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x10U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x10U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x11U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x11U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x11U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x11U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x12U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x12U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x12U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x12U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x13U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x13U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x13U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x13U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x14U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x14U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x14U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x14U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x15U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x15U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x15U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x15U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x16U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x16U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x16U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x16U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x17U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x17U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x17U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x17U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x18U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x18U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x18U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x18U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x19U][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x19U][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x19U][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x19U][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1aU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1aU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1aU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1aU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1bU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1bU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1bU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1bU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1cU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1cU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1cU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1cU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1dU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1dU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1dU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1dU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1eU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1eU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1eU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1eU][3U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1fU][0U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1fU][1U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1fU][2U] = 0U;
        this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[0x1fU][3U] = 0U;
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46);
    }
    if (this->__Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47) {
        VL_ASSIGNSEL_WIII(8,(IData)(this->__Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47), 
                          this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
                          [this->__Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47], this->__Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47);
    }
}

VL_INLINE_OPT void Vcache_simX_VX_Cache_Bank__pi7::_combo__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure__12(Vcache_simX__Syms* __restrict vlSymsp) {
    VL_DEBUG_IF(VL_DBG_MSGF("+            Vcache_simX_VX_Cache_Bank__pi7::_combo__TOP__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure__12\n"); );
    Vcache_simX* __restrict vlTOPp VL_ATTR_UNUSED = vlSymsp->TOPp;
    // Variables
    WData/*31:0*/ __Vtemp327[4];
    WData/*31:0*/ __Vtemp328[4];
    WData/*31:0*/ __Vtemp329[4];
    WData/*31:0*/ __Vtemp330[4];
    WData/*31:0*/ __Vtemp331[4];
    WData/*31:0*/ __Vtemp332[4];
    WData/*31:0*/ __Vtemp333[4];
    // Body
    this->__PVT__write_from_mem = ((2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state)) 
                                   & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__use_valid_in));
    this->__PVT__access = ((0U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state)) 
                           & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__use_valid_in));
    this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))];
    this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))];
    __Vtemp327[0U] = 0U;
    __Vtemp327[1U] = 0U;
    __Vtemp327[2U] = 0U;
    __Vtemp327[3U] = 0U;
    __Vtemp328[0U] = 0U;
    __Vtemp328[1U] = 0U;
    __Vtemp328[2U] = 0U;
    __Vtemp328[3U] = 0U;
    __Vtemp329[0U] = 0U;
    __Vtemp329[1U] = 0U;
    __Vtemp329[2U] = 0U;
    __Vtemp329[3U] = 0U;
    __Vtemp330[0U] = 0U;
    __Vtemp330[1U] = 0U;
    __Vtemp330[2U] = 0U;
    __Vtemp330[3U] = 0U;
    __Vtemp331[0U] = 0U;
    __Vtemp331[1U] = 0U;
    __Vtemp331[2U] = 0U;
    __Vtemp331[3U] = 0U;
    __Vtemp332[0U] = 0U;
    __Vtemp332[1U] = 0U;
    __Vtemp332[2U] = 0U;
    __Vtemp332[3U] = 0U;
    __Vtemp333[0U] = 0U;
    __Vtemp333[1U] = 0U;
    __Vtemp333[2U] = 0U;
    __Vtemp333[3U] = 0U;
    this->__PVT__use_write_data = ((0U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write))
                                    ? ((1U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                        ? (0xff00U 
                                           & (__Vtemp327[
                                              (3U & 
                                               ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                >> 6U))] 
                                              << 8U))
                                        : ((2U == (3U 
                                                   & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                            ? (0xff0000U 
                                               & (__Vtemp328[
                                                  (3U 
                                                   & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                      >> 6U))] 
                                                  << 0x10U))
                                            : ((3U 
                                                == 
                                                (3U 
                                                 & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                                ? (0xff000000U 
                                                   & (__Vtemp329[
                                                      (3U 
                                                       & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                          >> 6U))] 
                                                      << 0x18U))
                                                : __Vtemp330[
                                               (3U 
                                                & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                   >> 6U))])))
                                    : ((1U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write))
                                        ? ((2U == (3U 
                                                   & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                            ? (0xffff0000U 
                                               & (__Vtemp331[
                                                  (3U 
                                                   & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                      >> 6U))] 
                                                  << 0x10U))
                                            : __Vtemp332[
                                           (3U & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                                  >> 6U))])
                                        : __Vtemp333[
                                       (3U & ((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank) 
                                              >> 6U))]));
    this->__PVT__sb_mask = ((0U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                             ? 1U : ((1U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                      ? 2U : ((2U == 
                                               (3U 
                                                & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                               ? 4U
                                               : 8U)));
    this->__PVT__data_structures__DOT__data_use_per_way[0U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))][0U];
    this->__PVT__data_structures__DOT__data_use_per_way[1U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))][1U];
    this->__PVT__data_structures__DOT__data_use_per_way[2U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))][2U];
    this->__PVT__data_structures__DOT__data_use_per_way[3U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))][3U];
    this->__PVT__data_structures__DOT__data_use_per_way[4U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))][0U];
    this->__PVT__data_structures__DOT__data_use_per_way[5U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))][1U];
    this->__PVT__data_structures__DOT__data_use_per_way[6U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))][2U];
    this->__PVT__data_structures__DOT__data_use_per_way[7U] 
        = this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data
        [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                   >> 6U))][3U];
    this->__PVT__data_structures__DOT__valid_use_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)) 
           | this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid
           [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                      >> 6U))]);
    this->__PVT__data_structures__DOT__valid_use_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)) 
           | (this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid
              [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                         >> 6U))] << 1U));
    this->__PVT__data_structures__DOT__tag_use_per_way 
        = ((VL_ULL(0x3ffffe00000) & this->__PVT__data_structures__DOT__tag_use_per_way) 
           | (IData)((IData)(this->__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag
                             [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                        >> 6U))])));
    this->__PVT__data_structures__DOT__tag_use_per_way 
        = ((VL_ULL(0x1fffff) & this->__PVT__data_structures__DOT__tag_use_per_way) 
           | ((QData)((IData)(this->__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag
                              [(0x1fU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                         >> 6U))])) 
              << 0x15U));
    this->__PVT__data_structures__DOT__dirty_use_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__dirty_use_per_way)) 
           | (IData)(this->data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use));
    this->__PVT__data_structures__DOT__dirty_use_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__dirty_use_per_way)) 
           | ((IData)(this->data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use) 
              << 1U));
    this->__PVT__data_write[0U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[0xcU]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[1U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[0xdU]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[2U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[0xeU]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_write[3U] = ((IData)(this->__PVT__write_from_mem)
                                    ? vlSymsp->TOP__cache_simX__DOT__VX_dram_req_rsp.i_m_readdata[0xfU]
                                    : this->__PVT__use_write_data);
    this->__PVT__data_structures__DOT__invalid_found = 0U;
    if ((1U & (~ ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                  >> 1U)))) {
        this->__PVT__data_structures__DOT__invalid_found = 1U;
    }
    if ((1U & (~ (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)))) {
        this->__PVT__data_structures__DOT__invalid_found = 1U;
    }
    this->__PVT__data_structures__DOT__invalid_index = 0U;
    if ((1U & (~ ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                  >> 1U)))) {
        this->__PVT__data_structures__DOT__invalid_index = 1U;
    }
    if ((1U & (~ (IData)(this->__PVT__data_structures__DOT__valid_use_per_way)))) {
        this->__PVT__data_structures__DOT__invalid_index = 0U;
    }
    this->__PVT__data_structures__DOT__hit_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way)) 
           | (((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
               & ((0x1fffffU & (IData)(this->__PVT__data_structures__DOT__tag_use_per_way)) 
                  == (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                   >> 0xbU)))) ? 1U
               : 0U));
    this->__PVT__data_structures__DOT__hit_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way)) 
           | (((((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                 >> 1U) & ((0x1fffffU & (IData)((this->__PVT__data_structures__DOT__tag_use_per_way 
                                                 >> 0x15U))) 
                           == (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                            >> 0xbU))))
                ? 1U : 0U) << 1U));
    this->__PVT__data_structures__DOT__data_write_per_way[0U] 
        = this->__PVT__data_write[0U];
    this->__PVT__data_structures__DOT__data_write_per_way[1U] 
        = this->__PVT__data_write[1U];
    this->__PVT__data_structures__DOT__data_write_per_way[2U] 
        = this->__PVT__data_write[2U];
    this->__PVT__data_structures__DOT__data_write_per_way[3U] 
        = this->__PVT__data_write[3U];
    this->__PVT__data_structures__DOT__data_write_per_way[4U] 
        = this->__PVT__data_write[0U];
    this->__PVT__data_structures__DOT__data_write_per_way[5U] 
        = this->__PVT__data_write[1U];
    this->__PVT__data_structures__DOT__data_write_per_way[6U] 
        = this->__PVT__data_write[2U];
    this->__PVT__data_structures__DOT__data_write_per_way[7U] 
        = this->__PVT__data_write[3U];
    this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 0U;
    if ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 1U;
    }
    if ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = 1U;
    }
    this->__PVT__data_structures__DOT__way_index = 0U;
    if ((2U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__way_index = 1U;
    }
    if ((1U & (IData)(this->__PVT__data_structures__DOT__hit_per_way))) {
        this->__PVT__data_structures__DOT__way_index = 0U;
    }
    this->__PVT__data_structures__DOT__way_use_Qual 
        = ((0U != (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state))
            ? (IData)(this->__PVT__way_to_update) : (IData)(this->__PVT__data_structures__DOT__way_index));
    this->__PVT__data_structures__DOT__write_from_mem_per_way 
        = ((2U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)) 
           | ((IData)(this->__PVT__write_from_mem) 
              & (~ (IData)(this->__PVT__data_structures__DOT__way_use_Qual))));
    this->__PVT__data_structures__DOT__write_from_mem_per_way 
        = ((1U & (IData)(this->__PVT__data_structures__DOT__write_from_mem_per_way)) 
           | (((IData)(this->__PVT__write_from_mem) 
               & (IData)(this->__PVT__data_structures__DOT__way_use_Qual)) 
              << 1U));
    this->__Vcellout__data_structures__data_use[0U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(1U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                     << 2U))] >> (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                           << 7U))));
    this->__Vcellout__data_structures__data_use[1U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(2U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(1U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__Vcellout__data_structures__data_use[2U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(3U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(2U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__Vcellout__data_structures__data_use[3U] 
        = (((0U == (0x1fU & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                             << 7U))) ? 0U : (this->__PVT__data_structures__DOT__data_use_per_way[
                                              ((IData)(4U) 
                                               + (4U 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 2U)))] 
                                              << ((IData)(0x20U) 
                                                  - 
                                                  (0x1fU 
                                                   & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                      << 7U))))) 
           | (this->__PVT__data_structures__DOT__data_use_per_way[
              ((IData)(3U) + (4U & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                    << 2U)))] >> (0x1fU 
                                                  & ((IData)(this->__PVT__data_structures__DOT__way_use_Qual) 
                                                     << 7U))));
    this->__PVT__valid_use = (1U & ((IData)(this->__PVT__data_structures__DOT__valid_use_per_way) 
                                    >> (IData)(this->__PVT__data_structures__DOT__way_use_Qual)));
    this->__PVT__tag_use = ((0x29U >= (0x3fU & ((IData)(0x15U) 
                                                * (IData)(this->__PVT__data_structures__DOT__way_use_Qual))))
                             ? (0x1fffffU & (IData)(
                                                    (this->__PVT__data_structures__DOT__tag_use_per_way 
                                                     >> 
                                                     (0x3fU 
                                                      & ((IData)(0x15U) 
                                                         * (IData)(this->__PVT__data_structures__DOT__way_use_Qual))))))
                             : 0U);
    this->__PVT__data_unQual = (((0U == (3U & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr)) 
                                 | (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_read)))
                                 ? this->__Vcellout__data_structures__data_use[
                                (3U & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                       >> 4U))] : (
                                                   (1U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                                    ? 
                                                   (this->__Vcellout__data_structures__data_use[
                                                    (3U 
                                                     & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                                        >> 4U))] 
                                                    >> 8U)
                                                    : 
                                                   ((2U 
                                                     == 
                                                     (3U 
                                                      & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                                     ? 
                                                    (this->__Vcellout__data_structures__data_use[
                                                     (3U 
                                                      & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                                         >> 4U))] 
                                                     >> 0x10U)
                                                     : 
                                                    (this->__Vcellout__data_structures__data_use[
                                                     (3U 
                                                      & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                                         >> 4U))] 
                                                     >> 0x18U))));
    this->__PVT__miss = (((this->__PVT__tag_use != 
                           (0x1fffffU & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                         >> 0xbU))) 
                          & (IData)(this->__PVT__valid_use)) 
                         & (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__use_valid_in));
    this->__PVT__genblk1__BRA__0__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (0U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__1__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (1U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__2__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (2U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__genblk1__BRA__3__KET____DOT__normal_write 
        = (((IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__read_or_write) 
            & ((IData)(this->__PVT__access) & (3U == 
                                               (3U 
                                                & (vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr 
                                                   >> 4U))))) 
           & (~ (IData)(this->__PVT__miss)));
    this->__PVT__we = ((0xfff0U & (IData)(this->__PVT__we)) 
                       | ((IData)(this->__PVT__write_from_mem)
                           ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                      & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                      ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                                 & (0U 
                                                    == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                 ? (IData)(this->__PVT__sb_mask)
                                                 : 
                                                (((IData)(this->__PVT__genblk1__BRA__0__KET____DOT__normal_write) 
                                                  & (1U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? 
                                                 ((0U 
                                                   == 
                                                   (3U 
                                                    & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                                   ? 3U
                                                   : 0xcU)
                                                  : 0U)))));
    this->__PVT__we = ((0xff0fU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__1__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 4U));
    this->__PVT__we = ((0xf0ffU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__2__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 8U));
    this->__PVT__we = ((0xfffU & (IData)(this->__PVT__we)) 
                       | (((IData)(this->__PVT__write_from_mem)
                            ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                       & (2U == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                       ? 0xfU : (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                                  & (0U 
                                                     == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                  ? (IData)(this->__PVT__sb_mask)
                                                  : 
                                                 (((IData)(this->__PVT__genblk1__BRA__3__KET____DOT__normal_write) 
                                                   & (1U 
                                                      == (IData)(vlTOPp->cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write)))
                                                   ? 
                                                  ((0U 
                                                    == 
                                                    (3U 
                                                     & vlTOPp->cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr))
                                                    ? 3U
                                                    : 0xcU)
                                                   : 0U)))) 
                          << 0xcU));
    this->__PVT__data_structures__DOT__we_per_way = 
        ((0xffff0000U & this->__PVT__data_structures__DOT__we_per_way) 
         | ((IData)(this->__PVT__data_structures__DOT__way_use_Qual)
             ? 0U : (0xffffU & (IData)(this->__PVT__we))));
    this->__PVT__data_structures__DOT__we_per_way = 
        ((0xffffU & this->__PVT__data_structures__DOT__we_per_way) 
         | (((IData)(this->__PVT__data_structures__DOT__way_use_Qual)
              ? (0xffffU & (IData)(this->__PVT__we))
              : 0U) << 0x10U));
}

void Vcache_simX_VX_Cache_Bank__pi7::_ctor_var_reset() {
    VL_DEBUG_IF(VL_DBG_MSGF("+            Vcache_simX_VX_Cache_Bank__pi7::_ctor_var_reset\n"); );
    // Body
    rst = VL_RAND_RESET_I(1);
    clk = VL_RAND_RESET_I(1);
    state = VL_RAND_RESET_I(4);
    actual_index = VL_RAND_RESET_I(5);
    o_tag = VL_RAND_RESET_I(21);
    block_offset = VL_RAND_RESET_I(2);
    writedata = VL_RAND_RESET_I(32);
    valid_in = VL_RAND_RESET_I(1);
    read_or_write = VL_RAND_RESET_I(1);
    VL_RAND_RESET_W(128,fetched_writedata);
    i_p_mem_read = VL_RAND_RESET_I(3);
    i_p_mem_write = VL_RAND_RESET_I(3);
    byte_select = VL_RAND_RESET_I(2);
    evicted_way = VL_RAND_RESET_I(1);
    readdata = VL_RAND_RESET_I(32);
    hit = VL_RAND_RESET_I(1);
    eviction_wb = VL_RAND_RESET_I(1);
    eviction_addr = VL_RAND_RESET_I(32);
    VL_RAND_RESET_W(128,data_evicted);
    __PVT__tag_use = VL_RAND_RESET_I(21);
    __PVT__valid_use = VL_RAND_RESET_I(1);
    __PVT__access = VL_RAND_RESET_I(1);
    __PVT__write_from_mem = VL_RAND_RESET_I(1);
    __PVT__miss = VL_RAND_RESET_I(1);
    __PVT__way_to_update = VL_RAND_RESET_I(1);
    __PVT__data_unQual = VL_RAND_RESET_I(32);
    __PVT__use_write_data = VL_RAND_RESET_I(32);
    __PVT__sb_mask = VL_RAND_RESET_I(4);
    __PVT__we = VL_RAND_RESET_I(16);
    VL_RAND_RESET_W(128,__PVT__data_write);
    VL_RAND_RESET_W(128,__Vcellout__data_structures__data_use);
    __PVT__genblk1__BRA__0__KET____DOT__normal_write = VL_RAND_RESET_I(1);
    __PVT__genblk1__BRA__1__KET____DOT__normal_write = VL_RAND_RESET_I(1);
    __PVT__genblk1__BRA__2__KET____DOT__normal_write = VL_RAND_RESET_I(1);
    __PVT__genblk1__BRA__3__KET____DOT__normal_write = VL_RAND_RESET_I(1);
    __PVT__data_structures__DOT__tag_use_per_way = VL_RAND_RESET_Q(42);
    VL_RAND_RESET_W(256,__PVT__data_structures__DOT__data_use_per_way);
    __PVT__data_structures__DOT__valid_use_per_way = VL_RAND_RESET_I(2);
    __PVT__data_structures__DOT__dirty_use_per_way = VL_RAND_RESET_I(2);
    __PVT__data_structures__DOT__hit_per_way = VL_RAND_RESET_I(2);
    __PVT__data_structures__DOT__we_per_way = VL_RAND_RESET_I(32);
    VL_RAND_RESET_W(256,__PVT__data_structures__DOT__data_write_per_way);
    __PVT__data_structures__DOT__write_from_mem_per_way = VL_RAND_RESET_I(2);
    __PVT__data_structures__DOT__invalid_found = VL_RAND_RESET_I(1);
    __PVT__data_structures__DOT__way_index = VL_RAND_RESET_I(1);
    __PVT__data_structures__DOT__invalid_index = VL_RAND_RESET_I(1);
    __PVT__data_structures__DOT__way_use_Qual = VL_RAND_RESET_I(1);
    data_structures__DOT____Vcellout__each_way__BRA__0__KET____DOT__data_structures__dirty_use = VL_RAND_RESET_I(1);
    data_structures__DOT____Vcellout__each_way__BRA__1__KET____DOT__data_structures__dirty_use = VL_RAND_RESET_I(1);
    __PVT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found = VL_RAND_RESET_I(1);
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
            VL_RAND_RESET_W(128,__PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[__Vi0]);
    }}
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
            __PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[__Vi0] = VL_RAND_RESET_I(21);
    }}
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
            __PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[__Vi0] = VL_RAND_RESET_I(1);
    }}
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
            __PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[__Vi0] = VL_RAND_RESET_I(1);
    }}
    __PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__f = VL_RAND_RESET_I(32);
    __PVT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__ini_ind = VL_RAND_RESET_I(32);
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
            VL_RAND_RESET_W(128,__PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[__Vi0]);
    }}
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
            __PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[__Vi0] = VL_RAND_RESET_I(21);
    }}
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
            __PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[__Vi0] = VL_RAND_RESET_I(1);
    }}
    { int __Vi0=0; for (; __Vi0<32; ++__Vi0) {
            __PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[__Vi0] = VL_RAND_RESET_I(1);
    }}
    __PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__f = VL_RAND_RESET_I(32);
    __PVT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__ini_ind = VL_RAND_RESET_I(32);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v0 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32 = 0;
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid__v32 = 0;
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v0 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 = VL_RAND_RESET_I(21);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag__v32 = 0;
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v0 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v32 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v33 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v34 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v35 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v36 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v37 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v38 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v39 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v40 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v41 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v42 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v43 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v44 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v45 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v46 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data__v47 = 0;
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v0 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 = VL_RAND_RESET_I(1);
    __Vdlyvset__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty__v32 = 0;
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v0 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32 = 0;
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid__v32 = 0;
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v0 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 = VL_RAND_RESET_I(21);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag__v32 = 0;
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v0 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v32 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v33 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v34 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v35 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v36 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v37 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v38 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v39 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v40 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v41 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v42 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v43 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v44 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v45 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v46 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 = 0;
    __Vdlyvlsb__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 = VL_RAND_RESET_I(8);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data__v47 = 0;
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v0 = 0;
    __Vdlyvdim0__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 = 0;
    __Vdlyvval__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 = VL_RAND_RESET_I(1);
    __Vdlyvset__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty__v32 = 0;
}
