// Verilated -*- C++ -*-
// DESCRIPTION: Verilator output: Primary design header
//
// This header should be included by all source files instantiating the design.
// The class here is then constructed to instantiate the design.
// See the Verilator manual for examples.

#ifndef _Vcache_simX_H_
#define _Vcache_simX_H_

#include "verilated.h"

class Vcache_simX__Syms;
class Vcache_simX_VX_dram_req_rsp_inter__N1_NB4;
class Vcache_simX_VX_dcache_request_inter;
class Vcache_simX_VX_dram_req_rsp_inter__N4_NB4;
class Vcache_simX_VX_Cache_Bank__pi8;
class VerilatedVcd;

//----------

VL_MODULE(Vcache_simX) {
  public:
    // CELLS
    // Public to allow access to /*verilator_public*/ items;
    // otherwise the application code can consider these internals.
    Vcache_simX_VX_dram_req_rsp_inter__N1_NB4*	__PVT__cache_simX__DOT__VX_dram_req_rsp_icache;
    Vcache_simX_VX_dcache_request_inter*	__PVT__cache_simX__DOT__VX_dcache_req;
    Vcache_simX_VX_dram_req_rsp_inter__N4_NB4*	__PVT__cache_simX__DOT__VX_dram_req_rsp;
    Vcache_simX_VX_Cache_Bank__pi8*	__PVT__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_structure;
    Vcache_simX_VX_Cache_Bank__pi8*	__PVT__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_structure;
    Vcache_simX_VX_Cache_Bank__pi8*	__PVT__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_structure;
    Vcache_simX_VX_Cache_Bank__pi8*	__PVT__cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_structure;
    
    // PORTS
    // The application code writes and reads these signals to
    // propagate new values into/out from the Verilated model.
    VL_IN8(clk,0,0);
    VL_IN8(reset,0,0);
    VL_IN8(in_icache_valid_pc_addr,0,0);
    VL_OUT8(out_icache_stall,0,0);
    VL_IN8(in_dcache_mem_read,2,0);
    VL_IN8(in_dcache_mem_write,2,0);
    VL_OUT8(out_dcache_stall,0,0);
    VL_IN(in_icache_pc_addr,31,0);
    VL_IN8(in_dcache_in_valid[4],0,0);
    VL_IN(in_dcache_in_address[4],31,0);
    
    // LOCAL SIGNALS
    // Internals; generally not touched by application code
    VL_SIG8(cache_simX__DOT__icache_i_m_ready,0,0);
    VL_SIG8(cache_simX__DOT__dcache_i_m_ready,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__sm_driver_in_valid,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__cache_driver_in_valid,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__read_or_write,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_read,2,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__cache_driver_in_mem_write,2,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__sm_driver_in_mem_read,2,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__sm_driver_in_mem_write,2,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache_driver_in_mem_read,2,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__temp_out_valid,3,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__block_addr,27,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__block_we,7,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__orig_in_valid,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__genblk2__BRA__0__KET____DOT__shm_write,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__genblk2__BRA__1__KET____DOT__shm_write,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__genblk2__BRA__2__KET____DOT__shm_write,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__genblk2__BRA__3__KET____DOT__shm_write,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__left_requests,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__serviced,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__use_valid,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__more_than_one_valid,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__internal_req_num,7,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__internal_out_valid,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__serviced_qual,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__new_left_requests,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk1__BRA__0__KET____DOT__num_valids,2,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk1__BRA__1__KET____DOT__num_valids,2,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk1__BRA__2__KET____DOT__num_valids,2,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk1__BRA__3__KET____DOT__num_valids,2,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__global_way_to_evict,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__index_per_bank,7,0);
    VL_SIG16(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__use_mask_per_bank,15,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__valid_per_bank,3,0);
    VL_SIG16(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__threads_serviced_per_bank,15,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__hit_per_bank,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__eviction_wb,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__eviction_wb_old,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__state,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__new_state,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__use_valid,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__stored_valid,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__new_stored_valid,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__threads_serviced_Qual,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__detect_bank_miss,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__miss_bank_index,1,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__miss_found,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__update_global_way_to_evict,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__use_valid_in,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__use_valid_in,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__use_valid_in,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__use_valid_in,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__global_way_to_evict,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__valid_per_bank,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__threads_serviced_per_bank,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__hit_per_bank,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__eviction_wb_old,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__state,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__new_state,3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__use_valid,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__stored_valid,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__new_stored_valid,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__detect_bank_miss,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__miss_bank_index,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__miss_found,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__update_global_way_to_evict,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__use_valid_in,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__valid_use,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__access,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__write_from_mem,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__way_to_update,0,0);
    VL_SIG16(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__we,15,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__valid_use_per_way,1,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__dirty_use_per_way,1,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__hit_per_way,1,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__we_per_way,31,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__write_from_mem_per_way,1,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__invalid_found,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__way_index,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__invalid_index,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__way_use_Qual,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__genblk1__DOT__way_indexing__DOT__found,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__update_dirty,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__update_dirty,0,0);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__temp_out_data,127,0,4);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__block_wdata,511,0,16);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__block_rdata,511,0,16);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__0__KET____DOT__vx_priority_encoder__DOT__i,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__1__KET____DOT__vx_priority_encoder__DOT__i,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__2__KET____DOT__vx_priority_encoder__DOT__i,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__3__KET____DOT__vx_priority_encoder__DOT__i,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__genblk2__BRA__0__KET____DOT__vx_shared_memory_block__DOT__curr_ind,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__genblk2__BRA__1__KET____DOT__vx_shared_memory_block__DOT__curr_ind,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__genblk2__BRA__2__KET____DOT__vx_shared_memory_block__DOT__curr_ind,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__genblk2__BRA__3__KET____DOT__vx_shared_memory_block__DOT__curr_ind,31,0);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__final_data_read,127,0,4);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__new_final_data_read,127,0,4);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__new_final_data_read_Qual,127,0,4);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__readdata_per_bank,127,0,4);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__eviction_addr_per_bank,127,0,4);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__miss_addr,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__init_b,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__0__KET____DOT__bank_addr,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__1__KET____DOT__bank_addr,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__2__KET____DOT__bank_addr,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk3__BRA__3__KET____DOT__bank_addr,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__get_miss_index__DOT__i,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__0__KET____DOT__choose_thread__DOT__i,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__1__KET____DOT__choose_thread__DOT__i,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__2__KET____DOT__choose_thread__DOT__i,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__3__KET____DOT__choose_thread__DOT__i,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__icache__DOT__final_data_read,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__icache__DOT__new_final_data_read,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__icache__DOT__new_final_data_read_Qual,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__icache__DOT__miss_addr,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__icache__DOT__init_b,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_addr,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__tag_use,22,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_unQual,31,0);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_write,127,0,4);
    VL_SIG64(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__tag_use_per_way,45,0);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__data_use_per_way,255,0,8);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__data_write_per_way,255,0,8);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__f,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__ini_ind,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__f,31,0);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__ini_ind,31,0);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__genblk2__BRA__0__KET____DOT__vx_shared_memory_block__DOT__shared_memory[128],127,0,4);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__genblk2__BRA__1__KET____DOT__vx_shared_memory_block__DOT__shared_memory[128],127,0,4);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__genblk2__BRA__2__KET____DOT__vx_shared_memory_block__DOT__shared_memory[128],127,0,4);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__genblk2__BRA__3__KET____DOT__vx_shared_memory_block__DOT__shared_memory[128],127,0,4);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__debug_hit_per_bank_mask[4],3,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__debug_hit_per_bank_mask[1],0,0);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__data[32],127,0,4);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__tag[32],22,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__valid[32],0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__each_way__BRA__0__KET____DOT__data_structures__DOT__dirty[32],0,0);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__data[32],127,0,4);
    VL_SIG(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__tag[32],22,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__valid[32],0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk3__BRA__0__KET____DOT__bank_structure__DOT__data_structures__DOT__each_way__BRA__1__KET____DOT__data_structures__DOT__dirty[32],0,0);
    
    // LOCAL VARIABLES
    // Internals; generally not touched by application code
    static VL_ST_SIG8(__Vtable1_cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__0__KET____DOT____Vcellout__vx_priority_encoder__index[16],1,0);
    static VL_ST_SIG8(__Vtable1_cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__0__KET____DOT____Vcellout__vx_priority_encoder__found[16],0,0);
    static VL_ST_SIG(__Vtable1_cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__0__KET____DOT__vx_priority_encoder__DOT__i[16],31,0);
    static VL_ST_SIG8(__Vtable2_cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__1__KET____DOT____Vcellout__vx_priority_encoder__index[16],1,0);
    static VL_ST_SIG8(__Vtable2_cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__1__KET____DOT____Vcellout__vx_priority_encoder__found[16],0,0);
    static VL_ST_SIG(__Vtable2_cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__1__KET____DOT__vx_priority_encoder__DOT__i[16],31,0);
    static VL_ST_SIG8(__Vtable3_cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__2__KET____DOT____Vcellout__vx_priority_encoder__index[16],1,0);
    static VL_ST_SIG8(__Vtable3_cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__2__KET____DOT____Vcellout__vx_priority_encoder__found[16],0,0);
    static VL_ST_SIG(__Vtable3_cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__2__KET____DOT__vx_priority_encoder__DOT__i[16],31,0);
    static VL_ST_SIG8(__Vtable4_cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__3__KET____DOT____Vcellout__vx_priority_encoder__index[16],1,0);
    static VL_ST_SIG8(__Vtable4_cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__3__KET____DOT____Vcellout__vx_priority_encoder__found[16],0,0);
    static VL_ST_SIG(__Vtable4_cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__3__KET____DOT__vx_priority_encoder__DOT__i[16],31,0);
    static VL_ST_SIG8(__Vtable5_cache_simX__DOT__dmem_controller__DOT__dcache__DOT__miss_bank_index[16],1,0);
    static VL_ST_SIG8(__Vtable5_cache_simX__DOT__dmem_controller__DOT__dcache__DOT__miss_found[16],0,0);
    static VL_ST_SIG(__Vtable5_cache_simX__DOT__dmem_controller__DOT__dcache__DOT__get_miss_index__DOT__i[16],31,0);
    static VL_ST_SIG8(__Vtable6_cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__0__KET____DOT____Vcellout__choose_thread__index[16],1,0);
    static VL_ST_SIG8(__Vtable6_cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__0__KET____DOT____Vcellout__choose_thread__found[16],0,0);
    static VL_ST_SIG(__Vtable6_cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__0__KET____DOT__choose_thread__DOT__i[16],31,0);
    static VL_ST_SIG8(__Vtable7_cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__1__KET____DOT____Vcellout__choose_thread__index[16],1,0);
    static VL_ST_SIG8(__Vtable7_cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__1__KET____DOT____Vcellout__choose_thread__found[16],0,0);
    static VL_ST_SIG(__Vtable7_cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__1__KET____DOT__choose_thread__DOT__i[16],31,0);
    static VL_ST_SIG8(__Vtable8_cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__2__KET____DOT____Vcellout__choose_thread__index[16],1,0);
    static VL_ST_SIG8(__Vtable8_cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__2__KET____DOT____Vcellout__choose_thread__found[16],0,0);
    static VL_ST_SIG(__Vtable8_cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__2__KET____DOT__choose_thread__DOT__i[16],31,0);
    static VL_ST_SIG8(__Vtable9_cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__3__KET____DOT____Vcellout__choose_thread__index[16],1,0);
    static VL_ST_SIG8(__Vtable9_cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__3__KET____DOT____Vcellout__choose_thread__found[16],0,0);
    static VL_ST_SIG(__Vtable9_cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__3__KET____DOT__choose_thread__DOT__i[16],31,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT____Vlvbound1,6,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT____Vlvbound2,6,0);
    VL_SIG16(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT____Vcellout__vx_bank_valid__bank_valids,15,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__0__KET____DOT____Vcellout__vx_priority_encoder__found,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__0__KET____DOT____Vcellout__vx_priority_encoder__index,1,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__1__KET____DOT____Vcellout__vx_priority_encoder__found,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__1__KET____DOT____Vcellout__vx_priority_encoder__index,1,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__2__KET____DOT____Vcellout__vx_priority_encoder__found,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__2__KET____DOT____Vcellout__vx_priority_encoder__index,1,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__3__KET____DOT____Vcellout__vx_priority_encoder__found,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__vx_priority_encoder_sm__DOT__genblk2__BRA__3__KET____DOT____Vcellout__vx_priority_encoder__index,1,0);
    VL_SIG16(cache_simX__DOT__dmem_controller__DOT__dcache__DOT____Vcellout__multip_banks__thread_track_banks,15,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__0__KET____DOT____Vcellout__choose_thread__found,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__0__KET____DOT____Vcellout__choose_thread__index,1,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__1__KET____DOT____Vcellout__choose_thread__found,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__1__KET____DOT____Vcellout__choose_thread__index,1,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__2__KET____DOT____Vcellout__choose_thread__found,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__2__KET____DOT____Vcellout__choose_thread__index,1,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__3__KET____DOT____Vcellout__choose_thread__found,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__dcache__DOT__genblk1__BRA__3__KET____DOT____Vcellout__choose_thread__index,1,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT____Vcellout__multip_banks__thread_track_banks,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__genblk1__BRA__0__KET____DOT____Vcellout__choose_thread__index,0,0);
    VL_SIG8(cache_simX__DOT__dmem_controller__DOT__icache__DOT__multip_banks__DOT____Vlvbound1,0,0);
    VL_SIG8(__Vtableidx1,3,0);
    VL_SIG8(__Vtableidx2,3,0);
    VL_SIG8(__Vtableidx3,3,0);
    VL_SIG8(__Vtableidx4,3,0);
    VL_SIG8(__Vtableidx5,3,0);
    VL_SIG8(__Vtableidx6,3,0);
    VL_SIG8(__Vtableidx7,3,0);
    VL_SIG8(__Vtableidx8,3,0);
    VL_SIG8(__Vtableidx9,3,0);
    VL_SIG8(__Vclklast__TOP__clk,0,0);
    VL_SIG8(__Vclklast__TOP__reset,0,0);
    VL_SIG(__Vchglast__TOP__cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT__block_addr,27,0);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT____Vcellout__dcache__o_m_writedata,511,0,16);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT____Vcellout__vx_priority_encoder_sm__out_data,127,0,4);
    VL_SIGW(cache_simX__DOT__dmem_controller__DOT__shared_memory__DOT____Vcellout__vx_priority_encoder_sm__out_address,127,0,4);
    VL_SIG(__Vm_traceActivity,31,0);
    
    // INTERNAL VARIABLES
    // Internals; generally not touched by application code
    Vcache_simX__Syms* __VlSymsp;  // Symbol table
    
    // PARAMETERS
    // Parameters marked /*verilator public*/ for use by application code
    
    // CONSTRUCTORS
  private:
    Vcache_simX& operator= (const Vcache_simX&);  ///< Copying not allowed
    Vcache_simX(const Vcache_simX&);  ///< Copying not allowed
  public:
    /// Construct the model; called by application code
    /// The special name  may be used to make a wrapper with a
    /// single model invisible WRT DPI scope names.
    Vcache_simX(const char* name="TOP");
    /// Destroy the model; called (often implicitly) by application code
    ~Vcache_simX();
    /// Trace signals in the model; called by application code
    void trace (VerilatedVcdC* tfp, int levels, int options=0);
    
    // API METHODS
    /// Evaluate the model.  Application must call when inputs change.
    void eval();
    /// Simulation complete, run final blocks.  Application must call on completion.
    void final();
    
    // INTERNAL METHODS
  private:
    static void _eval_initial_loop(Vcache_simX__Syms* __restrict vlSymsp);
  public:
    void __Vconfigure(Vcache_simX__Syms* symsp, bool first);
  private:
    static QData _change_request(Vcache_simX__Syms* __restrict vlSymsp);
  public:
    static void _combo__TOP__1(Vcache_simX__Syms* __restrict vlSymsp);
    static void _combo__TOP__5(Vcache_simX__Syms* __restrict vlSymsp);
  private:
    void _ctor_var_reset();
  public:
    static void _eval(Vcache_simX__Syms* __restrict vlSymsp);
  private:
#ifdef VL_DEBUG
    void _eval_debug_assertions();
#endif // VL_DEBUG
  public:
    static void _eval_initial(Vcache_simX__Syms* __restrict vlSymsp);
    static void _eval_settle(Vcache_simX__Syms* __restrict vlSymsp);
    static void _sequent__TOP__4(Vcache_simX__Syms* __restrict vlSymsp);
    static void _settle__TOP__2(Vcache_simX__Syms* __restrict vlSymsp);
    static void _settle__TOP__3(Vcache_simX__Syms* __restrict vlSymsp);
    static void traceChgThis(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void traceChgThis__2(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void traceChgThis__3(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void traceChgThis__4(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void traceChgThis__5(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void traceChgThis__6(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void traceFullThis(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void traceFullThis__1(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void traceInitThis(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void traceInitThis__1(Vcache_simX__Syms* __restrict vlSymsp, VerilatedVcd* vcdp, uint32_t code);
    static void traceInit (VerilatedVcd* vcdp, void* userthis, uint32_t code);
    static void traceFull (VerilatedVcd* vcdp, void* userthis, uint32_t code);
    static void traceChg  (VerilatedVcd* vcdp, void* userthis, uint32_t code);
} VL_ATTR_ALIGNED(128);

#endif // guard
