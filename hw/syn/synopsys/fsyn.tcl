# Copyright © 2019-2023
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set search_path			[concat ../../models/memory/cln28hpm/rf2_128x128_wm1 ../../models/memory/cln28hpm/rf2_256x128_wm1 ../../models/memory/cln28hpm/rf2_256_19_wm0 ../../models/memory/cln28hpm/rf2_32x128_wm1  ../../rtl/ ../../rtl/interfaces ../../rtl/pipe_regs ../../rtl/shared_memory ../../rtl/cache]
set link_library		[concat NanGate_15nm_OCL.db]
set symbol_library		{}
set target_library		[concat NanGate_15nm_OCL.db]
set verilog_files 	[ list VX_countones.v VX_priority_encoder_w_mask.v VX_dram_req_rsp_inter.v VX_cache_bank_valid.v VX_cache_data_per_index.v VX_Cache_Bank.v VX_cache_data.v VX_d_cache.v VX_bank_valids.v VX_priority_encoder_sm.v VX_shared_memory.v VX_shared_memory_block.v VX_dmem_controller.v VX_generic_priority_encoder.v VX_generic_stack.v VX_join_inter.v VX_csr_wrapper.v VX_csr_req_inter.v VX_csr_wb_inter.v  VX_gpgpu_inst.v VX_gpu_inst_req_inter.v VX_wstall_inter.v VX_inst_exec_wb_inter.v VX_lsu.v VX_execute_unit.v VX_lsu_addr_gen.v VX_inst_multiplex.v VX_exec_unit_req_inter.v VX_lsu_req_inter.v VX_alu.v VX_back_end.v VX_gpr_stage.v VX_gpr_data_inter.v VX_csr_handler.v VX_decode.v VX_define.vh VX_scheduler.v VX_fetch.v VX_front_end.v VX_generic_register.v VX_gpr.v VX_gpr_wrapper.v VX_priority_encoder.v VX_warp_scheduler.v VX_writeback.v byte_enabled_simple_dual_port_ram.v VX_branch_response_inter.v VX_dcache_request_inter.v VX_dcache_response_inter.v VX_frE_to_bckE_req_inter.v VX_gpr_jal_inter.v VX_gpr_read_inter.v VX_icache_request_inter.v VX_icache_response_inter.v VX_inst_mem_wb_inter.v VX_inst_meta_inter.v VX_jal_response_inter.v VX_mem_req_inter.v VX_mw_wb_inter.v VX_warp_ctl_inter.v VX_wb_inter.v VX_d_e_reg.v VX_f_d_reg.v Vortex.v rf2_128x128_wm1.v \
					]
# set verilog_files 	[ list VX_countones.v VX_priority_encoder_w_mask.v VX_dram_req_rsp_inter.v cache_set.v VX_Cache_Bank.v VX_Cache_Block_DM.v VX_cache_data.v VX_d_cache.v VX_bank_valids.v VX_priority_encoder_sm.v VX_shared_memory.v VX_shared_memory_block.v VX_dmem_controller.v VX_generic_priority_encoder.v VX_generic_stack.v VX_join_inter.v VX_csr_wrapper.v VX_csr_req_inter.v VX_csr_wb_inter.v  VX_gpgpu_inst.v VX_gpu_inst_req_inter.v VX_wstall_inter.v VX_inst_exec_wb_inter.v VX_lsu.v VX_execute_unit.v VX_lsu_addr_gen.v VX_inst_multiplex.v VX_exec_unit_req_inter.v VX_lsu_req_inter.v VX_alu.v VX_back_end.v VX_gpr_stage.v VX_gpr_data_inter.v VX_csr_handler.v VX_decode.v VX_define.vh VX_scheduler.v VX_fetch.v VX_front_end.v VX_generic_register.v VX_gpr.v VX_gpr_wrapper.v VX_one_counter.v VX_priority_encoder.v VX_warp_scheduler.v VX_writeback.v byte_enabled_simple_dual_port_ram.v VX_branch_response_inter.v VX_dcache_request_inter.v VX_dcache_response_inter.v VX_frE_to_bckE_req_inter.v VX_gpr_jal_inter.v VX_gpr_read_inter.v VX_icache_request_inter.v VX_icache_response_inter.v VX_inst_mem_wb_inter.v VX_inst_meta_inter.v VX_jal_response_inter.v VX_mem_req_inter.v VX_mw_wb_inter.v VX_warp_ctl_inter.v VX_wb_inter.v VX_d_e_reg.v VX_f_d_reg.v Vortex.v \
# 					]

set top_level Vortex
analyze -format sverilog $verilog_files
elaborate Vortex
link

set clk_freq 100
set clk_period [expr 100.0 / $clk_freq / 1.0]
create_clock [get_ports clk] -period $clk_period
set_max_fanout 20 [get_ports clk]
set_ideal_network [get_ports clk]

set_max_fanout 20 [get_ports reset]
set_false_path -from [get_ports reset]

compile -no_map
exit
