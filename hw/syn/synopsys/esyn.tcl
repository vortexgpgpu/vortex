#set search_path		[concat /nethome/dshim8/Desktop/GTCAD-3DPKG-v3/example/tech/cln28hpm/2d_db/ /nethome/dshim8/Desktop/GTCAD-3DPKG-v3/example/tech/cln28hpm/2d_hard_db/ ../../rtl/ ../../rtl/interfaces ../../rtl/pipe_regs ../../rtl/shared_memory ../../rtl/cache ../../models/memory/cln28hpm/2d_hardmacro_db]
set search_path			[concat ../../rtl/ ../../rtl/interfaces ../../rtl/pipe_regs ../../rtl/shared_memory ../../rtl/cache ../../models/memory/cln28hpm/2d_hardmacro_db]
set link_library		[concat ./NanGate_15nm_OCL.db]
set symbol_library		{}
set target_library		[concat ./NanGate_15nm_OCL.db]

set verilog_files 	[ list VX_countones.v VX_priority_encoder_w_mask.v VX_dram_req_rsp_inter.v VX_cache_data_per_index.v VX_Cache_Bank.v VX_cache_data.v VX_d_cache.v VX_bank_valids.v VX_priority_encoder_sm.v VX_shared_memory.v VX_shared_memory_block.v VX_dmem_controller.v VX_generic_priority_encoder.v VX_generic_stack.v VX_join_inter.v VX_csr_wrapper.v VX_csr_req_inter.v VX_csr_wb_inter.v  VX_gpgpu_inst.v VX_gpu_inst_req_inter.v VX_wstall_inter.v VX_inst_exec_wb_inter.v VX_lsu.v VX_execute_unit.v VX_lsu_addr_gen.v VX_inst_multiplex.v VX_exec_unit_req_inter.v VX_lsu_req_inter.v VX_alu.v VX_back_end.v VX_gpr_stage.v VX_gpr_data_inter.v VX_csr_handler.v VX_decode.v VX_define.vh VX_config.vh VX_user_config.vh VX_scheduler.v VX_fetch.v VX_front_end.v VX_generic_register.v VX_gpr.v VX_gpr_wrapper.v VX_priority_encoder.v VX_warp_scheduler.v VX_writeback.v byte_enabled_simple_dual_port_ram.v VX_branch_response_inter.v VX_dcache_request_inter.v VX_dcache_response_inter.v VX_frE_to_bckE_req_inter.v VX_gpr_jal_inter.v VX_gpr_read_inter.v VX_icache_request_inter.v VX_icache_response_inter.v VX_inst_mem_wb_inter.v VX_inst_meta_inter.v VX_jal_response_inter.v VX_mem_req_inter.v VX_mw_wb_inter.v VX_warp_ctl_inter.v VX_wb_inter.v VX_d_e_reg.v VX_f_d_reg.v Vortex.v VX_cache_bank_valid.v \
					]
# set verilog_files 	[ list Vortex.v VX_countones.v VX_priority_encoder_w_mask.v VX_dram_req_rsp_inter.v cache_set.v VX_Cache_Bank.v VX_Cache_Block_DM.v VX_cache_data.v VX_d_cache.v VX_generic_pc.v VX_bank_valids.v VX_priority_encoder_sm.v VX_shared_memory.v VX_shared_memory_block.v VX_dmem_controller.v VX_generic_priority_encoder.v VX_generic_stack.v VX_join_inter.v VX_csr_wrapper.v VX_csr_req_inter.v VX_csr_wb_inter.v  VX_gpgpu_inst.v VX_gpu_inst_req_inter.v VX_wstall_inter.v VX_inst_exec_wb_inter.v VX_lsu.v VX_execute_unit.v VX_lsu_addr_gen.v VX_inst_multiplex.v VX_exec_unit_req_inter.v VX_lsu_req_inter.v VX_alu.v VX_back_end.v VX_gpr_stage.v VX_gpr_data_inter.v VX_csr_handler.v VX_decode.v VX_define.vh VX_scheduler.v VX_fetch.v VX_front_end.v VX_generic_register.v VX_gpr.v VX_gpr_wrapper.v VX_one_counter.v VX_priority_encoder.v VX_warp_scheduler.v VX_writeback.v byte_enabled_simple_dual_port_ram.v VX_branch_response_inter.v VX_dcache_request_inter.v VX_dcache_response_inter.v VX_frE_to_bckE_req_inter.v VX_gpr_jal_inter.v VX_gpr_read_inter.v VX_icache_request_inter.v VX_icache_response_inter.v VX_inst_mem_wb_inter.v VX_inst_meta_inter.v VX_jal_response_inter.v VX_mem_req_inter.v VX_mw_wb_inter.v VX_warp_ctl_inter.v VX_wb_inter.v VX_d_e_reg.v VX_f_d_reg.v \
# 					]

set top_level Vortex
analyze -format sverilog $verilog_files
#analyze -format sverilog -error=LINT-66 $verilog_files
elaborate Vortex
link

set clk_freq 0.4
set clk_period [expr 1000.0 / $clk_freq / 1.0]
create_clock [get_ports clk] -period $clk_period
set_max_fanout 20 [get_ports clk]
set_ideal_network [get_ports clk]

set_max_fanout 20 [get_ports reset]
set_false_path -from [get_ports reset]
all_high_fanout -net -threshold 20

# set_register_merging Vortex FALSE
# set compile_seqmap_propagate_constants false
# set compile_seqmap_propagate_high_effort false

check_design
compile_ultra -no_autoungroup
ungroup -all -flatten
uniquify

define_name_rules verilog -remove_internal_net_bus -remove_port_bus
change_names -rule verilog -hierarchy

# report_qor 
report_area
report_hierarchy
report_cell
report_reference
report_port
report_power

write -hierarchy -format verilog -output Vortex.netlist.v
remove_ideal_network [get_ports clk]
set_propagated_clock [get_ports clk]
write_sdc -version 1.9 Vortex.sdc
write_file -format ddc -output Vortex.ddc
exit