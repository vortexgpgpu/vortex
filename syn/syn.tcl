set search_path			[concat /nethome/dshim8/Desktop/GTCAD-3DPKG-v3/example/tech/cln28hpm/2d_db/ /nethome/dshim8/Desktop/GTCAD-3DPKG-v3/example/tech/cln28hpm/2d_hard_db/ ../rtl/ ../rtl/interfaces ../rtl/pipe_regs]
set link_library		[concat * sc12mc_cln28hpm_base_ulvt_c35_ssg_typical_max_0p81v_m40c.db rf2_32x128_wm1_ss_0p81v_0p81v_125c.db dw_foundation.sldb]
set symbol_library		{}
set target_library		[concat sc12mc_cln28hpm_base_ulvt_c35_ssg_typical_max_0p81v_m40c.db]

set verilog_files 	[ list VX_inst_exec_wb_inter.v VX_lsu.v VX_execute_unit.v VX_lsu_addr_gen.v VX_inst_multiplex.v VX_exec_unit_req_inter.v VX_lsu_req_inter.v VX_alu.v VX_back_end.v VX_gpr_stage.v VX_gpr_data_inter.v VX_csr_handler.v VX_decode.v VX_define.v VX_scheduler.v VX_fetch.v VX_front_end.v VX_generic_register.v VX_gpr.v VX_gpr_wrapper.v VX_one_counter.v VX_priority_encoder.v VX_warp.v VX_warp_scheduler.v VX_writeback.v Vortex.v byte_enabled_simple_dual_port_ram.v VX_branch_response_inter.v VX_csr_write_request_inter.v VX_dcache_request_inter.v VX_dcache_response_inter.v VX_frE_to_bckE_req_inter.v VX_gpr_clone_inter.v VX_gpr_jal_inter.v VX_gpr_read_inter.v VX_gpr_wspawn_inter.v VX_icache_request_inter.v VX_icache_response_inter.v VX_inst_mem_wb_inter.v VX_inst_meta_inter.v VX_jal_response_inter.v VX_mem_req_inter.v VX_mw_wb_inter.v VX_warp_ctl_inter.v VX_wb_inter.v VX_d_e_reg.v VX_f_d_reg.v \
					]

analyze -format sverilog $verilog_files
elaborate Vortex
link

set clk_freq 100
set clk_period [expr 1000.0 / $clk_freq / 1.0]
create_clock [get_ports clk] -period $clk_period
set_max_fanout 20 [get_ports clk]
set_ideal_network [get_ports clk]

set_max_fanout 20 [get_ports reset]
set_false_path -from [get_ports reset]

compile_ultra -no_autoungroup
ungroup -all -flatten
uniquify

define_name_rules verilog -remove_internal_net_bus -remove_port_bus
change_names -rule verilog -hierarchy

report_qor 
report_area
report_hierarchy
report_cell
report_reference
report_port

write -hierarchy -format verilog -output Vortex.netlist.v
remove_ideal_network [get_ports clk]
set_propagated_clock [get_ports clk]
write_sdc -version 1.9 Vortex.sdc
exit
