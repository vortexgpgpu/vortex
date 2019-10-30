

ALL:sim

#TOOL INPUT
SRC = \
  vortex_dpi.cpp  \
  vortex_tb.v \
../VX_define.v \
../interfaces/VX_branch_response_inter.v \
../interfaces/VX_csr_req_inter.v \
../interfaces/VX_csr_wb_inter.v \
../interfaces/VX_dcache_request_inter.v \
../interfaces/VX_dcache_response_inter.v \
../interfaces/VX_dram_req_rsp_inter.v \
../interfaces/VX_exec_unit_req_inter.v \
../interfaces/VX_frE_to_bckE_req_inter.v \
../interfaces/VX_gpr_clone_inter.v \
../interfaces/VX_gpr_data_inter.v \
../interfaces/VX_gpr_jal_inter.v \
../interfaces/VX_gpr_read_inter.v \
../interfaces/VX_gpr_wspawn_inter.v \
../interfaces/VX_gpu_inst_req_inter.v \
../interfaces/VX_icache_request_inter.v \
../interfaces/VX_icache_response_inter.v \
../interfaces/VX_inst_exec_wb_inter.v \
../interfaces/VX_inst_mem_wb_inter.v \
../interfaces/VX_inst_meta_inter.v \
../interfaces/VX_jal_response_inter.v \
../interfaces/VX_join_inter.v \
../interfaces/VX_lsu_req_inter.v \
../interfaces/VX_mem_req_inter.v \
../interfaces/VX_mw_wb_inter.v \
../interfaces/VX_warp_ctl_inter.v \
../interfaces/VX_wb_inter.v \
../interfaces/VX_wstall_inter.v \
../VX_alu.v \
../VX_back_end.v \
../VX_csr_handler.v \
../VX_csr_wrapper.v \
../VX_decode.v \
../VX_dmem_controller.v \
../VX_execute_unit.v \
../VX_fetch.v \
../VX_front_end.v \
../VX_generic_priority_encoder.v \
../VX_generic_register.v \
../VX_generic_stack.v \
../VX_gpgpu_inst.v \
../VX_gpr.v \
../VX_gpr_stage.v \
../VX_gpr_wrapper.v \
../VX_inst_multiplex.v \
../VX_lsu.v \
../VX_lsu_addr_gen.v \
../VX_priority_encoder.v \
../VX_priority_encoder_w_mask.v \
../VX_scheduler.v \
../VX_warp.v \
../VX_countones.v \
../VX_warp_scheduler.v \
../VX_writeback.v \
../Vortex.v \
../byte_enabled_simple_dual_port_ram.v \
../cache/VX_Cache_Bank.v \
../cache/VX_cache_bank_valid.v \
../cache/VX_cache_data.v \
../cache/VX_d_cache.v \
../cache/VX_generic_pe.v \
../cache/cache_set.v \
../cache/VX_cache_data_per_index.v \
../pipe_regs/VX_d_e_reg.v \
../pipe_regs/VX_f_d_reg.v \
../shared_memory/VX_bank_valids.v \
../shared_memory/VX_priority_encoder_sm.v \
../shared_memory/VX_shared_memory.v \
../shared_memory/VX_shared_memory_block.v \
../../models/memory/cln28hpm/rf2_128x128_wm1/rf2_128x128_wm1.v \
../../models/memory/cln28hpm/rf2_256x128_wm1/rf2_256x128_wm1.v \
../../models/memory/cln28hpm/rf2_256x19_wm0/rf2_256x19_wm0.v \
../../models/memory/cln28hpm/rf2_32x128_wm1/rf2_32x128_wm1.v 
# ../../models/memory/cln28hpc/rf2_32x128_wm1/rf2_32x128_wm1.v

# vortex_dpi.h 


CMD= \
-do "VoptFlow = 0; \
	 vcd file vortex.vcd; \
	 vcd add -r /vortex_tb/*; \
	 vcd add -r /vortex/*; \
	 run -all; \
	 quit -f"


OPT=-sv -sv12compat

LIB = vortex_lib

# LOG=-logfile vortex_tb.log
LOG=

# setup: 	source cshrc.modelsim
#           vlib 

comp:
	vlog $(OPT) -work $(LIB) $(SRC)
	# vlog -O0 -dpiheader vortex_dpi.h $(OPT) -work $(LIB) $(SRC)


sim: comp
	vsim vortex_tb $(LOG) -c -lib $(LIB) $(CMD) > vortex_sim.log
	# vsim -novopt vortex_tb $(LOG) -c -lib $(LIB) $(CMD) > vortex_sim.log





