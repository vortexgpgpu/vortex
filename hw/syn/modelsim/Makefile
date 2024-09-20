include ../common.mk

ALL:sim

#TOOL INPUT
SRC = \
  vortex_dpi.cpp  \
  vortex_tb.v \
../rtl/VX_user_config.vh \
../rtl/VX_config.vh \
../rtl/VX_define.vh \
../rtl/interfaces/VX_branch_response_inter.v \
../rtl/interfaces/VX_csr_req_inter.v \
../rtl/interfaces/VX_csr_wb_inter.v \
../rtl/interfaces/VX_dcache_request_inter.v \
../rtl/interfaces/VX_dcache_response_inter.v \
../rtl/interfaces/VX_dram_req_rsp_inter.v \
../rtl/interfaces/VX_exec_unit_req_inter.v \
../rtl/interfaces/VX_frE_to_bckE_req_inter.v \
../rtl/interfaces/VX_gpr_data_inter.v \
../rtl/interfaces/VX_gpr_jal_inter.v \
../rtl/interfaces/VX_gpr_read_inter.v \
../rtl/interfaces/VX_gpu_inst_req_inter.v \
../rtl/interfaces/VX_icache_request_inter.v \
../rtl/interfaces/VX_icache_response_inter.v \
../rtl/interfaces/VX_inst_exec_wb_inter.v \
../rtl/interfaces/VX_inst_mem_wb_inter.v \
../rtl/interfaces/VX_inst_meta_inter.v \
../rtl/interfaces/VX_jal_response_inter.v \
../rtl/interfaces/VX_join_inter.v \
../rtl/interfaces/VX_lsu_req_inter.v \
../rtl/interfaces/VX_mem_req_inter.v \
../rtl/interfaces/VX_mw_wb_inter.v \
../rtl/interfaces/VX_warp_ctl_inter.v \
../rtl/interfaces/VX_wb_inter.v \
../rtl/interfaces/VX_wstall_inter.v \
../rtl/VX_alu.v \
../rtl/VX_back_end.v \
../rtl/VX_csr_handler.v \
../rtl/VX_csr_wrapper.v \
../rtl/VX_decode.v \
../rtl/VX_dmem_controller.v \
../rtl/VX_execute_unit.v \
../rtl/VX_fetch.v \
../rtl/VX_front_end.v \
../rtl/VX_generic_priority_encoder.v \
../rtl/VX_generic_register.v \
../rtl/VX_generic_stack.v \
../rtl/VX_gpgpu_inst.v \
../rtl/VX_gpr.v \
../rtl/VX_gpr_stage.v \
../rtl/VX_gpr_wrapper.v \
../rtl/VX_inst_multiplex.v \
../rtl/VX_lsu.v \
../rtl/VX_lsu_addr_gen.v \
../rtl/VX_priority_encoder.v \
../rtl/VX_priority_encoder_w_mask.v \
../rtl/VX_scheduler.v \
../rtl/VX_warp.v \
../rtl/VX_countones.v \
../rtl/VX_warp_scheduler.v \
../rtl/VX_writeback.v \
../rtl/Vortex.v \
../rtl/byte_enabled_simple_dual_port_ram.v \
../rtl/cache/VX_Cache_Bank.v \
../rtl/cache/VX_cache_bank_valid.v \
../rtl/cache/VX_cache_data.v \
../rtl/cache/VX_d_cache.v \
../rtl/cache/VX_generic_pe.v \
../rtl/cache/cache_set.v \
../rtl/cache/VX_cache_data_per_index.v \
../rtl/VX_d_e_reg.v \
../rtl/VX_f_d_reg.v \
../models/memory/cln28hpm/rf2_128x128_wm1/rf2_128x128_wm1.v \
../models/memory/cln28hpm/rf2_256x128_wm1/rf2_256x128_wm1.v \
../models/memory/cln28hpm/rf2_256x19_wm0/rf2_256x19_wm0.v \
../models/memory/cln28hpm/rf2_32x128_wm1/rf2_32x128_wm1.v \
../models/memory/cln28hpm/rf2_32x19_wm0/rf2_32x19_wm0.v 

# ../models/memory/cln28hpc/rf2_32x128_wm1/rf2_32x128_wm1.v

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

lib:
	vlib vortex_lib

comp:
	vlog $(OPT) -work $(LIB) $(SRC)
	# vlog -O0 -dpiheader vortex_dpi.h $(OPT) -work $(LIB) $(SRC)


sim: comp
	# vsim vortex_tb $(LOG) -c -lib $(LIB) $(CMD) > vortex_sim.log
	vsim -novopt vortex_tb $(LOG) -c -lib $(LIB) $(CMD) > vortex_sim.log





