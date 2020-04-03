onerror {resume}
quietly WaveActivateNextPane {} 0
add wave -noupdate -label clk /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/clk
add wave -noupdate -label reset /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/SoftReset
add wave -noupdate -label state /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/state
add wave -noupdate -label cci_write_pending /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/cci_write_pending
add wave -noupdate -label cci_write_ctr -radix decimal -radixshowbase 0 /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/cci_write_ctr
add wave -noupdate -label csr_data_size -radix decimal -radixshowbase 0 /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/csr_data_size
add wave -noupdate -label avs_read_ctr -radix decimal -radixshowbase 0 /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/avs_read_ctr
add wave -noupdate -label avs_waitrequest /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/avs_waitrequest
add wave -noupdate -label avs_address -radix hexadecimal -radixshowbase 0 /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/avs_address
add wave -noupdate -label avs_readdata -radix hexadecimal /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/avs_readdata
add wave -noupdate -label avs_writedata -radix hexadecimal /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/avs_writedata
add wave -noupdate -label avs_write /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/avs_write
add wave -noupdate -label avs_read /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/avs_read
add wave -noupdate -label avs_readdatavalid /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/avs_readdatavalid
add wave -noupdate -label sRx.c0.rspValid /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/cp2af_sRxPort.c0.rspValid
add wave -noupdate -label sRx.c1.rspValid /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/cp2af_sRxPort.c1.rspValid
add wave -noupdate -label sTx.c0.valid /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/af2cp_sTxPort.c0.valid
add wave -noupdate -label sTx.c1.valid /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/af2cp_sTxPort.c1.valid
add wave -noupdate -label cci_write_req /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/cci_write_req
add wave -noupdate -label avs_raq_push /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/avs_raq_push
add wave -noupdate -label avs_rdq_push /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/avs_rdq_push
add wave -noupdate -label avs_raq_pop /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/avs_raq_pop
add wave -noupdate -label avs_rdq_pop /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/avs_rdq_pop
add wave -noupdate -label avs_raq_full /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/avs_raq_full
add wave -noupdate -label avs_rdq_full /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/avs_rdq_full
add wave -noupdate -label avs_raq_empty /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/avs_raq_empty
add wave -noupdate -label avs_rdq_empty /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/avs_rdq_empty
add wave -noupdate -label vx_dram_req_write /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/vx_dram_req_write
add wave -noupdate -label vx_dram_req_delay /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/vx_dram_req_delay
add wave -noupdate -label vx_dram_req_read /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/vx_dram_req_read
add wave -noupdate -label vx_reset /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/vx_soc/reset
add wave -noupdate -label out_dram_fill_rsp /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/vx_soc/out_dram_fill_rsp
add wave -noupdate -label out_dram_fill_accept /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/vx_soc/out_dram_fill_accept
add wave -noupdate -label llc_snp_req /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/vx_soc/llc_snp_req
add wave -noupdate -label llc_snp_req_delay /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/vx_soc/llc_snp_req_delay
add wave -noupdate -label out_break /ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/vx_soc/out_ebreak
add wave -noupdate -label warp_pc -radix hexadecimal -radixshowbase 0 {/ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/vx_soc/genblk1/Vortex_Cluster/genblk1[0]/vortex_core/vx_front_end/vx_fetch/warp_pc}
add wave -noupdate -label scheduled_warp {/ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/vx_soc/genblk1/Vortex_Cluster/genblk1[0]/vortex_core/vx_front_end/vx_fetch/scheduled_warp}
add wave -noupdate -label thread_mask {/ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/vx_soc/genblk1/Vortex_Cluster/genblk1[0]/vortex_core/vx_front_end/vx_fetch/thread_mask}
add wave -noupdate -label warp_num {/ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/vx_soc/genblk1/Vortex_Cluster/genblk1[0]/vortex_core/vx_front_end/vx_fetch/warp_num}
add wave -noupdate -label warp_active {/ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/vx_soc/genblk1/Vortex_Cluster/genblk1[0]/vortex_core/vx_front_end/vx_fetch/warp_scheduler/warp_active}
add wave -noupdate -label warp_stalled {/ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/vx_soc/genblk1/Vortex_Cluster/genblk1[0]/vortex_core/vx_front_end/vx_fetch/warp_scheduler/warp_stalled}
add wave -noupdate -label warp_lock {/ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/vx_soc/genblk1/Vortex_Cluster/genblk1[0]/vortex_core/vx_front_end/vx_fetch/warp_scheduler/warp_lock}
add wave -noupdate -label use_active {/ase_top/ase_top_generic/platform_shim_ccip_std_afu/ccip_std_afu/vortex_afu_inst/vx_soc/genblk1/Vortex_Cluster/genblk1[0]/vortex_core/vx_front_end/vx_fetch/warp_scheduler/use_active}
TreeUpdate [SetDefaultTree]
WaveRestoreCursors {{Cursor 2} {66234495 ps} 0}
quietly wave cursor active 1
configure wave -namecolwidth 195
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 0
configure wave -snapdistance 10
configure wave -datasetprefix 0
configure wave -rowmargin 4
configure wave -childrowmargin 2
configure wave -gridoffset 0
configure wave -gridperiod 1
configure wave -griddelta 40
configure wave -timeline 0
configure wave -timelineunits ps
update
WaveRestoreZoom {66041656 ps} {66406344 ps}
