`include "VX_define.vh"

module VX_back_end
	#(
		parameter CORE_ID = 0
	)
	(
	input wire clk, 
	input wire reset, 
	input wire schedule_delay,

	VX_gpu_dcache_rsp_inter  vx_dcache_rsp,
	VX_gpu_dcache_req_inter  vx_dcache_req,

	output wire               out_mem_delay,
	output wire               out_exec_delay,
	output wire               gpr_stage_delay,
	VX_jal_response_inter     vx_jal_rsp,
	VX_branch_response_inter  vx_branch_rsp,

	VX_frE_to_bckE_req_inter  vx_bckE_req,
	VX_wb_inter               vx_writeback_inter,

	VX_warp_ctl_inter         vx_warp_ctl
);


VX_wb_inter             vx_writeback_temp();
assign vx_writeback_inter.wb           = vx_writeback_temp.wb;
assign vx_writeback_inter.rd           = vx_writeback_temp.rd;
assign vx_writeback_inter.write_data   = vx_writeback_temp.write_data;
assign vx_writeback_inter.wb_valid     = vx_writeback_temp.wb_valid;
assign vx_writeback_inter.wb_warp_num  = vx_writeback_temp.wb_warp_num;
assign vx_writeback_inter.wb_pc        = vx_writeback_temp.wb_pc;

// assign VX_writeback_inter(vx_writeback_temp);

wire                     no_slot_mem;
wire                     no_slot_exec;

// LSU input + output
VX_lsu_req_inter         vx_lsu_req();
VX_inst_mem_wb_inter     vx_mem_wb();

// Exec unit input + output
VX_exec_unit_req_inter   vx_exec_unit_req();
VX_inst_exec_wb_inter    vx_inst_exec_wb();

// GPU unit input
VX_gpu_inst_req_inter    vx_gpu_inst_req();

// CSR unit inputs
VX_csr_req_inter         vx_csr_req();
VX_csr_wb_inter          vx_csr_wb();
wire                     no_slot_csr;
wire                     stall_gpr_csr;

VX_gpr_stage vx_gpr_stage(
	.clk               (clk),
	.reset             (reset),
	.schedule_delay    (schedule_delay),
	.vx_writeback_inter(vx_writeback_temp),
	.vx_bckE_req       (vx_bckE_req),
	// New
	.vx_exec_unit_req(vx_exec_unit_req),
	.vx_lsu_req      (vx_lsu_req),
	.vx_gpu_inst_req (vx_gpu_inst_req),
	.vx_csr_req      (vx_csr_req),
	.stall_gpr_csr   (stall_gpr_csr),
	// End new
	.memory_delay      (out_mem_delay),
	.exec_delay        (out_exec_delay),
	.gpr_stage_delay   (gpr_stage_delay)
	);

VX_lsu load_store_unit (
	.clk          (clk),
	.reset        (reset),
	.vx_lsu_req   (vx_lsu_req),
	.vx_mem_wb    (vx_mem_wb),
	.vx_dcache_rsp(vx_dcache_rsp),
	.vx_dcache_req(vx_dcache_req),
	.out_delay    (out_mem_delay),
	.no_slot_mem  (no_slot_mem)
);

VX_execute_unit vx_execUnit (
	.clk             (clk),
	.reset           (reset),
	.vx_exec_unit_req(vx_exec_unit_req),
	.vx_inst_exec_wb (vx_inst_exec_wb),
	.vx_jal_rsp      (vx_jal_rsp),
	.vx_branch_rsp   (vx_branch_rsp),
	.out_delay       (out_exec_delay),
	.no_slot_exec    (no_slot_exec)
);

VX_gpgpu_inst vx_gpgpu_inst (
	.vx_gpu_inst_req(vx_gpu_inst_req),
	.vx_warp_ctl    (vx_warp_ctl)
);

// VX_csr_wrapper vx_csr_wrapper(
// 	.vx_csr_req(vx_csr_req),
// 	.vx_csr_wb (vx_csr_wb)
// 	);

VX_csr_pipe #(
	.CORE_ID(CORE_ID)
) vx_csr_pipe (
	.clk         (clk),
	.reset       (reset),
	.no_slot_csr (no_slot_csr),
	.vx_csr_req  (vx_csr_req),
	.vx_writeback(vx_writeback_temp),
	.vx_csr_wb   (vx_csr_wb),
	.stall_gpr_csr(stall_gpr_csr)
);

VX_writeback vx_wb (
	.clk               (clk),
	.reset             (reset),
	.vx_mem_wb         (vx_mem_wb),
	.vx_inst_exec_wb   (vx_inst_exec_wb),
	.vx_csr_wb         (vx_csr_wb),

	.vx_writeback_inter(vx_writeback_temp),
	.no_slot_mem       (no_slot_mem),
	.no_slot_exec      (no_slot_exec),
	.no_slot_csr       (no_slot_csr)
);

endmodule