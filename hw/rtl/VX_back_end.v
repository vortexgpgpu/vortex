`include "VX_define.v"

module VX_back_end	#(
	parameter CORE_ID = 0
) (
	input wire clk, 
	input wire reset, 
	input wire schedule_delay,

	VX_gpu_dcache_rsp_if   dcache_rsp_if,
	VX_gpu_dcache_req_if   dcache_req_if,

	output wire            mem_delay_o,
	output wire            exec_delay_o,
	output wire            gpr_stage_delay,
	VX_jal_response_if     jal_rsp_if,
	VX_branch_response_if  branch_rsp_if,

	VX_frE_to_bckE_req_if  bckE_req_if,
	VX_wb_if               writeback_if,

	VX_warp_ctl_if         warp_ctl_if
);

VX_wb_if writeback_temp_if();
assign writeback_if.wb           = writeback_temp_if.wb;
assign writeback_if.rd           = writeback_temp_if.rd;
assign writeback_if.write_data   = writeback_temp_if.write_data;
assign writeback_if.wb_valid     = writeback_temp_if.wb_valid;
assign writeback_if.wb_warp_num  = writeback_temp_if.wb_warp_num;
assign writeback_if.wb_pc        = writeback_temp_if.wb_pc;

// assign VX_writeback_if(writeback_temp_if);

wire                no_slot_mem;
wire               	no_slot_exec;

// LSU input + output
VX_lsu_req_if       lsu_req_if();
VX_inst_mem_wb_if   mem_wb_if();

// Exec unit input + output
VX_exec_unit_req_if exec_unit_req_if();
VX_inst_exec_wb_if  inst_exec_wb_if();

// GPU unit input
VX_gpu_inst_req_if  gpu_inst_req_if();

// CSR unit inputs
VX_csr_req_if		csr_req_if();
VX_csr_wb_if    	csr_wb_if();
wire            	no_slot_csr;
wire         		stall_gpr_csr;

VX_gpr_stage gpr_stage (
	.clk               	(clk),
	.reset             	(reset),
	.schedule_delay    	(schedule_delay),
	.writeback_if     	(writeback_temp_if),
	.bckE_req_if       	(bckE_req_if),
	// New
	.exec_unit_req_if	(exec_unit_req_if),
	.lsu_req_if      	(lsu_req_if),
	.gpu_inst_req_if 	(gpu_inst_req_if),
	.csr_req_if      	(csr_req_if),
	.stall_gpr_csr   	(stall_gpr_csr),
	// End new
	.memory_delay    	(mem_delay_o),
	.exec_delay      	(exec_delay_o),
	.gpr_stage_delay	(gpr_stage_delay)
);

VX_lsu load_store_unit (
	.clk         	(clk),
	.reset        	(reset),
	.lsu_req_if  	(lsu_req_if),
	.mem_wb_if    	(mem_wb_if),
	.dcache_rsp_if	(dcache_rsp_if),
	.dcache_req_if	(dcache_req_if),
	.delay_o    	(mem_delay_o),
	.no_slot_mem_i	(no_slot_mem)
);

VX_execute_unit execUnit (
	.clk             (clk),
	.reset           (reset),
	.exec_unit_req_if(exec_unit_req_if),
	.inst_exec_wb_if (inst_exec_wb_if),
	.jal_rsp_if      (jal_rsp_if),
	.branch_rsp_if   (branch_rsp_if),
	.delay_o         (exec_delay_o),
	.no_slot_exec_i  (no_slot_exec)
);

VX_gpgpu_inst gpgpu_inst (
	.gpu_inst_req_if(gpu_inst_req_if),
	.warp_ctl_if    (warp_ctl_if)
);

VX_csr_pipe #(
	.CORE_ID(CORE_ID)
) csr_pipe (
	.clk         (clk),
	.reset       (reset),
	.no_slot_csr (no_slot_csr),
	.csr_req_if  (csr_req_if),
	.writeback_if(writeback_temp_if),
	.csr_wb_if   (csr_wb_if),
	.stall_gpr_csr(stall_gpr_csr)
);

VX_writeback wb (
	.clk               (clk),
	.reset             (reset),
	.mem_wb_if         (mem_wb_if),
	.inst_exec_wb_if   (inst_exec_wb_if),
	.csr_wb_if         (csr_wb_if),

	.writeback_if	   (writeback_temp_if),
	.no_slot_mem_o     (no_slot_mem),
	.no_slot_exec_o    (no_slot_exec),
	.no_slot_csr_o     (no_slot_csr)
);

endmodule