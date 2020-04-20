`include "VX_define.vh"

module VX_icache_stage (
	input  wire             clk,
	input  wire             reset,
	input  wire             total_freeze,
	output wire             icache_stage_delay,
	output wire[`NW_BITS-1:0] icache_stage_wid,
	output wire[`NUM_THREADS-1:0] icache_stage_valids,
	VX_inst_meta_if       	fe_inst_meta_fi,
	VX_inst_meta_if       	fe_inst_meta_id,

	VX_gpu_dcache_rsp_if  	icache_rsp_if,
	VX_gpu_dcache_req_if  	icache_req_if
);

	reg[`NUM_THREADS-1:0] threads_active[`NUM_WARPS-1:0];

	wire   valid_inst = (|fe_inst_meta_fi.valid);

	// Icache Request
	assign icache_req_if.core_req_valid      = valid_inst && !total_freeze;
	assign icache_req_if.core_req_addr       = fe_inst_meta_fi.inst_pc;
	assign icache_req_if.core_req_writedata  = 32'b0;
	assign icache_req_if.core_req_mem_read   = `LW_MEM_READ;
	assign icache_req_if.core_req_mem_write  = `NO_MEM_WRITE;
	assign icache_req_if.core_req_rd         = 5'b0;
	assign icache_req_if.core_req_wb         = {1{2'b1}};
	assign icache_req_if.core_req_warp_num   = fe_inst_meta_fi.warp_num;
	assign icache_req_if.core_req_pc         = fe_inst_meta_fi.inst_pc;

	assign fe_inst_meta_id.instruction = icache_rsp_if.core_rsp_readdata[0][31:0];
	assign fe_inst_meta_id.inst_pc     = icache_rsp_if.core_rsp_pc[0];
	assign fe_inst_meta_id.warp_num    = icache_rsp_if.core_rsp_warp_num;
	
	assign fe_inst_meta_id.valid       = icache_rsp_if.core_rsp_valid ? threads_active[icache_rsp_if.core_rsp_warp_num] : 0;

	assign icache_stage_wid            = fe_inst_meta_id.warp_num;
	assign icache_stage_valids         = fe_inst_meta_id.valid & {`NUM_THREADS{!icache_stage_delay}};

	// Cache can't accept request
	assign icache_stage_delay = ~icache_req_if.core_req_ready;

	// Core can't accept response
	assign icache_rsp_if.core_rsp_ready = ~total_freeze;

	integer curr_w;
	always @(posedge clk) begin
		if (reset) begin
			for (curr_w = 0; curr_w < `NUM_WARPS; curr_w=curr_w+1) begin
				threads_active[curr_w] <= 0;
			end
		end else begin
			if (valid_inst && !icache_stage_delay) begin
				threads_active[fe_inst_meta_fi.warp_num] <= fe_inst_meta_fi.valid;
			end
		end
	end

endmodule