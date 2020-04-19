`include "VX_define.vh"

module VX_icache_stage (
	input  wire              clk,
	input  wire              reset,
	input  wire              total_freeze,
	output wire              icache_stage_delay,
	output wire[`NW_BITS-1:0] icache_stage_wid,
	output wire[`NUM_THREADS-1:0] icache_stage_valids,
	VX_inst_meta_inter       fe_inst_meta_fi,
	VX_inst_meta_inter       fe_inst_meta_id,

	VX_gpu_dcache_rsp_inter  vx_icache_rsp,
	VX_gpu_dcache_req_inter  vx_icache_req
);

	reg[`NUM_THREADS-1:0] threads_active[`NUM_WARPS-1:0];

	wire   valid_inst = (|fe_inst_meta_fi.valid);

	// Icache Request
	assign vx_icache_req.core_req_valid      = valid_inst && !total_freeze;
	assign vx_icache_req.core_req_addr       = fe_inst_meta_fi.inst_pc;
	assign vx_icache_req.core_req_writedata  = 32'b0;
	assign vx_icache_req.core_req_mem_read   = `LW_MEM_READ;
	assign vx_icache_req.core_req_mem_write  = `NO_MEM_WRITE;
	assign vx_icache_req.core_req_rd         = 5'b0;
	assign vx_icache_req.core_req_wb         = {1{2'b1}};
	assign vx_icache_req.core_req_warp_num   = fe_inst_meta_fi.warp_num;
	assign vx_icache_req.core_req_pc         = fe_inst_meta_fi.inst_pc;

	assign fe_inst_meta_id.instruction = vx_icache_rsp.core_wb_readdata[0][31:0];
	assign fe_inst_meta_id.inst_pc     = vx_icache_rsp.core_wb_pc[0];
	assign fe_inst_meta_id.warp_num    = vx_icache_rsp.core_wb_warp_num;
	
	assign fe_inst_meta_id.valid       = vx_icache_rsp.core_wb_valid ? threads_active[vx_icache_rsp.core_wb_warp_num] : 0;

	assign icache_stage_wid            = fe_inst_meta_id.warp_num;
	assign icache_stage_valids         = fe_inst_meta_id.valid & {`NUM_THREADS{!icache_stage_delay}};

	// Cache can't accept request
	assign icache_stage_delay = vx_icache_rsp.delay_req;

	// Core can't accept response
	assign vx_icache_req.core_no_wb_slot = total_freeze;

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