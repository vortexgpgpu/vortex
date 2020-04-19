`include "VX_define.vh"

module VX_scheduler (
	input wire                clk,
	input wire                reset,
	input wire                memory_delay,
	input wire 				  exec_delay,
	input wire                gpr_stage_delay,
	VX_frE_to_bckE_req_if  vx_bckE_req,
	VX_wb_if               vx_writeback_if,

	output wire schedule_delay,
	output wire is_empty	
);
	reg[31:0] count_valid;

	assign is_empty = count_valid == 0;

	reg[31:0][`NUM_THREADS-1:0] rename_table[`NUM_WARPS-1:0];

	wire valid_wb  = (vx_writeback_if.wb != 0) && (|vx_writeback_if.wb_valid) && (vx_writeback_if.rd != 0);
	wire wb_inc    = (vx_bckE_req.wb != 0) && (vx_bckE_req.rd != 0);

	wire rs1_rename = rename_table[vx_bckE_req.warp_num][vx_bckE_req.rs1] != 0;
	wire rs2_rename = rename_table[vx_bckE_req.warp_num][vx_bckE_req.rs2] != 0;
	wire rd_rename  = rename_table[vx_bckE_req.warp_num][vx_bckE_req.rd ] != 0;

	wire is_store = (vx_bckE_req.mem_write != `NO_MEM_WRITE);
	wire is_load  = (vx_bckE_req.mem_read  != `NO_MEM_READ);

	// classify our next instruction.
	wire is_mem  = is_store || is_load;
	wire is_gpu  = (vx_bckE_req.is_wspawn || vx_bckE_req.is_tmc || vx_bckE_req.is_barrier || vx_bckE_req.is_split);
	wire is_csr  = vx_bckE_req.is_csr;
	wire is_exec = !is_mem && !is_gpu && !is_csr;

	wire using_rs2       = (vx_bckE_req.rs2_src == `RS2_REG) || is_store || vx_bckE_req.is_barrier || vx_bckE_req.is_wspawn;

	wire rs1_rename_qual = ((rs1_rename) && (vx_bckE_req.rs1 != 0));
	wire rs2_rename_qual = ((rs2_rename) && (vx_bckE_req.rs2 != 0 && using_rs2));
	wire  rd_rename_qual = ((rd_rename ) && (vx_bckE_req.rd  != 0));

	wire rename_valid = rs1_rename_qual || rs2_rename_qual || rd_rename_qual;

	assign schedule_delay = ((rename_valid) && (|vx_bckE_req.valid))
				         || (memory_delay && is_mem)
					     || (gpr_stage_delay && (is_mem || is_exec))
						 || (exec_delay && is_exec);

	integer i;
	integer w;
	always @(posedge clk) begin

		if (reset) begin
			for (w = 0; w < `NUM_WARPS; w=w+1) begin
				for (i = 0; i < 32; i = i + 1) begin
				 	rename_table[w][i] <= 0;
				end
			end
		end else begin
			if (valid_wb) begin
				rename_table[vx_writeback_if.wb_warp_num][vx_writeback_if.rd] <= rename_table[vx_writeback_if.wb_warp_num][vx_writeback_if.rd] & (~vx_writeback_if.wb_valid);
			end

			if (!schedule_delay && wb_inc) begin
				rename_table[vx_bckE_req.warp_num][vx_bckE_req.rd] <= vx_bckE_req.valid;
			end
		
			if (valid_wb 
			 && (0 == (rename_table[vx_writeback_if.wb_warp_num][vx_writeback_if.rd] & ~vx_writeback_if.wb_valid))) begin
			   	count_valid <= count_valid - 1;
			end

			if (!schedule_delay && wb_inc) begin
				count_valid <= count_valid + 1;
			end
		end
	end

endmodule