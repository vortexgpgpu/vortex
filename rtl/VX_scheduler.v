

`include "VX_define.v"

module VX_scheduler (
	input wire                clk,
	input wire                reset,
	input wire                memory_delay,
	input wire 				  exec_delay,
	input wire                gpr_stage_delay,
	VX_frE_to_bckE_req_inter  VX_bckE_req,
	VX_wb_inter               VX_writeback_inter,

	output wire schedule_delay
	
);



	reg[31:0][`NT-1:0] rename_table[`NW-1:0];

	wire valid_wb  = (VX_writeback_inter.wb != 0) && (|VX_writeback_inter.wb_valid) && (VX_writeback_inter.rd != 0);
	wire wb_inc    = (VX_bckE_req.wb != 0) && (VX_bckE_req.rd != 0);

	wire rs1_rename = rename_table[VX_bckE_req.warp_num][VX_bckE_req.rs1] != 0;
	wire rs2_rename = rename_table[VX_bckE_req.warp_num][VX_bckE_req.rs2] != 0;
	wire rd_rename  = rename_table[VX_bckE_req.warp_num][VX_bckE_req.rd ] != 0;

	wire is_store = (VX_bckE_req.mem_write != `NO_MEM_WRITE);
	wire is_load  = (VX_bckE_req.mem_read  != `NO_MEM_READ);

	// classify our next instruction.
	wire is_mem   = is_store || is_load;
	wire is_gpu = (VX_bckE_req.is_wspawn || VX_bckE_req.is_tmc || VX_bckE_req.is_barrier || VX_bckE_req.is_split);
	wire is_csr = VX_bckE_req.is_csr;
	wire is_exec = !is_mem && !is_gpu && !is_csr;



	// wire rs1_pass        = 0;
	// wire rs2_pass        = 0;

	wire using_rs2       = (VX_bckE_req.rs2_src == `RS2_REG) || is_store || VX_bckE_req.is_barrier || VX_bckE_req.is_wspawn;

	wire rs1_rename_qual = ((rs1_rename) && (VX_bckE_req.rs1 != 0));
	wire rs2_rename_qual = ((rs2_rename) && (VX_bckE_req.rs2 != 0 && using_rs2));
	wire  rd_rename_qual = ((rd_rename ) && (VX_bckE_req.rd  != 0));


	wire rename_valid = rs1_rename_qual || rs2_rename_qual || rd_rename_qual;

	assign schedule_delay = ((rename_valid) && (|VX_bckE_req.valid))
		|| (memory_delay && is_mem)
		|| (gpr_stage_delay && (is_mem || is_exec))
		|| (exec_delay && is_exec);

	integer i;
	integer w;
	always @(posedge clk or posedge reset) begin

		if (reset) begin
			for (w = 0; w < `NW; w=w+1)
			begin
				for (i = 0; i < 32; i = i + 1)
				begin
				 rename_table[w][i] <= 0;
				end
			end
		end else begin
			if (valid_wb                 ) rename_table[VX_writeback_inter.wb_warp_num][VX_writeback_inter.rd] <= rename_table[VX_writeback_inter.wb_warp_num][VX_writeback_inter.rd] & (~VX_writeback_inter.wb_valid);
			if (!schedule_delay && wb_inc) rename_table[VX_bckE_req.warp_num          ][VX_bckE_req.rd       ] <= VX_bckE_req.valid;
		end
	end


endmodule