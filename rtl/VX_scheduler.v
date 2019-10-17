

`include "VX_define.v"

module VX_scheduler (
	input wire                clk,
	VX_frE_to_bckE_req_inter  VX_bckE_req,
	VX_wb_inter               VX_writeback_inter,

	output wire schedule_delay
	
);



	reg rename_table[31:0];

	initial begin
		integer i;
		for (i = 0; i < 32; i = i + 1) rename_table[i] = 0;
	end

	wire valid_wb  = (VX_writeback_inter.wb != 0) && (|VX_writeback_inter.wb_valid) && (VX_writeback_inter.rd != 0);
	wire wb_inc    = (VX_bckE_req.wb != 0) && (VX_bckE_req.rd != 0);


	// wire pass_through = ((VX_bckE_req.rs1 == VX_writeback_inter.rd) || (VX_bckE_req.rs2 == VX_writeback_inter.rd)) && valid_wb;
	// wire pass_through = 0;

	wire rs1_rename = rename_table[VX_bckE_req.rs1];
	wire rs2_rename = rename_table[VX_bckE_req.rs2];

	wire is_store = (VX_bckE_req.mem_write != `NO_MEM_WRITE);

	wire rs1_rename_qual = (rs1_rename && (VX_bckE_req.rs1 != 0));
	wire rs2_rename_qual = (rs2_rename && (VX_bckE_req.rs2 != 0) && ((VX_bckE_req.rs2_src == `RS2_REG) || is_store));

	wire rename_valid = rs1_rename_qual || rs2_rename_qual ;


	assign schedule_delay = (rename_valid) && (|VX_bckE_req.valid);


	always @(posedge clk) begin
		if (valid_wb                 ) rename_table[VX_writeback_inter.rd] <= 0;
		if (!schedule_delay && wb_inc) rename_table[VX_bckE_req.rd]        <= 1;
	end


endmodule