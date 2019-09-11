
`include "VX_define.v"

module VX_gpr (
	input wire                  clk,
	input wire                  valid_write_request,
	VX_gpr_read_inter           VX_gpr_read,
	VX_wb_inter                 VX_writeback_inter,

	output reg[`NT_M1:0][31:0] out_a_reg_data,
	output reg[`NT_M1:0][31:0] out_b_reg_data
);


	logic[`NT_M1:0][31:0] gpr[31:0]; // gpr[register_number][thread_number][data_bits]

	wire write_enable;

	assign write_enable = valid_write_request && ((VX_writeback_inter.wb != 0) && (VX_writeback_inter.rd != 5'h0));
	// assign read_enable  = valid_request;

		// // Using Registers
		//  integer thread_index;
		//  always_ff@(posedge clk)
		//  begin
		//  	if (write_enable) begin
		//  		for (thread_index = 0; thread_index <= `NT_M1; thread_index = thread_index + 1) begin
		//  			if (VX_writeback_inter.wb_valid[thread_index]) begin
		//  				gpr[VX_writeback_inter.rd][thread_index] <= VX_writeback_inter.write_data[thread_index];
		//  			end
		//  		end
		//  	end
	 // 		out_a_reg_data <= gpr[VX_gpr_read.rs1];
		// 	out_b_reg_data <= gpr[VX_gpr_read.rs2];
		//  end




		 // USING RAM blocks

		 // First RAM
		 integer thread_index_1;
		 always_ff@(posedge clk)
		 begin
		 	if (write_enable) begin
		 		for (thread_index_1 = 0; thread_index_1 <= `NT_M1; thread_index_1 = thread_index_1 + 1) begin
		 			if (VX_writeback_inter.wb_valid[thread_index_1]) begin
		 				gpr[VX_writeback_inter.rd][thread_index_1] <= VX_writeback_inter.write_data[thread_index_1];
		 			end
		 		end
		 	end
		 end

		 always @(negedge clk) begin
	 		out_a_reg_data <= gpr[VX_gpr_read.rs1];
		 end


		 // Second RAM
		 integer thread_index_2;
		 always_ff@(posedge clk)
		 begin
		 	if (write_enable) begin
		 		for (thread_index_2 = 0; thread_index_2 <= `NT_M1; thread_index_2 = thread_index_2 + 1) begin
		 			if (VX_writeback_inter.wb_valid[thread_index_2]) begin
		 				gpr[VX_writeback_inter.rd][thread_index_2] <= VX_writeback_inter.write_data[thread_index_2];
		 			end
		 		end
		 	end
		 end

		 always @(negedge clk) begin
	 		out_b_reg_data <= gpr[VX_gpr_read.rs2];
		 end


endmodule