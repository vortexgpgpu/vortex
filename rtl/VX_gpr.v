
`include "VX_define.v"

module VX_gpr (
	input wire                  clk,
	input wire                  valid_write_request,
	VX_gpr_read_inter           VX_gpr_read,
	VX_wb_inter                 VX_writeback_inter,

	output reg[`NT_M1:0][31:0] out_a_reg_data,
	output reg[`NT_M1:0][31:0] out_b_reg_data
);

	wire write_enable;

	assign write_enable = valid_write_request && ((VX_writeback_inter.wb != 0) && (VX_writeback_inter.rd != 5'h0));
	 // USING RAM blocks
	 // First RAM
	 byte_enabled_simple_dual_port_ram first_ram(
	 	.we   (write_enable),
	 	.clk  (clk),
	 	.waddr(VX_writeback_inter.rd),
	 	.raddr(VX_gpr_read.rs1),
	 	.be   (VX_writeback_inter.wb_valid),
	 	.wdata(VX_writeback_inter.write_data),
	 	.q    (out_a_reg_data)
	 	);

	 // Second RAM block
	 byte_enabled_simple_dual_port_ram second_ram(
	 	.we   (write_enable),
	 	.clk  (clk),
	 	.waddr(VX_writeback_inter.rd),
	 	.raddr(VX_gpr_read.rs2),
	 	.be   (VX_writeback_inter.wb_valid),
	 	.wdata(VX_writeback_inter.write_data),
	 	.q    (out_b_reg_data)
	 	);




	// logic[`NT_M1:0][31:0] gpr[31:0]; // gpr[register_number][thread_number][data_bits]

	// wire write_enable;

	// assign write_enable = valid_write_request && ((VX_writeback_inter.wb != 0) && (VX_writeback_inter.rd != 5'h0));
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


endmodule