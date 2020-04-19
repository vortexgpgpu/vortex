`include "VX_define.vh"

module VX_gpr_wrapper (
	input wire                  clk,
	input wire                  reset,
	VX_gpr_read_if           gpr_read_if,
	VX_wb_if                 writeback_if,	
	VX_gpr_jal_if            gpr_jal_if,

	output wire[`NUM_THREADS-1:0][31:0] out_a_reg_data,
	output wire[`NUM_THREADS-1:0][31:0] out_b_reg_data
	
);

	wire[`NUM_WARPS-1:0][`NUM_THREADS-1:0][31:0] temp_a_reg_data;
	wire[`NUM_WARPS-1:0][`NUM_THREADS-1:0][31:0] temp_b_reg_data;

	wire[`NUM_THREADS-1:0][31:0] jal_data;
	genvar index;
	generate 
	for (index = 0; index < `NUM_THREADS; index = index + 1) begin : jal_data_assign
		assign jal_data[index] = gpr_jal_if.curr_PC;
	end
	endgenerate

	`ifndef ASIC
		assign out_a_reg_data = (gpr_jal_if.is_jal   ? jal_data :  (temp_a_reg_data[gpr_read_if.warp_num]));
		assign out_b_reg_data =                                    (temp_b_reg_data[gpr_read_if.warp_num]);
	`else 

		wire zer = 0;

		wire[`NW_BITS-1:0] old_warp_num;	
		VX_generic_register #(
			.N(`NW_BITS-1+1)
		) store_wn (
			.clk  (clk),
			.reset(reset),
			.stall(zer),
			.flush(zer),
			.in   (gpr_read_if.warp_num),
			.out  (old_warp_num)
		);

		assign out_a_reg_data = (gpr_jal_if.is_jal   ? jal_data :  (temp_a_reg_data[old_warp_num]));
		assign out_b_reg_data =                                    (temp_b_reg_data[old_warp_num]);
		
	`endif

	genvar warp_index;
	generate
		
		for (warp_index = 0; warp_index < `NUM_WARPS; warp_index = warp_index + 1) begin : warp_gprs
			wire valid_write_request = warp_index == writeback_if.wb_warp_num;
			VX_gpr gpr(
				.clk                (clk),
				.reset              (reset),
				.valid_write_request(valid_write_request),
				.gpr_read_if        (gpr_read_if),
				.writeback_if 		(writeback_if),
				.out_a_reg_data     (temp_a_reg_data[warp_index]),
				.out_b_reg_data     (temp_b_reg_data[warp_index])
			);
		end

	endgenerate	

endmodule


