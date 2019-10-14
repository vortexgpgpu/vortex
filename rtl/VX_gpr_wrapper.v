`include "VX_define.v"

module VX_gpr_wrapper (
	input wire                  clk,
	VX_gpr_read_inter           VX_gpr_read,
	VX_wb_inter                 VX_writeback_inter,
	VX_forward_response_inter   VX_fwd_rsp,
	
	VX_gpr_jal_inter            VX_gpr_jal,

	output wire[`NT_M1:0][31:0] out_a_reg_data,
	output wire[`NT_M1:0][31:0] out_b_reg_data,
	output wire                 out_gpr_stall
	
);

	wire[`NW-1:0][`NT_M1:0][31:0] temp_a_reg_data;
	wire[`NW-1:0][`NT_M1:0][31:0] temp_b_reg_data;

	wire[`NT_M1:0][31:0] jal_data;
	genvar index;
	for (index = 0; index <= `NT_M1; index = index + 1) assign jal_data[index] = VX_gpr_jal.curr_PC;


	assign out_a_reg_data = (VX_gpr_jal.is_jal   ? jal_data :  (VX_fwd_rsp.src1_fwd ? VX_fwd_rsp.src1_fwd_data : temp_a_reg_data[VX_gpr_read.warp_num]));
	assign out_b_reg_data =                                    (VX_fwd_rsp.src2_fwd ? VX_fwd_rsp.src2_fwd_data : temp_b_reg_data[VX_gpr_read.warp_num]);

	genvar warp_index;
	generate
		
		for (warp_index = 0; warp_index < `NW; warp_index = warp_index + 1) begin

			wire valid_write_request = warp_index == VX_writeback_inter.wb_warp_num;
			VX_gpr vx_gpr(
				.clk                (clk),
				.valid_write_request(valid_write_request),
				.VX_gpr_read        (VX_gpr_read),
				.VX_writeback_inter (VX_writeback_inter),
				.out_a_reg_data     (temp_a_reg_data[warp_index]),
				.out_b_reg_data     (temp_b_reg_data[warp_index])
				);

		end

	endgenerate

	assign out_gpr_stall = 0;
	


endmodule


