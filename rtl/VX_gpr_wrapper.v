`include "VX_define.v"

module VX_gpr_wrapper (
	input wire                  clk,
	input wire[`NW_M1:0]        in_warp_num,
	input wire[`NW_M1:0]        in_wb_warp_num,
	input wire                  is_clone,
	input wire[`NT_M1:0]        in_wb_valid,
	input wire[4:0]             in_rd,
	input wire[4:0]             in_rs1,
	input wire[4:0]             in_rs2,
	input wire[31:0]            in_curr_PC,
	input wire                  is_jal,
	input wire                  in_src1_fwd,
	input wire[`NT_M1:0][31:0]  in_src1_fwd_data,
	input wire                  in_src2_fwd,
	input wire[`NT_M1:0][31:0]  in_src2_fwd_data,
	input wire                  write_register, // WB
	input wire[`NT_M1:0][31:0]  in_write_data,
	input wire                  is_wspawn,
	input wire[`NW_M1:0]        in_which_wspawn,

	output wire[`NT_M1:0][31:0] out_a_reg_data,
	output wire[`NT_M1:0][31:0] out_b_reg_data,
	output wire                 out_clone_stall
	
);

	wire[`NW-1:0][`NT_M1:0][31:0] temp_a_reg_data;
	wire[`NW-1:0][`NT_M1:0][31:0] temp_b_reg_data;

	assign out_a_reg_data = temp_a_reg_data[in_warp_num];
	assign out_b_reg_data = temp_b_reg_data[in_warp_num];

	wire[31:0][31:0] w0_t0_registers;

	wire[`NW-1:0]  temp_clone_stall;

	assign out_clone_stall = (|temp_clone_stall);


	wire       curr_warp_zero     = in_warp_num == 0;
	wire       context_zero_valid = (in_wb_warp_num == 0);
	wire       real_zero_isclone  = is_clone  && (in_warp_num == 0);  
	VX_context VX_Context_zero(
		.clk              (clk),
		.in_warp          (curr_warp_zero),
		.in_wb_warp       (context_zero_valid),
		.in_valid         (in_wb_valid),
		.in_rd            (in_rd),
		.in_src1          (in_rs1),
		.in_src2          (in_rs2),
		.in_curr_PC       (in_curr_PC),
		.in_is_clone      (real_zero_isclone),
		.in_is_jal        (is_jal),
		.in_src1_fwd      (in_src1_fwd),
		.in_src1_fwd_data (in_src1_fwd_data),
		.in_src2_fwd      (in_src2_fwd),
		.in_src2_fwd_data (in_src2_fwd_data),
		.in_write_register(write_register),
		.in_write_data    (in_write_data),
		.out_a_reg_data   (temp_a_reg_data[0]),
		.out_b_reg_data   (temp_b_reg_data[0]),
		.out_clone_stall  (temp_clone_stall[0]),
		.w0_t0_registers  (w0_t0_registers)
	);

	genvar r;
	generate
		for (r = 1; r < `NW; r = r + 1) begin
			wire context_glob_valid = (in_wb_warp_num == r);
			wire curr_warp_glob     = in_warp_num == r;
			wire real_wspawn        = is_wspawn && (in_which_wspawn == r); 
			wire real_isclone       = is_clone  && (in_warp_num == r);      
			VX_context_slave VX_Context_one(
				.clk              (clk),
				.in_warp          (curr_warp_glob),
				.in_wb_warp       (context_glob_valid),
				.in_valid         (in_wb_valid),
				.in_rd            (in_rd),
				.in_src1          (in_rs1),
				.in_src2          (in_rs2),
				.in_curr_PC       (in_curr_PC),
				.in_is_clone      (real_isclone),
				.in_is_jal        (is_jal),
				.in_src1_fwd      (in_src1_fwd),
				.in_src1_fwd_data (in_src1_fwd_data),
				.in_src2_fwd      (in_src2_fwd),
				.in_src2_fwd_data (in_src2_fwd_data),
				.in_write_register(write_register),
				.in_write_data    (in_write_data),
				.in_wspawn_regs   (w0_t0_registers),
				.in_wspawn        (real_wspawn),
				.out_a_reg_data   (temp_a_reg_data[r]),
				.out_b_reg_data   (temp_b_reg_data[r]),
				.out_clone_stall  (temp_clone_stall[r])
			);
		end
	endgenerate

endmodule