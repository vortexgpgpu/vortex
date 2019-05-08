
`include "VX_define.v"

module VX_context (
  input wire        clk,
  input wire        in_valid[`NT_M1:0],
  input wire        in_write_register,
  input wire[4:0]   in_rd,
  input wire[31:0]  in_write_data[`NT_M1:0],
  input wire[4:0]   in_src1,
  input wire[4:0]   in_src2,
  input wire[31:0]  in_curr_PC,
  input wire        in_is_clone,
  input wire        in_is_jal,
  input wire        in_src1_fwd,
  input wire[31:0]  in_src1_fwd_data[`NT_M1:0],
  input wire        in_src2_fwd,
  input wire[31:0]  in_src2_fwd_data[`NT_M1:0],

  output reg[31:0]   out_a_reg_data[`NT_M1:0],
  output reg[31:0]   out_b_reg_data[`NT_M1:0],
  output wire        out_clone_stall
	
);
		wire[31:0] rd1_register[`NT_M1:0];
		wire[31:0] rd2_register[`NT_M1:0];
		/* verilator lint_off UNUSED */
		wire[31:0] clone_regsiters[31:0];
		/* verilator lint_on UNUSED */
		
		VX_register_file vx_register_file_master(
			.clk               (clk),
			.in_valid          (in_valid[0]),
			.in_write_register (in_write_register),
			.in_rd             (in_rd),
			.in_data           (in_write_data[0]),
			.in_src1           (in_src1),
			.in_src2           (in_src2),
			.out_regs          (clone_regsiters),
			.out_src1_data     (rd1_register[0]),
			.out_src2_data     (rd2_register[0])
		);

		genvar index;
		generate  
		for (index=1; index < `NT; index=index+1)  
		  begin: gen_code_label  
			wire to_clone;
		  	assign to_clone = (index == rd1_register[0]) && (state_stall == 1);
			VX_register_file_slave vx_register_file_slave(
				.clk               (clk),
				.in_valid          (in_valid[index]),
				.in_write_register (in_write_register),
				.in_rd             (in_rd),
				.in_data           (in_write_data[index]),
				.in_src1           (in_src1),
				.in_src2           (in_src2),
				.in_clone          (in_is_clone),
				.in_to_clone       (to_clone),
				.in_regs           (clone_regsiters),
				.out_src1_data     (rd1_register[index]),
				.out_src2_data     (rd2_register[index])
			);
		  end  
		endgenerate


		reg[5:0] state_stall = 0;
		always @(posedge clk) begin
			if ((in_is_clone) && state_stall == 0) begin
				state_stall <= 10;
				// $display("CLONEEE BITCH %d, 1 =? %h = %h -- %d", state_stall, rd1_register[0], to_clone_1, in_is_clone);
			end else if (state_stall == 1) begin
				// $display("ENDING CLONE, 1 =? %h = %h -- %d", rd1_register[0], to_clone_1, in_is_clone);
				state_stall <= 0;
			end else if (state_stall > 0) begin
				state_stall <= state_stall - 1;
				// $display("CLONEEE BITCH %d, 1 =? %h = %h -- %d", state_stall, rd1_register[0], to_clone_1, in_is_clone);
			end
		end

		genvar index_out_reg;
		generate
			for (index_out_reg = 0; index_out_reg < `NT; index_out_reg = index_out_reg + 1)
				begin
					assign out_a_reg_data[index_out_reg]   = (    (in_is_jal == 1'b1) ? in_curr_PC : ((in_src1_fwd == 1'b1) ? in_src1_fwd_data[index_out_reg] : rd1_register[index_out_reg]));
					assign out_b_reg_data[index_out_reg]   = (in_src2_fwd == 1'b1) ?  in_src2_fwd_data[index_out_reg] : rd2_register[index_out_reg];
				end
		endgenerate

		assign out_clone_stall = ((state_stall == 0) && in_is_clone) || ((state_stall != 1) && in_is_clone);

endmodule