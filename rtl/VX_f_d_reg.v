`include "buses.vh"

`include "VX_define.v"

module VX_f_d_reg (
	  input wire             clk,
	  input wire             reset,
	  input wire             in_valid[`NT_M1:0],
	  input wire             in_fwd_stall,
	  input wire             in_freeze,
	  input wire             in_clone_stall,

	  output wire[31:0]      out_instruction,
	  output wire[31:0]      out_curr_PC,
	  output wire            out_valid[`NT_M1:0],
	  output wire[`NW_M1:0]  out_warp_num,
	  /* verilator lint_off UNUSED */
	  input wire[31:0]       in_instruction,
	  input wire[31:0]       in_curr_PC,
	  input wire[`NW_M1:0]   in_warp_num,
	  input fe_inst_meta_de_t fe_inst_meta_fd
	  /* verilator lint_on UNUSED */
);

	// genvar index;
	// always @(posedge clk) begin
	// 	// $display("in_instruction: %x\tfe_inst_meta_fd.instruction: %x",in_instruction, fe_inst_meta_fd.instruction);
	// 	$error("finally");
	// 	assert (in_instruction == fe_inst_meta_fd.instruction);
	// 	assert (in_curr_PC     == fe_inst_meta_fd.inst_pc);
	// 	assert (in_warp_num    == fe_inst_meta_fd.warp_num);
	// 	for (index = 0; index <= `NT_M1; index = index + 1) assert (in_valid[index] == fe_inst_meta_fd.valid[index]);
	// end

	// var match;
	// always @(*) begin
	// 	match = ;
	// 	if (!match)
	// 		$display("in_instruction: %x, fe_inst_meta_fd.instruction: %x",in_instruction ,fe_inst_meta_fd.instruction);
	// end

	reg[31:0]      instruction;
	reg[31:0]      curr_PC;
	reg            valid[`NT_M1:0];
	reg[`NW_M1:0]  warp_num;

	integer reset_cur_thread = 0;


	always @(posedge clk or posedge reset) begin
		if(reset) begin
			instruction <= 32'h0;
			curr_PC     <= 32'h0;
			warp_num    <= 0;
			for (reset_cur_thread = 0; reset_cur_thread < `NT; reset_cur_thread = reset_cur_thread + 1)
				valid[reset_cur_thread]    <=  1'b0;

		end else if (in_fwd_stall == 1'b1 || in_freeze == 1'b1 || in_clone_stall) begin
			// if (in_clone_stall) begin
			// 	$display("STALL BECAUSE OF CLONE");
			// end
		end else begin
			instruction <= in_instruction;
			valid       <= in_valid;
			curr_PC     <= in_curr_PC;
			warp_num    <= in_warp_num;

			// instruction <= fe_inst_meta_fd.instruction;
			// valid       <= fe_inst_meta_fd.valid;
			// curr_PC     <= fe_inst_meta_fd.inst_pc;
			// warp_num    <= fe_inst_meta_fd.warp_num;
		end
	end

	always @(*) begin
		// $display("PC in VX_f_d_reg: %h", curr_PC);
	end

	assign out_instruction = instruction;
	assign out_curr_PC     = curr_PC;
	assign out_valid       = valid;
	assign out_warp_num    = warp_num;



endmodule