
`include "VX_define.v"

module VX_f_d_reg (
	  input wire             clk,
	  input wire             reset,
	  input wire[31:0]       in_instruction,
	  input wire[`NT_M1:0]   in_valid,
	  input wire[31:0]       in_curr_PC,
	  input wire             in_fwd_stall,
	  input wire             in_freeze,
	  output wire[31:0]      out_instruction,
	  output wire[31:0]      out_curr_PC,
	  output wire[`NT_M1:0]  out_valid
);

	// always @(posedge clk) begin
		// $display("Fetch Inst: %d\tDecode Inst: %d", in_instruction, out_instruction);
	// end

	reg[31:0]      instruction;
	reg[31:0]      curr_PC;
	reg[`NT_M1:0]  valid;

	integer reset_cur_thread = 0;


	always @(posedge clk or posedge reset) begin
		if(reset) begin
			instruction <= 32'h0;
			curr_PC     <= 32'h0;
			for (reset_cur_thread = 0; reset_cur_thread < `NT; reset_cur_thread = reset_cur_thread + 1)
				valid[reset_cur_thread]    <=  1'b0;

		end else if (in_fwd_stall == 1'b1 || in_freeze == 1'b1) begin
			// $display("Because of FWD stall keeping pc: %h", curr_PC);
		end else begin
			instruction <= in_instruction;
			valid       <= in_valid;
			curr_PC     <= in_curr_PC;
		end
	end

	always @(*) begin
		// $display("PC in VX_f_d_reg: %h", curr_PC);
	end

	assign out_instruction = instruction;
	assign out_curr_PC     = curr_PC;
	assign out_valid       = valid;



endmodule