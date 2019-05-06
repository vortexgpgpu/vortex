
module VX_warp (
	input wire       clk,
	input wire       reset,
	input wire       stall,
	input wire       in_jal,
	input wire[31:0] in_jal_dest,
	input wire       in_branch_dir,
	input wire[31:0] in_branch_dest,


	output wire[31:0] out_PC
);

		reg[31:0] real_PC;

		initial begin
			real_PC = 0;
		end

		var[31:0] temp_PC;
		always @(*) begin
			if (in_jal == 1'b1) begin
				temp_PC = in_jal_dest;
			end else if (in_branch_dir == 1'b1) begin
				temp_PC = in_branch_dest;
			end else begin
				temp_PC = real_PC;
			end
		end

		assign out_PC = temp_PC;

		always @(posedge clk or posedge reset) begin
			if (reset) begin
				real_PC <= 0;
			end else if (stall != 1'b1) begin
				real_PC <= temp_PC + 32'h4;
			end

		end
		

endmodule