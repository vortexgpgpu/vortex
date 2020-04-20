`include "VX_define.v"


module VX_warp (
	input  wire       clk,
	input  wire       reset,
	input  wire       stall,
	input  wire       remove,
	input  wire[`NUM_THREADS-1:0] in_thread_mask,
	input  wire       in_change_mask,
	input  wire       in_jal,
	input  wire[31:0] in_jal_dest,
	input  wire       in_branch_dir,
	input  wire[31:0] in_branch_dest,
	input  wire       in_wspawn,
	input  wire[31:0] in_wspawn_pc,

	output wire[31:0] out_PC,
	output wire[`NUM_THREADS-1:0] out_valid
);

		reg[31:0] real_PC;
		logic [31:0] temp_PC;
		logic [31:0] use_PC;
		reg[`NUM_THREADS-1:0] valid;

		reg[`NUM_THREADS-1:0] valid_zero;

		integer ini_cur_th = 0;
		initial begin
			real_PC = 0;
			for (ini_cur_th = 1; ini_cur_th < `NUM_THREADS; ini_cur_th=ini_cur_th+1) begin
				valid[ini_cur_th]   = 0; // Thread 1 active
				valid_zero[ini_cur_th] = 0;
			end
			valid[0]      = 1;
			valid_zero[0] = 0;
		end


		always @(posedge clk) begin
			if (remove) begin
				valid <= valid_zero;
			end else if (in_change_mask) begin
				valid <= in_thread_mask;
			end
		end


		genvar out_cur_th;
		generate
			for (out_cur_th = 0; out_cur_th < `NUM_THREADS; out_cur_th = out_cur_th+1) begin : out_valid_assign
				assign out_valid[out_cur_th] = in_change_mask ? in_thread_mask[out_cur_th] : stall ? 1'b0  : valid[out_cur_th];
			end
		endgenerate


		always @(*) begin
			if (in_jal == 1'b1) begin
				temp_PC = in_jal_dest;
				// $display("LINKING TO %h", temp_PC);
			end else if (in_branch_dir == 1'b1) begin
				temp_PC = in_branch_dest;
			end else begin
				temp_PC = real_PC;
			end
		end

		assign use_PC = temp_PC;
		assign out_PC = temp_PC;

		always @(posedge clk) begin
			if (reset) begin
				real_PC <= 0;
			end else if (in_wspawn == 1'b1) begin
				// $display("Inside warp ***** Spawn @ %H",in_wspawn_pc);
				real_PC <= in_wspawn_pc;
			end else if (!stall) begin
				real_PC <= use_PC + 32'h4;
			end else begin
				real_PC <= use_PC;
			end

		end
		

endmodule