
`include "VX_define.v"

module VX_fetch (
	input  wire       clk,
	input  wire       reset,
	input  wire       in_branch_dir,
	input  wire       in_freeze,
	input  wire[31:0] in_branch_dest,
	input  wire       in_branch_stall,
	input  wire       in_fwd_stall,
	input  wire       in_branch_stall_exe,
	input  wire       in_clone_stall,
	input  wire       in_jal,
	input  wire[31:0] in_jal_dest,
	input  wire       in_interrupt,
	input  wire       in_debug,
	input  wire[31:0] in_instruction,
	input  wire       in_thread_mask[`NT_M1:0],
	input  wire       in_change_mask,

	output wire[31:0]      out_instruction,
	output wire            out_delay,
	output wire[31:0]      out_curr_PC,
	output wire            out_valid[`NT_M1:0]
);


		reg       stall;
		reg[31:0] out_PC;


		reg valid[`NT_M1:0];


		integer ini_cur_th = 0;
		genvar out_cur_th;

		initial begin
			for (ini_cur_th = 1; ini_cur_th < `NT; ini_cur_th=ini_cur_th+1)
				valid[ini_cur_th]   = 0; // Thread 1 active
			valid[0]   = 1;
		end


		always @(*) begin : proc_
			if (in_change_mask) begin
				// $display("CHANGING MASK: [%d %d]",in_thread_mask[0], in_thread_mask[1]);
				assign valid = in_thread_mask;
			end
		end



		assign out_delay = 0;

		assign stall = in_clone_stall || in_branch_stall || in_fwd_stall || in_branch_stall_exe || in_interrupt || in_freeze || in_debug;

		assign out_instruction = stall ? 32'b0 : in_instruction;
		// assign out_instruction = in_instruction;

		generate
			for (out_cur_th = 0; out_cur_th < `NT; out_cur_th = out_cur_th+1)
				assign out_valid[out_cur_th] = in_change_mask ? in_thread_mask[out_cur_th] : stall ? 1'b0  : valid[out_cur_th];
		endgenerate



		wire[31:0] warp_pc;
		VX_warp VX_Warp(
			.clk           (clk),
			.reset         (reset),
			.stall         (stall),
			.in_jal        (in_jal),
			.in_jal_dest   (in_jal_dest),
			.in_branch_dir (in_branch_dir),
			.in_branch_dest(in_branch_dest),
			.out_PC        (warp_pc)
			);


		assign out_PC = warp_pc;

		// always @(*) begin
		// 	$display("FETCH PC: %h (%h, %h, %h)",delete, delete, in_jal_dest, in_branch_dest);
		// end


		assign out_curr_PC = out_PC;
		



		// always @(*) begin
		// 	$display("Fetch out pc: %h", out_PC);
		// end




endmodule