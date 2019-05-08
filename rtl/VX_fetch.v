
`include "VX_define.v"

module VX_fetch (
	input  wire        clk,
	input  wire        reset,
	input  wire        in_branch_dir,
	input  wire        in_freeze,
	input  wire[31:0]  in_branch_dest,
	input  wire        in_branch_stall,
	input  wire        in_fwd_stall,
	input  wire        in_branch_stall_exe,
	input  wire        in_clone_stall,
	input  wire        in_jal,
	input  wire[31:0]  in_jal_dest,
	input  wire        in_interrupt,
	input  wire        in_debug,
	input  wire[31:0]  in_instruction,
	input  wire        in_thread_mask[`NT_M1:0],
	input  wire        in_change_mask,

	output wire[31:0]  out_instruction,
	output wire        out_delay,
	// output wire[1:0]   out_warp_num,
	output wire[31:0]  out_curr_PC,
	output wire        out_valid[`NT_M1:0]
);

		reg       stall;
		reg[31:0] out_PC;

		// reg[1:0] warp_num;

		// initial begin
		// 	warp_num = 0;
		// end




		assign stall = in_clone_stall || in_branch_stall || in_fwd_stall || in_branch_stall_exe || in_interrupt || in_freeze || in_debug;


		wire[31:0] warp_pc;
		wire       warp_valid[`NT_M1:0];

		VX_warp VX_Warp(
			.clk           (clk),
			.reset         (reset),
			.stall         (stall),
			.in_thread_mask(in_thread_mask),
			.in_change_mask(in_change_mask),
			.in_jal        (in_jal),
			.in_jal_dest   (in_jal_dest),
			.in_branch_dir (in_branch_dir),
			.in_branch_dest(in_branch_dest),
			.out_PC        (warp_pc),
			.out_valid     (warp_valid)
			);


		assign out_PC = warp_pc;

		// always @(*) begin
		// 	$display("FETCH PC: %h (%h, %h, %h)",delete, delete, in_jal_dest, in_branch_dest);
		// end


		assign out_curr_PC     = out_PC;
		assign out_valid       = warp_valid;
		// assign out_warp_num    = warp_num;
		assign out_delay       = 0;
		assign out_instruction = stall ? 32'b0 : in_instruction;



endmodule