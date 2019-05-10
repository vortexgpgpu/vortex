
`include "VX_define.v"

module VX_fetch (
	input  wire           clk,
	input  wire           reset,
	input  wire           in_branch_dir,
	input  wire           in_freeze,
	input  wire[31:0]     in_branch_dest,
	input  wire           in_branch_stall,
	input  wire           in_fwd_stall,
	input  wire           in_branch_stall_exe,
	input  wire           in_clone_stall,
	input  wire           in_jal,
	input  wire[31:0]     in_jal_dest,
	input  wire           in_interrupt,
	input  wire           in_debug,
	input  wire[31:0]     in_instruction,
	input  wire           in_thread_mask[`NT_M1:0],
	input  wire           in_change_mask,
	input  wire[`NW_M1:0] in_decode_warp_num,
	input  wire[`NW_M1:0] in_memory_warp_num,
	input  wire           in_wspawn,
	input  wire[31:0]     in_wspawn_pc,
	input  wire           in_ebreak,

	output wire[31:0]     out_instruction,
	output wire           out_delay,
	output wire[`NW_M1:0] out_warp_num,
	output wire[31:0]     out_curr_PC,
	output wire           out_valid[`NT_M1:0],
	output wire           out_ebreak
);

		reg       stall;
		reg[31:0] out_PC;

		reg[`NW_M1:0] warp_num;
		reg[`NW_M1:0] warp_state;

		initial begin
			warp_num   = 0;
			warp_state = 0;
		end

		wire add_warp    = in_wspawn && !in_ebreak && !in_clone_stall;
		wire remove_warp = in_ebreak && !in_wspawn && !in_clone_stall;

		always @(posedge clk or posedge reset) begin
			if (reset || (warp_num == warp_state) || remove_warp || add_warp) begin
				warp_num   <= 0;
			end else begin
				warp_num   <= warp_num + 1;
			end

			if (add_warp) begin
				// $display("Adding a new warp %h", warp_state);
				warp_state <= warp_state + 1;
			end else if (remove_warp) begin
				// $display("Removing a warp %h", warp_state);
				warp_state <= warp_state - 1;
			end
		end

		assign out_ebreak = (warp_state == 0) && in_ebreak;


		assign stall = in_clone_stall || in_branch_stall || in_fwd_stall || in_branch_stall_exe || in_interrupt || in_freeze || in_debug;



		wire       warp_zero_change_mask = in_change_mask && (in_decode_warp_num == 0);
		wire       warp_zero_jal         = in_jal         && (in_memory_warp_num == 0);
		wire       warp_zero_branch      = in_branch_dir  && (in_memory_warp_num == 0);
		wire       warp_zero_stall       = stall          || (warp_num == 1);
		wire       warp_zero_wspawn      = 0;
		wire[31:0] warp_zero_wspawn_pc   = 32'h0;

		wire[31:0] warp_zero_pc;
		wire       warp_zero_valid[`NT_M1:0];
		VX_warp VX_Warp_zero(
			.clk           (clk),
			.reset         (reset),
			.stall         (warp_zero_stall),
			.in_thread_mask(in_thread_mask),
			.in_change_mask(warp_zero_change_mask),
			.in_jal        (warp_zero_jal),
			.in_jal_dest   (in_jal_dest),
			.in_branch_dir (warp_zero_branch),
			.in_branch_dest(in_branch_dest),
			.in_wspawn     (warp_zero_wspawn),
			.in_wspawn_pc  (warp_zero_wspawn_pc),
			.out_PC        (warp_zero_pc),
			.out_valid     (warp_zero_valid)
			);


		wire       warp_one_change_mask = in_change_mask && (in_decode_warp_num == 1);
		wire       warp_one_jal         = in_jal         && (in_memory_warp_num == 1);
		wire       warp_one_branch      = in_branch_dir  && (in_memory_warp_num == 1);
		wire       warp_one_stall       = stall          || (warp_num == 0);
		wire[31:0] warp_one_pc;
		wire       warp_one_valid[`NT_M1:0];
		VX_warp VX_Warp_one(
			.clk           (clk),
			.reset         (reset),
			.stall         (warp_one_stall),
			.in_thread_mask(in_thread_mask),
			.in_change_mask(warp_one_change_mask),
			.in_jal        (warp_one_jal),
			.in_jal_dest   (in_jal_dest),
			.in_branch_dir (warp_one_branch),
			.in_branch_dest(in_branch_dest),
			.in_wspawn     (in_wspawn),
			.in_wspawn_pc  (in_wspawn_pc),
			.out_PC        (warp_one_pc),
			.out_valid     (warp_one_valid)
			);

		// always @(*) begin
		// 	if (in_wspawn) begin
		// 		$display("Spawning a warp @ %h",in_wspawn_pc);
		// 	end
		// end

		// always @(posedge clk) begin 
		// 	$display("curr warp: %h Threads:%d%d PC: %h", warp_num, out_valid[0],out_valid[1], out_PC);
		// end

		// always @(*) begin
		// 	if (warp_num == 1) begin
		// 		$display("Going to PC: %h", warp_one_pc);
		// 	end
		// end

		assign out_PC    = (warp_num == 0) ? warp_zero_pc    : warp_one_pc;
		assign out_valid = (warp_num == 0) ? warp_zero_valid : warp_one_valid;

		// always @(*) begin
		// 	$display("FETCH PC: %h (%h, %h, %h)",delete, delete, in_jal_dest, in_branch_dest);
		// end


		assign out_curr_PC     = out_PC;
		assign out_warp_num    = warp_num;
		assign out_delay       = 0;

		assign out_instruction = stall ? 32'b0 : in_instruction;



endmodule