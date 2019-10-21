`include "VX_define.v"

module VX_warp_scheduler (
	input wire           clk,    // Clock
	input wire           reset,
	input wire           stall,
	// Wspawn
	input wire           wspawn,
	input wire[31:0]     wsapwn_pc,

	// CTM
	input  wire           ctm,
	input  wire[`NT_M1:0] ctm_mask,
	input  wire[`NW_M1:0] ctm_warp_num,

	// WHALT
	input  wire           whalt,
	input  wire[`NW_M1:0] whalt_warp_num,

	// WSTALL
	input  wire           wstall,
	input  wire[`NW_M1:0] wstall_warp_num,

	// Split
	input wire            is_split,
	input wire[`NT_M1:0]  split_new_mask,
	input wire[`NT_M1:0]  split_later_mask,
	input wire[31:0]      split_save_pc,	
	input wire[`NW_M1:0]  split_warp_num,

	// Join
	input wire            is_join,
	input wire[`NW_M1:0]  join_warp_num,

	// JAL
	input wire           jal,
	input wire[31:0]     jal_dest,
	input wire[`NW_M1:0] jal_warp_num,

	// Branch
	input wire           branch_valid,
	input wire           branch_dir,
	input wire[31:0]     branch_dest,
	input wire[`NW_M1:0] branch_warp_num,

	output wire[`NT_M1:0] thread_mask,
	output wire[`NW_M1:0] warp_num,
	output wire[31:0]     warp_pc,
	output wire           out_ebreak

);

	wire[(1+32+`NT_M1):0] d;

	wire           join_fall;
	wire[31:0]     join_pc;
	wire[`NT_M1:0] join_tm;

	wire in_wspawn = wspawn;
	wire in_ctm = ctm;
	wire in_whalt = whalt;
	wire in_wstall = wstall;

	reg[`NW-1:0] warp_active;
	reg[`NW-1:0] warp_stalled;

	reg[`NW-1:0] visible_active;
	wire[`NW-1:0] use_active;


	reg[`NT_M1:0] thread_masks[`NW-1:0];
	reg[31:0]     warp_pcs[`NW-1:0];


	// Choosing a warp to wsapwn
	wire[`NW_M1:0] warp_to_wsapwn;
	wire           found_wspawn;

	wire[`NW_M1:0] warp_to_schedule;
	wire           schedule;

	wire hazard;
	wire global_stall;

	wire real_schedule;

	wire[31:0] new_pc;

	/* verilator lint_off UNUSED */
	wire[`NW_M1:0] num_active;
	/* verilator lint_on UNUSED */

	reg[1:0] start;
	// initial begin
	// 	warp_pcs[0] = (32'h80000000 - 4);
	// 	start = 0;
	// 	warp_active[0]     = 1; // Activating first warp
	// 	visible_active[0]  = 1; // Activating first warp
	// 	thread_masks[0][0] = 1; // Activating first thread in first warp
	// end


	always @(posedge clk or posedge reset) begin
		if (reset) begin
			warp_pcs[0] = (32'h80000000 - 4);
			start = 0;
			warp_active[0]     = 1; // Activating first warp
			visible_active[0]  = 1; // Activating first warp
			thread_masks[0][0] = 1; // Activating first thread in first warp
		end

		// Wsapwning warps
		if (wspawn && found_wspawn) begin
			warp_pcs[warp_to_wsapwn]       <= wsapwn_pc;
			warp_active[warp_to_wsapwn]    <= 1;
			visible_active[warp_to_wsapwn] <= 1;
		end
		// Halting warps
		if (whalt) begin
			warp_active[whalt_warp_num]    <= 0;
			visible_active[whalt_warp_num] <= 0;
		end

		// Changing thread masks
		if (ctm) begin
			thread_masks[ctm_warp_num] <= ctm_mask;
			warp_stalled[ctm_warp_num] <= 0;
		end

		// Stalling the scheduling of warps
		if (wstall) begin
			warp_stalled[wstall_warp_num]   <= 1;
			visible_active[wstall_warp_num] <= 0;
		end

		if (is_split) begin
			warp_stalled[split_warp_num] <= 0;
			thread_masks[split_warp_num] <= split_new_mask;
		end

		if (is_join) begin
			if (!join_fall) begin
				warp_pcs[join_warp_num] <= join_pc;
			end
			thread_masks[join_warp_num] <= join_tm;
		end

		// Refilling active warps
		if ((visible_active == 0) && !(stall || wstall || hazard || is_join)) begin
			visible_active <= warp_active & (~warp_stalled);
		end

		// First cycle
		if (start <= 2) begin
			start <= 1;
			visible_active <= warp_active & (~warp_stalled);
		end

		// Don't change state if stall
		if (!global_stall && real_schedule && (thread_mask != 0)) begin
			visible_active[warp_to_schedule] <= 0;
			warp_pcs[warp_to_schedule]       <= new_pc;
		end

		// Jal
		if (jal) begin
			warp_pcs[jal_warp_num]     <= jal_dest;
			warp_stalled[jal_warp_num] <= 0;
		end

		// Branch
		if (branch_valid) begin
			if (branch_dir) warp_pcs[branch_warp_num]     <= branch_dest;
			warp_stalled[branch_warp_num] <= 0;
		end
	end

	wire[(1+32+`NT_M1):0] q1 = {1'b1, warp_pcs[split_warp_num], thread_masks[split_warp_num]};
	wire[(1+32+`NT_M1):0] q2 = {1'b0, split_save_pc           , split_later_mask};


	assign {join_fall, join_pc, join_tm} = d;



	genvar curr_warp;
	for (curr_warp = 0; curr_warp < `NW; curr_warp = curr_warp + 1) begin
		wire correct_warp_s = (curr_warp == split_warp_num);
		wire correct_warp_j = (curr_warp == join_warp_num);

		wire push = is_split && correct_warp_s;
		wire pop  = is_join  && correct_warp_j;
		VX_generic_stack #(.WIDTH(1+32+`NT), .DEPTH($clog2(`NT))) ipdom_stack(
			.clk  (clk),
			.reset(reset),
			.push (push),
			.pop  (pop),
			.d    (d),
			.q1   (q1),
			.q2   (q2)
			);
	end

	// wire should_stall = stall || (jal && (warp_to_schedule == jal_warp_num)) || (branch_dir && (warp_to_schedule == branch_warp_num));

	wire should_jal = (jal && (warp_to_schedule == jal_warp_num));
	wire should_bra = (branch_dir && (warp_to_schedule == branch_warp_num));

	assign hazard = (should_jal || should_bra) && schedule;

	assign real_schedule = schedule && !warp_stalled[warp_to_schedule];

	assign global_stall = (stall || wstall || hazard || !real_schedule || is_join);


	assign warp_pc     = warp_pcs[warp_to_schedule];
	assign thread_mask = (global_stall) ? 0 : thread_masks[warp_to_schedule];
	assign warp_num    = warp_to_schedule;


	assign new_pc = warp_pc + 4;


	assign use_active = (num_active <= 1) ? (warp_active & (~warp_stalled)) : visible_active;

	// Choosing a warp to schedule
	VX_priority_encoder choose_schedule(
		.valids(use_active),
		.index (warp_to_schedule),
		.found (schedule)
	);


	VX_priority_encoder choose_wsapwn(
		.valids(~warp_active),
		.index (warp_to_wsapwn),
		.found (found_wspawn)
	);


	// Valid counter
	VX_one_counter valid_counter(
		.valids(visible_active),
		.ones_found(num_active)
		);


	assign out_ebreak = (warp_active == 0);



endmodule