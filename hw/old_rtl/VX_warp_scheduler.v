`include "VX_define.v"

module VX_warp_scheduler (
	input wire           clk,    // Clock
	input wire           reset,
	input wire           stall,
	// Wspawn
	input wire           wspawn,
	input wire[31:0]     wsapwn_pc,
	input wire[`NW-1:0]  wspawn_new_active,

	// CTM
	input  wire           ctm,
	input  wire[`NT_M1:0] ctm_mask,
	input  wire[`NW_M1:0] ctm_warp_num,

	// WHALT
	input  wire           whalt,
	input  wire[`NW_M1:0] whalt_warp_num,

	input wire                 is_barrier,
	input wire[31:0]           barrier_id,
	input wire[$clog2(`NW):0]  num_warps,
	input wire[`NW_M1:0]       barrier_warp_num,

	// WSTALL
	input  wire           wstall,
	input  wire[`NW_M1:0] wstall_warp_num,

	// Split
	input wire            is_split,
	input wire            dont_split,
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
	output wire           out_ebreak,
	output wire           scheduled_warp

);

	wire update_use_wspawn;

	wire update_visible_active;

	wire[(1+32+`NT_M1):0] d[`NW-1:0];

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

	wire wstall_this_cycle;

	reg[`NT_M1:0] thread_masks[`NW-1:0];
	reg[31:0]     warp_pcs[`NW-1:0];

	// barriers
	reg[`NW-1:0] barrier_stall_mask[(`NUM_BARRIERS-1):0];
	wire         reached_barrier_limit;
	wire[`NW-1:0] curr_barrier_mask;
	wire[$clog2(`NW):0] curr_barrier_count;

	// wsapwn
	reg[31:0]    use_wsapwn_pc;
	reg[`NW-1:0] use_wsapwn;

	wire[`NW_M1:0] warp_to_schedule;
	wire           schedule;

	wire hazard;
	wire global_stall;

	wire real_schedule;

	wire[31:0] new_pc;

	reg[`NW-1:0] total_barrier_stall;

	reg didnt_split;

	/* verilator lint_off UNUSED */
	// wire[$clog2(`NW):0] num_active;
	/* verilator lint_on UNUSED */

	integer curr_w_help;
	integer curr_barrier;
	always @(posedge clk or posedge reset) begin
		if (reset) begin
			for (curr_barrier = 0; curr_barrier < `NUM_BARRIERS; curr_barrier=curr_barrier+1) begin
				barrier_stall_mask[curr_barrier] <= 0;
			end
			use_wsapwn_pc         <= 0;
			use_wsapwn            <= 0;
			warp_pcs[0]           <= (32'h80000000 - 4);
			warp_active[0]        <= 1; // Activating first warp
			visible_active[0]     <= 1; // Activating first warp
			thread_masks[0]       <= 1; // Activating first thread in first warp
			warp_stalled          <= 0;
			didnt_split           <= 0;
			// total_barrier_stall    = 0;
			for (curr_w_help = 1; curr_w_help < `NW; curr_w_help=curr_w_help+1) begin
				warp_pcs[curr_w_help]        <= 0;
				warp_active[curr_w_help]     <= 0; // Activating first warp
				visible_active[curr_w_help]  <= 0; // Activating first warp
				thread_masks[curr_w_help]    <= 1; // Activating first thread in first warp
			end

		end else begin
			// Wsapwning warps
			if (wspawn) begin
				warp_active    <= wspawn_new_active;
				use_wsapwn_pc  <= wsapwn_pc;
				use_wsapwn     <= wspawn_new_active & (~`NW'b1);
			end

			if (is_barrier) begin
				warp_stalled[barrier_warp_num]       <= 0;
				if (reached_barrier_limit) begin
					barrier_stall_mask[barrier_id] <= 0;
				end else begin
					barrier_stall_mask[barrier_id][barrier_warp_num] <= 1;
				end
			end else if (ctm) begin
				thread_masks[ctm_warp_num] <= ctm_mask;
				warp_stalled[ctm_warp_num] <= 0;
			end else if (is_join && !didnt_split) begin
				if (!join_fall) begin
					warp_pcs[join_warp_num] <= join_pc;
				end
				thread_masks[join_warp_num] <= join_tm;
				didnt_split                 <= 0;
			end else if (is_split) begin
				warp_stalled[split_warp_num] <= 0;
				if (!dont_split) begin
					thread_masks[split_warp_num] <= split_new_mask;
					didnt_split                  <= 0;
				end else begin
					didnt_split                  <= 1;
				end
			end
			
			if (whalt) begin
				warp_active[whalt_warp_num]    <= 0;
				visible_active[whalt_warp_num] <= 0;
			end

			if (update_use_wspawn) begin
				use_wsapwn[warp_to_schedule]   <= 0;
				thread_masks[warp_to_schedule] <= 1;
			end


			// Stalling the scheduling of warps
			if (wstall) begin
				warp_stalled[wstall_warp_num]   <= 1;
				visible_active[wstall_warp_num] <= 0;
			end

			// Refilling active warps
			if (update_visible_active) begin
				visible_active <= warp_active & (~warp_stalled) & (~total_barrier_stall);
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
	end

	VX_countones #(.N(`NW)) barrier_count(
		.valids(curr_barrier_mask),
		.count (curr_barrier_count)
		);

	wire[$clog2(`NW):0] count_visible_active;
	VX_countones #(.N(`NW)) num_visible(
		.valids(visible_active),
		.count (count_visible_active)
		);

	// assign curr_barrier_count    = $countones(curr_barrier_mask);

	assign curr_barrier_mask     = barrier_stall_mask[barrier_id][`NW-1:0];
	assign reached_barrier_limit = curr_barrier_count == (num_warps);

	assign wstall_this_cycle = wstall && (wstall_warp_num == warp_to_schedule); // Maybe bug

	assign total_barrier_stall = barrier_stall_mask[0] | barrier_stall_mask[1] | barrier_stall_mask[2] | barrier_stall_mask[3];
	// integer curr_b;
	// always @(*) begin
	// 	total_barrier_stall = 0;
	// 	for (curr_b = 0; curr_b < `NUM_BARRIERS; curr_b=curr_b+1)
	// 	begin
	// 		total_barrier_stall[`NW-1:0] = total_barrier_stall[`NW-1:0] | barrier_stall_mask[curr_b];
	// 	end
	// end


	assign update_visible_active = (count_visible_active < 1) && !(stall || wstall_this_cycle || hazard || is_join);

	wire[(1+32+`NT_M1):0] q1 = {1'b1, 32'b0                   , thread_masks[split_warp_num]};
	wire[(1+32+`NT_M1):0] q2 = {1'b0, split_save_pc           , split_later_mask};


	assign {join_fall, join_pc, join_tm} = d[join_warp_num];



	genvar curr_warp;
	for (curr_warp = 0; curr_warp < `NW; curr_warp = curr_warp + 1) begin
		wire correct_warp_s = (curr_warp == split_warp_num);
		wire correct_warp_j = (curr_warp == join_warp_num);

		wire push = (is_split && !dont_split) && correct_warp_s;
		wire pop  = is_join  && correct_warp_j;
		VX_generic_stack #(.WIDTH(1+32+`NT), .DEPTH($clog2(`NT)+1)) ipdom_stack(
			.clk  (clk),
			.reset(reset),
			.push (push),
			.pop  (pop),
			.d    (d[curr_warp]),
			.q1   (q1),
			.q2   (q2)
			);
	end

	// wire should_stall = stall || (jal && (warp_to_schedule == jal_warp_num)) || (branch_dir && (warp_to_schedule == branch_warp_num));

	wire should_jal = (jal && (warp_to_schedule == jal_warp_num));
	wire should_bra = (branch_dir && (warp_to_schedule == branch_warp_num));

	assign hazard = (should_jal || should_bra) && schedule;

	assign real_schedule = schedule && !warp_stalled[warp_to_schedule] && !total_barrier_stall[warp_to_schedule];

	assign global_stall = (stall || wstall_this_cycle || hazard || !real_schedule || is_join);

	assign scheduled_warp = !(wstall_this_cycle || hazard || !real_schedule || is_join);

	wire real_use_wspawn = use_wsapwn[warp_to_schedule];

	assign warp_pc     = real_use_wspawn ? use_wsapwn_pc : warp_pcs[warp_to_schedule];
	assign thread_mask = (global_stall) ? 0 : (real_use_wspawn ? `NT'b1 : thread_masks[warp_to_schedule]);
	assign warp_num    = warp_to_schedule;

	assign update_use_wspawn = use_wsapwn[warp_to_schedule] && !global_stall;

	assign new_pc = warp_pc + 4;


	assign use_active = (count_visible_active < 1) ? (warp_active & (~warp_stalled) & (~total_barrier_stall)) : visible_active;

	// Choosing a warp to schedule
	VX_priority_encoder choose_schedule(
		.valids(use_active),
		.index (warp_to_schedule),
		.found (schedule)
	);

	// always @(*) begin
	// 	$display("WarpPC: %h",warp_pc);
	// 	$display("real_schedule: %d, schedule: %d, warp_stalled: %d, warp_to_schedule: %d, total_barrier_stall: %d",real_schedule, schedule, warp_stalled[warp_to_schedule], warp_to_schedule,  total_barrier_stall[warp_to_schedule]);
	// end


	// Valid counter
	// assign num_active = $countones(visible_active);
	// VX_one_counter valid_counter(
	// 	.valids(visible_active),
	// 	.ones_found()
	// 	);


	wire ebreak = (warp_active == 0);
	assign out_ebreak = ebreak;

endmodule