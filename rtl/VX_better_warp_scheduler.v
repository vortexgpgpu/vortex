`include "VX_define.v"



typedef struct packed
{
  logic[31:0]     pc;
  logic[`NT_M1:0] thread_mask;
} warp_meta_t;


typedef struct packed
{
	logic[`NW-1:0] valid;
	logic[`NW-1:0] visible;
	logic[`NW-1:0] stalled;
	warp_meta_t[`NW-1:0] warp_data;

} warps_meta_t;


module VX_better_warp_scheduler (
	input wire           clk,    // Clock
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


	warps_meta_t warps_meta;


	initial begin
		warps_meta.valid[0]                 = 1;
		warps_meta.warp_data[0].thread_mask = 1;
	end


	always @(posedge clk) begin
		$display("JAL %d DI %h",jal, jal_dest);
		if (external_stall) begin


			// Wsapwning warps
			if (wspawn && found_wspawn) begin
				warps_meta.warp_data[warp_to_wsapwn].pc           <= wsapwn_pc;
				warps_meta.warp_data[warp_to_wsapwn].thread_mask  <= 1;
				warps_meta.valid[warp_to_wsapwn]                  <= 1;
			end
			// Halting warps
			if (whalt) begin
				warps_meta.valid[whalt_warp_num]   <= 0;
				warps_meta.visible[whalt_warp_num] <= 0;
			end

			// Changing thread masks
			if (ctm) begin
				warps_meta.warp_data[ctm_warp_num].thread_mask <= ctm_mask;
			end

			// Stalling the scheduling of warps
			if (wstall) begin
				warps_meta.stalled[wstall_warp_num] <= 1;
				warps_meta.visible[wstall_warp_num] <= 0;
			end
			// Jal
			if (jal) begin
				$display("UPDATING PC JAL: %h", jal_dest);
				warps_meta.warp_data[jal_warp_num].pc <= jal_dest;
				warps_meta.stalled[jal_warp_num]      <= 0;
			end

			// Branch
			if (branch_valid) begin
				if (branch_dir) warps_meta.warp_data[branch_warp_num].pc <= branch_dest;
				warps_meta.stalled[branch_warp_num]                      <= 0;
			end


		end else if (real_schedule) begin


			// Refilling active warps
			if (warps_meta.visible == 0) begin
				warps_meta.visible <= warps_meta.valid & (~warps_meta.stalled);
			end

			// Don't change state if stall
			warps_meta.visible[warp_to_schedule]      <= 0;
			warps_meta.warp_data[warp_to_schedule].pc <= warp_pc;


		end else begin

			// Refilling active warps
			if (warps_meta.visible == 0) begin
				warps_meta.visible <= warps_meta.valid & (~warps_meta.stalled);
			end

		end
	end


	wire external_stall = stall || wspawn || ctm || whalt || wstall || jal || branch_valid;

	wire real_schedule = schedule && !warps_meta.stalled[warp_to_schedule];


	assign warp_pc     = warps_meta.warp_data[warp_to_schedule].pc + 4;
	assign thread_mask = (external_stall || !real_schedule) ? 0 : warps_meta.warp_data[warp_to_schedule].thread_mask;
	assign warp_num    = warp_to_schedule;

	// Choosing a warp to schedule
	wire[`NW_M1:0] warp_to_schedule;
	wire           schedule;
	VX_priority_encoder choose_schedule(
		.valids(warps_meta.visible),
		.index (warp_to_schedule),
		.found (schedule)
	);

	// Choosing a warp to wsapwn
	wire[`NW_M1:0] warp_to_wsapwn;
	wire           found_wspawn;
	VX_priority_encoder choose_wsapwn(
		.valids(~warps_meta.valid),
		.index (warp_to_wsapwn),
		.found (found_wspawn)
	);

	assign out_ebreak = (warps_meta.valid == 0);

endmodule





