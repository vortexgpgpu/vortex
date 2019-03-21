
// `include "vx_fetch.v"
// `include "vx_f_d_reg.v"

module vortex(
	input  wire       clk,
	input  wire       reset,
	input  wire[31:0] fe_instruction,
	output wire[31:0] curr_PC,
	output wire[31:0] de_instruction,
	output wire       fe_delay
	);

wire branch_dir;
assign branch_dir = 0;

wire freeze;
assign freeze = 0;

wire[31:0] branch_dest;
wire       branch_stall;
wire       fwd_stall;
wire       branch_stall_exe;
wire       jal;
wire[31:0] jal_dest;
wire       interrupt;
wire       debug;

assign branch_dest       = 32'h0;
assign branch_stall      = 1'b0;
assign fwd_stall         = 1'b0;
assign branch_stall_exe  = 1'b0;
assign jal               = 1'b0;
assign jal_dest          = 32'h0;
assign interrupt         = 1'b0;
assign debug             = 1'b0;


wire[31:0] f_instruction;
wire       f_delay; /* verilator lint_off UNUSED */
wire[31:0] f_curr_pc;
wire       f_valid;

assign curr_PC = f_curr_pc;
assign fe_delay = f_delay;

VX_fetch vx_fetch (
	.clk(clk),
	.reset(reset),
	.in_branch_dir(branch_dir),
	.in_freeze(freeze),
	.in_branch_dest(branch_dest),
	.in_branch_stall(branch_stall),
	.in_fwd_stall(fwd_stall),
	.in_branch_stall_exe(branch_stall_exe),
	.in_jal(jal),
	.in_jal_dest(jal_dest),
	.in_interrupt(interrupt),
	.in_debug(debug),
	.in_instruction(fe_instruction),

	.out_instruction(f_instruction),
	.out_delay(f_delay),
	.out_curr_PC(f_curr_pc),
	.out_valid(f_valid)
);


wire[31:0] d_curr_pc;
wire[31:0] d_instruction;
wire       d_valid;

VX_f_d_reg vx_f_d_reg (
	  .clk(clk),
	  .reset(reset),
	  .in_instruction(f_instruction),
	  .in_valid(f_valid),
	  .in_curr_PC(f_curr_pc),
	  .in_fwd_stall(fwd_stall),
	  .in_freeze(freeze),
	  .out_instruction(d_instruction),
	  .out_curr_PC(d_curr_pc),
	  .out_valid(d_valid)
	);

assign de_instruction = d_instruction;


endmodule // Vortex





