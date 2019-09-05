
`include "VX_define.v"

`ifndef BUSES

`define BUSES

typedef struct packed
{
  // logic       valid;
  logic[31:0] pc_address;
} icache_request_t;

typedef struct packed
{
	// logic       ready;
	// logic       stall;
	logic[31:0] instruction;
} icache_response_t;

// typedef struct packed
// {
// 	logic[31:0]       instruction;
// 	logic[31:0]       inst_pc;
// 	logic[`NW_M1:0]   warp_num;
// 	logic[`NT_M1:0]   valid;

// } fe_inst_meta_de_t;


`endif


	// wire flush = 1'b0;
	// wire stall = in_fwd_stall == 1'b1 || in_freeze == 1'b1 || in_clone_stall;


	// fe_inst_meta_de_t meta_out;

	// VX_generic_register #(.N(72)) f_d_reg 
	// (
	// 	.clk  (clk),
	// 	.reset(reset),
	// 	.stall(stall),
	// 	.flush(flush),
	// 	.in   ({fe_inst_meta_de}),
	// 	.out  ({meta_out})
	// );

	// genvar index;
	// generate
	// 	for (index = 0; index <= `NT_M1; index = index + 1) assign out_valid[index] =  meta_out.valid[index];
	// endgenerate
	// // assign out_valid[`NT_M1:0] = meta_out.valid[`NT_M1:0];


	// assign out_instruction     = meta_out.instruction;
	// assign out_curr_PC         = meta_out.inst_pc;
	// assign out_warp_num        = meta_out.warp_num;

	// always @(*) begin
	// 	$display("Inst: %x, PC: %x, Valid: %x, warpNum: %x", fe_inst_meta_de.instruction, fe_inst_meta_de.inst_pc, fe_inst_meta_de.valid, fe_inst_meta_de.warp_num);
	// end