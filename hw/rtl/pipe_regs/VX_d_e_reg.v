`include "../VX_define.vh"

module VX_d_e_reg (
	input wire               clk,
	input wire               reset,
	input wire               in_branch_stall,
	input wire               in_freeze,
	VX_frE_to_bckE_req_inter vx_frE_to_bckE_req,
	VX_frE_to_bckE_req_inter vx_bckE_req
);

	wire stall = in_freeze;
	wire flush = (in_branch_stall == `STALL);

	VX_generic_register #(
		.N(233 + `NW_BITS-1 + 1 + `NUM_THREADS)
	) d_e_reg (
		.clk   (clk),
		.reset (reset),
		.stall (stall),
		.flush (flush),
		.in   ({vx_frE_to_bckE_req.csr_address, vx_frE_to_bckE_req.jalQual, vx_frE_to_bckE_req.ebreak, vx_frE_to_bckE_req.is_csr, vx_frE_to_bckE_req.csr_immed, vx_frE_to_bckE_req.csr_mask, vx_frE_to_bckE_req.rd, vx_frE_to_bckE_req.rs1, vx_frE_to_bckE_req.rs2, vx_frE_to_bckE_req.alu_op, vx_frE_to_bckE_req.wb, vx_frE_to_bckE_req.rs2_src, vx_frE_to_bckE_req.itype_immed, vx_frE_to_bckE_req.mem_read, vx_frE_to_bckE_req.mem_write, vx_frE_to_bckE_req.branch_type, vx_frE_to_bckE_req.upper_immed, vx_frE_to_bckE_req.curr_PC, vx_frE_to_bckE_req.jal, vx_frE_to_bckE_req.jal_offset, vx_frE_to_bckE_req.PC_next, vx_frE_to_bckE_req.valid, vx_frE_to_bckE_req.warp_num, vx_frE_to_bckE_req.is_wspawn, vx_frE_to_bckE_req.is_tmc, vx_frE_to_bckE_req.is_split, vx_frE_to_bckE_req.is_barrier}),
		.out  ({vx_bckE_req.csr_address       , vx_bckE_req.jalQual       , vx_bckE_req.ebreak       ,vx_bckE_req.is_csr       , vx_bckE_req.csr_immed       , vx_bckE_req.csr_mask       , vx_bckE_req.rd       , vx_bckE_req.rs1       , vx_bckE_req.rs2       , vx_bckE_req.alu_op       , vx_bckE_req.wb       , vx_bckE_req.rs2_src       , vx_bckE_req.itype_immed       , vx_bckE_req.mem_read       , vx_bckE_req.mem_write       , vx_bckE_req.branch_type       , vx_bckE_req.upper_immed       , vx_bckE_req.curr_PC       , vx_bckE_req.jal       , vx_bckE_req.jal_offset       , vx_bckE_req.PC_next       , vx_bckE_req.valid       , vx_bckE_req.warp_num        , vx_bckE_req.is_wspawn       , vx_bckE_req.is_tmc       , vx_bckE_req.is_split       , vx_bckE_req.is_barrier       })
	);

endmodule




