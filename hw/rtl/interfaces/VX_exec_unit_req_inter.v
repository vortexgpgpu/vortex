`ifndef VX_EXE_UNIT_REQ_INTER
`define VX_EXE_UNIT_REQ_INTER

`include "../VX_define.vh"

interface VX_exec_unit_req_inter ();

	// Meta
	wire[`NUM_THREADS-1:0]      valid;
	wire[`NW_BITS-1:0]       	warp_num;
	wire[31:0]           		curr_PC;
	wire[31:0]           		PC_next;

	// Write Back Info
	wire[4:0]            rd;
	wire[1:0]            wb;

	// Data and alu op
	wire[`NUM_THREADS-1:0][31:0] a_reg_data;
	wire[`NUM_THREADS-1:0][31:0] b_reg_data;
	wire[4:0]            alu_op;
	wire[4:0]            rs1;
	wire[4:0]            rs2;
	wire                 rs2_src;
	wire[31:0]           itype_immed;
	wire[19:0]           upper_immed;

	// Branch type
	wire[2:0]            branch_type;

	// Jal info
	wire                 jalQual;
	wire                 jal;
	wire[31:0]           jal_offset;

	/* verilator lint_off UNUSED */
	wire                 ebreak;
	wire                 wspawn;
	/* verilator lint_on UNUSED */

	// CSR info
	wire                 is_csr;
	wire[11:0]           csr_address;
	wire                 csr_immed;
	wire[31:0]           csr_mask;

endinterface

`endif