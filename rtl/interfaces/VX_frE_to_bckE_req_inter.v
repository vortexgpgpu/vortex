
`include "VX_define.v"

`ifndef VX_FrE_to_BE_INTER

`define VX_FrE_to_BE_INTER

interface VX_frE_to_bckE_req_inter ();

	wire[11:0]           csr_address;
	wire                 is_csr;
	wire[31:0]           csr_mask;
	wire[4:0]            rd;
	wire[4:0]            rs1;
	wire[4:0]            rs2;
	wire[`NT_M1:0][31:0] a_reg_data;
	wire[`NT_M1:0][31:0] b_reg_data;
	wire[4:0]            alu_op;
	wire[1:0]            wb;
	wire                 rs2_src;
	wire[31:0]           itype_immed;
	wire[2:0]            mem_read;
	wire[2:0]            mem_write;
	wire[2:0]            branch_type;
	wire[19:0]           upper_immed;
	wire[31:0]           curr_PC;
	wire                 jal;
	wire[31:0]           jal_offset;
	wire[31:0]           PC_next;
	wire[`NT_M1:0]       valid;
	wire[`NW_M1:0]       warp_num;

	// source-side view
	modport snk (
		input csr_address,
		input is_csr,
		input csr_mask,
		input rd,
		input rs1,
		input rs2,
		input a_reg_data,
		input b_reg_data,
		input alu_op,
		input wb,
		input rs2_src,
		input itype_immed,
		input mem_read,
		input mem_write,
		input branch_type,
		input upper_immed,
		input curr_PC,
		input jal,
		input jal_offset,
		input PC_next,
		input valid,
		input warp_num
	);


	// source-side view
	modport src (
		output csr_address,
		output is_csr,
		output csr_mask,
		output rd,
		output rs1,
		output rs2,
		output a_reg_data,
		output b_reg_data,
		output alu_op,
		output wb,
		output rs2_src,
		output itype_immed,
		output mem_read,
		output mem_write,
		output branch_type,
		output upper_immed,
		output curr_PC,
		output jal,
		output jal_offset,
		output PC_next,
		output valid,
		output warp_num
	);


endinterface


`endif