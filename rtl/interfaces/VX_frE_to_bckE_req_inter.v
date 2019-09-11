
`include "../VX_define.v"

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


endinterface


`endif