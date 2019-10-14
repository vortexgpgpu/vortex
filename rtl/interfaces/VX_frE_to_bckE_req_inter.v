
`include "../VX_define.v"

`ifndef VX_FrE_to_BE_INTER

`define VX_FrE_to_BE_INTER

interface VX_frE_to_bckE_req_inter ();

	wire[11:0]           csr_address;
	wire                 is_csr;
	/* verilator lint_off UNUSED */
	wire                 csr_immed;
	/* verilator lint_on UNUSED */
	wire[31:0]           csr_mask;
	wire[4:0]            rd;
	wire[4:0]            rs1;
	wire[4:0]            rs2;
	wire[4:0]            alu_op;
	wire[1:0]            wb;
	wire                 rs2_src;
	wire[31:0]           itype_immed;
	wire[2:0]            mem_read;
	wire[2:0]            mem_write;
	wire[2:0]            branch_type;
	wire[19:0]           upper_immed;
	wire[31:0]           curr_PC;
	/* verilator lint_off UNUSED */
	wire                 ebreak;
	wire                 wspawn;
	/* verilator lint_on UNUSED */
	wire                 jalQual;
	wire                 jal;
	wire[31:0]           jal_offset;
	wire[31:0]           PC_next;
	wire[`NT_M1:0]       valid;
	wire[`NW_M1:0]       warp_num;


endinterface


`endif