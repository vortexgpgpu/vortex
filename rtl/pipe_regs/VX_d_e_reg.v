

`include "VX_define.v"

module VX_d_e_reg (
		input wire               clk,
		input wire               reset,
		input wire               in_fwd_stall,
		input wire               in_branch_stall,
		input wire               in_freeze,
		input wire               in_clone_stall,
		VX_frE_to_bckE_req_inter VX_frE_to_bckE_req,


		VX_frE_to_bckE_req_inter VX_bckE_req
	);


		wire stall = in_freeze;
		wire flush = (in_fwd_stall == `STALL) || (in_branch_stall == `STALL) || (in_clone_stall == `STALL);


	VX_generic_register #(.N(490)) d_e_reg 
	(
		.clk  (clk),
		.reset(reset),
		.stall(stall),
		.flush(flush),
		.in   ({VX_frE_to_bckE_req.csr_address, VX_frE_to_bckE_req.is_csr, VX_frE_to_bckE_req.csr_mask, VX_frE_to_bckE_req.rd, VX_frE_to_bckE_req.rs1, VX_frE_to_bckE_req.rs2, VX_frE_to_bckE_req.a_reg_data, VX_frE_to_bckE_req.b_reg_data, VX_frE_to_bckE_req.alu_op, VX_frE_to_bckE_req.wb, VX_frE_to_bckE_req.rs2_src, VX_frE_to_bckE_req.itype_immed, VX_frE_to_bckE_req.mem_read, VX_frE_to_bckE_req.mem_write, VX_frE_to_bckE_req.branch_type, VX_frE_to_bckE_req.upper_immed, VX_frE_to_bckE_req.curr_PC, VX_frE_to_bckE_req.jal, VX_frE_to_bckE_req.jal_offset, VX_frE_to_bckE_req.PC_next, VX_frE_to_bckE_req.valid, VX_frE_to_bckE_req.warp_num}),
		.out  ({VX_bckE_req.csr_address       , VX_bckE_req.is_csr       , VX_bckE_req.csr_mask       , VX_bckE_req.rd       , VX_bckE_req.rs1       , VX_bckE_req.rs2       , VX_bckE_req.a_reg_data       , VX_bckE_req.b_reg_data       , VX_bckE_req.alu_op       , VX_bckE_req.wb       , VX_bckE_req.rs2_src       , VX_bckE_req.itype_immed       , VX_bckE_req.mem_read       , VX_bckE_req.mem_write       , VX_bckE_req.branch_type       , VX_bckE_req.upper_immed       , VX_bckE_req.curr_PC       , VX_bckE_req.jal       , VX_bckE_req.jal_offset       , VX_bckE_req.PC_next       , VX_bckE_req.valid       , VX_bckE_req.warp_num})
	);


	// wire[`NT_M1:0][31:0] temp_out_a_reg_data;
	// wire[`NT_M1:0][31:0] temp_out_b_reg_data;
	// wire[`NT_M1:0]       temp_out_valid;


	// genvar index;
	// for (index = 0; index <= `NT_M1; index = index + 1) begin

	// 	assign out_valid[index]      = temp_out_valid[index];
	// 	assign out_a_reg_data[index] = temp_out_a_reg_data[index];
	// 	assign out_b_reg_data[index] = temp_out_b_reg_data[index];

	// end


		// reg[4:0]  rd;
		// reg[4:0]  rs1;
		// reg[4:0]  rs2;
		// reg[31:0] a_reg_data[`NT_M1:0];
		// reg[31:0] b_reg_data[`NT_M1:0];
		// reg[4:0]  alu_op;
		// reg[1:0]  wb;
		// reg[31:0] PC_next_out;
		// reg       rs2_src;
		// reg[31:0] itype_immed;
		// reg[2:0]  mem_read;
		// reg[2:0]  mem_write;
		// reg[2:0]  branch_type;
		// reg[19:0] upper_immed;
		// reg[11:0] csr_address;
		// reg       is_csr;
		// reg[31:0] csr_mask;
		// reg[31:0] curr_PC;
		// reg       jal;
		// reg[31:0] jal_offset;
		// reg       valid[`NT_M1:0];

		// reg[31:0] reg_data_z[`NT_M1:0];
		// reg       valid_z[`NT_M1:0];

		// reg[`NW_M1:0] warp_num;

		// integer ini_reg;
		// initial begin
		// 	rd          = 0;
		// 	rs1         = 0;
		// 	for (ini_reg = 0; ini_reg < `NT; ini_reg = ini_reg + 1)
		// 	begin
		// 		a_reg_data[ini_reg]   = 0;
		// 		b_reg_data[ini_reg]   = 0;
		// 		reg_data_z[ini_reg]   = 0;
		// 		valid[ini_reg]        = 0;
		// 		valid_z[ini_reg]      = 0;
		// 	end
		// 	rs2         = 0;
		// 	alu_op      = 0;
		// 	wb          = `NO_WB;
		// 	PC_next_out = 0;
		// 	rs2_src     = 0;
		// 	itype_immed = 0;
		// 	mem_read    = `NO_MEM_READ;
		// 	mem_write   = `NO_MEM_WRITE;
		// 	branch_type = `NO_BRANCH;
		// 	upper_immed = 0;
		// 	csr_address = 0;
		// 	is_csr      = 0;
		// 	csr_mask    = 0;
		// 	curr_PC     = 0;
		// 	jal         = `NO_JUMP;
		// 	jal_offset  = 0;
		// 	warp_num    = 0;
		// end

		// wire stalling;

		// assign stalling = (in_fwd_stall == `STALL) || (in_branch_stall == `STALL) || (in_clone_stall == `STALL);

// Freeze stall
// Stalling flush

		// assign out_rd          = rd;
		// assign out_rs1         = rs1;
		// assign out_rs2         = rs2;
		// assign out_a_reg_data  = a_reg_data;
		// assign out_b_reg_data  = b_reg_data;
		// assign out_alu_op      = alu_op;
		// assign out_wb          = wb;
		// assign out_PC_next     = PC_next_out;
		// assign out_rs2_src     = rs2_src;
		// assign out_itype_immed = itype_immed;
		// assign out_mem_read    = mem_read;
		// assign out_mem_write   = mem_write;
		// assign out_branch_type = branch_type;
		// assign out_upper_immed = upper_immed;
		// assign out_csr_address = csr_address;
		// assign out_is_csr      = is_csr;
		// assign out_csr_mask    = csr_mask;
		// assign out_jal         = jal;
		// assign out_jal_offset  = jal_offset;
		// assign out_curr_PC     = curr_PC;
		// assign out_valid       = valid;
		// assign out_warp_num    = warp_num;


		// always @(posedge clk) begin
		// 	if (in_freeze == 1'h0) begin
		// 		rd          <= stalling ? 5'h0         : in_rd;
		// 		rs1         <= stalling ? 5'h0         : in_rs1;
		// 		rs2         <= stalling ? 5'h0         : in_rs2;
		// 		a_reg_data  <= stalling ? reg_data_z   : in_a_reg_data;
		// 		b_reg_data  <= stalling ? reg_data_z   : in_b_reg_data;
		// 		alu_op      <= stalling ? `NO_ALU      : in_alu_op;
		// 		wb          <= stalling ? `NO_WB       : in_wb;
		// 		PC_next_out <= stalling ? 32'h0        : in_PC_next;
		// 		rs2_src     <= stalling ? `RS2_REG     : in_rs2_src;
		// 		itype_immed <= stalling ? 32'hdeadbeef : in_itype_immed;
		// 		mem_read    <= stalling ? `NO_MEM_READ : in_mem_read;
		// 		mem_write   <= stalling ? `NO_MEM_WRITE: in_mem_write;
		// 		branch_type <= stalling ? `NO_BRANCH   : in_branch_type;
		// 		upper_immed <= stalling ? 20'h0        : in_upper_immed;
		// 		csr_address <= stalling ? 12'h0        : in_csr_address;
		// 		is_csr      <= stalling ? 1'h0         : in_is_csr;
		// 		csr_mask    <= stalling ? 32'h0        : in_csr_mask;
		// 		jal         <= stalling ? `NO_JUMP     : in_jal;
		// 		jal_offset  <= stalling ? 32'h0        : in_jal_offset;
		// 		curr_PC     <= stalling ? 32'h0        : in_curr_PC;
		// 		valid       <= stalling ? valid_z      : in_valid;
		// 		warp_num    <= stalling ? 0            : in_warp_num;
		// 	end
		// end

endmodule




