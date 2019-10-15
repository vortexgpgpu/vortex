
`include "VX_define.v"

module VX_gpr (
	input wire                  clk,
	input wire                  valid_write_request,
	VX_gpr_read_inter           VX_gpr_read,
	VX_wb_inter                 VX_writeback_inter,

	output reg[`NT_M1:0][31:0] out_a_reg_data,
	output reg[`NT_M1:0][31:0] out_b_reg_data
);

	wire write_enable;

	assign write_enable = valid_write_request && ((VX_writeback_inter.wb != 0) && (VX_writeback_inter.rd != 5'h0));
	
// <<<<<<< HEAD


			// always @(*) begin
			// 	if(write_enable) $display("Writing to %d: %d = %h",VX_writeback_inter.wb_warp_num, VX_writeback_inter.rd, VX_writeback_inter.write_data[0][31:0]);
			// end

	// byte_enabled_simple_dual_port_ram first_ram(
	// 	.we    (write_enable),
	// 	.clk   (clk),
	// 	.waddr (VX_writeback_inter.rd),
	// 	.raddr1(VX_gpr_read.rs1),
	// 	.raddr2(VX_gpr_read.rs2),
	// 	.be    (VX_writeback_inter.wb_valid),
	// 	.wdata (VX_writeback_inter.write_data),
	// 	.q1    (out_a_reg_data),
	// 	.q2    (out_b_reg_data)
	// );

// =======
	// byte_enabled_simple_dual_port_ram first_ram(
		// .we    (write_enable),
		// .clk   (clk),
		// .waddr (VX_writeback_inter.rd),
		// .raddr1(VX_gpr_read.rs1),
		// .be    (VX_writeback_inter.wb_valid),
		// .wdata (VX_writeback_inter.write_data),
		// .q1    (out_a_reg_data)
	// );

	// byte_enabled_simple_dual_port_ram first_ram(
		// .we    (write_enable),
		// .clk   (clk),
		// .waddr (VX_writeback_inter.rd),
		// .raddr1(VX_gpr_read.rs2),
		// .be    (VX_writeback_inter.wb_valid),
		// .wdata (VX_writeback_inter.write_data),
		// .q1    (out_b_reg_data)
	// );

	wire[127:0] write_bit_mask = {{32{~(VX_writeback_inter.wb_valid[3])}}, {32{~(VX_writeback_inter.wb_valid[2])}}, {32{~(VX_writeback_inter.wb_valid[1])}}, {32{~(VX_writeback_inter.wb_valid[0])}}};

	// Port A is a read port, Port B is a write port

	/* verilator lint_off PINCONNECTEMPTY */
   rf2_32x128_wm1 first_ram (
         .CENYA(),
         .AYA(),
         .CENYB(),
         .WENYB(),
         .AYB(),
         .QA(out_a_reg_data),
         .SOA(),
         .SOB(),
         .CLKA(clk),
         .CENA(1'b0),
         .AA(VX_gpr_read.rs1),
         .CLKB(clk),
         .CENB(1'b0),
         .WENB(write_bit_mask),
         .AB(VX_writeback_inter.rd),
         .DB(VX_writeback_inter.write_data),
         .EMAA(3'b011),
         .EMASA(1'b0),
         .EMAB(3'b011),
         .TENA(1'b1),
         .TCENA(1'b0),
         .TAA(5'b0),
         .TENB(1'b1),
         .TCENB(1'b0),
         .TWENB(128'b0),
         .TAB(5'b0),
         .TDB(128'b0),
         .RET1N(1'b1),
         .SIA(2'b0),
         .SEA(1'b0),
         .DFTRAMBYP(1'b0),
         .SIB(2'b0),
         .SEB(1'b0),
         .COLLDISN(1'b1)
   );
   /* verilator lint_on PINCONNECTEMPTY */

   /* verilator lint_off PINCONNECTEMPTY */
   rf2_32x128_wm1 second_ram (
         .CENYA(),
         .AYA(),
         .CENYB(),
         .WENYB(),
         .AYB(),
         .QA(out_b_reg_data),
         .SOA(),
         .SOB(),
         .CLKA(clk),
         .CENA(1'b0),
         .AA(VX_gpr_read.rs2),
         .CLKB(clk),
         .CENB(1'b0),
         .WENB(write_bit_mask),
         .AB(VX_writeback_inter.rd),
         .DB(VX_writeback_inter.write_data),
         .EMAA(3'b011),
         .EMASA(1'b0),
         .EMAB(3'b011),
         .TENA(1'b1),
         .TCENA(1'b0),
         .TAA(5'b0),
         .TENB(1'b1),
         .TCENB(1'b0),
         .TWENB(128'b0),
         .TAB(5'b0),
         .TDB(128'b0),
         .RET1N(1'b1),
         .SIA(2'b0),
         .SEA(1'b0),
         .DFTRAMBYP(1'b0),
         .SIB(2'b0),
         .SEB(1'b0),
         .COLLDISN(1'b1)
   );
   /* verilator lint_on PINCONNECTEMPTY */
// >>>>>>> 5680b997b599ce2900997cab976681fe3881e880






	 // // USING RAM blocks
	 // // First RAM
	 // byte_enabled_simple_dual_port_ram first_ram(
	 // 	.we   (write_enable),
	 // 	.clk  (clk),
	 // 	.waddr(VX_writeback_inter.rd),
	 // 	.raddr(VX_gpr_read.rs1),
	 // 	.be   (VX_writeback_inter.wb_valid),
	 // 	.wdata(VX_writeback_inter.write_data),
	 // 	.q    (out_a_reg_data)
	 // 	);

	 // // Second RAM block
	 // byte_enabled_simple_dual_port_ram second_ram(
	 // 	.we   (write_enable),
	 // 	.clk  (clk),
	 // 	.waddr(VX_writeback_inter.rd),
	 // 	.raddr(VX_gpr_read.rs2),
	 // 	.be   (VX_writeback_inter.wb_valid),
	 // 	.wdata(VX_writeback_inter.write_data),
	 // 	.q    (out_b_reg_data)
	 // 	);



	// logic[`NT_M1:0][31:0] gpr[31:0]; // gpr[register_number][thread_number][data_bits]

	// wire write_enable;

	// assign write_enable = valid_write_request && ((VX_writeback_inter.wb != 0) && (VX_writeback_inter.rd != 5'h0));
	// assign read_enable  = valid_request;

		// // Using Registers
		//  integer thread_index;
		//  always_ff@(posedge clk)
		//  begin
		//  	if (write_enable) begin
		//  		for (thread_index = 0; thread_index <= `NT_M1; thread_index = thread_index + 1) begin
		//  			if (VX_writeback_inter.wb_valid[thread_index]) begin
		//  				gpr[VX_writeback_inter.rd][thread_index] <= VX_writeback_inter.write_data[thread_index];
		//  			end
		//  		end
		//  	end
	 // 		out_a_reg_data <= gpr[VX_gpr_read.rs1];
		// 	out_b_reg_data <= gpr[VX_gpr_read.rs2];
		//  end


endmodule
