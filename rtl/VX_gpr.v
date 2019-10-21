
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


	wire[`NT_M1:0][31:0] write_bit_mask;

	genvar curr_t;
	for (curr_t = 0; curr_t < `NT; curr_t=curr_t+1) begin
		wire local_write = write_enable & VX_writeback_inter.wb_valid[curr_t];
		assign write_bit_mask[curr_t] = {32{~local_write}};
	end

	wire going_to_write = write_enable & (|VX_writeback_inter.wb_valid);


	wire cenb    = !going_to_write;

	wire cena_1  = (VX_gpr_read.rs1 == 0);
	wire cena_2  = (VX_gpr_read.rs2 == 0);

	wire[127:0] write_bit_mask = {{32{~(VX_writeback_inter.wb_valid[3])}}, {32{~(VX_writeback_inter.wb_valid[2])}}, {32{~(VX_writeback_inter.wb_valid[1])}}, {32{~(VX_writeback_inter.wb_valid[0])}}};
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
         .CENA(cena_1),
         .AA(VX_gpr_read.rs1),
         .CLKB(clk),
         .CENB(cenb),
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
         .CENA(cena_2),
         .AA(VX_gpr_read.rs2),
         .CLKB(clk),
         .CENB(cenb),
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

endmodule
