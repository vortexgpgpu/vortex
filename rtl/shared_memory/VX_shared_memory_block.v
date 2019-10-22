module VX_shared_memory_block (
	input clk,    // Clock
	input wire[6:0]       addr,
	input wire[3:0][31:0] wdata,
	input wire[1:0] we,
	input wire shm_write,

	output wire[3:0][31:0] data_out
	
);

	logic [3:0][31:0] shared_memory[127:0];

	//wire need_to_write = (|we);

	always @(posedge clk) begin
		if(shm_write) begin
			if (we == 2'b00) shared_memory[addr][0][31:0] <= wdata[0][31:0]; 
			if (we == 2'b01) shared_memory[addr][1][31:0] <= wdata[1][31:0]; 
			if (we == 2'b10) shared_memory[addr][2][31:0] <= wdata[2][31:0]; 
			if (we == 2'b11) shared_memory[addr][3][31:0] <= wdata[3][31:0]; 
		end
	end


	assign data_out = shm_write ? 0 : shared_memory[addr];




	// wire cena = 1;
	// wire cenb = shm_write;

	// wire[3:0][31:0] write_bit_mask;

	// assign write_bit_mask[0] = (we == 2'b00) ? 1 : {32{1'b0}};
	// assign write_bit_mask[1] = (we == 2'b01) ? 1 : {32{1'b0}};
	// assign write_bit_mask[2] = (we == 2'b10) ? 1 : {32{1'b0}};
	// assign write_bit_mask[3] = (we == 2'b11) ? 1 : {32{1'b0}};

	// // Using ASIC MEM
	// /* verilator lint_off PINCONNECTEMPTY */
 //   rf2_128x128_wm1 first_ram (
 //         .CENYA(),
 //         .AYA(),
 //         .CENYB(),
 //         .WENYB(),
 //         .AYB(),
 //         .QA(data_out),
 //         .SOA(),
 //         .SOB(),
 //         .CLKA(clk),
 //         .CENA(cena),
 //         .AA(addr),
 //         .CLKB(clk),
 //         .CENB(cenb),
 //         .WENB(write_bit_mask),
 //         .AB(addr),
 //         .DB(wdata),
 //         .EMAA(3'b011),
 //         .EMASA(1'b0),
 //         .EMAB(3'b011),
 //         .TENA(1'b1),
 //         .TCENA(1'b0),
 //         .TAA(5'b0),
 //         .TENB(1'b1),
 //         .TCENB(1'b0),
 //         .TWENB(128'b0),
 //         .TAB(5'b0),
 //         .TDB(128'b0),
 //         .RET1N(1'b1),
 //         .SIA(2'b0),
 //         .SEA(1'b0),
 //         .DFTRAMBYP(1'b0),
 //         .SIB(2'b0),
 //         .SEB(1'b0),
 //         .COLLDISN(1'b1)
 //   );
 //   /* verilator lint_on PINCONNECTEMPTY */

endmodule