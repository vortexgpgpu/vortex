module VX_shared_memory_block
#(
		parameter SMB_SIZE               = 4096, // Bytes
		parameter SMB_BYTES_PER_READ     = 16,
		parameter SMB_WORDS_PER_READ     = 4,
		parameter SMB_LOG_WORDS_PER_READ = 2,
		parameter SMB_HEIGHT             = 128, // Bytes
		parameter BITS_PER_BANK          = 3
)
(
	input wire             clk,    // Clock
	input wire             reset,
	//input wire[6:0]        addr,
	//input wire[3:0][31:0]  wdata,
	//input wire[1:0]        we,
	//input wire             shm_write,

	//output wire[3:0][31:0] data_out
	input wire[$clog2(SMB_HEIGHT) - 1:0]        addr,
	input wire[SMB_WORDS_PER_READ-1:0][31:0]    wdata,
	input wire[SMB_LOG_WORDS_PER_READ-1:0]      we,
	input wire                                  shm_write,

	output wire[SMB_WORDS_PER_READ-1:0][31:0]   data_out

);


	`ifndef SYN

		reg[SMB_WORDS_PER_READ-1:0][3:0][7:0] shared_memory[SMB_HEIGHT-1:0];
		
		wire [$clog2(SMB_HEIGHT) - 1:0]reg_addr;

		//wire need_to_write = (|we);
		integer curr_ind;
		// initial begin
		// 		for (curr_ind = 0; curr_ind < SMB_HEIGHT; curr_ind = curr_ind + 1)
		// 		begin
		// 			shared_memory[curr_ind] = 0;
		// 		end
		// end
		always @(posedge clk, posedge reset) begin
			if (reset) begin
				//for (curr_ind = 0; curr_ind < 128; curr_ind = curr_ind + 1)
			end else if(shm_write) begin
				if (we == 2'b00) shared_memory[reg_addr][0] <= wdata[0];
				if (we == 2'b01) shared_memory[reg_addr][1] <= wdata[1];
				if (we == 2'b10) shared_memory[reg_addr][2] <= wdata[2];
				if (we == 2'b11) shared_memory[reg_addr][3] <= wdata[3];
			end
		end
       	
		assign reg_addr = addr;
		// always @(posedge clk)
 		// 	reg_addr <= addr;


		assign data_out = shm_write ? 0 : shared_memory[reg_addr];

	`else

		wire cena = 0;
		wire cenb = !shm_write;

		wire[3:0][31:0] write_bit_mask;

		//assign write_bit_mask[0] = (we == 2'b00) ? {32{1'b1}} : {32{1'b0}};
		//assign write_bit_mask[1] = (we == 2'b01) ? {32{1'b1}} : {32{1'b0}};
		//assign write_bit_mask[2] = (we == 2'b10) ? {32{1'b1}} : {32{1'b0}};
		//assign write_bit_mask[3] = (we == 2'b11) ? {32{1'b1}} : {32{1'b0}};
		genvar curr_word;
		for (curr_word = 0; curr_word < SMB_WORDS_PER_READ; curr_word = curr_word + 1)
		begin
			assign write_bit_mask[curr_word] = (we == curr_word) ? 1 : {32{1'b0}};
		end

		// Using ASIC MEM
		/* verilator lint_off PINCONNECTEMPTY */
	   rf2_128x128_wm1 first_ram (
	         .CENYA(),
	         .AYA(),
	         .CENYB(),
	         .WENYB(),
	         .AYB(),
	         .QA(data_out),
	         .SOA(),
	         .SOB(),
	         .CLKA(clk),
	         .CENA(cena),
	         .AA(addr),
	         .CLKB(clk),
	         .CENB(cenb),
	         .WENB(write_bit_mask),
	         .AB(addr),
	         .DB(wdata),
	         .EMAA(3'b011),
	         .EMASA(1'b0),
	         .EMAB(3'b011),
	         .TENA(1'b1),
	         .TCENA(1'b0),
	         .TAA(7'b0),
	         .TENB(1'b1),
	         .TCENB(1'b0),
	         .TWENB(128'b0),
	         .TAB(7'b0),
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


 	`endif

endmodule
