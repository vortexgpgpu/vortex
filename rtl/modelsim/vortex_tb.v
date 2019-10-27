
`include "../VX_define.v"

`define NUMBER_BANKS 8
`define NUM_WORDS_PER_BLOCK 4

`timescale 1ns/1ps

import "DPI-C" load_file   = function void load_file(input string filename);

import "DPI-C" ibus_driver = function void ibus_driver(input  int pc_addr,
	                                                   output int instruction);

import "DPI-C" dbus_driver = function void dbus_driver( input int o_m_read_addr,
													    input int o_m_evict_addr,
													    input reg o_m_valid,
													    input reg [31:0] o_m_writedata[`NUMBER_BANKS - 1:0][`NUM_WORDS_PER_BLOCK-1:0],
													    input reg o_m_read_or_write,

													    // Rsp
													    output reg [31:0] i_m_readdata[`NUMBER_BANKS - 1:0][`NUM_WORDS_PER_BLOCK-1:0],
													    output reg        i_m_ready);

import "DPI-C" io_handler  = function void io_handler(input reg io_valid, input int io_data);


module vortex_tb (
	
);

	reg           clk;
	reg           reset;
	reg[31:0] icache_response_instruction;
	reg[31:0] icache_request_pc_address;
	// IO
	reg        io_valid;
	reg[31:0]  io_data;
	// Req
    reg [31:0]  o_m_read_addr;
    reg [31:0]  o_m_evict_addr;
    reg         o_m_valid;
    reg [31:0]  o_m_writedata[8 - 1:0][4-1:0];
    reg         o_m_read_or_write;

    // Rsp
    reg [31:0] i_m_readdata[8 - 1:0][4-1:0];
    reg        i_m_ready;
	reg        out_ebreak;


	reg[31:0] hi;

	integer temp;
	integer num_cycles;

	initial begin
		// $fdumpfile("vortex1.vcd");
		load_file("../../kernel/vortex_test.hex");
		$dumpvars(0, vortex_tb);
		reset = 1;
		clk = 0;
		#5 reset = 1;
		clk = 1;
		num_cycles = 0;
	end

	Vortex vortex(
		.clk                        (clk),
		.reset                      (reset),
		.icache_response_instruction(icache_response_instruction),
		.icache_request_pc_address  (icache_request_pc_address),
		.io_valid                   (io_valid),
		.io_data                    (io_data),
		.o_m_read_addr              (o_m_read_addr),
		.o_m_evict_addr             (o_m_evict_addr),
		.o_m_valid                  (o_m_valid),
		.o_m_writedata              (o_m_writedata),
		.o_m_read_or_write          (o_m_read_or_write),
		.i_m_readdata               (i_m_readdata),
		.i_m_ready                  (i_m_ready),
		.out_ebreak                 (out_ebreak)
		);


	always @(clk, posedge reset) begin
		// $display("FROM ALWAYS");
		// $display("num_cycles: %d",num_cycles);
		num_cycles = num_cycles + 1;
		if (num_cycles == 1000) begin
			// $dumpall;
			// $dumpflush;
			// $finish;
		end
		// if (num_cycles == 1000) $stop;
		if (reset) begin
			reset = 0;
			clk = 0;
		end

		if (clk == 0) begin
			ibus_driver(icache_request_pc_address, icache_response_instruction);
			dbus_driver(o_m_read_addr, o_m_evict_addr, o_m_valid, o_m_writedata, o_m_read_or_write, i_m_readdata, i_m_ready);
			io_handler(io_valid, io_data);
		end

		// $display("clk: %d, out_ebreak: %d",clk, out_ebreak);
		#5 clk <= ~clk;
		if (out_ebreak) $finish;
	end

endmodule







