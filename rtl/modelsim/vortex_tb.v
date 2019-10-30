
`include "../VX_define.v"

`define NUMBER_BANKS 8
`define NUM_WORDS_PER_BLOCK 4

`define ARM_UD_MODEL

`timescale 1ns/1ps

import "DPI-C" load_file   = function void load_file(input string filename);

import "DPI-C" ibus_driver = function void ibus_driver(input logic clk, input  int pc_addr,
	                                                   output int instruction);

import "DPI-C" dbus_driver = function void dbus_driver( input logic clk,
														input int o_m_read_addr,
													    input int o_m_evict_addr,
													    input logic o_m_valid,
													    input reg[31:0] o_m_writedata[`NUMBER_BANKS - 1:0][`NUM_WORDS_PER_BLOCK-1:0],
													    input logic o_m_read_or_write,

													    // Rsp
													    output reg[31:0] i_m_readdata[`NUMBER_BANKS - 1:0][`NUM_WORDS_PER_BLOCK-1:0],
													    output logic        i_m_ready);


import "DPI-C" io_handler  = function void io_handler(input logic clk, input logic io_valid, input int io_data);

import "DPI-C" gracefulExit = function void gracefulExit();

module vortex_tb (
	
);

	reg[31:0]     cycle_num;

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
    reg [31:0]  i_m_readdata[8 - 1:0][4-1:0];
    reg         i_m_ready;
	reg         out_ebreak;


	reg[31:0] hi;

	integer temp;

	initial begin
		// $fdumpfile("vortex1.vcd");
		load_file("../../kernel/vortex_test.hex");
		$dumpvars(0, vortex_tb);
		reset = 1;
		clk = 0;
		#5 reset = 1;
		clk = 1;
		cycle_num = 0;
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

	always @(*) begin
		ibus_driver(clk, icache_request_pc_address, icache_response_instruction);
		dbus_driver(clk, o_m_read_addr, o_m_evict_addr, o_m_valid, o_m_writedata, o_m_read_or_write, i_m_readdata, i_m_ready);
		io_handler (clk, io_valid, io_data);
		
	end

	always @(clk, posedge reset) begin
		if (reset) begin
			reset = 0;
			clk = 0;
		end

		#5 clk <= ~clk;

		if (out_ebreak) begin
			gracefulExit();
			#20 $finish;
		end

	end

endmodule







