// Copyright © 2019-2023
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "VX_define.vh"

//`define NUM_BANKS 8
//`define NUM_WORDS_PER_BLOCK 4

`define ARM_UD_MODEL

`timescale 1ns/1ps

import "DPI-C" load_file   = function void load_file(input string filename);

/*
import "DPI-C" ibus_driver = function void ibus_driver(input logic clk, input  int pc_addr,
	                                                   output int instruction);
	                                                   */

import "DPI-C" ibus_driver = function void ibus_driver( input logic clk,
														input int o_m_read_addr,
													    input int o_m_evict_addr,
													    input logic o_m_valid,
													    input reg[31:0] o_m_writedata[`ICACHE_BANKS - 1:0][`ICACHE_NUM_WORDS_PER_BLOCK-1:0],
													    input logic o_m_read_or_write,
													    input int cache_banks,
													    input int words_per_block,
													    // Rsp
													    output reg[31:0] i_m_readdata[`ICACHE_BANKS - 1:0][`ICACHE_NUM_WORDS_PER_BLOCK-1:0],
													    output logic        i_m_ready);

import "DPI-C" dbus_driver = function void dbus_driver( input logic clk,
														input int o_m_read_addr,
													    input int o_m_evict_addr,
													    input logic o_m_valid,
													    input reg[31:0] o_m_writedata[`DCACHE_BANKS - 1:0][`DCACHE_NUM_WORDS_PER_BLOCK-1:0],
													    input logic o_m_read_or_write,
													    input int cache_banks,
													    input int words_per_block,
													    // Rsp
													    output reg[31:0] i_m_readdata[`DCACHE_BANKS - 1:0][`DCACHE_NUM_WORDS_PER_BLOCK-1:0],
													    output logic        i_m_ready);


import "DPI-C" io_handler  = function void io_handler(input logic clk, input logic io_valid, input int io_data);

import "DPI-C" gracefulExit = function void gracefulExit(input int cycle_num);

module vortex_tb (
	
);

	int     cycle_num;

reg           clk;
reg           reset;
reg[31:0] icache_response_instruction;
reg[31:0] icache_request_pc_address;
// IO
reg        io_valid;
reg[31:0]  io_data;
// Req
	reg [31:0]  o_m_read_addr_d;
	reg [31:0]  o_m_evict_addr_d;
	reg         o_m_valid_d;
	reg [31:0]  o_m_writedata_d[`DCACHE_BANKS - 1:0][`DCACHE_NUM_WORDS_PER_BLOCK-1:0];
	reg         o_m_read_or_write_d;

	// Rsp
	reg [31:0]  i_m_readdata_d[`DCACHE_BANKS - 1:0][`DCACHE_NUM_WORDS_PER_BLOCK-1:0];
	reg         i_m_ready_d;

// Req
	reg [31:0]  o_m_read_addr_i;
	reg [31:0]  o_m_evict_addr_i;
	reg         o_m_valid_i;
	reg [31:0]  o_m_writedata_i[`ICACHE_BANKS - 1:0][`ICACHE_NUM_WORDS_PER_BLOCK-1:0];
	reg         o_m_read_or_write_i;

	// Rsp
	reg [31:0]  i_m_readdata_i[`ICACHE_BANKS - 1:0][`ICACHE_NUM_WORDS_PER_BLOCK-1:0];
	reg         i_m_ready_i;
    reg         out_ebreak;


	reg[31:0] hi;

	initial begin
		// $fdumpfile("vortex1.vcd");
		load_file("../../runtime/tests/simple/simple_main_if.hex");
		$dumpvars(0, vortex_tb);
		reset = 1;
		clk = 0;
		#5 reset = 1;
		clk = 1;
		cycle_num = 0;
	end

	Vortex vortex(
		.clk                          (clk),
		.reset                        (reset),
		.icache_response_instruction  (icache_response_instruction),
		.icache_request_pc_address    (icache_request_pc_address),
		.io_valid                     (io_valid),
		.io_data                      (io_data),
		.m_read_addr_d                (o_m_read_addr_d),
		.m_evict_addr_d               (o_m_evict_addr_d),
		.m_valid_d                    (o_m_valid_d),
		.m_writedata_d                (o_m_writedata_d),
		.m_read_or_write_d            (o_m_read_or_write_d),
		.m_readdata_d                 (i_m_readdata_d),
		.m_ready_d                    (i_m_ready_d),
		.m_read_addr                  (o_m_read_addr_i),
		.m_evict_addr                 (o_m_evict_addr_i),
		.m_valid                      (o_m_valid_i),
		.writedata                    (o_m_writedata_i),
		.m_read_or_write              (o_m_read_or_write_i),
		.m_readdata                   (i_m_readdata_i),
		.m_ready                      (i_m_ready_i),
		.ebreak                       (out_ebreak)
	);

	always @(negedge clk) begin
		ibus_driver(clk, o_m_read_addr_i, o_m_evict_addr_i, o_m_valid_i, o_m_writedata_i, o_m_read_or_write_i, `ICACHE_BANKS, `ICACHE_NUM_WORDS_PER_BLOCK, i_m_readdata_i, i_m_ready_i);
		dbus_driver(clk, o_m_read_addr_d, o_m_evict_addr_d, o_m_valid_d, o_m_writedata_d, o_m_read_or_write_d, `DCACHE_BANKS, `DCACHE_NUM_WORDS_PER_BLOCK, i_m_readdata_d, i_m_ready_d);
		io_handler (clk, io_valid, io_data);		
	end

	always @(posedge clk) begin
		if (out_ebreak) begin
			gracefulExit(cycle_num);
			#40 $finish;
		end
	end

	always @(posedge clk) begin
		cycle_num = cycle_num + 1;
	end

	always @(clk) begin
		if (reset) begin
			reset = 0;
			clk = 0;
		end

		#5 clk <= !clk;
	end

endmodule







