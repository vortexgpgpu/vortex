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

`timescale 1ns/1ps


module rf2_32x128_wm1_tb (
	output [127 : 0] out_a_reg_data,
	output reg clk,
	output reg [4 : 0] rs1,
	output reg [127 : 0] write_bit_mask,
	output reg [4 : 0] rd,
	output reg [127 : 0] write_data,
	output reg cena,
	output reg cenb
);

	initial begin
		clk <= 1'b0;
		rs1 <= 5'b0;
		write_bit_mask <= {128{1'b1}};
		rd <= 5'b0;
		write_data <= 128'b0;
		cena <= 1'b1;
		cenb <= 1'b1;


		
		#100
		cenb <= 1'b0;
		write_bit_mask <= {{96{1'b1}}, {32{1'b0}}};
		rd <= 5'h0a;
		write_data <= 128'h0000_0002_0000_0002_0000_0002_0000_0002;
		#10
		cenb <= 1'b1;
		write_bit_mask <= {128{1'b1}};
		rd <= 5'b0;
		write_data <= 128'b0;
		
		#100
		cena <= 1'b0;
		rs1 <= 5'h0a;



	end

	always @(clk) #5 clk <= ~clk;



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
         .CENA(cena),
         .AA(rs1),
         .CLKB(clk),
         .CENB(cenb),
         .WENB(write_bit_mask),
         .AB(rd),
         .DB(write_data),
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

endmodule 

