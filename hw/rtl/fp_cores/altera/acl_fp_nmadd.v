// (C) 1992-2014 Altera Corporation. All rights reserved.                         
// Your use of Altera Corporation's design tools, logic functions and other       
// software and tools, and its AMPP partner logic functions, and any output       
// files any of the foregoing (including device programming or simulation         
// files), and any associated documentation or information are expressly subject  
// to the terms and conditions of the Altera Program License Subscription         
// Agreement, Altera MegaCore Function License Agreement, or other applicable     
// license agreement, including, without limitation, that your use is for the     
// sole purpose of programming logic devices manufactured by Altera and sold by   
// Altera or its authorized distributors.  Please refer to the applicable         
// agreement for further details.                                   
 
module acl_fp_multadd(dataa, datab, datac, clock, enable, result);
// a*b + c
input	[31:0] dataa;
input	[31:0] datab;
input	[31:0] datac;
input clock;
input enable;
output [31:0] result;

// FP MAC wysiwyg
twentynm_fp_mac mac_fp_wys (
    // inputs
    .accumulate(),
    .chainin_overflow(),
    .chainin_invalid(),
    .chainin_underflow(),
    .chainin_inexact(),
	.ax(datac),
	.ay(datab),
    .az(dataa),
	.clk({2'b00,clock}),
	.ena({2'b11,enable}),
	.aclr(2'b00),
    .chainin(),
    // outputs
    .overflow(),
    .invalid(),
    .underflow(),
    .inexact(),
    .chainout_overflow(),
    .chainout_invalid(),
    .chainout_underflow(),
    .chainout_inexact(),
	.resulta(result),
    .chainout()
);
defparam mac_fp_wys.operation_mode = "sp_mult_add"; 
defparam mac_fp_wys.use_chainin = "false"; 
defparam mac_fp_wys.adder_subtract = "false"; 
defparam mac_fp_wys.ax_clock = "0"; 
defparam mac_fp_wys.ay_clock = "0"; 
defparam mac_fp_wys.az_clock = "0"; 
defparam mac_fp_wys.output_clock = "0"; 
defparam mac_fp_wys.accumulate_clock = "none"; 
defparam mac_fp_wys.ax_chainin_pl_clock = "0"; 
defparam mac_fp_wys.accum_pipeline_clock = "none"; 
defparam mac_fp_wys.mult_pipeline_clock = "0"; 
defparam mac_fp_wys.adder_input_clock = "0"; 
defparam mac_fp_wys.accum_adder_clock = "none"; 

endmodule