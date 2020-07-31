// (C) 1992-2016 Intel Corporation.                            
// Intel, the Intel logo, Intel, MegaCore, NIOS II, Quartus and TalkBack words    
// and logos are trademarks of Intel Corporation or its subsidiaries in the U.S.  
// and/or other countries. Other marks and brands may be claimed as the property  
// of others. See Trademarks on intel.com for full list of Intel trademarks or    
// the Trademarks & Brands Names Database (if Intel) or See www.Intel.com/legal (if Altera) 
// Your use of Intel Corporation's design tools, logic functions and other        
// software and tools, and its AMPP partner logic functions, and any output       
// files any of the foregoing (including device programming or simulation         
// files), and any associated documentation or information are expressly subject  
// to the terms and conditions of the Altera Program License Subscription         
// Agreement, Intel MegaCore Function License Agreement, or other applicable      
// license agreement, including, without limitation, that your use is for the     
// sole purpose of programming logic devices manufactured by Intel and sold by    
// Intel or its authorized distributors.  Please refer to the applicable          
// agreement for further details.                                                 

module acl_fp_mul(dataa, datab, clock, enable, result);

input	[31:0] dataa;
input	[31:0] datab;
input clock, enable;
   
output [31:0] result;

// FP MAC wysiwyg
twentynm_fp_mac mac_fp_wys (
    // inputs
    .accumulate(),
    .chainin_overflow(),
    .chainin_invalid(),
    .chainin_underflow(),
    .chainin_inexact(),
	.ax(),
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
defparam mac_fp_wys.operation_mode = "sp_mult"; 
defparam mac_fp_wys.use_chainin = "false"; 
defparam mac_fp_wys.adder_subtract = "false"; 
defparam mac_fp_wys.ax_clock = "none"; 
defparam mac_fp_wys.ay_clock = "0"; 
defparam mac_fp_wys.az_clock = "0"; 
defparam mac_fp_wys.output_clock = "0"; 
defparam mac_fp_wys.accumulate_clock = "none"; 
defparam mac_fp_wys.ax_chainin_pl_clock = "none"; 
defparam mac_fp_wys.accum_pipeline_clock = "none"; 
defparam mac_fp_wys.mult_pipeline_clock = "0"; 
defparam mac_fp_wys.adder_input_clock = "none"; 
defparam mac_fp_wys.accum_adder_clock = "none"; 

endmodule
