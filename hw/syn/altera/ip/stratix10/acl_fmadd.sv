// ------------------------------------------------------------------------- 
// High Level Design Compiler for Intel(R) FPGAs Version 18.1 (Release Build #277)
// Quartus Prime development tool and MATLAB/Simulink Interface
// 
// Legal Notice: Copyright 2019 Intel Corporation.  All rights reserved.
// Your use of  Intel Corporation's design tools,  logic functions and other
// software and  tools, and its AMPP partner logic functions, and any output
// files any  of the foregoing (including  device programming  or simulation
// files), and  any associated  documentation  or information  are expressly
// subject  to the terms and  conditions of the  Intel FPGA Software License
// Agreement, Intel MegaCore Function License Agreement, or other applicable
// license agreement,  including,  without limitation,  that your use is for
// the  sole  purpose of  programming  logic devices  manufactured by  Intel
// and  sold by Intel  or its authorized  distributors. Please refer  to the
// applicable agreement for further details.
// ---------------------------------------------------------------------------

// SystemVerilog created from acl_fmadd
// SystemVerilog created on Sun Dec 27 09:48:58 2020


(* altera_attribute = "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007; -name MESSAGE_DISABLE 10958" *)
module acl_fmadd (
    input wire [31:0] a,
    input wire [31:0] b,
    input wire [31:0] c,
    input wire [0:0] en,
    output wire [31:0] q,
    input wire clk,
    input wire areset
    );

    wire fpMultAddTest_impl_reset0;
    wire fpMultAddTest_impl_ena0;
    wire [31:0] fpMultAddTest_impl_ax0;
    wire [31:0] fpMultAddTest_impl_ay0;
    wire [31:0] fpMultAddTest_impl_az0;
    wire [31:0] fpMultAddTest_impl_q0;


    // fpMultAddTest_impl(FPCOLUMN,5)@0
    // out q0@4
    assign fpMultAddTest_impl_ax0 = c;
    assign fpMultAddTest_impl_ay0 = b;
    assign fpMultAddTest_impl_az0 = a;
    assign fpMultAddTest_impl_reset0 = areset;
    assign fpMultAddTest_impl_ena0 = en[0] | fpMultAddTest_impl_reset0;
    fourteennm_fp_mac #(
        .operation_mode("sp_mult_add"),
        .ax_clock("0"),
        .ay_clock("0"),
        .az_clock("0"),
        .mult_2nd_pipeline_clock("0"),
        .adder_input_clock("0"),
        .ax_chainin_pl_clock("0"),
        .output_clock("0"),
        .clear_type("sclr")
    ) fpMultAddTest_impl_DSP0 (
        .clk({1'b0,1'b0,clk}),
        .ena({ 1'b0, 1'b0, fpMultAddTest_impl_ena0 }),
        .clr({ fpMultAddTest_impl_reset0, fpMultAddTest_impl_reset0 }),
        .ax(fpMultAddTest_impl_ax0),
        .ay(fpMultAddTest_impl_ay0),
        .az(fpMultAddTest_impl_az0),
        .resulta(fpMultAddTest_impl_q0),
        .accumulate(),
        .chainin(),
        .chainout()
    );

    // xOut(GPOUT,4)@4
    assign q = fpMultAddTest_impl_q0;

endmodule
