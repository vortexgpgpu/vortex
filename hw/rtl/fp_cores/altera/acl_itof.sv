// ------------------------------------------------------------------------- 
// High Level Design Compiler for Intel(R) FPGAs Version 17.1 (Release Build #273)
// Quartus Prime development tool and MATLAB/Simulink Interface
// 
// Legal Notice: Copyright 2017 Intel Corporation.  All rights reserved.
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

// SystemVerilog created from acl_itof
// SystemVerilog created on Wed Dec  9 01:17:51 2020


(* altera_attribute = "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007; -name MESSAGE_DISABLE 10958" *)
module acl_itof (
    input wire [31:0] a,
    input wire [0:0] en,
    output wire [31:0] q,
    input wire clk,
    input wire areset
    );

    wire [0:0] GND_q;
    wire [0:0] signX_uid6_fxpToFPTest_b;
    wire [31:0] xXorSign_uid7_fxpToFPTest_b;
    wire [31:0] xXorSign_uid7_fxpToFPTest_qi;
    reg [31:0] xXorSign_uid7_fxpToFPTest_q;
    wire [32:0] yE_uid8_fxpToFPTest_a;
    wire [32:0] yE_uid8_fxpToFPTest_b;
    logic [32:0] yE_uid8_fxpToFPTest_o;
    wire [32:0] yE_uid8_fxpToFPTest_q;
    wire [31:0] y_uid9_fxpToFPTest_in;
    wire [31:0] y_uid9_fxpToFPTest_b;
    wire [5:0] maxCount_uid11_fxpToFPTest_q;
    wire [0:0] inIsZero_uid12_fxpToFPTest_qi;
    reg [0:0] inIsZero_uid12_fxpToFPTest_q;
    wire [7:0] msbIn_uid13_fxpToFPTest_q;
    wire [8:0] expPreRnd_uid14_fxpToFPTest_a;
    wire [8:0] expPreRnd_uid14_fxpToFPTest_b;
    logic [8:0] expPreRnd_uid14_fxpToFPTest_o;
    wire [8:0] expPreRnd_uid14_fxpToFPTest_q;
    wire [32:0] expFracRnd_uid16_fxpToFPTest_q;
    wire [0:0] sticky_uid20_fxpToFPTest_qi;
    reg [0:0] sticky_uid20_fxpToFPTest_q;
    wire [0:0] nr_uid21_fxpToFPTest_q;
    wire [0:0] rnd_uid22_fxpToFPTest_q;
    wire [34:0] expFracR_uid24_fxpToFPTest_a;
    wire [34:0] expFracR_uid24_fxpToFPTest_b;
    logic [34:0] expFracR_uid24_fxpToFPTest_o;
    wire [33:0] expFracR_uid24_fxpToFPTest_q;
    wire [23:0] fracR_uid25_fxpToFPTest_in;
    wire [22:0] fracR_uid25_fxpToFPTest_b;
    wire [9:0] expR_uid26_fxpToFPTest_b;
    wire [11:0] udf_uid27_fxpToFPTest_a;
    wire [11:0] udf_uid27_fxpToFPTest_b;
    logic [11:0] udf_uid27_fxpToFPTest_o;
    wire [0:0] udf_uid27_fxpToFPTest_n;
    wire [7:0] expInf_uid28_fxpToFPTest_q;
    wire [11:0] ovf_uid29_fxpToFPTest_a;
    wire [11:0] ovf_uid29_fxpToFPTest_b;
    logic [11:0] ovf_uid29_fxpToFPTest_o;
    wire [0:0] ovf_uid29_fxpToFPTest_n;
    wire [0:0] excSelector_uid30_fxpToFPTest_q;
    wire [22:0] fracZ_uid31_fxpToFPTest_q;
    wire [0:0] fracRPostExc_uid32_fxpToFPTest_s;
    reg [22:0] fracRPostExc_uid32_fxpToFPTest_q;
    wire [0:0] udfOrInZero_uid33_fxpToFPTest_q;
    wire [1:0] excSelector_uid34_fxpToFPTest_q;
    wire [7:0] expZ_uid37_fxpToFPTest_q;
    wire [7:0] expR_uid38_fxpToFPTest_in;
    wire [7:0] expR_uid38_fxpToFPTest_b;
    wire [1:0] expRPostExc_uid39_fxpToFPTest_s;
    reg [7:0] expRPostExc_uid39_fxpToFPTest_q;
    wire [31:0] outRes_uid40_fxpToFPTest_q;
    wire [31:0] zs_uid42_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [0:0] vCount_uid44_lzcShifterZ1_uid10_fxpToFPTest_qi;
    reg [0:0] vCount_uid44_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [0:0] vStagei_uid46_lzcShifterZ1_uid10_fxpToFPTest_s;
    reg [31:0] vStagei_uid46_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [15:0] zs_uid47_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [0:0] vCount_uid49_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [31:0] cStage_uid52_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [0:0] vStagei_uid53_lzcShifterZ1_uid10_fxpToFPTest_s;
    reg [31:0] vStagei_uid53_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [0:0] vCount_uid56_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [31:0] cStage_uid59_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [0:0] vStagei_uid60_lzcShifterZ1_uid10_fxpToFPTest_s;
    reg [31:0] vStagei_uid60_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [3:0] zs_uid61_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [0:0] vCount_uid63_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [31:0] cStage_uid66_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [0:0] vStagei_uid67_lzcShifterZ1_uid10_fxpToFPTest_s;
    reg [31:0] vStagei_uid67_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [1:0] zs_uid68_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [0:0] vCount_uid70_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [31:0] cStage_uid73_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [0:0] vStagei_uid74_lzcShifterZ1_uid10_fxpToFPTest_s;
    reg [31:0] vStagei_uid74_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [0:0] vCount_uid77_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [31:0] cStage_uid80_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [0:0] vStagei_uid81_lzcShifterZ1_uid10_fxpToFPTest_s;
    reg [31:0] vStagei_uid81_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [5:0] vCount_uid82_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [7:0] vCountBig_uid84_lzcShifterZ1_uid10_fxpToFPTest_a;
    wire [7:0] vCountBig_uid84_lzcShifterZ1_uid10_fxpToFPTest_b;
    logic [7:0] vCountBig_uid84_lzcShifterZ1_uid10_fxpToFPTest_o;
    wire [0:0] vCountBig_uid84_lzcShifterZ1_uid10_fxpToFPTest_c;
    wire [0:0] vCountFinal_uid86_lzcShifterZ1_uid10_fxpToFPTest_s;
    reg [5:0] vCountFinal_uid86_lzcShifterZ1_uid10_fxpToFPTest_q;
    wire [1:0] l_uid17_fxpToFPTest_merged_bit_select_in;
    wire [0:0] l_uid17_fxpToFPTest_merged_bit_select_b;
    wire [0:0] l_uid17_fxpToFPTest_merged_bit_select_c;
    wire [15:0] rVStage_uid48_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_b;
    wire [15:0] rVStage_uid48_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_c;
    wire [7:0] rVStage_uid55_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_b;
    wire [23:0] rVStage_uid55_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_c;
    wire [3:0] rVStage_uid62_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_b;
    wire [27:0] rVStage_uid62_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_c;
    wire [1:0] rVStage_uid69_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_b;
    wire [29:0] rVStage_uid69_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_c;
    wire [0:0] rVStage_uid76_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_b;
    wire [30:0] rVStage_uid76_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_c;
    wire [30:0] fracRnd_uid15_fxpToFPTest_merged_bit_select_in;
    wire [23:0] fracRnd_uid15_fxpToFPTest_merged_bit_select_b;
    wire [6:0] fracRnd_uid15_fxpToFPTest_merged_bit_select_c;
    reg [23:0] redist0_fracRnd_uid15_fxpToFPTest_merged_bit_select_b_2_q;
    reg [0:0] redist1_vCount_uid63_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    reg [0:0] redist2_vCount_uid56_lzcShifterZ1_uid10_fxpToFPTest_q_1_q;
    reg [0:0] redist3_vCount_uid49_lzcShifterZ1_uid10_fxpToFPTest_q_2_q;
    reg [0:0] redist4_vCount_uid44_lzcShifterZ1_uid10_fxpToFPTest_q_3_q;
    reg [9:0] redist5_expR_uid26_fxpToFPTest_b_1_q;
    reg [22:0] redist6_fracR_uid25_fxpToFPTest_b_1_q;
    reg [0:0] redist7_sticky_uid20_fxpToFPTest_q_2_q;
    reg [0:0] redist8_inIsZero_uid12_fxpToFPTest_q_2_q;
    reg [31:0] redist9_y_uid9_fxpToFPTest_b_1_q;
    reg [0:0] redist10_signX_uid6_fxpToFPTest_b_1_q;
    reg [0:0] redist11_signX_uid6_fxpToFPTest_b_7_q;


    // signX_uid6_fxpToFPTest(BITSELECT,5)@0
    assign signX_uid6_fxpToFPTest_b = a[31:31];

    // redist10_signX_uid6_fxpToFPTest_b_1(DELAY,105)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist10_signX_uid6_fxpToFPTest_b_1 ( .xin(signX_uid6_fxpToFPTest_b), .xout(redist10_signX_uid6_fxpToFPTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist11_signX_uid6_fxpToFPTest_b_7(DELAY,106)
    dspba_delay_ver #( .width(1), .depth(6), .reset_kind("ASYNC") )
    redist11_signX_uid6_fxpToFPTest_b_7 ( .xin(redist10_signX_uid6_fxpToFPTest_b_1_q), .xout(redist11_signX_uid6_fxpToFPTest_b_7_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expInf_uid28_fxpToFPTest(CONSTANT,27)
    assign expInf_uid28_fxpToFPTest_q = 8'b11111111;

    // expZ_uid37_fxpToFPTest(CONSTANT,36)
    assign expZ_uid37_fxpToFPTest_q = 8'b00000000;

    // rVStage_uid76_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select(BITSELECT,93)@4
    assign rVStage_uid76_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_b = vStagei_uid74_lzcShifterZ1_uid10_fxpToFPTest_q[31:31];
    assign rVStage_uid76_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_c = vStagei_uid74_lzcShifterZ1_uid10_fxpToFPTest_q[30:0];

    // GND(CONSTANT,0)
    assign GND_q = 1'b0;

    // cStage_uid80_lzcShifterZ1_uid10_fxpToFPTest(BITJOIN,79)@4
    assign cStage_uid80_lzcShifterZ1_uid10_fxpToFPTest_q = {rVStage_uid76_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_c, GND_q};

    // rVStage_uid69_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select(BITSELECT,92)@4
    assign rVStage_uid69_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_b = vStagei_uid67_lzcShifterZ1_uid10_fxpToFPTest_q[31:30];
    assign rVStage_uid69_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_c = vStagei_uid67_lzcShifterZ1_uid10_fxpToFPTest_q[29:0];

    // zs_uid68_lzcShifterZ1_uid10_fxpToFPTest(CONSTANT,67)
    assign zs_uid68_lzcShifterZ1_uid10_fxpToFPTest_q = 2'b00;

    // cStage_uid73_lzcShifterZ1_uid10_fxpToFPTest(BITJOIN,72)@4
    assign cStage_uid73_lzcShifterZ1_uid10_fxpToFPTest_q = {rVStage_uid69_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_c, zs_uid68_lzcShifterZ1_uid10_fxpToFPTest_q};

    // rVStage_uid62_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select(BITSELECT,91)@3
    assign rVStage_uid62_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_b = vStagei_uid60_lzcShifterZ1_uid10_fxpToFPTest_q[31:28];
    assign rVStage_uid62_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_c = vStagei_uid60_lzcShifterZ1_uid10_fxpToFPTest_q[27:0];

    // zs_uid61_lzcShifterZ1_uid10_fxpToFPTest(CONSTANT,60)
    assign zs_uid61_lzcShifterZ1_uid10_fxpToFPTest_q = 4'b0000;

    // cStage_uid66_lzcShifterZ1_uid10_fxpToFPTest(BITJOIN,65)@3
    assign cStage_uid66_lzcShifterZ1_uid10_fxpToFPTest_q = {rVStage_uid62_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_c, zs_uid61_lzcShifterZ1_uid10_fxpToFPTest_q};

    // rVStage_uid55_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select(BITSELECT,90)@3
    assign rVStage_uid55_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_b = vStagei_uid53_lzcShifterZ1_uid10_fxpToFPTest_q[31:24];
    assign rVStage_uid55_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_c = vStagei_uid53_lzcShifterZ1_uid10_fxpToFPTest_q[23:0];

    // cStage_uid59_lzcShifterZ1_uid10_fxpToFPTest(BITJOIN,58)@3
    assign cStage_uid59_lzcShifterZ1_uid10_fxpToFPTest_q = {rVStage_uid55_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_c, expZ_uid37_fxpToFPTest_q};

    // rVStage_uid48_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select(BITSELECT,89)@2
    assign rVStage_uid48_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_b = vStagei_uid46_lzcShifterZ1_uid10_fxpToFPTest_q[31:16];
    assign rVStage_uid48_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_c = vStagei_uid46_lzcShifterZ1_uid10_fxpToFPTest_q[15:0];

    // zs_uid47_lzcShifterZ1_uid10_fxpToFPTest(CONSTANT,46)
    assign zs_uid47_lzcShifterZ1_uid10_fxpToFPTest_q = 16'b0000000000000000;

    // cStage_uid52_lzcShifterZ1_uid10_fxpToFPTest(BITJOIN,51)@2
    assign cStage_uid52_lzcShifterZ1_uid10_fxpToFPTest_q = {rVStage_uid48_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_c, zs_uid47_lzcShifterZ1_uid10_fxpToFPTest_q};

    // zs_uid42_lzcShifterZ1_uid10_fxpToFPTest(CONSTANT,41)
    assign zs_uid42_lzcShifterZ1_uid10_fxpToFPTest_q = 32'b00000000000000000000000000000000;

    // xXorSign_uid7_fxpToFPTest(LOGICAL,6)@0 + 1
    assign xXorSign_uid7_fxpToFPTest_b = {{31{signX_uid6_fxpToFPTest_b[0]}}, signX_uid6_fxpToFPTest_b};
    assign xXorSign_uid7_fxpToFPTest_qi = a ^ xXorSign_uid7_fxpToFPTest_b;
    dspba_delay_ver #( .width(32), .depth(1), .reset_kind("ASYNC") )
    xXorSign_uid7_fxpToFPTest_delay ( .xin(xXorSign_uid7_fxpToFPTest_qi), .xout(xXorSign_uid7_fxpToFPTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // yE_uid8_fxpToFPTest(ADD,7)@1
    assign yE_uid8_fxpToFPTest_a = {1'b0, xXorSign_uid7_fxpToFPTest_q};
    assign yE_uid8_fxpToFPTest_b = {32'b00000000000000000000000000000000, redist10_signX_uid6_fxpToFPTest_b_1_q};
    assign yE_uid8_fxpToFPTest_o = $unsigned(yE_uid8_fxpToFPTest_a) + $unsigned(yE_uid8_fxpToFPTest_b);
    assign yE_uid8_fxpToFPTest_q = yE_uid8_fxpToFPTest_o[32:0];

    // y_uid9_fxpToFPTest(BITSELECT,8)@1
    assign y_uid9_fxpToFPTest_in = yE_uid8_fxpToFPTest_q[31:0];
    assign y_uid9_fxpToFPTest_b = y_uid9_fxpToFPTest_in[31:0];

    // redist9_y_uid9_fxpToFPTest_b_1(DELAY,104)
    dspba_delay_ver #( .width(32), .depth(1), .reset_kind("ASYNC") )
    redist9_y_uid9_fxpToFPTest_b_1 ( .xin(y_uid9_fxpToFPTest_b), .xout(redist9_y_uid9_fxpToFPTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // vCount_uid44_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,43)@1 + 1
    assign vCount_uid44_lzcShifterZ1_uid10_fxpToFPTest_qi = y_uid9_fxpToFPTest_b == zs_uid42_lzcShifterZ1_uid10_fxpToFPTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    vCount_uid44_lzcShifterZ1_uid10_fxpToFPTest_delay ( .xin(vCount_uid44_lzcShifterZ1_uid10_fxpToFPTest_qi), .xout(vCount_uid44_lzcShifterZ1_uid10_fxpToFPTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // vStagei_uid46_lzcShifterZ1_uid10_fxpToFPTest(MUX,45)@2
    assign vStagei_uid46_lzcShifterZ1_uid10_fxpToFPTest_s = vCount_uid44_lzcShifterZ1_uid10_fxpToFPTest_q;
    always @(vStagei_uid46_lzcShifterZ1_uid10_fxpToFPTest_s or en or redist9_y_uid9_fxpToFPTest_b_1_q or zs_uid42_lzcShifterZ1_uid10_fxpToFPTest_q)
    begin
        unique case (vStagei_uid46_lzcShifterZ1_uid10_fxpToFPTest_s)
            1'b0 : vStagei_uid46_lzcShifterZ1_uid10_fxpToFPTest_q = redist9_y_uid9_fxpToFPTest_b_1_q;
            1'b1 : vStagei_uid46_lzcShifterZ1_uid10_fxpToFPTest_q = zs_uid42_lzcShifterZ1_uid10_fxpToFPTest_q;
            default : vStagei_uid46_lzcShifterZ1_uid10_fxpToFPTest_q = 32'b0;
        endcase
    end

    // vCount_uid49_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,48)@2
    assign vCount_uid49_lzcShifterZ1_uid10_fxpToFPTest_q = rVStage_uid48_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_b == zs_uid47_lzcShifterZ1_uid10_fxpToFPTest_q ? 1'b1 : 1'b0;

    // vStagei_uid53_lzcShifterZ1_uid10_fxpToFPTest(MUX,52)@2 + 1
    assign vStagei_uid53_lzcShifterZ1_uid10_fxpToFPTest_s = vCount_uid49_lzcShifterZ1_uid10_fxpToFPTest_q;
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            vStagei_uid53_lzcShifterZ1_uid10_fxpToFPTest_q <= 32'b0;
        end
        else if (en == 1'b1)
        begin
            unique case (vStagei_uid53_lzcShifterZ1_uid10_fxpToFPTest_s)
                1'b0 : vStagei_uid53_lzcShifterZ1_uid10_fxpToFPTest_q <= vStagei_uid46_lzcShifterZ1_uid10_fxpToFPTest_q;
                1'b1 : vStagei_uid53_lzcShifterZ1_uid10_fxpToFPTest_q <= cStage_uid52_lzcShifterZ1_uid10_fxpToFPTest_q;
                default : vStagei_uid53_lzcShifterZ1_uid10_fxpToFPTest_q <= 32'b0;
            endcase
        end
    end

    // vCount_uid56_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,55)@3
    assign vCount_uid56_lzcShifterZ1_uid10_fxpToFPTest_q = rVStage_uid55_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_b == expZ_uid37_fxpToFPTest_q ? 1'b1 : 1'b0;

    // vStagei_uid60_lzcShifterZ1_uid10_fxpToFPTest(MUX,59)@3
    assign vStagei_uid60_lzcShifterZ1_uid10_fxpToFPTest_s = vCount_uid56_lzcShifterZ1_uid10_fxpToFPTest_q;
    always @(vStagei_uid60_lzcShifterZ1_uid10_fxpToFPTest_s or en or vStagei_uid53_lzcShifterZ1_uid10_fxpToFPTest_q or cStage_uid59_lzcShifterZ1_uid10_fxpToFPTest_q)
    begin
        unique case (vStagei_uid60_lzcShifterZ1_uid10_fxpToFPTest_s)
            1'b0 : vStagei_uid60_lzcShifterZ1_uid10_fxpToFPTest_q = vStagei_uid53_lzcShifterZ1_uid10_fxpToFPTest_q;
            1'b1 : vStagei_uid60_lzcShifterZ1_uid10_fxpToFPTest_q = cStage_uid59_lzcShifterZ1_uid10_fxpToFPTest_q;
            default : vStagei_uid60_lzcShifterZ1_uid10_fxpToFPTest_q = 32'b0;
        endcase
    end

    // vCount_uid63_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,62)@3
    assign vCount_uid63_lzcShifterZ1_uid10_fxpToFPTest_q = rVStage_uid62_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_b == zs_uid61_lzcShifterZ1_uid10_fxpToFPTest_q ? 1'b1 : 1'b0;

    // vStagei_uid67_lzcShifterZ1_uid10_fxpToFPTest(MUX,66)@3 + 1
    assign vStagei_uid67_lzcShifterZ1_uid10_fxpToFPTest_s = vCount_uid63_lzcShifterZ1_uid10_fxpToFPTest_q;
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            vStagei_uid67_lzcShifterZ1_uid10_fxpToFPTest_q <= 32'b0;
        end
        else if (en == 1'b1)
        begin
            unique case (vStagei_uid67_lzcShifterZ1_uid10_fxpToFPTest_s)
                1'b0 : vStagei_uid67_lzcShifterZ1_uid10_fxpToFPTest_q <= vStagei_uid60_lzcShifterZ1_uid10_fxpToFPTest_q;
                1'b1 : vStagei_uid67_lzcShifterZ1_uid10_fxpToFPTest_q <= cStage_uid66_lzcShifterZ1_uid10_fxpToFPTest_q;
                default : vStagei_uid67_lzcShifterZ1_uid10_fxpToFPTest_q <= 32'b0;
            endcase
        end
    end

    // vCount_uid70_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,69)@4
    assign vCount_uid70_lzcShifterZ1_uid10_fxpToFPTest_q = rVStage_uid69_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_b == zs_uid68_lzcShifterZ1_uid10_fxpToFPTest_q ? 1'b1 : 1'b0;

    // vStagei_uid74_lzcShifterZ1_uid10_fxpToFPTest(MUX,73)@4
    assign vStagei_uid74_lzcShifterZ1_uid10_fxpToFPTest_s = vCount_uid70_lzcShifterZ1_uid10_fxpToFPTest_q;
    always @(vStagei_uid74_lzcShifterZ1_uid10_fxpToFPTest_s or en or vStagei_uid67_lzcShifterZ1_uid10_fxpToFPTest_q or cStage_uid73_lzcShifterZ1_uid10_fxpToFPTest_q)
    begin
        unique case (vStagei_uid74_lzcShifterZ1_uid10_fxpToFPTest_s)
            1'b0 : vStagei_uid74_lzcShifterZ1_uid10_fxpToFPTest_q = vStagei_uid67_lzcShifterZ1_uid10_fxpToFPTest_q;
            1'b1 : vStagei_uid74_lzcShifterZ1_uid10_fxpToFPTest_q = cStage_uid73_lzcShifterZ1_uid10_fxpToFPTest_q;
            default : vStagei_uid74_lzcShifterZ1_uid10_fxpToFPTest_q = 32'b0;
        endcase
    end

    // vCount_uid77_lzcShifterZ1_uid10_fxpToFPTest(LOGICAL,76)@4
    assign vCount_uid77_lzcShifterZ1_uid10_fxpToFPTest_q = rVStage_uid76_lzcShifterZ1_uid10_fxpToFPTest_merged_bit_select_b == GND_q ? 1'b1 : 1'b0;

    // vStagei_uid81_lzcShifterZ1_uid10_fxpToFPTest(MUX,80)@4
    assign vStagei_uid81_lzcShifterZ1_uid10_fxpToFPTest_s = vCount_uid77_lzcShifterZ1_uid10_fxpToFPTest_q;
    always @(vStagei_uid81_lzcShifterZ1_uid10_fxpToFPTest_s or en or vStagei_uid74_lzcShifterZ1_uid10_fxpToFPTest_q or cStage_uid80_lzcShifterZ1_uid10_fxpToFPTest_q)
    begin
        unique case (vStagei_uid81_lzcShifterZ1_uid10_fxpToFPTest_s)
            1'b0 : vStagei_uid81_lzcShifterZ1_uid10_fxpToFPTest_q = vStagei_uid74_lzcShifterZ1_uid10_fxpToFPTest_q;
            1'b1 : vStagei_uid81_lzcShifterZ1_uid10_fxpToFPTest_q = cStage_uid80_lzcShifterZ1_uid10_fxpToFPTest_q;
            default : vStagei_uid81_lzcShifterZ1_uid10_fxpToFPTest_q = 32'b0;
        endcase
    end

    // fracRnd_uid15_fxpToFPTest_merged_bit_select(BITSELECT,94)@4
    assign fracRnd_uid15_fxpToFPTest_merged_bit_select_in = vStagei_uid81_lzcShifterZ1_uid10_fxpToFPTest_q[30:0];
    assign fracRnd_uid15_fxpToFPTest_merged_bit_select_b = fracRnd_uid15_fxpToFPTest_merged_bit_select_in[30:7];
    assign fracRnd_uid15_fxpToFPTest_merged_bit_select_c = fracRnd_uid15_fxpToFPTest_merged_bit_select_in[6:0];

    // sticky_uid20_fxpToFPTest(LOGICAL,19)@4 + 1
    assign sticky_uid20_fxpToFPTest_qi = fracRnd_uid15_fxpToFPTest_merged_bit_select_c != 7'b0000000 ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    sticky_uid20_fxpToFPTest_delay ( .xin(sticky_uid20_fxpToFPTest_qi), .xout(sticky_uid20_fxpToFPTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist7_sticky_uid20_fxpToFPTest_q_2(DELAY,102)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist7_sticky_uid20_fxpToFPTest_q_2 ( .xin(sticky_uid20_fxpToFPTest_q), .xout(redist7_sticky_uid20_fxpToFPTest_q_2_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // nr_uid21_fxpToFPTest(LOGICAL,20)@6
    assign nr_uid21_fxpToFPTest_q = ~ (l_uid17_fxpToFPTest_merged_bit_select_c);

    // l_uid17_fxpToFPTest_merged_bit_select(BITSELECT,88)@6
    assign l_uid17_fxpToFPTest_merged_bit_select_in = expFracRnd_uid16_fxpToFPTest_q[1:0];
    assign l_uid17_fxpToFPTest_merged_bit_select_b = l_uid17_fxpToFPTest_merged_bit_select_in[1:1];
    assign l_uid17_fxpToFPTest_merged_bit_select_c = l_uid17_fxpToFPTest_merged_bit_select_in[0:0];

    // rnd_uid22_fxpToFPTest(LOGICAL,21)@6
    assign rnd_uid22_fxpToFPTest_q = l_uid17_fxpToFPTest_merged_bit_select_b | nr_uid21_fxpToFPTest_q | redist7_sticky_uid20_fxpToFPTest_q_2_q;

    // maxCount_uid11_fxpToFPTest(CONSTANT,10)
    assign maxCount_uid11_fxpToFPTest_q = 6'b100000;

    // redist4_vCount_uid44_lzcShifterZ1_uid10_fxpToFPTest_q_3(DELAY,99)
    dspba_delay_ver #( .width(1), .depth(2), .reset_kind("ASYNC") )
    redist4_vCount_uid44_lzcShifterZ1_uid10_fxpToFPTest_q_3 ( .xin(vCount_uid44_lzcShifterZ1_uid10_fxpToFPTest_q), .xout(redist4_vCount_uid44_lzcShifterZ1_uid10_fxpToFPTest_q_3_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist3_vCount_uid49_lzcShifterZ1_uid10_fxpToFPTest_q_2(DELAY,98)
    dspba_delay_ver #( .width(1), .depth(2), .reset_kind("ASYNC") )
    redist3_vCount_uid49_lzcShifterZ1_uid10_fxpToFPTest_q_2 ( .xin(vCount_uid49_lzcShifterZ1_uid10_fxpToFPTest_q), .xout(redist3_vCount_uid49_lzcShifterZ1_uid10_fxpToFPTest_q_2_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist2_vCount_uid56_lzcShifterZ1_uid10_fxpToFPTest_q_1(DELAY,97)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist2_vCount_uid56_lzcShifterZ1_uid10_fxpToFPTest_q_1 ( .xin(vCount_uid56_lzcShifterZ1_uid10_fxpToFPTest_q), .xout(redist2_vCount_uid56_lzcShifterZ1_uid10_fxpToFPTest_q_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist1_vCount_uid63_lzcShifterZ1_uid10_fxpToFPTest_q_1(DELAY,96)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist1_vCount_uid63_lzcShifterZ1_uid10_fxpToFPTest_q_1 ( .xin(vCount_uid63_lzcShifterZ1_uid10_fxpToFPTest_q), .xout(redist1_vCount_uid63_lzcShifterZ1_uid10_fxpToFPTest_q_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // vCount_uid82_lzcShifterZ1_uid10_fxpToFPTest(BITJOIN,81)@4
    assign vCount_uid82_lzcShifterZ1_uid10_fxpToFPTest_q = {redist4_vCount_uid44_lzcShifterZ1_uid10_fxpToFPTest_q_3_q, redist3_vCount_uid49_lzcShifterZ1_uid10_fxpToFPTest_q_2_q, redist2_vCount_uid56_lzcShifterZ1_uid10_fxpToFPTest_q_1_q, redist1_vCount_uid63_lzcShifterZ1_uid10_fxpToFPTest_q_1_q, vCount_uid70_lzcShifterZ1_uid10_fxpToFPTest_q, vCount_uid77_lzcShifterZ1_uid10_fxpToFPTest_q};

    // vCountBig_uid84_lzcShifterZ1_uid10_fxpToFPTest(COMPARE,83)@4
    assign vCountBig_uid84_lzcShifterZ1_uid10_fxpToFPTest_a = {2'b00, maxCount_uid11_fxpToFPTest_q};
    assign vCountBig_uid84_lzcShifterZ1_uid10_fxpToFPTest_b = {2'b00, vCount_uid82_lzcShifterZ1_uid10_fxpToFPTest_q};
    assign vCountBig_uid84_lzcShifterZ1_uid10_fxpToFPTest_o = $unsigned(vCountBig_uid84_lzcShifterZ1_uid10_fxpToFPTest_a) - $unsigned(vCountBig_uid84_lzcShifterZ1_uid10_fxpToFPTest_b);
    assign vCountBig_uid84_lzcShifterZ1_uid10_fxpToFPTest_c[0] = vCountBig_uid84_lzcShifterZ1_uid10_fxpToFPTest_o[7];

    // vCountFinal_uid86_lzcShifterZ1_uid10_fxpToFPTest(MUX,85)@4 + 1
    assign vCountFinal_uid86_lzcShifterZ1_uid10_fxpToFPTest_s = vCountBig_uid84_lzcShifterZ1_uid10_fxpToFPTest_c;
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            vCountFinal_uid86_lzcShifterZ1_uid10_fxpToFPTest_q <= 6'b0;
        end
        else if (en == 1'b1)
        begin
            unique case (vCountFinal_uid86_lzcShifterZ1_uid10_fxpToFPTest_s)
                1'b0 : vCountFinal_uid86_lzcShifterZ1_uid10_fxpToFPTest_q <= vCount_uid82_lzcShifterZ1_uid10_fxpToFPTest_q;
                1'b1 : vCountFinal_uid86_lzcShifterZ1_uid10_fxpToFPTest_q <= maxCount_uid11_fxpToFPTest_q;
                default : vCountFinal_uid86_lzcShifterZ1_uid10_fxpToFPTest_q <= 6'b0;
            endcase
        end
    end

    // msbIn_uid13_fxpToFPTest(CONSTANT,12)
    assign msbIn_uid13_fxpToFPTest_q = 8'b10011110;

    // expPreRnd_uid14_fxpToFPTest(SUB,13)@5 + 1
    assign expPreRnd_uid14_fxpToFPTest_a = {1'b0, msbIn_uid13_fxpToFPTest_q};
    assign expPreRnd_uid14_fxpToFPTest_b = {3'b000, vCountFinal_uid86_lzcShifterZ1_uid10_fxpToFPTest_q};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            expPreRnd_uid14_fxpToFPTest_o <= 9'b0;
        end
        else if (en == 1'b1)
        begin
            expPreRnd_uid14_fxpToFPTest_o <= $unsigned(expPreRnd_uid14_fxpToFPTest_a) - $unsigned(expPreRnd_uid14_fxpToFPTest_b);
        end
    end
    assign expPreRnd_uid14_fxpToFPTest_q = expPreRnd_uid14_fxpToFPTest_o[8:0];

    // redist0_fracRnd_uid15_fxpToFPTest_merged_bit_select_b_2(DELAY,95)
    dspba_delay_ver #( .width(24), .depth(2), .reset_kind("ASYNC") )
    redist0_fracRnd_uid15_fxpToFPTest_merged_bit_select_b_2 ( .xin(fracRnd_uid15_fxpToFPTest_merged_bit_select_b), .xout(redist0_fracRnd_uid15_fxpToFPTest_merged_bit_select_b_2_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expFracRnd_uid16_fxpToFPTest(BITJOIN,15)@6
    assign expFracRnd_uid16_fxpToFPTest_q = {expPreRnd_uid14_fxpToFPTest_q, redist0_fracRnd_uid15_fxpToFPTest_merged_bit_select_b_2_q};

    // expFracR_uid24_fxpToFPTest(ADD,23)@6
    assign expFracR_uid24_fxpToFPTest_a = {{2{expFracRnd_uid16_fxpToFPTest_q[32]}}, expFracRnd_uid16_fxpToFPTest_q};
    assign expFracR_uid24_fxpToFPTest_b = {34'b0000000000000000000000000000000000, rnd_uid22_fxpToFPTest_q};
    assign expFracR_uid24_fxpToFPTest_o = $signed(expFracR_uid24_fxpToFPTest_a) + $signed(expFracR_uid24_fxpToFPTest_b);
    assign expFracR_uid24_fxpToFPTest_q = expFracR_uid24_fxpToFPTest_o[33:0];

    // expR_uid26_fxpToFPTest(BITSELECT,25)@6
    assign expR_uid26_fxpToFPTest_b = expFracR_uid24_fxpToFPTest_q[33:24];

    // redist5_expR_uid26_fxpToFPTest_b_1(DELAY,100)
    dspba_delay_ver #( .width(10), .depth(1), .reset_kind("ASYNC") )
    redist5_expR_uid26_fxpToFPTest_b_1 ( .xin(expR_uid26_fxpToFPTest_b), .xout(redist5_expR_uid26_fxpToFPTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expR_uid38_fxpToFPTest(BITSELECT,37)@7
    assign expR_uid38_fxpToFPTest_in = redist5_expR_uid26_fxpToFPTest_b_1_q[7:0];
    assign expR_uid38_fxpToFPTest_b = expR_uid38_fxpToFPTest_in[7:0];

    // ovf_uid29_fxpToFPTest(COMPARE,28)@7
    assign ovf_uid29_fxpToFPTest_a = {{2{redist5_expR_uid26_fxpToFPTest_b_1_q[9]}}, redist5_expR_uid26_fxpToFPTest_b_1_q};
    assign ovf_uid29_fxpToFPTest_b = {4'b0000, expInf_uid28_fxpToFPTest_q};
    assign ovf_uid29_fxpToFPTest_o = $signed(ovf_uid29_fxpToFPTest_a) - $signed(ovf_uid29_fxpToFPTest_b);
    assign ovf_uid29_fxpToFPTest_n[0] = ~ (ovf_uid29_fxpToFPTest_o[11]);

    // inIsZero_uid12_fxpToFPTest(LOGICAL,11)@5 + 1
    assign inIsZero_uid12_fxpToFPTest_qi = vCountFinal_uid86_lzcShifterZ1_uid10_fxpToFPTest_q == maxCount_uid11_fxpToFPTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    inIsZero_uid12_fxpToFPTest_delay ( .xin(inIsZero_uid12_fxpToFPTest_qi), .xout(inIsZero_uid12_fxpToFPTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist8_inIsZero_uid12_fxpToFPTest_q_2(DELAY,103)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist8_inIsZero_uid12_fxpToFPTest_q_2 ( .xin(inIsZero_uid12_fxpToFPTest_q), .xout(redist8_inIsZero_uid12_fxpToFPTest_q_2_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // udf_uid27_fxpToFPTest(COMPARE,26)@7
    assign udf_uid27_fxpToFPTest_a = {11'b00000000000, GND_q};
    assign udf_uid27_fxpToFPTest_b = {{2{redist5_expR_uid26_fxpToFPTest_b_1_q[9]}}, redist5_expR_uid26_fxpToFPTest_b_1_q};
    assign udf_uid27_fxpToFPTest_o = $signed(udf_uid27_fxpToFPTest_a) - $signed(udf_uid27_fxpToFPTest_b);
    assign udf_uid27_fxpToFPTest_n[0] = ~ (udf_uid27_fxpToFPTest_o[11]);

    // udfOrInZero_uid33_fxpToFPTest(LOGICAL,32)@7
    assign udfOrInZero_uid33_fxpToFPTest_q = udf_uid27_fxpToFPTest_n | redist8_inIsZero_uid12_fxpToFPTest_q_2_q;

    // excSelector_uid34_fxpToFPTest(BITJOIN,33)@7
    assign excSelector_uid34_fxpToFPTest_q = {ovf_uid29_fxpToFPTest_n, udfOrInZero_uid33_fxpToFPTest_q};

    // expRPostExc_uid39_fxpToFPTest(MUX,38)@7
    assign expRPostExc_uid39_fxpToFPTest_s = excSelector_uid34_fxpToFPTest_q;
    always @(expRPostExc_uid39_fxpToFPTest_s or en or expR_uid38_fxpToFPTest_b or expZ_uid37_fxpToFPTest_q or expInf_uid28_fxpToFPTest_q)
    begin
        unique case (expRPostExc_uid39_fxpToFPTest_s)
            2'b00 : expRPostExc_uid39_fxpToFPTest_q = expR_uid38_fxpToFPTest_b;
            2'b01 : expRPostExc_uid39_fxpToFPTest_q = expZ_uid37_fxpToFPTest_q;
            2'b10 : expRPostExc_uid39_fxpToFPTest_q = expInf_uid28_fxpToFPTest_q;
            2'b11 : expRPostExc_uid39_fxpToFPTest_q = expInf_uid28_fxpToFPTest_q;
            default : expRPostExc_uid39_fxpToFPTest_q = 8'b0;
        endcase
    end

    // fracZ_uid31_fxpToFPTest(CONSTANT,30)
    assign fracZ_uid31_fxpToFPTest_q = 23'b00000000000000000000000;

    // fracR_uid25_fxpToFPTest(BITSELECT,24)@6
    assign fracR_uid25_fxpToFPTest_in = expFracR_uid24_fxpToFPTest_q[23:0];
    assign fracR_uid25_fxpToFPTest_b = fracR_uid25_fxpToFPTest_in[23:1];

    // redist6_fracR_uid25_fxpToFPTest_b_1(DELAY,101)
    dspba_delay_ver #( .width(23), .depth(1), .reset_kind("ASYNC") )
    redist6_fracR_uid25_fxpToFPTest_b_1 ( .xin(fracR_uid25_fxpToFPTest_b), .xout(redist6_fracR_uid25_fxpToFPTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // excSelector_uid30_fxpToFPTest(LOGICAL,29)@7
    assign excSelector_uid30_fxpToFPTest_q = redist8_inIsZero_uid12_fxpToFPTest_q_2_q | ovf_uid29_fxpToFPTest_n | udf_uid27_fxpToFPTest_n;

    // fracRPostExc_uid32_fxpToFPTest(MUX,31)@7
    assign fracRPostExc_uid32_fxpToFPTest_s = excSelector_uid30_fxpToFPTest_q;
    always @(fracRPostExc_uid32_fxpToFPTest_s or en or redist6_fracR_uid25_fxpToFPTest_b_1_q or fracZ_uid31_fxpToFPTest_q)
    begin
        unique case (fracRPostExc_uid32_fxpToFPTest_s)
            1'b0 : fracRPostExc_uid32_fxpToFPTest_q = redist6_fracR_uid25_fxpToFPTest_b_1_q;
            1'b1 : fracRPostExc_uid32_fxpToFPTest_q = fracZ_uid31_fxpToFPTest_q;
            default : fracRPostExc_uid32_fxpToFPTest_q = 23'b0;
        endcase
    end

    // outRes_uid40_fxpToFPTest(BITJOIN,39)@7
    assign outRes_uid40_fxpToFPTest_q = {redist11_signX_uid6_fxpToFPTest_b_7_q, expRPostExc_uid39_fxpToFPTest_q, fracRPostExc_uid32_fxpToFPTest_q};

    // xOut(GPOUT,4)@7
    assign q = outRes_uid40_fxpToFPTest_q;

endmodule
