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

// SystemVerilog created from acl_ftoi
// SystemVerilog created on Wed Dec  9 01:17:51 2020


(* altera_attribute = "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007; -name MESSAGE_DISABLE 10958" *)
module acl_ftoi (
    input wire [31:0] a,
    input wire [0:0] en,
    output wire [31:0] q,
    input wire clk,
    input wire areset
    );

    wire [0:0] GND_q;
    wire [7:0] cstAllOWE_uid6_fpToFxPTest_q;
    wire [22:0] cstZeroWF_uid7_fpToFxPTest_q;
    wire [7:0] cstAllZWE_uid8_fpToFxPTest_q;
    wire [7:0] exp_x_uid9_fpToFxPTest_b;
    wire [22:0] frac_x_uid10_fpToFxPTest_b;
    wire [0:0] excZ_x_uid11_fpToFxPTest_qi;
    reg [0:0] excZ_x_uid11_fpToFxPTest_q;
    wire [0:0] expXIsMax_uid12_fpToFxPTest_qi;
    reg [0:0] expXIsMax_uid12_fpToFxPTest_q;
    wire [0:0] fracXIsZero_uid13_fpToFxPTest_qi;
    reg [0:0] fracXIsZero_uid13_fpToFxPTest_q;
    wire [0:0] fracXIsNotZero_uid14_fpToFxPTest_q;
    wire [0:0] excI_x_uid15_fpToFxPTest_q;
    wire [0:0] excN_x_uid16_fpToFxPTest_q;
    wire [0:0] invExcXZ_uid22_fpToFxPTest_q;
    wire [23:0] oFracX_uid23_fpToFxPTest_q;
    wire [0:0] signX_uid25_fpToFxPTest_b;
    wire [8:0] ovfExpVal_uid26_fpToFxPTest_q;
    wire [10:0] ovfExpRange_uid27_fpToFxPTest_a;
    wire [10:0] ovfExpRange_uid27_fpToFxPTest_b;
    logic [10:0] ovfExpRange_uid27_fpToFxPTest_o;
    wire [0:0] ovfExpRange_uid27_fpToFxPTest_n;
    wire [7:0] udfExpVal_uid28_fpToFxPTest_q;
    wire [10:0] udf_uid29_fpToFxPTest_a;
    wire [10:0] udf_uid29_fpToFxPTest_b;
    logic [10:0] udf_uid29_fpToFxPTest_o;
    wire [0:0] udf_uid29_fpToFxPTest_n;
    wire [8:0] ovfExpVal_uid30_fpToFxPTest_q;
    wire [10:0] shiftValE_uid31_fpToFxPTest_a;
    wire [10:0] shiftValE_uid31_fpToFxPTest_b;
    logic [10:0] shiftValE_uid31_fpToFxPTest_o;
    wire [9:0] shiftValE_uid31_fpToFxPTest_q;
    wire [5:0] shiftValRaw_uid32_fpToFxPTest_in;
    wire [5:0] shiftValRaw_uid32_fpToFxPTest_b;
    wire [5:0] maxShiftCst_uid33_fpToFxPTest_q;
    wire [11:0] shiftOutOfRange_uid34_fpToFxPTest_a;
    wire [11:0] shiftOutOfRange_uid34_fpToFxPTest_b;
    logic [11:0] shiftOutOfRange_uid34_fpToFxPTest_o;
    wire [0:0] shiftOutOfRange_uid34_fpToFxPTest_n;
    wire [0:0] shiftVal_uid35_fpToFxPTest_s;
    reg [5:0] shiftVal_uid35_fpToFxPTest_q;
    wire [31:0] shifterIn_uid37_fpToFxPTest_q;
    wire [31:0] maxPosValueS_uid39_fpToFxPTest_q;
    wire [31:0] maxNegValueS_uid40_fpToFxPTest_q;
    wire [32:0] zRightShiferNoStickyOut_uid41_fpToFxPTest_q;
    wire [32:0] xXorSignE_uid42_fpToFxPTest_b;
    wire [32:0] xXorSignE_uid42_fpToFxPTest_q;
    wire [2:0] d0_uid43_fpToFxPTest_q;
    wire [33:0] sPostRndFull_uid44_fpToFxPTest_a;
    wire [33:0] sPostRndFull_uid44_fpToFxPTest_b;
    logic [33:0] sPostRndFull_uid44_fpToFxPTest_o;
    wire [33:0] sPostRndFull_uid44_fpToFxPTest_q;
    wire [32:0] sPostRnd_uid45_fpToFxPTest_in;
    wire [31:0] sPostRnd_uid45_fpToFxPTest_b;
    wire [34:0] sPostRnd_uid46_fpToFxPTest_in;
    wire [33:0] sPostRnd_uid46_fpToFxPTest_b;
    wire [35:0] rndOvfPos_uid47_fpToFxPTest_a;
    wire [35:0] rndOvfPos_uid47_fpToFxPTest_b;
    logic [35:0] rndOvfPos_uid47_fpToFxPTest_o;
    wire [0:0] rndOvfPos_uid47_fpToFxPTest_c;
    wire [0:0] ovfPostRnd_uid48_fpToFxPTest_q;
    wire [2:0] muxSelConc_uid49_fpToFxPTest_q;
    reg [1:0] muxSel_uid50_fpToFxPTest_q;
    wire [31:0] maxNegValueU_uid51_fpToFxPTest_q;
    wire [1:0] finalOut_uid52_fpToFxPTest_s;
    reg [31:0] finalOut_uid52_fpToFxPTest_q;
    wire [15:0] rightShiftStage0Idx1Rng16_uid56_rightShiferNoStickyOut_uid38_fpToFxPTest_b;
    wire [15:0] rightShiftStage0Idx1Pad16_uid57_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
    wire [31:0] rightShiftStage0Idx1_uid58_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
    wire [1:0] rightShiftStage0_uid62_rightShiferNoStickyOut_uid38_fpToFxPTest_s;
    reg [31:0] rightShiftStage0_uid62_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
    wire [27:0] rightShiftStage1Idx1Rng4_uid63_rightShiferNoStickyOut_uid38_fpToFxPTest_b;
    wire [3:0] rightShiftStage1Idx1Pad4_uid64_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
    wire [31:0] rightShiftStage1Idx1_uid65_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
    wire [23:0] rightShiftStage1Idx2Rng8_uid66_rightShiferNoStickyOut_uid38_fpToFxPTest_b;
    wire [31:0] rightShiftStage1Idx2_uid68_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
    wire [19:0] rightShiftStage1Idx3Rng12_uid69_rightShiferNoStickyOut_uid38_fpToFxPTest_b;
    wire [11:0] rightShiftStage1Idx3Pad12_uid70_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
    wire [31:0] rightShiftStage1Idx3_uid71_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
    wire [1:0] rightShiftStage1_uid73_rightShiferNoStickyOut_uid38_fpToFxPTest_s;
    reg [31:0] rightShiftStage1_uid73_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
    wire [30:0] rightShiftStage2Idx1Rng1_uid74_rightShiferNoStickyOut_uid38_fpToFxPTest_b;
    wire [31:0] rightShiftStage2Idx1_uid76_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
    wire [29:0] rightShiftStage2Idx2Rng2_uid77_rightShiferNoStickyOut_uid38_fpToFxPTest_b;
    wire [1:0] rightShiftStage2Idx2Pad2_uid78_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
    wire [31:0] rightShiftStage2Idx2_uid79_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
    wire [28:0] rightShiftStage2Idx3Rng3_uid80_rightShiferNoStickyOut_uid38_fpToFxPTest_b;
    wire [2:0] rightShiftStage2Idx3Pad3_uid81_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
    wire [31:0] rightShiftStage2Idx3_uid82_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
    wire [1:0] rightShiftStage2_uid84_rightShiferNoStickyOut_uid38_fpToFxPTest_s;
    reg [31:0] rightShiftStage2_uid84_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
    wire [1:0] rightShiftStageSel5Dto4_uid61_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_b;
    wire [1:0] rightShiftStageSel5Dto4_uid61_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_c;
    wire [1:0] rightShiftStageSel5Dto4_uid61_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_d;
    reg [31:0] redist0_sPostRnd_uid45_fpToFxPTest_b_1_q;
    reg [5:0] redist1_shiftValRaw_uid32_fpToFxPTest_b_1_q;
    reg [0:0] redist2_udf_uid29_fpToFxPTest_n_3_q;
    reg [0:0] redist3_ovfExpRange_uid27_fpToFxPTest_n_3_q;
    reg [0:0] redist4_signX_uid25_fpToFxPTest_b_2_q;
    reg [0:0] redist5_signX_uid25_fpToFxPTest_b_3_q;
    reg [0:0] redist6_fracXIsZero_uid13_fpToFxPTest_q_2_q;
    reg [0:0] redist7_expXIsMax_uid12_fpToFxPTest_q_3_q;
    reg [22:0] redist8_frac_x_uid10_fpToFxPTest_b_1_q;


    // maxNegValueU_uid51_fpToFxPTest(CONSTANT,50)
    assign maxNegValueU_uid51_fpToFxPTest_q = 32'b00000000000000000000000000000000;

    // maxNegValueS_uid40_fpToFxPTest(CONSTANT,39)
    assign maxNegValueS_uid40_fpToFxPTest_q = 32'b10000000000000000000000000000000;

    // maxPosValueS_uid39_fpToFxPTest(CONSTANT,38)
    assign maxPosValueS_uid39_fpToFxPTest_q = 32'b01111111111111111111111111111111;

    // d0_uid43_fpToFxPTest(CONSTANT,42)
    assign d0_uid43_fpToFxPTest_q = 3'b001;

    // signX_uid25_fpToFxPTest(BITSELECT,24)@0
    assign signX_uid25_fpToFxPTest_b = a[31:31];

    // redist4_signX_uid25_fpToFxPTest_b_2(DELAY,90)
    dspba_delay_ver #( .width(1), .depth(2), .reset_kind("ASYNC") )
    redist4_signX_uid25_fpToFxPTest_b_2 ( .xin(signX_uid25_fpToFxPTest_b), .xout(redist4_signX_uid25_fpToFxPTest_b_2_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // GND(CONSTANT,0)
    assign GND_q = 1'b0;

    // rightShiftStage2Idx3Pad3_uid81_rightShiferNoStickyOut_uid38_fpToFxPTest(CONSTANT,80)
    assign rightShiftStage2Idx3Pad3_uid81_rightShiferNoStickyOut_uid38_fpToFxPTest_q = 3'b000;

    // rightShiftStage2Idx3Rng3_uid80_rightShiferNoStickyOut_uid38_fpToFxPTest(BITSELECT,79)@1
    assign rightShiftStage2Idx3Rng3_uid80_rightShiferNoStickyOut_uid38_fpToFxPTest_b = rightShiftStage1_uid73_rightShiferNoStickyOut_uid38_fpToFxPTest_q[31:3];

    // rightShiftStage2Idx3_uid82_rightShiferNoStickyOut_uid38_fpToFxPTest(BITJOIN,81)@1
    assign rightShiftStage2Idx3_uid82_rightShiferNoStickyOut_uid38_fpToFxPTest_q = {rightShiftStage2Idx3Pad3_uid81_rightShiferNoStickyOut_uid38_fpToFxPTest_q, rightShiftStage2Idx3Rng3_uid80_rightShiferNoStickyOut_uid38_fpToFxPTest_b};

    // rightShiftStage2Idx2Pad2_uid78_rightShiferNoStickyOut_uid38_fpToFxPTest(CONSTANT,77)
    assign rightShiftStage2Idx2Pad2_uid78_rightShiferNoStickyOut_uid38_fpToFxPTest_q = 2'b00;

    // rightShiftStage2Idx2Rng2_uid77_rightShiferNoStickyOut_uid38_fpToFxPTest(BITSELECT,76)@1
    assign rightShiftStage2Idx2Rng2_uid77_rightShiferNoStickyOut_uid38_fpToFxPTest_b = rightShiftStage1_uid73_rightShiferNoStickyOut_uid38_fpToFxPTest_q[31:2];

    // rightShiftStage2Idx2_uid79_rightShiferNoStickyOut_uid38_fpToFxPTest(BITJOIN,78)@1
    assign rightShiftStage2Idx2_uid79_rightShiferNoStickyOut_uid38_fpToFxPTest_q = {rightShiftStage2Idx2Pad2_uid78_rightShiferNoStickyOut_uid38_fpToFxPTest_q, rightShiftStage2Idx2Rng2_uid77_rightShiferNoStickyOut_uid38_fpToFxPTest_b};

    // rightShiftStage2Idx1Rng1_uid74_rightShiferNoStickyOut_uid38_fpToFxPTest(BITSELECT,73)@1
    assign rightShiftStage2Idx1Rng1_uid74_rightShiferNoStickyOut_uid38_fpToFxPTest_b = rightShiftStage1_uid73_rightShiferNoStickyOut_uid38_fpToFxPTest_q[31:1];

    // rightShiftStage2Idx1_uid76_rightShiferNoStickyOut_uid38_fpToFxPTest(BITJOIN,75)@1
    assign rightShiftStage2Idx1_uid76_rightShiferNoStickyOut_uid38_fpToFxPTest_q = {GND_q, rightShiftStage2Idx1Rng1_uid74_rightShiferNoStickyOut_uid38_fpToFxPTest_b};

    // rightShiftStage1Idx3Pad12_uid70_rightShiferNoStickyOut_uid38_fpToFxPTest(CONSTANT,69)
    assign rightShiftStage1Idx3Pad12_uid70_rightShiferNoStickyOut_uid38_fpToFxPTest_q = 12'b000000000000;

    // rightShiftStage1Idx3Rng12_uid69_rightShiferNoStickyOut_uid38_fpToFxPTest(BITSELECT,68)@1
    assign rightShiftStage1Idx3Rng12_uid69_rightShiferNoStickyOut_uid38_fpToFxPTest_b = rightShiftStage0_uid62_rightShiferNoStickyOut_uid38_fpToFxPTest_q[31:12];

    // rightShiftStage1Idx3_uid71_rightShiferNoStickyOut_uid38_fpToFxPTest(BITJOIN,70)@1
    assign rightShiftStage1Idx3_uid71_rightShiferNoStickyOut_uid38_fpToFxPTest_q = {rightShiftStage1Idx3Pad12_uid70_rightShiferNoStickyOut_uid38_fpToFxPTest_q, rightShiftStage1Idx3Rng12_uid69_rightShiferNoStickyOut_uid38_fpToFxPTest_b};

    // cstAllZWE_uid8_fpToFxPTest(CONSTANT,7)
    assign cstAllZWE_uid8_fpToFxPTest_q = 8'b00000000;

    // rightShiftStage1Idx2Rng8_uid66_rightShiferNoStickyOut_uid38_fpToFxPTest(BITSELECT,65)@1
    assign rightShiftStage1Idx2Rng8_uid66_rightShiferNoStickyOut_uid38_fpToFxPTest_b = rightShiftStage0_uid62_rightShiferNoStickyOut_uid38_fpToFxPTest_q[31:8];

    // rightShiftStage1Idx2_uid68_rightShiferNoStickyOut_uid38_fpToFxPTest(BITJOIN,67)@1
    assign rightShiftStage1Idx2_uid68_rightShiferNoStickyOut_uid38_fpToFxPTest_q = {cstAllZWE_uid8_fpToFxPTest_q, rightShiftStage1Idx2Rng8_uid66_rightShiferNoStickyOut_uid38_fpToFxPTest_b};

    // rightShiftStage1Idx1Pad4_uid64_rightShiferNoStickyOut_uid38_fpToFxPTest(CONSTANT,63)
    assign rightShiftStage1Idx1Pad4_uid64_rightShiferNoStickyOut_uid38_fpToFxPTest_q = 4'b0000;

    // rightShiftStage1Idx1Rng4_uid63_rightShiferNoStickyOut_uid38_fpToFxPTest(BITSELECT,62)@1
    assign rightShiftStage1Idx1Rng4_uid63_rightShiferNoStickyOut_uid38_fpToFxPTest_b = rightShiftStage0_uid62_rightShiferNoStickyOut_uid38_fpToFxPTest_q[31:4];

    // rightShiftStage1Idx1_uid65_rightShiferNoStickyOut_uid38_fpToFxPTest(BITJOIN,64)@1
    assign rightShiftStage1Idx1_uid65_rightShiferNoStickyOut_uid38_fpToFxPTest_q = {rightShiftStage1Idx1Pad4_uid64_rightShiferNoStickyOut_uid38_fpToFxPTest_q, rightShiftStage1Idx1Rng4_uid63_rightShiferNoStickyOut_uid38_fpToFxPTest_b};

    // rightShiftStage0Idx1Pad16_uid57_rightShiferNoStickyOut_uid38_fpToFxPTest(CONSTANT,56)
    assign rightShiftStage0Idx1Pad16_uid57_rightShiferNoStickyOut_uid38_fpToFxPTest_q = 16'b0000000000000000;

    // rightShiftStage0Idx1Rng16_uid56_rightShiferNoStickyOut_uid38_fpToFxPTest(BITSELECT,55)@1
    assign rightShiftStage0Idx1Rng16_uid56_rightShiferNoStickyOut_uid38_fpToFxPTest_b = shifterIn_uid37_fpToFxPTest_q[31:16];

    // rightShiftStage0Idx1_uid58_rightShiferNoStickyOut_uid38_fpToFxPTest(BITJOIN,57)@1
    assign rightShiftStage0Idx1_uid58_rightShiferNoStickyOut_uid38_fpToFxPTest_q = {rightShiftStage0Idx1Pad16_uid57_rightShiferNoStickyOut_uid38_fpToFxPTest_q, rightShiftStage0Idx1Rng16_uid56_rightShiferNoStickyOut_uid38_fpToFxPTest_b};

    // exp_x_uid9_fpToFxPTest(BITSELECT,8)@0
    assign exp_x_uid9_fpToFxPTest_b = a[30:23];

    // excZ_x_uid11_fpToFxPTest(LOGICAL,10)@0 + 1
    assign excZ_x_uid11_fpToFxPTest_qi = exp_x_uid9_fpToFxPTest_b == cstAllZWE_uid8_fpToFxPTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    excZ_x_uid11_fpToFxPTest_delay ( .xin(excZ_x_uid11_fpToFxPTest_qi), .xout(excZ_x_uid11_fpToFxPTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // invExcXZ_uid22_fpToFxPTest(LOGICAL,21)@1
    assign invExcXZ_uid22_fpToFxPTest_q = ~ (excZ_x_uid11_fpToFxPTest_q);

    // frac_x_uid10_fpToFxPTest(BITSELECT,9)@0
    assign frac_x_uid10_fpToFxPTest_b = a[22:0];

    // redist8_frac_x_uid10_fpToFxPTest_b_1(DELAY,94)
    dspba_delay_ver #( .width(23), .depth(1), .reset_kind("ASYNC") )
    redist8_frac_x_uid10_fpToFxPTest_b_1 ( .xin(frac_x_uid10_fpToFxPTest_b), .xout(redist8_frac_x_uid10_fpToFxPTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // oFracX_uid23_fpToFxPTest(BITJOIN,22)@1
    assign oFracX_uid23_fpToFxPTest_q = {invExcXZ_uid22_fpToFxPTest_q, redist8_frac_x_uid10_fpToFxPTest_b_1_q};

    // shifterIn_uid37_fpToFxPTest(BITJOIN,36)@1
    assign shifterIn_uid37_fpToFxPTest_q = {oFracX_uid23_fpToFxPTest_q, cstAllZWE_uid8_fpToFxPTest_q};

    // rightShiftStage0_uid62_rightShiferNoStickyOut_uid38_fpToFxPTest(MUX,61)@1
    assign rightShiftStage0_uid62_rightShiferNoStickyOut_uid38_fpToFxPTest_s = rightShiftStageSel5Dto4_uid61_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_b;
    always @(rightShiftStage0_uid62_rightShiferNoStickyOut_uid38_fpToFxPTest_s or en or shifterIn_uid37_fpToFxPTest_q or rightShiftStage0Idx1_uid58_rightShiferNoStickyOut_uid38_fpToFxPTest_q or maxNegValueU_uid51_fpToFxPTest_q)
    begin
        unique case (rightShiftStage0_uid62_rightShiferNoStickyOut_uid38_fpToFxPTest_s)
            2'b00 : rightShiftStage0_uid62_rightShiferNoStickyOut_uid38_fpToFxPTest_q = shifterIn_uid37_fpToFxPTest_q;
            2'b01 : rightShiftStage0_uid62_rightShiferNoStickyOut_uid38_fpToFxPTest_q = rightShiftStage0Idx1_uid58_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
            2'b10 : rightShiftStage0_uid62_rightShiferNoStickyOut_uid38_fpToFxPTest_q = maxNegValueU_uid51_fpToFxPTest_q;
            2'b11 : rightShiftStage0_uid62_rightShiferNoStickyOut_uid38_fpToFxPTest_q = maxNegValueU_uid51_fpToFxPTest_q;
            default : rightShiftStage0_uid62_rightShiferNoStickyOut_uid38_fpToFxPTest_q = 32'b0;
        endcase
    end

    // rightShiftStage1_uid73_rightShiferNoStickyOut_uid38_fpToFxPTest(MUX,72)@1
    assign rightShiftStage1_uid73_rightShiferNoStickyOut_uid38_fpToFxPTest_s = rightShiftStageSel5Dto4_uid61_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_c;
    always @(rightShiftStage1_uid73_rightShiferNoStickyOut_uid38_fpToFxPTest_s or en or rightShiftStage0_uid62_rightShiferNoStickyOut_uid38_fpToFxPTest_q or rightShiftStage1Idx1_uid65_rightShiferNoStickyOut_uid38_fpToFxPTest_q or rightShiftStage1Idx2_uid68_rightShiferNoStickyOut_uid38_fpToFxPTest_q or rightShiftStage1Idx3_uid71_rightShiferNoStickyOut_uid38_fpToFxPTest_q)
    begin
        unique case (rightShiftStage1_uid73_rightShiferNoStickyOut_uid38_fpToFxPTest_s)
            2'b00 : rightShiftStage1_uid73_rightShiferNoStickyOut_uid38_fpToFxPTest_q = rightShiftStage0_uid62_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
            2'b01 : rightShiftStage1_uid73_rightShiferNoStickyOut_uid38_fpToFxPTest_q = rightShiftStage1Idx1_uid65_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
            2'b10 : rightShiftStage1_uid73_rightShiferNoStickyOut_uid38_fpToFxPTest_q = rightShiftStage1Idx2_uid68_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
            2'b11 : rightShiftStage1_uid73_rightShiferNoStickyOut_uid38_fpToFxPTest_q = rightShiftStage1Idx3_uid71_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
            default : rightShiftStage1_uid73_rightShiferNoStickyOut_uid38_fpToFxPTest_q = 32'b0;
        endcase
    end

    // maxShiftCst_uid33_fpToFxPTest(CONSTANT,32)
    assign maxShiftCst_uid33_fpToFxPTest_q = 6'b100000;

    // ovfExpVal_uid30_fpToFxPTest(CONSTANT,29)
    assign ovfExpVal_uid30_fpToFxPTest_q = 9'b010011101;

    // shiftValE_uid31_fpToFxPTest(SUB,30)@0
    assign shiftValE_uid31_fpToFxPTest_a = {{2{ovfExpVal_uid30_fpToFxPTest_q[8]}}, ovfExpVal_uid30_fpToFxPTest_q};
    assign shiftValE_uid31_fpToFxPTest_b = {3'b000, exp_x_uid9_fpToFxPTest_b};
    assign shiftValE_uid31_fpToFxPTest_o = $signed(shiftValE_uid31_fpToFxPTest_a) - $signed(shiftValE_uid31_fpToFxPTest_b);
    assign shiftValE_uid31_fpToFxPTest_q = shiftValE_uid31_fpToFxPTest_o[9:0];

    // shiftValRaw_uid32_fpToFxPTest(BITSELECT,31)@0
    assign shiftValRaw_uid32_fpToFxPTest_in = shiftValE_uid31_fpToFxPTest_q[5:0];
    assign shiftValRaw_uid32_fpToFxPTest_b = shiftValRaw_uid32_fpToFxPTest_in[5:0];

    // redist1_shiftValRaw_uid32_fpToFxPTest_b_1(DELAY,87)
    dspba_delay_ver #( .width(6), .depth(1), .reset_kind("ASYNC") )
    redist1_shiftValRaw_uid32_fpToFxPTest_b_1 ( .xin(shiftValRaw_uid32_fpToFxPTest_b), .xout(redist1_shiftValRaw_uid32_fpToFxPTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // shiftOutOfRange_uid34_fpToFxPTest(COMPARE,33)@0 + 1
    assign shiftOutOfRange_uid34_fpToFxPTest_a = {{2{shiftValE_uid31_fpToFxPTest_q[9]}}, shiftValE_uid31_fpToFxPTest_q};
    assign shiftOutOfRange_uid34_fpToFxPTest_b = {6'b000000, maxShiftCst_uid33_fpToFxPTest_q};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            shiftOutOfRange_uid34_fpToFxPTest_o <= 12'b0;
        end
        else if (en == 1'b1)
        begin
            shiftOutOfRange_uid34_fpToFxPTest_o <= $signed(shiftOutOfRange_uid34_fpToFxPTest_a) - $signed(shiftOutOfRange_uid34_fpToFxPTest_b);
        end
    end
    assign shiftOutOfRange_uid34_fpToFxPTest_n[0] = ~ (shiftOutOfRange_uid34_fpToFxPTest_o[11]);

    // shiftVal_uid35_fpToFxPTest(MUX,34)@1
    assign shiftVal_uid35_fpToFxPTest_s = shiftOutOfRange_uid34_fpToFxPTest_n;
    always @(shiftVal_uid35_fpToFxPTest_s or en or redist1_shiftValRaw_uid32_fpToFxPTest_b_1_q or maxShiftCst_uid33_fpToFxPTest_q)
    begin
        unique case (shiftVal_uid35_fpToFxPTest_s)
            1'b0 : shiftVal_uid35_fpToFxPTest_q = redist1_shiftValRaw_uid32_fpToFxPTest_b_1_q;
            1'b1 : shiftVal_uid35_fpToFxPTest_q = maxShiftCst_uid33_fpToFxPTest_q;
            default : shiftVal_uid35_fpToFxPTest_q = 6'b0;
        endcase
    end

    // rightShiftStageSel5Dto4_uid61_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select(BITSELECT,85)@1
    assign rightShiftStageSel5Dto4_uid61_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_b = shiftVal_uid35_fpToFxPTest_q[5:4];
    assign rightShiftStageSel5Dto4_uid61_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_c = shiftVal_uid35_fpToFxPTest_q[3:2];
    assign rightShiftStageSel5Dto4_uid61_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_d = shiftVal_uid35_fpToFxPTest_q[1:0];

    // rightShiftStage2_uid84_rightShiferNoStickyOut_uid38_fpToFxPTest(MUX,83)@1 + 1
    assign rightShiftStage2_uid84_rightShiferNoStickyOut_uid38_fpToFxPTest_s = rightShiftStageSel5Dto4_uid61_rightShiferNoStickyOut_uid38_fpToFxPTest_merged_bit_select_d;
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            rightShiftStage2_uid84_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= 32'b0;
        end
        else if (en == 1'b1)
        begin
            unique case (rightShiftStage2_uid84_rightShiferNoStickyOut_uid38_fpToFxPTest_s)
                2'b00 : rightShiftStage2_uid84_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage1_uid73_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
                2'b01 : rightShiftStage2_uid84_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage2Idx1_uid76_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
                2'b10 : rightShiftStage2_uid84_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage2Idx2_uid79_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
                2'b11 : rightShiftStage2_uid84_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= rightShiftStage2Idx3_uid82_rightShiferNoStickyOut_uid38_fpToFxPTest_q;
                default : rightShiftStage2_uid84_rightShiferNoStickyOut_uid38_fpToFxPTest_q <= 32'b0;
            endcase
        end
    end

    // zRightShiferNoStickyOut_uid41_fpToFxPTest(BITJOIN,40)@2
    assign zRightShiferNoStickyOut_uid41_fpToFxPTest_q = {GND_q, rightShiftStage2_uid84_rightShiferNoStickyOut_uid38_fpToFxPTest_q};

    // xXorSignE_uid42_fpToFxPTest(LOGICAL,41)@2
    assign xXorSignE_uid42_fpToFxPTest_b = {{32{redist4_signX_uid25_fpToFxPTest_b_2_q[0]}}, redist4_signX_uid25_fpToFxPTest_b_2_q};
    assign xXorSignE_uid42_fpToFxPTest_q = zRightShiferNoStickyOut_uid41_fpToFxPTest_q ^ xXorSignE_uid42_fpToFxPTest_b;

    // sPostRndFull_uid44_fpToFxPTest(ADD,43)@2
    assign sPostRndFull_uid44_fpToFxPTest_a = {{1{xXorSignE_uid42_fpToFxPTest_q[32]}}, xXorSignE_uid42_fpToFxPTest_q};
    assign sPostRndFull_uid44_fpToFxPTest_b = {{31{d0_uid43_fpToFxPTest_q[2]}}, d0_uid43_fpToFxPTest_q};
    assign sPostRndFull_uid44_fpToFxPTest_o = $signed(sPostRndFull_uid44_fpToFxPTest_a) + $signed(sPostRndFull_uid44_fpToFxPTest_b);
    assign sPostRndFull_uid44_fpToFxPTest_q = sPostRndFull_uid44_fpToFxPTest_o[33:0];

    // sPostRnd_uid45_fpToFxPTest(BITSELECT,44)@2
    assign sPostRnd_uid45_fpToFxPTest_in = sPostRndFull_uid44_fpToFxPTest_q[32:0];
    assign sPostRnd_uid45_fpToFxPTest_b = sPostRnd_uid45_fpToFxPTest_in[32:1];

    // redist0_sPostRnd_uid45_fpToFxPTest_b_1(DELAY,86)
    dspba_delay_ver #( .width(32), .depth(1), .reset_kind("ASYNC") )
    redist0_sPostRnd_uid45_fpToFxPTest_b_1 ( .xin(sPostRnd_uid45_fpToFxPTest_b), .xout(redist0_sPostRnd_uid45_fpToFxPTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist5_signX_uid25_fpToFxPTest_b_3(DELAY,91)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist5_signX_uid25_fpToFxPTest_b_3 ( .xin(redist4_signX_uid25_fpToFxPTest_b_2_q), .xout(redist5_signX_uid25_fpToFxPTest_b_3_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // udfExpVal_uid28_fpToFxPTest(CONSTANT,27)
    assign udfExpVal_uid28_fpToFxPTest_q = 8'b01111101;

    // udf_uid29_fpToFxPTest(COMPARE,28)@0 + 1
    assign udf_uid29_fpToFxPTest_a = {{3{udfExpVal_uid28_fpToFxPTest_q[7]}}, udfExpVal_uid28_fpToFxPTest_q};
    assign udf_uid29_fpToFxPTest_b = {3'b000, exp_x_uid9_fpToFxPTest_b};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            udf_uid29_fpToFxPTest_o <= 11'b0;
        end
        else if (en == 1'b1)
        begin
            udf_uid29_fpToFxPTest_o <= $signed(udf_uid29_fpToFxPTest_a) - $signed(udf_uid29_fpToFxPTest_b);
        end
    end
    assign udf_uid29_fpToFxPTest_n[0] = ~ (udf_uid29_fpToFxPTest_o[10]);

    // redist2_udf_uid29_fpToFxPTest_n_3(DELAY,88)
    dspba_delay_ver #( .width(1), .depth(2), .reset_kind("ASYNC") )
    redist2_udf_uid29_fpToFxPTest_n_3 ( .xin(udf_uid29_fpToFxPTest_n), .xout(redist2_udf_uid29_fpToFxPTest_n_3_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // sPostRnd_uid46_fpToFxPTest(BITSELECT,45)@2
    assign sPostRnd_uid46_fpToFxPTest_in = {{1{sPostRndFull_uid44_fpToFxPTest_q[33]}}, sPostRndFull_uid44_fpToFxPTest_q};
    assign sPostRnd_uid46_fpToFxPTest_b = sPostRnd_uid46_fpToFxPTest_in[34:1];

    // rndOvfPos_uid47_fpToFxPTest(COMPARE,46)@2 + 1
    assign rndOvfPos_uid47_fpToFxPTest_a = {4'b0000, maxPosValueS_uid39_fpToFxPTest_q};
    assign rndOvfPos_uid47_fpToFxPTest_b = {{2{sPostRnd_uid46_fpToFxPTest_b[33]}}, sPostRnd_uid46_fpToFxPTest_b};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            rndOvfPos_uid47_fpToFxPTest_o <= 36'b0;
        end
        else if (en == 1'b1)
        begin
            rndOvfPos_uid47_fpToFxPTest_o <= $signed(rndOvfPos_uid47_fpToFxPTest_a) - $signed(rndOvfPos_uid47_fpToFxPTest_b);
        end
    end
    assign rndOvfPos_uid47_fpToFxPTest_c[0] = rndOvfPos_uid47_fpToFxPTest_o[35];

    // ovfExpVal_uid26_fpToFxPTest(CONSTANT,25)
    assign ovfExpVal_uid26_fpToFxPTest_q = 9'b010011110;

    // ovfExpRange_uid27_fpToFxPTest(COMPARE,26)@0 + 1
    assign ovfExpRange_uid27_fpToFxPTest_a = {3'b000, exp_x_uid9_fpToFxPTest_b};
    assign ovfExpRange_uid27_fpToFxPTest_b = {{2{ovfExpVal_uid26_fpToFxPTest_q[8]}}, ovfExpVal_uid26_fpToFxPTest_q};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            ovfExpRange_uid27_fpToFxPTest_o <= 11'b0;
        end
        else if (en == 1'b1)
        begin
            ovfExpRange_uid27_fpToFxPTest_o <= $signed(ovfExpRange_uid27_fpToFxPTest_a) - $signed(ovfExpRange_uid27_fpToFxPTest_b);
        end
    end
    assign ovfExpRange_uid27_fpToFxPTest_n[0] = ~ (ovfExpRange_uid27_fpToFxPTest_o[10]);

    // redist3_ovfExpRange_uid27_fpToFxPTest_n_3(DELAY,89)
    dspba_delay_ver #( .width(1), .depth(2), .reset_kind("ASYNC") )
    redist3_ovfExpRange_uid27_fpToFxPTest_n_3 ( .xin(ovfExpRange_uid27_fpToFxPTest_n), .xout(redist3_ovfExpRange_uid27_fpToFxPTest_n_3_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // cstZeroWF_uid7_fpToFxPTest(CONSTANT,6)
    assign cstZeroWF_uid7_fpToFxPTest_q = 23'b00000000000000000000000;

    // fracXIsZero_uid13_fpToFxPTest(LOGICAL,12)@1 + 1
    assign fracXIsZero_uid13_fpToFxPTest_qi = cstZeroWF_uid7_fpToFxPTest_q == redist8_frac_x_uid10_fpToFxPTest_b_1_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    fracXIsZero_uid13_fpToFxPTest_delay ( .xin(fracXIsZero_uid13_fpToFxPTest_qi), .xout(fracXIsZero_uid13_fpToFxPTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist6_fracXIsZero_uid13_fpToFxPTest_q_2(DELAY,92)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist6_fracXIsZero_uid13_fpToFxPTest_q_2 ( .xin(fracXIsZero_uid13_fpToFxPTest_q), .xout(redist6_fracXIsZero_uid13_fpToFxPTest_q_2_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // cstAllOWE_uid6_fpToFxPTest(CONSTANT,5)
    assign cstAllOWE_uid6_fpToFxPTest_q = 8'b11111111;

    // expXIsMax_uid12_fpToFxPTest(LOGICAL,11)@0 + 1
    assign expXIsMax_uid12_fpToFxPTest_qi = exp_x_uid9_fpToFxPTest_b == cstAllOWE_uid6_fpToFxPTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    expXIsMax_uid12_fpToFxPTest_delay ( .xin(expXIsMax_uid12_fpToFxPTest_qi), .xout(expXIsMax_uid12_fpToFxPTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist7_expXIsMax_uid12_fpToFxPTest_q_3(DELAY,93)
    dspba_delay_ver #( .width(1), .depth(2), .reset_kind("ASYNC") )
    redist7_expXIsMax_uid12_fpToFxPTest_q_3 ( .xin(expXIsMax_uid12_fpToFxPTest_q), .xout(redist7_expXIsMax_uid12_fpToFxPTest_q_3_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // excI_x_uid15_fpToFxPTest(LOGICAL,14)@3
    assign excI_x_uid15_fpToFxPTest_q = redist7_expXIsMax_uid12_fpToFxPTest_q_3_q & redist6_fracXIsZero_uid13_fpToFxPTest_q_2_q;

    // fracXIsNotZero_uid14_fpToFxPTest(LOGICAL,13)@3
    assign fracXIsNotZero_uid14_fpToFxPTest_q = ~ (redist6_fracXIsZero_uid13_fpToFxPTest_q_2_q);

    // excN_x_uid16_fpToFxPTest(LOGICAL,15)@3
    assign excN_x_uid16_fpToFxPTest_q = redist7_expXIsMax_uid12_fpToFxPTest_q_3_q & fracXIsNotZero_uid14_fpToFxPTest_q;

    // ovfPostRnd_uid48_fpToFxPTest(LOGICAL,47)@3
    assign ovfPostRnd_uid48_fpToFxPTest_q = excN_x_uid16_fpToFxPTest_q | excI_x_uid15_fpToFxPTest_q | redist3_ovfExpRange_uid27_fpToFxPTest_n_3_q | rndOvfPos_uid47_fpToFxPTest_c;

    // muxSelConc_uid49_fpToFxPTest(BITJOIN,48)@3
    assign muxSelConc_uid49_fpToFxPTest_q = {redist5_signX_uid25_fpToFxPTest_b_3_q, redist2_udf_uid29_fpToFxPTest_n_3_q, ovfPostRnd_uid48_fpToFxPTest_q};

    // muxSel_uid50_fpToFxPTest(LOOKUP,49)@3
    always @(muxSelConc_uid49_fpToFxPTest_q)
    begin
        // Begin reserved scope level
        unique case (muxSelConc_uid49_fpToFxPTest_q)
            3'b000 : muxSel_uid50_fpToFxPTest_q = 2'b00;
            3'b001 : muxSel_uid50_fpToFxPTest_q = 2'b01;
            3'b010 : muxSel_uid50_fpToFxPTest_q = 2'b11;
            3'b011 : muxSel_uid50_fpToFxPTest_q = 2'b11;
            3'b100 : muxSel_uid50_fpToFxPTest_q = 2'b00;
            3'b101 : muxSel_uid50_fpToFxPTest_q = 2'b10;
            3'b110 : muxSel_uid50_fpToFxPTest_q = 2'b11;
            3'b111 : muxSel_uid50_fpToFxPTest_q = 2'b11;
            default : begin
                          // unreachable
                          muxSel_uid50_fpToFxPTest_q = 2'bxx;
                      end
        endcase
        // End reserved scope level
    end

    // finalOut_uid52_fpToFxPTest(MUX,51)@3
    assign finalOut_uid52_fpToFxPTest_s = muxSel_uid50_fpToFxPTest_q;
    always @(finalOut_uid52_fpToFxPTest_s or en or redist0_sPostRnd_uid45_fpToFxPTest_b_1_q or maxPosValueS_uid39_fpToFxPTest_q or maxNegValueS_uid40_fpToFxPTest_q or maxNegValueU_uid51_fpToFxPTest_q)
    begin
        unique case (finalOut_uid52_fpToFxPTest_s)
            2'b00 : finalOut_uid52_fpToFxPTest_q = redist0_sPostRnd_uid45_fpToFxPTest_b_1_q;
            2'b01 : finalOut_uid52_fpToFxPTest_q = maxPosValueS_uid39_fpToFxPTest_q;
            2'b10 : finalOut_uid52_fpToFxPTest_q = maxNegValueS_uid40_fpToFxPTest_q;
            2'b11 : finalOut_uid52_fpToFxPTest_q = maxNegValueU_uid51_fpToFxPTest_q;
            default : finalOut_uid52_fpToFxPTest_q = 32'b0;
        endcase
    end

    // xOut(GPOUT,4)@3
    assign q = finalOut_uid52_fpToFxPTest_q;

endmodule
