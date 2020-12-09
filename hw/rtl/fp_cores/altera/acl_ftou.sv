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

// SystemVerilog created from acl_ftou
// SystemVerilog created on Wed Dec  9 01:17:51 2020


(* altera_attribute = "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007; -name MESSAGE_DISABLE 10958" *)
module acl_ftou (
    input wire [31:0] a,
    input wire [0:0] en,
    output wire [31:0] q,
    input wire clk,
    input wire areset
    );

    wire [0:0] GND_q;
    wire [0:0] VCC_q;
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
    wire [10:0] ovf_uid27_fpToFxPTest_a;
    wire [10:0] ovf_uid27_fpToFxPTest_b;
    logic [10:0] ovf_uid27_fpToFxPTest_o;
    wire [0:0] ovf_uid27_fpToFxPTest_n;
    wire [0:0] negOrOvf_uid28_fpToFxPTest_q;
    wire [7:0] udfExpVal_uid29_fpToFxPTest_q;
    wire [10:0] udf_uid30_fpToFxPTest_a;
    wire [10:0] udf_uid30_fpToFxPTest_b;
    logic [10:0] udf_uid30_fpToFxPTest_o;
    wire [0:0] udf_uid30_fpToFxPTest_n;
    wire [8:0] ovfExpVal_uid31_fpToFxPTest_q;
    wire [10:0] shiftValE_uid32_fpToFxPTest_a;
    wire [10:0] shiftValE_uid32_fpToFxPTest_b;
    logic [10:0] shiftValE_uid32_fpToFxPTest_o;
    wire [9:0] shiftValE_uid32_fpToFxPTest_q;
    wire [5:0] shiftValRaw_uid33_fpToFxPTest_in;
    wire [5:0] shiftValRaw_uid33_fpToFxPTest_b;
    wire [5:0] maxShiftCst_uid34_fpToFxPTest_q;
    wire [11:0] shiftOutOfRange_uid35_fpToFxPTest_a;
    wire [11:0] shiftOutOfRange_uid35_fpToFxPTest_b;
    logic [11:0] shiftOutOfRange_uid35_fpToFxPTest_o;
    wire [0:0] shiftOutOfRange_uid35_fpToFxPTest_n;
    wire [0:0] shiftVal_uid36_fpToFxPTest_s;
    reg [5:0] shiftVal_uid36_fpToFxPTest_q;
    wire [8:0] zPadd_uid37_fpToFxPTest_q;
    wire [32:0] shifterIn_uid38_fpToFxPTest_q;
    wire [31:0] maxPosValueU_uid40_fpToFxPTest_q;
    wire [31:0] maxNegValueU_uid41_fpToFxPTest_q;
    wire [33:0] zRightShiferNoStickyOut_uid43_fpToFxPTest_q;
    wire [34:0] sPostRndFull_uid44_fpToFxPTest_a;
    wire [34:0] sPostRndFull_uid44_fpToFxPTest_b;
    logic [34:0] sPostRndFull_uid44_fpToFxPTest_o;
    wire [34:0] sPostRndFull_uid44_fpToFxPTest_q;
    wire [32:0] sPostRnd_uid45_fpToFxPTest_in;
    wire [31:0] sPostRnd_uid45_fpToFxPTest_b;
    wire [33:0] sPostRndFullMSBU_uid46_fpToFxPTest_in;
    wire [0:0] sPostRndFullMSBU_uid46_fpToFxPTest_b;
    wire [0:0] ovfPostRnd_uid47_fpToFxPTest_q;
    wire [2:0] muxSelConc_uid48_fpToFxPTest_q;
    reg [1:0] muxSel_uid49_fpToFxPTest_q;
    wire [1:0] finalOut_uid51_fpToFxPTest_s;
    reg [31:0] finalOut_uid51_fpToFxPTest_q;
    wire [16:0] rightShiftStage0Idx1Rng16_uid55_rightShiferNoStickyOut_uid39_fpToFxPTest_b;
    wire [15:0] rightShiftStage0Idx1Pad16_uid56_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
    wire [32:0] rightShiftStage0Idx1_uid57_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
    wire [0:0] rightShiftStage0Idx2Rng32_uid58_rightShiferNoStickyOut_uid39_fpToFxPTest_b;
    wire [32:0] rightShiftStage0Idx2_uid60_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
    wire [32:0] rightShiftStage0Idx3_uid61_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
    wire [1:0] rightShiftStage0_uid63_rightShiferNoStickyOut_uid39_fpToFxPTest_s;
    reg [32:0] rightShiftStage0_uid63_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
    wire [28:0] rightShiftStage1Idx1Rng4_uid64_rightShiferNoStickyOut_uid39_fpToFxPTest_b;
    wire [3:0] rightShiftStage1Idx1Pad4_uid65_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
    wire [32:0] rightShiftStage1Idx1_uid66_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
    wire [24:0] rightShiftStage1Idx2Rng8_uid67_rightShiferNoStickyOut_uid39_fpToFxPTest_b;
    wire [32:0] rightShiftStage1Idx2_uid69_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
    wire [20:0] rightShiftStage1Idx3Rng12_uid70_rightShiferNoStickyOut_uid39_fpToFxPTest_b;
    wire [11:0] rightShiftStage1Idx3Pad12_uid71_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
    wire [32:0] rightShiftStage1Idx3_uid72_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
    wire [1:0] rightShiftStage1_uid74_rightShiferNoStickyOut_uid39_fpToFxPTest_s;
    reg [32:0] rightShiftStage1_uid74_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
    wire [31:0] rightShiftStage2Idx1Rng1_uid75_rightShiferNoStickyOut_uid39_fpToFxPTest_b;
    wire [32:0] rightShiftStage2Idx1_uid77_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
    wire [30:0] rightShiftStage2Idx2Rng2_uid78_rightShiferNoStickyOut_uid39_fpToFxPTest_b;
    wire [1:0] rightShiftStage2Idx2Pad2_uid79_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
    wire [32:0] rightShiftStage2Idx2_uid80_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
    wire [29:0] rightShiftStage2Idx3Rng3_uid81_rightShiferNoStickyOut_uid39_fpToFxPTest_b;
    wire [2:0] rightShiftStage2Idx3Pad3_uid82_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
    wire [32:0] rightShiftStage2Idx3_uid83_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
    wire [1:0] rightShiftStage2_uid85_rightShiferNoStickyOut_uid39_fpToFxPTest_s;
    reg [32:0] rightShiftStage2_uid85_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
    wire [1:0] rightShiftStageSel5Dto4_uid62_rightShiferNoStickyOut_uid39_fpToFxPTest_merged_bit_select_b;
    wire [1:0] rightShiftStageSel5Dto4_uid62_rightShiferNoStickyOut_uid39_fpToFxPTest_merged_bit_select_c;
    wire [1:0] rightShiftStageSel5Dto4_uid62_rightShiferNoStickyOut_uid39_fpToFxPTest_merged_bit_select_d;
    reg [31:0] redist0_sPostRnd_uid45_fpToFxPTest_b_1_q;
    reg [5:0] redist1_shiftValRaw_uid33_fpToFxPTest_b_1_q;
    reg [0:0] redist2_udf_uid30_fpToFxPTest_n_2_q;
    reg [0:0] redist3_ovf_uid27_fpToFxPTest_n_2_q;
    reg [0:0] redist4_signX_uid25_fpToFxPTest_b_2_q;
    reg [0:0] redist5_expXIsMax_uid12_fpToFxPTest_q_2_q;
    reg [22:0] redist6_frac_x_uid10_fpToFxPTest_b_1_q;


    // maxNegValueU_uid41_fpToFxPTest(CONSTANT,40)
    assign maxNegValueU_uid41_fpToFxPTest_q = 32'b00000000000000000000000000000000;

    // maxPosValueU_uid40_fpToFxPTest(CONSTANT,39)
    assign maxPosValueU_uid40_fpToFxPTest_q = 32'b11111111111111111111111111111111;

    // VCC(CONSTANT,1)
    assign VCC_q = 1'b1;

    // GND(CONSTANT,0)
    assign GND_q = 1'b0;

    // rightShiftStage2Idx3Pad3_uid82_rightShiferNoStickyOut_uid39_fpToFxPTest(CONSTANT,81)
    assign rightShiftStage2Idx3Pad3_uid82_rightShiferNoStickyOut_uid39_fpToFxPTest_q = 3'b000;

    // rightShiftStage2Idx3Rng3_uid81_rightShiferNoStickyOut_uid39_fpToFxPTest(BITSELECT,80)@1
    assign rightShiftStage2Idx3Rng3_uid81_rightShiferNoStickyOut_uid39_fpToFxPTest_b = rightShiftStage1_uid74_rightShiferNoStickyOut_uid39_fpToFxPTest_q[32:3];

    // rightShiftStage2Idx3_uid83_rightShiferNoStickyOut_uid39_fpToFxPTest(BITJOIN,82)@1
    assign rightShiftStage2Idx3_uid83_rightShiferNoStickyOut_uid39_fpToFxPTest_q = {rightShiftStage2Idx3Pad3_uid82_rightShiferNoStickyOut_uid39_fpToFxPTest_q, rightShiftStage2Idx3Rng3_uid81_rightShiferNoStickyOut_uid39_fpToFxPTest_b};

    // rightShiftStage2Idx2Pad2_uid79_rightShiferNoStickyOut_uid39_fpToFxPTest(CONSTANT,78)
    assign rightShiftStage2Idx2Pad2_uid79_rightShiferNoStickyOut_uid39_fpToFxPTest_q = 2'b00;

    // rightShiftStage2Idx2Rng2_uid78_rightShiferNoStickyOut_uid39_fpToFxPTest(BITSELECT,77)@1
    assign rightShiftStage2Idx2Rng2_uid78_rightShiferNoStickyOut_uid39_fpToFxPTest_b = rightShiftStage1_uid74_rightShiferNoStickyOut_uid39_fpToFxPTest_q[32:2];

    // rightShiftStage2Idx2_uid80_rightShiferNoStickyOut_uid39_fpToFxPTest(BITJOIN,79)@1
    assign rightShiftStage2Idx2_uid80_rightShiferNoStickyOut_uid39_fpToFxPTest_q = {rightShiftStage2Idx2Pad2_uid79_rightShiferNoStickyOut_uid39_fpToFxPTest_q, rightShiftStage2Idx2Rng2_uid78_rightShiferNoStickyOut_uid39_fpToFxPTest_b};

    // rightShiftStage2Idx1Rng1_uid75_rightShiferNoStickyOut_uid39_fpToFxPTest(BITSELECT,74)@1
    assign rightShiftStage2Idx1Rng1_uid75_rightShiferNoStickyOut_uid39_fpToFxPTest_b = rightShiftStage1_uid74_rightShiferNoStickyOut_uid39_fpToFxPTest_q[32:1];

    // rightShiftStage2Idx1_uid77_rightShiferNoStickyOut_uid39_fpToFxPTest(BITJOIN,76)@1
    assign rightShiftStage2Idx1_uid77_rightShiferNoStickyOut_uid39_fpToFxPTest_q = {GND_q, rightShiftStage2Idx1Rng1_uid75_rightShiferNoStickyOut_uid39_fpToFxPTest_b};

    // rightShiftStage1Idx3Pad12_uid71_rightShiferNoStickyOut_uid39_fpToFxPTest(CONSTANT,70)
    assign rightShiftStage1Idx3Pad12_uid71_rightShiferNoStickyOut_uid39_fpToFxPTest_q = 12'b000000000000;

    // rightShiftStage1Idx3Rng12_uid70_rightShiferNoStickyOut_uid39_fpToFxPTest(BITSELECT,69)@1
    assign rightShiftStage1Idx3Rng12_uid70_rightShiferNoStickyOut_uid39_fpToFxPTest_b = rightShiftStage0_uid63_rightShiferNoStickyOut_uid39_fpToFxPTest_q[32:12];

    // rightShiftStage1Idx3_uid72_rightShiferNoStickyOut_uid39_fpToFxPTest(BITJOIN,71)@1
    assign rightShiftStage1Idx3_uid72_rightShiferNoStickyOut_uid39_fpToFxPTest_q = {rightShiftStage1Idx3Pad12_uid71_rightShiferNoStickyOut_uid39_fpToFxPTest_q, rightShiftStage1Idx3Rng12_uid70_rightShiferNoStickyOut_uid39_fpToFxPTest_b};

    // cstAllZWE_uid8_fpToFxPTest(CONSTANT,7)
    assign cstAllZWE_uid8_fpToFxPTest_q = 8'b00000000;

    // rightShiftStage1Idx2Rng8_uid67_rightShiferNoStickyOut_uid39_fpToFxPTest(BITSELECT,66)@1
    assign rightShiftStage1Idx2Rng8_uid67_rightShiferNoStickyOut_uid39_fpToFxPTest_b = rightShiftStage0_uid63_rightShiferNoStickyOut_uid39_fpToFxPTest_q[32:8];

    // rightShiftStage1Idx2_uid69_rightShiferNoStickyOut_uid39_fpToFxPTest(BITJOIN,68)@1
    assign rightShiftStage1Idx2_uid69_rightShiferNoStickyOut_uid39_fpToFxPTest_q = {cstAllZWE_uid8_fpToFxPTest_q, rightShiftStage1Idx2Rng8_uid67_rightShiferNoStickyOut_uid39_fpToFxPTest_b};

    // rightShiftStage1Idx1Pad4_uid65_rightShiferNoStickyOut_uid39_fpToFxPTest(CONSTANT,64)
    assign rightShiftStage1Idx1Pad4_uid65_rightShiferNoStickyOut_uid39_fpToFxPTest_q = 4'b0000;

    // rightShiftStage1Idx1Rng4_uid64_rightShiferNoStickyOut_uid39_fpToFxPTest(BITSELECT,63)@1
    assign rightShiftStage1Idx1Rng4_uid64_rightShiferNoStickyOut_uid39_fpToFxPTest_b = rightShiftStage0_uid63_rightShiferNoStickyOut_uid39_fpToFxPTest_q[32:4];

    // rightShiftStage1Idx1_uid66_rightShiferNoStickyOut_uid39_fpToFxPTest(BITJOIN,65)@1
    assign rightShiftStage1Idx1_uid66_rightShiferNoStickyOut_uid39_fpToFxPTest_q = {rightShiftStage1Idx1Pad4_uid65_rightShiferNoStickyOut_uid39_fpToFxPTest_q, rightShiftStage1Idx1Rng4_uid64_rightShiferNoStickyOut_uid39_fpToFxPTest_b};

    // rightShiftStage0Idx3_uid61_rightShiferNoStickyOut_uid39_fpToFxPTest(CONSTANT,60)
    assign rightShiftStage0Idx3_uid61_rightShiferNoStickyOut_uid39_fpToFxPTest_q = 33'b000000000000000000000000000000000;

    // rightShiftStage0Idx2Rng32_uid58_rightShiferNoStickyOut_uid39_fpToFxPTest(BITSELECT,57)@1
    assign rightShiftStage0Idx2Rng32_uid58_rightShiferNoStickyOut_uid39_fpToFxPTest_b = shifterIn_uid38_fpToFxPTest_q[32:32];

    // rightShiftStage0Idx2_uid60_rightShiferNoStickyOut_uid39_fpToFxPTest(BITJOIN,59)@1
    assign rightShiftStage0Idx2_uid60_rightShiferNoStickyOut_uid39_fpToFxPTest_q = {maxNegValueU_uid41_fpToFxPTest_q, rightShiftStage0Idx2Rng32_uid58_rightShiferNoStickyOut_uid39_fpToFxPTest_b};

    // rightShiftStage0Idx1Pad16_uid56_rightShiferNoStickyOut_uid39_fpToFxPTest(CONSTANT,55)
    assign rightShiftStage0Idx1Pad16_uid56_rightShiferNoStickyOut_uid39_fpToFxPTest_q = 16'b0000000000000000;

    // rightShiftStage0Idx1Rng16_uid55_rightShiferNoStickyOut_uid39_fpToFxPTest(BITSELECT,54)@1
    assign rightShiftStage0Idx1Rng16_uid55_rightShiferNoStickyOut_uid39_fpToFxPTest_b = shifterIn_uid38_fpToFxPTest_q[32:16];

    // rightShiftStage0Idx1_uid57_rightShiferNoStickyOut_uid39_fpToFxPTest(BITJOIN,56)@1
    assign rightShiftStage0Idx1_uid57_rightShiferNoStickyOut_uid39_fpToFxPTest_q = {rightShiftStage0Idx1Pad16_uid56_rightShiferNoStickyOut_uid39_fpToFxPTest_q, rightShiftStage0Idx1Rng16_uid55_rightShiferNoStickyOut_uid39_fpToFxPTest_b};

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

    // redist6_frac_x_uid10_fpToFxPTest_b_1(DELAY,93)
    dspba_delay_ver #( .width(23), .depth(1), .reset_kind("ASYNC") )
    redist6_frac_x_uid10_fpToFxPTest_b_1 ( .xin(frac_x_uid10_fpToFxPTest_b), .xout(redist6_frac_x_uid10_fpToFxPTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // oFracX_uid23_fpToFxPTest(BITJOIN,22)@1
    assign oFracX_uid23_fpToFxPTest_q = {invExcXZ_uid22_fpToFxPTest_q, redist6_frac_x_uid10_fpToFxPTest_b_1_q};

    // zPadd_uid37_fpToFxPTest(CONSTANT,36)
    assign zPadd_uid37_fpToFxPTest_q = 9'b000000000;

    // shifterIn_uid38_fpToFxPTest(BITJOIN,37)@1
    assign shifterIn_uid38_fpToFxPTest_q = {oFracX_uid23_fpToFxPTest_q, zPadd_uid37_fpToFxPTest_q};

    // rightShiftStage0_uid63_rightShiferNoStickyOut_uid39_fpToFxPTest(MUX,62)@1
    assign rightShiftStage0_uid63_rightShiferNoStickyOut_uid39_fpToFxPTest_s = rightShiftStageSel5Dto4_uid62_rightShiferNoStickyOut_uid39_fpToFxPTest_merged_bit_select_b;
    always @(rightShiftStage0_uid63_rightShiferNoStickyOut_uid39_fpToFxPTest_s or en or shifterIn_uid38_fpToFxPTest_q or rightShiftStage0Idx1_uid57_rightShiferNoStickyOut_uid39_fpToFxPTest_q or rightShiftStage0Idx2_uid60_rightShiferNoStickyOut_uid39_fpToFxPTest_q or rightShiftStage0Idx3_uid61_rightShiferNoStickyOut_uid39_fpToFxPTest_q)
    begin
        unique case (rightShiftStage0_uid63_rightShiferNoStickyOut_uid39_fpToFxPTest_s)
            2'b00 : rightShiftStage0_uid63_rightShiferNoStickyOut_uid39_fpToFxPTest_q = shifterIn_uid38_fpToFxPTest_q;
            2'b01 : rightShiftStage0_uid63_rightShiferNoStickyOut_uid39_fpToFxPTest_q = rightShiftStage0Idx1_uid57_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
            2'b10 : rightShiftStage0_uid63_rightShiferNoStickyOut_uid39_fpToFxPTest_q = rightShiftStage0Idx2_uid60_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
            2'b11 : rightShiftStage0_uid63_rightShiferNoStickyOut_uid39_fpToFxPTest_q = rightShiftStage0Idx3_uid61_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
            default : rightShiftStage0_uid63_rightShiferNoStickyOut_uid39_fpToFxPTest_q = 33'b0;
        endcase
    end

    // rightShiftStage1_uid74_rightShiferNoStickyOut_uid39_fpToFxPTest(MUX,73)@1
    assign rightShiftStage1_uid74_rightShiferNoStickyOut_uid39_fpToFxPTest_s = rightShiftStageSel5Dto4_uid62_rightShiferNoStickyOut_uid39_fpToFxPTest_merged_bit_select_c;
    always @(rightShiftStage1_uid74_rightShiferNoStickyOut_uid39_fpToFxPTest_s or en or rightShiftStage0_uid63_rightShiferNoStickyOut_uid39_fpToFxPTest_q or rightShiftStage1Idx1_uid66_rightShiferNoStickyOut_uid39_fpToFxPTest_q or rightShiftStage1Idx2_uid69_rightShiferNoStickyOut_uid39_fpToFxPTest_q or rightShiftStage1Idx3_uid72_rightShiferNoStickyOut_uid39_fpToFxPTest_q)
    begin
        unique case (rightShiftStage1_uid74_rightShiferNoStickyOut_uid39_fpToFxPTest_s)
            2'b00 : rightShiftStage1_uid74_rightShiferNoStickyOut_uid39_fpToFxPTest_q = rightShiftStage0_uid63_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
            2'b01 : rightShiftStage1_uid74_rightShiferNoStickyOut_uid39_fpToFxPTest_q = rightShiftStage1Idx1_uid66_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
            2'b10 : rightShiftStage1_uid74_rightShiferNoStickyOut_uid39_fpToFxPTest_q = rightShiftStage1Idx2_uid69_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
            2'b11 : rightShiftStage1_uid74_rightShiferNoStickyOut_uid39_fpToFxPTest_q = rightShiftStage1Idx3_uid72_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
            default : rightShiftStage1_uid74_rightShiferNoStickyOut_uid39_fpToFxPTest_q = 33'b0;
        endcase
    end

    // maxShiftCst_uid34_fpToFxPTest(CONSTANT,33)
    assign maxShiftCst_uid34_fpToFxPTest_q = 6'b100001;

    // ovfExpVal_uid31_fpToFxPTest(CONSTANT,30)
    assign ovfExpVal_uid31_fpToFxPTest_q = 9'b010011110;

    // shiftValE_uid32_fpToFxPTest(SUB,31)@0
    assign shiftValE_uid32_fpToFxPTest_a = {{2{ovfExpVal_uid31_fpToFxPTest_q[8]}}, ovfExpVal_uid31_fpToFxPTest_q};
    assign shiftValE_uid32_fpToFxPTest_b = {3'b000, exp_x_uid9_fpToFxPTest_b};
    assign shiftValE_uid32_fpToFxPTest_o = $signed(shiftValE_uid32_fpToFxPTest_a) - $signed(shiftValE_uid32_fpToFxPTest_b);
    assign shiftValE_uid32_fpToFxPTest_q = shiftValE_uid32_fpToFxPTest_o[9:0];

    // shiftValRaw_uid33_fpToFxPTest(BITSELECT,32)@0
    assign shiftValRaw_uid33_fpToFxPTest_in = shiftValE_uid32_fpToFxPTest_q[5:0];
    assign shiftValRaw_uid33_fpToFxPTest_b = shiftValRaw_uid33_fpToFxPTest_in[5:0];

    // redist1_shiftValRaw_uid33_fpToFxPTest_b_1(DELAY,88)
    dspba_delay_ver #( .width(6), .depth(1), .reset_kind("ASYNC") )
    redist1_shiftValRaw_uid33_fpToFxPTest_b_1 ( .xin(shiftValRaw_uid33_fpToFxPTest_b), .xout(redist1_shiftValRaw_uid33_fpToFxPTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // shiftOutOfRange_uid35_fpToFxPTest(COMPARE,34)@0 + 1
    assign shiftOutOfRange_uid35_fpToFxPTest_a = {{2{shiftValE_uid32_fpToFxPTest_q[9]}}, shiftValE_uid32_fpToFxPTest_q};
    assign shiftOutOfRange_uid35_fpToFxPTest_b = {6'b000000, maxShiftCst_uid34_fpToFxPTest_q};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            shiftOutOfRange_uid35_fpToFxPTest_o <= 12'b0;
        end
        else if (en == 1'b1)
        begin
            shiftOutOfRange_uid35_fpToFxPTest_o <= $signed(shiftOutOfRange_uid35_fpToFxPTest_a) - $signed(shiftOutOfRange_uid35_fpToFxPTest_b);
        end
    end
    assign shiftOutOfRange_uid35_fpToFxPTest_n[0] = ~ (shiftOutOfRange_uid35_fpToFxPTest_o[11]);

    // shiftVal_uid36_fpToFxPTest(MUX,35)@1
    assign shiftVal_uid36_fpToFxPTest_s = shiftOutOfRange_uid35_fpToFxPTest_n;
    always @(shiftVal_uid36_fpToFxPTest_s or en or redist1_shiftValRaw_uid33_fpToFxPTest_b_1_q or maxShiftCst_uid34_fpToFxPTest_q)
    begin
        unique case (shiftVal_uid36_fpToFxPTest_s)
            1'b0 : shiftVal_uid36_fpToFxPTest_q = redist1_shiftValRaw_uid33_fpToFxPTest_b_1_q;
            1'b1 : shiftVal_uid36_fpToFxPTest_q = maxShiftCst_uid34_fpToFxPTest_q;
            default : shiftVal_uid36_fpToFxPTest_q = 6'b0;
        endcase
    end

    // rightShiftStageSel5Dto4_uid62_rightShiferNoStickyOut_uid39_fpToFxPTest_merged_bit_select(BITSELECT,86)@1
    assign rightShiftStageSel5Dto4_uid62_rightShiferNoStickyOut_uid39_fpToFxPTest_merged_bit_select_b = shiftVal_uid36_fpToFxPTest_q[5:4];
    assign rightShiftStageSel5Dto4_uid62_rightShiferNoStickyOut_uid39_fpToFxPTest_merged_bit_select_c = shiftVal_uid36_fpToFxPTest_q[3:2];
    assign rightShiftStageSel5Dto4_uid62_rightShiferNoStickyOut_uid39_fpToFxPTest_merged_bit_select_d = shiftVal_uid36_fpToFxPTest_q[1:0];

    // rightShiftStage2_uid85_rightShiferNoStickyOut_uid39_fpToFxPTest(MUX,84)@1 + 1
    assign rightShiftStage2_uid85_rightShiferNoStickyOut_uid39_fpToFxPTest_s = rightShiftStageSel5Dto4_uid62_rightShiferNoStickyOut_uid39_fpToFxPTest_merged_bit_select_d;
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            rightShiftStage2_uid85_rightShiferNoStickyOut_uid39_fpToFxPTest_q <= 33'b0;
        end
        else if (en == 1'b1)
        begin
            unique case (rightShiftStage2_uid85_rightShiferNoStickyOut_uid39_fpToFxPTest_s)
                2'b00 : rightShiftStage2_uid85_rightShiferNoStickyOut_uid39_fpToFxPTest_q <= rightShiftStage1_uid74_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
                2'b01 : rightShiftStage2_uid85_rightShiferNoStickyOut_uid39_fpToFxPTest_q <= rightShiftStage2Idx1_uid77_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
                2'b10 : rightShiftStage2_uid85_rightShiferNoStickyOut_uid39_fpToFxPTest_q <= rightShiftStage2Idx2_uid80_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
                2'b11 : rightShiftStage2_uid85_rightShiferNoStickyOut_uid39_fpToFxPTest_q <= rightShiftStage2Idx3_uid83_rightShiferNoStickyOut_uid39_fpToFxPTest_q;
                default : rightShiftStage2_uid85_rightShiferNoStickyOut_uid39_fpToFxPTest_q <= 33'b0;
            endcase
        end
    end

    // zRightShiferNoStickyOut_uid43_fpToFxPTest(BITJOIN,42)@2
    assign zRightShiferNoStickyOut_uid43_fpToFxPTest_q = {GND_q, rightShiftStage2_uid85_rightShiferNoStickyOut_uid39_fpToFxPTest_q};

    // sPostRndFull_uid44_fpToFxPTest(ADD,43)@2
    assign sPostRndFull_uid44_fpToFxPTest_a = {1'b0, zRightShiferNoStickyOut_uid43_fpToFxPTest_q};
    assign sPostRndFull_uid44_fpToFxPTest_b = {34'b0000000000000000000000000000000000, VCC_q};
    assign sPostRndFull_uid44_fpToFxPTest_o = $unsigned(sPostRndFull_uid44_fpToFxPTest_a) + $unsigned(sPostRndFull_uid44_fpToFxPTest_b);
    assign sPostRndFull_uid44_fpToFxPTest_q = sPostRndFull_uid44_fpToFxPTest_o[34:0];

    // sPostRnd_uid45_fpToFxPTest(BITSELECT,44)@2
    assign sPostRnd_uid45_fpToFxPTest_in = sPostRndFull_uid44_fpToFxPTest_q[32:0];
    assign sPostRnd_uid45_fpToFxPTest_b = sPostRnd_uid45_fpToFxPTest_in[32:1];

    // redist0_sPostRnd_uid45_fpToFxPTest_b_1(DELAY,87)
    dspba_delay_ver #( .width(32), .depth(1), .reset_kind("ASYNC") )
    redist0_sPostRnd_uid45_fpToFxPTest_b_1 ( .xin(sPostRnd_uid45_fpToFxPTest_b), .xout(redist0_sPostRnd_uid45_fpToFxPTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // signX_uid25_fpToFxPTest(BITSELECT,24)@0
    assign signX_uid25_fpToFxPTest_b = a[31:31];

    // redist4_signX_uid25_fpToFxPTest_b_2(DELAY,91)
    dspba_delay_ver #( .width(1), .depth(2), .reset_kind("ASYNC") )
    redist4_signX_uid25_fpToFxPTest_b_2 ( .xin(signX_uid25_fpToFxPTest_b), .xout(redist4_signX_uid25_fpToFxPTest_b_2_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // udfExpVal_uid29_fpToFxPTest(CONSTANT,28)
    assign udfExpVal_uid29_fpToFxPTest_q = 8'b01111101;

    // udf_uid30_fpToFxPTest(COMPARE,29)@0 + 1
    assign udf_uid30_fpToFxPTest_a = {{3{udfExpVal_uid29_fpToFxPTest_q[7]}}, udfExpVal_uid29_fpToFxPTest_q};
    assign udf_uid30_fpToFxPTest_b = {3'b000, exp_x_uid9_fpToFxPTest_b};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            udf_uid30_fpToFxPTest_o <= 11'b0;
        end
        else if (en == 1'b1)
        begin
            udf_uid30_fpToFxPTest_o <= $signed(udf_uid30_fpToFxPTest_a) - $signed(udf_uid30_fpToFxPTest_b);
        end
    end
    assign udf_uid30_fpToFxPTest_n[0] = ~ (udf_uid30_fpToFxPTest_o[10]);

    // redist2_udf_uid30_fpToFxPTest_n_2(DELAY,89)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist2_udf_uid30_fpToFxPTest_n_2 ( .xin(udf_uid30_fpToFxPTest_n), .xout(redist2_udf_uid30_fpToFxPTest_n_2_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // sPostRndFullMSBU_uid46_fpToFxPTest(BITSELECT,45)@2
    assign sPostRndFullMSBU_uid46_fpToFxPTest_in = sPostRndFull_uid44_fpToFxPTest_q[33:0];
    assign sPostRndFullMSBU_uid46_fpToFxPTest_b = sPostRndFullMSBU_uid46_fpToFxPTest_in[33:33];

    // ovfExpVal_uid26_fpToFxPTest(CONSTANT,25)
    assign ovfExpVal_uid26_fpToFxPTest_q = 9'b010011111;

    // ovf_uid27_fpToFxPTest(COMPARE,26)@0 + 1
    assign ovf_uid27_fpToFxPTest_a = {3'b000, exp_x_uid9_fpToFxPTest_b};
    assign ovf_uid27_fpToFxPTest_b = {{2{ovfExpVal_uid26_fpToFxPTest_q[8]}}, ovfExpVal_uid26_fpToFxPTest_q};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            ovf_uid27_fpToFxPTest_o <= 11'b0;
        end
        else if (en == 1'b1)
        begin
            ovf_uid27_fpToFxPTest_o <= $signed(ovf_uid27_fpToFxPTest_a) - $signed(ovf_uid27_fpToFxPTest_b);
        end
    end
    assign ovf_uid27_fpToFxPTest_n[0] = ~ (ovf_uid27_fpToFxPTest_o[10]);

    // redist3_ovf_uid27_fpToFxPTest_n_2(DELAY,90)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist3_ovf_uid27_fpToFxPTest_n_2 ( .xin(ovf_uid27_fpToFxPTest_n), .xout(redist3_ovf_uid27_fpToFxPTest_n_2_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // negOrOvf_uid28_fpToFxPTest(LOGICAL,27)@2
    assign negOrOvf_uid28_fpToFxPTest_q = redist4_signX_uid25_fpToFxPTest_b_2_q | redist3_ovf_uid27_fpToFxPTest_n_2_q;

    // cstZeroWF_uid7_fpToFxPTest(CONSTANT,6)
    assign cstZeroWF_uid7_fpToFxPTest_q = 23'b00000000000000000000000;

    // fracXIsZero_uid13_fpToFxPTest(LOGICAL,12)@1 + 1
    assign fracXIsZero_uid13_fpToFxPTest_qi = cstZeroWF_uid7_fpToFxPTest_q == redist6_frac_x_uid10_fpToFxPTest_b_1_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    fracXIsZero_uid13_fpToFxPTest_delay ( .xin(fracXIsZero_uid13_fpToFxPTest_qi), .xout(fracXIsZero_uid13_fpToFxPTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // cstAllOWE_uid6_fpToFxPTest(CONSTANT,5)
    assign cstAllOWE_uid6_fpToFxPTest_q = 8'b11111111;

    // expXIsMax_uid12_fpToFxPTest(LOGICAL,11)@0 + 1
    assign expXIsMax_uid12_fpToFxPTest_qi = exp_x_uid9_fpToFxPTest_b == cstAllOWE_uid6_fpToFxPTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    expXIsMax_uid12_fpToFxPTest_delay ( .xin(expXIsMax_uid12_fpToFxPTest_qi), .xout(expXIsMax_uid12_fpToFxPTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist5_expXIsMax_uid12_fpToFxPTest_q_2(DELAY,92)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist5_expXIsMax_uid12_fpToFxPTest_q_2 ( .xin(expXIsMax_uid12_fpToFxPTest_q), .xout(redist5_expXIsMax_uid12_fpToFxPTest_q_2_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // excI_x_uid15_fpToFxPTest(LOGICAL,14)@2
    assign excI_x_uid15_fpToFxPTest_q = redist5_expXIsMax_uid12_fpToFxPTest_q_2_q & fracXIsZero_uid13_fpToFxPTest_q;

    // fracXIsNotZero_uid14_fpToFxPTest(LOGICAL,13)@2
    assign fracXIsNotZero_uid14_fpToFxPTest_q = ~ (fracXIsZero_uid13_fpToFxPTest_q);

    // excN_x_uid16_fpToFxPTest(LOGICAL,15)@2
    assign excN_x_uid16_fpToFxPTest_q = redist5_expXIsMax_uid12_fpToFxPTest_q_2_q & fracXIsNotZero_uid14_fpToFxPTest_q;

    // ovfPostRnd_uid47_fpToFxPTest(LOGICAL,46)@2
    assign ovfPostRnd_uid47_fpToFxPTest_q = excN_x_uid16_fpToFxPTest_q | excI_x_uid15_fpToFxPTest_q | negOrOvf_uid28_fpToFxPTest_q | sPostRndFullMSBU_uid46_fpToFxPTest_b;

    // muxSelConc_uid48_fpToFxPTest(BITJOIN,47)@2
    assign muxSelConc_uid48_fpToFxPTest_q = {redist4_signX_uid25_fpToFxPTest_b_2_q, redist2_udf_uid30_fpToFxPTest_n_2_q, ovfPostRnd_uid47_fpToFxPTest_q};

    // muxSel_uid49_fpToFxPTest(LOOKUP,48)@2 + 1
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            muxSel_uid49_fpToFxPTest_q <= 2'b00;
        end
        else if (en == 1'b1)
        begin
            unique case (muxSelConc_uid48_fpToFxPTest_q)
                3'b000 : muxSel_uid49_fpToFxPTest_q <= 2'b00;
                3'b001 : muxSel_uid49_fpToFxPTest_q <= 2'b01;
                3'b010 : muxSel_uid49_fpToFxPTest_q <= 2'b11;
                3'b011 : muxSel_uid49_fpToFxPTest_q <= 2'b00;
                3'b100 : muxSel_uid49_fpToFxPTest_q <= 2'b10;
                3'b101 : muxSel_uid49_fpToFxPTest_q <= 2'b10;
                3'b110 : muxSel_uid49_fpToFxPTest_q <= 2'b10;
                3'b111 : muxSel_uid49_fpToFxPTest_q <= 2'b10;
                default : begin
                              // unreachable
                              muxSel_uid49_fpToFxPTest_q <= 2'bxx;
                          end
            endcase
        end
    end

    // finalOut_uid51_fpToFxPTest(MUX,50)@3
    assign finalOut_uid51_fpToFxPTest_s = muxSel_uid49_fpToFxPTest_q;
    always @(finalOut_uid51_fpToFxPTest_s or en or redist0_sPostRnd_uid45_fpToFxPTest_b_1_q or maxPosValueU_uid40_fpToFxPTest_q or maxNegValueU_uid41_fpToFxPTest_q)
    begin
        unique case (finalOut_uid51_fpToFxPTest_s)
            2'b00 : finalOut_uid51_fpToFxPTest_q = redist0_sPostRnd_uid45_fpToFxPTest_b_1_q;
            2'b01 : finalOut_uid51_fpToFxPTest_q = maxPosValueU_uid40_fpToFxPTest_q;
            2'b10 : finalOut_uid51_fpToFxPTest_q = maxNegValueU_uid41_fpToFxPTest_q;
            2'b11 : finalOut_uid51_fpToFxPTest_q = maxNegValueU_uid41_fpToFxPTest_q;
            default : finalOut_uid51_fpToFxPTest_q = 32'b0;
        endcase
    end

    // xOut(GPOUT,4)@3
    assign q = finalOut_uid51_fpToFxPTest_q;

endmodule
