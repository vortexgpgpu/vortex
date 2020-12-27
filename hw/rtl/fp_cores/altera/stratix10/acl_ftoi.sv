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

// SystemVerilog created from acl_ftoi
// SystemVerilog created on Sun Dec 27 09:48:58 2020


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
    wire [0:0] excZ_x_uid11_fpToFxPTest_q;
    wire [0:0] expXIsMax_uid12_fpToFxPTest_q;
    wire [0:0] fracXIsZero_uid13_fpToFxPTest_q;
    wire [0:0] fracXIsNotZero_uid14_fpToFxPTest_q;
    wire [0:0] excI_x_uid15_fpToFxPTest_qi;
    reg [0:0] excI_x_uid15_fpToFxPTest_q;
    wire [0:0] excN_x_uid16_fpToFxPTest_q;
    wire [0:0] fracPostZ_uid23_fpToFxPTest_s;
    reg [22:0] fracPostZ_uid23_fpToFxPTest_q;
    wire [0:0] invExcXZ_uid24_fpToFxPTest_qi;
    reg [0:0] invExcXZ_uid24_fpToFxPTest_q;
    wire [23:0] oFracX_uid25_fpToFxPTest_q;
    wire [0:0] signX_uid27_fpToFxPTest_b;
    wire [0:0] notNan_uid28_fpToFxPTest_q;
    wire [0:0] signX_uid29_fpToFxPTest_qi;
    reg [0:0] signX_uid29_fpToFxPTest_q;
    wire [8:0] ovfExpVal_uid30_fpToFxPTest_q;
    wire [10:0] ovfExpRange_uid31_fpToFxPTest_a;
    wire [10:0] ovfExpRange_uid31_fpToFxPTest_b;
    logic [10:0] ovfExpRange_uid31_fpToFxPTest_o;
    wire [0:0] ovfExpRange_uid31_fpToFxPTest_n;
    wire [7:0] udfExpVal_uid32_fpToFxPTest_q;
    wire [10:0] udf_uid33_fpToFxPTest_a;
    wire [10:0] udf_uid33_fpToFxPTest_b;
    logic [10:0] udf_uid33_fpToFxPTest_o;
    wire [0:0] udf_uid33_fpToFxPTest_n;
    wire [8:0] ovfExpVal_uid34_fpToFxPTest_q;
    wire [10:0] shiftValE_uid35_fpToFxPTest_a;
    wire [10:0] shiftValE_uid35_fpToFxPTest_b;
    logic [10:0] shiftValE_uid35_fpToFxPTest_o;
    wire [9:0] shiftValE_uid35_fpToFxPTest_q;
    wire [5:0] shiftValRaw_uid36_fpToFxPTest_in;
    wire [5:0] shiftValRaw_uid36_fpToFxPTest_b;
    wire [5:0] maxShiftCst_uid37_fpToFxPTest_q;
    wire [11:0] shiftOutOfRange_uid38_fpToFxPTest_a;
    wire [11:0] shiftOutOfRange_uid38_fpToFxPTest_b;
    logic [11:0] shiftOutOfRange_uid38_fpToFxPTest_o;
    wire [0:0] shiftOutOfRange_uid38_fpToFxPTest_n;
    wire [0:0] shiftVal_uid39_fpToFxPTest_s;
    reg [5:0] shiftVal_uid39_fpToFxPTest_q;
    wire [31:0] shifterIn_uid41_fpToFxPTest_q;
    wire [31:0] maxPosValueS_uid43_fpToFxPTest_q;
    wire [31:0] maxNegValueS_uid44_fpToFxPTest_q;
    wire [32:0] zRightShiferNoStickyOut_uid45_fpToFxPTest_q;
    wire [32:0] xXorSignE_uid46_fpToFxPTest_b;
    wire [32:0] xXorSignE_uid46_fpToFxPTest_qi;
    reg [32:0] xXorSignE_uid46_fpToFxPTest_q;
    wire [2:0] d0_uid47_fpToFxPTest_q;
    wire [33:0] sPostRndFull_uid48_fpToFxPTest_a;
    wire [33:0] sPostRndFull_uid48_fpToFxPTest_b;
    logic [33:0] sPostRndFull_uid48_fpToFxPTest_o;
    wire [33:0] sPostRndFull_uid48_fpToFxPTest_q;
    wire [32:0] sPostRnd_uid49_fpToFxPTest_in;
    wire [31:0] sPostRnd_uid49_fpToFxPTest_b;
    wire [34:0] sPostRnd_uid50_fpToFxPTest_in;
    wire [33:0] sPostRnd_uid50_fpToFxPTest_b;
    wire [35:0] rndOvfPos_uid51_fpToFxPTest_a;
    wire [35:0] rndOvfPos_uid51_fpToFxPTest_b;
    logic [35:0] rndOvfPos_uid51_fpToFxPTest_o;
    wire [0:0] rndOvfPos_uid51_fpToFxPTest_c;
    wire [0:0] ovfPostRnd_uid52_fpToFxPTest_q;
    wire [2:0] muxSelConc_uid53_fpToFxPTest_q;
    reg [1:0] muxSel_uid54_fpToFxPTest_q;
    wire [31:0] maxNegValueU_uid55_fpToFxPTest_q;
    wire [1:0] finalOut_uid56_fpToFxPTest_s;
    reg [31:0] finalOut_uid56_fpToFxPTest_q;
    wire [30:0] rightShiftStage0Idx1Rng1_uid60_rightShiferNoStickyOut_uid42_fpToFxPTest_b;
    wire [31:0] rightShiftStage0Idx1_uid62_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
    wire [29:0] rightShiftStage0Idx2Rng2_uid63_rightShiferNoStickyOut_uid42_fpToFxPTest_b;
    wire [1:0] rightShiftStage0Idx2Pad2_uid64_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
    wire [31:0] rightShiftStage0Idx2_uid65_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
    wire [28:0] rightShiftStage0Idx3Rng3_uid66_rightShiferNoStickyOut_uid42_fpToFxPTest_b;
    wire [2:0] rightShiftStage0Idx3Pad3_uid67_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
    wire [31:0] rightShiftStage0Idx3_uid68_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
    wire [1:0] rightShiftStage0_uid70_rightShiferNoStickyOut_uid42_fpToFxPTest_s;
    reg [31:0] rightShiftStage0_uid70_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
    wire [27:0] rightShiftStage1Idx1Rng4_uid71_rightShiferNoStickyOut_uid42_fpToFxPTest_b;
    wire [3:0] rightShiftStage1Idx1Pad4_uid72_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
    wire [31:0] rightShiftStage1Idx1_uid73_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
    wire [23:0] rightShiftStage1Idx2Rng8_uid74_rightShiferNoStickyOut_uid42_fpToFxPTest_b;
    wire [31:0] rightShiftStage1Idx2_uid76_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
    wire [19:0] rightShiftStage1Idx3Rng12_uid77_rightShiferNoStickyOut_uid42_fpToFxPTest_b;
    wire [11:0] rightShiftStage1Idx3Pad12_uid78_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
    wire [31:0] rightShiftStage1Idx3_uid79_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
    wire [1:0] rightShiftStage1_uid81_rightShiferNoStickyOut_uid42_fpToFxPTest_s;
    reg [31:0] rightShiftStage1_uid81_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
    wire [15:0] rightShiftStage2Idx1Rng16_uid82_rightShiferNoStickyOut_uid42_fpToFxPTest_b;
    wire [15:0] rightShiftStage2Idx1Pad16_uid83_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
    wire [31:0] rightShiftStage2Idx1_uid84_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
    wire [1:0] rightShiftStage2_uid88_rightShiferNoStickyOut_uid42_fpToFxPTest_s;
    reg [31:0] rightShiftStage2_uid88_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
    wire [1:0] rightShiftStageSel0Dto0_uid69_rightShiferNoStickyOut_uid42_fpToFxPTest_merged_bit_select_b;
    wire [1:0] rightShiftStageSel0Dto0_uid69_rightShiferNoStickyOut_uid42_fpToFxPTest_merged_bit_select_c;
    wire [1:0] rightShiftStageSel0Dto0_uid69_rightShiferNoStickyOut_uid42_fpToFxPTest_merged_bit_select_d;
    reg [31:0] redist0_sPostRnd_uid49_fpToFxPTest_b_1_q;
    reg [0:0] redist1_udf_uid33_fpToFxPTest_n_3_q;
    reg [0:0] redist1_udf_uid33_fpToFxPTest_n_3_delay_0;
    reg [0:0] redist2_ovfExpRange_uid31_fpToFxPTest_n_3_q;
    reg [0:0] redist2_ovfExpRange_uid31_fpToFxPTest_n_3_delay_0;
    reg [0:0] redist3_signX_uid29_fpToFxPTest_q_3_q;
    reg [0:0] redist3_signX_uid29_fpToFxPTest_q_3_delay_0;
    reg [0:0] redist4_excN_x_uid16_fpToFxPTest_q_3_q;
    reg [0:0] redist4_excN_x_uid16_fpToFxPTest_q_3_delay_0;
    reg [0:0] redist4_excN_x_uid16_fpToFxPTest_q_3_delay_1;
    reg [0:0] redist5_excI_x_uid15_fpToFxPTest_q_3_q;
    reg [0:0] redist5_excI_x_uid15_fpToFxPTest_q_3_delay_0;


    // maxNegValueU_uid55_fpToFxPTest(CONSTANT,54)
    assign maxNegValueU_uid55_fpToFxPTest_q = 32'b00000000000000000000000000000000;

    // maxNegValueS_uid44_fpToFxPTest(CONSTANT,43)
    assign maxNegValueS_uid44_fpToFxPTest_q = 32'b10000000000000000000000000000000;

    // maxPosValueS_uid43_fpToFxPTest(CONSTANT,42)
    assign maxPosValueS_uid43_fpToFxPTest_q = 32'b01111111111111111111111111111111;

    // d0_uid47_fpToFxPTest(CONSTANT,46)
    assign d0_uid47_fpToFxPTest_q = 3'b001;

    // signX_uid27_fpToFxPTest(BITSELECT,26)@0
    assign signX_uid27_fpToFxPTest_b = a[31:31];

    // frac_x_uid10_fpToFxPTest(BITSELECT,9)@0
    assign frac_x_uid10_fpToFxPTest_b = a[22:0];

    // cstZeroWF_uid7_fpToFxPTest(CONSTANT,6)
    assign cstZeroWF_uid7_fpToFxPTest_q = 23'b00000000000000000000000;

    // fracXIsZero_uid13_fpToFxPTest(LOGICAL,12)@0
    assign fracXIsZero_uid13_fpToFxPTest_q = cstZeroWF_uid7_fpToFxPTest_q == frac_x_uid10_fpToFxPTest_b ? 1'b1 : 1'b0;

    // fracXIsNotZero_uid14_fpToFxPTest(LOGICAL,13)@0
    assign fracXIsNotZero_uid14_fpToFxPTest_q = ~ (fracXIsZero_uid13_fpToFxPTest_q);

    // cstAllOWE_uid6_fpToFxPTest(CONSTANT,5)
    assign cstAllOWE_uid6_fpToFxPTest_q = 8'b11111111;

    // exp_x_uid9_fpToFxPTest(BITSELECT,8)@0
    assign exp_x_uid9_fpToFxPTest_b = a[30:23];

    // expXIsMax_uid12_fpToFxPTest(LOGICAL,11)@0
    assign expXIsMax_uid12_fpToFxPTest_q = exp_x_uid9_fpToFxPTest_b == cstAllOWE_uid6_fpToFxPTest_q ? 1'b1 : 1'b0;

    // excN_x_uid16_fpToFxPTest(LOGICAL,15)@0
    assign excN_x_uid16_fpToFxPTest_q = expXIsMax_uid12_fpToFxPTest_q & fracXIsNotZero_uid14_fpToFxPTest_q;

    // notNan_uid28_fpToFxPTest(LOGICAL,27)@0
    assign notNan_uid28_fpToFxPTest_q = ~ (excN_x_uid16_fpToFxPTest_q);

    // signX_uid29_fpToFxPTest(LOGICAL,28)@0 + 1
    assign signX_uid29_fpToFxPTest_qi = notNan_uid28_fpToFxPTest_q & signX_uid27_fpToFxPTest_b;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    signX_uid29_fpToFxPTest_delay ( .xin(signX_uid29_fpToFxPTest_qi), .xout(signX_uid29_fpToFxPTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // GND(CONSTANT,0)
    assign GND_q = 1'b0;

    // rightShiftStage2Idx1Pad16_uid83_rightShiferNoStickyOut_uid42_fpToFxPTest(CONSTANT,82)
    assign rightShiftStage2Idx1Pad16_uid83_rightShiferNoStickyOut_uid42_fpToFxPTest_q = 16'b0000000000000000;

    // rightShiftStage2Idx1Rng16_uid82_rightShiferNoStickyOut_uid42_fpToFxPTest(BITSELECT,81)@1
    assign rightShiftStage2Idx1Rng16_uid82_rightShiferNoStickyOut_uid42_fpToFxPTest_b = rightShiftStage1_uid81_rightShiferNoStickyOut_uid42_fpToFxPTest_q[31:16];

    // rightShiftStage2Idx1_uid84_rightShiferNoStickyOut_uid42_fpToFxPTest(BITJOIN,83)@1
    assign rightShiftStage2Idx1_uid84_rightShiferNoStickyOut_uid42_fpToFxPTest_q = {rightShiftStage2Idx1Pad16_uid83_rightShiferNoStickyOut_uid42_fpToFxPTest_q, rightShiftStage2Idx1Rng16_uid82_rightShiferNoStickyOut_uid42_fpToFxPTest_b};

    // rightShiftStage1Idx3Pad12_uid78_rightShiferNoStickyOut_uid42_fpToFxPTest(CONSTANT,77)
    assign rightShiftStage1Idx3Pad12_uid78_rightShiferNoStickyOut_uid42_fpToFxPTest_q = 12'b000000000000;

    // rightShiftStage1Idx3Rng12_uid77_rightShiferNoStickyOut_uid42_fpToFxPTest(BITSELECT,76)@1
    assign rightShiftStage1Idx3Rng12_uid77_rightShiferNoStickyOut_uid42_fpToFxPTest_b = rightShiftStage0_uid70_rightShiferNoStickyOut_uid42_fpToFxPTest_q[31:12];

    // rightShiftStage1Idx3_uid79_rightShiferNoStickyOut_uid42_fpToFxPTest(BITJOIN,78)@1
    assign rightShiftStage1Idx3_uid79_rightShiferNoStickyOut_uid42_fpToFxPTest_q = {rightShiftStage1Idx3Pad12_uid78_rightShiferNoStickyOut_uid42_fpToFxPTest_q, rightShiftStage1Idx3Rng12_uid77_rightShiferNoStickyOut_uid42_fpToFxPTest_b};

    // cstAllZWE_uid8_fpToFxPTest(CONSTANT,7)
    assign cstAllZWE_uid8_fpToFxPTest_q = 8'b00000000;

    // rightShiftStage1Idx2Rng8_uid74_rightShiferNoStickyOut_uid42_fpToFxPTest(BITSELECT,73)@1
    assign rightShiftStage1Idx2Rng8_uid74_rightShiferNoStickyOut_uid42_fpToFxPTest_b = rightShiftStage0_uid70_rightShiferNoStickyOut_uid42_fpToFxPTest_q[31:8];

    // rightShiftStage1Idx2_uid76_rightShiferNoStickyOut_uid42_fpToFxPTest(BITJOIN,75)@1
    assign rightShiftStage1Idx2_uid76_rightShiferNoStickyOut_uid42_fpToFxPTest_q = {cstAllZWE_uid8_fpToFxPTest_q, rightShiftStage1Idx2Rng8_uid74_rightShiferNoStickyOut_uid42_fpToFxPTest_b};

    // rightShiftStage1Idx1Pad4_uid72_rightShiferNoStickyOut_uid42_fpToFxPTest(CONSTANT,71)
    assign rightShiftStage1Idx1Pad4_uid72_rightShiferNoStickyOut_uid42_fpToFxPTest_q = 4'b0000;

    // rightShiftStage1Idx1Rng4_uid71_rightShiferNoStickyOut_uid42_fpToFxPTest(BITSELECT,70)@1
    assign rightShiftStage1Idx1Rng4_uid71_rightShiferNoStickyOut_uid42_fpToFxPTest_b = rightShiftStage0_uid70_rightShiferNoStickyOut_uid42_fpToFxPTest_q[31:4];

    // rightShiftStage1Idx1_uid73_rightShiferNoStickyOut_uid42_fpToFxPTest(BITJOIN,72)@1
    assign rightShiftStage1Idx1_uid73_rightShiferNoStickyOut_uid42_fpToFxPTest_q = {rightShiftStage1Idx1Pad4_uid72_rightShiferNoStickyOut_uid42_fpToFxPTest_q, rightShiftStage1Idx1Rng4_uid71_rightShiferNoStickyOut_uid42_fpToFxPTest_b};

    // rightShiftStage0Idx3Pad3_uid67_rightShiferNoStickyOut_uid42_fpToFxPTest(CONSTANT,66)
    assign rightShiftStage0Idx3Pad3_uid67_rightShiferNoStickyOut_uid42_fpToFxPTest_q = 3'b000;

    // rightShiftStage0Idx3Rng3_uid66_rightShiferNoStickyOut_uid42_fpToFxPTest(BITSELECT,65)@1
    assign rightShiftStage0Idx3Rng3_uid66_rightShiferNoStickyOut_uid42_fpToFxPTest_b = shifterIn_uid41_fpToFxPTest_q[31:3];

    // rightShiftStage0Idx3_uid68_rightShiferNoStickyOut_uid42_fpToFxPTest(BITJOIN,67)@1
    assign rightShiftStage0Idx3_uid68_rightShiferNoStickyOut_uid42_fpToFxPTest_q = {rightShiftStage0Idx3Pad3_uid67_rightShiferNoStickyOut_uid42_fpToFxPTest_q, rightShiftStage0Idx3Rng3_uid66_rightShiferNoStickyOut_uid42_fpToFxPTest_b};

    // rightShiftStage0Idx2Pad2_uid64_rightShiferNoStickyOut_uid42_fpToFxPTest(CONSTANT,63)
    assign rightShiftStage0Idx2Pad2_uid64_rightShiferNoStickyOut_uid42_fpToFxPTest_q = 2'b00;

    // rightShiftStage0Idx2Rng2_uid63_rightShiferNoStickyOut_uid42_fpToFxPTest(BITSELECT,62)@1
    assign rightShiftStage0Idx2Rng2_uid63_rightShiferNoStickyOut_uid42_fpToFxPTest_b = shifterIn_uid41_fpToFxPTest_q[31:2];

    // rightShiftStage0Idx2_uid65_rightShiferNoStickyOut_uid42_fpToFxPTest(BITJOIN,64)@1
    assign rightShiftStage0Idx2_uid65_rightShiferNoStickyOut_uid42_fpToFxPTest_q = {rightShiftStage0Idx2Pad2_uid64_rightShiferNoStickyOut_uid42_fpToFxPTest_q, rightShiftStage0Idx2Rng2_uid63_rightShiferNoStickyOut_uid42_fpToFxPTest_b};

    // rightShiftStage0Idx1Rng1_uid60_rightShiferNoStickyOut_uid42_fpToFxPTest(BITSELECT,59)@1
    assign rightShiftStage0Idx1Rng1_uid60_rightShiferNoStickyOut_uid42_fpToFxPTest_b = shifterIn_uid41_fpToFxPTest_q[31:1];

    // rightShiftStage0Idx1_uid62_rightShiferNoStickyOut_uid42_fpToFxPTest(BITJOIN,61)@1
    assign rightShiftStage0Idx1_uid62_rightShiferNoStickyOut_uid42_fpToFxPTest_q = {GND_q, rightShiftStage0Idx1Rng1_uid60_rightShiferNoStickyOut_uid42_fpToFxPTest_b};

    // excZ_x_uid11_fpToFxPTest(LOGICAL,10)@0
    assign excZ_x_uid11_fpToFxPTest_q = exp_x_uid9_fpToFxPTest_b == cstAllZWE_uid8_fpToFxPTest_q ? 1'b1 : 1'b0;

    // invExcXZ_uid24_fpToFxPTest(LOGICAL,23)@0 + 1
    assign invExcXZ_uid24_fpToFxPTest_qi = ~ (excZ_x_uid11_fpToFxPTest_q);
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    invExcXZ_uid24_fpToFxPTest_delay ( .xin(invExcXZ_uid24_fpToFxPTest_qi), .xout(invExcXZ_uid24_fpToFxPTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // fracPostZ_uid23_fpToFxPTest(MUX,22)@0 + 1
    assign fracPostZ_uid23_fpToFxPTest_s = excZ_x_uid11_fpToFxPTest_q;
    always @ (posedge clk)
    begin
        if (areset)
        begin
            fracPostZ_uid23_fpToFxPTest_q <= 23'b0;
        end
        else if (en == 1'b1)
        begin
            unique case (fracPostZ_uid23_fpToFxPTest_s)
                1'b0 : fracPostZ_uid23_fpToFxPTest_q <= frac_x_uid10_fpToFxPTest_b;
                1'b1 : fracPostZ_uid23_fpToFxPTest_q <= cstZeroWF_uid7_fpToFxPTest_q;
                default : fracPostZ_uid23_fpToFxPTest_q <= 23'b0;
            endcase
        end
    end

    // oFracX_uid25_fpToFxPTest(BITJOIN,24)@1
    assign oFracX_uid25_fpToFxPTest_q = {invExcXZ_uid24_fpToFxPTest_q, fracPostZ_uid23_fpToFxPTest_q};

    // shifterIn_uid41_fpToFxPTest(BITJOIN,40)@1
    assign shifterIn_uid41_fpToFxPTest_q = {oFracX_uid25_fpToFxPTest_q, cstAllZWE_uid8_fpToFxPTest_q};

    // rightShiftStage0_uid70_rightShiferNoStickyOut_uid42_fpToFxPTest(MUX,69)@1
    assign rightShiftStage0_uid70_rightShiferNoStickyOut_uid42_fpToFxPTest_s = rightShiftStageSel0Dto0_uid69_rightShiferNoStickyOut_uid42_fpToFxPTest_merged_bit_select_b;
    always @(rightShiftStage0_uid70_rightShiferNoStickyOut_uid42_fpToFxPTest_s or en or shifterIn_uid41_fpToFxPTest_q or rightShiftStage0Idx1_uid62_rightShiferNoStickyOut_uid42_fpToFxPTest_q or rightShiftStage0Idx2_uid65_rightShiferNoStickyOut_uid42_fpToFxPTest_q or rightShiftStage0Idx3_uid68_rightShiferNoStickyOut_uid42_fpToFxPTest_q)
    begin
        unique case (rightShiftStage0_uid70_rightShiferNoStickyOut_uid42_fpToFxPTest_s)
            2'b00 : rightShiftStage0_uid70_rightShiferNoStickyOut_uid42_fpToFxPTest_q = shifterIn_uid41_fpToFxPTest_q;
            2'b01 : rightShiftStage0_uid70_rightShiferNoStickyOut_uid42_fpToFxPTest_q = rightShiftStage0Idx1_uid62_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
            2'b10 : rightShiftStage0_uid70_rightShiferNoStickyOut_uid42_fpToFxPTest_q = rightShiftStage0Idx2_uid65_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
            2'b11 : rightShiftStage0_uid70_rightShiferNoStickyOut_uid42_fpToFxPTest_q = rightShiftStage0Idx3_uid68_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
            default : rightShiftStage0_uid70_rightShiferNoStickyOut_uid42_fpToFxPTest_q = 32'b0;
        endcase
    end

    // rightShiftStage1_uid81_rightShiferNoStickyOut_uid42_fpToFxPTest(MUX,80)@1
    assign rightShiftStage1_uid81_rightShiferNoStickyOut_uid42_fpToFxPTest_s = rightShiftStageSel0Dto0_uid69_rightShiferNoStickyOut_uid42_fpToFxPTest_merged_bit_select_c;
    always @(rightShiftStage1_uid81_rightShiferNoStickyOut_uid42_fpToFxPTest_s or en or rightShiftStage0_uid70_rightShiferNoStickyOut_uid42_fpToFxPTest_q or rightShiftStage1Idx1_uid73_rightShiferNoStickyOut_uid42_fpToFxPTest_q or rightShiftStage1Idx2_uid76_rightShiferNoStickyOut_uid42_fpToFxPTest_q or rightShiftStage1Idx3_uid79_rightShiferNoStickyOut_uid42_fpToFxPTest_q)
    begin
        unique case (rightShiftStage1_uid81_rightShiferNoStickyOut_uid42_fpToFxPTest_s)
            2'b00 : rightShiftStage1_uid81_rightShiferNoStickyOut_uid42_fpToFxPTest_q = rightShiftStage0_uid70_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
            2'b01 : rightShiftStage1_uid81_rightShiferNoStickyOut_uid42_fpToFxPTest_q = rightShiftStage1Idx1_uid73_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
            2'b10 : rightShiftStage1_uid81_rightShiferNoStickyOut_uid42_fpToFxPTest_q = rightShiftStage1Idx2_uid76_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
            2'b11 : rightShiftStage1_uid81_rightShiferNoStickyOut_uid42_fpToFxPTest_q = rightShiftStage1Idx3_uid79_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
            default : rightShiftStage1_uid81_rightShiferNoStickyOut_uid42_fpToFxPTest_q = 32'b0;
        endcase
    end

    // maxShiftCst_uid37_fpToFxPTest(CONSTANT,36)
    assign maxShiftCst_uid37_fpToFxPTest_q = 6'b100000;

    // ovfExpVal_uid34_fpToFxPTest(CONSTANT,33)
    assign ovfExpVal_uid34_fpToFxPTest_q = 9'b010011101;

    // shiftValE_uid35_fpToFxPTest(SUB,34)@0
    assign shiftValE_uid35_fpToFxPTest_a = {{2{ovfExpVal_uid34_fpToFxPTest_q[8]}}, ovfExpVal_uid34_fpToFxPTest_q};
    assign shiftValE_uid35_fpToFxPTest_b = {3'b000, exp_x_uid9_fpToFxPTest_b};
    assign shiftValE_uid35_fpToFxPTest_o = $signed(shiftValE_uid35_fpToFxPTest_a) - $signed(shiftValE_uid35_fpToFxPTest_b);
    assign shiftValE_uid35_fpToFxPTest_q = shiftValE_uid35_fpToFxPTest_o[9:0];

    // shiftValRaw_uid36_fpToFxPTest(BITSELECT,35)@0
    assign shiftValRaw_uid36_fpToFxPTest_in = shiftValE_uid35_fpToFxPTest_q[5:0];
    assign shiftValRaw_uid36_fpToFxPTest_b = shiftValRaw_uid36_fpToFxPTest_in[5:0];

    // shiftOutOfRange_uid38_fpToFxPTest(COMPARE,37)@0
    assign shiftOutOfRange_uid38_fpToFxPTest_a = {{2{shiftValE_uid35_fpToFxPTest_q[9]}}, shiftValE_uid35_fpToFxPTest_q};
    assign shiftOutOfRange_uid38_fpToFxPTest_b = {6'b000000, maxShiftCst_uid37_fpToFxPTest_q};
    assign shiftOutOfRange_uid38_fpToFxPTest_o = $signed(shiftOutOfRange_uid38_fpToFxPTest_a) - $signed(shiftOutOfRange_uid38_fpToFxPTest_b);
    assign shiftOutOfRange_uid38_fpToFxPTest_n[0] = ~ (shiftOutOfRange_uid38_fpToFxPTest_o[11]);

    // shiftVal_uid39_fpToFxPTest(MUX,38)@0 + 1
    assign shiftVal_uid39_fpToFxPTest_s = shiftOutOfRange_uid38_fpToFxPTest_n;
    always @ (posedge clk)
    begin
        if (areset)
        begin
            shiftVal_uid39_fpToFxPTest_q <= 6'b0;
        end
        else if (en == 1'b1)
        begin
            unique case (shiftVal_uid39_fpToFxPTest_s)
                1'b0 : shiftVal_uid39_fpToFxPTest_q <= shiftValRaw_uid36_fpToFxPTest_b;
                1'b1 : shiftVal_uid39_fpToFxPTest_q <= maxShiftCst_uid37_fpToFxPTest_q;
                default : shiftVal_uid39_fpToFxPTest_q <= 6'b0;
            endcase
        end
    end

    // rightShiftStageSel0Dto0_uid69_rightShiferNoStickyOut_uid42_fpToFxPTest_merged_bit_select(BITSELECT,89)@1
    assign rightShiftStageSel0Dto0_uid69_rightShiferNoStickyOut_uid42_fpToFxPTest_merged_bit_select_b = shiftVal_uid39_fpToFxPTest_q[1:0];
    assign rightShiftStageSel0Dto0_uid69_rightShiferNoStickyOut_uid42_fpToFxPTest_merged_bit_select_c = shiftVal_uid39_fpToFxPTest_q[3:2];
    assign rightShiftStageSel0Dto0_uid69_rightShiferNoStickyOut_uid42_fpToFxPTest_merged_bit_select_d = shiftVal_uid39_fpToFxPTest_q[5:4];

    // rightShiftStage2_uid88_rightShiferNoStickyOut_uid42_fpToFxPTest(MUX,87)@1
    assign rightShiftStage2_uid88_rightShiferNoStickyOut_uid42_fpToFxPTest_s = rightShiftStageSel0Dto0_uid69_rightShiferNoStickyOut_uid42_fpToFxPTest_merged_bit_select_d;
    always @(rightShiftStage2_uid88_rightShiferNoStickyOut_uid42_fpToFxPTest_s or en or rightShiftStage1_uid81_rightShiferNoStickyOut_uid42_fpToFxPTest_q or rightShiftStage2Idx1_uid84_rightShiferNoStickyOut_uid42_fpToFxPTest_q or maxNegValueU_uid55_fpToFxPTest_q)
    begin
        unique case (rightShiftStage2_uid88_rightShiferNoStickyOut_uid42_fpToFxPTest_s)
            2'b00 : rightShiftStage2_uid88_rightShiferNoStickyOut_uid42_fpToFxPTest_q = rightShiftStage1_uid81_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
            2'b01 : rightShiftStage2_uid88_rightShiferNoStickyOut_uid42_fpToFxPTest_q = rightShiftStage2Idx1_uid84_rightShiferNoStickyOut_uid42_fpToFxPTest_q;
            2'b10 : rightShiftStage2_uid88_rightShiferNoStickyOut_uid42_fpToFxPTest_q = maxNegValueU_uid55_fpToFxPTest_q;
            2'b11 : rightShiftStage2_uid88_rightShiferNoStickyOut_uid42_fpToFxPTest_q = maxNegValueU_uid55_fpToFxPTest_q;
            default : rightShiftStage2_uid88_rightShiferNoStickyOut_uid42_fpToFxPTest_q = 32'b0;
        endcase
    end

    // zRightShiferNoStickyOut_uid45_fpToFxPTest(BITJOIN,44)@1
    assign zRightShiferNoStickyOut_uid45_fpToFxPTest_q = {GND_q, rightShiftStage2_uid88_rightShiferNoStickyOut_uid42_fpToFxPTest_q};

    // xXorSignE_uid46_fpToFxPTest(LOGICAL,45)@1 + 1
    assign xXorSignE_uid46_fpToFxPTest_b = {{32{signX_uid29_fpToFxPTest_q[0]}}, signX_uid29_fpToFxPTest_q};
    assign xXorSignE_uid46_fpToFxPTest_qi = zRightShiferNoStickyOut_uid45_fpToFxPTest_q ^ xXorSignE_uid46_fpToFxPTest_b;
    dspba_delay_ver #( .width(33), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    xXorSignE_uid46_fpToFxPTest_delay ( .xin(xXorSignE_uid46_fpToFxPTest_qi), .xout(xXorSignE_uid46_fpToFxPTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // sPostRndFull_uid48_fpToFxPTest(ADD,47)@2
    assign sPostRndFull_uid48_fpToFxPTest_a = {{1{xXorSignE_uid46_fpToFxPTest_q[32]}}, xXorSignE_uid46_fpToFxPTest_q};
    assign sPostRndFull_uid48_fpToFxPTest_b = {{31{d0_uid47_fpToFxPTest_q[2]}}, d0_uid47_fpToFxPTest_q};
    assign sPostRndFull_uid48_fpToFxPTest_o = $signed(sPostRndFull_uid48_fpToFxPTest_a) + $signed(sPostRndFull_uid48_fpToFxPTest_b);
    assign sPostRndFull_uid48_fpToFxPTest_q = sPostRndFull_uid48_fpToFxPTest_o[33:0];

    // sPostRnd_uid49_fpToFxPTest(BITSELECT,48)@2
    assign sPostRnd_uid49_fpToFxPTest_in = sPostRndFull_uid48_fpToFxPTest_q[32:0];
    assign sPostRnd_uid49_fpToFxPTest_b = sPostRnd_uid49_fpToFxPTest_in[32:1];

    // redist0_sPostRnd_uid49_fpToFxPTest_b_1(DELAY,90)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist0_sPostRnd_uid49_fpToFxPTest_b_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist0_sPostRnd_uid49_fpToFxPTest_b_1_q <= sPostRnd_uid49_fpToFxPTest_b;
        end
    end

    // redist3_signX_uid29_fpToFxPTest_q_3(DELAY,93)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist3_signX_uid29_fpToFxPTest_q_3_delay_0 <= '0;
            redist3_signX_uid29_fpToFxPTest_q_3_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist3_signX_uid29_fpToFxPTest_q_3_delay_0 <= signX_uid29_fpToFxPTest_q;
            redist3_signX_uid29_fpToFxPTest_q_3_q <= redist3_signX_uid29_fpToFxPTest_q_3_delay_0;
        end
    end

    // udfExpVal_uid32_fpToFxPTest(CONSTANT,31)
    assign udfExpVal_uid32_fpToFxPTest_q = 8'b01111101;

    // udf_uid33_fpToFxPTest(COMPARE,32)@0 + 1
    assign udf_uid33_fpToFxPTest_a = {{3{udfExpVal_uid32_fpToFxPTest_q[7]}}, udfExpVal_uid32_fpToFxPTest_q};
    assign udf_uid33_fpToFxPTest_b = {3'b000, exp_x_uid9_fpToFxPTest_b};
    always @ (posedge clk)
    begin
        if (areset)
        begin
            udf_uid33_fpToFxPTest_o <= 11'b0;
        end
        else if (en == 1'b1)
        begin
            udf_uid33_fpToFxPTest_o <= $signed(udf_uid33_fpToFxPTest_a) - $signed(udf_uid33_fpToFxPTest_b);
        end
    end
    assign udf_uid33_fpToFxPTest_n[0] = ~ (udf_uid33_fpToFxPTest_o[10]);

    // redist1_udf_uid33_fpToFxPTest_n_3(DELAY,91)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist1_udf_uid33_fpToFxPTest_n_3_delay_0 <= '0;
            redist1_udf_uid33_fpToFxPTest_n_3_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist1_udf_uid33_fpToFxPTest_n_3_delay_0 <= udf_uid33_fpToFxPTest_n;
            redist1_udf_uid33_fpToFxPTest_n_3_q <= redist1_udf_uid33_fpToFxPTest_n_3_delay_0;
        end
    end

    // sPostRnd_uid50_fpToFxPTest(BITSELECT,49)@2
    assign sPostRnd_uid50_fpToFxPTest_in = {{1{sPostRndFull_uid48_fpToFxPTest_q[33]}}, sPostRndFull_uid48_fpToFxPTest_q};
    assign sPostRnd_uid50_fpToFxPTest_b = sPostRnd_uid50_fpToFxPTest_in[34:1];

    // rndOvfPos_uid51_fpToFxPTest(COMPARE,50)@2 + 1
    assign rndOvfPos_uid51_fpToFxPTest_a = {4'b0000, maxPosValueS_uid43_fpToFxPTest_q};
    assign rndOvfPos_uid51_fpToFxPTest_b = {{2{sPostRnd_uid50_fpToFxPTest_b[33]}}, sPostRnd_uid50_fpToFxPTest_b};
    always @ (posedge clk)
    begin
        if (areset)
        begin
            rndOvfPos_uid51_fpToFxPTest_o <= 36'b0;
        end
        else if (en == 1'b1)
        begin
            rndOvfPos_uid51_fpToFxPTest_o <= $signed(rndOvfPos_uid51_fpToFxPTest_a) - $signed(rndOvfPos_uid51_fpToFxPTest_b);
        end
    end
    assign rndOvfPos_uid51_fpToFxPTest_c[0] = rndOvfPos_uid51_fpToFxPTest_o[35];

    // ovfExpVal_uid30_fpToFxPTest(CONSTANT,29)
    assign ovfExpVal_uid30_fpToFxPTest_q = 9'b010011110;

    // ovfExpRange_uid31_fpToFxPTest(COMPARE,30)@0 + 1
    assign ovfExpRange_uid31_fpToFxPTest_a = {3'b000, exp_x_uid9_fpToFxPTest_b};
    assign ovfExpRange_uid31_fpToFxPTest_b = {{2{ovfExpVal_uid30_fpToFxPTest_q[8]}}, ovfExpVal_uid30_fpToFxPTest_q};
    always @ (posedge clk)
    begin
        if (areset)
        begin
            ovfExpRange_uid31_fpToFxPTest_o <= 11'b0;
        end
        else if (en == 1'b1)
        begin
            ovfExpRange_uid31_fpToFxPTest_o <= $signed(ovfExpRange_uid31_fpToFxPTest_a) - $signed(ovfExpRange_uid31_fpToFxPTest_b);
        end
    end
    assign ovfExpRange_uid31_fpToFxPTest_n[0] = ~ (ovfExpRange_uid31_fpToFxPTest_o[10]);

    // redist2_ovfExpRange_uid31_fpToFxPTest_n_3(DELAY,92)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist2_ovfExpRange_uid31_fpToFxPTest_n_3_delay_0 <= '0;
            redist2_ovfExpRange_uid31_fpToFxPTest_n_3_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist2_ovfExpRange_uid31_fpToFxPTest_n_3_delay_0 <= ovfExpRange_uid31_fpToFxPTest_n;
            redist2_ovfExpRange_uid31_fpToFxPTest_n_3_q <= redist2_ovfExpRange_uid31_fpToFxPTest_n_3_delay_0;
        end
    end

    // excI_x_uid15_fpToFxPTest(LOGICAL,14)@0 + 1
    assign excI_x_uid15_fpToFxPTest_qi = expXIsMax_uid12_fpToFxPTest_q & fracXIsZero_uid13_fpToFxPTest_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    excI_x_uid15_fpToFxPTest_delay ( .xin(excI_x_uid15_fpToFxPTest_qi), .xout(excI_x_uid15_fpToFxPTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist5_excI_x_uid15_fpToFxPTest_q_3(DELAY,95)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist5_excI_x_uid15_fpToFxPTest_q_3_delay_0 <= '0;
            redist5_excI_x_uid15_fpToFxPTest_q_3_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist5_excI_x_uid15_fpToFxPTest_q_3_delay_0 <= excI_x_uid15_fpToFxPTest_q;
            redist5_excI_x_uid15_fpToFxPTest_q_3_q <= redist5_excI_x_uid15_fpToFxPTest_q_3_delay_0;
        end
    end

    // redist4_excN_x_uid16_fpToFxPTest_q_3(DELAY,94)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist4_excN_x_uid16_fpToFxPTest_q_3_delay_0 <= '0;
            redist4_excN_x_uid16_fpToFxPTest_q_3_delay_1 <= '0;
            redist4_excN_x_uid16_fpToFxPTest_q_3_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist4_excN_x_uid16_fpToFxPTest_q_3_delay_0 <= excN_x_uid16_fpToFxPTest_q;
            redist4_excN_x_uid16_fpToFxPTest_q_3_delay_1 <= redist4_excN_x_uid16_fpToFxPTest_q_3_delay_0;
            redist4_excN_x_uid16_fpToFxPTest_q_3_q <= redist4_excN_x_uid16_fpToFxPTest_q_3_delay_1;
        end
    end

    // ovfPostRnd_uid52_fpToFxPTest(LOGICAL,51)@3
    assign ovfPostRnd_uid52_fpToFxPTest_q = redist4_excN_x_uid16_fpToFxPTest_q_3_q | redist5_excI_x_uid15_fpToFxPTest_q_3_q | redist2_ovfExpRange_uid31_fpToFxPTest_n_3_q | rndOvfPos_uid51_fpToFxPTest_c;

    // muxSelConc_uid53_fpToFxPTest(BITJOIN,52)@3
    assign muxSelConc_uid53_fpToFxPTest_q = {redist3_signX_uid29_fpToFxPTest_q_3_q, redist1_udf_uid33_fpToFxPTest_n_3_q, ovfPostRnd_uid52_fpToFxPTest_q};

    // muxSel_uid54_fpToFxPTest(LOOKUP,53)@3
    always @(muxSelConc_uid53_fpToFxPTest_q)
    begin
        // Begin reserved scope level
        unique case (muxSelConc_uid53_fpToFxPTest_q)
            3'b000 : muxSel_uid54_fpToFxPTest_q = 2'b00;
            3'b001 : muxSel_uid54_fpToFxPTest_q = 2'b01;
            3'b010 : muxSel_uid54_fpToFxPTest_q = 2'b11;
            3'b011 : muxSel_uid54_fpToFxPTest_q = 2'b11;
            3'b100 : muxSel_uid54_fpToFxPTest_q = 2'b00;
            3'b101 : muxSel_uid54_fpToFxPTest_q = 2'b10;
            3'b110 : muxSel_uid54_fpToFxPTest_q = 2'b11;
            3'b111 : muxSel_uid54_fpToFxPTest_q = 2'b11;
            default : begin
                          // unreachable
                          muxSel_uid54_fpToFxPTest_q = 2'bxx;
                      end
        endcase
        // End reserved scope level
    end

    // finalOut_uid56_fpToFxPTest(MUX,55)@3
    assign finalOut_uid56_fpToFxPTest_s = muxSel_uid54_fpToFxPTest_q;
    always @(finalOut_uid56_fpToFxPTest_s or en or redist0_sPostRnd_uid49_fpToFxPTest_b_1_q or maxPosValueS_uid43_fpToFxPTest_q or maxNegValueS_uid44_fpToFxPTest_q or maxNegValueU_uid55_fpToFxPTest_q)
    begin
        unique case (finalOut_uid56_fpToFxPTest_s)
            2'b00 : finalOut_uid56_fpToFxPTest_q = redist0_sPostRnd_uid49_fpToFxPTest_b_1_q;
            2'b01 : finalOut_uid56_fpToFxPTest_q = maxPosValueS_uid43_fpToFxPTest_q;
            2'b10 : finalOut_uid56_fpToFxPTest_q = maxNegValueS_uid44_fpToFxPTest_q;
            2'b11 : finalOut_uid56_fpToFxPTest_q = maxNegValueU_uid55_fpToFxPTest_q;
            default : finalOut_uid56_fpToFxPTest_q = 32'b0;
        endcase
    end

    // xOut(GPOUT,4)@3
    assign q = finalOut_uid56_fpToFxPTest_q;

endmodule
