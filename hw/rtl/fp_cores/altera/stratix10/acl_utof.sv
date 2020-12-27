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

// SystemVerilog created from acl_utof
// SystemVerilog created on Sun Dec 27 09:48:58 2020


(* altera_attribute = "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007; -name MESSAGE_DISABLE 10958" *)
module acl_utof (
    input wire [31:0] a,
    input wire [0:0] en,
    output wire [31:0] q,
    input wire clk,
    input wire areset
    );

    wire [0:0] GND_q;
    wire [5:0] maxCount_uid7_fxpToFPTest_q;
    wire [0:0] inIsZero_uid8_fxpToFPTest_qi;
    reg [0:0] inIsZero_uid8_fxpToFPTest_q;
    wire [7:0] msbIn_uid9_fxpToFPTest_q;
    wire [8:0] expPreRnd_uid10_fxpToFPTest_a;
    wire [8:0] expPreRnd_uid10_fxpToFPTest_b;
    logic [8:0] expPreRnd_uid10_fxpToFPTest_o;
    wire [8:0] expPreRnd_uid10_fxpToFPTest_q;
    wire [32:0] expFracRnd_uid12_fxpToFPTest_q;
    wire [0:0] sticky_uid16_fxpToFPTest_q;
    wire [0:0] nr_uid17_fxpToFPTest_q;
    wire [0:0] rnd_uid18_fxpToFPTest_qi;
    reg [0:0] rnd_uid18_fxpToFPTest_q;
    wire [34:0] expFracR_uid20_fxpToFPTest_a;
    wire [34:0] expFracR_uid20_fxpToFPTest_b;
    logic [34:0] expFracR_uid20_fxpToFPTest_o;
    wire [33:0] expFracR_uid20_fxpToFPTest_q;
    wire [23:0] fracR_uid21_fxpToFPTest_in;
    wire [22:0] fracR_uid21_fxpToFPTest_b;
    wire [9:0] expR_uid22_fxpToFPTest_b;
    wire [11:0] udf_uid23_fxpToFPTest_a;
    wire [11:0] udf_uid23_fxpToFPTest_b;
    logic [11:0] udf_uid23_fxpToFPTest_o;
    wire [0:0] udf_uid23_fxpToFPTest_n;
    wire [7:0] expInf_uid24_fxpToFPTest_q;
    wire [11:0] ovf_uid25_fxpToFPTest_a;
    wire [11:0] ovf_uid25_fxpToFPTest_b;
    logic [11:0] ovf_uid25_fxpToFPTest_o;
    wire [0:0] ovf_uid25_fxpToFPTest_n;
    wire [0:0] excSelector_uid26_fxpToFPTest_q;
    wire [22:0] fracZ_uid27_fxpToFPTest_q;
    wire [0:0] fracRPostExc_uid28_fxpToFPTest_s;
    reg [22:0] fracRPostExc_uid28_fxpToFPTest_q;
    wire [0:0] udfOrInZero_uid29_fxpToFPTest_q;
    wire [1:0] excSelector_uid30_fxpToFPTest_q;
    wire [7:0] expZ_uid33_fxpToFPTest_q;
    wire [7:0] expR_uid34_fxpToFPTest_in;
    wire [7:0] expR_uid34_fxpToFPTest_b;
    wire [1:0] expRPostExc_uid35_fxpToFPTest_s;
    reg [7:0] expRPostExc_uid35_fxpToFPTest_q;
    wire [31:0] outRes_uid36_fxpToFPTest_q;
    wire [31:0] zs_uid38_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [0:0] vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [0:0] vStagei_uid42_lzcShifterZ1_uid6_fxpToFPTest_s;
    reg [31:0] vStagei_uid42_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [15:0] zs_uid43_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [0:0] vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [31:0] cStage_uid48_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [0:0] vStagei_uid49_lzcShifterZ1_uid6_fxpToFPTest_s;
    reg [31:0] vStagei_uid49_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [0:0] vCount_uid52_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [31:0] cStage_uid55_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [0:0] vStagei_uid56_lzcShifterZ1_uid6_fxpToFPTest_s;
    reg [31:0] vStagei_uid56_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [3:0] zs_uid57_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [0:0] vCount_uid59_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [31:0] cStage_uid62_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [0:0] vStagei_uid63_lzcShifterZ1_uid6_fxpToFPTest_s;
    reg [31:0] vStagei_uid63_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [1:0] zs_uid64_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [0:0] vCount_uid66_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [31:0] cStage_uid69_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [0:0] vStagei_uid70_lzcShifterZ1_uid6_fxpToFPTest_s;
    reg [31:0] vStagei_uid70_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [0:0] vCount_uid73_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [31:0] cStage_uid76_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [0:0] vStagei_uid77_lzcShifterZ1_uid6_fxpToFPTest_s;
    reg [31:0] vStagei_uid77_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [5:0] vCount_uid78_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [7:0] vCountBig_uid80_lzcShifterZ1_uid6_fxpToFPTest_a;
    wire [7:0] vCountBig_uid80_lzcShifterZ1_uid6_fxpToFPTest_b;
    logic [7:0] vCountBig_uid80_lzcShifterZ1_uid6_fxpToFPTest_o;
    wire [0:0] vCountBig_uid80_lzcShifterZ1_uid6_fxpToFPTest_c;
    wire [0:0] vCountFinal_uid82_lzcShifterZ1_uid6_fxpToFPTest_s;
    reg [5:0] vCountFinal_uid82_lzcShifterZ1_uid6_fxpToFPTest_q;
    wire [1:0] l_uid13_fxpToFPTest_merged_bit_select_in;
    wire [0:0] l_uid13_fxpToFPTest_merged_bit_select_b;
    wire [0:0] l_uid13_fxpToFPTest_merged_bit_select_c;
    wire [15:0] rVStage_uid44_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_b;
    wire [15:0] rVStage_uid44_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_c;
    wire [7:0] rVStage_uid51_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_b;
    wire [23:0] rVStage_uid51_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_c;
    wire [3:0] rVStage_uid58_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_b;
    wire [27:0] rVStage_uid58_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_c;
    wire [1:0] rVStage_uid65_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_b;
    wire [29:0] rVStage_uid65_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_c;
    wire [0:0] rVStage_uid72_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_b;
    wire [30:0] rVStage_uid72_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_c;
    wire [30:0] fracRnd_uid11_fxpToFPTest_merged_bit_select_in;
    wire [23:0] fracRnd_uid11_fxpToFPTest_merged_bit_select_b;
    wire [6:0] fracRnd_uid11_fxpToFPTest_merged_bit_select_c;
    reg [23:0] redist0_fracRnd_uid11_fxpToFPTest_merged_bit_select_b_1_q;
    reg [6:0] redist1_fracRnd_uid11_fxpToFPTest_merged_bit_select_c_1_q;
    reg [5:0] redist2_vCount_uid78_lzcShifterZ1_uid6_fxpToFPTest_q_1_q;
    reg [0:0] redist3_vCount_uid59_lzcShifterZ1_uid6_fxpToFPTest_q_1_q;
    reg [0:0] redist4_vCount_uid52_lzcShifterZ1_uid6_fxpToFPTest_q_2_q;
    reg [0:0] redist4_vCount_uid52_lzcShifterZ1_uid6_fxpToFPTest_q_2_delay_0;
    reg [0:0] redist5_vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest_q_3_q;
    reg [0:0] redist5_vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest_q_3_delay_0;
    reg [0:0] redist5_vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest_q_3_delay_1;
    reg [0:0] redist6_vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q_4_q;
    reg [0:0] redist6_vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q_4_delay_0;
    reg [0:0] redist6_vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q_4_delay_1;
    reg [0:0] redist6_vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q_4_delay_2;
    reg [9:0] redist7_expR_uid22_fxpToFPTest_b_1_q;
    reg [22:0] redist8_fracR_uid21_fxpToFPTest_b_1_q;
    reg [32:0] redist9_expFracRnd_uid12_fxpToFPTest_q_1_q;
    reg [0:0] redist10_inIsZero_uid8_fxpToFPTest_q_2_q;


    // GND(CONSTANT,0)
    assign GND_q = 1'b0;

    // expInf_uid24_fxpToFPTest(CONSTANT,23)
    assign expInf_uid24_fxpToFPTest_q = 8'b11111111;

    // expZ_uid33_fxpToFPTest(CONSTANT,32)
    assign expZ_uid33_fxpToFPTest_q = 8'b00000000;

    // rVStage_uid72_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select(BITSELECT,89)@4
    assign rVStage_uid72_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_b = vStagei_uid70_lzcShifterZ1_uid6_fxpToFPTest_q[31:31];
    assign rVStage_uid72_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_c = vStagei_uid70_lzcShifterZ1_uid6_fxpToFPTest_q[30:0];

    // cStage_uid76_lzcShifterZ1_uid6_fxpToFPTest(BITJOIN,75)@4
    assign cStage_uid76_lzcShifterZ1_uid6_fxpToFPTest_q = {rVStage_uid72_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_c, GND_q};

    // rVStage_uid65_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select(BITSELECT,88)@4
    assign rVStage_uid65_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_b = vStagei_uid63_lzcShifterZ1_uid6_fxpToFPTest_q[31:30];
    assign rVStage_uid65_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_c = vStagei_uid63_lzcShifterZ1_uid6_fxpToFPTest_q[29:0];

    // zs_uid64_lzcShifterZ1_uid6_fxpToFPTest(CONSTANT,63)
    assign zs_uid64_lzcShifterZ1_uid6_fxpToFPTest_q = 2'b00;

    // cStage_uid69_lzcShifterZ1_uid6_fxpToFPTest(BITJOIN,68)@4
    assign cStage_uid69_lzcShifterZ1_uid6_fxpToFPTest_q = {rVStage_uid65_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_c, zs_uid64_lzcShifterZ1_uid6_fxpToFPTest_q};

    // rVStage_uid58_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select(BITSELECT,87)@3
    assign rVStage_uid58_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_b = vStagei_uid56_lzcShifterZ1_uid6_fxpToFPTest_q[31:28];
    assign rVStage_uid58_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_c = vStagei_uid56_lzcShifterZ1_uid6_fxpToFPTest_q[27:0];

    // zs_uid57_lzcShifterZ1_uid6_fxpToFPTest(CONSTANT,56)
    assign zs_uid57_lzcShifterZ1_uid6_fxpToFPTest_q = 4'b0000;

    // cStage_uid62_lzcShifterZ1_uid6_fxpToFPTest(BITJOIN,61)@3
    assign cStage_uid62_lzcShifterZ1_uid6_fxpToFPTest_q = {rVStage_uid58_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_c, zs_uid57_lzcShifterZ1_uid6_fxpToFPTest_q};

    // rVStage_uid51_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select(BITSELECT,86)@2
    assign rVStage_uid51_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_b = vStagei_uid49_lzcShifterZ1_uid6_fxpToFPTest_q[31:24];
    assign rVStage_uid51_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_c = vStagei_uid49_lzcShifterZ1_uid6_fxpToFPTest_q[23:0];

    // cStage_uid55_lzcShifterZ1_uid6_fxpToFPTest(BITJOIN,54)@2
    assign cStage_uid55_lzcShifterZ1_uid6_fxpToFPTest_q = {rVStage_uid51_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_c, expZ_uid33_fxpToFPTest_q};

    // rVStage_uid44_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select(BITSELECT,85)@1
    assign rVStage_uid44_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_b = vStagei_uid42_lzcShifterZ1_uid6_fxpToFPTest_q[31:16];
    assign rVStage_uid44_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_c = vStagei_uid42_lzcShifterZ1_uid6_fxpToFPTest_q[15:0];

    // zs_uid43_lzcShifterZ1_uid6_fxpToFPTest(CONSTANT,42)
    assign zs_uid43_lzcShifterZ1_uid6_fxpToFPTest_q = 16'b0000000000000000;

    // cStage_uid48_lzcShifterZ1_uid6_fxpToFPTest(BITJOIN,47)@1
    assign cStage_uid48_lzcShifterZ1_uid6_fxpToFPTest_q = {rVStage_uid44_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_c, zs_uid43_lzcShifterZ1_uid6_fxpToFPTest_q};

    // zs_uid38_lzcShifterZ1_uid6_fxpToFPTest(CONSTANT,37)
    assign zs_uid38_lzcShifterZ1_uid6_fxpToFPTest_q = 32'b00000000000000000000000000000000;

    // vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest(LOGICAL,39)@0
    assign vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q = a == zs_uid38_lzcShifterZ1_uid6_fxpToFPTest_q ? 1'b1 : 1'b0;

    // vStagei_uid42_lzcShifterZ1_uid6_fxpToFPTest(MUX,41)@0 + 1
    assign vStagei_uid42_lzcShifterZ1_uid6_fxpToFPTest_s = vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q;
    always @ (posedge clk)
    begin
        if (areset)
        begin
            vStagei_uid42_lzcShifterZ1_uid6_fxpToFPTest_q <= 32'b0;
        end
        else if (en == 1'b1)
        begin
            unique case (vStagei_uid42_lzcShifterZ1_uid6_fxpToFPTest_s)
                1'b0 : vStagei_uid42_lzcShifterZ1_uid6_fxpToFPTest_q <= a;
                1'b1 : vStagei_uid42_lzcShifterZ1_uid6_fxpToFPTest_q <= zs_uid38_lzcShifterZ1_uid6_fxpToFPTest_q;
                default : vStagei_uid42_lzcShifterZ1_uid6_fxpToFPTest_q <= 32'b0;
            endcase
        end
    end

    // vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest(LOGICAL,44)@1
    assign vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest_q = rVStage_uid44_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_b == zs_uid43_lzcShifterZ1_uid6_fxpToFPTest_q ? 1'b1 : 1'b0;

    // vStagei_uid49_lzcShifterZ1_uid6_fxpToFPTest(MUX,48)@1 + 1
    assign vStagei_uid49_lzcShifterZ1_uid6_fxpToFPTest_s = vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest_q;
    always @ (posedge clk)
    begin
        if (areset)
        begin
            vStagei_uid49_lzcShifterZ1_uid6_fxpToFPTest_q <= 32'b0;
        end
        else if (en == 1'b1)
        begin
            unique case (vStagei_uid49_lzcShifterZ1_uid6_fxpToFPTest_s)
                1'b0 : vStagei_uid49_lzcShifterZ1_uid6_fxpToFPTest_q <= vStagei_uid42_lzcShifterZ1_uid6_fxpToFPTest_q;
                1'b1 : vStagei_uid49_lzcShifterZ1_uid6_fxpToFPTest_q <= cStage_uid48_lzcShifterZ1_uid6_fxpToFPTest_q;
                default : vStagei_uid49_lzcShifterZ1_uid6_fxpToFPTest_q <= 32'b0;
            endcase
        end
    end

    // vCount_uid52_lzcShifterZ1_uid6_fxpToFPTest(LOGICAL,51)@2
    assign vCount_uid52_lzcShifterZ1_uid6_fxpToFPTest_q = rVStage_uid51_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_b == expZ_uid33_fxpToFPTest_q ? 1'b1 : 1'b0;

    // vStagei_uid56_lzcShifterZ1_uid6_fxpToFPTest(MUX,55)@2 + 1
    assign vStagei_uid56_lzcShifterZ1_uid6_fxpToFPTest_s = vCount_uid52_lzcShifterZ1_uid6_fxpToFPTest_q;
    always @ (posedge clk)
    begin
        if (areset)
        begin
            vStagei_uid56_lzcShifterZ1_uid6_fxpToFPTest_q <= 32'b0;
        end
        else if (en == 1'b1)
        begin
            unique case (vStagei_uid56_lzcShifterZ1_uid6_fxpToFPTest_s)
                1'b0 : vStagei_uid56_lzcShifterZ1_uid6_fxpToFPTest_q <= vStagei_uid49_lzcShifterZ1_uid6_fxpToFPTest_q;
                1'b1 : vStagei_uid56_lzcShifterZ1_uid6_fxpToFPTest_q <= cStage_uid55_lzcShifterZ1_uid6_fxpToFPTest_q;
                default : vStagei_uid56_lzcShifterZ1_uid6_fxpToFPTest_q <= 32'b0;
            endcase
        end
    end

    // vCount_uid59_lzcShifterZ1_uid6_fxpToFPTest(LOGICAL,58)@3
    assign vCount_uid59_lzcShifterZ1_uid6_fxpToFPTest_q = rVStage_uid58_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_b == zs_uid57_lzcShifterZ1_uid6_fxpToFPTest_q ? 1'b1 : 1'b0;

    // vStagei_uid63_lzcShifterZ1_uid6_fxpToFPTest(MUX,62)@3 + 1
    assign vStagei_uid63_lzcShifterZ1_uid6_fxpToFPTest_s = vCount_uid59_lzcShifterZ1_uid6_fxpToFPTest_q;
    always @ (posedge clk)
    begin
        if (areset)
        begin
            vStagei_uid63_lzcShifterZ1_uid6_fxpToFPTest_q <= 32'b0;
        end
        else if (en == 1'b1)
        begin
            unique case (vStagei_uid63_lzcShifterZ1_uid6_fxpToFPTest_s)
                1'b0 : vStagei_uid63_lzcShifterZ1_uid6_fxpToFPTest_q <= vStagei_uid56_lzcShifterZ1_uid6_fxpToFPTest_q;
                1'b1 : vStagei_uid63_lzcShifterZ1_uid6_fxpToFPTest_q <= cStage_uid62_lzcShifterZ1_uid6_fxpToFPTest_q;
                default : vStagei_uid63_lzcShifterZ1_uid6_fxpToFPTest_q <= 32'b0;
            endcase
        end
    end

    // vCount_uid66_lzcShifterZ1_uid6_fxpToFPTest(LOGICAL,65)@4
    assign vCount_uid66_lzcShifterZ1_uid6_fxpToFPTest_q = rVStage_uid65_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_b == zs_uid64_lzcShifterZ1_uid6_fxpToFPTest_q ? 1'b1 : 1'b0;

    // vStagei_uid70_lzcShifterZ1_uid6_fxpToFPTest(MUX,69)@4
    assign vStagei_uid70_lzcShifterZ1_uid6_fxpToFPTest_s = vCount_uid66_lzcShifterZ1_uid6_fxpToFPTest_q;
    always @(vStagei_uid70_lzcShifterZ1_uid6_fxpToFPTest_s or en or vStagei_uid63_lzcShifterZ1_uid6_fxpToFPTest_q or cStage_uid69_lzcShifterZ1_uid6_fxpToFPTest_q)
    begin
        unique case (vStagei_uid70_lzcShifterZ1_uid6_fxpToFPTest_s)
            1'b0 : vStagei_uid70_lzcShifterZ1_uid6_fxpToFPTest_q = vStagei_uid63_lzcShifterZ1_uid6_fxpToFPTest_q;
            1'b1 : vStagei_uid70_lzcShifterZ1_uid6_fxpToFPTest_q = cStage_uid69_lzcShifterZ1_uid6_fxpToFPTest_q;
            default : vStagei_uid70_lzcShifterZ1_uid6_fxpToFPTest_q = 32'b0;
        endcase
    end

    // vCount_uid73_lzcShifterZ1_uid6_fxpToFPTest(LOGICAL,72)@4
    assign vCount_uid73_lzcShifterZ1_uid6_fxpToFPTest_q = rVStage_uid72_lzcShifterZ1_uid6_fxpToFPTest_merged_bit_select_b == GND_q ? 1'b1 : 1'b0;

    // vStagei_uid77_lzcShifterZ1_uid6_fxpToFPTest(MUX,76)@4
    assign vStagei_uid77_lzcShifterZ1_uid6_fxpToFPTest_s = vCount_uid73_lzcShifterZ1_uid6_fxpToFPTest_q;
    always @(vStagei_uid77_lzcShifterZ1_uid6_fxpToFPTest_s or en or vStagei_uid70_lzcShifterZ1_uid6_fxpToFPTest_q or cStage_uid76_lzcShifterZ1_uid6_fxpToFPTest_q)
    begin
        unique case (vStagei_uid77_lzcShifterZ1_uid6_fxpToFPTest_s)
            1'b0 : vStagei_uid77_lzcShifterZ1_uid6_fxpToFPTest_q = vStagei_uid70_lzcShifterZ1_uid6_fxpToFPTest_q;
            1'b1 : vStagei_uid77_lzcShifterZ1_uid6_fxpToFPTest_q = cStage_uid76_lzcShifterZ1_uid6_fxpToFPTest_q;
            default : vStagei_uid77_lzcShifterZ1_uid6_fxpToFPTest_q = 32'b0;
        endcase
    end

    // fracRnd_uid11_fxpToFPTest_merged_bit_select(BITSELECT,90)@4
    assign fracRnd_uid11_fxpToFPTest_merged_bit_select_in = vStagei_uid77_lzcShifterZ1_uid6_fxpToFPTest_q[30:0];
    assign fracRnd_uid11_fxpToFPTest_merged_bit_select_b = fracRnd_uid11_fxpToFPTest_merged_bit_select_in[30:7];
    assign fracRnd_uid11_fxpToFPTest_merged_bit_select_c = fracRnd_uid11_fxpToFPTest_merged_bit_select_in[6:0];

    // redist1_fracRnd_uid11_fxpToFPTest_merged_bit_select_c_1(DELAY,92)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist1_fracRnd_uid11_fxpToFPTest_merged_bit_select_c_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist1_fracRnd_uid11_fxpToFPTest_merged_bit_select_c_1_q <= fracRnd_uid11_fxpToFPTest_merged_bit_select_c;
        end
    end

    // sticky_uid16_fxpToFPTest(LOGICAL,15)@5
    assign sticky_uid16_fxpToFPTest_q = redist1_fracRnd_uid11_fxpToFPTest_merged_bit_select_c_1_q != 7'b0000000 ? 1'b1 : 1'b0;

    // nr_uid17_fxpToFPTest(LOGICAL,16)@5
    assign nr_uid17_fxpToFPTest_q = ~ (l_uid13_fxpToFPTest_merged_bit_select_c);

    // maxCount_uid7_fxpToFPTest(CONSTANT,6)
    assign maxCount_uid7_fxpToFPTest_q = 6'b100000;

    // redist6_vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q_4(DELAY,97)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist6_vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q_4_delay_0 <= '0;
            redist6_vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q_4_delay_1 <= '0;
            redist6_vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q_4_delay_2 <= '0;
            redist6_vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q_4_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist6_vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q_4_delay_0 <= vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q;
            redist6_vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q_4_delay_1 <= redist6_vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q_4_delay_0;
            redist6_vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q_4_delay_2 <= redist6_vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q_4_delay_1;
            redist6_vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q_4_q <= redist6_vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q_4_delay_2;
        end
    end

    // redist5_vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest_q_3(DELAY,96)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist5_vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest_q_3_delay_0 <= '0;
            redist5_vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest_q_3_delay_1 <= '0;
            redist5_vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest_q_3_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist5_vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest_q_3_delay_0 <= vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest_q;
            redist5_vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest_q_3_delay_1 <= redist5_vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest_q_3_delay_0;
            redist5_vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest_q_3_q <= redist5_vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest_q_3_delay_1;
        end
    end

    // redist4_vCount_uid52_lzcShifterZ1_uid6_fxpToFPTest_q_2(DELAY,95)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist4_vCount_uid52_lzcShifterZ1_uid6_fxpToFPTest_q_2_delay_0 <= '0;
            redist4_vCount_uid52_lzcShifterZ1_uid6_fxpToFPTest_q_2_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist4_vCount_uid52_lzcShifterZ1_uid6_fxpToFPTest_q_2_delay_0 <= vCount_uid52_lzcShifterZ1_uid6_fxpToFPTest_q;
            redist4_vCount_uid52_lzcShifterZ1_uid6_fxpToFPTest_q_2_q <= redist4_vCount_uid52_lzcShifterZ1_uid6_fxpToFPTest_q_2_delay_0;
        end
    end

    // redist3_vCount_uid59_lzcShifterZ1_uid6_fxpToFPTest_q_1(DELAY,94)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist3_vCount_uid59_lzcShifterZ1_uid6_fxpToFPTest_q_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist3_vCount_uid59_lzcShifterZ1_uid6_fxpToFPTest_q_1_q <= vCount_uid59_lzcShifterZ1_uid6_fxpToFPTest_q;
        end
    end

    // vCount_uid78_lzcShifterZ1_uid6_fxpToFPTest(BITJOIN,77)@4
    assign vCount_uid78_lzcShifterZ1_uid6_fxpToFPTest_q = {redist6_vCount_uid40_lzcShifterZ1_uid6_fxpToFPTest_q_4_q, redist5_vCount_uid45_lzcShifterZ1_uid6_fxpToFPTest_q_3_q, redist4_vCount_uid52_lzcShifterZ1_uid6_fxpToFPTest_q_2_q, redist3_vCount_uid59_lzcShifterZ1_uid6_fxpToFPTest_q_1_q, vCount_uid66_lzcShifterZ1_uid6_fxpToFPTest_q, vCount_uid73_lzcShifterZ1_uid6_fxpToFPTest_q};

    // redist2_vCount_uid78_lzcShifterZ1_uid6_fxpToFPTest_q_1(DELAY,93)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist2_vCount_uid78_lzcShifterZ1_uid6_fxpToFPTest_q_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist2_vCount_uid78_lzcShifterZ1_uid6_fxpToFPTest_q_1_q <= vCount_uid78_lzcShifterZ1_uid6_fxpToFPTest_q;
        end
    end

    // vCountBig_uid80_lzcShifterZ1_uid6_fxpToFPTest(COMPARE,79)@4 + 1
    assign vCountBig_uid80_lzcShifterZ1_uid6_fxpToFPTest_a = {2'b00, maxCount_uid7_fxpToFPTest_q};
    assign vCountBig_uid80_lzcShifterZ1_uid6_fxpToFPTest_b = {2'b00, vCount_uid78_lzcShifterZ1_uid6_fxpToFPTest_q};
    always @ (posedge clk)
    begin
        if (areset)
        begin
            vCountBig_uid80_lzcShifterZ1_uid6_fxpToFPTest_o <= 8'b0;
        end
        else if (en == 1'b1)
        begin
            vCountBig_uid80_lzcShifterZ1_uid6_fxpToFPTest_o <= $unsigned(vCountBig_uid80_lzcShifterZ1_uid6_fxpToFPTest_a) - $unsigned(vCountBig_uid80_lzcShifterZ1_uid6_fxpToFPTest_b);
        end
    end
    assign vCountBig_uid80_lzcShifterZ1_uid6_fxpToFPTest_c[0] = vCountBig_uid80_lzcShifterZ1_uid6_fxpToFPTest_o[7];

    // vCountFinal_uid82_lzcShifterZ1_uid6_fxpToFPTest(MUX,81)@5
    assign vCountFinal_uid82_lzcShifterZ1_uid6_fxpToFPTest_s = vCountBig_uid80_lzcShifterZ1_uid6_fxpToFPTest_c;
    always @(vCountFinal_uid82_lzcShifterZ1_uid6_fxpToFPTest_s or en or redist2_vCount_uid78_lzcShifterZ1_uid6_fxpToFPTest_q_1_q or maxCount_uid7_fxpToFPTest_q)
    begin
        unique case (vCountFinal_uid82_lzcShifterZ1_uid6_fxpToFPTest_s)
            1'b0 : vCountFinal_uid82_lzcShifterZ1_uid6_fxpToFPTest_q = redist2_vCount_uid78_lzcShifterZ1_uid6_fxpToFPTest_q_1_q;
            1'b1 : vCountFinal_uid82_lzcShifterZ1_uid6_fxpToFPTest_q = maxCount_uid7_fxpToFPTest_q;
            default : vCountFinal_uid82_lzcShifterZ1_uid6_fxpToFPTest_q = 6'b0;
        endcase
    end

    // msbIn_uid9_fxpToFPTest(CONSTANT,8)
    assign msbIn_uid9_fxpToFPTest_q = 8'b10011110;

    // expPreRnd_uid10_fxpToFPTest(SUB,9)@5
    assign expPreRnd_uid10_fxpToFPTest_a = {1'b0, msbIn_uid9_fxpToFPTest_q};
    assign expPreRnd_uid10_fxpToFPTest_b = {3'b000, vCountFinal_uid82_lzcShifterZ1_uid6_fxpToFPTest_q};
    assign expPreRnd_uid10_fxpToFPTest_o = $unsigned(expPreRnd_uid10_fxpToFPTest_a) - $unsigned(expPreRnd_uid10_fxpToFPTest_b);
    assign expPreRnd_uid10_fxpToFPTest_q = expPreRnd_uid10_fxpToFPTest_o[8:0];

    // redist0_fracRnd_uid11_fxpToFPTest_merged_bit_select_b_1(DELAY,91)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist0_fracRnd_uid11_fxpToFPTest_merged_bit_select_b_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist0_fracRnd_uid11_fxpToFPTest_merged_bit_select_b_1_q <= fracRnd_uid11_fxpToFPTest_merged_bit_select_b;
        end
    end

    // expFracRnd_uid12_fxpToFPTest(BITJOIN,11)@5
    assign expFracRnd_uid12_fxpToFPTest_q = {expPreRnd_uid10_fxpToFPTest_q, redist0_fracRnd_uid11_fxpToFPTest_merged_bit_select_b_1_q};

    // l_uid13_fxpToFPTest_merged_bit_select(BITSELECT,84)@5
    assign l_uid13_fxpToFPTest_merged_bit_select_in = expFracRnd_uid12_fxpToFPTest_q[1:0];
    assign l_uid13_fxpToFPTest_merged_bit_select_b = l_uid13_fxpToFPTest_merged_bit_select_in[1:1];
    assign l_uid13_fxpToFPTest_merged_bit_select_c = l_uid13_fxpToFPTest_merged_bit_select_in[0:0];

    // rnd_uid18_fxpToFPTest(LOGICAL,17)@5 + 1
    assign rnd_uid18_fxpToFPTest_qi = l_uid13_fxpToFPTest_merged_bit_select_b | nr_uid17_fxpToFPTest_q | sticky_uid16_fxpToFPTest_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    rnd_uid18_fxpToFPTest_delay ( .xin(rnd_uid18_fxpToFPTest_qi), .xout(rnd_uid18_fxpToFPTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist9_expFracRnd_uid12_fxpToFPTest_q_1(DELAY,100)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist9_expFracRnd_uid12_fxpToFPTest_q_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist9_expFracRnd_uid12_fxpToFPTest_q_1_q <= expFracRnd_uid12_fxpToFPTest_q;
        end
    end

    // expFracR_uid20_fxpToFPTest(ADD,19)@6
    assign expFracR_uid20_fxpToFPTest_a = {{2{redist9_expFracRnd_uid12_fxpToFPTest_q_1_q[32]}}, redist9_expFracRnd_uid12_fxpToFPTest_q_1_q};
    assign expFracR_uid20_fxpToFPTest_b = {34'b0000000000000000000000000000000000, rnd_uid18_fxpToFPTest_q};
    assign expFracR_uid20_fxpToFPTest_o = $signed(expFracR_uid20_fxpToFPTest_a) + $signed(expFracR_uid20_fxpToFPTest_b);
    assign expFracR_uid20_fxpToFPTest_q = expFracR_uid20_fxpToFPTest_o[33:0];

    // expR_uid22_fxpToFPTest(BITSELECT,21)@6
    assign expR_uid22_fxpToFPTest_b = expFracR_uid20_fxpToFPTest_q[33:24];

    // redist7_expR_uid22_fxpToFPTest_b_1(DELAY,98)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist7_expR_uid22_fxpToFPTest_b_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist7_expR_uid22_fxpToFPTest_b_1_q <= expR_uid22_fxpToFPTest_b;
        end
    end

    // expR_uid34_fxpToFPTest(BITSELECT,33)@7
    assign expR_uid34_fxpToFPTest_in = redist7_expR_uid22_fxpToFPTest_b_1_q[7:0];
    assign expR_uid34_fxpToFPTest_b = expR_uid34_fxpToFPTest_in[7:0];

    // ovf_uid25_fxpToFPTest(COMPARE,24)@7
    assign ovf_uid25_fxpToFPTest_a = {{2{redist7_expR_uid22_fxpToFPTest_b_1_q[9]}}, redist7_expR_uid22_fxpToFPTest_b_1_q};
    assign ovf_uid25_fxpToFPTest_b = {4'b0000, expInf_uid24_fxpToFPTest_q};
    assign ovf_uid25_fxpToFPTest_o = $signed(ovf_uid25_fxpToFPTest_a) - $signed(ovf_uid25_fxpToFPTest_b);
    assign ovf_uid25_fxpToFPTest_n[0] = ~ (ovf_uid25_fxpToFPTest_o[11]);

    // inIsZero_uid8_fxpToFPTest(LOGICAL,7)@5 + 1
    assign inIsZero_uid8_fxpToFPTest_qi = vCountFinal_uid82_lzcShifterZ1_uid6_fxpToFPTest_q == maxCount_uid7_fxpToFPTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    inIsZero_uid8_fxpToFPTest_delay ( .xin(inIsZero_uid8_fxpToFPTest_qi), .xout(inIsZero_uid8_fxpToFPTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist10_inIsZero_uid8_fxpToFPTest_q_2(DELAY,101)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist10_inIsZero_uid8_fxpToFPTest_q_2_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist10_inIsZero_uid8_fxpToFPTest_q_2_q <= inIsZero_uid8_fxpToFPTest_q;
        end
    end

    // udf_uid23_fxpToFPTest(COMPARE,22)@7
    assign udf_uid23_fxpToFPTest_a = {11'b00000000000, GND_q};
    assign udf_uid23_fxpToFPTest_b = {{2{redist7_expR_uid22_fxpToFPTest_b_1_q[9]}}, redist7_expR_uid22_fxpToFPTest_b_1_q};
    assign udf_uid23_fxpToFPTest_o = $signed(udf_uid23_fxpToFPTest_a) - $signed(udf_uid23_fxpToFPTest_b);
    assign udf_uid23_fxpToFPTest_n[0] = ~ (udf_uid23_fxpToFPTest_o[11]);

    // udfOrInZero_uid29_fxpToFPTest(LOGICAL,28)@7
    assign udfOrInZero_uid29_fxpToFPTest_q = udf_uid23_fxpToFPTest_n | redist10_inIsZero_uid8_fxpToFPTest_q_2_q;

    // excSelector_uid30_fxpToFPTest(BITJOIN,29)@7
    assign excSelector_uid30_fxpToFPTest_q = {ovf_uid25_fxpToFPTest_n, udfOrInZero_uid29_fxpToFPTest_q};

    // expRPostExc_uid35_fxpToFPTest(MUX,34)@7
    assign expRPostExc_uid35_fxpToFPTest_s = excSelector_uid30_fxpToFPTest_q;
    always @(expRPostExc_uid35_fxpToFPTest_s or en or expR_uid34_fxpToFPTest_b or expZ_uid33_fxpToFPTest_q or expInf_uid24_fxpToFPTest_q)
    begin
        unique case (expRPostExc_uid35_fxpToFPTest_s)
            2'b00 : expRPostExc_uid35_fxpToFPTest_q = expR_uid34_fxpToFPTest_b;
            2'b01 : expRPostExc_uid35_fxpToFPTest_q = expZ_uid33_fxpToFPTest_q;
            2'b10 : expRPostExc_uid35_fxpToFPTest_q = expInf_uid24_fxpToFPTest_q;
            2'b11 : expRPostExc_uid35_fxpToFPTest_q = expInf_uid24_fxpToFPTest_q;
            default : expRPostExc_uid35_fxpToFPTest_q = 8'b0;
        endcase
    end

    // fracZ_uid27_fxpToFPTest(CONSTANT,26)
    assign fracZ_uid27_fxpToFPTest_q = 23'b00000000000000000000000;

    // fracR_uid21_fxpToFPTest(BITSELECT,20)@6
    assign fracR_uid21_fxpToFPTest_in = expFracR_uid20_fxpToFPTest_q[23:0];
    assign fracR_uid21_fxpToFPTest_b = fracR_uid21_fxpToFPTest_in[23:1];

    // redist8_fracR_uid21_fxpToFPTest_b_1(DELAY,99)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist8_fracR_uid21_fxpToFPTest_b_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist8_fracR_uid21_fxpToFPTest_b_1_q <= fracR_uid21_fxpToFPTest_b;
        end
    end

    // excSelector_uid26_fxpToFPTest(LOGICAL,25)@7
    assign excSelector_uid26_fxpToFPTest_q = redist10_inIsZero_uid8_fxpToFPTest_q_2_q | ovf_uid25_fxpToFPTest_n | udf_uid23_fxpToFPTest_n;

    // fracRPostExc_uid28_fxpToFPTest(MUX,27)@7
    assign fracRPostExc_uid28_fxpToFPTest_s = excSelector_uid26_fxpToFPTest_q;
    always @(fracRPostExc_uid28_fxpToFPTest_s or en or redist8_fracR_uid21_fxpToFPTest_b_1_q or fracZ_uid27_fxpToFPTest_q)
    begin
        unique case (fracRPostExc_uid28_fxpToFPTest_s)
            1'b0 : fracRPostExc_uid28_fxpToFPTest_q = redist8_fracR_uid21_fxpToFPTest_b_1_q;
            1'b1 : fracRPostExc_uid28_fxpToFPTest_q = fracZ_uid27_fxpToFPTest_q;
            default : fracRPostExc_uid28_fxpToFPTest_q = 23'b0;
        endcase
    end

    // outRes_uid36_fxpToFPTest(BITJOIN,35)@7
    assign outRes_uid36_fxpToFPTest_q = {GND_q, expRPostExc_uid35_fxpToFPTest_q, fracRPostExc_uid28_fxpToFPTest_q};

    // xOut(GPOUT,4)@7
    assign q = outRes_uid36_fxpToFPTest_q;

endmodule
