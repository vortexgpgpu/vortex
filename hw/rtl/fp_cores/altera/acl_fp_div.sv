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

// SystemVerilog created from acl_fp_div
// SystemVerilog created on Wed Aug  5 12:58:14 2020


(* altera_attribute = "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007; -name MESSAGE_DISABLE 10958" *)
module acl_fp_div (
    input wire [31:0] a,
    input wire [31:0] b,
    input wire [0:0] en,
    output wire [31:0] q,
    input wire clk,
    input wire areset
    );

    wire [0:0] GND_q;
    wire [0:0] VCC_q;
    wire [7:0] cstBiasM1_uid6_fpDivTest_q;
    wire [7:0] expX_uid9_fpDivTest_b;
    wire [22:0] fracX_uid10_fpDivTest_b;
    wire [0:0] signX_uid11_fpDivTest_b;
    wire [7:0] expY_uid12_fpDivTest_b;
    wire [22:0] fracY_uid13_fpDivTest_b;
    wire [0:0] signY_uid14_fpDivTest_b;
    wire [22:0] paddingY_uid15_fpDivTest_q;
    wire [23:0] updatedY_uid16_fpDivTest_q;
    wire [23:0] fracYZero_uid15_fpDivTest_a;
    wire [0:0] fracYZero_uid15_fpDivTest_qi;
    reg [0:0] fracYZero_uid15_fpDivTest_q;
    wire [7:0] cstAllOWE_uid18_fpDivTest_q;
    wire [7:0] cstAllZWE_uid20_fpDivTest_q;
    wire [0:0] excZ_x_uid23_fpDivTest_qi;
    reg [0:0] excZ_x_uid23_fpDivTest_q;
    wire [0:0] expXIsMax_uid24_fpDivTest_qi;
    reg [0:0] expXIsMax_uid24_fpDivTest_q;
    wire [0:0] fracXIsZero_uid25_fpDivTest_qi;
    reg [0:0] fracXIsZero_uid25_fpDivTest_q;
    wire [0:0] fracXIsNotZero_uid26_fpDivTest_q;
    wire [0:0] excI_x_uid27_fpDivTest_q;
    wire [0:0] excN_x_uid28_fpDivTest_q;
    wire [0:0] invExpXIsMax_uid29_fpDivTest_q;
    wire [0:0] InvExpXIsZero_uid30_fpDivTest_q;
    wire [0:0] excR_x_uid31_fpDivTest_q;
    wire [0:0] excZ_y_uid37_fpDivTest_qi;
    reg [0:0] excZ_y_uid37_fpDivTest_q;
    wire [0:0] expXIsMax_uid38_fpDivTest_qi;
    reg [0:0] expXIsMax_uid38_fpDivTest_q;
    wire [0:0] fracXIsZero_uid39_fpDivTest_qi;
    reg [0:0] fracXIsZero_uid39_fpDivTest_q;
    wire [0:0] fracXIsNotZero_uid40_fpDivTest_q;
    wire [0:0] excI_y_uid41_fpDivTest_q;
    wire [0:0] excN_y_uid42_fpDivTest_q;
    wire [0:0] invExpXIsMax_uid43_fpDivTest_q;
    wire [0:0] InvExpXIsZero_uid44_fpDivTest_q;
    wire [0:0] excR_y_uid45_fpDivTest_q;
    wire [0:0] signR_uid46_fpDivTest_qi;
    reg [0:0] signR_uid46_fpDivTest_q;
    wire [8:0] expXmY_uid47_fpDivTest_a;
    wire [8:0] expXmY_uid47_fpDivTest_b;
    logic [8:0] expXmY_uid47_fpDivTest_o;
    wire [8:0] expXmY_uid47_fpDivTest_q;
    wire [10:0] expR_uid48_fpDivTest_a;
    wire [10:0] expR_uid48_fpDivTest_b;
    logic [10:0] expR_uid48_fpDivTest_o;
    wire [9:0] expR_uid48_fpDivTest_q;
    wire [8:0] yAddr_uid51_fpDivTest_b;
    wire [13:0] yPE_uid52_fpDivTest_b;
    wire [0:0] fracYPostZ_uid56_fpDivTest_qi;
    reg [0:0] fracYPostZ_uid56_fpDivTest_q;
    wire [23:0] lOAdded_uid58_fpDivTest_q;
    wire [1:0] oFracXSE_bottomExtension_uid61_fpDivTest_q;
    wire [25:0] oFracXSE_mergedSignalTM_uid63_fpDivTest_q;
    wire [0:0] divValPreNormTrunc_uid66_fpDivTest_s;
    reg [25:0] divValPreNormTrunc_uid66_fpDivTest_q;
    wire [0:0] norm_uid67_fpDivTest_b;
    wire [24:0] divValPreNormHigh_uid68_fpDivTest_in;
    wire [23:0] divValPreNormHigh_uid68_fpDivTest_b;
    wire [23:0] divValPreNormLow_uid69_fpDivTest_in;
    wire [23:0] divValPreNormLow_uid69_fpDivTest_b;
    wire [0:0] normFracRnd_uid70_fpDivTest_s;
    reg [23:0] normFracRnd_uid70_fpDivTest_q;
    wire [33:0] expFracRnd_uid71_fpDivTest_q;
    wire [24:0] rndOp_uid75_fpDivTest_q;
    wire [35:0] expFracPostRnd_uid76_fpDivTest_a;
    wire [35:0] expFracPostRnd_uid76_fpDivTest_b;
    logic [35:0] expFracPostRnd_uid76_fpDivTest_o;
    wire [34:0] expFracPostRnd_uid76_fpDivTest_q;
    wire [23:0] fracRPreExc_uid78_fpDivTest_in;
    wire [22:0] fracRPreExc_uid78_fpDivTest_b;
    wire [31:0] excRPreExc_uid79_fpDivTest_in;
    wire [7:0] excRPreExc_uid79_fpDivTest_b;
    wire [10:0] expRExt_uid80_fpDivTest_b;
    wire [12:0] expUdf_uid81_fpDivTest_a;
    wire [12:0] expUdf_uid81_fpDivTest_b;
    logic [12:0] expUdf_uid81_fpDivTest_o;
    wire [0:0] expUdf_uid81_fpDivTest_n;
    wire [12:0] expOvf_uid84_fpDivTest_a;
    wire [12:0] expOvf_uid84_fpDivTest_b;
    logic [12:0] expOvf_uid84_fpDivTest_o;
    wire [0:0] expOvf_uid84_fpDivTest_n;
    wire [0:0] zeroOverReg_uid85_fpDivTest_q;
    wire [0:0] regOverRegWithUf_uid86_fpDivTest_q;
    wire [0:0] xRegOrZero_uid87_fpDivTest_q;
    wire [0:0] regOrZeroOverInf_uid88_fpDivTest_q;
    wire [0:0] excRZero_uid89_fpDivTest_qi;
    reg [0:0] excRZero_uid89_fpDivTest_q;
    wire [0:0] excXRYZ_uid90_fpDivTest_q;
    wire [0:0] excXRYROvf_uid91_fpDivTest_q;
    wire [0:0] excXIYZ_uid92_fpDivTest_q;
    wire [0:0] excXIYR_uid93_fpDivTest_q;
    wire [0:0] excRInf_uid94_fpDivTest_qi;
    reg [0:0] excRInf_uid94_fpDivTest_q;
    wire [0:0] excXZYZ_uid95_fpDivTest_q;
    wire [0:0] excXIYI_uid96_fpDivTest_q;
    wire [0:0] excRNaN_uid97_fpDivTest_qi;
    reg [0:0] excRNaN_uid97_fpDivTest_q;
    wire [2:0] concExc_uid98_fpDivTest_q;
    reg [1:0] excREnc_uid99_fpDivTest_q;
    wire [22:0] oneFracRPostExc2_uid100_fpDivTest_q;
    wire [1:0] fracRPostExc_uid103_fpDivTest_s;
    reg [22:0] fracRPostExc_uid103_fpDivTest_q;
    wire [1:0] expRPostExc_uid107_fpDivTest_s;
    reg [7:0] expRPostExc_uid107_fpDivTest_q;
    wire [0:0] invExcRNaN_uid108_fpDivTest_q;
    wire [0:0] sRPostExc_uid109_fpDivTest_q;
    wire [31:0] divR_uid110_fpDivTest_q;
    wire [11:0] yT1_uid124_invPolyEval_b;
    wire [0:0] lowRangeB_uid126_invPolyEval_in;
    wire [0:0] lowRangeB_uid126_invPolyEval_b;
    wire [11:0] highBBits_uid127_invPolyEval_b;
    wire [21:0] s1sumAHighB_uid128_invPolyEval_a;
    wire [21:0] s1sumAHighB_uid128_invPolyEval_b;
    logic [21:0] s1sumAHighB_uid128_invPolyEval_o;
    wire [21:0] s1sumAHighB_uid128_invPolyEval_q;
    wire [22:0] s1_uid129_invPolyEval_q;
    wire [1:0] lowRangeB_uid132_invPolyEval_in;
    wire [1:0] lowRangeB_uid132_invPolyEval_b;
    wire [21:0] highBBits_uid133_invPolyEval_b;
    wire [31:0] s2sumAHighB_uid134_invPolyEval_a;
    wire [31:0] s2sumAHighB_uid134_invPolyEval_b;
    logic [31:0] s2sumAHighB_uid134_invPolyEval_o;
    wire [31:0] s2sumAHighB_uid134_invPolyEval_q;
    wire [33:0] s2_uid135_invPolyEval_q;
    wire [25:0] osig_uid138_prodDivPreNormProd_uid60_fpDivTest_b;
    wire [12:0] osig_uid141_pT1_uid125_invPolyEval_b;
    wire [23:0] osig_uid144_pT2_uid131_invPolyEval_b;
    wire memoryC0_uid112_invTables_lutmem_reset0;
    wire [30:0] memoryC0_uid112_invTables_lutmem_ia;
    wire [8:0] memoryC0_uid112_invTables_lutmem_aa;
    wire [8:0] memoryC0_uid112_invTables_lutmem_ab;
    wire [30:0] memoryC0_uid112_invTables_lutmem_ir;
    wire [30:0] memoryC0_uid112_invTables_lutmem_r;
    wire memoryC1_uid115_invTables_lutmem_reset0;
    wire [20:0] memoryC1_uid115_invTables_lutmem_ia;
    wire [8:0] memoryC1_uid115_invTables_lutmem_aa;
    wire [8:0] memoryC1_uid115_invTables_lutmem_ab;
    wire [20:0] memoryC1_uid115_invTables_lutmem_ir;
    wire [20:0] memoryC1_uid115_invTables_lutmem_r;
    wire memoryC2_uid118_invTables_lutmem_reset0;
    wire [11:0] memoryC2_uid118_invTables_lutmem_ia;
    wire [8:0] memoryC2_uid118_invTables_lutmem_aa;
    wire [8:0] memoryC2_uid118_invTables_lutmem_ab;
    wire [11:0] memoryC2_uid118_invTables_lutmem_ir;
    wire [11:0] memoryC2_uid118_invTables_lutmem_r;
    wire prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_reset;
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [25:0] prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_a0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [25:0] prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_a1 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [23:0] prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_c0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [23:0] prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_c1 [0:0];
    wire [49:0] prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_p [0:0];
    wire [49:0] prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_u [0:0];
    wire [49:0] prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_w [0:0];
    wire [49:0] prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_x [0:0];
    wire [49:0] prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_y [0:0];
    reg [49:0] prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_s [0:0];
    wire [49:0] prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_qq;
    wire [49:0] prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_q;
    wire prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_ena0;
    wire prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_ena1;
    wire prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_ena2;
    wire prodXY_uid140_pT1_uid125_invPolyEval_cma_reset;
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [11:0] prodXY_uid140_pT1_uid125_invPolyEval_cma_a0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [11:0] prodXY_uid140_pT1_uid125_invPolyEval_cma_a1 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg signed [11:0] prodXY_uid140_pT1_uid125_invPolyEval_cma_c0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg signed [11:0] prodXY_uid140_pT1_uid125_invPolyEval_cma_c1 [0:0];
    wire signed [12:0] prodXY_uid140_pT1_uid125_invPolyEval_cma_l [0:0];
    wire signed [24:0] prodXY_uid140_pT1_uid125_invPolyEval_cma_p [0:0];
    wire signed [24:0] prodXY_uid140_pT1_uid125_invPolyEval_cma_u [0:0];
    wire signed [24:0] prodXY_uid140_pT1_uid125_invPolyEval_cma_w [0:0];
    wire signed [24:0] prodXY_uid140_pT1_uid125_invPolyEval_cma_x [0:0];
    wire signed [24:0] prodXY_uid140_pT1_uid125_invPolyEval_cma_y [0:0];
    reg signed [24:0] prodXY_uid140_pT1_uid125_invPolyEval_cma_s [0:0];
    wire [23:0] prodXY_uid140_pT1_uid125_invPolyEval_cma_qq;
    wire [23:0] prodXY_uid140_pT1_uid125_invPolyEval_cma_q;
    wire prodXY_uid140_pT1_uid125_invPolyEval_cma_ena0;
    wire prodXY_uid140_pT1_uid125_invPolyEval_cma_ena1;
    wire prodXY_uid140_pT1_uid125_invPolyEval_cma_ena2;
    wire prodXY_uid143_pT2_uid131_invPolyEval_cma_reset;
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [13:0] prodXY_uid143_pT2_uid131_invPolyEval_cma_a0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [13:0] prodXY_uid143_pT2_uid131_invPolyEval_cma_a1 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg signed [22:0] prodXY_uid143_pT2_uid131_invPolyEval_cma_c0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg signed [22:0] prodXY_uid143_pT2_uid131_invPolyEval_cma_c1 [0:0];
    wire signed [14:0] prodXY_uid143_pT2_uid131_invPolyEval_cma_l [0:0];
    wire signed [37:0] prodXY_uid143_pT2_uid131_invPolyEval_cma_p [0:0];
    wire signed [37:0] prodXY_uid143_pT2_uid131_invPolyEval_cma_u [0:0];
    wire signed [37:0] prodXY_uid143_pT2_uid131_invPolyEval_cma_w [0:0];
    wire signed [37:0] prodXY_uid143_pT2_uid131_invPolyEval_cma_x [0:0];
    wire signed [37:0] prodXY_uid143_pT2_uid131_invPolyEval_cma_y [0:0];
    reg signed [37:0] prodXY_uid143_pT2_uid131_invPolyEval_cma_s [0:0];
    wire [36:0] prodXY_uid143_pT2_uid131_invPolyEval_cma_qq;
    wire [36:0] prodXY_uid143_pT2_uid131_invPolyEval_cma_q;
    wire prodXY_uid143_pT2_uid131_invPolyEval_cma_ena0;
    wire prodXY_uid143_pT2_uid131_invPolyEval_cma_ena1;
    wire prodXY_uid143_pT2_uid131_invPolyEval_cma_ena2;
    wire [31:0] invY_uid54_fpDivTest_merged_bit_select_in;
    wire [25:0] invY_uid54_fpDivTest_merged_bit_select_b;
    wire [0:0] invY_uid54_fpDivTest_merged_bit_select_c;
    reg [25:0] redist0_invY_uid54_fpDivTest_merged_bit_select_b_1_q;
    reg [0:0] redist1_lowRangeB_uid126_invPolyEval_b_1_q;
    reg [7:0] redist2_excRPreExc_uid79_fpDivTest_b_1_q;
    reg [22:0] redist3_fracRPreExc_uid78_fpDivTest_b_1_q;
    reg [0:0] redist4_norm_uid67_fpDivTest_b_1_q;
    reg [23:0] redist5_lOAdded_uid58_fpDivTest_q_3_q;
    reg [0:0] redist6_fracYPostZ_uid56_fpDivTest_q_4_q;
    reg [13:0] redist7_yPE_uid52_fpDivTest_b_2_q;
    reg [8:0] redist9_yAddr_uid51_fpDivTest_b_3_q;
    reg [8:0] redist10_yAddr_uid51_fpDivTest_b_7_q;
    reg [0:0] redist12_signR_uid46_fpDivTest_q_16_q;
    reg [0:0] redist13_fracXIsZero_uid39_fpDivTest_q_15_q;
    reg [0:0] redist14_expXIsMax_uid38_fpDivTest_q_15_q;
    reg [0:0] redist15_excZ_y_uid37_fpDivTest_q_15_q;
    reg [0:0] redist16_fracXIsZero_uid25_fpDivTest_q_5_q;
    reg [0:0] redist17_expXIsMax_uid24_fpDivTest_q_15_q;
    reg [0:0] redist18_excZ_x_uid23_fpDivTest_q_15_q;
    reg [0:0] redist19_fracYZero_uid15_fpDivTest_q_9_q;
    wire redist8_yPE_uid52_fpDivTest_b_6_mem_reset0;
    wire [13:0] redist8_yPE_uid52_fpDivTest_b_6_mem_ia;
    wire [1:0] redist8_yPE_uid52_fpDivTest_b_6_mem_aa;
    wire [1:0] redist8_yPE_uid52_fpDivTest_b_6_mem_ab;
    wire [13:0] redist8_yPE_uid52_fpDivTest_b_6_mem_iq;
    wire [13:0] redist8_yPE_uid52_fpDivTest_b_6_mem_q;
    wire [1:0] redist8_yPE_uid52_fpDivTest_b_6_rdcnt_q;
    (* preserve *) reg [1:0] redist8_yPE_uid52_fpDivTest_b_6_rdcnt_i;
    (* preserve *) reg redist8_yPE_uid52_fpDivTest_b_6_rdcnt_eq;
    wire [0:0] redist8_yPE_uid52_fpDivTest_b_6_rdmux_s;
    reg [1:0] redist8_yPE_uid52_fpDivTest_b_6_rdmux_q;
    reg [1:0] redist8_yPE_uid52_fpDivTest_b_6_wraddr_q;
    wire [1:0] redist8_yPE_uid52_fpDivTest_b_6_mem_last_q;
    wire [0:0] redist8_yPE_uid52_fpDivTest_b_6_cmp_q;
    reg [0:0] redist8_yPE_uid52_fpDivTest_b_6_cmpReg_q;
    wire [0:0] redist8_yPE_uid52_fpDivTest_b_6_notEnable_q;
    wire [0:0] redist8_yPE_uid52_fpDivTest_b_6_nor_q;
    (* preserve_syn_only *) reg [0:0] redist8_yPE_uid52_fpDivTest_b_6_sticky_ena_q;
    wire [0:0] redist8_yPE_uid52_fpDivTest_b_6_enaAnd_q;
    reg [8:0] redist11_expXmY_uid47_fpDivTest_q_13_outputreg_q;
    wire redist11_expXmY_uid47_fpDivTest_q_13_mem_reset0;
    wire [8:0] redist11_expXmY_uid47_fpDivTest_q_13_mem_ia;
    wire [3:0] redist11_expXmY_uid47_fpDivTest_q_13_mem_aa;
    wire [3:0] redist11_expXmY_uid47_fpDivTest_q_13_mem_ab;
    wire [8:0] redist11_expXmY_uid47_fpDivTest_q_13_mem_iq;
    wire [8:0] redist11_expXmY_uid47_fpDivTest_q_13_mem_q;
    wire [3:0] redist11_expXmY_uid47_fpDivTest_q_13_rdcnt_q;
    (* preserve *) reg [3:0] redist11_expXmY_uid47_fpDivTest_q_13_rdcnt_i;
    (* preserve *) reg redist11_expXmY_uid47_fpDivTest_q_13_rdcnt_eq;
    wire [0:0] redist11_expXmY_uid47_fpDivTest_q_13_rdmux_s;
    reg [3:0] redist11_expXmY_uid47_fpDivTest_q_13_rdmux_q;
    reg [3:0] redist11_expXmY_uid47_fpDivTest_q_13_wraddr_q;
    wire [4:0] redist11_expXmY_uid47_fpDivTest_q_13_mem_last_q;
    wire [4:0] redist11_expXmY_uid47_fpDivTest_q_13_cmp_b;
    wire [0:0] redist11_expXmY_uid47_fpDivTest_q_13_cmp_q;
    reg [0:0] redist11_expXmY_uid47_fpDivTest_q_13_cmpReg_q;
    wire [0:0] redist11_expXmY_uid47_fpDivTest_q_13_notEnable_q;
    wire [0:0] redist11_expXmY_uid47_fpDivTest_q_13_nor_q;
    (* preserve_syn_only *) reg [0:0] redist11_expXmY_uid47_fpDivTest_q_13_sticky_ena_q;
    wire [0:0] redist11_expXmY_uid47_fpDivTest_q_13_enaAnd_q;
    reg [22:0] redist20_fracX_uid10_fpDivTest_b_10_outputreg_q;
    wire redist20_fracX_uid10_fpDivTest_b_10_mem_reset0;
    wire [22:0] redist20_fracX_uid10_fpDivTest_b_10_mem_ia;
    wire [2:0] redist20_fracX_uid10_fpDivTest_b_10_mem_aa;
    wire [2:0] redist20_fracX_uid10_fpDivTest_b_10_mem_ab;
    wire [22:0] redist20_fracX_uid10_fpDivTest_b_10_mem_iq;
    wire [22:0] redist20_fracX_uid10_fpDivTest_b_10_mem_q;
    wire [2:0] redist20_fracX_uid10_fpDivTest_b_10_rdcnt_q;
    (* preserve *) reg [2:0] redist20_fracX_uid10_fpDivTest_b_10_rdcnt_i;
    wire [0:0] redist20_fracX_uid10_fpDivTest_b_10_rdmux_s;
    reg [2:0] redist20_fracX_uid10_fpDivTest_b_10_rdmux_q;
    reg [2:0] redist20_fracX_uid10_fpDivTest_b_10_wraddr_q;
    wire [3:0] redist20_fracX_uid10_fpDivTest_b_10_mem_last_q;
    wire [3:0] redist20_fracX_uid10_fpDivTest_b_10_cmp_b;
    wire [0:0] redist20_fracX_uid10_fpDivTest_b_10_cmp_q;
    reg [0:0] redist20_fracX_uid10_fpDivTest_b_10_cmpReg_q;
    wire [0:0] redist20_fracX_uid10_fpDivTest_b_10_notEnable_q;
    wire [0:0] redist20_fracX_uid10_fpDivTest_b_10_nor_q;
    (* preserve_syn_only *) reg [0:0] redist20_fracX_uid10_fpDivTest_b_10_sticky_ena_q;
    wire [0:0] redist20_fracX_uid10_fpDivTest_b_10_enaAnd_q;


    // fracY_uid13_fpDivTest(BITSELECT,12)@0
    assign fracY_uid13_fpDivTest_b = b[22:0];

    // paddingY_uid15_fpDivTest(CONSTANT,14)
    assign paddingY_uid15_fpDivTest_q = 23'b00000000000000000000000;

    // fracXIsZero_uid39_fpDivTest(LOGICAL,38)@0 + 1
    assign fracXIsZero_uid39_fpDivTest_qi = paddingY_uid15_fpDivTest_q == fracY_uid13_fpDivTest_b ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    fracXIsZero_uid39_fpDivTest_delay ( .xin(fracXIsZero_uid39_fpDivTest_qi), .xout(fracXIsZero_uid39_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist13_fracXIsZero_uid39_fpDivTest_q_15(DELAY,165)
    dspba_delay_ver #( .width(1), .depth(14), .reset_kind("ASYNC") )
    redist13_fracXIsZero_uid39_fpDivTest_q_15 ( .xin(fracXIsZero_uid39_fpDivTest_q), .xout(redist13_fracXIsZero_uid39_fpDivTest_q_15_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // cstAllOWE_uid18_fpDivTest(CONSTANT,17)
    assign cstAllOWE_uid18_fpDivTest_q = 8'b11111111;

    // expY_uid12_fpDivTest(BITSELECT,11)@0
    assign expY_uid12_fpDivTest_b = b[30:23];

    // expXIsMax_uid38_fpDivTest(LOGICAL,37)@0 + 1
    assign expXIsMax_uid38_fpDivTest_qi = expY_uid12_fpDivTest_b == cstAllOWE_uid18_fpDivTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    expXIsMax_uid38_fpDivTest_delay ( .xin(expXIsMax_uid38_fpDivTest_qi), .xout(expXIsMax_uid38_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist14_expXIsMax_uid38_fpDivTest_q_15(DELAY,166)
    dspba_delay_ver #( .width(1), .depth(14), .reset_kind("ASYNC") )
    redist14_expXIsMax_uid38_fpDivTest_q_15 ( .xin(expXIsMax_uid38_fpDivTest_q), .xout(redist14_expXIsMax_uid38_fpDivTest_q_15_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // excI_y_uid41_fpDivTest(LOGICAL,40)@15
    assign excI_y_uid41_fpDivTest_q = redist14_expXIsMax_uid38_fpDivTest_q_15_q & redist13_fracXIsZero_uid39_fpDivTest_q_15_q;

    // redist20_fracX_uid10_fpDivTest_b_10_notEnable(LOGICAL,204)
    assign redist20_fracX_uid10_fpDivTest_b_10_notEnable_q = ~ (en);

    // redist20_fracX_uid10_fpDivTest_b_10_nor(LOGICAL,205)
    assign redist20_fracX_uid10_fpDivTest_b_10_nor_q = ~ (redist20_fracX_uid10_fpDivTest_b_10_notEnable_q | redist20_fracX_uid10_fpDivTest_b_10_sticky_ena_q);

    // redist20_fracX_uid10_fpDivTest_b_10_mem_last(CONSTANT,201)
    assign redist20_fracX_uid10_fpDivTest_b_10_mem_last_q = 4'b0110;

    // redist20_fracX_uid10_fpDivTest_b_10_cmp(LOGICAL,202)
    assign redist20_fracX_uid10_fpDivTest_b_10_cmp_b = {1'b0, redist20_fracX_uid10_fpDivTest_b_10_rdmux_q};
    assign redist20_fracX_uid10_fpDivTest_b_10_cmp_q = redist20_fracX_uid10_fpDivTest_b_10_mem_last_q == redist20_fracX_uid10_fpDivTest_b_10_cmp_b ? 1'b1 : 1'b0;

    // redist20_fracX_uid10_fpDivTest_b_10_cmpReg(REG,203)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist20_fracX_uid10_fpDivTest_b_10_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist20_fracX_uid10_fpDivTest_b_10_cmpReg_q <= redist20_fracX_uid10_fpDivTest_b_10_cmp_q;
        end
    end

    // redist20_fracX_uid10_fpDivTest_b_10_sticky_ena(REG,206)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist20_fracX_uid10_fpDivTest_b_10_sticky_ena_q <= 1'b0;
        end
        else if (redist20_fracX_uid10_fpDivTest_b_10_nor_q == 1'b1)
        begin
            redist20_fracX_uid10_fpDivTest_b_10_sticky_ena_q <= redist20_fracX_uid10_fpDivTest_b_10_cmpReg_q;
        end
    end

    // redist20_fracX_uid10_fpDivTest_b_10_enaAnd(LOGICAL,207)
    assign redist20_fracX_uid10_fpDivTest_b_10_enaAnd_q = redist20_fracX_uid10_fpDivTest_b_10_sticky_ena_q & en;

    // redist20_fracX_uid10_fpDivTest_b_10_rdcnt(COUNTER,198)
    // low=0, high=7, step=1, init=0
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist20_fracX_uid10_fpDivTest_b_10_rdcnt_i <= 3'd0;
        end
        else if (en == 1'b1)
        begin
            redist20_fracX_uid10_fpDivTest_b_10_rdcnt_i <= $unsigned(redist20_fracX_uid10_fpDivTest_b_10_rdcnt_i) + $unsigned(3'd1);
        end
    end
    assign redist20_fracX_uid10_fpDivTest_b_10_rdcnt_q = redist20_fracX_uid10_fpDivTest_b_10_rdcnt_i[2:0];

    // redist20_fracX_uid10_fpDivTest_b_10_rdmux(MUX,199)
    assign redist20_fracX_uid10_fpDivTest_b_10_rdmux_s = en;
    always @(redist20_fracX_uid10_fpDivTest_b_10_rdmux_s or redist20_fracX_uid10_fpDivTest_b_10_wraddr_q or redist20_fracX_uid10_fpDivTest_b_10_rdcnt_q)
    begin
        unique case (redist20_fracX_uid10_fpDivTest_b_10_rdmux_s)
            1'b0 : redist20_fracX_uid10_fpDivTest_b_10_rdmux_q = redist20_fracX_uid10_fpDivTest_b_10_wraddr_q;
            1'b1 : redist20_fracX_uid10_fpDivTest_b_10_rdmux_q = redist20_fracX_uid10_fpDivTest_b_10_rdcnt_q;
            default : redist20_fracX_uid10_fpDivTest_b_10_rdmux_q = 3'b0;
        endcase
    end

    // VCC(CONSTANT,1)
    assign VCC_q = 1'b1;

    // fracX_uid10_fpDivTest(BITSELECT,9)@0
    assign fracX_uid10_fpDivTest_b = a[22:0];

    // redist20_fracX_uid10_fpDivTest_b_10_wraddr(REG,200)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist20_fracX_uid10_fpDivTest_b_10_wraddr_q <= 3'b111;
        end
        else
        begin
            redist20_fracX_uid10_fpDivTest_b_10_wraddr_q <= redist20_fracX_uid10_fpDivTest_b_10_rdmux_q;
        end
    end

    // redist20_fracX_uid10_fpDivTest_b_10_mem(DUALMEM,197)
    assign redist20_fracX_uid10_fpDivTest_b_10_mem_ia = fracX_uid10_fpDivTest_b;
    assign redist20_fracX_uid10_fpDivTest_b_10_mem_aa = redist20_fracX_uid10_fpDivTest_b_10_wraddr_q;
    assign redist20_fracX_uid10_fpDivTest_b_10_mem_ab = redist20_fracX_uid10_fpDivTest_b_10_rdmux_q;
    assign redist20_fracX_uid10_fpDivTest_b_10_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(23),
        .widthad_a(3),
        .numwords_a(8),
        .width_b(23),
        .widthad_b(3),
        .numwords_b(8),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .rdcontrol_reg_b("CLOCK0"),
        .byteena_reg_b("CLOCK0"),
        .outdata_reg_b("CLOCK1"),
        .outdata_aclr_b("CLEAR1"),
        .clock_enable_input_a("NORMAL"),
        .clock_enable_input_b("NORMAL"),
        .clock_enable_output_b("NORMAL"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("TRUE"),
        .intended_device_family("Arria 10")
    ) redist20_fracX_uid10_fpDivTest_b_10_mem_dmem (
        .clocken1(redist20_fracX_uid10_fpDivTest_b_10_enaAnd_q[0]),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .aclr1(redist20_fracX_uid10_fpDivTest_b_10_mem_reset0),
        .clock1(clk),
        .address_a(redist20_fracX_uid10_fpDivTest_b_10_mem_aa),
        .data_a(redist20_fracX_uid10_fpDivTest_b_10_mem_ia),
        .wren_a(en[0]),
        .address_b(redist20_fracX_uid10_fpDivTest_b_10_mem_ab),
        .q_b(redist20_fracX_uid10_fpDivTest_b_10_mem_iq),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_b(),
        .clocken2(),
        .clocken3(),
        .aclr0(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .sclr(),
        .address2_a(),
        .address2_b(),
        .q_a(),
        .eccstatus()
    );
    assign redist20_fracX_uid10_fpDivTest_b_10_mem_q = redist20_fracX_uid10_fpDivTest_b_10_mem_iq[22:0];

    // redist20_fracX_uid10_fpDivTest_b_10_outputreg(DELAY,196)
    dspba_delay_ver #( .width(23), .depth(1), .reset_kind("ASYNC") )
    redist20_fracX_uid10_fpDivTest_b_10_outputreg ( .xin(redist20_fracX_uid10_fpDivTest_b_10_mem_q), .xout(redist20_fracX_uid10_fpDivTest_b_10_outputreg_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // fracXIsZero_uid25_fpDivTest(LOGICAL,24)@10 + 1
    assign fracXIsZero_uid25_fpDivTest_qi = paddingY_uid15_fpDivTest_q == redist20_fracX_uid10_fpDivTest_b_10_outputreg_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    fracXIsZero_uid25_fpDivTest_delay ( .xin(fracXIsZero_uid25_fpDivTest_qi), .xout(fracXIsZero_uid25_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist16_fracXIsZero_uid25_fpDivTest_q_5(DELAY,168)
    dspba_delay_ver #( .width(1), .depth(4), .reset_kind("ASYNC") )
    redist16_fracXIsZero_uid25_fpDivTest_q_5 ( .xin(fracXIsZero_uid25_fpDivTest_q), .xout(redist16_fracXIsZero_uid25_fpDivTest_q_5_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expX_uid9_fpDivTest(BITSELECT,8)@0
    assign expX_uid9_fpDivTest_b = a[30:23];

    // expXIsMax_uid24_fpDivTest(LOGICAL,23)@0 + 1
    assign expXIsMax_uid24_fpDivTest_qi = expX_uid9_fpDivTest_b == cstAllOWE_uid18_fpDivTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    expXIsMax_uid24_fpDivTest_delay ( .xin(expXIsMax_uid24_fpDivTest_qi), .xout(expXIsMax_uid24_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist17_expXIsMax_uid24_fpDivTest_q_15(DELAY,169)
    dspba_delay_ver #( .width(1), .depth(14), .reset_kind("ASYNC") )
    redist17_expXIsMax_uid24_fpDivTest_q_15 ( .xin(expXIsMax_uid24_fpDivTest_q), .xout(redist17_expXIsMax_uid24_fpDivTest_q_15_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // excI_x_uid27_fpDivTest(LOGICAL,26)@15
    assign excI_x_uid27_fpDivTest_q = redist17_expXIsMax_uid24_fpDivTest_q_15_q & redist16_fracXIsZero_uid25_fpDivTest_q_5_q;

    // excXIYI_uid96_fpDivTest(LOGICAL,95)@15
    assign excXIYI_uid96_fpDivTest_q = excI_x_uid27_fpDivTest_q & excI_y_uid41_fpDivTest_q;

    // fracXIsNotZero_uid40_fpDivTest(LOGICAL,39)@15
    assign fracXIsNotZero_uid40_fpDivTest_q = ~ (redist13_fracXIsZero_uid39_fpDivTest_q_15_q);

    // excN_y_uid42_fpDivTest(LOGICAL,41)@15
    assign excN_y_uid42_fpDivTest_q = redist14_expXIsMax_uid38_fpDivTest_q_15_q & fracXIsNotZero_uid40_fpDivTest_q;

    // fracXIsNotZero_uid26_fpDivTest(LOGICAL,25)@15
    assign fracXIsNotZero_uid26_fpDivTest_q = ~ (redist16_fracXIsZero_uid25_fpDivTest_q_5_q);

    // excN_x_uid28_fpDivTest(LOGICAL,27)@15
    assign excN_x_uid28_fpDivTest_q = redist17_expXIsMax_uid24_fpDivTest_q_15_q & fracXIsNotZero_uid26_fpDivTest_q;

    // cstAllZWE_uid20_fpDivTest(CONSTANT,19)
    assign cstAllZWE_uid20_fpDivTest_q = 8'b00000000;

    // excZ_y_uid37_fpDivTest(LOGICAL,36)@0 + 1
    assign excZ_y_uid37_fpDivTest_qi = expY_uid12_fpDivTest_b == cstAllZWE_uid20_fpDivTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    excZ_y_uid37_fpDivTest_delay ( .xin(excZ_y_uid37_fpDivTest_qi), .xout(excZ_y_uid37_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist15_excZ_y_uid37_fpDivTest_q_15(DELAY,167)
    dspba_delay_ver #( .width(1), .depth(14), .reset_kind("ASYNC") )
    redist15_excZ_y_uid37_fpDivTest_q_15 ( .xin(excZ_y_uid37_fpDivTest_q), .xout(redist15_excZ_y_uid37_fpDivTest_q_15_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // excZ_x_uid23_fpDivTest(LOGICAL,22)@0 + 1
    assign excZ_x_uid23_fpDivTest_qi = expX_uid9_fpDivTest_b == cstAllZWE_uid20_fpDivTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    excZ_x_uid23_fpDivTest_delay ( .xin(excZ_x_uid23_fpDivTest_qi), .xout(excZ_x_uid23_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist18_excZ_x_uid23_fpDivTest_q_15(DELAY,170)
    dspba_delay_ver #( .width(1), .depth(14), .reset_kind("ASYNC") )
    redist18_excZ_x_uid23_fpDivTest_q_15 ( .xin(excZ_x_uid23_fpDivTest_q), .xout(redist18_excZ_x_uid23_fpDivTest_q_15_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // excXZYZ_uid95_fpDivTest(LOGICAL,94)@15
    assign excXZYZ_uid95_fpDivTest_q = redist18_excZ_x_uid23_fpDivTest_q_15_q & redist15_excZ_y_uid37_fpDivTest_q_15_q;

    // excRNaN_uid97_fpDivTest(LOGICAL,96)@15 + 1
    assign excRNaN_uid97_fpDivTest_qi = excXZYZ_uid95_fpDivTest_q | excN_x_uid28_fpDivTest_q | excN_y_uid42_fpDivTest_q | excXIYI_uid96_fpDivTest_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    excRNaN_uid97_fpDivTest_delay ( .xin(excRNaN_uid97_fpDivTest_qi), .xout(excRNaN_uid97_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // invExcRNaN_uid108_fpDivTest(LOGICAL,107)@16
    assign invExcRNaN_uid108_fpDivTest_q = ~ (excRNaN_uid97_fpDivTest_q);

    // signY_uid14_fpDivTest(BITSELECT,13)@0
    assign signY_uid14_fpDivTest_b = b[31:31];

    // signX_uid11_fpDivTest(BITSELECT,10)@0
    assign signX_uid11_fpDivTest_b = a[31:31];

    // signR_uid46_fpDivTest(LOGICAL,45)@0 + 1
    assign signR_uid46_fpDivTest_qi = signX_uid11_fpDivTest_b ^ signY_uid14_fpDivTest_b;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    signR_uid46_fpDivTest_delay ( .xin(signR_uid46_fpDivTest_qi), .xout(signR_uid46_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist12_signR_uid46_fpDivTest_q_16(DELAY,164)
    dspba_delay_ver #( .width(1), .depth(15), .reset_kind("ASYNC") )
    redist12_signR_uid46_fpDivTest_q_16 ( .xin(signR_uid46_fpDivTest_q), .xout(redist12_signR_uid46_fpDivTest_q_16_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // sRPostExc_uid109_fpDivTest(LOGICAL,108)@16
    assign sRPostExc_uid109_fpDivTest_q = redist12_signR_uid46_fpDivTest_q_16_q & invExcRNaN_uid108_fpDivTest_q;

    // lOAdded_uid58_fpDivTest(BITJOIN,57)@10
    assign lOAdded_uid58_fpDivTest_q = {VCC_q, redist20_fracX_uid10_fpDivTest_b_10_outputreg_q};

    // redist5_lOAdded_uid58_fpDivTest_q_3(DELAY,157)
    dspba_delay_ver #( .width(24), .depth(3), .reset_kind("ASYNC") )
    redist5_lOAdded_uid58_fpDivTest_q_3 ( .xin(lOAdded_uid58_fpDivTest_q), .xout(redist5_lOAdded_uid58_fpDivTest_q_3_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // oFracXSE_bottomExtension_uid61_fpDivTest(CONSTANT,60)
    assign oFracXSE_bottomExtension_uid61_fpDivTest_q = 2'b00;

    // oFracXSE_mergedSignalTM_uid63_fpDivTest(BITJOIN,62)@13
    assign oFracXSE_mergedSignalTM_uid63_fpDivTest_q = {redist5_lOAdded_uid58_fpDivTest_q_3_q, oFracXSE_bottomExtension_uid61_fpDivTest_q};

    // yAddr_uid51_fpDivTest(BITSELECT,50)@0
    assign yAddr_uid51_fpDivTest_b = fracY_uid13_fpDivTest_b[22:14];

    // memoryC2_uid118_invTables_lutmem(DUALMEM,147)@0 + 2
    // in j@20000000
    assign memoryC2_uid118_invTables_lutmem_aa = yAddr_uid51_fpDivTest_b;
    assign memoryC2_uid118_invTables_lutmem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("M20K"),
        .operation_mode("ROM"),
        .width_a(12),
        .widthad_a(9),
        .numwords_a(512),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .outdata_reg_a("CLOCK0"),
        .outdata_aclr_a("CLEAR0"),
        .clock_enable_input_a("NORMAL"),
        .power_up_uninitialized("FALSE"),
        .init_file("acl_fp_div_memoryC2_uid118_invTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Arria 10")
    ) memoryC2_uid118_invTables_lutmem_dmem (
        .clocken0(en[0]),
        .aclr0(memoryC2_uid118_invTables_lutmem_reset0),
        .clock0(clk),
        .address_a(memoryC2_uid118_invTables_lutmem_aa),
        .q_a(memoryC2_uid118_invTables_lutmem_ir),
        .wren_a(),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_a(),
        .data_b(),
        .address_b(),
        .clock1(),
        .clocken1(),
        .clocken2(),
        .clocken3(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .sclr(),
        .address2_a(),
        .address2_b(),
        .q_b(),
        .eccstatus()
    );
    assign memoryC2_uid118_invTables_lutmem_r = memoryC2_uid118_invTables_lutmem_ir[11:0];

    // yPE_uid52_fpDivTest(BITSELECT,51)@0
    assign yPE_uid52_fpDivTest_b = b[13:0];

    // redist7_yPE_uid52_fpDivTest_b_2(DELAY,159)
    dspba_delay_ver #( .width(14), .depth(2), .reset_kind("ASYNC") )
    redist7_yPE_uid52_fpDivTest_b_2 ( .xin(yPE_uid52_fpDivTest_b), .xout(redist7_yPE_uid52_fpDivTest_b_2_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // yT1_uid124_invPolyEval(BITSELECT,123)@2
    assign yT1_uid124_invPolyEval_b = redist7_yPE_uid52_fpDivTest_b_2_q[13:2];

    // prodXY_uid140_pT1_uid125_invPolyEval_cma(CHAINMULTADD,149)@2 + 3
    assign prodXY_uid140_pT1_uid125_invPolyEval_cma_reset = areset;
    assign prodXY_uid140_pT1_uid125_invPolyEval_cma_ena0 = en[0];
    assign prodXY_uid140_pT1_uid125_invPolyEval_cma_ena1 = prodXY_uid140_pT1_uid125_invPolyEval_cma_ena0;
    assign prodXY_uid140_pT1_uid125_invPolyEval_cma_ena2 = prodXY_uid140_pT1_uid125_invPolyEval_cma_ena0;
    assign prodXY_uid140_pT1_uid125_invPolyEval_cma_l[0] = $signed({1'b0, prodXY_uid140_pT1_uid125_invPolyEval_cma_a1[0][11:0]});
    assign prodXY_uid140_pT1_uid125_invPolyEval_cma_p[0] = prodXY_uid140_pT1_uid125_invPolyEval_cma_l[0] * prodXY_uid140_pT1_uid125_invPolyEval_cma_c1[0];
    assign prodXY_uid140_pT1_uid125_invPolyEval_cma_u[0] = prodXY_uid140_pT1_uid125_invPolyEval_cma_p[0][24:0];
    assign prodXY_uid140_pT1_uid125_invPolyEval_cma_w[0] = prodXY_uid140_pT1_uid125_invPolyEval_cma_u[0];
    assign prodXY_uid140_pT1_uid125_invPolyEval_cma_x[0] = prodXY_uid140_pT1_uid125_invPolyEval_cma_w[0];
    assign prodXY_uid140_pT1_uid125_invPolyEval_cma_y[0] = prodXY_uid140_pT1_uid125_invPolyEval_cma_x[0];
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid140_pT1_uid125_invPolyEval_cma_a0 <= '{default: '0};
            prodXY_uid140_pT1_uid125_invPolyEval_cma_c0 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid140_pT1_uid125_invPolyEval_cma_ena0 == 1'b1)
            begin
                prodXY_uid140_pT1_uid125_invPolyEval_cma_a0[0] <= yT1_uid124_invPolyEval_b;
                prodXY_uid140_pT1_uid125_invPolyEval_cma_c0[0] <= memoryC2_uid118_invTables_lutmem_r;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid140_pT1_uid125_invPolyEval_cma_a1 <= '{default: '0};
            prodXY_uid140_pT1_uid125_invPolyEval_cma_c1 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid140_pT1_uid125_invPolyEval_cma_ena2 == 1'b1)
            begin
                prodXY_uid140_pT1_uid125_invPolyEval_cma_a1 <= prodXY_uid140_pT1_uid125_invPolyEval_cma_a0;
                prodXY_uid140_pT1_uid125_invPolyEval_cma_c1 <= prodXY_uid140_pT1_uid125_invPolyEval_cma_c0;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid140_pT1_uid125_invPolyEval_cma_s <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid140_pT1_uid125_invPolyEval_cma_ena1 == 1'b1)
            begin
                prodXY_uid140_pT1_uid125_invPolyEval_cma_s[0] <= prodXY_uid140_pT1_uid125_invPolyEval_cma_y[0];
            end
        end
    end
    dspba_delay_ver #( .width(24), .depth(0), .reset_kind("ASYNC") )
    prodXY_uid140_pT1_uid125_invPolyEval_cma_delay ( .xin(prodXY_uid140_pT1_uid125_invPolyEval_cma_s[0][23:0]), .xout(prodXY_uid140_pT1_uid125_invPolyEval_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign prodXY_uid140_pT1_uid125_invPolyEval_cma_q = prodXY_uid140_pT1_uid125_invPolyEval_cma_qq[23:0];

    // osig_uid141_pT1_uid125_invPolyEval(BITSELECT,140)@5
    assign osig_uid141_pT1_uid125_invPolyEval_b = prodXY_uid140_pT1_uid125_invPolyEval_cma_q[23:11];

    // highBBits_uid127_invPolyEval(BITSELECT,126)@5
    assign highBBits_uid127_invPolyEval_b = osig_uid141_pT1_uid125_invPolyEval_b[12:1];

    // redist9_yAddr_uid51_fpDivTest_b_3(DELAY,161)
    dspba_delay_ver #( .width(9), .depth(3), .reset_kind("ASYNC") )
    redist9_yAddr_uid51_fpDivTest_b_3 ( .xin(yAddr_uid51_fpDivTest_b), .xout(redist9_yAddr_uid51_fpDivTest_b_3_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // memoryC1_uid115_invTables_lutmem(DUALMEM,146)@3 + 2
    // in j@20000000
    assign memoryC1_uid115_invTables_lutmem_aa = redist9_yAddr_uid51_fpDivTest_b_3_q;
    assign memoryC1_uid115_invTables_lutmem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("M20K"),
        .operation_mode("ROM"),
        .width_a(21),
        .widthad_a(9),
        .numwords_a(512),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .outdata_reg_a("CLOCK0"),
        .outdata_aclr_a("CLEAR0"),
        .clock_enable_input_a("NORMAL"),
        .power_up_uninitialized("FALSE"),
        .init_file("acl_fp_div_memoryC1_uid115_invTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Arria 10")
    ) memoryC1_uid115_invTables_lutmem_dmem (
        .clocken0(en[0]),
        .aclr0(memoryC1_uid115_invTables_lutmem_reset0),
        .clock0(clk),
        .address_a(memoryC1_uid115_invTables_lutmem_aa),
        .q_a(memoryC1_uid115_invTables_lutmem_ir),
        .wren_a(),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_a(),
        .data_b(),
        .address_b(),
        .clock1(),
        .clocken1(),
        .clocken2(),
        .clocken3(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .sclr(),
        .address2_a(),
        .address2_b(),
        .q_b(),
        .eccstatus()
    );
    assign memoryC1_uid115_invTables_lutmem_r = memoryC1_uid115_invTables_lutmem_ir[20:0];

    // s1sumAHighB_uid128_invPolyEval(ADD,127)@5 + 1
    assign s1sumAHighB_uid128_invPolyEval_a = {{1{memoryC1_uid115_invTables_lutmem_r[20]}}, memoryC1_uid115_invTables_lutmem_r};
    assign s1sumAHighB_uid128_invPolyEval_b = {{10{highBBits_uid127_invPolyEval_b[11]}}, highBBits_uid127_invPolyEval_b};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            s1sumAHighB_uid128_invPolyEval_o <= 22'b0;
        end
        else if (en == 1'b1)
        begin
            s1sumAHighB_uid128_invPolyEval_o <= $signed(s1sumAHighB_uid128_invPolyEval_a) + $signed(s1sumAHighB_uid128_invPolyEval_b);
        end
    end
    assign s1sumAHighB_uid128_invPolyEval_q = s1sumAHighB_uid128_invPolyEval_o[21:0];

    // lowRangeB_uid126_invPolyEval(BITSELECT,125)@5
    assign lowRangeB_uid126_invPolyEval_in = osig_uid141_pT1_uid125_invPolyEval_b[0:0];
    assign lowRangeB_uid126_invPolyEval_b = lowRangeB_uid126_invPolyEval_in[0:0];

    // redist1_lowRangeB_uid126_invPolyEval_b_1(DELAY,153)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist1_lowRangeB_uid126_invPolyEval_b_1 ( .xin(lowRangeB_uid126_invPolyEval_b), .xout(redist1_lowRangeB_uid126_invPolyEval_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // s1_uid129_invPolyEval(BITJOIN,128)@6
    assign s1_uid129_invPolyEval_q = {s1sumAHighB_uid128_invPolyEval_q, redist1_lowRangeB_uid126_invPolyEval_b_1_q};

    // redist8_yPE_uid52_fpDivTest_b_6_notEnable(LOGICAL,180)
    assign redist8_yPE_uid52_fpDivTest_b_6_notEnable_q = ~ (en);

    // redist8_yPE_uid52_fpDivTest_b_6_nor(LOGICAL,181)
    assign redist8_yPE_uid52_fpDivTest_b_6_nor_q = ~ (redist8_yPE_uid52_fpDivTest_b_6_notEnable_q | redist8_yPE_uid52_fpDivTest_b_6_sticky_ena_q);

    // redist8_yPE_uid52_fpDivTest_b_6_mem_last(CONSTANT,177)
    assign redist8_yPE_uid52_fpDivTest_b_6_mem_last_q = 2'b01;

    // redist8_yPE_uid52_fpDivTest_b_6_cmp(LOGICAL,178)
    assign redist8_yPE_uid52_fpDivTest_b_6_cmp_q = redist8_yPE_uid52_fpDivTest_b_6_mem_last_q == redist8_yPE_uid52_fpDivTest_b_6_rdmux_q ? 1'b1 : 1'b0;

    // redist8_yPE_uid52_fpDivTest_b_6_cmpReg(REG,179)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist8_yPE_uid52_fpDivTest_b_6_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist8_yPE_uid52_fpDivTest_b_6_cmpReg_q <= redist8_yPE_uid52_fpDivTest_b_6_cmp_q;
        end
    end

    // redist8_yPE_uid52_fpDivTest_b_6_sticky_ena(REG,182)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist8_yPE_uid52_fpDivTest_b_6_sticky_ena_q <= 1'b0;
        end
        else if (redist8_yPE_uid52_fpDivTest_b_6_nor_q == 1'b1)
        begin
            redist8_yPE_uid52_fpDivTest_b_6_sticky_ena_q <= redist8_yPE_uid52_fpDivTest_b_6_cmpReg_q;
        end
    end

    // redist8_yPE_uid52_fpDivTest_b_6_enaAnd(LOGICAL,183)
    assign redist8_yPE_uid52_fpDivTest_b_6_enaAnd_q = redist8_yPE_uid52_fpDivTest_b_6_sticky_ena_q & en;

    // redist8_yPE_uid52_fpDivTest_b_6_rdcnt(COUNTER,174)
    // low=0, high=2, step=1, init=0
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist8_yPE_uid52_fpDivTest_b_6_rdcnt_i <= 2'd0;
            redist8_yPE_uid52_fpDivTest_b_6_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist8_yPE_uid52_fpDivTest_b_6_rdcnt_i == 2'd1)
            begin
                redist8_yPE_uid52_fpDivTest_b_6_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist8_yPE_uid52_fpDivTest_b_6_rdcnt_eq <= 1'b0;
            end
            if (redist8_yPE_uid52_fpDivTest_b_6_rdcnt_eq == 1'b1)
            begin
                redist8_yPE_uid52_fpDivTest_b_6_rdcnt_i <= $unsigned(redist8_yPE_uid52_fpDivTest_b_6_rdcnt_i) + $unsigned(2'd2);
            end
            else
            begin
                redist8_yPE_uid52_fpDivTest_b_6_rdcnt_i <= $unsigned(redist8_yPE_uid52_fpDivTest_b_6_rdcnt_i) + $unsigned(2'd1);
            end
        end
    end
    assign redist8_yPE_uid52_fpDivTest_b_6_rdcnt_q = redist8_yPE_uid52_fpDivTest_b_6_rdcnt_i[1:0];

    // redist8_yPE_uid52_fpDivTest_b_6_rdmux(MUX,175)
    assign redist8_yPE_uid52_fpDivTest_b_6_rdmux_s = en;
    always @(redist8_yPE_uid52_fpDivTest_b_6_rdmux_s or redist8_yPE_uid52_fpDivTest_b_6_wraddr_q or redist8_yPE_uid52_fpDivTest_b_6_rdcnt_q)
    begin
        unique case (redist8_yPE_uid52_fpDivTest_b_6_rdmux_s)
            1'b0 : redist8_yPE_uid52_fpDivTest_b_6_rdmux_q = redist8_yPE_uid52_fpDivTest_b_6_wraddr_q;
            1'b1 : redist8_yPE_uid52_fpDivTest_b_6_rdmux_q = redist8_yPE_uid52_fpDivTest_b_6_rdcnt_q;
            default : redist8_yPE_uid52_fpDivTest_b_6_rdmux_q = 2'b0;
        endcase
    end

    // redist8_yPE_uid52_fpDivTest_b_6_wraddr(REG,176)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist8_yPE_uid52_fpDivTest_b_6_wraddr_q <= 2'b10;
        end
        else
        begin
            redist8_yPE_uid52_fpDivTest_b_6_wraddr_q <= redist8_yPE_uid52_fpDivTest_b_6_rdmux_q;
        end
    end

    // redist8_yPE_uid52_fpDivTest_b_6_mem(DUALMEM,173)
    assign redist8_yPE_uid52_fpDivTest_b_6_mem_ia = redist7_yPE_uid52_fpDivTest_b_2_q;
    assign redist8_yPE_uid52_fpDivTest_b_6_mem_aa = redist8_yPE_uid52_fpDivTest_b_6_wraddr_q;
    assign redist8_yPE_uid52_fpDivTest_b_6_mem_ab = redist8_yPE_uid52_fpDivTest_b_6_rdmux_q;
    assign redist8_yPE_uid52_fpDivTest_b_6_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(14),
        .widthad_a(2),
        .numwords_a(3),
        .width_b(14),
        .widthad_b(2),
        .numwords_b(3),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .rdcontrol_reg_b("CLOCK0"),
        .byteena_reg_b("CLOCK0"),
        .outdata_reg_b("CLOCK1"),
        .outdata_aclr_b("CLEAR1"),
        .clock_enable_input_a("NORMAL"),
        .clock_enable_input_b("NORMAL"),
        .clock_enable_output_b("NORMAL"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("TRUE"),
        .intended_device_family("Arria 10")
    ) redist8_yPE_uid52_fpDivTest_b_6_mem_dmem (
        .clocken1(redist8_yPE_uid52_fpDivTest_b_6_enaAnd_q[0]),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .aclr1(redist8_yPE_uid52_fpDivTest_b_6_mem_reset0),
        .clock1(clk),
        .address_a(redist8_yPE_uid52_fpDivTest_b_6_mem_aa),
        .data_a(redist8_yPE_uid52_fpDivTest_b_6_mem_ia),
        .wren_a(en[0]),
        .address_b(redist8_yPE_uid52_fpDivTest_b_6_mem_ab),
        .q_b(redist8_yPE_uid52_fpDivTest_b_6_mem_iq),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_b(),
        .clocken2(),
        .clocken3(),
        .aclr0(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .sclr(),
        .address2_a(),
        .address2_b(),
        .q_a(),
        .eccstatus()
    );
    assign redist8_yPE_uid52_fpDivTest_b_6_mem_q = redist8_yPE_uid52_fpDivTest_b_6_mem_iq[13:0];

    // prodXY_uid143_pT2_uid131_invPolyEval_cma(CHAINMULTADD,150)@6 + 3
    assign prodXY_uid143_pT2_uid131_invPolyEval_cma_reset = areset;
    assign prodXY_uid143_pT2_uid131_invPolyEval_cma_ena0 = en[0];
    assign prodXY_uid143_pT2_uid131_invPolyEval_cma_ena1 = prodXY_uid143_pT2_uid131_invPolyEval_cma_ena0;
    assign prodXY_uid143_pT2_uid131_invPolyEval_cma_ena2 = prodXY_uid143_pT2_uid131_invPolyEval_cma_ena0;
    assign prodXY_uid143_pT2_uid131_invPolyEval_cma_l[0] = $signed({1'b0, prodXY_uid143_pT2_uid131_invPolyEval_cma_a1[0][13:0]});
    assign prodXY_uid143_pT2_uid131_invPolyEval_cma_p[0] = prodXY_uid143_pT2_uid131_invPolyEval_cma_l[0] * prodXY_uid143_pT2_uid131_invPolyEval_cma_c1[0];
    assign prodXY_uid143_pT2_uid131_invPolyEval_cma_u[0] = prodXY_uid143_pT2_uid131_invPolyEval_cma_p[0][37:0];
    assign prodXY_uid143_pT2_uid131_invPolyEval_cma_w[0] = prodXY_uid143_pT2_uid131_invPolyEval_cma_u[0];
    assign prodXY_uid143_pT2_uid131_invPolyEval_cma_x[0] = prodXY_uid143_pT2_uid131_invPolyEval_cma_w[0];
    assign prodXY_uid143_pT2_uid131_invPolyEval_cma_y[0] = prodXY_uid143_pT2_uid131_invPolyEval_cma_x[0];
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid143_pT2_uid131_invPolyEval_cma_a0 <= '{default: '0};
            prodXY_uid143_pT2_uid131_invPolyEval_cma_c0 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid143_pT2_uid131_invPolyEval_cma_ena0 == 1'b1)
            begin
                prodXY_uid143_pT2_uid131_invPolyEval_cma_a0[0] <= redist8_yPE_uid52_fpDivTest_b_6_mem_q;
                prodXY_uid143_pT2_uid131_invPolyEval_cma_c0[0] <= s1_uid129_invPolyEval_q;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid143_pT2_uid131_invPolyEval_cma_a1 <= '{default: '0};
            prodXY_uid143_pT2_uid131_invPolyEval_cma_c1 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid143_pT2_uid131_invPolyEval_cma_ena2 == 1'b1)
            begin
                prodXY_uid143_pT2_uid131_invPolyEval_cma_a1 <= prodXY_uid143_pT2_uid131_invPolyEval_cma_a0;
                prodXY_uid143_pT2_uid131_invPolyEval_cma_c1 <= prodXY_uid143_pT2_uid131_invPolyEval_cma_c0;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid143_pT2_uid131_invPolyEval_cma_s <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid143_pT2_uid131_invPolyEval_cma_ena1 == 1'b1)
            begin
                prodXY_uid143_pT2_uid131_invPolyEval_cma_s[0] <= prodXY_uid143_pT2_uid131_invPolyEval_cma_y[0];
            end
        end
    end
    dspba_delay_ver #( .width(37), .depth(0), .reset_kind("ASYNC") )
    prodXY_uid143_pT2_uid131_invPolyEval_cma_delay ( .xin(prodXY_uid143_pT2_uid131_invPolyEval_cma_s[0][36:0]), .xout(prodXY_uid143_pT2_uid131_invPolyEval_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign prodXY_uid143_pT2_uid131_invPolyEval_cma_q = prodXY_uid143_pT2_uid131_invPolyEval_cma_qq[36:0];

    // osig_uid144_pT2_uid131_invPolyEval(BITSELECT,143)@9
    assign osig_uid144_pT2_uid131_invPolyEval_b = prodXY_uid143_pT2_uid131_invPolyEval_cma_q[36:13];

    // highBBits_uid133_invPolyEval(BITSELECT,132)@9
    assign highBBits_uid133_invPolyEval_b = osig_uid144_pT2_uid131_invPolyEval_b[23:2];

    // redist10_yAddr_uid51_fpDivTest_b_7(DELAY,162)
    dspba_delay_ver #( .width(9), .depth(4), .reset_kind("ASYNC") )
    redist10_yAddr_uid51_fpDivTest_b_7 ( .xin(redist9_yAddr_uid51_fpDivTest_b_3_q), .xout(redist10_yAddr_uid51_fpDivTest_b_7_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // memoryC0_uid112_invTables_lutmem(DUALMEM,145)@7 + 2
    // in j@20000000
    assign memoryC0_uid112_invTables_lutmem_aa = redist10_yAddr_uid51_fpDivTest_b_7_q;
    assign memoryC0_uid112_invTables_lutmem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("M20K"),
        .operation_mode("ROM"),
        .width_a(31),
        .widthad_a(9),
        .numwords_a(512),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .outdata_reg_a("CLOCK0"),
        .outdata_aclr_a("CLEAR0"),
        .clock_enable_input_a("NORMAL"),
        .power_up_uninitialized("FALSE"),
        .init_file("acl_fp_div_memoryC0_uid112_invTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Arria 10")
    ) memoryC0_uid112_invTables_lutmem_dmem (
        .clocken0(en[0]),
        .aclr0(memoryC0_uid112_invTables_lutmem_reset0),
        .clock0(clk),
        .address_a(memoryC0_uid112_invTables_lutmem_aa),
        .q_a(memoryC0_uid112_invTables_lutmem_ir),
        .wren_a(),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_a(),
        .data_b(),
        .address_b(),
        .clock1(),
        .clocken1(),
        .clocken2(),
        .clocken3(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .sclr(),
        .address2_a(),
        .address2_b(),
        .q_b(),
        .eccstatus()
    );
    assign memoryC0_uid112_invTables_lutmem_r = memoryC0_uid112_invTables_lutmem_ir[30:0];

    // s2sumAHighB_uid134_invPolyEval(ADD,133)@9
    assign s2sumAHighB_uid134_invPolyEval_a = {{1{memoryC0_uid112_invTables_lutmem_r[30]}}, memoryC0_uid112_invTables_lutmem_r};
    assign s2sumAHighB_uid134_invPolyEval_b = {{10{highBBits_uid133_invPolyEval_b[21]}}, highBBits_uid133_invPolyEval_b};
    assign s2sumAHighB_uid134_invPolyEval_o = $signed(s2sumAHighB_uid134_invPolyEval_a) + $signed(s2sumAHighB_uid134_invPolyEval_b);
    assign s2sumAHighB_uid134_invPolyEval_q = s2sumAHighB_uid134_invPolyEval_o[31:0];

    // lowRangeB_uid132_invPolyEval(BITSELECT,131)@9
    assign lowRangeB_uid132_invPolyEval_in = osig_uid144_pT2_uid131_invPolyEval_b[1:0];
    assign lowRangeB_uid132_invPolyEval_b = lowRangeB_uid132_invPolyEval_in[1:0];

    // s2_uid135_invPolyEval(BITJOIN,134)@9
    assign s2_uid135_invPolyEval_q = {s2sumAHighB_uid134_invPolyEval_q, lowRangeB_uid132_invPolyEval_b};

    // invY_uid54_fpDivTest_merged_bit_select(BITSELECT,151)@9
    assign invY_uid54_fpDivTest_merged_bit_select_in = s2_uid135_invPolyEval_q[31:0];
    assign invY_uid54_fpDivTest_merged_bit_select_b = invY_uid54_fpDivTest_merged_bit_select_in[30:5];
    assign invY_uid54_fpDivTest_merged_bit_select_c = invY_uid54_fpDivTest_merged_bit_select_in[31:31];

    // redist0_invY_uid54_fpDivTest_merged_bit_select_b_1(DELAY,152)
    dspba_delay_ver #( .width(26), .depth(1), .reset_kind("ASYNC") )
    redist0_invY_uid54_fpDivTest_merged_bit_select_b_1 ( .xin(invY_uid54_fpDivTest_merged_bit_select_b), .xout(redist0_invY_uid54_fpDivTest_merged_bit_select_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // GND(CONSTANT,0)
    assign GND_q = 1'b0;

    // prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma(CHAINMULTADD,148)@10 + 3
    assign prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_reset = areset;
    assign prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_ena0 = en[0];
    assign prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_ena1 = prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_ena0;
    assign prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_ena2 = prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_ena0;
    assign prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_p[0] = prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_a1[0] * prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_c1[0];
    assign prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_u[0] = prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_p[0][49:0];
    assign prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_w[0] = prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_u[0];
    assign prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_x[0] = prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_w[0];
    assign prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_y[0] = prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_x[0];
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_a0 <= '{default: '0};
            prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_c0 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_ena0 == 1'b1)
            begin
                prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_a0[0] <= redist0_invY_uid54_fpDivTest_merged_bit_select_b_1_q;
                prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_c0[0] <= lOAdded_uid58_fpDivTest_q;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_a1 <= '{default: '0};
            prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_c1 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_ena2 == 1'b1)
            begin
                prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_a1 <= prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_a0;
                prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_c1 <= prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_c0;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_s <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_ena1 == 1'b1)
            begin
                prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_s[0] <= prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_y[0];
            end
        end
    end
    dspba_delay_ver #( .width(50), .depth(0), .reset_kind("ASYNC") )
    prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_delay ( .xin(prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_s[0][49:0]), .xout(prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_q = prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_qq[49:0];

    // osig_uid138_prodDivPreNormProd_uid60_fpDivTest(BITSELECT,137)@13
    assign osig_uid138_prodDivPreNormProd_uid60_fpDivTest_b = prodXY_uid137_prodDivPreNormProd_uid60_fpDivTest_cma_q[49:24];

    // updatedY_uid16_fpDivTest(BITJOIN,15)@0
    assign updatedY_uid16_fpDivTest_q = {GND_q, paddingY_uid15_fpDivTest_q};

    // fracYZero_uid15_fpDivTest(LOGICAL,16)@0 + 1
    assign fracYZero_uid15_fpDivTest_a = {1'b0, fracY_uid13_fpDivTest_b};
    assign fracYZero_uid15_fpDivTest_qi = fracYZero_uid15_fpDivTest_a == updatedY_uid16_fpDivTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    fracYZero_uid15_fpDivTest_delay ( .xin(fracYZero_uid15_fpDivTest_qi), .xout(fracYZero_uid15_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist19_fracYZero_uid15_fpDivTest_q_9(DELAY,171)
    dspba_delay_ver #( .width(1), .depth(8), .reset_kind("ASYNC") )
    redist19_fracYZero_uid15_fpDivTest_q_9 ( .xin(fracYZero_uid15_fpDivTest_q), .xout(redist19_fracYZero_uid15_fpDivTest_q_9_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // fracYPostZ_uid56_fpDivTest(LOGICAL,55)@9 + 1
    assign fracYPostZ_uid56_fpDivTest_qi = redist19_fracYZero_uid15_fpDivTest_q_9_q | invY_uid54_fpDivTest_merged_bit_select_c;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    fracYPostZ_uid56_fpDivTest_delay ( .xin(fracYPostZ_uid56_fpDivTest_qi), .xout(fracYPostZ_uid56_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist6_fracYPostZ_uid56_fpDivTest_q_4(DELAY,158)
    dspba_delay_ver #( .width(1), .depth(3), .reset_kind("ASYNC") )
    redist6_fracYPostZ_uid56_fpDivTest_q_4 ( .xin(fracYPostZ_uid56_fpDivTest_q), .xout(redist6_fracYPostZ_uid56_fpDivTest_q_4_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // divValPreNormTrunc_uid66_fpDivTest(MUX,65)@13
    assign divValPreNormTrunc_uid66_fpDivTest_s = redist6_fracYPostZ_uid56_fpDivTest_q_4_q;
    always @(divValPreNormTrunc_uid66_fpDivTest_s or en or osig_uid138_prodDivPreNormProd_uid60_fpDivTest_b or oFracXSE_mergedSignalTM_uid63_fpDivTest_q)
    begin
        unique case (divValPreNormTrunc_uid66_fpDivTest_s)
            1'b0 : divValPreNormTrunc_uid66_fpDivTest_q = osig_uid138_prodDivPreNormProd_uid60_fpDivTest_b;
            1'b1 : divValPreNormTrunc_uid66_fpDivTest_q = oFracXSE_mergedSignalTM_uid63_fpDivTest_q;
            default : divValPreNormTrunc_uid66_fpDivTest_q = 26'b0;
        endcase
    end

    // norm_uid67_fpDivTest(BITSELECT,66)@13
    assign norm_uid67_fpDivTest_b = divValPreNormTrunc_uid66_fpDivTest_q[25:25];

    // redist4_norm_uid67_fpDivTest_b_1(DELAY,156)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist4_norm_uid67_fpDivTest_b_1 ( .xin(norm_uid67_fpDivTest_b), .xout(redist4_norm_uid67_fpDivTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // rndOp_uid75_fpDivTest(BITJOIN,74)@14
    assign rndOp_uid75_fpDivTest_q = {redist4_norm_uid67_fpDivTest_b_1_q, paddingY_uid15_fpDivTest_q, VCC_q};

    // cstBiasM1_uid6_fpDivTest(CONSTANT,5)
    assign cstBiasM1_uid6_fpDivTest_q = 8'b01111110;

    // redist11_expXmY_uid47_fpDivTest_q_13_notEnable(LOGICAL,192)
    assign redist11_expXmY_uid47_fpDivTest_q_13_notEnable_q = ~ (en);

    // redist11_expXmY_uid47_fpDivTest_q_13_nor(LOGICAL,193)
    assign redist11_expXmY_uid47_fpDivTest_q_13_nor_q = ~ (redist11_expXmY_uid47_fpDivTest_q_13_notEnable_q | redist11_expXmY_uid47_fpDivTest_q_13_sticky_ena_q);

    // redist11_expXmY_uid47_fpDivTest_q_13_mem_last(CONSTANT,189)
    assign redist11_expXmY_uid47_fpDivTest_q_13_mem_last_q = 5'b01000;

    // redist11_expXmY_uid47_fpDivTest_q_13_cmp(LOGICAL,190)
    assign redist11_expXmY_uid47_fpDivTest_q_13_cmp_b = {1'b0, redist11_expXmY_uid47_fpDivTest_q_13_rdmux_q};
    assign redist11_expXmY_uid47_fpDivTest_q_13_cmp_q = redist11_expXmY_uid47_fpDivTest_q_13_mem_last_q == redist11_expXmY_uid47_fpDivTest_q_13_cmp_b ? 1'b1 : 1'b0;

    // redist11_expXmY_uid47_fpDivTest_q_13_cmpReg(REG,191)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist11_expXmY_uid47_fpDivTest_q_13_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist11_expXmY_uid47_fpDivTest_q_13_cmpReg_q <= redist11_expXmY_uid47_fpDivTest_q_13_cmp_q;
        end
    end

    // redist11_expXmY_uid47_fpDivTest_q_13_sticky_ena(REG,194)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist11_expXmY_uid47_fpDivTest_q_13_sticky_ena_q <= 1'b0;
        end
        else if (redist11_expXmY_uid47_fpDivTest_q_13_nor_q == 1'b1)
        begin
            redist11_expXmY_uid47_fpDivTest_q_13_sticky_ena_q <= redist11_expXmY_uid47_fpDivTest_q_13_cmpReg_q;
        end
    end

    // redist11_expXmY_uid47_fpDivTest_q_13_enaAnd(LOGICAL,195)
    assign redist11_expXmY_uid47_fpDivTest_q_13_enaAnd_q = redist11_expXmY_uid47_fpDivTest_q_13_sticky_ena_q & en;

    // redist11_expXmY_uid47_fpDivTest_q_13_rdcnt(COUNTER,186)
    // low=0, high=9, step=1, init=0
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist11_expXmY_uid47_fpDivTest_q_13_rdcnt_i <= 4'd0;
            redist11_expXmY_uid47_fpDivTest_q_13_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist11_expXmY_uid47_fpDivTest_q_13_rdcnt_i == 4'd8)
            begin
                redist11_expXmY_uid47_fpDivTest_q_13_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist11_expXmY_uid47_fpDivTest_q_13_rdcnt_eq <= 1'b0;
            end
            if (redist11_expXmY_uid47_fpDivTest_q_13_rdcnt_eq == 1'b1)
            begin
                redist11_expXmY_uid47_fpDivTest_q_13_rdcnt_i <= $unsigned(redist11_expXmY_uid47_fpDivTest_q_13_rdcnt_i) + $unsigned(4'd7);
            end
            else
            begin
                redist11_expXmY_uid47_fpDivTest_q_13_rdcnt_i <= $unsigned(redist11_expXmY_uid47_fpDivTest_q_13_rdcnt_i) + $unsigned(4'd1);
            end
        end
    end
    assign redist11_expXmY_uid47_fpDivTest_q_13_rdcnt_q = redist11_expXmY_uid47_fpDivTest_q_13_rdcnt_i[3:0];

    // redist11_expXmY_uid47_fpDivTest_q_13_rdmux(MUX,187)
    assign redist11_expXmY_uid47_fpDivTest_q_13_rdmux_s = en;
    always @(redist11_expXmY_uid47_fpDivTest_q_13_rdmux_s or redist11_expXmY_uid47_fpDivTest_q_13_wraddr_q or redist11_expXmY_uid47_fpDivTest_q_13_rdcnt_q)
    begin
        unique case (redist11_expXmY_uid47_fpDivTest_q_13_rdmux_s)
            1'b0 : redist11_expXmY_uid47_fpDivTest_q_13_rdmux_q = redist11_expXmY_uid47_fpDivTest_q_13_wraddr_q;
            1'b1 : redist11_expXmY_uid47_fpDivTest_q_13_rdmux_q = redist11_expXmY_uid47_fpDivTest_q_13_rdcnt_q;
            default : redist11_expXmY_uid47_fpDivTest_q_13_rdmux_q = 4'b0;
        endcase
    end

    // expXmY_uid47_fpDivTest(SUB,46)@0 + 1
    assign expXmY_uid47_fpDivTest_a = {1'b0, expX_uid9_fpDivTest_b};
    assign expXmY_uid47_fpDivTest_b = {1'b0, expY_uid12_fpDivTest_b};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            expXmY_uid47_fpDivTest_o <= 9'b0;
        end
        else if (en == 1'b1)
        begin
            expXmY_uid47_fpDivTest_o <= $unsigned(expXmY_uid47_fpDivTest_a) - $unsigned(expXmY_uid47_fpDivTest_b);
        end
    end
    assign expXmY_uid47_fpDivTest_q = expXmY_uid47_fpDivTest_o[8:0];

    // redist11_expXmY_uid47_fpDivTest_q_13_wraddr(REG,188)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist11_expXmY_uid47_fpDivTest_q_13_wraddr_q <= 4'b1001;
        end
        else
        begin
            redist11_expXmY_uid47_fpDivTest_q_13_wraddr_q <= redist11_expXmY_uid47_fpDivTest_q_13_rdmux_q;
        end
    end

    // redist11_expXmY_uid47_fpDivTest_q_13_mem(DUALMEM,185)
    assign redist11_expXmY_uid47_fpDivTest_q_13_mem_ia = expXmY_uid47_fpDivTest_q;
    assign redist11_expXmY_uid47_fpDivTest_q_13_mem_aa = redist11_expXmY_uid47_fpDivTest_q_13_wraddr_q;
    assign redist11_expXmY_uid47_fpDivTest_q_13_mem_ab = redist11_expXmY_uid47_fpDivTest_q_13_rdmux_q;
    assign redist11_expXmY_uid47_fpDivTest_q_13_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(9),
        .widthad_a(4),
        .numwords_a(10),
        .width_b(9),
        .widthad_b(4),
        .numwords_b(10),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .rdcontrol_reg_b("CLOCK0"),
        .byteena_reg_b("CLOCK0"),
        .outdata_reg_b("CLOCK1"),
        .outdata_aclr_b("CLEAR1"),
        .clock_enable_input_a("NORMAL"),
        .clock_enable_input_b("NORMAL"),
        .clock_enable_output_b("NORMAL"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("TRUE"),
        .intended_device_family("Arria 10")
    ) redist11_expXmY_uid47_fpDivTest_q_13_mem_dmem (
        .clocken1(redist11_expXmY_uid47_fpDivTest_q_13_enaAnd_q[0]),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .aclr1(redist11_expXmY_uid47_fpDivTest_q_13_mem_reset0),
        .clock1(clk),
        .address_a(redist11_expXmY_uid47_fpDivTest_q_13_mem_aa),
        .data_a(redist11_expXmY_uid47_fpDivTest_q_13_mem_ia),
        .wren_a(en[0]),
        .address_b(redist11_expXmY_uid47_fpDivTest_q_13_mem_ab),
        .q_b(redist11_expXmY_uid47_fpDivTest_q_13_mem_iq),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_b(),
        .clocken2(),
        .clocken3(),
        .aclr0(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .sclr(),
        .address2_a(),
        .address2_b(),
        .q_a(),
        .eccstatus()
    );
    assign redist11_expXmY_uid47_fpDivTest_q_13_mem_q = redist11_expXmY_uid47_fpDivTest_q_13_mem_iq[8:0];

    // redist11_expXmY_uid47_fpDivTest_q_13_outputreg(DELAY,184)
    dspba_delay_ver #( .width(9), .depth(1), .reset_kind("ASYNC") )
    redist11_expXmY_uid47_fpDivTest_q_13_outputreg ( .xin(redist11_expXmY_uid47_fpDivTest_q_13_mem_q), .xout(redist11_expXmY_uid47_fpDivTest_q_13_outputreg_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expR_uid48_fpDivTest(ADD,47)@13 + 1
    assign expR_uid48_fpDivTest_a = {{2{redist11_expXmY_uid47_fpDivTest_q_13_outputreg_q[8]}}, redist11_expXmY_uid47_fpDivTest_q_13_outputreg_q};
    assign expR_uid48_fpDivTest_b = {3'b000, cstBiasM1_uid6_fpDivTest_q};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            expR_uid48_fpDivTest_o <= 11'b0;
        end
        else if (en == 1'b1)
        begin
            expR_uid48_fpDivTest_o <= $signed(expR_uid48_fpDivTest_a) + $signed(expR_uid48_fpDivTest_b);
        end
    end
    assign expR_uid48_fpDivTest_q = expR_uid48_fpDivTest_o[9:0];

    // divValPreNormHigh_uid68_fpDivTest(BITSELECT,67)@13
    assign divValPreNormHigh_uid68_fpDivTest_in = divValPreNormTrunc_uid66_fpDivTest_q[24:0];
    assign divValPreNormHigh_uid68_fpDivTest_b = divValPreNormHigh_uid68_fpDivTest_in[24:1];

    // divValPreNormLow_uid69_fpDivTest(BITSELECT,68)@13
    assign divValPreNormLow_uid69_fpDivTest_in = divValPreNormTrunc_uid66_fpDivTest_q[23:0];
    assign divValPreNormLow_uid69_fpDivTest_b = divValPreNormLow_uid69_fpDivTest_in[23:0];

    // normFracRnd_uid70_fpDivTest(MUX,69)@13 + 1
    assign normFracRnd_uid70_fpDivTest_s = norm_uid67_fpDivTest_b;
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            normFracRnd_uid70_fpDivTest_q <= 24'b0;
        end
        else if (en == 1'b1)
        begin
            unique case (normFracRnd_uid70_fpDivTest_s)
                1'b0 : normFracRnd_uid70_fpDivTest_q <= divValPreNormLow_uid69_fpDivTest_b;
                1'b1 : normFracRnd_uid70_fpDivTest_q <= divValPreNormHigh_uid68_fpDivTest_b;
                default : normFracRnd_uid70_fpDivTest_q <= 24'b0;
            endcase
        end
    end

    // expFracRnd_uid71_fpDivTest(BITJOIN,70)@14
    assign expFracRnd_uid71_fpDivTest_q = {expR_uid48_fpDivTest_q, normFracRnd_uid70_fpDivTest_q};

    // expFracPostRnd_uid76_fpDivTest(ADD,75)@14 + 1
    assign expFracPostRnd_uid76_fpDivTest_a = {{2{expFracRnd_uid71_fpDivTest_q[33]}}, expFracRnd_uid71_fpDivTest_q};
    assign expFracPostRnd_uid76_fpDivTest_b = {11'b00000000000, rndOp_uid75_fpDivTest_q};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            expFracPostRnd_uid76_fpDivTest_o <= 36'b0;
        end
        else if (en == 1'b1)
        begin
            expFracPostRnd_uid76_fpDivTest_o <= $signed(expFracPostRnd_uid76_fpDivTest_a) + $signed(expFracPostRnd_uid76_fpDivTest_b);
        end
    end
    assign expFracPostRnd_uid76_fpDivTest_q = expFracPostRnd_uid76_fpDivTest_o[34:0];

    // excRPreExc_uid79_fpDivTest(BITSELECT,78)@15
    assign excRPreExc_uid79_fpDivTest_in = expFracPostRnd_uid76_fpDivTest_q[31:0];
    assign excRPreExc_uid79_fpDivTest_b = excRPreExc_uid79_fpDivTest_in[31:24];

    // redist2_excRPreExc_uid79_fpDivTest_b_1(DELAY,154)
    dspba_delay_ver #( .width(8), .depth(1), .reset_kind("ASYNC") )
    redist2_excRPreExc_uid79_fpDivTest_b_1 ( .xin(excRPreExc_uid79_fpDivTest_b), .xout(redist2_excRPreExc_uid79_fpDivTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // invExpXIsMax_uid43_fpDivTest(LOGICAL,42)@15
    assign invExpXIsMax_uid43_fpDivTest_q = ~ (redist14_expXIsMax_uid38_fpDivTest_q_15_q);

    // InvExpXIsZero_uid44_fpDivTest(LOGICAL,43)@15
    assign InvExpXIsZero_uid44_fpDivTest_q = ~ (redist15_excZ_y_uid37_fpDivTest_q_15_q);

    // excR_y_uid45_fpDivTest(LOGICAL,44)@15
    assign excR_y_uid45_fpDivTest_q = InvExpXIsZero_uid44_fpDivTest_q & invExpXIsMax_uid43_fpDivTest_q;

    // excXIYR_uid93_fpDivTest(LOGICAL,92)@15
    assign excXIYR_uid93_fpDivTest_q = excI_x_uid27_fpDivTest_q & excR_y_uid45_fpDivTest_q;

    // excXIYZ_uid92_fpDivTest(LOGICAL,91)@15
    assign excXIYZ_uid92_fpDivTest_q = excI_x_uid27_fpDivTest_q & redist15_excZ_y_uid37_fpDivTest_q_15_q;

    // expRExt_uid80_fpDivTest(BITSELECT,79)@15
    assign expRExt_uid80_fpDivTest_b = expFracPostRnd_uid76_fpDivTest_q[34:24];

    // expOvf_uid84_fpDivTest(COMPARE,83)@15
    assign expOvf_uid84_fpDivTest_a = {{2{expRExt_uid80_fpDivTest_b[10]}}, expRExt_uid80_fpDivTest_b};
    assign expOvf_uid84_fpDivTest_b = {5'b00000, cstAllOWE_uid18_fpDivTest_q};
    assign expOvf_uid84_fpDivTest_o = $signed(expOvf_uid84_fpDivTest_a) - $signed(expOvf_uid84_fpDivTest_b);
    assign expOvf_uid84_fpDivTest_n[0] = ~ (expOvf_uid84_fpDivTest_o[12]);

    // invExpXIsMax_uid29_fpDivTest(LOGICAL,28)@15
    assign invExpXIsMax_uid29_fpDivTest_q = ~ (redist17_expXIsMax_uid24_fpDivTest_q_15_q);

    // InvExpXIsZero_uid30_fpDivTest(LOGICAL,29)@15
    assign InvExpXIsZero_uid30_fpDivTest_q = ~ (redist18_excZ_x_uid23_fpDivTest_q_15_q);

    // excR_x_uid31_fpDivTest(LOGICAL,30)@15
    assign excR_x_uid31_fpDivTest_q = InvExpXIsZero_uid30_fpDivTest_q & invExpXIsMax_uid29_fpDivTest_q;

    // excXRYROvf_uid91_fpDivTest(LOGICAL,90)@15
    assign excXRYROvf_uid91_fpDivTest_q = excR_x_uid31_fpDivTest_q & excR_y_uid45_fpDivTest_q & expOvf_uid84_fpDivTest_n;

    // excXRYZ_uid90_fpDivTest(LOGICAL,89)@15
    assign excXRYZ_uid90_fpDivTest_q = excR_x_uid31_fpDivTest_q & redist15_excZ_y_uid37_fpDivTest_q_15_q;

    // excRInf_uid94_fpDivTest(LOGICAL,93)@15 + 1
    assign excRInf_uid94_fpDivTest_qi = excXRYZ_uid90_fpDivTest_q | excXRYROvf_uid91_fpDivTest_q | excXIYZ_uid92_fpDivTest_q | excXIYR_uid93_fpDivTest_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    excRInf_uid94_fpDivTest_delay ( .xin(excRInf_uid94_fpDivTest_qi), .xout(excRInf_uid94_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // xRegOrZero_uid87_fpDivTest(LOGICAL,86)@15
    assign xRegOrZero_uid87_fpDivTest_q = excR_x_uid31_fpDivTest_q | redist18_excZ_x_uid23_fpDivTest_q_15_q;

    // regOrZeroOverInf_uid88_fpDivTest(LOGICAL,87)@15
    assign regOrZeroOverInf_uid88_fpDivTest_q = xRegOrZero_uid87_fpDivTest_q & excI_y_uid41_fpDivTest_q;

    // expUdf_uid81_fpDivTest(COMPARE,80)@15
    assign expUdf_uid81_fpDivTest_a = {12'b000000000000, GND_q};
    assign expUdf_uid81_fpDivTest_b = {{2{expRExt_uid80_fpDivTest_b[10]}}, expRExt_uid80_fpDivTest_b};
    assign expUdf_uid81_fpDivTest_o = $signed(expUdf_uid81_fpDivTest_a) - $signed(expUdf_uid81_fpDivTest_b);
    assign expUdf_uid81_fpDivTest_n[0] = ~ (expUdf_uid81_fpDivTest_o[12]);

    // regOverRegWithUf_uid86_fpDivTest(LOGICAL,85)@15
    assign regOverRegWithUf_uid86_fpDivTest_q = expUdf_uid81_fpDivTest_n & excR_x_uid31_fpDivTest_q & excR_y_uid45_fpDivTest_q;

    // zeroOverReg_uid85_fpDivTest(LOGICAL,84)@15
    assign zeroOverReg_uid85_fpDivTest_q = redist18_excZ_x_uid23_fpDivTest_q_15_q & excR_y_uid45_fpDivTest_q;

    // excRZero_uid89_fpDivTest(LOGICAL,88)@15 + 1
    assign excRZero_uid89_fpDivTest_qi = zeroOverReg_uid85_fpDivTest_q | regOverRegWithUf_uid86_fpDivTest_q | regOrZeroOverInf_uid88_fpDivTest_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    excRZero_uid89_fpDivTest_delay ( .xin(excRZero_uid89_fpDivTest_qi), .xout(excRZero_uid89_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // concExc_uid98_fpDivTest(BITJOIN,97)@16
    assign concExc_uid98_fpDivTest_q = {excRNaN_uid97_fpDivTest_q, excRInf_uid94_fpDivTest_q, excRZero_uid89_fpDivTest_q};

    // excREnc_uid99_fpDivTest(LOOKUP,98)@16
    always @(concExc_uid98_fpDivTest_q)
    begin
        // Begin reserved scope level
        unique case (concExc_uid98_fpDivTest_q)
            3'b000 : excREnc_uid99_fpDivTest_q = 2'b01;
            3'b001 : excREnc_uid99_fpDivTest_q = 2'b00;
            3'b010 : excREnc_uid99_fpDivTest_q = 2'b10;
            3'b011 : excREnc_uid99_fpDivTest_q = 2'b00;
            3'b100 : excREnc_uid99_fpDivTest_q = 2'b11;
            3'b101 : excREnc_uid99_fpDivTest_q = 2'b00;
            3'b110 : excREnc_uid99_fpDivTest_q = 2'b00;
            3'b111 : excREnc_uid99_fpDivTest_q = 2'b00;
            default : begin
                          // unreachable
                          excREnc_uid99_fpDivTest_q = 2'bxx;
                      end
        endcase
        // End reserved scope level
    end

    // expRPostExc_uid107_fpDivTest(MUX,106)@16
    assign expRPostExc_uid107_fpDivTest_s = excREnc_uid99_fpDivTest_q;
    always @(expRPostExc_uid107_fpDivTest_s or en or cstAllZWE_uid20_fpDivTest_q or redist2_excRPreExc_uid79_fpDivTest_b_1_q or cstAllOWE_uid18_fpDivTest_q)
    begin
        unique case (expRPostExc_uid107_fpDivTest_s)
            2'b00 : expRPostExc_uid107_fpDivTest_q = cstAllZWE_uid20_fpDivTest_q;
            2'b01 : expRPostExc_uid107_fpDivTest_q = redist2_excRPreExc_uid79_fpDivTest_b_1_q;
            2'b10 : expRPostExc_uid107_fpDivTest_q = cstAllOWE_uid18_fpDivTest_q;
            2'b11 : expRPostExc_uid107_fpDivTest_q = cstAllOWE_uid18_fpDivTest_q;
            default : expRPostExc_uid107_fpDivTest_q = 8'b0;
        endcase
    end

    // oneFracRPostExc2_uid100_fpDivTest(CONSTANT,99)
    assign oneFracRPostExc2_uid100_fpDivTest_q = 23'b00000000000000000000001;

    // fracRPreExc_uid78_fpDivTest(BITSELECT,77)@15
    assign fracRPreExc_uid78_fpDivTest_in = expFracPostRnd_uid76_fpDivTest_q[23:0];
    assign fracRPreExc_uid78_fpDivTest_b = fracRPreExc_uid78_fpDivTest_in[23:1];

    // redist3_fracRPreExc_uid78_fpDivTest_b_1(DELAY,155)
    dspba_delay_ver #( .width(23), .depth(1), .reset_kind("ASYNC") )
    redist3_fracRPreExc_uid78_fpDivTest_b_1 ( .xin(fracRPreExc_uid78_fpDivTest_b), .xout(redist3_fracRPreExc_uid78_fpDivTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // fracRPostExc_uid103_fpDivTest(MUX,102)@16
    assign fracRPostExc_uid103_fpDivTest_s = excREnc_uid99_fpDivTest_q;
    always @(fracRPostExc_uid103_fpDivTest_s or en or paddingY_uid15_fpDivTest_q or redist3_fracRPreExc_uid78_fpDivTest_b_1_q or oneFracRPostExc2_uid100_fpDivTest_q)
    begin
        unique case (fracRPostExc_uid103_fpDivTest_s)
            2'b00 : fracRPostExc_uid103_fpDivTest_q = paddingY_uid15_fpDivTest_q;
            2'b01 : fracRPostExc_uid103_fpDivTest_q = redist3_fracRPreExc_uid78_fpDivTest_b_1_q;
            2'b10 : fracRPostExc_uid103_fpDivTest_q = paddingY_uid15_fpDivTest_q;
            2'b11 : fracRPostExc_uid103_fpDivTest_q = oneFracRPostExc2_uid100_fpDivTest_q;
            default : fracRPostExc_uid103_fpDivTest_q = 23'b0;
        endcase
    end

    // divR_uid110_fpDivTest(BITJOIN,109)@16
    assign divR_uid110_fpDivTest_q = {sRPostExc_uid109_fpDivTest_q, expRPostExc_uid107_fpDivTest_q, fracRPostExc_uid103_fpDivTest_q};

    // xOut(GPOUT,4)@16
    assign q = divR_uid110_fpDivTest_q;

endmodule
