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

// SystemVerilog created from acl_fdiv
// SystemVerilog created on Sun Dec 27 09:48:58 2020


(* altera_attribute = "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007; -name MESSAGE_DISABLE 10958" *)
module acl_fdiv (
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
    wire [7:0] cstBias_uid7_fpDivTest_q;
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
    wire [0:0] excZ_x_uid23_fpDivTest_q;
    wire [0:0] expXIsMax_uid24_fpDivTest_q;
    wire [0:0] fracXIsZero_uid25_fpDivTest_qi;
    reg [0:0] fracXIsZero_uid25_fpDivTest_q;
    wire [0:0] fracXIsNotZero_uid26_fpDivTest_q;
    wire [0:0] excI_x_uid27_fpDivTest_q;
    wire [0:0] excN_x_uid28_fpDivTest_q;
    wire [0:0] invExpXIsMax_uid29_fpDivTest_q;
    wire [0:0] InvExpXIsZero_uid30_fpDivTest_q;
    wire [0:0] excR_x_uid31_fpDivTest_qi;
    reg [0:0] excR_x_uid31_fpDivTest_q;
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
    wire [31:0] invY_uid54_fpDivTest_in;
    wire [26:0] invY_uid54_fpDivTest_b;
    wire [32:0] invYO_uid55_fpDivTest_in;
    wire [0:0] invYO_uid55_fpDivTest_b;
    wire [23:0] lOAdded_uid57_fpDivTest_q;
    wire [3:0] z4_uid60_fpDivTest_q;
    wire [27:0] oFracXZ4_uid61_fpDivTest_q;
    wire [0:0] divValPreNormYPow2Exc_uid63_fpDivTest_s;
    reg [27:0] divValPreNormYPow2Exc_uid63_fpDivTest_q;
    wire [0:0] norm_uid64_fpDivTest_b;
    wire [26:0] divValPreNormHigh_uid65_fpDivTest_in;
    wire [24:0] divValPreNormHigh_uid65_fpDivTest_b;
    wire [25:0] divValPreNormLow_uid66_fpDivTest_in;
    wire [24:0] divValPreNormLow_uid66_fpDivTest_b;
    wire [0:0] normFracRnd_uid67_fpDivTest_s;
    reg [24:0] normFracRnd_uid67_fpDivTest_q;
    wire [34:0] expFracRnd_uid68_fpDivTest_q;
    wire [23:0] zeroPaddingInAddition_uid74_fpDivTest_q;
    wire [25:0] expFracPostRnd_uid75_fpDivTest_q;
    wire [36:0] expFracPostRnd_uid76_fpDivTest_a;
    wire [36:0] expFracPostRnd_uid76_fpDivTest_b;
    logic [36:0] expFracPostRnd_uid76_fpDivTest_o;
    wire [35:0] expFracPostRnd_uid76_fpDivTest_q;
    wire [23:0] fracXExt_uid77_fpDivTest_q;
    wire [24:0] fracPostRndF_uid79_fpDivTest_in;
    wire [23:0] fracPostRndF_uid79_fpDivTest_b;
    wire [0:0] fracPostRndF_uid80_fpDivTest_s;
    reg [23:0] fracPostRndF_uid80_fpDivTest_q;
    wire [32:0] expPostRndFR_uid81_fpDivTest_in;
    wire [7:0] expPostRndFR_uid81_fpDivTest_b;
    wire [0:0] expPostRndF_uid82_fpDivTest_s;
    reg [7:0] expPostRndF_uid82_fpDivTest_q;
    wire [24:0] lOAdded_uid84_fpDivTest_q;
    wire [23:0] lOAdded_uid87_fpDivTest_q;
    wire [0:0] qDivProdNorm_uid90_fpDivTest_b;
    wire [47:0] qDivProdFracHigh_uid91_fpDivTest_in;
    wire [23:0] qDivProdFracHigh_uid91_fpDivTest_b;
    wire [46:0] qDivProdFracLow_uid92_fpDivTest_in;
    wire [23:0] qDivProdFracLow_uid92_fpDivTest_b;
    wire [0:0] qDivProdFrac_uid93_fpDivTest_s;
    reg [23:0] qDivProdFrac_uid93_fpDivTest_q;
    wire [8:0] qDivProdExp_opA_uid94_fpDivTest_a;
    wire [8:0] qDivProdExp_opA_uid94_fpDivTest_b;
    logic [8:0] qDivProdExp_opA_uid94_fpDivTest_o;
    wire [8:0] qDivProdExp_opA_uid94_fpDivTest_q;
    wire [8:0] qDivProdExp_opBs_uid95_fpDivTest_a;
    wire [8:0] qDivProdExp_opBs_uid95_fpDivTest_b;
    logic [8:0] qDivProdExp_opBs_uid95_fpDivTest_o;
    wire [8:0] qDivProdExp_opBs_uid95_fpDivTest_q;
    wire [11:0] qDivProdExp_uid96_fpDivTest_a;
    wire [11:0] qDivProdExp_uid96_fpDivTest_b;
    logic [11:0] qDivProdExp_uid96_fpDivTest_o;
    wire [10:0] qDivProdExp_uid96_fpDivTest_q;
    wire [22:0] qDivProdFracWF_uid97_fpDivTest_b;
    wire [7:0] qDivProdLTX_opA_uid98_fpDivTest_in;
    wire [7:0] qDivProdLTX_opA_uid98_fpDivTest_b;
    wire [30:0] qDivProdLTX_opA_uid99_fpDivTest_q;
    wire [30:0] qDivProdLTX_opB_uid100_fpDivTest_q;
    wire [32:0] qDividerProdLTX_uid101_fpDivTest_a;
    wire [32:0] qDividerProdLTX_uid101_fpDivTest_b;
    logic [32:0] qDividerProdLTX_uid101_fpDivTest_o;
    wire [0:0] qDividerProdLTX_uid101_fpDivTest_c;
    wire [0:0] betweenFPwF_uid102_fpDivTest_in;
    wire [0:0] betweenFPwF_uid102_fpDivTest_b;
    wire [0:0] extraUlp_uid103_fpDivTest_qi;
    reg [0:0] extraUlp_uid103_fpDivTest_q;
    wire [22:0] fracPostRndFT_uid104_fpDivTest_b;
    wire [23:0] fracRPreExcExt_uid105_fpDivTest_a;
    wire [23:0] fracRPreExcExt_uid105_fpDivTest_b;
    logic [23:0] fracRPreExcExt_uid105_fpDivTest_o;
    wire [23:0] fracRPreExcExt_uid105_fpDivTest_q;
    wire [22:0] fracPostRndFPostUlp_uid106_fpDivTest_in;
    wire [22:0] fracPostRndFPostUlp_uid106_fpDivTest_b;
    wire [0:0] fracRPreExc_uid107_fpDivTest_s;
    reg [22:0] fracRPreExc_uid107_fpDivTest_q;
    wire [0:0] ovfIncRnd_uid109_fpDivTest_b;
    wire [8:0] expFracPostRndInc_uid110_fpDivTest_a;
    wire [8:0] expFracPostRndInc_uid110_fpDivTest_b;
    logic [8:0] expFracPostRndInc_uid110_fpDivTest_o;
    wire [8:0] expFracPostRndInc_uid110_fpDivTest_q;
    wire [7:0] expFracPostRndR_uid111_fpDivTest_in;
    wire [7:0] expFracPostRndR_uid111_fpDivTest_b;
    wire [0:0] expRPreExc_uid112_fpDivTest_s;
    reg [7:0] expRPreExc_uid112_fpDivTest_q;
    wire [10:0] expRExt_uid114_fpDivTest_b;
    wire [12:0] expUdf_uid115_fpDivTest_a;
    wire [12:0] expUdf_uid115_fpDivTest_b;
    logic [12:0] expUdf_uid115_fpDivTest_o;
    wire [0:0] expUdf_uid115_fpDivTest_n;
    wire [12:0] expOvf_uid118_fpDivTest_a;
    wire [12:0] expOvf_uid118_fpDivTest_b;
    logic [12:0] expOvf_uid118_fpDivTest_o;
    wire [0:0] expOvf_uid118_fpDivTest_n;
    wire [0:0] zeroOverReg_uid119_fpDivTest_q;
    wire [0:0] regOverRegWithUf_uid120_fpDivTest_q;
    wire [0:0] xRegOrZero_uid121_fpDivTest_q;
    wire [0:0] regOrZeroOverInf_uid122_fpDivTest_q;
    wire [0:0] excRZero_uid123_fpDivTest_q;
    wire [0:0] excXRYZ_uid124_fpDivTest_q;
    wire [0:0] excXRYROvf_uid125_fpDivTest_q;
    wire [0:0] excXIYZ_uid126_fpDivTest_q;
    wire [0:0] excXIYR_uid127_fpDivTest_q;
    wire [0:0] excRInf_uid128_fpDivTest_q;
    wire [0:0] excXZYZ_uid129_fpDivTest_q;
    wire [0:0] excXIYI_uid130_fpDivTest_q;
    wire [0:0] excRNaN_uid131_fpDivTest_q;
    wire [2:0] concExc_uid132_fpDivTest_q;
    reg [1:0] excREnc_uid133_fpDivTest_q;
    wire [22:0] oneFracRPostExc2_uid134_fpDivTest_q;
    wire [1:0] fracRPostExc_uid137_fpDivTest_s;
    reg [22:0] fracRPostExc_uid137_fpDivTest_q;
    wire [1:0] expRPostExc_uid141_fpDivTest_s;
    reg [7:0] expRPostExc_uid141_fpDivTest_q;
    wire [0:0] invExcRNaN_uid142_fpDivTest_q;
    wire [0:0] sRPostExc_uid143_fpDivTest_qi;
    reg [0:0] sRPostExc_uid143_fpDivTest_q;
    wire [31:0] divR_uid144_fpDivTest_q;
    wire [12:0] yT1_uid158_invPolyEval_b;
    wire [0:0] lowRangeB_uid160_invPolyEval_in;
    wire [0:0] lowRangeB_uid160_invPolyEval_b;
    wire [12:0] highBBits_uid161_invPolyEval_b;
    wire [22:0] s1sumAHighB_uid162_invPolyEval_a;
    wire [22:0] s1sumAHighB_uid162_invPolyEval_b;
    logic [22:0] s1sumAHighB_uid162_invPolyEval_o;
    wire [22:0] s1sumAHighB_uid162_invPolyEval_q;
    wire [23:0] s1_uid163_invPolyEval_q;
    wire [1:0] lowRangeB_uid166_invPolyEval_in;
    wire [1:0] lowRangeB_uid166_invPolyEval_b;
    wire [22:0] highBBits_uid167_invPolyEval_b;
    wire [32:0] s2sumAHighB_uid168_invPolyEval_a;
    wire [32:0] s2sumAHighB_uid168_invPolyEval_b;
    logic [32:0] s2sumAHighB_uid168_invPolyEval_o;
    wire [32:0] s2sumAHighB_uid168_invPolyEval_q;
    wire [34:0] s2_uid169_invPolyEval_q;
    wire [27:0] osig_uid172_divValPreNorm_uid59_fpDivTest_b;
    wire [13:0] osig_uid175_pT1_uid159_invPolyEval_b;
    wire [24:0] osig_uid178_pT2_uid165_invPolyEval_b;
    wire memoryC0_uid146_invTables_lutmem_reset0;
    wire [31:0] memoryC0_uid146_invTables_lutmem_ia;
    wire [8:0] memoryC0_uid146_invTables_lutmem_aa;
    wire [8:0] memoryC0_uid146_invTables_lutmem_ab;
    wire [31:0] memoryC0_uid146_invTables_lutmem_ir;
    wire [31:0] memoryC0_uid146_invTables_lutmem_r;
    wire memoryC0_uid146_invTables_lutmem_enaOr_rst;
    wire memoryC1_uid149_invTables_lutmem_reset0;
    wire [21:0] memoryC1_uid149_invTables_lutmem_ia;
    wire [8:0] memoryC1_uid149_invTables_lutmem_aa;
    wire [8:0] memoryC1_uid149_invTables_lutmem_ab;
    wire [21:0] memoryC1_uid149_invTables_lutmem_ir;
    wire [21:0] memoryC1_uid149_invTables_lutmem_r;
    wire memoryC1_uid149_invTables_lutmem_enaOr_rst;
    wire memoryC2_uid152_invTables_lutmem_reset0;
    wire [12:0] memoryC2_uid152_invTables_lutmem_ia;
    wire [8:0] memoryC2_uid152_invTables_lutmem_aa;
    wire [8:0] memoryC2_uid152_invTables_lutmem_ab;
    wire [12:0] memoryC2_uid152_invTables_lutmem_ir;
    wire [12:0] memoryC2_uid152_invTables_lutmem_r;
    wire memoryC2_uid152_invTables_lutmem_enaOr_rst;
    wire qDivProd_uid89_fpDivTest_cma_reset;
    (* preserve_syn_only *) reg [24:0] qDivProd_uid89_fpDivTest_cma_ah [0:0];
    (* preserve_syn_only *) reg [23:0] qDivProd_uid89_fpDivTest_cma_ch [0:0];
    wire [24:0] qDivProd_uid89_fpDivTest_cma_a0;
    wire [23:0] qDivProd_uid89_fpDivTest_cma_c0;
    wire [48:0] qDivProd_uid89_fpDivTest_cma_s0;
    wire [48:0] qDivProd_uid89_fpDivTest_cma_qq;
    reg [48:0] qDivProd_uid89_fpDivTest_cma_q;
    wire qDivProd_uid89_fpDivTest_cma_ena0;
    wire qDivProd_uid89_fpDivTest_cma_ena1;
    wire qDivProd_uid89_fpDivTest_cma_ena2;
    wire prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_reset;
    (* preserve_syn_only *) reg [26:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ah [0:0];
    (* preserve_syn_only *) reg [23:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ch [0:0];
    wire [26:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_a0;
    wire [23:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_c0;
    wire [50:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_s0;
    wire [50:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_qq;
    reg [50:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_q;
    wire prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena0;
    wire prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena1;
    wire prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena2;
    wire prodXY_uid174_pT1_uid159_invPolyEval_cma_reset;
    (* preserve_syn_only *) reg [12:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_ah [0:0];
    (* preserve_syn_only *) reg signed [12:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_ch [0:0];
    wire [12:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_a0;
    wire [12:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_c0;
    wire [25:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_s0;
    wire [25:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_qq;
    reg [25:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_q;
    wire prodXY_uid174_pT1_uid159_invPolyEval_cma_ena0;
    wire prodXY_uid174_pT1_uid159_invPolyEval_cma_ena1;
    wire prodXY_uid174_pT1_uid159_invPolyEval_cma_ena2;
    wire prodXY_uid177_pT2_uid165_invPolyEval_cma_reset;
    (* preserve_syn_only *) reg [13:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_ah [0:0];
    (* preserve_syn_only *) reg signed [23:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_ch [0:0];
    wire [13:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_a0;
    wire [23:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_c0;
    wire [37:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_s0;
    wire [37:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_qq;
    reg [37:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_q;
    wire prodXY_uid177_pT2_uid165_invPolyEval_cma_ena0;
    wire prodXY_uid177_pT2_uid165_invPolyEval_cma_ena1;
    wire prodXY_uid177_pT2_uid165_invPolyEval_cma_ena2;
    reg [12:0] redist0_memoryC2_uid152_invTables_lutmem_r_1_q;
    reg [0:0] redist1_lowRangeB_uid160_invPolyEval_b_1_q;
    reg [0:0] redist2_sRPostExc_uid143_fpDivTest_q_9_q;
    reg [1:0] redist3_excREnc_uid133_fpDivTest_q_9_q;
    reg [0:0] redist5_betweenFPwF_uid102_fpDivTest_b_7_q;
    reg [7:0] redist6_qDivProdLTX_opA_uid98_fpDivTest_b_1_q;
    reg [22:0] redist7_qDivProdFracWF_uid97_fpDivTest_b_1_q;
    reg [7:0] redist9_expPostRndFR_uid81_fpDivTest_b_9_q;
    reg [7:0] redist9_expPostRndFR_uid81_fpDivTest_b_9_delay_0;
    reg [23:0] redist10_fracPostRndF_uid79_fpDivTest_b_1_q;
    reg [0:0] redist11_norm_uid64_fpDivTest_b_1_q;
    reg [0:0] redist13_invYO_uid55_fpDivTest_b_9_q;
    reg [0:0] redist14_invYO_uid55_fpDivTest_b_15_q;
    reg [26:0] redist15_invY_uid54_fpDivTest_b_1_q;
    reg [13:0] redist16_yPE_uid52_fpDivTest_b_3_q;
    reg [13:0] redist16_yPE_uid52_fpDivTest_b_3_delay_0;
    reg [13:0] redist16_yPE_uid52_fpDivTest_b_3_delay_1;
    reg [0:0] redist20_signR_uid46_fpDivTest_q_25_q;
    reg [0:0] redist21_expXIsMax_uid24_fpDivTest_q_1_q;
    reg [0:0] redist22_excZ_x_uid23_fpDivTest_q_1_q;
    reg [22:0] redist24_fracY_uid13_fpDivTest_b_24_q;
    reg [22:0] redist24_fracY_uid13_fpDivTest_b_24_delay_0;
    reg [22:0] redist25_fracY_uid13_fpDivTest_b_25_q;
    reg [7:0] redist27_expY_uid12_fpDivTest_b_24_q;
    reg [22:0] redist31_fracX_uid10_fpDivTest_b_25_q;
    reg [7:0] redist34_expX_uid9_fpDivTest_b_24_q;
    reg [7:0] redist36_expX_uid9_fpDivTest_b_32_q;
    wire redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_reset0;
    wire [22:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_ia;
    wire [2:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_aa;
    wire [2:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_ab;
    wire [22:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_iq;
    wire [22:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_q;
    wire redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_enaOr_rst;
    wire [2:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt_q;
    (* preserve_syn_only *) reg [2:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt_i;
    (* preserve_syn_only *) reg redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt_eq;
    wire [0:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdmux_s;
    reg [2:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdmux_q;
    reg [2:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_wraddr_q;
    wire [3:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_last_q;
    wire [3:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_cmp_b;
    wire [0:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_cmp_q;
    reg [0:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_cmpReg_q;
    wire [0:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_notEnable_q;
    wire [0:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_nor_q;
    (* preserve_syn_only *) reg [0:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_sticky_ena_q;
    wire [0:0] redist4_fracPostRndFT_uid104_fpDivTest_b_8_enaAnd_q;
    reg [7:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_outputreg0_q;
    wire redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_reset0;
    wire [7:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_ia;
    wire [2:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_aa;
    wire [2:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_ab;
    wire [7:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_iq;
    wire [7:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_q;
    wire redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_enaOr_rst;
    wire [2:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt_q;
    (* preserve_syn_only *) reg [2:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt_i;
    (* preserve_syn_only *) reg redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt_eq;
    wire [0:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_rdmux_s;
    reg [2:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_rdmux_q;
    reg [2:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_wraddr_q;
    wire [2:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_last_q;
    wire [0:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_cmp_q;
    reg [0:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_cmpReg_q;
    wire [0:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_notEnable_q;
    wire [0:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_nor_q;
    (* preserve_syn_only *) reg [0:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_sticky_ena_q;
    wire [0:0] redist8_expPostRndFR_uid81_fpDivTest_b_7_enaAnd_q;
    wire redist12_lOAdded_uid57_fpDivTest_q_6_mem_reset0;
    wire [23:0] redist12_lOAdded_uid57_fpDivTest_q_6_mem_ia;
    wire [2:0] redist12_lOAdded_uid57_fpDivTest_q_6_mem_aa;
    wire [2:0] redist12_lOAdded_uid57_fpDivTest_q_6_mem_ab;
    wire [23:0] redist12_lOAdded_uid57_fpDivTest_q_6_mem_iq;
    wire [23:0] redist12_lOAdded_uid57_fpDivTest_q_6_mem_q;
    wire redist12_lOAdded_uid57_fpDivTest_q_6_mem_enaOr_rst;
    wire [2:0] redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt_q;
    (* preserve_syn_only *) reg [2:0] redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt_i;
    (* preserve_syn_only *) reg redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt_eq;
    wire [0:0] redist12_lOAdded_uid57_fpDivTest_q_6_rdmux_s;
    reg [2:0] redist12_lOAdded_uid57_fpDivTest_q_6_rdmux_q;
    reg [2:0] redist12_lOAdded_uid57_fpDivTest_q_6_wraddr_q;
    wire [2:0] redist12_lOAdded_uid57_fpDivTest_q_6_mem_last_q;
    wire [0:0] redist12_lOAdded_uid57_fpDivTest_q_6_cmp_q;
    reg [0:0] redist12_lOAdded_uid57_fpDivTest_q_6_cmpReg_q;
    wire [0:0] redist12_lOAdded_uid57_fpDivTest_q_6_notEnable_q;
    wire [0:0] redist12_lOAdded_uid57_fpDivTest_q_6_nor_q;
    (* preserve_syn_only *) reg [0:0] redist12_lOAdded_uid57_fpDivTest_q_6_sticky_ena_q;
    wire [0:0] redist12_lOAdded_uid57_fpDivTest_q_6_enaAnd_q;
    reg [13:0] redist17_yPE_uid52_fpDivTest_b_10_outputreg0_q;
    wire redist17_yPE_uid52_fpDivTest_b_10_mem_reset0;
    wire [13:0] redist17_yPE_uid52_fpDivTest_b_10_mem_ia;
    wire [2:0] redist17_yPE_uid52_fpDivTest_b_10_mem_aa;
    wire [2:0] redist17_yPE_uid52_fpDivTest_b_10_mem_ab;
    wire [13:0] redist17_yPE_uid52_fpDivTest_b_10_mem_iq;
    wire [13:0] redist17_yPE_uid52_fpDivTest_b_10_mem_q;
    wire redist17_yPE_uid52_fpDivTest_b_10_mem_enaOr_rst;
    wire [2:0] redist17_yPE_uid52_fpDivTest_b_10_rdcnt_q;
    (* preserve_syn_only *) reg [2:0] redist17_yPE_uid52_fpDivTest_b_10_rdcnt_i;
    (* preserve_syn_only *) reg redist17_yPE_uid52_fpDivTest_b_10_rdcnt_eq;
    wire [0:0] redist17_yPE_uid52_fpDivTest_b_10_rdmux_s;
    reg [2:0] redist17_yPE_uid52_fpDivTest_b_10_rdmux_q;
    reg [2:0] redist17_yPE_uid52_fpDivTest_b_10_wraddr_q;
    wire [2:0] redist17_yPE_uid52_fpDivTest_b_10_mem_last_q;
    wire [0:0] redist17_yPE_uid52_fpDivTest_b_10_cmp_q;
    reg [0:0] redist17_yPE_uid52_fpDivTest_b_10_cmpReg_q;
    wire [0:0] redist17_yPE_uid52_fpDivTest_b_10_notEnable_q;
    wire [0:0] redist17_yPE_uid52_fpDivTest_b_10_nor_q;
    (* preserve_syn_only *) reg [0:0] redist17_yPE_uid52_fpDivTest_b_10_sticky_ena_q;
    wire [0:0] redist17_yPE_uid52_fpDivTest_b_10_enaAnd_q;
    reg [8:0] redist18_yAddr_uid51_fpDivTest_b_7_outputreg0_q;
    wire redist18_yAddr_uid51_fpDivTest_b_7_mem_reset0;
    wire [8:0] redist18_yAddr_uid51_fpDivTest_b_7_mem_ia;
    wire [2:0] redist18_yAddr_uid51_fpDivTest_b_7_mem_aa;
    wire [2:0] redist18_yAddr_uid51_fpDivTest_b_7_mem_ab;
    wire [8:0] redist18_yAddr_uid51_fpDivTest_b_7_mem_iq;
    wire [8:0] redist18_yAddr_uid51_fpDivTest_b_7_mem_q;
    wire redist18_yAddr_uid51_fpDivTest_b_7_mem_enaOr_rst;
    wire [2:0] redist18_yAddr_uid51_fpDivTest_b_7_rdcnt_q;
    (* preserve_syn_only *) reg [2:0] redist18_yAddr_uid51_fpDivTest_b_7_rdcnt_i;
    (* preserve_syn_only *) reg redist18_yAddr_uid51_fpDivTest_b_7_rdcnt_eq;
    wire [0:0] redist18_yAddr_uid51_fpDivTest_b_7_rdmux_s;
    reg [2:0] redist18_yAddr_uid51_fpDivTest_b_7_rdmux_q;
    reg [2:0] redist18_yAddr_uid51_fpDivTest_b_7_wraddr_q;
    wire [2:0] redist18_yAddr_uid51_fpDivTest_b_7_mem_last_q;
    wire [0:0] redist18_yAddr_uid51_fpDivTest_b_7_cmp_q;
    reg [0:0] redist18_yAddr_uid51_fpDivTest_b_7_cmpReg_q;
    wire [0:0] redist18_yAddr_uid51_fpDivTest_b_7_notEnable_q;
    wire [0:0] redist18_yAddr_uid51_fpDivTest_b_7_nor_q;
    (* preserve_syn_only *) reg [0:0] redist18_yAddr_uid51_fpDivTest_b_7_sticky_ena_q;
    wire [0:0] redist18_yAddr_uid51_fpDivTest_b_7_enaAnd_q;
    reg [8:0] redist19_yAddr_uid51_fpDivTest_b_14_outputreg0_q;
    wire redist19_yAddr_uid51_fpDivTest_b_14_mem_reset0;
    wire [8:0] redist19_yAddr_uid51_fpDivTest_b_14_mem_ia;
    wire [2:0] redist19_yAddr_uid51_fpDivTest_b_14_mem_aa;
    wire [2:0] redist19_yAddr_uid51_fpDivTest_b_14_mem_ab;
    wire [8:0] redist19_yAddr_uid51_fpDivTest_b_14_mem_iq;
    wire [8:0] redist19_yAddr_uid51_fpDivTest_b_14_mem_q;
    wire redist19_yAddr_uid51_fpDivTest_b_14_mem_enaOr_rst;
    wire [2:0] redist19_yAddr_uid51_fpDivTest_b_14_rdcnt_q;
    (* preserve_syn_only *) reg [2:0] redist19_yAddr_uid51_fpDivTest_b_14_rdcnt_i;
    (* preserve_syn_only *) reg redist19_yAddr_uid51_fpDivTest_b_14_rdcnt_eq;
    wire [0:0] redist19_yAddr_uid51_fpDivTest_b_14_rdmux_s;
    reg [2:0] redist19_yAddr_uid51_fpDivTest_b_14_rdmux_q;
    reg [2:0] redist19_yAddr_uid51_fpDivTest_b_14_wraddr_q;
    wire [2:0] redist19_yAddr_uid51_fpDivTest_b_14_mem_last_q;
    wire [0:0] redist19_yAddr_uid51_fpDivTest_b_14_cmp_q;
    reg [0:0] redist19_yAddr_uid51_fpDivTest_b_14_cmpReg_q;
    wire [0:0] redist19_yAddr_uid51_fpDivTest_b_14_notEnable_q;
    wire [0:0] redist19_yAddr_uid51_fpDivTest_b_14_nor_q;
    (* preserve_syn_only *) reg [0:0] redist19_yAddr_uid51_fpDivTest_b_14_sticky_ena_q;
    wire [0:0] redist19_yAddr_uid51_fpDivTest_b_14_enaAnd_q;
    wire redist23_fracY_uid13_fpDivTest_b_22_mem_reset0;
    wire [22:0] redist23_fracY_uid13_fpDivTest_b_22_mem_ia;
    wire [4:0] redist23_fracY_uid13_fpDivTest_b_22_mem_aa;
    wire [4:0] redist23_fracY_uid13_fpDivTest_b_22_mem_ab;
    wire [22:0] redist23_fracY_uid13_fpDivTest_b_22_mem_iq;
    wire [22:0] redist23_fracY_uid13_fpDivTest_b_22_mem_q;
    wire redist23_fracY_uid13_fpDivTest_b_22_mem_enaOr_rst;
    wire [4:0] redist23_fracY_uid13_fpDivTest_b_22_rdcnt_q;
    (* preserve_syn_only *) reg [4:0] redist23_fracY_uid13_fpDivTest_b_22_rdcnt_i;
    (* preserve_syn_only *) reg redist23_fracY_uid13_fpDivTest_b_22_rdcnt_eq;
    wire [0:0] redist23_fracY_uid13_fpDivTest_b_22_rdmux_s;
    reg [4:0] redist23_fracY_uid13_fpDivTest_b_22_rdmux_q;
    reg [4:0] redist23_fracY_uid13_fpDivTest_b_22_wraddr_q;
    wire [5:0] redist23_fracY_uid13_fpDivTest_b_22_mem_last_q;
    wire [5:0] redist23_fracY_uid13_fpDivTest_b_22_cmp_b;
    wire [0:0] redist23_fracY_uid13_fpDivTest_b_22_cmp_q;
    reg [0:0] redist23_fracY_uid13_fpDivTest_b_22_cmpReg_q;
    wire [0:0] redist23_fracY_uid13_fpDivTest_b_22_notEnable_q;
    wire [0:0] redist23_fracY_uid13_fpDivTest_b_22_nor_q;
    (* preserve_syn_only *) reg [0:0] redist23_fracY_uid13_fpDivTest_b_22_sticky_ena_q;
    wire [0:0] redist23_fracY_uid13_fpDivTest_b_22_enaAnd_q;
    wire redist26_expY_uid12_fpDivTest_b_23_mem_reset0;
    wire [7:0] redist26_expY_uid12_fpDivTest_b_23_mem_ia;
    wire [4:0] redist26_expY_uid12_fpDivTest_b_23_mem_aa;
    wire [4:0] redist26_expY_uid12_fpDivTest_b_23_mem_ab;
    wire [7:0] redist26_expY_uid12_fpDivTest_b_23_mem_iq;
    wire [7:0] redist26_expY_uid12_fpDivTest_b_23_mem_q;
    wire redist26_expY_uid12_fpDivTest_b_23_mem_enaOr_rst;
    wire [4:0] redist26_expY_uid12_fpDivTest_b_23_rdcnt_q;
    (* preserve_syn_only *) reg [4:0] redist26_expY_uid12_fpDivTest_b_23_rdcnt_i;
    (* preserve_syn_only *) reg redist26_expY_uid12_fpDivTest_b_23_rdcnt_eq;
    wire [0:0] redist26_expY_uid12_fpDivTest_b_23_rdmux_s;
    reg [4:0] redist26_expY_uid12_fpDivTest_b_23_rdmux_q;
    reg [4:0] redist26_expY_uid12_fpDivTest_b_23_wraddr_q;
    wire [5:0] redist26_expY_uid12_fpDivTest_b_23_mem_last_q;
    wire [5:0] redist26_expY_uid12_fpDivTest_b_23_cmp_b;
    wire [0:0] redist26_expY_uid12_fpDivTest_b_23_cmp_q;
    reg [0:0] redist26_expY_uid12_fpDivTest_b_23_cmpReg_q;
    wire [0:0] redist26_expY_uid12_fpDivTest_b_23_notEnable_q;
    wire [0:0] redist26_expY_uid12_fpDivTest_b_23_nor_q;
    (* preserve_syn_only *) reg [0:0] redist26_expY_uid12_fpDivTest_b_23_sticky_ena_q;
    wire [0:0] redist26_expY_uid12_fpDivTest_b_23_enaAnd_q;
    reg [7:0] redist28_expY_uid12_fpDivTest_b_31_outputreg0_q;
    wire redist28_expY_uid12_fpDivTest_b_31_mem_reset0;
    wire [7:0] redist28_expY_uid12_fpDivTest_b_31_mem_ia;
    wire [2:0] redist28_expY_uid12_fpDivTest_b_31_mem_aa;
    wire [2:0] redist28_expY_uid12_fpDivTest_b_31_mem_ab;
    wire [7:0] redist28_expY_uid12_fpDivTest_b_31_mem_iq;
    wire [7:0] redist28_expY_uid12_fpDivTest_b_31_mem_q;
    wire redist28_expY_uid12_fpDivTest_b_31_mem_enaOr_rst;
    wire [2:0] redist28_expY_uid12_fpDivTest_b_31_rdcnt_q;
    (* preserve_syn_only *) reg [2:0] redist28_expY_uid12_fpDivTest_b_31_rdcnt_i;
    (* preserve_syn_only *) reg redist28_expY_uid12_fpDivTest_b_31_rdcnt_eq;
    wire [0:0] redist28_expY_uid12_fpDivTest_b_31_rdmux_s;
    reg [2:0] redist28_expY_uid12_fpDivTest_b_31_rdmux_q;
    reg [2:0] redist28_expY_uid12_fpDivTest_b_31_wraddr_q;
    wire [2:0] redist28_expY_uid12_fpDivTest_b_31_mem_last_q;
    wire [0:0] redist28_expY_uid12_fpDivTest_b_31_cmp_q;
    reg [0:0] redist28_expY_uid12_fpDivTest_b_31_cmpReg_q;
    wire [0:0] redist28_expY_uid12_fpDivTest_b_31_notEnable_q;
    wire [0:0] redist28_expY_uid12_fpDivTest_b_31_nor_q;
    (* preserve_syn_only *) reg [0:0] redist28_expY_uid12_fpDivTest_b_31_sticky_ena_q;
    wire [0:0] redist28_expY_uid12_fpDivTest_b_31_enaAnd_q;
    reg [22:0] redist29_fracX_uid10_fpDivTest_b_17_outputreg0_q;
    wire redist29_fracX_uid10_fpDivTest_b_17_mem_reset0;
    wire [22:0] redist29_fracX_uid10_fpDivTest_b_17_mem_ia;
    wire [3:0] redist29_fracX_uid10_fpDivTest_b_17_mem_aa;
    wire [3:0] redist29_fracX_uid10_fpDivTest_b_17_mem_ab;
    wire [22:0] redist29_fracX_uid10_fpDivTest_b_17_mem_iq;
    wire [22:0] redist29_fracX_uid10_fpDivTest_b_17_mem_q;
    wire redist29_fracX_uid10_fpDivTest_b_17_mem_enaOr_rst;
    wire [3:0] redist29_fracX_uid10_fpDivTest_b_17_rdcnt_q;
    (* preserve_syn_only *) reg [3:0] redist29_fracX_uid10_fpDivTest_b_17_rdcnt_i;
    (* preserve_syn_only *) reg redist29_fracX_uid10_fpDivTest_b_17_rdcnt_eq;
    wire [0:0] redist29_fracX_uid10_fpDivTest_b_17_rdmux_s;
    reg [3:0] redist29_fracX_uid10_fpDivTest_b_17_rdmux_q;
    reg [3:0] redist29_fracX_uid10_fpDivTest_b_17_wraddr_q;
    wire [4:0] redist29_fracX_uid10_fpDivTest_b_17_mem_last_q;
    wire [4:0] redist29_fracX_uid10_fpDivTest_b_17_cmp_b;
    wire [0:0] redist29_fracX_uid10_fpDivTest_b_17_cmp_q;
    reg [0:0] redist29_fracX_uid10_fpDivTest_b_17_cmpReg_q;
    wire [0:0] redist29_fracX_uid10_fpDivTest_b_17_notEnable_q;
    wire [0:0] redist29_fracX_uid10_fpDivTest_b_17_nor_q;
    (* preserve_syn_only *) reg [0:0] redist29_fracX_uid10_fpDivTest_b_17_sticky_ena_q;
    wire [0:0] redist29_fracX_uid10_fpDivTest_b_17_enaAnd_q;
    wire redist30_fracX_uid10_fpDivTest_b_24_mem_reset0;
    wire [22:0] redist30_fracX_uid10_fpDivTest_b_24_mem_ia;
    wire [2:0] redist30_fracX_uid10_fpDivTest_b_24_mem_aa;
    wire [2:0] redist30_fracX_uid10_fpDivTest_b_24_mem_ab;
    wire [22:0] redist30_fracX_uid10_fpDivTest_b_24_mem_iq;
    wire [22:0] redist30_fracX_uid10_fpDivTest_b_24_mem_q;
    wire redist30_fracX_uid10_fpDivTest_b_24_mem_enaOr_rst;
    wire [2:0] redist30_fracX_uid10_fpDivTest_b_24_rdcnt_q;
    (* preserve_syn_only *) reg [2:0] redist30_fracX_uid10_fpDivTest_b_24_rdcnt_i;
    (* preserve_syn_only *) reg redist30_fracX_uid10_fpDivTest_b_24_rdcnt_eq;
    wire [0:0] redist30_fracX_uid10_fpDivTest_b_24_rdmux_s;
    reg [2:0] redist30_fracX_uid10_fpDivTest_b_24_rdmux_q;
    reg [2:0] redist30_fracX_uid10_fpDivTest_b_24_wraddr_q;
    wire [3:0] redist30_fracX_uid10_fpDivTest_b_24_mem_last_q;
    wire [3:0] redist30_fracX_uid10_fpDivTest_b_24_cmp_b;
    wire [0:0] redist30_fracX_uid10_fpDivTest_b_24_cmp_q;
    reg [0:0] redist30_fracX_uid10_fpDivTest_b_24_cmpReg_q;
    wire [0:0] redist30_fracX_uid10_fpDivTest_b_24_notEnable_q;
    wire [0:0] redist30_fracX_uid10_fpDivTest_b_24_nor_q;
    (* preserve_syn_only *) reg [0:0] redist30_fracX_uid10_fpDivTest_b_24_sticky_ena_q;
    wire [0:0] redist30_fracX_uid10_fpDivTest_b_24_enaAnd_q;
    wire redist32_fracX_uid10_fpDivTest_b_32_mem_reset0;
    wire [22:0] redist32_fracX_uid10_fpDivTest_b_32_mem_ia;
    wire [2:0] redist32_fracX_uid10_fpDivTest_b_32_mem_aa;
    wire [2:0] redist32_fracX_uid10_fpDivTest_b_32_mem_ab;
    wire [22:0] redist32_fracX_uid10_fpDivTest_b_32_mem_iq;
    wire [22:0] redist32_fracX_uid10_fpDivTest_b_32_mem_q;
    wire redist32_fracX_uid10_fpDivTest_b_32_mem_enaOr_rst;
    wire [2:0] redist32_fracX_uid10_fpDivTest_b_32_rdcnt_q;
    (* preserve_syn_only *) reg [2:0] redist32_fracX_uid10_fpDivTest_b_32_rdcnt_i;
    (* preserve_syn_only *) reg redist32_fracX_uid10_fpDivTest_b_32_rdcnt_eq;
    wire [0:0] redist32_fracX_uid10_fpDivTest_b_32_rdmux_s;
    reg [2:0] redist32_fracX_uid10_fpDivTest_b_32_rdmux_q;
    reg [2:0] redist32_fracX_uid10_fpDivTest_b_32_wraddr_q;
    wire [3:0] redist32_fracX_uid10_fpDivTest_b_32_mem_last_q;
    wire [3:0] redist32_fracX_uid10_fpDivTest_b_32_cmp_b;
    wire [0:0] redist32_fracX_uid10_fpDivTest_b_32_cmp_q;
    reg [0:0] redist32_fracX_uid10_fpDivTest_b_32_cmpReg_q;
    wire [0:0] redist32_fracX_uid10_fpDivTest_b_32_notEnable_q;
    wire [0:0] redist32_fracX_uid10_fpDivTest_b_32_nor_q;
    (* preserve_syn_only *) reg [0:0] redist32_fracX_uid10_fpDivTest_b_32_sticky_ena_q;
    wire [0:0] redist32_fracX_uid10_fpDivTest_b_32_enaAnd_q;
    wire redist33_expX_uid9_fpDivTest_b_23_mem_reset0;
    wire [7:0] redist33_expX_uid9_fpDivTest_b_23_mem_ia;
    wire [4:0] redist33_expX_uid9_fpDivTest_b_23_mem_aa;
    wire [4:0] redist33_expX_uid9_fpDivTest_b_23_mem_ab;
    wire [7:0] redist33_expX_uid9_fpDivTest_b_23_mem_iq;
    wire [7:0] redist33_expX_uid9_fpDivTest_b_23_mem_q;
    wire redist33_expX_uid9_fpDivTest_b_23_mem_enaOr_rst;
    wire [4:0] redist33_expX_uid9_fpDivTest_b_23_rdcnt_q;
    (* preserve_syn_only *) reg [4:0] redist33_expX_uid9_fpDivTest_b_23_rdcnt_i;
    (* preserve_syn_only *) reg redist33_expX_uid9_fpDivTest_b_23_rdcnt_eq;
    wire [0:0] redist33_expX_uid9_fpDivTest_b_23_rdmux_s;
    reg [4:0] redist33_expX_uid9_fpDivTest_b_23_rdmux_q;
    reg [4:0] redist33_expX_uid9_fpDivTest_b_23_wraddr_q;
    wire [5:0] redist33_expX_uid9_fpDivTest_b_23_mem_last_q;
    wire [5:0] redist33_expX_uid9_fpDivTest_b_23_cmp_b;
    wire [0:0] redist33_expX_uid9_fpDivTest_b_23_cmp_q;
    reg [0:0] redist33_expX_uid9_fpDivTest_b_23_cmpReg_q;
    wire [0:0] redist33_expX_uid9_fpDivTest_b_23_notEnable_q;
    wire [0:0] redist33_expX_uid9_fpDivTest_b_23_nor_q;
    (* preserve_syn_only *) reg [0:0] redist33_expX_uid9_fpDivTest_b_23_sticky_ena_q;
    wire [0:0] redist33_expX_uid9_fpDivTest_b_23_enaAnd_q;
    reg [7:0] redist35_expX_uid9_fpDivTest_b_31_outputreg0_q;
    wire redist35_expX_uid9_fpDivTest_b_31_mem_reset0;
    wire [7:0] redist35_expX_uid9_fpDivTest_b_31_mem_ia;
    wire [2:0] redist35_expX_uid9_fpDivTest_b_31_mem_aa;
    wire [2:0] redist35_expX_uid9_fpDivTest_b_31_mem_ab;
    wire [7:0] redist35_expX_uid9_fpDivTest_b_31_mem_iq;
    wire [7:0] redist35_expX_uid9_fpDivTest_b_31_mem_q;
    wire redist35_expX_uid9_fpDivTest_b_31_mem_enaOr_rst;
    wire [2:0] redist35_expX_uid9_fpDivTest_b_31_rdcnt_q;
    (* preserve_syn_only *) reg [2:0] redist35_expX_uid9_fpDivTest_b_31_rdcnt_i;
    (* preserve_syn_only *) reg redist35_expX_uid9_fpDivTest_b_31_rdcnt_eq;
    wire [0:0] redist35_expX_uid9_fpDivTest_b_31_rdmux_s;
    reg [2:0] redist35_expX_uid9_fpDivTest_b_31_rdmux_q;
    reg [2:0] redist35_expX_uid9_fpDivTest_b_31_wraddr_q;
    wire [2:0] redist35_expX_uid9_fpDivTest_b_31_mem_last_q;
    wire [0:0] redist35_expX_uid9_fpDivTest_b_31_cmp_q;
    reg [0:0] redist35_expX_uid9_fpDivTest_b_31_cmpReg_q;
    wire [0:0] redist35_expX_uid9_fpDivTest_b_31_notEnable_q;
    wire [0:0] redist35_expX_uid9_fpDivTest_b_31_nor_q;
    (* preserve_syn_only *) reg [0:0] redist35_expX_uid9_fpDivTest_b_31_sticky_ena_q;
    wire [0:0] redist35_expX_uid9_fpDivTest_b_31_enaAnd_q;


    // redist23_fracY_uid13_fpDivTest_b_22_notEnable(LOGICAL,300)
    assign redist23_fracY_uid13_fpDivTest_b_22_notEnable_q = ~ (en);

    // redist23_fracY_uid13_fpDivTest_b_22_nor(LOGICAL,301)
    assign redist23_fracY_uid13_fpDivTest_b_22_nor_q = ~ (redist23_fracY_uid13_fpDivTest_b_22_notEnable_q | redist23_fracY_uid13_fpDivTest_b_22_sticky_ena_q);

    // redist23_fracY_uid13_fpDivTest_b_22_mem_last(CONSTANT,297)
    assign redist23_fracY_uid13_fpDivTest_b_22_mem_last_q = 6'b010011;

    // redist23_fracY_uid13_fpDivTest_b_22_cmp(LOGICAL,298)
    assign redist23_fracY_uid13_fpDivTest_b_22_cmp_b = {1'b0, redist23_fracY_uid13_fpDivTest_b_22_rdmux_q};
    assign redist23_fracY_uid13_fpDivTest_b_22_cmp_q = redist23_fracY_uid13_fpDivTest_b_22_mem_last_q == redist23_fracY_uid13_fpDivTest_b_22_cmp_b ? 1'b1 : 1'b0;

    // redist23_fracY_uid13_fpDivTest_b_22_cmpReg(REG,299)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist23_fracY_uid13_fpDivTest_b_22_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist23_fracY_uid13_fpDivTest_b_22_cmpReg_q <= redist23_fracY_uid13_fpDivTest_b_22_cmp_q;
        end
    end

    // redist23_fracY_uid13_fpDivTest_b_22_sticky_ena(REG,302)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist23_fracY_uid13_fpDivTest_b_22_sticky_ena_q <= 1'b0;
        end
        else if (redist23_fracY_uid13_fpDivTest_b_22_nor_q == 1'b1)
        begin
            redist23_fracY_uid13_fpDivTest_b_22_sticky_ena_q <= redist23_fracY_uid13_fpDivTest_b_22_cmpReg_q;
        end
    end

    // redist23_fracY_uid13_fpDivTest_b_22_enaAnd(LOGICAL,303)
    assign redist23_fracY_uid13_fpDivTest_b_22_enaAnd_q = redist23_fracY_uid13_fpDivTest_b_22_sticky_ena_q & en;

    // redist23_fracY_uid13_fpDivTest_b_22_rdcnt(COUNTER,294)
    // low=0, high=20, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist23_fracY_uid13_fpDivTest_b_22_rdcnt_i <= 5'd0;
            redist23_fracY_uid13_fpDivTest_b_22_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist23_fracY_uid13_fpDivTest_b_22_rdcnt_i == 5'd19)
            begin
                redist23_fracY_uid13_fpDivTest_b_22_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist23_fracY_uid13_fpDivTest_b_22_rdcnt_eq <= 1'b0;
            end
            if (redist23_fracY_uid13_fpDivTest_b_22_rdcnt_eq == 1'b1)
            begin
                redist23_fracY_uid13_fpDivTest_b_22_rdcnt_i <= $unsigned(redist23_fracY_uid13_fpDivTest_b_22_rdcnt_i) + $unsigned(5'd12);
            end
            else
            begin
                redist23_fracY_uid13_fpDivTest_b_22_rdcnt_i <= $unsigned(redist23_fracY_uid13_fpDivTest_b_22_rdcnt_i) + $unsigned(5'd1);
            end
        end
    end
    assign redist23_fracY_uid13_fpDivTest_b_22_rdcnt_q = redist23_fracY_uid13_fpDivTest_b_22_rdcnt_i[4:0];

    // redist23_fracY_uid13_fpDivTest_b_22_rdmux(MUX,295)
    assign redist23_fracY_uid13_fpDivTest_b_22_rdmux_s = en;
    always @(redist23_fracY_uid13_fpDivTest_b_22_rdmux_s or redist23_fracY_uid13_fpDivTest_b_22_wraddr_q or redist23_fracY_uid13_fpDivTest_b_22_rdcnt_q)
    begin
        unique case (redist23_fracY_uid13_fpDivTest_b_22_rdmux_s)
            1'b0 : redist23_fracY_uid13_fpDivTest_b_22_rdmux_q = redist23_fracY_uid13_fpDivTest_b_22_wraddr_q;
            1'b1 : redist23_fracY_uid13_fpDivTest_b_22_rdmux_q = redist23_fracY_uid13_fpDivTest_b_22_rdcnt_q;
            default : redist23_fracY_uid13_fpDivTest_b_22_rdmux_q = 5'b0;
        endcase
    end

    // VCC(CONSTANT,1)
    assign VCC_q = 1'b1;

    // fracY_uid13_fpDivTest(BITSELECT,12)@0
    assign fracY_uid13_fpDivTest_b = b[22:0];

    // redist23_fracY_uid13_fpDivTest_b_22_wraddr(REG,296)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist23_fracY_uid13_fpDivTest_b_22_wraddr_q <= 5'b10100;
        end
        else
        begin
            redist23_fracY_uid13_fpDivTest_b_22_wraddr_q <= redist23_fracY_uid13_fpDivTest_b_22_rdmux_q;
        end
    end

    // redist23_fracY_uid13_fpDivTest_b_22_mem(DUALMEM,293)
    assign redist23_fracY_uid13_fpDivTest_b_22_mem_ia = fracY_uid13_fpDivTest_b;
    assign redist23_fracY_uid13_fpDivTest_b_22_mem_aa = redist23_fracY_uid13_fpDivTest_b_22_wraddr_q;
    assign redist23_fracY_uid13_fpDivTest_b_22_mem_ab = redist23_fracY_uid13_fpDivTest_b_22_rdmux_q;
    assign redist23_fracY_uid13_fpDivTest_b_22_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(23),
        .widthad_a(5),
        .numwords_a(21),
        .width_b(23),
        .widthad_b(5),
        .numwords_b(21),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .rdcontrol_reg_b("CLOCK0"),
        .byteena_reg_b("CLOCK0"),
        .outdata_reg_b("CLOCK1"),
        .outdata_sclr_b("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .clock_enable_input_b("NORMAL"),
        .clock_enable_output_b("NORMAL"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("TRUE"),
        .intended_device_family("Stratix 10")
    ) redist23_fracY_uid13_fpDivTest_b_22_mem_dmem (
        .clocken1(redist23_fracY_uid13_fpDivTest_b_22_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist23_fracY_uid13_fpDivTest_b_22_mem_reset0),
        .clock1(clk),
        .address_a(redist23_fracY_uid13_fpDivTest_b_22_mem_aa),
        .data_a(redist23_fracY_uid13_fpDivTest_b_22_mem_ia),
        .wren_a(en[0]),
        .address_b(redist23_fracY_uid13_fpDivTest_b_22_mem_ab),
        .q_b(redist23_fracY_uid13_fpDivTest_b_22_mem_iq),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_b(),
        .clocken2(),
        .clocken3(),
        .aclr0(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .address2_a(),
        .address2_b(),
        .q_a(),
        .eccstatus()
    );
    assign redist23_fracY_uid13_fpDivTest_b_22_mem_q = redist23_fracY_uid13_fpDivTest_b_22_mem_iq[22:0];
    assign redist23_fracY_uid13_fpDivTest_b_22_mem_enaOr_rst = redist23_fracY_uid13_fpDivTest_b_22_enaAnd_q[0] | redist23_fracY_uid13_fpDivTest_b_22_mem_reset0;

    // redist24_fracY_uid13_fpDivTest_b_24(DELAY,210)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist24_fracY_uid13_fpDivTest_b_24_delay_0 <= '0;
            redist24_fracY_uid13_fpDivTest_b_24_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist24_fracY_uid13_fpDivTest_b_24_delay_0 <= redist23_fracY_uid13_fpDivTest_b_22_mem_q;
            redist24_fracY_uid13_fpDivTest_b_24_q <= redist24_fracY_uid13_fpDivTest_b_24_delay_0;
        end
    end

    // paddingY_uid15_fpDivTest(CONSTANT,14)
    assign paddingY_uid15_fpDivTest_q = 23'b00000000000000000000000;

    // fracXIsZero_uid39_fpDivTest(LOGICAL,38)@24 + 1
    assign fracXIsZero_uid39_fpDivTest_qi = paddingY_uid15_fpDivTest_q == redist24_fracY_uid13_fpDivTest_b_24_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    fracXIsZero_uid39_fpDivTest_delay ( .xin(fracXIsZero_uid39_fpDivTest_qi), .xout(fracXIsZero_uid39_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // cstAllOWE_uid18_fpDivTest(CONSTANT,17)
    assign cstAllOWE_uid18_fpDivTest_q = 8'b11111111;

    // redist26_expY_uid12_fpDivTest_b_23_notEnable(LOGICAL,311)
    assign redist26_expY_uid12_fpDivTest_b_23_notEnable_q = ~ (en);

    // redist26_expY_uid12_fpDivTest_b_23_nor(LOGICAL,312)
    assign redist26_expY_uid12_fpDivTest_b_23_nor_q = ~ (redist26_expY_uid12_fpDivTest_b_23_notEnable_q | redist26_expY_uid12_fpDivTest_b_23_sticky_ena_q);

    // redist26_expY_uid12_fpDivTest_b_23_mem_last(CONSTANT,308)
    assign redist26_expY_uid12_fpDivTest_b_23_mem_last_q = 6'b010100;

    // redist26_expY_uid12_fpDivTest_b_23_cmp(LOGICAL,309)
    assign redist26_expY_uid12_fpDivTest_b_23_cmp_b = {1'b0, redist26_expY_uid12_fpDivTest_b_23_rdmux_q};
    assign redist26_expY_uid12_fpDivTest_b_23_cmp_q = redist26_expY_uid12_fpDivTest_b_23_mem_last_q == redist26_expY_uid12_fpDivTest_b_23_cmp_b ? 1'b1 : 1'b0;

    // redist26_expY_uid12_fpDivTest_b_23_cmpReg(REG,310)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist26_expY_uid12_fpDivTest_b_23_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist26_expY_uid12_fpDivTest_b_23_cmpReg_q <= redist26_expY_uid12_fpDivTest_b_23_cmp_q;
        end
    end

    // redist26_expY_uid12_fpDivTest_b_23_sticky_ena(REG,313)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist26_expY_uid12_fpDivTest_b_23_sticky_ena_q <= 1'b0;
        end
        else if (redist26_expY_uid12_fpDivTest_b_23_nor_q == 1'b1)
        begin
            redist26_expY_uid12_fpDivTest_b_23_sticky_ena_q <= redist26_expY_uid12_fpDivTest_b_23_cmpReg_q;
        end
    end

    // redist26_expY_uid12_fpDivTest_b_23_enaAnd(LOGICAL,314)
    assign redist26_expY_uid12_fpDivTest_b_23_enaAnd_q = redist26_expY_uid12_fpDivTest_b_23_sticky_ena_q & en;

    // redist26_expY_uid12_fpDivTest_b_23_rdcnt(COUNTER,305)
    // low=0, high=21, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist26_expY_uid12_fpDivTest_b_23_rdcnt_i <= 5'd0;
            redist26_expY_uid12_fpDivTest_b_23_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist26_expY_uid12_fpDivTest_b_23_rdcnt_i == 5'd20)
            begin
                redist26_expY_uid12_fpDivTest_b_23_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist26_expY_uid12_fpDivTest_b_23_rdcnt_eq <= 1'b0;
            end
            if (redist26_expY_uid12_fpDivTest_b_23_rdcnt_eq == 1'b1)
            begin
                redist26_expY_uid12_fpDivTest_b_23_rdcnt_i <= $unsigned(redist26_expY_uid12_fpDivTest_b_23_rdcnt_i) + $unsigned(5'd11);
            end
            else
            begin
                redist26_expY_uid12_fpDivTest_b_23_rdcnt_i <= $unsigned(redist26_expY_uid12_fpDivTest_b_23_rdcnt_i) + $unsigned(5'd1);
            end
        end
    end
    assign redist26_expY_uid12_fpDivTest_b_23_rdcnt_q = redist26_expY_uid12_fpDivTest_b_23_rdcnt_i[4:0];

    // redist26_expY_uid12_fpDivTest_b_23_rdmux(MUX,306)
    assign redist26_expY_uid12_fpDivTest_b_23_rdmux_s = en;
    always @(redist26_expY_uid12_fpDivTest_b_23_rdmux_s or redist26_expY_uid12_fpDivTest_b_23_wraddr_q or redist26_expY_uid12_fpDivTest_b_23_rdcnt_q)
    begin
        unique case (redist26_expY_uid12_fpDivTest_b_23_rdmux_s)
            1'b0 : redist26_expY_uid12_fpDivTest_b_23_rdmux_q = redist26_expY_uid12_fpDivTest_b_23_wraddr_q;
            1'b1 : redist26_expY_uid12_fpDivTest_b_23_rdmux_q = redist26_expY_uid12_fpDivTest_b_23_rdcnt_q;
            default : redist26_expY_uid12_fpDivTest_b_23_rdmux_q = 5'b0;
        endcase
    end

    // expY_uid12_fpDivTest(BITSELECT,11)@0
    assign expY_uid12_fpDivTest_b = b[30:23];

    // redist26_expY_uid12_fpDivTest_b_23_wraddr(REG,307)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist26_expY_uid12_fpDivTest_b_23_wraddr_q <= 5'b10101;
        end
        else
        begin
            redist26_expY_uid12_fpDivTest_b_23_wraddr_q <= redist26_expY_uid12_fpDivTest_b_23_rdmux_q;
        end
    end

    // redist26_expY_uid12_fpDivTest_b_23_mem(DUALMEM,304)
    assign redist26_expY_uid12_fpDivTest_b_23_mem_ia = expY_uid12_fpDivTest_b;
    assign redist26_expY_uid12_fpDivTest_b_23_mem_aa = redist26_expY_uid12_fpDivTest_b_23_wraddr_q;
    assign redist26_expY_uid12_fpDivTest_b_23_mem_ab = redist26_expY_uid12_fpDivTest_b_23_rdmux_q;
    assign redist26_expY_uid12_fpDivTest_b_23_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(8),
        .widthad_a(5),
        .numwords_a(22),
        .width_b(8),
        .widthad_b(5),
        .numwords_b(22),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .rdcontrol_reg_b("CLOCK0"),
        .byteena_reg_b("CLOCK0"),
        .outdata_reg_b("CLOCK1"),
        .outdata_sclr_b("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .clock_enable_input_b("NORMAL"),
        .clock_enable_output_b("NORMAL"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("TRUE"),
        .intended_device_family("Stratix 10")
    ) redist26_expY_uid12_fpDivTest_b_23_mem_dmem (
        .clocken1(redist26_expY_uid12_fpDivTest_b_23_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist26_expY_uid12_fpDivTest_b_23_mem_reset0),
        .clock1(clk),
        .address_a(redist26_expY_uid12_fpDivTest_b_23_mem_aa),
        .data_a(redist26_expY_uid12_fpDivTest_b_23_mem_ia),
        .wren_a(en[0]),
        .address_b(redist26_expY_uid12_fpDivTest_b_23_mem_ab),
        .q_b(redist26_expY_uid12_fpDivTest_b_23_mem_iq),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_b(),
        .clocken2(),
        .clocken3(),
        .aclr0(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .address2_a(),
        .address2_b(),
        .q_a(),
        .eccstatus()
    );
    assign redist26_expY_uid12_fpDivTest_b_23_mem_q = redist26_expY_uid12_fpDivTest_b_23_mem_iq[7:0];
    assign redist26_expY_uid12_fpDivTest_b_23_mem_enaOr_rst = redist26_expY_uid12_fpDivTest_b_23_enaAnd_q[0] | redist26_expY_uid12_fpDivTest_b_23_mem_reset0;

    // redist27_expY_uid12_fpDivTest_b_24(DELAY,213)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist27_expY_uid12_fpDivTest_b_24_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist27_expY_uid12_fpDivTest_b_24_q <= redist26_expY_uid12_fpDivTest_b_23_mem_q;
        end
    end

    // expXIsMax_uid38_fpDivTest(LOGICAL,37)@24 + 1
    assign expXIsMax_uid38_fpDivTest_qi = redist27_expY_uid12_fpDivTest_b_24_q == cstAllOWE_uid18_fpDivTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    expXIsMax_uid38_fpDivTest_delay ( .xin(expXIsMax_uid38_fpDivTest_qi), .xout(expXIsMax_uid38_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // excI_y_uid41_fpDivTest(LOGICAL,40)@25
    assign excI_y_uid41_fpDivTest_q = expXIsMax_uid38_fpDivTest_q & fracXIsZero_uid39_fpDivTest_q;

    // redist30_fracX_uid10_fpDivTest_b_24_notEnable(LOGICAL,346)
    assign redist30_fracX_uid10_fpDivTest_b_24_notEnable_q = ~ (en);

    // redist30_fracX_uid10_fpDivTest_b_24_nor(LOGICAL,347)
    assign redist30_fracX_uid10_fpDivTest_b_24_nor_q = ~ (redist30_fracX_uid10_fpDivTest_b_24_notEnable_q | redist30_fracX_uid10_fpDivTest_b_24_sticky_ena_q);

    // redist30_fracX_uid10_fpDivTest_b_24_mem_last(CONSTANT,343)
    assign redist30_fracX_uid10_fpDivTest_b_24_mem_last_q = 4'b0100;

    // redist30_fracX_uid10_fpDivTest_b_24_cmp(LOGICAL,344)
    assign redist30_fracX_uid10_fpDivTest_b_24_cmp_b = {1'b0, redist30_fracX_uid10_fpDivTest_b_24_rdmux_q};
    assign redist30_fracX_uid10_fpDivTest_b_24_cmp_q = redist30_fracX_uid10_fpDivTest_b_24_mem_last_q == redist30_fracX_uid10_fpDivTest_b_24_cmp_b ? 1'b1 : 1'b0;

    // redist30_fracX_uid10_fpDivTest_b_24_cmpReg(REG,345)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist30_fracX_uid10_fpDivTest_b_24_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist30_fracX_uid10_fpDivTest_b_24_cmpReg_q <= redist30_fracX_uid10_fpDivTest_b_24_cmp_q;
        end
    end

    // redist30_fracX_uid10_fpDivTest_b_24_sticky_ena(REG,348)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist30_fracX_uid10_fpDivTest_b_24_sticky_ena_q <= 1'b0;
        end
        else if (redist30_fracX_uid10_fpDivTest_b_24_nor_q == 1'b1)
        begin
            redist30_fracX_uid10_fpDivTest_b_24_sticky_ena_q <= redist30_fracX_uid10_fpDivTest_b_24_cmpReg_q;
        end
    end

    // redist30_fracX_uid10_fpDivTest_b_24_enaAnd(LOGICAL,349)
    assign redist30_fracX_uid10_fpDivTest_b_24_enaAnd_q = redist30_fracX_uid10_fpDivTest_b_24_sticky_ena_q & en;

    // redist30_fracX_uid10_fpDivTest_b_24_rdcnt(COUNTER,340)
    // low=0, high=5, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist30_fracX_uid10_fpDivTest_b_24_rdcnt_i <= 3'd0;
            redist30_fracX_uid10_fpDivTest_b_24_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist30_fracX_uid10_fpDivTest_b_24_rdcnt_i == 3'd4)
            begin
                redist30_fracX_uid10_fpDivTest_b_24_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist30_fracX_uid10_fpDivTest_b_24_rdcnt_eq <= 1'b0;
            end
            if (redist30_fracX_uid10_fpDivTest_b_24_rdcnt_eq == 1'b1)
            begin
                redist30_fracX_uid10_fpDivTest_b_24_rdcnt_i <= $unsigned(redist30_fracX_uid10_fpDivTest_b_24_rdcnt_i) + $unsigned(3'd3);
            end
            else
            begin
                redist30_fracX_uid10_fpDivTest_b_24_rdcnt_i <= $unsigned(redist30_fracX_uid10_fpDivTest_b_24_rdcnt_i) + $unsigned(3'd1);
            end
        end
    end
    assign redist30_fracX_uid10_fpDivTest_b_24_rdcnt_q = redist30_fracX_uid10_fpDivTest_b_24_rdcnt_i[2:0];

    // redist30_fracX_uid10_fpDivTest_b_24_rdmux(MUX,341)
    assign redist30_fracX_uid10_fpDivTest_b_24_rdmux_s = en;
    always @(redist30_fracX_uid10_fpDivTest_b_24_rdmux_s or redist30_fracX_uid10_fpDivTest_b_24_wraddr_q or redist30_fracX_uid10_fpDivTest_b_24_rdcnt_q)
    begin
        unique case (redist30_fracX_uid10_fpDivTest_b_24_rdmux_s)
            1'b0 : redist30_fracX_uid10_fpDivTest_b_24_rdmux_q = redist30_fracX_uid10_fpDivTest_b_24_wraddr_q;
            1'b1 : redist30_fracX_uid10_fpDivTest_b_24_rdmux_q = redist30_fracX_uid10_fpDivTest_b_24_rdcnt_q;
            default : redist30_fracX_uid10_fpDivTest_b_24_rdmux_q = 3'b0;
        endcase
    end

    // redist29_fracX_uid10_fpDivTest_b_17_notEnable(LOGICAL,335)
    assign redist29_fracX_uid10_fpDivTest_b_17_notEnable_q = ~ (en);

    // redist29_fracX_uid10_fpDivTest_b_17_nor(LOGICAL,336)
    assign redist29_fracX_uid10_fpDivTest_b_17_nor_q = ~ (redist29_fracX_uid10_fpDivTest_b_17_notEnable_q | redist29_fracX_uid10_fpDivTest_b_17_sticky_ena_q);

    // redist29_fracX_uid10_fpDivTest_b_17_mem_last(CONSTANT,332)
    assign redist29_fracX_uid10_fpDivTest_b_17_mem_last_q = 5'b01101;

    // redist29_fracX_uid10_fpDivTest_b_17_cmp(LOGICAL,333)
    assign redist29_fracX_uid10_fpDivTest_b_17_cmp_b = {1'b0, redist29_fracX_uid10_fpDivTest_b_17_rdmux_q};
    assign redist29_fracX_uid10_fpDivTest_b_17_cmp_q = redist29_fracX_uid10_fpDivTest_b_17_mem_last_q == redist29_fracX_uid10_fpDivTest_b_17_cmp_b ? 1'b1 : 1'b0;

    // redist29_fracX_uid10_fpDivTest_b_17_cmpReg(REG,334)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist29_fracX_uid10_fpDivTest_b_17_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist29_fracX_uid10_fpDivTest_b_17_cmpReg_q <= redist29_fracX_uid10_fpDivTest_b_17_cmp_q;
        end
    end

    // redist29_fracX_uid10_fpDivTest_b_17_sticky_ena(REG,337)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist29_fracX_uid10_fpDivTest_b_17_sticky_ena_q <= 1'b0;
        end
        else if (redist29_fracX_uid10_fpDivTest_b_17_nor_q == 1'b1)
        begin
            redist29_fracX_uid10_fpDivTest_b_17_sticky_ena_q <= redist29_fracX_uid10_fpDivTest_b_17_cmpReg_q;
        end
    end

    // redist29_fracX_uid10_fpDivTest_b_17_enaAnd(LOGICAL,338)
    assign redist29_fracX_uid10_fpDivTest_b_17_enaAnd_q = redist29_fracX_uid10_fpDivTest_b_17_sticky_ena_q & en;

    // redist29_fracX_uid10_fpDivTest_b_17_rdcnt(COUNTER,329)
    // low=0, high=14, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist29_fracX_uid10_fpDivTest_b_17_rdcnt_i <= 4'd0;
            redist29_fracX_uid10_fpDivTest_b_17_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist29_fracX_uid10_fpDivTest_b_17_rdcnt_i == 4'd13)
            begin
                redist29_fracX_uid10_fpDivTest_b_17_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist29_fracX_uid10_fpDivTest_b_17_rdcnt_eq <= 1'b0;
            end
            if (redist29_fracX_uid10_fpDivTest_b_17_rdcnt_eq == 1'b1)
            begin
                redist29_fracX_uid10_fpDivTest_b_17_rdcnt_i <= $unsigned(redist29_fracX_uid10_fpDivTest_b_17_rdcnt_i) + $unsigned(4'd2);
            end
            else
            begin
                redist29_fracX_uid10_fpDivTest_b_17_rdcnt_i <= $unsigned(redist29_fracX_uid10_fpDivTest_b_17_rdcnt_i) + $unsigned(4'd1);
            end
        end
    end
    assign redist29_fracX_uid10_fpDivTest_b_17_rdcnt_q = redist29_fracX_uid10_fpDivTest_b_17_rdcnt_i[3:0];

    // redist29_fracX_uid10_fpDivTest_b_17_rdmux(MUX,330)
    assign redist29_fracX_uid10_fpDivTest_b_17_rdmux_s = en;
    always @(redist29_fracX_uid10_fpDivTest_b_17_rdmux_s or redist29_fracX_uid10_fpDivTest_b_17_wraddr_q or redist29_fracX_uid10_fpDivTest_b_17_rdcnt_q)
    begin
        unique case (redist29_fracX_uid10_fpDivTest_b_17_rdmux_s)
            1'b0 : redist29_fracX_uid10_fpDivTest_b_17_rdmux_q = redist29_fracX_uid10_fpDivTest_b_17_wraddr_q;
            1'b1 : redist29_fracX_uid10_fpDivTest_b_17_rdmux_q = redist29_fracX_uid10_fpDivTest_b_17_rdcnt_q;
            default : redist29_fracX_uid10_fpDivTest_b_17_rdmux_q = 4'b0;
        endcase
    end

    // fracX_uid10_fpDivTest(BITSELECT,9)@0
    assign fracX_uid10_fpDivTest_b = a[22:0];

    // redist29_fracX_uid10_fpDivTest_b_17_wraddr(REG,331)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist29_fracX_uid10_fpDivTest_b_17_wraddr_q <= 4'b1110;
        end
        else
        begin
            redist29_fracX_uid10_fpDivTest_b_17_wraddr_q <= redist29_fracX_uid10_fpDivTest_b_17_rdmux_q;
        end
    end

    // redist29_fracX_uid10_fpDivTest_b_17_mem(DUALMEM,328)
    assign redist29_fracX_uid10_fpDivTest_b_17_mem_ia = fracX_uid10_fpDivTest_b;
    assign redist29_fracX_uid10_fpDivTest_b_17_mem_aa = redist29_fracX_uid10_fpDivTest_b_17_wraddr_q;
    assign redist29_fracX_uid10_fpDivTest_b_17_mem_ab = redist29_fracX_uid10_fpDivTest_b_17_rdmux_q;
    assign redist29_fracX_uid10_fpDivTest_b_17_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(23),
        .widthad_a(4),
        .numwords_a(15),
        .width_b(23),
        .widthad_b(4),
        .numwords_b(15),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .rdcontrol_reg_b("CLOCK0"),
        .byteena_reg_b("CLOCK0"),
        .outdata_reg_b("CLOCK1"),
        .outdata_sclr_b("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .clock_enable_input_b("NORMAL"),
        .clock_enable_output_b("NORMAL"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("TRUE"),
        .intended_device_family("Stratix 10")
    ) redist29_fracX_uid10_fpDivTest_b_17_mem_dmem (
        .clocken1(redist29_fracX_uid10_fpDivTest_b_17_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist29_fracX_uid10_fpDivTest_b_17_mem_reset0),
        .clock1(clk),
        .address_a(redist29_fracX_uid10_fpDivTest_b_17_mem_aa),
        .data_a(redist29_fracX_uid10_fpDivTest_b_17_mem_ia),
        .wren_a(en[0]),
        .address_b(redist29_fracX_uid10_fpDivTest_b_17_mem_ab),
        .q_b(redist29_fracX_uid10_fpDivTest_b_17_mem_iq),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_b(),
        .clocken2(),
        .clocken3(),
        .aclr0(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .address2_a(),
        .address2_b(),
        .q_a(),
        .eccstatus()
    );
    assign redist29_fracX_uid10_fpDivTest_b_17_mem_q = redist29_fracX_uid10_fpDivTest_b_17_mem_iq[22:0];
    assign redist29_fracX_uid10_fpDivTest_b_17_mem_enaOr_rst = redist29_fracX_uid10_fpDivTest_b_17_enaAnd_q[0] | redist29_fracX_uid10_fpDivTest_b_17_mem_reset0;

    // redist29_fracX_uid10_fpDivTest_b_17_outputreg0(DELAY,327)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist29_fracX_uid10_fpDivTest_b_17_outputreg0_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist29_fracX_uid10_fpDivTest_b_17_outputreg0_q <= redist29_fracX_uid10_fpDivTest_b_17_mem_q;
        end
    end

    // redist30_fracX_uid10_fpDivTest_b_24_wraddr(REG,342)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist30_fracX_uid10_fpDivTest_b_24_wraddr_q <= 3'b101;
        end
        else
        begin
            redist30_fracX_uid10_fpDivTest_b_24_wraddr_q <= redist30_fracX_uid10_fpDivTest_b_24_rdmux_q;
        end
    end

    // redist30_fracX_uid10_fpDivTest_b_24_mem(DUALMEM,339)
    assign redist30_fracX_uid10_fpDivTest_b_24_mem_ia = redist29_fracX_uid10_fpDivTest_b_17_outputreg0_q;
    assign redist30_fracX_uid10_fpDivTest_b_24_mem_aa = redist30_fracX_uid10_fpDivTest_b_24_wraddr_q;
    assign redist30_fracX_uid10_fpDivTest_b_24_mem_ab = redist30_fracX_uid10_fpDivTest_b_24_rdmux_q;
    assign redist30_fracX_uid10_fpDivTest_b_24_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(23),
        .widthad_a(3),
        .numwords_a(6),
        .width_b(23),
        .widthad_b(3),
        .numwords_b(6),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .rdcontrol_reg_b("CLOCK0"),
        .byteena_reg_b("CLOCK0"),
        .outdata_reg_b("CLOCK1"),
        .outdata_sclr_b("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .clock_enable_input_b("NORMAL"),
        .clock_enable_output_b("NORMAL"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("TRUE"),
        .intended_device_family("Stratix 10")
    ) redist30_fracX_uid10_fpDivTest_b_24_mem_dmem (
        .clocken1(redist30_fracX_uid10_fpDivTest_b_24_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist30_fracX_uid10_fpDivTest_b_24_mem_reset0),
        .clock1(clk),
        .address_a(redist30_fracX_uid10_fpDivTest_b_24_mem_aa),
        .data_a(redist30_fracX_uid10_fpDivTest_b_24_mem_ia),
        .wren_a(en[0]),
        .address_b(redist30_fracX_uid10_fpDivTest_b_24_mem_ab),
        .q_b(redist30_fracX_uid10_fpDivTest_b_24_mem_iq),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_b(),
        .clocken2(),
        .clocken3(),
        .aclr0(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .address2_a(),
        .address2_b(),
        .q_a(),
        .eccstatus()
    );
    assign redist30_fracX_uid10_fpDivTest_b_24_mem_q = redist30_fracX_uid10_fpDivTest_b_24_mem_iq[22:0];
    assign redist30_fracX_uid10_fpDivTest_b_24_mem_enaOr_rst = redist30_fracX_uid10_fpDivTest_b_24_enaAnd_q[0] | redist30_fracX_uid10_fpDivTest_b_24_mem_reset0;

    // fracXIsZero_uid25_fpDivTest(LOGICAL,24)@24 + 1
    assign fracXIsZero_uid25_fpDivTest_qi = paddingY_uid15_fpDivTest_q == redist30_fracX_uid10_fpDivTest_b_24_mem_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    fracXIsZero_uid25_fpDivTest_delay ( .xin(fracXIsZero_uid25_fpDivTest_qi), .xout(fracXIsZero_uid25_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist33_expX_uid9_fpDivTest_b_23_notEnable(LOGICAL,368)
    assign redist33_expX_uid9_fpDivTest_b_23_notEnable_q = ~ (en);

    // redist33_expX_uid9_fpDivTest_b_23_nor(LOGICAL,369)
    assign redist33_expX_uid9_fpDivTest_b_23_nor_q = ~ (redist33_expX_uid9_fpDivTest_b_23_notEnable_q | redist33_expX_uid9_fpDivTest_b_23_sticky_ena_q);

    // redist33_expX_uid9_fpDivTest_b_23_mem_last(CONSTANT,365)
    assign redist33_expX_uid9_fpDivTest_b_23_mem_last_q = 6'b010100;

    // redist33_expX_uid9_fpDivTest_b_23_cmp(LOGICAL,366)
    assign redist33_expX_uid9_fpDivTest_b_23_cmp_b = {1'b0, redist33_expX_uid9_fpDivTest_b_23_rdmux_q};
    assign redist33_expX_uid9_fpDivTest_b_23_cmp_q = redist33_expX_uid9_fpDivTest_b_23_mem_last_q == redist33_expX_uid9_fpDivTest_b_23_cmp_b ? 1'b1 : 1'b0;

    // redist33_expX_uid9_fpDivTest_b_23_cmpReg(REG,367)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist33_expX_uid9_fpDivTest_b_23_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist33_expX_uid9_fpDivTest_b_23_cmpReg_q <= redist33_expX_uid9_fpDivTest_b_23_cmp_q;
        end
    end

    // redist33_expX_uid9_fpDivTest_b_23_sticky_ena(REG,370)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist33_expX_uid9_fpDivTest_b_23_sticky_ena_q <= 1'b0;
        end
        else if (redist33_expX_uid9_fpDivTest_b_23_nor_q == 1'b1)
        begin
            redist33_expX_uid9_fpDivTest_b_23_sticky_ena_q <= redist33_expX_uid9_fpDivTest_b_23_cmpReg_q;
        end
    end

    // redist33_expX_uid9_fpDivTest_b_23_enaAnd(LOGICAL,371)
    assign redist33_expX_uid9_fpDivTest_b_23_enaAnd_q = redist33_expX_uid9_fpDivTest_b_23_sticky_ena_q & en;

    // redist33_expX_uid9_fpDivTest_b_23_rdcnt(COUNTER,362)
    // low=0, high=21, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist33_expX_uid9_fpDivTest_b_23_rdcnt_i <= 5'd0;
            redist33_expX_uid9_fpDivTest_b_23_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist33_expX_uid9_fpDivTest_b_23_rdcnt_i == 5'd20)
            begin
                redist33_expX_uid9_fpDivTest_b_23_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist33_expX_uid9_fpDivTest_b_23_rdcnt_eq <= 1'b0;
            end
            if (redist33_expX_uid9_fpDivTest_b_23_rdcnt_eq == 1'b1)
            begin
                redist33_expX_uid9_fpDivTest_b_23_rdcnt_i <= $unsigned(redist33_expX_uid9_fpDivTest_b_23_rdcnt_i) + $unsigned(5'd11);
            end
            else
            begin
                redist33_expX_uid9_fpDivTest_b_23_rdcnt_i <= $unsigned(redist33_expX_uid9_fpDivTest_b_23_rdcnt_i) + $unsigned(5'd1);
            end
        end
    end
    assign redist33_expX_uid9_fpDivTest_b_23_rdcnt_q = redist33_expX_uid9_fpDivTest_b_23_rdcnt_i[4:0];

    // redist33_expX_uid9_fpDivTest_b_23_rdmux(MUX,363)
    assign redist33_expX_uid9_fpDivTest_b_23_rdmux_s = en;
    always @(redist33_expX_uid9_fpDivTest_b_23_rdmux_s or redist33_expX_uid9_fpDivTest_b_23_wraddr_q or redist33_expX_uid9_fpDivTest_b_23_rdcnt_q)
    begin
        unique case (redist33_expX_uid9_fpDivTest_b_23_rdmux_s)
            1'b0 : redist33_expX_uid9_fpDivTest_b_23_rdmux_q = redist33_expX_uid9_fpDivTest_b_23_wraddr_q;
            1'b1 : redist33_expX_uid9_fpDivTest_b_23_rdmux_q = redist33_expX_uid9_fpDivTest_b_23_rdcnt_q;
            default : redist33_expX_uid9_fpDivTest_b_23_rdmux_q = 5'b0;
        endcase
    end

    // expX_uid9_fpDivTest(BITSELECT,8)@0
    assign expX_uid9_fpDivTest_b = a[30:23];

    // redist33_expX_uid9_fpDivTest_b_23_wraddr(REG,364)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist33_expX_uid9_fpDivTest_b_23_wraddr_q <= 5'b10101;
        end
        else
        begin
            redist33_expX_uid9_fpDivTest_b_23_wraddr_q <= redist33_expX_uid9_fpDivTest_b_23_rdmux_q;
        end
    end

    // redist33_expX_uid9_fpDivTest_b_23_mem(DUALMEM,361)
    assign redist33_expX_uid9_fpDivTest_b_23_mem_ia = expX_uid9_fpDivTest_b;
    assign redist33_expX_uid9_fpDivTest_b_23_mem_aa = redist33_expX_uid9_fpDivTest_b_23_wraddr_q;
    assign redist33_expX_uid9_fpDivTest_b_23_mem_ab = redist33_expX_uid9_fpDivTest_b_23_rdmux_q;
    assign redist33_expX_uid9_fpDivTest_b_23_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(8),
        .widthad_a(5),
        .numwords_a(22),
        .width_b(8),
        .widthad_b(5),
        .numwords_b(22),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .rdcontrol_reg_b("CLOCK0"),
        .byteena_reg_b("CLOCK0"),
        .outdata_reg_b("CLOCK1"),
        .outdata_sclr_b("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .clock_enable_input_b("NORMAL"),
        .clock_enable_output_b("NORMAL"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("TRUE"),
        .intended_device_family("Stratix 10")
    ) redist33_expX_uid9_fpDivTest_b_23_mem_dmem (
        .clocken1(redist33_expX_uid9_fpDivTest_b_23_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist33_expX_uid9_fpDivTest_b_23_mem_reset0),
        .clock1(clk),
        .address_a(redist33_expX_uid9_fpDivTest_b_23_mem_aa),
        .data_a(redist33_expX_uid9_fpDivTest_b_23_mem_ia),
        .wren_a(en[0]),
        .address_b(redist33_expX_uid9_fpDivTest_b_23_mem_ab),
        .q_b(redist33_expX_uid9_fpDivTest_b_23_mem_iq),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_b(),
        .clocken2(),
        .clocken3(),
        .aclr0(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .address2_a(),
        .address2_b(),
        .q_a(),
        .eccstatus()
    );
    assign redist33_expX_uid9_fpDivTest_b_23_mem_q = redist33_expX_uid9_fpDivTest_b_23_mem_iq[7:0];
    assign redist33_expX_uid9_fpDivTest_b_23_mem_enaOr_rst = redist33_expX_uid9_fpDivTest_b_23_enaAnd_q[0] | redist33_expX_uid9_fpDivTest_b_23_mem_reset0;

    // redist34_expX_uid9_fpDivTest_b_24(DELAY,220)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist34_expX_uid9_fpDivTest_b_24_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist34_expX_uid9_fpDivTest_b_24_q <= redist33_expX_uid9_fpDivTest_b_23_mem_q;
        end
    end

    // expXIsMax_uid24_fpDivTest(LOGICAL,23)@24
    assign expXIsMax_uid24_fpDivTest_q = redist34_expX_uid9_fpDivTest_b_24_q == cstAllOWE_uid18_fpDivTest_q ? 1'b1 : 1'b0;

    // redist21_expXIsMax_uid24_fpDivTest_q_1(DELAY,207)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist21_expXIsMax_uid24_fpDivTest_q_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist21_expXIsMax_uid24_fpDivTest_q_1_q <= expXIsMax_uid24_fpDivTest_q;
        end
    end

    // excI_x_uid27_fpDivTest(LOGICAL,26)@25
    assign excI_x_uid27_fpDivTest_q = redist21_expXIsMax_uid24_fpDivTest_q_1_q & fracXIsZero_uid25_fpDivTest_q;

    // excXIYI_uid130_fpDivTest(LOGICAL,129)@25
    assign excXIYI_uid130_fpDivTest_q = excI_x_uid27_fpDivTest_q & excI_y_uid41_fpDivTest_q;

    // fracXIsNotZero_uid40_fpDivTest(LOGICAL,39)@25
    assign fracXIsNotZero_uid40_fpDivTest_q = ~ (fracXIsZero_uid39_fpDivTest_q);

    // excN_y_uid42_fpDivTest(LOGICAL,41)@25
    assign excN_y_uid42_fpDivTest_q = expXIsMax_uid38_fpDivTest_q & fracXIsNotZero_uid40_fpDivTest_q;

    // fracXIsNotZero_uid26_fpDivTest(LOGICAL,25)@25
    assign fracXIsNotZero_uid26_fpDivTest_q = ~ (fracXIsZero_uid25_fpDivTest_q);

    // excN_x_uid28_fpDivTest(LOGICAL,27)@25
    assign excN_x_uid28_fpDivTest_q = redist21_expXIsMax_uid24_fpDivTest_q_1_q & fracXIsNotZero_uid26_fpDivTest_q;

    // cstAllZWE_uid20_fpDivTest(CONSTANT,19)
    assign cstAllZWE_uid20_fpDivTest_q = 8'b00000000;

    // excZ_y_uid37_fpDivTest(LOGICAL,36)@24 + 1
    assign excZ_y_uid37_fpDivTest_qi = redist27_expY_uid12_fpDivTest_b_24_q == cstAllZWE_uid20_fpDivTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    excZ_y_uid37_fpDivTest_delay ( .xin(excZ_y_uid37_fpDivTest_qi), .xout(excZ_y_uid37_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // excZ_x_uid23_fpDivTest(LOGICAL,22)@24
    assign excZ_x_uid23_fpDivTest_q = redist34_expX_uid9_fpDivTest_b_24_q == cstAllZWE_uid20_fpDivTest_q ? 1'b1 : 1'b0;

    // redist22_excZ_x_uid23_fpDivTest_q_1(DELAY,208)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist22_excZ_x_uid23_fpDivTest_q_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist22_excZ_x_uid23_fpDivTest_q_1_q <= excZ_x_uid23_fpDivTest_q;
        end
    end

    // excXZYZ_uid129_fpDivTest(LOGICAL,128)@25
    assign excXZYZ_uid129_fpDivTest_q = redist22_excZ_x_uid23_fpDivTest_q_1_q & excZ_y_uid37_fpDivTest_q;

    // excRNaN_uid131_fpDivTest(LOGICAL,130)@25
    assign excRNaN_uid131_fpDivTest_q = excXZYZ_uid129_fpDivTest_q | excN_x_uid28_fpDivTest_q | excN_y_uid42_fpDivTest_q | excXIYI_uid130_fpDivTest_q;

    // invExcRNaN_uid142_fpDivTest(LOGICAL,141)@25
    assign invExcRNaN_uid142_fpDivTest_q = ~ (excRNaN_uid131_fpDivTest_q);

    // signY_uid14_fpDivTest(BITSELECT,13)@0
    assign signY_uid14_fpDivTest_b = b[31:31];

    // signX_uid11_fpDivTest(BITSELECT,10)@0
    assign signX_uid11_fpDivTest_b = a[31:31];

    // signR_uid46_fpDivTest(LOGICAL,45)@0 + 1
    assign signR_uid46_fpDivTest_qi = signX_uid11_fpDivTest_b ^ signY_uid14_fpDivTest_b;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    signR_uid46_fpDivTest_delay ( .xin(signR_uid46_fpDivTest_qi), .xout(signR_uid46_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist20_signR_uid46_fpDivTest_q_25(DELAY,206)
    dspba_delay_ver #( .width(1), .depth(24), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    redist20_signR_uid46_fpDivTest_q_25 ( .xin(signR_uid46_fpDivTest_q), .xout(redist20_signR_uid46_fpDivTest_q_25_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // sRPostExc_uid143_fpDivTest(LOGICAL,142)@25 + 1
    assign sRPostExc_uid143_fpDivTest_qi = redist20_signR_uid46_fpDivTest_q_25_q & invExcRNaN_uid142_fpDivTest_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    sRPostExc_uid143_fpDivTest_delay ( .xin(sRPostExc_uid143_fpDivTest_qi), .xout(sRPostExc_uid143_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist2_sRPostExc_uid143_fpDivTest_q_9(DELAY,188)
    dspba_delay_ver #( .width(1), .depth(8), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    redist2_sRPostExc_uid143_fpDivTest_q_9 ( .xin(sRPostExc_uid143_fpDivTest_q), .xout(redist2_sRPostExc_uid143_fpDivTest_q_9_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist4_fracPostRndFT_uid104_fpDivTest_b_8_notEnable(LOGICAL,230)
    assign redist4_fracPostRndFT_uid104_fpDivTest_b_8_notEnable_q = ~ (en);

    // redist4_fracPostRndFT_uid104_fpDivTest_b_8_nor(LOGICAL,231)
    assign redist4_fracPostRndFT_uid104_fpDivTest_b_8_nor_q = ~ (redist4_fracPostRndFT_uid104_fpDivTest_b_8_notEnable_q | redist4_fracPostRndFT_uid104_fpDivTest_b_8_sticky_ena_q);

    // redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_last(CONSTANT,227)
    assign redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_last_q = 4'b0101;

    // redist4_fracPostRndFT_uid104_fpDivTest_b_8_cmp(LOGICAL,228)
    assign redist4_fracPostRndFT_uid104_fpDivTest_b_8_cmp_b = {1'b0, redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdmux_q};
    assign redist4_fracPostRndFT_uid104_fpDivTest_b_8_cmp_q = redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_last_q == redist4_fracPostRndFT_uid104_fpDivTest_b_8_cmp_b ? 1'b1 : 1'b0;

    // redist4_fracPostRndFT_uid104_fpDivTest_b_8_cmpReg(REG,229)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist4_fracPostRndFT_uid104_fpDivTest_b_8_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist4_fracPostRndFT_uid104_fpDivTest_b_8_cmpReg_q <= redist4_fracPostRndFT_uid104_fpDivTest_b_8_cmp_q;
        end
    end

    // redist4_fracPostRndFT_uid104_fpDivTest_b_8_sticky_ena(REG,232)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist4_fracPostRndFT_uid104_fpDivTest_b_8_sticky_ena_q <= 1'b0;
        end
        else if (redist4_fracPostRndFT_uid104_fpDivTest_b_8_nor_q == 1'b1)
        begin
            redist4_fracPostRndFT_uid104_fpDivTest_b_8_sticky_ena_q <= redist4_fracPostRndFT_uid104_fpDivTest_b_8_cmpReg_q;
        end
    end

    // redist4_fracPostRndFT_uid104_fpDivTest_b_8_enaAnd(LOGICAL,233)
    assign redist4_fracPostRndFT_uid104_fpDivTest_b_8_enaAnd_q = redist4_fracPostRndFT_uid104_fpDivTest_b_8_sticky_ena_q & en;

    // redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt(COUNTER,224)
    // low=0, high=6, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt_i <= 3'd0;
            redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt_i == 3'd5)
            begin
                redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt_eq <= 1'b0;
            end
            if (redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt_eq == 1'b1)
            begin
                redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt_i <= $unsigned(redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt_i) + $unsigned(3'd2);
            end
            else
            begin
                redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt_i <= $unsigned(redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt_i) + $unsigned(3'd1);
            end
        end
    end
    assign redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt_q = redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt_i[2:0];

    // redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdmux(MUX,225)
    assign redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdmux_s = en;
    always @(redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdmux_s or redist4_fracPostRndFT_uid104_fpDivTest_b_8_wraddr_q or redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt_q)
    begin
        unique case (redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdmux_s)
            1'b0 : redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdmux_q = redist4_fracPostRndFT_uid104_fpDivTest_b_8_wraddr_q;
            1'b1 : redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdmux_q = redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdcnt_q;
            default : redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdmux_q = 3'b0;
        endcase
    end

    // redist31_fracX_uid10_fpDivTest_b_25(DELAY,217)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist31_fracX_uid10_fpDivTest_b_25_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist31_fracX_uid10_fpDivTest_b_25_q <= redist30_fracX_uid10_fpDivTest_b_24_mem_q;
        end
    end

    // GND(CONSTANT,0)
    assign GND_q = 1'b0;

    // fracXExt_uid77_fpDivTest(BITJOIN,76)@25
    assign fracXExt_uid77_fpDivTest_q = {redist31_fracX_uid10_fpDivTest_b_25_q, GND_q};

    // redist12_lOAdded_uid57_fpDivTest_q_6_notEnable(LOGICAL,253)
    assign redist12_lOAdded_uid57_fpDivTest_q_6_notEnable_q = ~ (en);

    // redist12_lOAdded_uid57_fpDivTest_q_6_nor(LOGICAL,254)
    assign redist12_lOAdded_uid57_fpDivTest_q_6_nor_q = ~ (redist12_lOAdded_uid57_fpDivTest_q_6_notEnable_q | redist12_lOAdded_uid57_fpDivTest_q_6_sticky_ena_q);

    // redist12_lOAdded_uid57_fpDivTest_q_6_mem_last(CONSTANT,250)
    assign redist12_lOAdded_uid57_fpDivTest_q_6_mem_last_q = 3'b011;

    // redist12_lOAdded_uid57_fpDivTest_q_6_cmp(LOGICAL,251)
    assign redist12_lOAdded_uid57_fpDivTest_q_6_cmp_q = redist12_lOAdded_uid57_fpDivTest_q_6_mem_last_q == redist12_lOAdded_uid57_fpDivTest_q_6_rdmux_q ? 1'b1 : 1'b0;

    // redist12_lOAdded_uid57_fpDivTest_q_6_cmpReg(REG,252)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist12_lOAdded_uid57_fpDivTest_q_6_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist12_lOAdded_uid57_fpDivTest_q_6_cmpReg_q <= redist12_lOAdded_uid57_fpDivTest_q_6_cmp_q;
        end
    end

    // redist12_lOAdded_uid57_fpDivTest_q_6_sticky_ena(REG,255)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist12_lOAdded_uid57_fpDivTest_q_6_sticky_ena_q <= 1'b0;
        end
        else if (redist12_lOAdded_uid57_fpDivTest_q_6_nor_q == 1'b1)
        begin
            redist12_lOAdded_uid57_fpDivTest_q_6_sticky_ena_q <= redist12_lOAdded_uid57_fpDivTest_q_6_cmpReg_q;
        end
    end

    // redist12_lOAdded_uid57_fpDivTest_q_6_enaAnd(LOGICAL,256)
    assign redist12_lOAdded_uid57_fpDivTest_q_6_enaAnd_q = redist12_lOAdded_uid57_fpDivTest_q_6_sticky_ena_q & en;

    // redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt(COUNTER,247)
    // low=0, high=4, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt_i <= 3'd0;
            redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt_i == 3'd3)
            begin
                redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt_eq <= 1'b0;
            end
            if (redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt_eq == 1'b1)
            begin
                redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt_i <= $unsigned(redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt_i) + $unsigned(3'd4);
            end
            else
            begin
                redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt_i <= $unsigned(redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt_i) + $unsigned(3'd1);
            end
        end
    end
    assign redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt_q = redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt_i[2:0];

    // redist12_lOAdded_uid57_fpDivTest_q_6_rdmux(MUX,248)
    assign redist12_lOAdded_uid57_fpDivTest_q_6_rdmux_s = en;
    always @(redist12_lOAdded_uid57_fpDivTest_q_6_rdmux_s or redist12_lOAdded_uid57_fpDivTest_q_6_wraddr_q or redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt_q)
    begin
        unique case (redist12_lOAdded_uid57_fpDivTest_q_6_rdmux_s)
            1'b0 : redist12_lOAdded_uid57_fpDivTest_q_6_rdmux_q = redist12_lOAdded_uid57_fpDivTest_q_6_wraddr_q;
            1'b1 : redist12_lOAdded_uid57_fpDivTest_q_6_rdmux_q = redist12_lOAdded_uid57_fpDivTest_q_6_rdcnt_q;
            default : redist12_lOAdded_uid57_fpDivTest_q_6_rdmux_q = 3'b0;
        endcase
    end

    // lOAdded_uid57_fpDivTest(BITJOIN,56)@17
    assign lOAdded_uid57_fpDivTest_q = {VCC_q, redist29_fracX_uid10_fpDivTest_b_17_outputreg0_q};

    // redist12_lOAdded_uid57_fpDivTest_q_6_wraddr(REG,249)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist12_lOAdded_uid57_fpDivTest_q_6_wraddr_q <= 3'b100;
        end
        else
        begin
            redist12_lOAdded_uid57_fpDivTest_q_6_wraddr_q <= redist12_lOAdded_uid57_fpDivTest_q_6_rdmux_q;
        end
    end

    // redist12_lOAdded_uid57_fpDivTest_q_6_mem(DUALMEM,246)
    assign redist12_lOAdded_uid57_fpDivTest_q_6_mem_ia = lOAdded_uid57_fpDivTest_q;
    assign redist12_lOAdded_uid57_fpDivTest_q_6_mem_aa = redist12_lOAdded_uid57_fpDivTest_q_6_wraddr_q;
    assign redist12_lOAdded_uid57_fpDivTest_q_6_mem_ab = redist12_lOAdded_uid57_fpDivTest_q_6_rdmux_q;
    assign redist12_lOAdded_uid57_fpDivTest_q_6_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(24),
        .widthad_a(3),
        .numwords_a(5),
        .width_b(24),
        .widthad_b(3),
        .numwords_b(5),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .rdcontrol_reg_b("CLOCK0"),
        .byteena_reg_b("CLOCK0"),
        .outdata_reg_b("CLOCK1"),
        .outdata_sclr_b("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .clock_enable_input_b("NORMAL"),
        .clock_enable_output_b("NORMAL"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("TRUE"),
        .intended_device_family("Stratix 10")
    ) redist12_lOAdded_uid57_fpDivTest_q_6_mem_dmem (
        .clocken1(redist12_lOAdded_uid57_fpDivTest_q_6_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist12_lOAdded_uid57_fpDivTest_q_6_mem_reset0),
        .clock1(clk),
        .address_a(redist12_lOAdded_uid57_fpDivTest_q_6_mem_aa),
        .data_a(redist12_lOAdded_uid57_fpDivTest_q_6_mem_ia),
        .wren_a(en[0]),
        .address_b(redist12_lOAdded_uid57_fpDivTest_q_6_mem_ab),
        .q_b(redist12_lOAdded_uid57_fpDivTest_q_6_mem_iq),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_b(),
        .clocken2(),
        .clocken3(),
        .aclr0(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .address2_a(),
        .address2_b(),
        .q_a(),
        .eccstatus()
    );
    assign redist12_lOAdded_uid57_fpDivTest_q_6_mem_q = redist12_lOAdded_uid57_fpDivTest_q_6_mem_iq[23:0];
    assign redist12_lOAdded_uid57_fpDivTest_q_6_mem_enaOr_rst = redist12_lOAdded_uid57_fpDivTest_q_6_enaAnd_q[0] | redist12_lOAdded_uid57_fpDivTest_q_6_mem_reset0;

    // z4_uid60_fpDivTest(CONSTANT,59)
    assign z4_uid60_fpDivTest_q = 4'b0000;

    // oFracXZ4_uid61_fpDivTest(BITJOIN,60)@23
    assign oFracXZ4_uid61_fpDivTest_q = {redist12_lOAdded_uid57_fpDivTest_q_6_mem_q, z4_uid60_fpDivTest_q};

    // yAddr_uid51_fpDivTest(BITSELECT,50)@0
    assign yAddr_uid51_fpDivTest_b = fracY_uid13_fpDivTest_b[22:14];

    // memoryC2_uid152_invTables_lutmem(DUALMEM,181)@0 + 2
    // in j@20000000
    assign memoryC2_uid152_invTables_lutmem_aa = yAddr_uid51_fpDivTest_b;
    assign memoryC2_uid152_invTables_lutmem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("M20K"),
        .operation_mode("ROM"),
        .width_a(13),
        .widthad_a(9),
        .numwords_a(512),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .outdata_reg_a("CLOCK0"),
        .outdata_sclr_a("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .power_up_uninitialized("FALSE"),
        .init_file("acl_fdiv_memoryC2_uid152_invTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Stratix 10")
    ) memoryC2_uid152_invTables_lutmem_dmem (
        .clocken0(en[0]),
        .sclr(memoryC2_uid152_invTables_lutmem_reset0),
        .clock0(clk),
        .address_a(memoryC2_uid152_invTables_lutmem_aa),
        .q_a(memoryC2_uid152_invTables_lutmem_ir),
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
        .aclr0(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .address2_a(),
        .address2_b(),
        .q_b(),
        .eccstatus()
    );
    assign memoryC2_uid152_invTables_lutmem_r = memoryC2_uid152_invTables_lutmem_ir[12:0];
    assign memoryC2_uid152_invTables_lutmem_enaOr_rst = en[0] | memoryC2_uid152_invTables_lutmem_reset0;

    // redist0_memoryC2_uid152_invTables_lutmem_r_1(DELAY,186)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist0_memoryC2_uid152_invTables_lutmem_r_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist0_memoryC2_uid152_invTables_lutmem_r_1_q <= memoryC2_uid152_invTables_lutmem_r;
        end
    end

    // yPE_uid52_fpDivTest(BITSELECT,51)@0
    assign yPE_uid52_fpDivTest_b = b[13:0];

    // redist16_yPE_uid52_fpDivTest_b_3(DELAY,202)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist16_yPE_uid52_fpDivTest_b_3_delay_0 <= '0;
            redist16_yPE_uid52_fpDivTest_b_3_delay_1 <= '0;
            redist16_yPE_uid52_fpDivTest_b_3_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist16_yPE_uid52_fpDivTest_b_3_delay_0 <= yPE_uid52_fpDivTest_b;
            redist16_yPE_uid52_fpDivTest_b_3_delay_1 <= redist16_yPE_uid52_fpDivTest_b_3_delay_0;
            redist16_yPE_uid52_fpDivTest_b_3_q <= redist16_yPE_uid52_fpDivTest_b_3_delay_1;
        end
    end

    // yT1_uid158_invPolyEval(BITSELECT,157)@3
    assign yT1_uid158_invPolyEval_b = redist16_yPE_uid52_fpDivTest_b_3_q[13:1];

    // prodXY_uid174_pT1_uid159_invPolyEval_cma(CHAINMULTADD,184)@3 + 5
    // out q@9
    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_reset = areset;
    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_ena0 = en[0] | prodXY_uid174_pT1_uid159_invPolyEval_cma_reset;
    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_ena1 = prodXY_uid174_pT1_uid159_invPolyEval_cma_ena0;
    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_ena2 = prodXY_uid174_pT1_uid159_invPolyEval_cma_ena0;
    always @ (posedge clk)
    begin
        if (0)
        begin
        end
        else
        begin
            if (en == 1'b1)
            begin
                prodXY_uid174_pT1_uid159_invPolyEval_cma_ah[0] <= yT1_uid158_invPolyEval_b;
                prodXY_uid174_pT1_uid159_invPolyEval_cma_ch[0] <= redist0_memoryC2_uid152_invTables_lutmem_r_1_q;
            end
        end
    end

    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_a0 = prodXY_uid174_pT1_uid159_invPolyEval_cma_ah[0];
    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_c0 = prodXY_uid174_pT1_uid159_invPolyEval_cma_ch[0];
    fourteennm_mac #(
        .operation_mode("m18x18_full"),
        .clear_type("sclr"),
        .ay_scan_in_clock("0"),
        .ay_scan_in_width(13),
        .ax_clock("0"),
        .ax_width(13),
        .signed_may("false"),
        .signed_max("true"),
        .input_pipeline_clock("2"),
        .second_pipeline_clock("2"),
        .output_clock("1"),
        .result_a_width(26)
    ) prodXY_uid174_pT1_uid159_invPolyEval_cma_DSP0 (
        .clk({clk,clk,clk}),
        .ena({ prodXY_uid174_pT1_uid159_invPolyEval_cma_ena2, prodXY_uid174_pT1_uid159_invPolyEval_cma_ena1, prodXY_uid174_pT1_uid159_invPolyEval_cma_ena0 }),
        .clr({ prodXY_uid174_pT1_uid159_invPolyEval_cma_reset, prodXY_uid174_pT1_uid159_invPolyEval_cma_reset }),
        .ay(prodXY_uid174_pT1_uid159_invPolyEval_cma_a0),
        .ax(prodXY_uid174_pT1_uid159_invPolyEval_cma_c0),
        .resulta(prodXY_uid174_pT1_uid159_invPolyEval_cma_s0),
        .accumulate(),
        .loadconst(),
        .negate(),
        .sub(),
        .az(),
        .coefsela(),
        .bx(),
        .by(),
        .bz(),
        .coefselb(),
        .scanin(),
        .scanout(),
        .chainin(),
        .chainout(),
        .resultb(),
        .dfxlfsrena(),
        .dfxmisrena(),
        .dftout()
    );
    dspba_delay_ver #( .width(26), .depth(1), .reset_kind("NONE"), .phase(0), .modulus(1) )
    prodXY_uid174_pT1_uid159_invPolyEval_cma_delay ( .xin(prodXY_uid174_pT1_uid159_invPolyEval_cma_s0), .xout(prodXY_uid174_pT1_uid159_invPolyEval_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_q = prodXY_uid174_pT1_uid159_invPolyEval_cma_qq[25:0];

    // osig_uid175_pT1_uid159_invPolyEval(BITSELECT,174)@9
    assign osig_uid175_pT1_uid159_invPolyEval_b = prodXY_uid174_pT1_uid159_invPolyEval_cma_q[25:12];

    // highBBits_uid161_invPolyEval(BITSELECT,160)@9
    assign highBBits_uid161_invPolyEval_b = osig_uid175_pT1_uid159_invPolyEval_b[13:1];

    // redist18_yAddr_uid51_fpDivTest_b_7_notEnable(LOGICAL,277)
    assign redist18_yAddr_uid51_fpDivTest_b_7_notEnable_q = ~ (en);

    // redist18_yAddr_uid51_fpDivTest_b_7_nor(LOGICAL,278)
    assign redist18_yAddr_uid51_fpDivTest_b_7_nor_q = ~ (redist18_yAddr_uid51_fpDivTest_b_7_notEnable_q | redist18_yAddr_uid51_fpDivTest_b_7_sticky_ena_q);

    // redist18_yAddr_uid51_fpDivTest_b_7_mem_last(CONSTANT,274)
    assign redist18_yAddr_uid51_fpDivTest_b_7_mem_last_q = 3'b011;

    // redist18_yAddr_uid51_fpDivTest_b_7_cmp(LOGICAL,275)
    assign redist18_yAddr_uid51_fpDivTest_b_7_cmp_q = redist18_yAddr_uid51_fpDivTest_b_7_mem_last_q == redist18_yAddr_uid51_fpDivTest_b_7_rdmux_q ? 1'b1 : 1'b0;

    // redist18_yAddr_uid51_fpDivTest_b_7_cmpReg(REG,276)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist18_yAddr_uid51_fpDivTest_b_7_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist18_yAddr_uid51_fpDivTest_b_7_cmpReg_q <= redist18_yAddr_uid51_fpDivTest_b_7_cmp_q;
        end
    end

    // redist18_yAddr_uid51_fpDivTest_b_7_sticky_ena(REG,279)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist18_yAddr_uid51_fpDivTest_b_7_sticky_ena_q <= 1'b0;
        end
        else if (redist18_yAddr_uid51_fpDivTest_b_7_nor_q == 1'b1)
        begin
            redist18_yAddr_uid51_fpDivTest_b_7_sticky_ena_q <= redist18_yAddr_uid51_fpDivTest_b_7_cmpReg_q;
        end
    end

    // redist18_yAddr_uid51_fpDivTest_b_7_enaAnd(LOGICAL,280)
    assign redist18_yAddr_uid51_fpDivTest_b_7_enaAnd_q = redist18_yAddr_uid51_fpDivTest_b_7_sticky_ena_q & en;

    // redist18_yAddr_uid51_fpDivTest_b_7_rdcnt(COUNTER,271)
    // low=0, high=4, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist18_yAddr_uid51_fpDivTest_b_7_rdcnt_i <= 3'd0;
            redist18_yAddr_uid51_fpDivTest_b_7_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist18_yAddr_uid51_fpDivTest_b_7_rdcnt_i == 3'd3)
            begin
                redist18_yAddr_uid51_fpDivTest_b_7_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist18_yAddr_uid51_fpDivTest_b_7_rdcnt_eq <= 1'b0;
            end
            if (redist18_yAddr_uid51_fpDivTest_b_7_rdcnt_eq == 1'b1)
            begin
                redist18_yAddr_uid51_fpDivTest_b_7_rdcnt_i <= $unsigned(redist18_yAddr_uid51_fpDivTest_b_7_rdcnt_i) + $unsigned(3'd4);
            end
            else
            begin
                redist18_yAddr_uid51_fpDivTest_b_7_rdcnt_i <= $unsigned(redist18_yAddr_uid51_fpDivTest_b_7_rdcnt_i) + $unsigned(3'd1);
            end
        end
    end
    assign redist18_yAddr_uid51_fpDivTest_b_7_rdcnt_q = redist18_yAddr_uid51_fpDivTest_b_7_rdcnt_i[2:0];

    // redist18_yAddr_uid51_fpDivTest_b_7_rdmux(MUX,272)
    assign redist18_yAddr_uid51_fpDivTest_b_7_rdmux_s = en;
    always @(redist18_yAddr_uid51_fpDivTest_b_7_rdmux_s or redist18_yAddr_uid51_fpDivTest_b_7_wraddr_q or redist18_yAddr_uid51_fpDivTest_b_7_rdcnt_q)
    begin
        unique case (redist18_yAddr_uid51_fpDivTest_b_7_rdmux_s)
            1'b0 : redist18_yAddr_uid51_fpDivTest_b_7_rdmux_q = redist18_yAddr_uid51_fpDivTest_b_7_wraddr_q;
            1'b1 : redist18_yAddr_uid51_fpDivTest_b_7_rdmux_q = redist18_yAddr_uid51_fpDivTest_b_7_rdcnt_q;
            default : redist18_yAddr_uid51_fpDivTest_b_7_rdmux_q = 3'b0;
        endcase
    end

    // redist18_yAddr_uid51_fpDivTest_b_7_wraddr(REG,273)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist18_yAddr_uid51_fpDivTest_b_7_wraddr_q <= 3'b100;
        end
        else
        begin
            redist18_yAddr_uid51_fpDivTest_b_7_wraddr_q <= redist18_yAddr_uid51_fpDivTest_b_7_rdmux_q;
        end
    end

    // redist18_yAddr_uid51_fpDivTest_b_7_mem(DUALMEM,270)
    assign redist18_yAddr_uid51_fpDivTest_b_7_mem_ia = yAddr_uid51_fpDivTest_b;
    assign redist18_yAddr_uid51_fpDivTest_b_7_mem_aa = redist18_yAddr_uid51_fpDivTest_b_7_wraddr_q;
    assign redist18_yAddr_uid51_fpDivTest_b_7_mem_ab = redist18_yAddr_uid51_fpDivTest_b_7_rdmux_q;
    assign redist18_yAddr_uid51_fpDivTest_b_7_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(9),
        .widthad_a(3),
        .numwords_a(5),
        .width_b(9),
        .widthad_b(3),
        .numwords_b(5),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .rdcontrol_reg_b("CLOCK0"),
        .byteena_reg_b("CLOCK0"),
        .outdata_reg_b("CLOCK1"),
        .outdata_sclr_b("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .clock_enable_input_b("NORMAL"),
        .clock_enable_output_b("NORMAL"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("TRUE"),
        .intended_device_family("Stratix 10")
    ) redist18_yAddr_uid51_fpDivTest_b_7_mem_dmem (
        .clocken1(redist18_yAddr_uid51_fpDivTest_b_7_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist18_yAddr_uid51_fpDivTest_b_7_mem_reset0),
        .clock1(clk),
        .address_a(redist18_yAddr_uid51_fpDivTest_b_7_mem_aa),
        .data_a(redist18_yAddr_uid51_fpDivTest_b_7_mem_ia),
        .wren_a(en[0]),
        .address_b(redist18_yAddr_uid51_fpDivTest_b_7_mem_ab),
        .q_b(redist18_yAddr_uid51_fpDivTest_b_7_mem_iq),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_b(),
        .clocken2(),
        .clocken3(),
        .aclr0(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .address2_a(),
        .address2_b(),
        .q_a(),
        .eccstatus()
    );
    assign redist18_yAddr_uid51_fpDivTest_b_7_mem_q = redist18_yAddr_uid51_fpDivTest_b_7_mem_iq[8:0];
    assign redist18_yAddr_uid51_fpDivTest_b_7_mem_enaOr_rst = redist18_yAddr_uid51_fpDivTest_b_7_enaAnd_q[0] | redist18_yAddr_uid51_fpDivTest_b_7_mem_reset0;

    // redist18_yAddr_uid51_fpDivTest_b_7_outputreg0(DELAY,269)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist18_yAddr_uid51_fpDivTest_b_7_outputreg0_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist18_yAddr_uid51_fpDivTest_b_7_outputreg0_q <= redist18_yAddr_uid51_fpDivTest_b_7_mem_q;
        end
    end

    // memoryC1_uid149_invTables_lutmem(DUALMEM,180)@7 + 2
    // in j@20000000
    assign memoryC1_uid149_invTables_lutmem_aa = redist18_yAddr_uid51_fpDivTest_b_7_outputreg0_q;
    assign memoryC1_uid149_invTables_lutmem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("M20K"),
        .operation_mode("ROM"),
        .width_a(22),
        .widthad_a(9),
        .numwords_a(512),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .outdata_reg_a("CLOCK0"),
        .outdata_sclr_a("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .power_up_uninitialized("FALSE"),
        .init_file("acl_fdiv_memoryC1_uid149_invTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Stratix 10")
    ) memoryC1_uid149_invTables_lutmem_dmem (
        .clocken0(en[0]),
        .sclr(memoryC1_uid149_invTables_lutmem_reset0),
        .clock0(clk),
        .address_a(memoryC1_uid149_invTables_lutmem_aa),
        .q_a(memoryC1_uid149_invTables_lutmem_ir),
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
        .aclr0(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .address2_a(),
        .address2_b(),
        .q_b(),
        .eccstatus()
    );
    assign memoryC1_uid149_invTables_lutmem_r = memoryC1_uid149_invTables_lutmem_ir[21:0];
    assign memoryC1_uid149_invTables_lutmem_enaOr_rst = en[0] | memoryC1_uid149_invTables_lutmem_reset0;

    // s1sumAHighB_uid162_invPolyEval(ADD,161)@9 + 1
    assign s1sumAHighB_uid162_invPolyEval_a = {{1{memoryC1_uid149_invTables_lutmem_r[21]}}, memoryC1_uid149_invTables_lutmem_r};
    assign s1sumAHighB_uid162_invPolyEval_b = {{10{highBBits_uid161_invPolyEval_b[12]}}, highBBits_uid161_invPolyEval_b};
    always @ (posedge clk)
    begin
        if (areset)
        begin
            s1sumAHighB_uid162_invPolyEval_o <= 23'b0;
        end
        else if (en == 1'b1)
        begin
            s1sumAHighB_uid162_invPolyEval_o <= $signed(s1sumAHighB_uid162_invPolyEval_a) + $signed(s1sumAHighB_uid162_invPolyEval_b);
        end
    end
    assign s1sumAHighB_uid162_invPolyEval_q = s1sumAHighB_uid162_invPolyEval_o[22:0];

    // lowRangeB_uid160_invPolyEval(BITSELECT,159)@9
    assign lowRangeB_uid160_invPolyEval_in = osig_uid175_pT1_uid159_invPolyEval_b[0:0];
    assign lowRangeB_uid160_invPolyEval_b = lowRangeB_uid160_invPolyEval_in[0:0];

    // redist1_lowRangeB_uid160_invPolyEval_b_1(DELAY,187)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist1_lowRangeB_uid160_invPolyEval_b_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist1_lowRangeB_uid160_invPolyEval_b_1_q <= lowRangeB_uid160_invPolyEval_b;
        end
    end

    // s1_uid163_invPolyEval(BITJOIN,162)@10
    assign s1_uid163_invPolyEval_q = {s1sumAHighB_uid162_invPolyEval_q, redist1_lowRangeB_uid160_invPolyEval_b_1_q};

    // redist17_yPE_uid52_fpDivTest_b_10_notEnable(LOGICAL,265)
    assign redist17_yPE_uid52_fpDivTest_b_10_notEnable_q = ~ (en);

    // redist17_yPE_uid52_fpDivTest_b_10_nor(LOGICAL,266)
    assign redist17_yPE_uid52_fpDivTest_b_10_nor_q = ~ (redist17_yPE_uid52_fpDivTest_b_10_notEnable_q | redist17_yPE_uid52_fpDivTest_b_10_sticky_ena_q);

    // redist17_yPE_uid52_fpDivTest_b_10_mem_last(CONSTANT,262)
    assign redist17_yPE_uid52_fpDivTest_b_10_mem_last_q = 3'b011;

    // redist17_yPE_uid52_fpDivTest_b_10_cmp(LOGICAL,263)
    assign redist17_yPE_uid52_fpDivTest_b_10_cmp_q = redist17_yPE_uid52_fpDivTest_b_10_mem_last_q == redist17_yPE_uid52_fpDivTest_b_10_rdmux_q ? 1'b1 : 1'b0;

    // redist17_yPE_uid52_fpDivTest_b_10_cmpReg(REG,264)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist17_yPE_uid52_fpDivTest_b_10_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist17_yPE_uid52_fpDivTest_b_10_cmpReg_q <= redist17_yPE_uid52_fpDivTest_b_10_cmp_q;
        end
    end

    // redist17_yPE_uid52_fpDivTest_b_10_sticky_ena(REG,267)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist17_yPE_uid52_fpDivTest_b_10_sticky_ena_q <= 1'b0;
        end
        else if (redist17_yPE_uid52_fpDivTest_b_10_nor_q == 1'b1)
        begin
            redist17_yPE_uid52_fpDivTest_b_10_sticky_ena_q <= redist17_yPE_uid52_fpDivTest_b_10_cmpReg_q;
        end
    end

    // redist17_yPE_uid52_fpDivTest_b_10_enaAnd(LOGICAL,268)
    assign redist17_yPE_uid52_fpDivTest_b_10_enaAnd_q = redist17_yPE_uid52_fpDivTest_b_10_sticky_ena_q & en;

    // redist17_yPE_uid52_fpDivTest_b_10_rdcnt(COUNTER,259)
    // low=0, high=4, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist17_yPE_uid52_fpDivTest_b_10_rdcnt_i <= 3'd0;
            redist17_yPE_uid52_fpDivTest_b_10_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist17_yPE_uid52_fpDivTest_b_10_rdcnt_i == 3'd3)
            begin
                redist17_yPE_uid52_fpDivTest_b_10_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist17_yPE_uid52_fpDivTest_b_10_rdcnt_eq <= 1'b0;
            end
            if (redist17_yPE_uid52_fpDivTest_b_10_rdcnt_eq == 1'b1)
            begin
                redist17_yPE_uid52_fpDivTest_b_10_rdcnt_i <= $unsigned(redist17_yPE_uid52_fpDivTest_b_10_rdcnt_i) + $unsigned(3'd4);
            end
            else
            begin
                redist17_yPE_uid52_fpDivTest_b_10_rdcnt_i <= $unsigned(redist17_yPE_uid52_fpDivTest_b_10_rdcnt_i) + $unsigned(3'd1);
            end
        end
    end
    assign redist17_yPE_uid52_fpDivTest_b_10_rdcnt_q = redist17_yPE_uid52_fpDivTest_b_10_rdcnt_i[2:0];

    // redist17_yPE_uid52_fpDivTest_b_10_rdmux(MUX,260)
    assign redist17_yPE_uid52_fpDivTest_b_10_rdmux_s = en;
    always @(redist17_yPE_uid52_fpDivTest_b_10_rdmux_s or redist17_yPE_uid52_fpDivTest_b_10_wraddr_q or redist17_yPE_uid52_fpDivTest_b_10_rdcnt_q)
    begin
        unique case (redist17_yPE_uid52_fpDivTest_b_10_rdmux_s)
            1'b0 : redist17_yPE_uid52_fpDivTest_b_10_rdmux_q = redist17_yPE_uid52_fpDivTest_b_10_wraddr_q;
            1'b1 : redist17_yPE_uid52_fpDivTest_b_10_rdmux_q = redist17_yPE_uid52_fpDivTest_b_10_rdcnt_q;
            default : redist17_yPE_uid52_fpDivTest_b_10_rdmux_q = 3'b0;
        endcase
    end

    // redist17_yPE_uid52_fpDivTest_b_10_wraddr(REG,261)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist17_yPE_uid52_fpDivTest_b_10_wraddr_q <= 3'b100;
        end
        else
        begin
            redist17_yPE_uid52_fpDivTest_b_10_wraddr_q <= redist17_yPE_uid52_fpDivTest_b_10_rdmux_q;
        end
    end

    // redist17_yPE_uid52_fpDivTest_b_10_mem(DUALMEM,258)
    assign redist17_yPE_uid52_fpDivTest_b_10_mem_ia = redist16_yPE_uid52_fpDivTest_b_3_q;
    assign redist17_yPE_uid52_fpDivTest_b_10_mem_aa = redist17_yPE_uid52_fpDivTest_b_10_wraddr_q;
    assign redist17_yPE_uid52_fpDivTest_b_10_mem_ab = redist17_yPE_uid52_fpDivTest_b_10_rdmux_q;
    assign redist17_yPE_uid52_fpDivTest_b_10_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(14),
        .widthad_a(3),
        .numwords_a(5),
        .width_b(14),
        .widthad_b(3),
        .numwords_b(5),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .rdcontrol_reg_b("CLOCK0"),
        .byteena_reg_b("CLOCK0"),
        .outdata_reg_b("CLOCK1"),
        .outdata_sclr_b("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .clock_enable_input_b("NORMAL"),
        .clock_enable_output_b("NORMAL"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("TRUE"),
        .intended_device_family("Stratix 10")
    ) redist17_yPE_uid52_fpDivTest_b_10_mem_dmem (
        .clocken1(redist17_yPE_uid52_fpDivTest_b_10_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist17_yPE_uid52_fpDivTest_b_10_mem_reset0),
        .clock1(clk),
        .address_a(redist17_yPE_uid52_fpDivTest_b_10_mem_aa),
        .data_a(redist17_yPE_uid52_fpDivTest_b_10_mem_ia),
        .wren_a(en[0]),
        .address_b(redist17_yPE_uid52_fpDivTest_b_10_mem_ab),
        .q_b(redist17_yPE_uid52_fpDivTest_b_10_mem_iq),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_b(),
        .clocken2(),
        .clocken3(),
        .aclr0(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .address2_a(),
        .address2_b(),
        .q_a(),
        .eccstatus()
    );
    assign redist17_yPE_uid52_fpDivTest_b_10_mem_q = redist17_yPE_uid52_fpDivTest_b_10_mem_iq[13:0];
    assign redist17_yPE_uid52_fpDivTest_b_10_mem_enaOr_rst = redist17_yPE_uid52_fpDivTest_b_10_enaAnd_q[0] | redist17_yPE_uid52_fpDivTest_b_10_mem_reset0;

    // redist17_yPE_uid52_fpDivTest_b_10_outputreg0(DELAY,257)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist17_yPE_uid52_fpDivTest_b_10_outputreg0_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist17_yPE_uid52_fpDivTest_b_10_outputreg0_q <= redist17_yPE_uid52_fpDivTest_b_10_mem_q;
        end
    end

    // prodXY_uid177_pT2_uid165_invPolyEval_cma(CHAINMULTADD,185)@10 + 5
    // out q@16
    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_reset = areset;
    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_ena0 = en[0] | prodXY_uid177_pT2_uid165_invPolyEval_cma_reset;
    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_ena1 = prodXY_uid177_pT2_uid165_invPolyEval_cma_ena0;
    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_ena2 = prodXY_uid177_pT2_uid165_invPolyEval_cma_ena0;
    always @ (posedge clk)
    begin
        if (0)
        begin
        end
        else
        begin
            if (en == 1'b1)
            begin
                prodXY_uid177_pT2_uid165_invPolyEval_cma_ah[0] <= redist17_yPE_uid52_fpDivTest_b_10_outputreg0_q;
                prodXY_uid177_pT2_uid165_invPolyEval_cma_ch[0] <= s1_uid163_invPolyEval_q;
            end
        end
    end

    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_a0 = prodXY_uid177_pT2_uid165_invPolyEval_cma_ah[0];
    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_c0 = prodXY_uid177_pT2_uid165_invPolyEval_cma_ch[0];
    fourteennm_mac #(
        .operation_mode("m27x27"),
        .clear_type("sclr"),
        .use_chainadder("false"),
        .ay_scan_in_clock("0"),
        .ay_scan_in_width(14),
        .ax_clock("0"),
        .ax_width(24),
        .signed_may("false"),
        .signed_max("true"),
        .input_pipeline_clock("2"),
        .second_pipeline_clock("2"),
        .output_clock("1"),
        .result_a_width(38)
    ) prodXY_uid177_pT2_uid165_invPolyEval_cma_DSP0 (
        .clk({clk,clk,clk}),
        .ena({ prodXY_uid177_pT2_uid165_invPolyEval_cma_ena2, prodXY_uid177_pT2_uid165_invPolyEval_cma_ena1, prodXY_uid177_pT2_uid165_invPolyEval_cma_ena0 }),
        .clr({ prodXY_uid177_pT2_uid165_invPolyEval_cma_reset, prodXY_uid177_pT2_uid165_invPolyEval_cma_reset }),
        .ay(prodXY_uid177_pT2_uid165_invPolyEval_cma_a0),
        .ax(prodXY_uid177_pT2_uid165_invPolyEval_cma_c0),
        .resulta(prodXY_uid177_pT2_uid165_invPolyEval_cma_s0),
        .accumulate(),
        .loadconst(),
        .negate(),
        .sub(),
        .az(),
        .coefsela(),
        .bx(),
        .by(),
        .bz(),
        .coefselb(),
        .scanin(),
        .scanout(),
        .chainin(),
        .chainout(),
        .resultb(),
        .dfxlfsrena(),
        .dfxmisrena(),
        .dftout()
    );
    dspba_delay_ver #( .width(38), .depth(1), .reset_kind("NONE"), .phase(0), .modulus(1) )
    prodXY_uid177_pT2_uid165_invPolyEval_cma_delay ( .xin(prodXY_uid177_pT2_uid165_invPolyEval_cma_s0), .xout(prodXY_uid177_pT2_uid165_invPolyEval_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_q = prodXY_uid177_pT2_uid165_invPolyEval_cma_qq[37:0];

    // osig_uid178_pT2_uid165_invPolyEval(BITSELECT,177)@16
    assign osig_uid178_pT2_uid165_invPolyEval_b = prodXY_uid177_pT2_uid165_invPolyEval_cma_q[37:13];

    // highBBits_uid167_invPolyEval(BITSELECT,166)@16
    assign highBBits_uid167_invPolyEval_b = osig_uid178_pT2_uid165_invPolyEval_b[24:2];

    // redist19_yAddr_uid51_fpDivTest_b_14_notEnable(LOGICAL,289)
    assign redist19_yAddr_uid51_fpDivTest_b_14_notEnable_q = ~ (en);

    // redist19_yAddr_uid51_fpDivTest_b_14_nor(LOGICAL,290)
    assign redist19_yAddr_uid51_fpDivTest_b_14_nor_q = ~ (redist19_yAddr_uid51_fpDivTest_b_14_notEnable_q | redist19_yAddr_uid51_fpDivTest_b_14_sticky_ena_q);

    // redist19_yAddr_uid51_fpDivTest_b_14_mem_last(CONSTANT,286)
    assign redist19_yAddr_uid51_fpDivTest_b_14_mem_last_q = 3'b011;

    // redist19_yAddr_uid51_fpDivTest_b_14_cmp(LOGICAL,287)
    assign redist19_yAddr_uid51_fpDivTest_b_14_cmp_q = redist19_yAddr_uid51_fpDivTest_b_14_mem_last_q == redist19_yAddr_uid51_fpDivTest_b_14_rdmux_q ? 1'b1 : 1'b0;

    // redist19_yAddr_uid51_fpDivTest_b_14_cmpReg(REG,288)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist19_yAddr_uid51_fpDivTest_b_14_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist19_yAddr_uid51_fpDivTest_b_14_cmpReg_q <= redist19_yAddr_uid51_fpDivTest_b_14_cmp_q;
        end
    end

    // redist19_yAddr_uid51_fpDivTest_b_14_sticky_ena(REG,291)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist19_yAddr_uid51_fpDivTest_b_14_sticky_ena_q <= 1'b0;
        end
        else if (redist19_yAddr_uid51_fpDivTest_b_14_nor_q == 1'b1)
        begin
            redist19_yAddr_uid51_fpDivTest_b_14_sticky_ena_q <= redist19_yAddr_uid51_fpDivTest_b_14_cmpReg_q;
        end
    end

    // redist19_yAddr_uid51_fpDivTest_b_14_enaAnd(LOGICAL,292)
    assign redist19_yAddr_uid51_fpDivTest_b_14_enaAnd_q = redist19_yAddr_uid51_fpDivTest_b_14_sticky_ena_q & en;

    // redist19_yAddr_uid51_fpDivTest_b_14_rdcnt(COUNTER,283)
    // low=0, high=4, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist19_yAddr_uid51_fpDivTest_b_14_rdcnt_i <= 3'd0;
            redist19_yAddr_uid51_fpDivTest_b_14_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist19_yAddr_uid51_fpDivTest_b_14_rdcnt_i == 3'd3)
            begin
                redist19_yAddr_uid51_fpDivTest_b_14_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist19_yAddr_uid51_fpDivTest_b_14_rdcnt_eq <= 1'b0;
            end
            if (redist19_yAddr_uid51_fpDivTest_b_14_rdcnt_eq == 1'b1)
            begin
                redist19_yAddr_uid51_fpDivTest_b_14_rdcnt_i <= $unsigned(redist19_yAddr_uid51_fpDivTest_b_14_rdcnt_i) + $unsigned(3'd4);
            end
            else
            begin
                redist19_yAddr_uid51_fpDivTest_b_14_rdcnt_i <= $unsigned(redist19_yAddr_uid51_fpDivTest_b_14_rdcnt_i) + $unsigned(3'd1);
            end
        end
    end
    assign redist19_yAddr_uid51_fpDivTest_b_14_rdcnt_q = redist19_yAddr_uid51_fpDivTest_b_14_rdcnt_i[2:0];

    // redist19_yAddr_uid51_fpDivTest_b_14_rdmux(MUX,284)
    assign redist19_yAddr_uid51_fpDivTest_b_14_rdmux_s = en;
    always @(redist19_yAddr_uid51_fpDivTest_b_14_rdmux_s or redist19_yAddr_uid51_fpDivTest_b_14_wraddr_q or redist19_yAddr_uid51_fpDivTest_b_14_rdcnt_q)
    begin
        unique case (redist19_yAddr_uid51_fpDivTest_b_14_rdmux_s)
            1'b0 : redist19_yAddr_uid51_fpDivTest_b_14_rdmux_q = redist19_yAddr_uid51_fpDivTest_b_14_wraddr_q;
            1'b1 : redist19_yAddr_uid51_fpDivTest_b_14_rdmux_q = redist19_yAddr_uid51_fpDivTest_b_14_rdcnt_q;
            default : redist19_yAddr_uid51_fpDivTest_b_14_rdmux_q = 3'b0;
        endcase
    end

    // redist19_yAddr_uid51_fpDivTest_b_14_wraddr(REG,285)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist19_yAddr_uid51_fpDivTest_b_14_wraddr_q <= 3'b100;
        end
        else
        begin
            redist19_yAddr_uid51_fpDivTest_b_14_wraddr_q <= redist19_yAddr_uid51_fpDivTest_b_14_rdmux_q;
        end
    end

    // redist19_yAddr_uid51_fpDivTest_b_14_mem(DUALMEM,282)
    assign redist19_yAddr_uid51_fpDivTest_b_14_mem_ia = redist18_yAddr_uid51_fpDivTest_b_7_outputreg0_q;
    assign redist19_yAddr_uid51_fpDivTest_b_14_mem_aa = redist19_yAddr_uid51_fpDivTest_b_14_wraddr_q;
    assign redist19_yAddr_uid51_fpDivTest_b_14_mem_ab = redist19_yAddr_uid51_fpDivTest_b_14_rdmux_q;
    assign redist19_yAddr_uid51_fpDivTest_b_14_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(9),
        .widthad_a(3),
        .numwords_a(5),
        .width_b(9),
        .widthad_b(3),
        .numwords_b(5),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .rdcontrol_reg_b("CLOCK0"),
        .byteena_reg_b("CLOCK0"),
        .outdata_reg_b("CLOCK1"),
        .outdata_sclr_b("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .clock_enable_input_b("NORMAL"),
        .clock_enable_output_b("NORMAL"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("TRUE"),
        .intended_device_family("Stratix 10")
    ) redist19_yAddr_uid51_fpDivTest_b_14_mem_dmem (
        .clocken1(redist19_yAddr_uid51_fpDivTest_b_14_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist19_yAddr_uid51_fpDivTest_b_14_mem_reset0),
        .clock1(clk),
        .address_a(redist19_yAddr_uid51_fpDivTest_b_14_mem_aa),
        .data_a(redist19_yAddr_uid51_fpDivTest_b_14_mem_ia),
        .wren_a(en[0]),
        .address_b(redist19_yAddr_uid51_fpDivTest_b_14_mem_ab),
        .q_b(redist19_yAddr_uid51_fpDivTest_b_14_mem_iq),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_b(),
        .clocken2(),
        .clocken3(),
        .aclr0(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .address2_a(),
        .address2_b(),
        .q_a(),
        .eccstatus()
    );
    assign redist19_yAddr_uid51_fpDivTest_b_14_mem_q = redist19_yAddr_uid51_fpDivTest_b_14_mem_iq[8:0];
    assign redist19_yAddr_uid51_fpDivTest_b_14_mem_enaOr_rst = redist19_yAddr_uid51_fpDivTest_b_14_enaAnd_q[0] | redist19_yAddr_uid51_fpDivTest_b_14_mem_reset0;

    // redist19_yAddr_uid51_fpDivTest_b_14_outputreg0(DELAY,281)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist19_yAddr_uid51_fpDivTest_b_14_outputreg0_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist19_yAddr_uid51_fpDivTest_b_14_outputreg0_q <= redist19_yAddr_uid51_fpDivTest_b_14_mem_q;
        end
    end

    // memoryC0_uid146_invTables_lutmem(DUALMEM,179)@14 + 2
    // in j@20000000
    assign memoryC0_uid146_invTables_lutmem_aa = redist19_yAddr_uid51_fpDivTest_b_14_outputreg0_q;
    assign memoryC0_uid146_invTables_lutmem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("M20K"),
        .operation_mode("ROM"),
        .width_a(32),
        .widthad_a(9),
        .numwords_a(512),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .outdata_reg_a("CLOCK0"),
        .outdata_sclr_a("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .power_up_uninitialized("FALSE"),
        .init_file("acl_fdiv_memoryC0_uid146_invTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Stratix 10")
    ) memoryC0_uid146_invTables_lutmem_dmem (
        .clocken0(en[0]),
        .sclr(memoryC0_uid146_invTables_lutmem_reset0),
        .clock0(clk),
        .address_a(memoryC0_uid146_invTables_lutmem_aa),
        .q_a(memoryC0_uid146_invTables_lutmem_ir),
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
        .aclr0(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .address2_a(),
        .address2_b(),
        .q_b(),
        .eccstatus()
    );
    assign memoryC0_uid146_invTables_lutmem_r = memoryC0_uid146_invTables_lutmem_ir[31:0];
    assign memoryC0_uid146_invTables_lutmem_enaOr_rst = en[0] | memoryC0_uid146_invTables_lutmem_reset0;

    // s2sumAHighB_uid168_invPolyEval(ADD,167)@16
    assign s2sumAHighB_uid168_invPolyEval_a = {{1{memoryC0_uid146_invTables_lutmem_r[31]}}, memoryC0_uid146_invTables_lutmem_r};
    assign s2sumAHighB_uid168_invPolyEval_b = {{10{highBBits_uid167_invPolyEval_b[22]}}, highBBits_uid167_invPolyEval_b};
    assign s2sumAHighB_uid168_invPolyEval_o = $signed(s2sumAHighB_uid168_invPolyEval_a) + $signed(s2sumAHighB_uid168_invPolyEval_b);
    assign s2sumAHighB_uid168_invPolyEval_q = s2sumAHighB_uid168_invPolyEval_o[32:0];

    // lowRangeB_uid166_invPolyEval(BITSELECT,165)@16
    assign lowRangeB_uid166_invPolyEval_in = osig_uid178_pT2_uid165_invPolyEval_b[1:0];
    assign lowRangeB_uid166_invPolyEval_b = lowRangeB_uid166_invPolyEval_in[1:0];

    // s2_uid169_invPolyEval(BITJOIN,168)@16
    assign s2_uid169_invPolyEval_q = {s2sumAHighB_uid168_invPolyEval_q, lowRangeB_uid166_invPolyEval_b};

    // invY_uid54_fpDivTest(BITSELECT,53)@16
    assign invY_uid54_fpDivTest_in = s2_uid169_invPolyEval_q[31:0];
    assign invY_uid54_fpDivTest_b = invY_uid54_fpDivTest_in[31:5];

    // redist15_invY_uid54_fpDivTest_b_1(DELAY,201)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist15_invY_uid54_fpDivTest_b_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist15_invY_uid54_fpDivTest_b_1_q <= invY_uid54_fpDivTest_b;
        end
    end

    // prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma(CHAINMULTADD,183)@17 + 5
    // out q@23
    assign prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_reset = areset;
    assign prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena0 = en[0] | prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_reset;
    assign prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena1 = prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena0;
    assign prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena2 = prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena0;
    always @ (posedge clk)
    begin
        if (0)
        begin
        end
        else
        begin
            if (en == 1'b1)
            begin
                prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ah[0] <= redist15_invY_uid54_fpDivTest_b_1_q;
                prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ch[0] <= lOAdded_uid57_fpDivTest_q;
            end
        end
    end

    assign prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_a0 = prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ah[0];
    assign prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_c0 = prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ch[0];
    fourteennm_mac #(
        .operation_mode("m27x27"),
        .clear_type("sclr"),
        .use_chainadder("false"),
        .ay_scan_in_clock("0"),
        .ay_scan_in_width(27),
        .ax_clock("0"),
        .ax_width(24),
        .signed_may("false"),
        .signed_max("false"),
        .input_pipeline_clock("2"),
        .second_pipeline_clock("2"),
        .output_clock("1"),
        .result_a_width(51)
    ) prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_DSP0 (
        .clk({clk,clk,clk}),
        .ena({ prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena2, prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena1, prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena0 }),
        .clr({ prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_reset, prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_reset }),
        .ay(prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_a0),
        .ax(prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_c0),
        .resulta(prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_s0),
        .accumulate(),
        .loadconst(),
        .negate(),
        .sub(),
        .az(),
        .coefsela(),
        .bx(),
        .by(),
        .bz(),
        .coefselb(),
        .scanin(),
        .scanout(),
        .chainin(),
        .chainout(),
        .resultb(),
        .dfxlfsrena(),
        .dfxmisrena(),
        .dftout()
    );
    dspba_delay_ver #( .width(51), .depth(1), .reset_kind("NONE"), .phase(0), .modulus(1) )
    prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_delay ( .xin(prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_s0), .xout(prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_q = prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_qq[50:0];

    // osig_uid172_divValPreNorm_uid59_fpDivTest(BITSELECT,171)@23
    assign osig_uid172_divValPreNorm_uid59_fpDivTest_b = prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_q[50:23];

    // updatedY_uid16_fpDivTest(BITJOIN,15)@22
    assign updatedY_uid16_fpDivTest_q = {GND_q, paddingY_uid15_fpDivTest_q};

    // fracYZero_uid15_fpDivTest(LOGICAL,16)@22 + 1
    assign fracYZero_uid15_fpDivTest_a = {1'b0, redist23_fracY_uid13_fpDivTest_b_22_mem_q};
    assign fracYZero_uid15_fpDivTest_qi = fracYZero_uid15_fpDivTest_a == updatedY_uid16_fpDivTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    fracYZero_uid15_fpDivTest_delay ( .xin(fracYZero_uid15_fpDivTest_qi), .xout(fracYZero_uid15_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // divValPreNormYPow2Exc_uid63_fpDivTest(MUX,62)@23
    assign divValPreNormYPow2Exc_uid63_fpDivTest_s = fracYZero_uid15_fpDivTest_q;
    always @(divValPreNormYPow2Exc_uid63_fpDivTest_s or en or osig_uid172_divValPreNorm_uid59_fpDivTest_b or oFracXZ4_uid61_fpDivTest_q)
    begin
        unique case (divValPreNormYPow2Exc_uid63_fpDivTest_s)
            1'b0 : divValPreNormYPow2Exc_uid63_fpDivTest_q = osig_uid172_divValPreNorm_uid59_fpDivTest_b;
            1'b1 : divValPreNormYPow2Exc_uid63_fpDivTest_q = oFracXZ4_uid61_fpDivTest_q;
            default : divValPreNormYPow2Exc_uid63_fpDivTest_q = 28'b0;
        endcase
    end

    // norm_uid64_fpDivTest(BITSELECT,63)@23
    assign norm_uid64_fpDivTest_b = divValPreNormYPow2Exc_uid63_fpDivTest_q[27:27];

    // redist11_norm_uid64_fpDivTest_b_1(DELAY,197)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist11_norm_uid64_fpDivTest_b_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist11_norm_uid64_fpDivTest_b_1_q <= norm_uid64_fpDivTest_b;
        end
    end

    // zeroPaddingInAddition_uid74_fpDivTest(CONSTANT,73)
    assign zeroPaddingInAddition_uid74_fpDivTest_q = 24'b000000000000000000000000;

    // expFracPostRnd_uid75_fpDivTest(BITJOIN,74)@24
    assign expFracPostRnd_uid75_fpDivTest_q = {redist11_norm_uid64_fpDivTest_b_1_q, zeroPaddingInAddition_uid74_fpDivTest_q, VCC_q};

    // cstBiasM1_uid6_fpDivTest(CONSTANT,5)
    assign cstBiasM1_uid6_fpDivTest_q = 8'b01111110;

    // expXmY_uid47_fpDivTest(SUB,46)@23
    assign expXmY_uid47_fpDivTest_a = {1'b0, redist33_expX_uid9_fpDivTest_b_23_mem_q};
    assign expXmY_uid47_fpDivTest_b = {1'b0, redist26_expY_uid12_fpDivTest_b_23_mem_q};
    assign expXmY_uid47_fpDivTest_o = $unsigned(expXmY_uid47_fpDivTest_a) - $unsigned(expXmY_uid47_fpDivTest_b);
    assign expXmY_uid47_fpDivTest_q = expXmY_uid47_fpDivTest_o[8:0];

    // expR_uid48_fpDivTest(ADD,47)@23 + 1
    assign expR_uid48_fpDivTest_a = {{2{expXmY_uid47_fpDivTest_q[8]}}, expXmY_uid47_fpDivTest_q};
    assign expR_uid48_fpDivTest_b = {3'b000, cstBiasM1_uid6_fpDivTest_q};
    always @ (posedge clk)
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

    // divValPreNormHigh_uid65_fpDivTest(BITSELECT,64)@23
    assign divValPreNormHigh_uid65_fpDivTest_in = divValPreNormYPow2Exc_uid63_fpDivTest_q[26:0];
    assign divValPreNormHigh_uid65_fpDivTest_b = divValPreNormHigh_uid65_fpDivTest_in[26:2];

    // divValPreNormLow_uid66_fpDivTest(BITSELECT,65)@23
    assign divValPreNormLow_uid66_fpDivTest_in = divValPreNormYPow2Exc_uid63_fpDivTest_q[25:0];
    assign divValPreNormLow_uid66_fpDivTest_b = divValPreNormLow_uid66_fpDivTest_in[25:1];

    // normFracRnd_uid67_fpDivTest(MUX,66)@23 + 1
    assign normFracRnd_uid67_fpDivTest_s = norm_uid64_fpDivTest_b;
    always @ (posedge clk)
    begin
        if (areset)
        begin
            normFracRnd_uid67_fpDivTest_q <= 25'b0;
        end
        else if (en == 1'b1)
        begin
            unique case (normFracRnd_uid67_fpDivTest_s)
                1'b0 : normFracRnd_uid67_fpDivTest_q <= divValPreNormLow_uid66_fpDivTest_b;
                1'b1 : normFracRnd_uid67_fpDivTest_q <= divValPreNormHigh_uid65_fpDivTest_b;
                default : normFracRnd_uid67_fpDivTest_q <= 25'b0;
            endcase
        end
    end

    // expFracRnd_uid68_fpDivTest(BITJOIN,67)@24
    assign expFracRnd_uid68_fpDivTest_q = {expR_uid48_fpDivTest_q, normFracRnd_uid67_fpDivTest_q};

    // expFracPostRnd_uid76_fpDivTest(ADD,75)@24
    assign expFracPostRnd_uid76_fpDivTest_a = {{2{expFracRnd_uid68_fpDivTest_q[34]}}, expFracRnd_uid68_fpDivTest_q};
    assign expFracPostRnd_uid76_fpDivTest_b = {11'b00000000000, expFracPostRnd_uid75_fpDivTest_q};
    assign expFracPostRnd_uid76_fpDivTest_o = $signed(expFracPostRnd_uid76_fpDivTest_a) + $signed(expFracPostRnd_uid76_fpDivTest_b);
    assign expFracPostRnd_uid76_fpDivTest_q = expFracPostRnd_uid76_fpDivTest_o[35:0];

    // fracPostRndF_uid79_fpDivTest(BITSELECT,78)@24
    assign fracPostRndF_uid79_fpDivTest_in = expFracPostRnd_uid76_fpDivTest_q[24:0];
    assign fracPostRndF_uid79_fpDivTest_b = fracPostRndF_uid79_fpDivTest_in[24:1];

    // redist10_fracPostRndF_uid79_fpDivTest_b_1(DELAY,196)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist10_fracPostRndF_uid79_fpDivTest_b_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist10_fracPostRndF_uid79_fpDivTest_b_1_q <= fracPostRndF_uid79_fpDivTest_b;
        end
    end

    // invYO_uid55_fpDivTest(BITSELECT,54)@16
    assign invYO_uid55_fpDivTest_in = s2_uid169_invPolyEval_q[32:0];
    assign invYO_uid55_fpDivTest_b = invYO_uid55_fpDivTest_in[32:32];

    // redist13_invYO_uid55_fpDivTest_b_9(DELAY,199)
    dspba_delay_ver #( .width(1), .depth(9), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    redist13_invYO_uid55_fpDivTest_b_9 ( .xin(invYO_uid55_fpDivTest_b), .xout(redist13_invYO_uid55_fpDivTest_b_9_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // fracPostRndF_uid80_fpDivTest(MUX,79)@25
    assign fracPostRndF_uid80_fpDivTest_s = redist13_invYO_uid55_fpDivTest_b_9_q;
    always @(fracPostRndF_uid80_fpDivTest_s or en or redist10_fracPostRndF_uid79_fpDivTest_b_1_q or fracXExt_uid77_fpDivTest_q)
    begin
        unique case (fracPostRndF_uid80_fpDivTest_s)
            1'b0 : fracPostRndF_uid80_fpDivTest_q = redist10_fracPostRndF_uid79_fpDivTest_b_1_q;
            1'b1 : fracPostRndF_uid80_fpDivTest_q = fracXExt_uid77_fpDivTest_q;
            default : fracPostRndF_uid80_fpDivTest_q = 24'b0;
        endcase
    end

    // fracPostRndFT_uid104_fpDivTest(BITSELECT,103)@25
    assign fracPostRndFT_uid104_fpDivTest_b = fracPostRndF_uid80_fpDivTest_q[23:1];

    // redist4_fracPostRndFT_uid104_fpDivTest_b_8_wraddr(REG,226)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist4_fracPostRndFT_uid104_fpDivTest_b_8_wraddr_q <= 3'b110;
        end
        else
        begin
            redist4_fracPostRndFT_uid104_fpDivTest_b_8_wraddr_q <= redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdmux_q;
        end
    end

    // redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem(DUALMEM,223)
    assign redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_ia = fracPostRndFT_uid104_fpDivTest_b;
    assign redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_aa = redist4_fracPostRndFT_uid104_fpDivTest_b_8_wraddr_q;
    assign redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_ab = redist4_fracPostRndFT_uid104_fpDivTest_b_8_rdmux_q;
    assign redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(23),
        .widthad_a(3),
        .numwords_a(7),
        .width_b(23),
        .widthad_b(3),
        .numwords_b(7),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .rdcontrol_reg_b("CLOCK0"),
        .byteena_reg_b("CLOCK0"),
        .outdata_reg_b("CLOCK1"),
        .outdata_sclr_b("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .clock_enable_input_b("NORMAL"),
        .clock_enable_output_b("NORMAL"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("TRUE"),
        .intended_device_family("Stratix 10")
    ) redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_dmem (
        .clocken1(redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_reset0),
        .clock1(clk),
        .address_a(redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_aa),
        .data_a(redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_ia),
        .wren_a(en[0]),
        .address_b(redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_ab),
        .q_b(redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_iq),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_b(),
        .clocken2(),
        .clocken3(),
        .aclr0(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .address2_a(),
        .address2_b(),
        .q_a(),
        .eccstatus()
    );
    assign redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_q = redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_iq[22:0];
    assign redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_enaOr_rst = redist4_fracPostRndFT_uid104_fpDivTest_b_8_enaAnd_q[0] | redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_reset0;

    // fracRPreExcExt_uid105_fpDivTest(ADD,104)@33
    assign fracRPreExcExt_uid105_fpDivTest_a = {1'b0, redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_q};
    assign fracRPreExcExt_uid105_fpDivTest_b = {23'b00000000000000000000000, extraUlp_uid103_fpDivTest_q};
    assign fracRPreExcExt_uid105_fpDivTest_o = $unsigned(fracRPreExcExt_uid105_fpDivTest_a) + $unsigned(fracRPreExcExt_uid105_fpDivTest_b);
    assign fracRPreExcExt_uid105_fpDivTest_q = fracRPreExcExt_uid105_fpDivTest_o[23:0];

    // ovfIncRnd_uid109_fpDivTest(BITSELECT,108)@33
    assign ovfIncRnd_uid109_fpDivTest_b = fracRPreExcExt_uid105_fpDivTest_q[23:23];

    // expFracPostRndInc_uid110_fpDivTest(ADD,109)@33
    assign expFracPostRndInc_uid110_fpDivTest_a = {1'b0, redist9_expPostRndFR_uid81_fpDivTest_b_9_q};
    assign expFracPostRndInc_uid110_fpDivTest_b = {8'b00000000, ovfIncRnd_uid109_fpDivTest_b};
    assign expFracPostRndInc_uid110_fpDivTest_o = $unsigned(expFracPostRndInc_uid110_fpDivTest_a) + $unsigned(expFracPostRndInc_uid110_fpDivTest_b);
    assign expFracPostRndInc_uid110_fpDivTest_q = expFracPostRndInc_uid110_fpDivTest_o[8:0];

    // expFracPostRndR_uid111_fpDivTest(BITSELECT,110)@33
    assign expFracPostRndR_uid111_fpDivTest_in = expFracPostRndInc_uid110_fpDivTest_q[7:0];
    assign expFracPostRndR_uid111_fpDivTest_b = expFracPostRndR_uid111_fpDivTest_in[7:0];

    // redist8_expPostRndFR_uid81_fpDivTest_b_7_notEnable(LOGICAL,242)
    assign redist8_expPostRndFR_uid81_fpDivTest_b_7_notEnable_q = ~ (en);

    // redist8_expPostRndFR_uid81_fpDivTest_b_7_nor(LOGICAL,243)
    assign redist8_expPostRndFR_uid81_fpDivTest_b_7_nor_q = ~ (redist8_expPostRndFR_uid81_fpDivTest_b_7_notEnable_q | redist8_expPostRndFR_uid81_fpDivTest_b_7_sticky_ena_q);

    // redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_last(CONSTANT,239)
    assign redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_last_q = 3'b011;

    // redist8_expPostRndFR_uid81_fpDivTest_b_7_cmp(LOGICAL,240)
    assign redist8_expPostRndFR_uid81_fpDivTest_b_7_cmp_q = redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_last_q == redist8_expPostRndFR_uid81_fpDivTest_b_7_rdmux_q ? 1'b1 : 1'b0;

    // redist8_expPostRndFR_uid81_fpDivTest_b_7_cmpReg(REG,241)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist8_expPostRndFR_uid81_fpDivTest_b_7_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist8_expPostRndFR_uid81_fpDivTest_b_7_cmpReg_q <= redist8_expPostRndFR_uid81_fpDivTest_b_7_cmp_q;
        end
    end

    // redist8_expPostRndFR_uid81_fpDivTest_b_7_sticky_ena(REG,244)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist8_expPostRndFR_uid81_fpDivTest_b_7_sticky_ena_q <= 1'b0;
        end
        else if (redist8_expPostRndFR_uid81_fpDivTest_b_7_nor_q == 1'b1)
        begin
            redist8_expPostRndFR_uid81_fpDivTest_b_7_sticky_ena_q <= redist8_expPostRndFR_uid81_fpDivTest_b_7_cmpReg_q;
        end
    end

    // redist8_expPostRndFR_uid81_fpDivTest_b_7_enaAnd(LOGICAL,245)
    assign redist8_expPostRndFR_uid81_fpDivTest_b_7_enaAnd_q = redist8_expPostRndFR_uid81_fpDivTest_b_7_sticky_ena_q & en;

    // redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt(COUNTER,236)
    // low=0, high=4, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt_i <= 3'd0;
            redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt_i == 3'd3)
            begin
                redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt_eq <= 1'b0;
            end
            if (redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt_eq == 1'b1)
            begin
                redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt_i <= $unsigned(redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt_i) + $unsigned(3'd4);
            end
            else
            begin
                redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt_i <= $unsigned(redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt_i) + $unsigned(3'd1);
            end
        end
    end
    assign redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt_q = redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt_i[2:0];

    // redist8_expPostRndFR_uid81_fpDivTest_b_7_rdmux(MUX,237)
    assign redist8_expPostRndFR_uid81_fpDivTest_b_7_rdmux_s = en;
    always @(redist8_expPostRndFR_uid81_fpDivTest_b_7_rdmux_s or redist8_expPostRndFR_uid81_fpDivTest_b_7_wraddr_q or redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt_q)
    begin
        unique case (redist8_expPostRndFR_uid81_fpDivTest_b_7_rdmux_s)
            1'b0 : redist8_expPostRndFR_uid81_fpDivTest_b_7_rdmux_q = redist8_expPostRndFR_uid81_fpDivTest_b_7_wraddr_q;
            1'b1 : redist8_expPostRndFR_uid81_fpDivTest_b_7_rdmux_q = redist8_expPostRndFR_uid81_fpDivTest_b_7_rdcnt_q;
            default : redist8_expPostRndFR_uid81_fpDivTest_b_7_rdmux_q = 3'b0;
        endcase
    end

    // expPostRndFR_uid81_fpDivTest(BITSELECT,80)@24
    assign expPostRndFR_uid81_fpDivTest_in = expFracPostRnd_uid76_fpDivTest_q[32:0];
    assign expPostRndFR_uid81_fpDivTest_b = expPostRndFR_uid81_fpDivTest_in[32:25];

    // redist8_expPostRndFR_uid81_fpDivTest_b_7_wraddr(REG,238)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist8_expPostRndFR_uid81_fpDivTest_b_7_wraddr_q <= 3'b100;
        end
        else
        begin
            redist8_expPostRndFR_uid81_fpDivTest_b_7_wraddr_q <= redist8_expPostRndFR_uid81_fpDivTest_b_7_rdmux_q;
        end
    end

    // redist8_expPostRndFR_uid81_fpDivTest_b_7_mem(DUALMEM,235)
    assign redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_ia = expPostRndFR_uid81_fpDivTest_b;
    assign redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_aa = redist8_expPostRndFR_uid81_fpDivTest_b_7_wraddr_q;
    assign redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_ab = redist8_expPostRndFR_uid81_fpDivTest_b_7_rdmux_q;
    assign redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(8),
        .widthad_a(3),
        .numwords_a(5),
        .width_b(8),
        .widthad_b(3),
        .numwords_b(5),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .rdcontrol_reg_b("CLOCK0"),
        .byteena_reg_b("CLOCK0"),
        .outdata_reg_b("CLOCK1"),
        .outdata_sclr_b("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .clock_enable_input_b("NORMAL"),
        .clock_enable_output_b("NORMAL"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("TRUE"),
        .intended_device_family("Stratix 10")
    ) redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_dmem (
        .clocken1(redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_reset0),
        .clock1(clk),
        .address_a(redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_aa),
        .data_a(redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_ia),
        .wren_a(en[0]),
        .address_b(redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_ab),
        .q_b(redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_iq),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_b(),
        .clocken2(),
        .clocken3(),
        .aclr0(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .address2_a(),
        .address2_b(),
        .q_a(),
        .eccstatus()
    );
    assign redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_q = redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_iq[7:0];
    assign redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_enaOr_rst = redist8_expPostRndFR_uid81_fpDivTest_b_7_enaAnd_q[0] | redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_reset0;

    // redist8_expPostRndFR_uid81_fpDivTest_b_7_outputreg0(DELAY,234)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist8_expPostRndFR_uid81_fpDivTest_b_7_outputreg0_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist8_expPostRndFR_uid81_fpDivTest_b_7_outputreg0_q <= redist8_expPostRndFR_uid81_fpDivTest_b_7_mem_q;
        end
    end

    // redist9_expPostRndFR_uid81_fpDivTest_b_9(DELAY,195)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist9_expPostRndFR_uid81_fpDivTest_b_9_delay_0 <= '0;
            redist9_expPostRndFR_uid81_fpDivTest_b_9_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist9_expPostRndFR_uid81_fpDivTest_b_9_delay_0 <= redist8_expPostRndFR_uid81_fpDivTest_b_7_outputreg0_q;
            redist9_expPostRndFR_uid81_fpDivTest_b_9_q <= redist9_expPostRndFR_uid81_fpDivTest_b_9_delay_0;
        end
    end

    // betweenFPwF_uid102_fpDivTest(BITSELECT,101)@25
    assign betweenFPwF_uid102_fpDivTest_in = fracPostRndF_uid80_fpDivTest_q[0:0];
    assign betweenFPwF_uid102_fpDivTest_b = betweenFPwF_uid102_fpDivTest_in[0:0];

    // redist5_betweenFPwF_uid102_fpDivTest_b_7(DELAY,191)
    dspba_delay_ver #( .width(1), .depth(7), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    redist5_betweenFPwF_uid102_fpDivTest_b_7 ( .xin(betweenFPwF_uid102_fpDivTest_b), .xout(redist5_betweenFPwF_uid102_fpDivTest_b_7_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist35_expX_uid9_fpDivTest_b_31_notEnable(LOGICAL,380)
    assign redist35_expX_uid9_fpDivTest_b_31_notEnable_q = ~ (en);

    // redist35_expX_uid9_fpDivTest_b_31_nor(LOGICAL,381)
    assign redist35_expX_uid9_fpDivTest_b_31_nor_q = ~ (redist35_expX_uid9_fpDivTest_b_31_notEnable_q | redist35_expX_uid9_fpDivTest_b_31_sticky_ena_q);

    // redist35_expX_uid9_fpDivTest_b_31_mem_last(CONSTANT,377)
    assign redist35_expX_uid9_fpDivTest_b_31_mem_last_q = 3'b011;

    // redist35_expX_uid9_fpDivTest_b_31_cmp(LOGICAL,378)
    assign redist35_expX_uid9_fpDivTest_b_31_cmp_q = redist35_expX_uid9_fpDivTest_b_31_mem_last_q == redist35_expX_uid9_fpDivTest_b_31_rdmux_q ? 1'b1 : 1'b0;

    // redist35_expX_uid9_fpDivTest_b_31_cmpReg(REG,379)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist35_expX_uid9_fpDivTest_b_31_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist35_expX_uid9_fpDivTest_b_31_cmpReg_q <= redist35_expX_uid9_fpDivTest_b_31_cmp_q;
        end
    end

    // redist35_expX_uid9_fpDivTest_b_31_sticky_ena(REG,382)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist35_expX_uid9_fpDivTest_b_31_sticky_ena_q <= 1'b0;
        end
        else if (redist35_expX_uid9_fpDivTest_b_31_nor_q == 1'b1)
        begin
            redist35_expX_uid9_fpDivTest_b_31_sticky_ena_q <= redist35_expX_uid9_fpDivTest_b_31_cmpReg_q;
        end
    end

    // redist35_expX_uid9_fpDivTest_b_31_enaAnd(LOGICAL,383)
    assign redist35_expX_uid9_fpDivTest_b_31_enaAnd_q = redist35_expX_uid9_fpDivTest_b_31_sticky_ena_q & en;

    // redist35_expX_uid9_fpDivTest_b_31_rdcnt(COUNTER,374)
    // low=0, high=4, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist35_expX_uid9_fpDivTest_b_31_rdcnt_i <= 3'd0;
            redist35_expX_uid9_fpDivTest_b_31_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist35_expX_uid9_fpDivTest_b_31_rdcnt_i == 3'd3)
            begin
                redist35_expX_uid9_fpDivTest_b_31_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist35_expX_uid9_fpDivTest_b_31_rdcnt_eq <= 1'b0;
            end
            if (redist35_expX_uid9_fpDivTest_b_31_rdcnt_eq == 1'b1)
            begin
                redist35_expX_uid9_fpDivTest_b_31_rdcnt_i <= $unsigned(redist35_expX_uid9_fpDivTest_b_31_rdcnt_i) + $unsigned(3'd4);
            end
            else
            begin
                redist35_expX_uid9_fpDivTest_b_31_rdcnt_i <= $unsigned(redist35_expX_uid9_fpDivTest_b_31_rdcnt_i) + $unsigned(3'd1);
            end
        end
    end
    assign redist35_expX_uid9_fpDivTest_b_31_rdcnt_q = redist35_expX_uid9_fpDivTest_b_31_rdcnt_i[2:0];

    // redist35_expX_uid9_fpDivTest_b_31_rdmux(MUX,375)
    assign redist35_expX_uid9_fpDivTest_b_31_rdmux_s = en;
    always @(redist35_expX_uid9_fpDivTest_b_31_rdmux_s or redist35_expX_uid9_fpDivTest_b_31_wraddr_q or redist35_expX_uid9_fpDivTest_b_31_rdcnt_q)
    begin
        unique case (redist35_expX_uid9_fpDivTest_b_31_rdmux_s)
            1'b0 : redist35_expX_uid9_fpDivTest_b_31_rdmux_q = redist35_expX_uid9_fpDivTest_b_31_wraddr_q;
            1'b1 : redist35_expX_uid9_fpDivTest_b_31_rdmux_q = redist35_expX_uid9_fpDivTest_b_31_rdcnt_q;
            default : redist35_expX_uid9_fpDivTest_b_31_rdmux_q = 3'b0;
        endcase
    end

    // redist35_expX_uid9_fpDivTest_b_31_wraddr(REG,376)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist35_expX_uid9_fpDivTest_b_31_wraddr_q <= 3'b100;
        end
        else
        begin
            redist35_expX_uid9_fpDivTest_b_31_wraddr_q <= redist35_expX_uid9_fpDivTest_b_31_rdmux_q;
        end
    end

    // redist35_expX_uid9_fpDivTest_b_31_mem(DUALMEM,373)
    assign redist35_expX_uid9_fpDivTest_b_31_mem_ia = redist34_expX_uid9_fpDivTest_b_24_q;
    assign redist35_expX_uid9_fpDivTest_b_31_mem_aa = redist35_expX_uid9_fpDivTest_b_31_wraddr_q;
    assign redist35_expX_uid9_fpDivTest_b_31_mem_ab = redist35_expX_uid9_fpDivTest_b_31_rdmux_q;
    assign redist35_expX_uid9_fpDivTest_b_31_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(8),
        .widthad_a(3),
        .numwords_a(5),
        .width_b(8),
        .widthad_b(3),
        .numwords_b(5),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .rdcontrol_reg_b("CLOCK0"),
        .byteena_reg_b("CLOCK0"),
        .outdata_reg_b("CLOCK1"),
        .outdata_sclr_b("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .clock_enable_input_b("NORMAL"),
        .clock_enable_output_b("NORMAL"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("TRUE"),
        .intended_device_family("Stratix 10")
    ) redist35_expX_uid9_fpDivTest_b_31_mem_dmem (
        .clocken1(redist35_expX_uid9_fpDivTest_b_31_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist35_expX_uid9_fpDivTest_b_31_mem_reset0),
        .clock1(clk),
        .address_a(redist35_expX_uid9_fpDivTest_b_31_mem_aa),
        .data_a(redist35_expX_uid9_fpDivTest_b_31_mem_ia),
        .wren_a(en[0]),
        .address_b(redist35_expX_uid9_fpDivTest_b_31_mem_ab),
        .q_b(redist35_expX_uid9_fpDivTest_b_31_mem_iq),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_b(),
        .clocken2(),
        .clocken3(),
        .aclr0(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .address2_a(),
        .address2_b(),
        .q_a(),
        .eccstatus()
    );
    assign redist35_expX_uid9_fpDivTest_b_31_mem_q = redist35_expX_uid9_fpDivTest_b_31_mem_iq[7:0];
    assign redist35_expX_uid9_fpDivTest_b_31_mem_enaOr_rst = redist35_expX_uid9_fpDivTest_b_31_enaAnd_q[0] | redist35_expX_uid9_fpDivTest_b_31_mem_reset0;

    // redist35_expX_uid9_fpDivTest_b_31_outputreg0(DELAY,372)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist35_expX_uid9_fpDivTest_b_31_outputreg0_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist35_expX_uid9_fpDivTest_b_31_outputreg0_q <= redist35_expX_uid9_fpDivTest_b_31_mem_q;
        end
    end

    // redist36_expX_uid9_fpDivTest_b_32(DELAY,222)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist36_expX_uid9_fpDivTest_b_32_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist36_expX_uid9_fpDivTest_b_32_q <= redist35_expX_uid9_fpDivTest_b_31_outputreg0_q;
        end
    end

    // redist32_fracX_uid10_fpDivTest_b_32_notEnable(LOGICAL,357)
    assign redist32_fracX_uid10_fpDivTest_b_32_notEnable_q = ~ (en);

    // redist32_fracX_uid10_fpDivTest_b_32_nor(LOGICAL,358)
    assign redist32_fracX_uid10_fpDivTest_b_32_nor_q = ~ (redist32_fracX_uid10_fpDivTest_b_32_notEnable_q | redist32_fracX_uid10_fpDivTest_b_32_sticky_ena_q);

    // redist32_fracX_uid10_fpDivTest_b_32_mem_last(CONSTANT,354)
    assign redist32_fracX_uid10_fpDivTest_b_32_mem_last_q = 4'b0100;

    // redist32_fracX_uid10_fpDivTest_b_32_cmp(LOGICAL,355)
    assign redist32_fracX_uid10_fpDivTest_b_32_cmp_b = {1'b0, redist32_fracX_uid10_fpDivTest_b_32_rdmux_q};
    assign redist32_fracX_uid10_fpDivTest_b_32_cmp_q = redist32_fracX_uid10_fpDivTest_b_32_mem_last_q == redist32_fracX_uid10_fpDivTest_b_32_cmp_b ? 1'b1 : 1'b0;

    // redist32_fracX_uid10_fpDivTest_b_32_cmpReg(REG,356)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist32_fracX_uid10_fpDivTest_b_32_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist32_fracX_uid10_fpDivTest_b_32_cmpReg_q <= redist32_fracX_uid10_fpDivTest_b_32_cmp_q;
        end
    end

    // redist32_fracX_uid10_fpDivTest_b_32_sticky_ena(REG,359)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist32_fracX_uid10_fpDivTest_b_32_sticky_ena_q <= 1'b0;
        end
        else if (redist32_fracX_uid10_fpDivTest_b_32_nor_q == 1'b1)
        begin
            redist32_fracX_uid10_fpDivTest_b_32_sticky_ena_q <= redist32_fracX_uid10_fpDivTest_b_32_cmpReg_q;
        end
    end

    // redist32_fracX_uid10_fpDivTest_b_32_enaAnd(LOGICAL,360)
    assign redist32_fracX_uid10_fpDivTest_b_32_enaAnd_q = redist32_fracX_uid10_fpDivTest_b_32_sticky_ena_q & en;

    // redist32_fracX_uid10_fpDivTest_b_32_rdcnt(COUNTER,351)
    // low=0, high=5, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist32_fracX_uid10_fpDivTest_b_32_rdcnt_i <= 3'd0;
            redist32_fracX_uid10_fpDivTest_b_32_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist32_fracX_uid10_fpDivTest_b_32_rdcnt_i == 3'd4)
            begin
                redist32_fracX_uid10_fpDivTest_b_32_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist32_fracX_uid10_fpDivTest_b_32_rdcnt_eq <= 1'b0;
            end
            if (redist32_fracX_uid10_fpDivTest_b_32_rdcnt_eq == 1'b1)
            begin
                redist32_fracX_uid10_fpDivTest_b_32_rdcnt_i <= $unsigned(redist32_fracX_uid10_fpDivTest_b_32_rdcnt_i) + $unsigned(3'd3);
            end
            else
            begin
                redist32_fracX_uid10_fpDivTest_b_32_rdcnt_i <= $unsigned(redist32_fracX_uid10_fpDivTest_b_32_rdcnt_i) + $unsigned(3'd1);
            end
        end
    end
    assign redist32_fracX_uid10_fpDivTest_b_32_rdcnt_q = redist32_fracX_uid10_fpDivTest_b_32_rdcnt_i[2:0];

    // redist32_fracX_uid10_fpDivTest_b_32_rdmux(MUX,352)
    assign redist32_fracX_uid10_fpDivTest_b_32_rdmux_s = en;
    always @(redist32_fracX_uid10_fpDivTest_b_32_rdmux_s or redist32_fracX_uid10_fpDivTest_b_32_wraddr_q or redist32_fracX_uid10_fpDivTest_b_32_rdcnt_q)
    begin
        unique case (redist32_fracX_uid10_fpDivTest_b_32_rdmux_s)
            1'b0 : redist32_fracX_uid10_fpDivTest_b_32_rdmux_q = redist32_fracX_uid10_fpDivTest_b_32_wraddr_q;
            1'b1 : redist32_fracX_uid10_fpDivTest_b_32_rdmux_q = redist32_fracX_uid10_fpDivTest_b_32_rdcnt_q;
            default : redist32_fracX_uid10_fpDivTest_b_32_rdmux_q = 3'b0;
        endcase
    end

    // redist32_fracX_uid10_fpDivTest_b_32_wraddr(REG,353)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist32_fracX_uid10_fpDivTest_b_32_wraddr_q <= 3'b101;
        end
        else
        begin
            redist32_fracX_uid10_fpDivTest_b_32_wraddr_q <= redist32_fracX_uid10_fpDivTest_b_32_rdmux_q;
        end
    end

    // redist32_fracX_uid10_fpDivTest_b_32_mem(DUALMEM,350)
    assign redist32_fracX_uid10_fpDivTest_b_32_mem_ia = redist31_fracX_uid10_fpDivTest_b_25_q;
    assign redist32_fracX_uid10_fpDivTest_b_32_mem_aa = redist32_fracX_uid10_fpDivTest_b_32_wraddr_q;
    assign redist32_fracX_uid10_fpDivTest_b_32_mem_ab = redist32_fracX_uid10_fpDivTest_b_32_rdmux_q;
    assign redist32_fracX_uid10_fpDivTest_b_32_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(23),
        .widthad_a(3),
        .numwords_a(6),
        .width_b(23),
        .widthad_b(3),
        .numwords_b(6),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .rdcontrol_reg_b("CLOCK0"),
        .byteena_reg_b("CLOCK0"),
        .outdata_reg_b("CLOCK1"),
        .outdata_sclr_b("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .clock_enable_input_b("NORMAL"),
        .clock_enable_output_b("NORMAL"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("TRUE"),
        .intended_device_family("Stratix 10")
    ) redist32_fracX_uid10_fpDivTest_b_32_mem_dmem (
        .clocken1(redist32_fracX_uid10_fpDivTest_b_32_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist32_fracX_uid10_fpDivTest_b_32_mem_reset0),
        .clock1(clk),
        .address_a(redist32_fracX_uid10_fpDivTest_b_32_mem_aa),
        .data_a(redist32_fracX_uid10_fpDivTest_b_32_mem_ia),
        .wren_a(en[0]),
        .address_b(redist32_fracX_uid10_fpDivTest_b_32_mem_ab),
        .q_b(redist32_fracX_uid10_fpDivTest_b_32_mem_iq),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_b(),
        .clocken2(),
        .clocken3(),
        .aclr0(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .address2_a(),
        .address2_b(),
        .q_a(),
        .eccstatus()
    );
    assign redist32_fracX_uid10_fpDivTest_b_32_mem_q = redist32_fracX_uid10_fpDivTest_b_32_mem_iq[22:0];
    assign redist32_fracX_uid10_fpDivTest_b_32_mem_enaOr_rst = redist32_fracX_uid10_fpDivTest_b_32_enaAnd_q[0] | redist32_fracX_uid10_fpDivTest_b_32_mem_reset0;

    // qDivProdLTX_opB_uid100_fpDivTest(BITJOIN,99)@32
    assign qDivProdLTX_opB_uid100_fpDivTest_q = {redist36_expX_uid9_fpDivTest_b_32_q, redist32_fracX_uid10_fpDivTest_b_32_mem_q};

    // redist25_fracY_uid13_fpDivTest_b_25(DELAY,211)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist25_fracY_uid13_fpDivTest_b_25_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist25_fracY_uid13_fpDivTest_b_25_q <= redist24_fracY_uid13_fpDivTest_b_24_q;
        end
    end

    // lOAdded_uid87_fpDivTest(BITJOIN,86)@25
    assign lOAdded_uid87_fpDivTest_q = {VCC_q, redist25_fracY_uid13_fpDivTest_b_25_q};

    // lOAdded_uid84_fpDivTest(BITJOIN,83)@25
    assign lOAdded_uid84_fpDivTest_q = {VCC_q, fracPostRndF_uid80_fpDivTest_q};

    // qDivProd_uid89_fpDivTest_cma(CHAINMULTADD,182)@25 + 5
    // out q@31
    assign qDivProd_uid89_fpDivTest_cma_reset = areset;
    assign qDivProd_uid89_fpDivTest_cma_ena0 = en[0] | qDivProd_uid89_fpDivTest_cma_reset;
    assign qDivProd_uid89_fpDivTest_cma_ena1 = qDivProd_uid89_fpDivTest_cma_ena0;
    assign qDivProd_uid89_fpDivTest_cma_ena2 = qDivProd_uid89_fpDivTest_cma_ena0;
    always @ (posedge clk)
    begin
        if (0)
        begin
        end
        else
        begin
            if (en == 1'b1)
            begin
                qDivProd_uid89_fpDivTest_cma_ah[0] <= lOAdded_uid84_fpDivTest_q;
                qDivProd_uid89_fpDivTest_cma_ch[0] <= lOAdded_uid87_fpDivTest_q;
            end
        end
    end

    assign qDivProd_uid89_fpDivTest_cma_a0 = qDivProd_uid89_fpDivTest_cma_ah[0];
    assign qDivProd_uid89_fpDivTest_cma_c0 = qDivProd_uid89_fpDivTest_cma_ch[0];
    fourteennm_mac #(
        .operation_mode("m27x27"),
        .clear_type("sclr"),
        .use_chainadder("false"),
        .ay_scan_in_clock("0"),
        .ay_scan_in_width(25),
        .ax_clock("0"),
        .ax_width(24),
        .signed_may("false"),
        .signed_max("false"),
        .input_pipeline_clock("2"),
        .second_pipeline_clock("2"),
        .output_clock("1"),
        .result_a_width(49)
    ) qDivProd_uid89_fpDivTest_cma_DSP0 (
        .clk({clk,clk,clk}),
        .ena({ qDivProd_uid89_fpDivTest_cma_ena2, qDivProd_uid89_fpDivTest_cma_ena1, qDivProd_uid89_fpDivTest_cma_ena0 }),
        .clr({ qDivProd_uid89_fpDivTest_cma_reset, qDivProd_uid89_fpDivTest_cma_reset }),
        .ay(qDivProd_uid89_fpDivTest_cma_a0),
        .ax(qDivProd_uid89_fpDivTest_cma_c0),
        .resulta(qDivProd_uid89_fpDivTest_cma_s0),
        .accumulate(),
        .loadconst(),
        .negate(),
        .sub(),
        .az(),
        .coefsela(),
        .bx(),
        .by(),
        .bz(),
        .coefselb(),
        .scanin(),
        .scanout(),
        .chainin(),
        .chainout(),
        .resultb(),
        .dfxlfsrena(),
        .dfxmisrena(),
        .dftout()
    );
    dspba_delay_ver #( .width(49), .depth(1), .reset_kind("NONE"), .phase(0), .modulus(1) )
    qDivProd_uid89_fpDivTest_cma_delay ( .xin(qDivProd_uid89_fpDivTest_cma_s0), .xout(qDivProd_uid89_fpDivTest_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign qDivProd_uid89_fpDivTest_cma_q = qDivProd_uid89_fpDivTest_cma_qq[48:0];

    // qDivProdNorm_uid90_fpDivTest(BITSELECT,89)@31
    assign qDivProdNorm_uid90_fpDivTest_b = qDivProd_uid89_fpDivTest_cma_q[48:48];

    // cstBias_uid7_fpDivTest(CONSTANT,6)
    assign cstBias_uid7_fpDivTest_q = 8'b01111111;

    // qDivProdExp_opBs_uid95_fpDivTest(SUB,94)@31
    assign qDivProdExp_opBs_uid95_fpDivTest_a = {1'b0, cstBias_uid7_fpDivTest_q};
    assign qDivProdExp_opBs_uid95_fpDivTest_b = {8'b00000000, qDivProdNorm_uid90_fpDivTest_b};
    assign qDivProdExp_opBs_uid95_fpDivTest_o = $unsigned(qDivProdExp_opBs_uid95_fpDivTest_a) - $unsigned(qDivProdExp_opBs_uid95_fpDivTest_b);
    assign qDivProdExp_opBs_uid95_fpDivTest_q = qDivProdExp_opBs_uid95_fpDivTest_o[8:0];

    // redist14_invYO_uid55_fpDivTest_b_15(DELAY,200)
    dspba_delay_ver #( .width(1), .depth(6), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    redist14_invYO_uid55_fpDivTest_b_15 ( .xin(redist13_invYO_uid55_fpDivTest_b_9_q), .xout(redist14_invYO_uid55_fpDivTest_b_15_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expPostRndF_uid82_fpDivTest(MUX,81)@31
    assign expPostRndF_uid82_fpDivTest_s = redist14_invYO_uid55_fpDivTest_b_15_q;
    always @(expPostRndF_uid82_fpDivTest_s or en or redist8_expPostRndFR_uid81_fpDivTest_b_7_outputreg0_q or redist35_expX_uid9_fpDivTest_b_31_outputreg0_q)
    begin
        unique case (expPostRndF_uid82_fpDivTest_s)
            1'b0 : expPostRndF_uid82_fpDivTest_q = redist8_expPostRndFR_uid81_fpDivTest_b_7_outputreg0_q;
            1'b1 : expPostRndF_uid82_fpDivTest_q = redist35_expX_uid9_fpDivTest_b_31_outputreg0_q;
            default : expPostRndF_uid82_fpDivTest_q = 8'b0;
        endcase
    end

    // redist28_expY_uid12_fpDivTest_b_31_notEnable(LOGICAL,323)
    assign redist28_expY_uid12_fpDivTest_b_31_notEnable_q = ~ (en);

    // redist28_expY_uid12_fpDivTest_b_31_nor(LOGICAL,324)
    assign redist28_expY_uid12_fpDivTest_b_31_nor_q = ~ (redist28_expY_uid12_fpDivTest_b_31_notEnable_q | redist28_expY_uid12_fpDivTest_b_31_sticky_ena_q);

    // redist28_expY_uid12_fpDivTest_b_31_mem_last(CONSTANT,320)
    assign redist28_expY_uid12_fpDivTest_b_31_mem_last_q = 3'b011;

    // redist28_expY_uid12_fpDivTest_b_31_cmp(LOGICAL,321)
    assign redist28_expY_uid12_fpDivTest_b_31_cmp_q = redist28_expY_uid12_fpDivTest_b_31_mem_last_q == redist28_expY_uid12_fpDivTest_b_31_rdmux_q ? 1'b1 : 1'b0;

    // redist28_expY_uid12_fpDivTest_b_31_cmpReg(REG,322)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist28_expY_uid12_fpDivTest_b_31_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist28_expY_uid12_fpDivTest_b_31_cmpReg_q <= redist28_expY_uid12_fpDivTest_b_31_cmp_q;
        end
    end

    // redist28_expY_uid12_fpDivTest_b_31_sticky_ena(REG,325)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist28_expY_uid12_fpDivTest_b_31_sticky_ena_q <= 1'b0;
        end
        else if (redist28_expY_uid12_fpDivTest_b_31_nor_q == 1'b1)
        begin
            redist28_expY_uid12_fpDivTest_b_31_sticky_ena_q <= redist28_expY_uid12_fpDivTest_b_31_cmpReg_q;
        end
    end

    // redist28_expY_uid12_fpDivTest_b_31_enaAnd(LOGICAL,326)
    assign redist28_expY_uid12_fpDivTest_b_31_enaAnd_q = redist28_expY_uid12_fpDivTest_b_31_sticky_ena_q & en;

    // redist28_expY_uid12_fpDivTest_b_31_rdcnt(COUNTER,317)
    // low=0, high=4, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist28_expY_uid12_fpDivTest_b_31_rdcnt_i <= 3'd0;
            redist28_expY_uid12_fpDivTest_b_31_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist28_expY_uid12_fpDivTest_b_31_rdcnt_i == 3'd3)
            begin
                redist28_expY_uid12_fpDivTest_b_31_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist28_expY_uid12_fpDivTest_b_31_rdcnt_eq <= 1'b0;
            end
            if (redist28_expY_uid12_fpDivTest_b_31_rdcnt_eq == 1'b1)
            begin
                redist28_expY_uid12_fpDivTest_b_31_rdcnt_i <= $unsigned(redist28_expY_uid12_fpDivTest_b_31_rdcnt_i) + $unsigned(3'd4);
            end
            else
            begin
                redist28_expY_uid12_fpDivTest_b_31_rdcnt_i <= $unsigned(redist28_expY_uid12_fpDivTest_b_31_rdcnt_i) + $unsigned(3'd1);
            end
        end
    end
    assign redist28_expY_uid12_fpDivTest_b_31_rdcnt_q = redist28_expY_uid12_fpDivTest_b_31_rdcnt_i[2:0];

    // redist28_expY_uid12_fpDivTest_b_31_rdmux(MUX,318)
    assign redist28_expY_uid12_fpDivTest_b_31_rdmux_s = en;
    always @(redist28_expY_uid12_fpDivTest_b_31_rdmux_s or redist28_expY_uid12_fpDivTest_b_31_wraddr_q or redist28_expY_uid12_fpDivTest_b_31_rdcnt_q)
    begin
        unique case (redist28_expY_uid12_fpDivTest_b_31_rdmux_s)
            1'b0 : redist28_expY_uid12_fpDivTest_b_31_rdmux_q = redist28_expY_uid12_fpDivTest_b_31_wraddr_q;
            1'b1 : redist28_expY_uid12_fpDivTest_b_31_rdmux_q = redist28_expY_uid12_fpDivTest_b_31_rdcnt_q;
            default : redist28_expY_uid12_fpDivTest_b_31_rdmux_q = 3'b0;
        endcase
    end

    // redist28_expY_uid12_fpDivTest_b_31_wraddr(REG,319)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist28_expY_uid12_fpDivTest_b_31_wraddr_q <= 3'b100;
        end
        else
        begin
            redist28_expY_uid12_fpDivTest_b_31_wraddr_q <= redist28_expY_uid12_fpDivTest_b_31_rdmux_q;
        end
    end

    // redist28_expY_uid12_fpDivTest_b_31_mem(DUALMEM,316)
    assign redist28_expY_uid12_fpDivTest_b_31_mem_ia = redist27_expY_uid12_fpDivTest_b_24_q;
    assign redist28_expY_uid12_fpDivTest_b_31_mem_aa = redist28_expY_uid12_fpDivTest_b_31_wraddr_q;
    assign redist28_expY_uid12_fpDivTest_b_31_mem_ab = redist28_expY_uid12_fpDivTest_b_31_rdmux_q;
    assign redist28_expY_uid12_fpDivTest_b_31_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(8),
        .widthad_a(3),
        .numwords_a(5),
        .width_b(8),
        .widthad_b(3),
        .numwords_b(5),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .address_reg_b("CLOCK0"),
        .indata_reg_b("CLOCK0"),
        .rdcontrol_reg_b("CLOCK0"),
        .byteena_reg_b("CLOCK0"),
        .outdata_reg_b("CLOCK1"),
        .outdata_sclr_b("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .clock_enable_input_b("NORMAL"),
        .clock_enable_output_b("NORMAL"),
        .read_during_write_mode_mixed_ports("DONT_CARE"),
        .power_up_uninitialized("TRUE"),
        .intended_device_family("Stratix 10")
    ) redist28_expY_uid12_fpDivTest_b_31_mem_dmem (
        .clocken1(redist28_expY_uid12_fpDivTest_b_31_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist28_expY_uid12_fpDivTest_b_31_mem_reset0),
        .clock1(clk),
        .address_a(redist28_expY_uid12_fpDivTest_b_31_mem_aa),
        .data_a(redist28_expY_uid12_fpDivTest_b_31_mem_ia),
        .wren_a(en[0]),
        .address_b(redist28_expY_uid12_fpDivTest_b_31_mem_ab),
        .q_b(redist28_expY_uid12_fpDivTest_b_31_mem_iq),
        .wren_b(),
        .rden_a(),
        .rden_b(),
        .data_b(),
        .clocken2(),
        .clocken3(),
        .aclr0(),
        .aclr1(),
        .addressstall_a(),
        .addressstall_b(),
        .byteena_a(),
        .byteena_b(),
        .eccencbypass(),
        .eccencparity(),
        .address2_a(),
        .address2_b(),
        .q_a(),
        .eccstatus()
    );
    assign redist28_expY_uid12_fpDivTest_b_31_mem_q = redist28_expY_uid12_fpDivTest_b_31_mem_iq[7:0];
    assign redist28_expY_uid12_fpDivTest_b_31_mem_enaOr_rst = redist28_expY_uid12_fpDivTest_b_31_enaAnd_q[0] | redist28_expY_uid12_fpDivTest_b_31_mem_reset0;

    // redist28_expY_uid12_fpDivTest_b_31_outputreg0(DELAY,315)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist28_expY_uid12_fpDivTest_b_31_outputreg0_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist28_expY_uid12_fpDivTest_b_31_outputreg0_q <= redist28_expY_uid12_fpDivTest_b_31_mem_q;
        end
    end

    // qDivProdExp_opA_uid94_fpDivTest(ADD,93)@31
    assign qDivProdExp_opA_uid94_fpDivTest_a = {1'b0, redist28_expY_uid12_fpDivTest_b_31_outputreg0_q};
    assign qDivProdExp_opA_uid94_fpDivTest_b = {1'b0, expPostRndF_uid82_fpDivTest_q};
    assign qDivProdExp_opA_uid94_fpDivTest_o = $unsigned(qDivProdExp_opA_uid94_fpDivTest_a) + $unsigned(qDivProdExp_opA_uid94_fpDivTest_b);
    assign qDivProdExp_opA_uid94_fpDivTest_q = qDivProdExp_opA_uid94_fpDivTest_o[8:0];

    // qDivProdExp_uid96_fpDivTest(SUB,95)@31
    assign qDivProdExp_uid96_fpDivTest_a = {3'b000, qDivProdExp_opA_uid94_fpDivTest_q};
    assign qDivProdExp_uid96_fpDivTest_b = {{3{qDivProdExp_opBs_uid95_fpDivTest_q[8]}}, qDivProdExp_opBs_uid95_fpDivTest_q};
    assign qDivProdExp_uid96_fpDivTest_o = $signed(qDivProdExp_uid96_fpDivTest_a) - $signed(qDivProdExp_uid96_fpDivTest_b);
    assign qDivProdExp_uid96_fpDivTest_q = qDivProdExp_uid96_fpDivTest_o[10:0];

    // qDivProdLTX_opA_uid98_fpDivTest(BITSELECT,97)@31
    assign qDivProdLTX_opA_uid98_fpDivTest_in = qDivProdExp_uid96_fpDivTest_q[7:0];
    assign qDivProdLTX_opA_uid98_fpDivTest_b = qDivProdLTX_opA_uid98_fpDivTest_in[7:0];

    // redist6_qDivProdLTX_opA_uid98_fpDivTest_b_1(DELAY,192)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist6_qDivProdLTX_opA_uid98_fpDivTest_b_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist6_qDivProdLTX_opA_uid98_fpDivTest_b_1_q <= qDivProdLTX_opA_uid98_fpDivTest_b;
        end
    end

    // qDivProdFracHigh_uid91_fpDivTest(BITSELECT,90)@31
    assign qDivProdFracHigh_uid91_fpDivTest_in = qDivProd_uid89_fpDivTest_cma_q[47:0];
    assign qDivProdFracHigh_uid91_fpDivTest_b = qDivProdFracHigh_uid91_fpDivTest_in[47:24];

    // qDivProdFracLow_uid92_fpDivTest(BITSELECT,91)@31
    assign qDivProdFracLow_uid92_fpDivTest_in = qDivProd_uid89_fpDivTest_cma_q[46:0];
    assign qDivProdFracLow_uid92_fpDivTest_b = qDivProdFracLow_uid92_fpDivTest_in[46:23];

    // qDivProdFrac_uid93_fpDivTest(MUX,92)@31
    assign qDivProdFrac_uid93_fpDivTest_s = qDivProdNorm_uid90_fpDivTest_b;
    always @(qDivProdFrac_uid93_fpDivTest_s or en or qDivProdFracLow_uid92_fpDivTest_b or qDivProdFracHigh_uid91_fpDivTest_b)
    begin
        unique case (qDivProdFrac_uid93_fpDivTest_s)
            1'b0 : qDivProdFrac_uid93_fpDivTest_q = qDivProdFracLow_uid92_fpDivTest_b;
            1'b1 : qDivProdFrac_uid93_fpDivTest_q = qDivProdFracHigh_uid91_fpDivTest_b;
            default : qDivProdFrac_uid93_fpDivTest_q = 24'b0;
        endcase
    end

    // qDivProdFracWF_uid97_fpDivTest(BITSELECT,96)@31
    assign qDivProdFracWF_uid97_fpDivTest_b = qDivProdFrac_uid93_fpDivTest_q[23:1];

    // redist7_qDivProdFracWF_uid97_fpDivTest_b_1(DELAY,193)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist7_qDivProdFracWF_uid97_fpDivTest_b_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist7_qDivProdFracWF_uid97_fpDivTest_b_1_q <= qDivProdFracWF_uid97_fpDivTest_b;
        end
    end

    // qDivProdLTX_opA_uid99_fpDivTest(BITJOIN,98)@32
    assign qDivProdLTX_opA_uid99_fpDivTest_q = {redist6_qDivProdLTX_opA_uid98_fpDivTest_b_1_q, redist7_qDivProdFracWF_uid97_fpDivTest_b_1_q};

    // qDividerProdLTX_uid101_fpDivTest(COMPARE,100)@32
    assign qDividerProdLTX_uid101_fpDivTest_a = {2'b00, qDivProdLTX_opA_uid99_fpDivTest_q};
    assign qDividerProdLTX_uid101_fpDivTest_b = {2'b00, qDivProdLTX_opB_uid100_fpDivTest_q};
    assign qDividerProdLTX_uid101_fpDivTest_o = $unsigned(qDividerProdLTX_uid101_fpDivTest_a) - $unsigned(qDividerProdLTX_uid101_fpDivTest_b);
    assign qDividerProdLTX_uid101_fpDivTest_c[0] = qDividerProdLTX_uid101_fpDivTest_o[32];

    // extraUlp_uid103_fpDivTest(LOGICAL,102)@32 + 1
    assign extraUlp_uid103_fpDivTest_qi = qDividerProdLTX_uid101_fpDivTest_c & redist5_betweenFPwF_uid102_fpDivTest_b_7_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    extraUlp_uid103_fpDivTest_delay ( .xin(extraUlp_uid103_fpDivTest_qi), .xout(extraUlp_uid103_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expRPreExc_uid112_fpDivTest(MUX,111)@33 + 1
    assign expRPreExc_uid112_fpDivTest_s = extraUlp_uid103_fpDivTest_q;
    always @ (posedge clk)
    begin
        if (areset)
        begin
            expRPreExc_uid112_fpDivTest_q <= 8'b0;
        end
        else if (en == 1'b1)
        begin
            unique case (expRPreExc_uid112_fpDivTest_s)
                1'b0 : expRPreExc_uid112_fpDivTest_q <= redist9_expPostRndFR_uid81_fpDivTest_b_9_q;
                1'b1 : expRPreExc_uid112_fpDivTest_q <= expFracPostRndR_uid111_fpDivTest_b;
                default : expRPreExc_uid112_fpDivTest_q <= 8'b0;
            endcase
        end
    end

    // invExpXIsMax_uid43_fpDivTest(LOGICAL,42)@25
    assign invExpXIsMax_uid43_fpDivTest_q = ~ (expXIsMax_uid38_fpDivTest_q);

    // InvExpXIsZero_uid44_fpDivTest(LOGICAL,43)@25
    assign InvExpXIsZero_uid44_fpDivTest_q = ~ (excZ_y_uid37_fpDivTest_q);

    // excR_y_uid45_fpDivTest(LOGICAL,44)@25
    assign excR_y_uid45_fpDivTest_q = InvExpXIsZero_uid44_fpDivTest_q & invExpXIsMax_uid43_fpDivTest_q;

    // excXIYR_uid127_fpDivTest(LOGICAL,126)@25
    assign excXIYR_uid127_fpDivTest_q = excI_x_uid27_fpDivTest_q & excR_y_uid45_fpDivTest_q;

    // excXIYZ_uid126_fpDivTest(LOGICAL,125)@25
    assign excXIYZ_uid126_fpDivTest_q = excI_x_uid27_fpDivTest_q & excZ_y_uid37_fpDivTest_q;

    // expRExt_uid114_fpDivTest(BITSELECT,113)@24
    assign expRExt_uid114_fpDivTest_b = expFracPostRnd_uid76_fpDivTest_q[35:25];

    // expOvf_uid118_fpDivTest(COMPARE,117)@24 + 1
    assign expOvf_uid118_fpDivTest_a = {{2{expRExt_uid114_fpDivTest_b[10]}}, expRExt_uid114_fpDivTest_b};
    assign expOvf_uid118_fpDivTest_b = {5'b00000, cstAllOWE_uid18_fpDivTest_q};
    always @ (posedge clk)
    begin
        if (areset)
        begin
            expOvf_uid118_fpDivTest_o <= 13'b0;
        end
        else if (en == 1'b1)
        begin
            expOvf_uid118_fpDivTest_o <= $signed(expOvf_uid118_fpDivTest_a) - $signed(expOvf_uid118_fpDivTest_b);
        end
    end
    assign expOvf_uid118_fpDivTest_n[0] = ~ (expOvf_uid118_fpDivTest_o[12]);

    // invExpXIsMax_uid29_fpDivTest(LOGICAL,28)@24
    assign invExpXIsMax_uid29_fpDivTest_q = ~ (expXIsMax_uid24_fpDivTest_q);

    // InvExpXIsZero_uid30_fpDivTest(LOGICAL,29)@24
    assign InvExpXIsZero_uid30_fpDivTest_q = ~ (excZ_x_uid23_fpDivTest_q);

    // excR_x_uid31_fpDivTest(LOGICAL,30)@24 + 1
    assign excR_x_uid31_fpDivTest_qi = InvExpXIsZero_uid30_fpDivTest_q & invExpXIsMax_uid29_fpDivTest_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    excR_x_uid31_fpDivTest_delay ( .xin(excR_x_uid31_fpDivTest_qi), .xout(excR_x_uid31_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // excXRYROvf_uid125_fpDivTest(LOGICAL,124)@25
    assign excXRYROvf_uid125_fpDivTest_q = excR_x_uid31_fpDivTest_q & excR_y_uid45_fpDivTest_q & expOvf_uid118_fpDivTest_n;

    // excXRYZ_uid124_fpDivTest(LOGICAL,123)@25
    assign excXRYZ_uid124_fpDivTest_q = excR_x_uid31_fpDivTest_q & excZ_y_uid37_fpDivTest_q;

    // excRInf_uid128_fpDivTest(LOGICAL,127)@25
    assign excRInf_uid128_fpDivTest_q = excXRYZ_uid124_fpDivTest_q | excXRYROvf_uid125_fpDivTest_q | excXIYZ_uid126_fpDivTest_q | excXIYR_uid127_fpDivTest_q;

    // xRegOrZero_uid121_fpDivTest(LOGICAL,120)@25
    assign xRegOrZero_uid121_fpDivTest_q = excR_x_uid31_fpDivTest_q | redist22_excZ_x_uid23_fpDivTest_q_1_q;

    // regOrZeroOverInf_uid122_fpDivTest(LOGICAL,121)@25
    assign regOrZeroOverInf_uid122_fpDivTest_q = xRegOrZero_uid121_fpDivTest_q & excI_y_uid41_fpDivTest_q;

    // expUdf_uid115_fpDivTest(COMPARE,114)@24 + 1
    assign expUdf_uid115_fpDivTest_a = {12'b000000000000, GND_q};
    assign expUdf_uid115_fpDivTest_b = {{2{expRExt_uid114_fpDivTest_b[10]}}, expRExt_uid114_fpDivTest_b};
    always @ (posedge clk)
    begin
        if (areset)
        begin
            expUdf_uid115_fpDivTest_o <= 13'b0;
        end
        else if (en == 1'b1)
        begin
            expUdf_uid115_fpDivTest_o <= $signed(expUdf_uid115_fpDivTest_a) - $signed(expUdf_uid115_fpDivTest_b);
        end
    end
    assign expUdf_uid115_fpDivTest_n[0] = ~ (expUdf_uid115_fpDivTest_o[12]);

    // regOverRegWithUf_uid120_fpDivTest(LOGICAL,119)@25
    assign regOverRegWithUf_uid120_fpDivTest_q = expUdf_uid115_fpDivTest_n & excR_x_uid31_fpDivTest_q & excR_y_uid45_fpDivTest_q;

    // zeroOverReg_uid119_fpDivTest(LOGICAL,118)@25
    assign zeroOverReg_uid119_fpDivTest_q = redist22_excZ_x_uid23_fpDivTest_q_1_q & excR_y_uid45_fpDivTest_q;

    // excRZero_uid123_fpDivTest(LOGICAL,122)@25
    assign excRZero_uid123_fpDivTest_q = zeroOverReg_uid119_fpDivTest_q | regOverRegWithUf_uid120_fpDivTest_q | regOrZeroOverInf_uid122_fpDivTest_q;

    // concExc_uid132_fpDivTest(BITJOIN,131)@25
    assign concExc_uid132_fpDivTest_q = {excRNaN_uid131_fpDivTest_q, excRInf_uid128_fpDivTest_q, excRZero_uid123_fpDivTest_q};

    // excREnc_uid133_fpDivTest(LOOKUP,132)@25 + 1
    always @ (posedge clk)
    begin
        if (areset)
        begin
            excREnc_uid133_fpDivTest_q <= 2'b01;
        end
        else if (en == 1'b1)
        begin
            unique case (concExc_uid132_fpDivTest_q)
                3'b000 : excREnc_uid133_fpDivTest_q <= 2'b01;
                3'b001 : excREnc_uid133_fpDivTest_q <= 2'b00;
                3'b010 : excREnc_uid133_fpDivTest_q <= 2'b10;
                3'b011 : excREnc_uid133_fpDivTest_q <= 2'b00;
                3'b100 : excREnc_uid133_fpDivTest_q <= 2'b11;
                3'b101 : excREnc_uid133_fpDivTest_q <= 2'b00;
                3'b110 : excREnc_uid133_fpDivTest_q <= 2'b00;
                3'b111 : excREnc_uid133_fpDivTest_q <= 2'b00;
                default : begin
                              // unreachable
                              excREnc_uid133_fpDivTest_q <= 2'bxx;
                          end
            endcase
        end
    end

    // redist3_excREnc_uid133_fpDivTest_q_9(DELAY,189)
    dspba_delay_ver #( .width(2), .depth(8), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    redist3_excREnc_uid133_fpDivTest_q_9 ( .xin(excREnc_uid133_fpDivTest_q), .xout(redist3_excREnc_uid133_fpDivTest_q_9_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expRPostExc_uid141_fpDivTest(MUX,140)@34
    assign expRPostExc_uid141_fpDivTest_s = redist3_excREnc_uid133_fpDivTest_q_9_q;
    always @(expRPostExc_uid141_fpDivTest_s or en or cstAllZWE_uid20_fpDivTest_q or expRPreExc_uid112_fpDivTest_q or cstAllOWE_uid18_fpDivTest_q)
    begin
        unique case (expRPostExc_uid141_fpDivTest_s)
            2'b00 : expRPostExc_uid141_fpDivTest_q = cstAllZWE_uid20_fpDivTest_q;
            2'b01 : expRPostExc_uid141_fpDivTest_q = expRPreExc_uid112_fpDivTest_q;
            2'b10 : expRPostExc_uid141_fpDivTest_q = cstAllOWE_uid18_fpDivTest_q;
            2'b11 : expRPostExc_uid141_fpDivTest_q = cstAllOWE_uid18_fpDivTest_q;
            default : expRPostExc_uid141_fpDivTest_q = 8'b0;
        endcase
    end

    // oneFracRPostExc2_uid134_fpDivTest(CONSTANT,133)
    assign oneFracRPostExc2_uid134_fpDivTest_q = 23'b00000000000000000000001;

    // fracPostRndFPostUlp_uid106_fpDivTest(BITSELECT,105)@33
    assign fracPostRndFPostUlp_uid106_fpDivTest_in = fracRPreExcExt_uid105_fpDivTest_q[22:0];
    assign fracPostRndFPostUlp_uid106_fpDivTest_b = fracPostRndFPostUlp_uid106_fpDivTest_in[22:0];

    // fracRPreExc_uid107_fpDivTest(MUX,106)@33 + 1
    assign fracRPreExc_uid107_fpDivTest_s = extraUlp_uid103_fpDivTest_q;
    always @ (posedge clk)
    begin
        if (areset)
        begin
            fracRPreExc_uid107_fpDivTest_q <= 23'b0;
        end
        else if (en == 1'b1)
        begin
            unique case (fracRPreExc_uid107_fpDivTest_s)
                1'b0 : fracRPreExc_uid107_fpDivTest_q <= redist4_fracPostRndFT_uid104_fpDivTest_b_8_mem_q;
                1'b1 : fracRPreExc_uid107_fpDivTest_q <= fracPostRndFPostUlp_uid106_fpDivTest_b;
                default : fracRPreExc_uid107_fpDivTest_q <= 23'b0;
            endcase
        end
    end

    // fracRPostExc_uid137_fpDivTest(MUX,136)@34
    assign fracRPostExc_uid137_fpDivTest_s = redist3_excREnc_uid133_fpDivTest_q_9_q;
    always @(fracRPostExc_uid137_fpDivTest_s or en or paddingY_uid15_fpDivTest_q or fracRPreExc_uid107_fpDivTest_q or oneFracRPostExc2_uid134_fpDivTest_q)
    begin
        unique case (fracRPostExc_uid137_fpDivTest_s)
            2'b00 : fracRPostExc_uid137_fpDivTest_q = paddingY_uid15_fpDivTest_q;
            2'b01 : fracRPostExc_uid137_fpDivTest_q = fracRPreExc_uid107_fpDivTest_q;
            2'b10 : fracRPostExc_uid137_fpDivTest_q = paddingY_uid15_fpDivTest_q;
            2'b11 : fracRPostExc_uid137_fpDivTest_q = oneFracRPostExc2_uid134_fpDivTest_q;
            default : fracRPostExc_uid137_fpDivTest_q = 23'b0;
        endcase
    end

    // divR_uid144_fpDivTest(BITJOIN,143)@34
    assign divR_uid144_fpDivTest_q = {redist2_sRPostExc_uid143_fpDivTest_q_9_q, expRPostExc_uid141_fpDivTest_q, fracRPostExc_uid137_fpDivTest_q};

    // xOut(GPOUT,4)@34
    assign q = divR_uid144_fpDivTest_q;

endmodule
