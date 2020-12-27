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

// SystemVerilog created from acl_fdiv
// SystemVerilog created on Sun Dec 27 09:47:21 2020


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
    wire [0:0] fracXIsZero_uid25_fpDivTest_q;
    wire [0:0] fracXIsNotZero_uid26_fpDivTest_q;
    wire [0:0] excI_x_uid27_fpDivTest_q;
    wire [0:0] excN_x_uid28_fpDivTest_q;
    wire [0:0] invExpXIsMax_uid29_fpDivTest_q;
    wire [0:0] InvExpXIsZero_uid30_fpDivTest_q;
    wire [0:0] excR_x_uid31_fpDivTest_q;
    wire [0:0] excZ_y_uid37_fpDivTest_q;
    wire [0:0] expXIsMax_uid38_fpDivTest_q;
    wire [0:0] fracXIsZero_uid39_fpDivTest_q;
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
    wire [0:0] extraUlp_uid103_fpDivTest_q;
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
    wire [0:0] zeroOverReg_uid119_fpDivTest_qi;
    reg [0:0] zeroOverReg_uid119_fpDivTest_q;
    wire [0:0] regOverRegWithUf_uid120_fpDivTest_qi;
    reg [0:0] regOverRegWithUf_uid120_fpDivTest_q;
    wire [0:0] xRegOrZero_uid121_fpDivTest_q;
    wire [0:0] regOrZeroOverInf_uid122_fpDivTest_qi;
    reg [0:0] regOrZeroOverInf_uid122_fpDivTest_q;
    wire [0:0] excRZero_uid123_fpDivTest_q;
    wire [0:0] excXRYZ_uid124_fpDivTest_q;
    wire [0:0] excXRYROvf_uid125_fpDivTest_q;
    wire [0:0] excXIYZ_uid126_fpDivTest_q;
    wire [0:0] excXIYR_uid127_fpDivTest_q;
    wire [0:0] excRInf_uid128_fpDivTest_qi;
    reg [0:0] excRInf_uid128_fpDivTest_q;
    wire [0:0] excXZYZ_uid129_fpDivTest_q;
    wire [0:0] excXIYI_uid130_fpDivTest_q;
    wire [0:0] excRNaN_uid131_fpDivTest_qi;
    reg [0:0] excRNaN_uid131_fpDivTest_q;
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
    wire memoryC1_uid149_invTables_lutmem_reset0;
    wire [21:0] memoryC1_uid149_invTables_lutmem_ia;
    wire [8:0] memoryC1_uid149_invTables_lutmem_aa;
    wire [8:0] memoryC1_uid149_invTables_lutmem_ab;
    wire [21:0] memoryC1_uid149_invTables_lutmem_ir;
    wire [21:0] memoryC1_uid149_invTables_lutmem_r;
    wire memoryC2_uid152_invTables_lutmem_reset0;
    wire [12:0] memoryC2_uid152_invTables_lutmem_ia;
    wire [8:0] memoryC2_uid152_invTables_lutmem_aa;
    wire [8:0] memoryC2_uid152_invTables_lutmem_ab;
    wire [12:0] memoryC2_uid152_invTables_lutmem_ir;
    wire [12:0] memoryC2_uid152_invTables_lutmem_r;
    wire qDivProd_uid89_fpDivTest_cma_reset;
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [24:0] qDivProd_uid89_fpDivTest_cma_a0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [24:0] qDivProd_uid89_fpDivTest_cma_a1 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [23:0] qDivProd_uid89_fpDivTest_cma_c0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [23:0] qDivProd_uid89_fpDivTest_cma_c1 [0:0];
    wire [48:0] qDivProd_uid89_fpDivTest_cma_p [0:0];
    wire [48:0] qDivProd_uid89_fpDivTest_cma_u [0:0];
    wire [48:0] qDivProd_uid89_fpDivTest_cma_w [0:0];
    wire [48:0] qDivProd_uid89_fpDivTest_cma_x [0:0];
    wire [48:0] qDivProd_uid89_fpDivTest_cma_y [0:0];
    reg [48:0] qDivProd_uid89_fpDivTest_cma_s [0:0];
    wire [48:0] qDivProd_uid89_fpDivTest_cma_qq;
    wire [48:0] qDivProd_uid89_fpDivTest_cma_q;
    wire qDivProd_uid89_fpDivTest_cma_ena0;
    wire qDivProd_uid89_fpDivTest_cma_ena1;
    wire qDivProd_uid89_fpDivTest_cma_ena2;
    wire prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_reset;
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [26:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_a0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [26:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_a1 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [23:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_c0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [23:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_c1 [0:0];
    wire [50:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_p [0:0];
    wire [50:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_u [0:0];
    wire [50:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_w [0:0];
    wire [50:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_x [0:0];
    wire [50:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_y [0:0];
    reg [50:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_s [0:0];
    wire [50:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_qq;
    wire [50:0] prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_q;
    wire prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena0;
    wire prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena1;
    wire prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena2;
    wire prodXY_uid174_pT1_uid159_invPolyEval_cma_reset;
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [12:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_a0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [12:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_a1 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg signed [12:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_c0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg signed [12:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_c1 [0:0];
    wire signed [13:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_l [0:0];
    wire signed [26:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_p [0:0];
    wire signed [26:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_u [0:0];
    wire signed [26:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_w [0:0];
    wire signed [26:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_x [0:0];
    wire signed [26:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_y [0:0];
    reg signed [26:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_s [0:0];
    wire [25:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_qq;
    wire [25:0] prodXY_uid174_pT1_uid159_invPolyEval_cma_q;
    wire prodXY_uid174_pT1_uid159_invPolyEval_cma_ena0;
    wire prodXY_uid174_pT1_uid159_invPolyEval_cma_ena1;
    wire prodXY_uid174_pT1_uid159_invPolyEval_cma_ena2;
    wire prodXY_uid177_pT2_uid165_invPolyEval_cma_reset;
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [13:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_a0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [13:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_a1 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg signed [23:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_c0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg signed [23:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_c1 [0:0];
    wire signed [14:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_l [0:0];
    wire signed [38:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_p [0:0];
    wire signed [38:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_u [0:0];
    wire signed [38:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_w [0:0];
    wire signed [38:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_x [0:0];
    wire signed [38:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_y [0:0];
    reg signed [38:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_s [0:0];
    wire [37:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_qq;
    wire [37:0] prodXY_uid177_pT2_uid165_invPolyEval_cma_q;
    wire prodXY_uid177_pT2_uid165_invPolyEval_cma_ena0;
    wire prodXY_uid177_pT2_uid165_invPolyEval_cma_ena1;
    wire prodXY_uid177_pT2_uid165_invPolyEval_cma_ena2;
    reg [0:0] redist0_lowRangeB_uid160_invPolyEval_b_1_q;
    reg [0:0] redist1_sRPostExc_uid143_fpDivTest_q_5_q;
    reg [1:0] redist2_excREnc_uid133_fpDivTest_q_5_q;
    reg [0:0] redist3_ovfIncRnd_uid109_fpDivTest_b_1_q;
    reg [0:0] redist4_extraUlp_uid103_fpDivTest_q_1_q;
    reg [22:0] redist5_qDivProdFracWF_uid97_fpDivTest_b_1_q;
    reg [8:0] redist6_qDivProdExp_opA_uid94_fpDivTest_q_4_q;
    reg [23:0] redist9_lOAdded_uid57_fpDivTest_q_3_q;
    reg [0:0] redist10_invYO_uid55_fpDivTest_b_5_q;
    reg [26:0] redist11_invY_uid54_fpDivTest_b_1_q;
    reg [13:0] redist12_yPE_uid52_fpDivTest_b_2_q;
    reg [8:0] redist14_yAddr_uid51_fpDivTest_b_3_q;
    reg [8:0] redist15_yAddr_uid51_fpDivTest_b_7_q;
    reg [0:0] redist16_signR_uid46_fpDivTest_q_15_q;
    reg [22:0] redist18_fracY_uid13_fpDivTest_b_14_q;
    reg [7:0] redist20_expY_uid12_fpDivTest_b_14_q;
    reg [22:0] redist22_fracX_uid10_fpDivTest_b_14_q;
    reg [22:0] redist23_fracX_uid10_fpDivTest_b_18_q;
    reg [7:0] redist25_expX_uid9_fpDivTest_b_14_q;
    reg [7:0] redist26_expX_uid9_fpDivTest_b_18_q;
    reg [7:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_outputreg_q;
    wire redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_reset0;
    wire [7:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_ia;
    wire [1:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_aa;
    wire [1:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_ab;
    wire [7:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_iq;
    wire [7:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_q;
    wire [1:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_rdcnt_q;
    (* preserve *) reg [1:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_rdcnt_i;
    wire [0:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_rdmux_s;
    reg [1:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_rdmux_q;
    reg [1:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_wraddr_q;
    wire [2:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_last_q;
    wire [2:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_cmp_b;
    wire [0:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_cmp_q;
    reg [0:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_cmpReg_q;
    wire [0:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_notEnable_q;
    wire [0:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_nor_q;
    (* preserve_syn_only *) reg [0:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_sticky_ena_q;
    wire [0:0] redist7_expPostRndFR_uid81_fpDivTest_b_6_enaAnd_q;
    reg [23:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_outputreg_q;
    wire redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_reset0;
    wire [23:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_ia;
    wire [1:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_aa;
    wire [1:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_ab;
    wire [23:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_iq;
    wire [23:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_q;
    wire [1:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt_q;
    (* preserve *) reg [1:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt_i;
    (* preserve *) reg redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt_eq;
    wire [0:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_rdmux_s;
    reg [1:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_rdmux_q;
    reg [1:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_wraddr_q;
    wire [1:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_last_q;
    wire [0:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_cmp_q;
    reg [0:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_cmpReg_q;
    wire [0:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_notEnable_q;
    wire [0:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_nor_q;
    (* preserve_syn_only *) reg [0:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_sticky_ena_q;
    wire [0:0] redist8_fracPostRndF_uid80_fpDivTest_q_5_enaAnd_q;
    wire redist13_yPE_uid52_fpDivTest_b_6_mem_reset0;
    wire [13:0] redist13_yPE_uid52_fpDivTest_b_6_mem_ia;
    wire [1:0] redist13_yPE_uid52_fpDivTest_b_6_mem_aa;
    wire [1:0] redist13_yPE_uid52_fpDivTest_b_6_mem_ab;
    wire [13:0] redist13_yPE_uid52_fpDivTest_b_6_mem_iq;
    wire [13:0] redist13_yPE_uid52_fpDivTest_b_6_mem_q;
    wire [1:0] redist13_yPE_uid52_fpDivTest_b_6_rdcnt_q;
    (* preserve *) reg [1:0] redist13_yPE_uid52_fpDivTest_b_6_rdcnt_i;
    (* preserve *) reg redist13_yPE_uid52_fpDivTest_b_6_rdcnt_eq;
    wire [0:0] redist13_yPE_uid52_fpDivTest_b_6_rdmux_s;
    reg [1:0] redist13_yPE_uid52_fpDivTest_b_6_rdmux_q;
    reg [1:0] redist13_yPE_uid52_fpDivTest_b_6_wraddr_q;
    wire [1:0] redist13_yPE_uid52_fpDivTest_b_6_mem_last_q;
    wire [0:0] redist13_yPE_uid52_fpDivTest_b_6_cmp_q;
    reg [0:0] redist13_yPE_uid52_fpDivTest_b_6_cmpReg_q;
    wire [0:0] redist13_yPE_uid52_fpDivTest_b_6_notEnable_q;
    wire [0:0] redist13_yPE_uid52_fpDivTest_b_6_nor_q;
    (* preserve_syn_only *) reg [0:0] redist13_yPE_uid52_fpDivTest_b_6_sticky_ena_q;
    wire [0:0] redist13_yPE_uid52_fpDivTest_b_6_enaAnd_q;
    reg [22:0] redist17_fracY_uid13_fpDivTest_b_12_outputreg_q;
    wire redist17_fracY_uid13_fpDivTest_b_12_mem_reset0;
    wire [22:0] redist17_fracY_uid13_fpDivTest_b_12_mem_ia;
    wire [3:0] redist17_fracY_uid13_fpDivTest_b_12_mem_aa;
    wire [3:0] redist17_fracY_uid13_fpDivTest_b_12_mem_ab;
    wire [22:0] redist17_fracY_uid13_fpDivTest_b_12_mem_iq;
    wire [22:0] redist17_fracY_uid13_fpDivTest_b_12_mem_q;
    wire [3:0] redist17_fracY_uid13_fpDivTest_b_12_rdcnt_q;
    (* preserve *) reg [3:0] redist17_fracY_uid13_fpDivTest_b_12_rdcnt_i;
    (* preserve *) reg redist17_fracY_uid13_fpDivTest_b_12_rdcnt_eq;
    wire [0:0] redist17_fracY_uid13_fpDivTest_b_12_rdmux_s;
    reg [3:0] redist17_fracY_uid13_fpDivTest_b_12_rdmux_q;
    reg [3:0] redist17_fracY_uid13_fpDivTest_b_12_wraddr_q;
    wire [4:0] redist17_fracY_uid13_fpDivTest_b_12_mem_last_q;
    wire [4:0] redist17_fracY_uid13_fpDivTest_b_12_cmp_b;
    wire [0:0] redist17_fracY_uid13_fpDivTest_b_12_cmp_q;
    reg [0:0] redist17_fracY_uid13_fpDivTest_b_12_cmpReg_q;
    wire [0:0] redist17_fracY_uid13_fpDivTest_b_12_notEnable_q;
    wire [0:0] redist17_fracY_uid13_fpDivTest_b_12_nor_q;
    (* preserve_syn_only *) reg [0:0] redist17_fracY_uid13_fpDivTest_b_12_sticky_ena_q;
    wire [0:0] redist17_fracY_uid13_fpDivTest_b_12_enaAnd_q;
    reg [7:0] redist19_expY_uid12_fpDivTest_b_12_outputreg_q;
    wire redist19_expY_uid12_fpDivTest_b_12_mem_reset0;
    wire [7:0] redist19_expY_uid12_fpDivTest_b_12_mem_ia;
    wire [3:0] redist19_expY_uid12_fpDivTest_b_12_mem_aa;
    wire [3:0] redist19_expY_uid12_fpDivTest_b_12_mem_ab;
    wire [7:0] redist19_expY_uid12_fpDivTest_b_12_mem_iq;
    wire [7:0] redist19_expY_uid12_fpDivTest_b_12_mem_q;
    wire [3:0] redist19_expY_uid12_fpDivTest_b_12_rdcnt_q;
    (* preserve *) reg [3:0] redist19_expY_uid12_fpDivTest_b_12_rdcnt_i;
    (* preserve *) reg redist19_expY_uid12_fpDivTest_b_12_rdcnt_eq;
    wire [0:0] redist19_expY_uid12_fpDivTest_b_12_rdmux_s;
    reg [3:0] redist19_expY_uid12_fpDivTest_b_12_rdmux_q;
    reg [3:0] redist19_expY_uid12_fpDivTest_b_12_wraddr_q;
    wire [4:0] redist19_expY_uid12_fpDivTest_b_12_mem_last_q;
    wire [4:0] redist19_expY_uid12_fpDivTest_b_12_cmp_b;
    wire [0:0] redist19_expY_uid12_fpDivTest_b_12_cmp_q;
    reg [0:0] redist19_expY_uid12_fpDivTest_b_12_cmpReg_q;
    wire [0:0] redist19_expY_uid12_fpDivTest_b_12_notEnable_q;
    wire [0:0] redist19_expY_uid12_fpDivTest_b_12_nor_q;
    (* preserve_syn_only *) reg [0:0] redist19_expY_uid12_fpDivTest_b_12_sticky_ena_q;
    wire [0:0] redist19_expY_uid12_fpDivTest_b_12_enaAnd_q;
    reg [22:0] redist21_fracX_uid10_fpDivTest_b_10_outputreg_q;
    wire redist21_fracX_uid10_fpDivTest_b_10_mem_reset0;
    wire [22:0] redist21_fracX_uid10_fpDivTest_b_10_mem_ia;
    wire [2:0] redist21_fracX_uid10_fpDivTest_b_10_mem_aa;
    wire [2:0] redist21_fracX_uid10_fpDivTest_b_10_mem_ab;
    wire [22:0] redist21_fracX_uid10_fpDivTest_b_10_mem_iq;
    wire [22:0] redist21_fracX_uid10_fpDivTest_b_10_mem_q;
    wire [2:0] redist21_fracX_uid10_fpDivTest_b_10_rdcnt_q;
    (* preserve *) reg [2:0] redist21_fracX_uid10_fpDivTest_b_10_rdcnt_i;
    wire [0:0] redist21_fracX_uid10_fpDivTest_b_10_rdmux_s;
    reg [2:0] redist21_fracX_uid10_fpDivTest_b_10_rdmux_q;
    reg [2:0] redist21_fracX_uid10_fpDivTest_b_10_wraddr_q;
    wire [3:0] redist21_fracX_uid10_fpDivTest_b_10_mem_last_q;
    wire [3:0] redist21_fracX_uid10_fpDivTest_b_10_cmp_b;
    wire [0:0] redist21_fracX_uid10_fpDivTest_b_10_cmp_q;
    reg [0:0] redist21_fracX_uid10_fpDivTest_b_10_cmpReg_q;
    wire [0:0] redist21_fracX_uid10_fpDivTest_b_10_notEnable_q;
    wire [0:0] redist21_fracX_uid10_fpDivTest_b_10_nor_q;
    (* preserve_syn_only *) reg [0:0] redist21_fracX_uid10_fpDivTest_b_10_sticky_ena_q;
    wire [0:0] redist21_fracX_uid10_fpDivTest_b_10_enaAnd_q;
    reg [22:0] redist22_fracX_uid10_fpDivTest_b_14_inputreg_q;
    reg [22:0] redist23_fracX_uid10_fpDivTest_b_18_inputreg_q;
    reg [7:0] redist24_expX_uid9_fpDivTest_b_12_outputreg_q;
    wire redist24_expX_uid9_fpDivTest_b_12_mem_reset0;
    wire [7:0] redist24_expX_uid9_fpDivTest_b_12_mem_ia;
    wire [3:0] redist24_expX_uid9_fpDivTest_b_12_mem_aa;
    wire [3:0] redist24_expX_uid9_fpDivTest_b_12_mem_ab;
    wire [7:0] redist24_expX_uid9_fpDivTest_b_12_mem_iq;
    wire [7:0] redist24_expX_uid9_fpDivTest_b_12_mem_q;
    wire [3:0] redist24_expX_uid9_fpDivTest_b_12_rdcnt_q;
    (* preserve *) reg [3:0] redist24_expX_uid9_fpDivTest_b_12_rdcnt_i;
    (* preserve *) reg redist24_expX_uid9_fpDivTest_b_12_rdcnt_eq;
    wire [0:0] redist24_expX_uid9_fpDivTest_b_12_rdmux_s;
    reg [3:0] redist24_expX_uid9_fpDivTest_b_12_rdmux_q;
    reg [3:0] redist24_expX_uid9_fpDivTest_b_12_wraddr_q;
    wire [4:0] redist24_expX_uid9_fpDivTest_b_12_mem_last_q;
    wire [4:0] redist24_expX_uid9_fpDivTest_b_12_cmp_b;
    wire [0:0] redist24_expX_uid9_fpDivTest_b_12_cmp_q;
    reg [0:0] redist24_expX_uid9_fpDivTest_b_12_cmpReg_q;
    wire [0:0] redist24_expX_uid9_fpDivTest_b_12_notEnable_q;
    wire [0:0] redist24_expX_uid9_fpDivTest_b_12_nor_q;
    (* preserve_syn_only *) reg [0:0] redist24_expX_uid9_fpDivTest_b_12_sticky_ena_q;
    wire [0:0] redist24_expX_uid9_fpDivTest_b_12_enaAnd_q;


    // redist17_fracY_uid13_fpDivTest_b_12_notEnable(LOGICAL,256)
    assign redist17_fracY_uid13_fpDivTest_b_12_notEnable_q = ~ (en);

    // redist17_fracY_uid13_fpDivTest_b_12_nor(LOGICAL,257)
    assign redist17_fracY_uid13_fpDivTest_b_12_nor_q = ~ (redist17_fracY_uid13_fpDivTest_b_12_notEnable_q | redist17_fracY_uid13_fpDivTest_b_12_sticky_ena_q);

    // redist17_fracY_uid13_fpDivTest_b_12_mem_last(CONSTANT,253)
    assign redist17_fracY_uid13_fpDivTest_b_12_mem_last_q = 5'b01000;

    // redist17_fracY_uid13_fpDivTest_b_12_cmp(LOGICAL,254)
    assign redist17_fracY_uid13_fpDivTest_b_12_cmp_b = {1'b0, redist17_fracY_uid13_fpDivTest_b_12_rdmux_q};
    assign redist17_fracY_uid13_fpDivTest_b_12_cmp_q = redist17_fracY_uid13_fpDivTest_b_12_mem_last_q == redist17_fracY_uid13_fpDivTest_b_12_cmp_b ? 1'b1 : 1'b0;

    // redist17_fracY_uid13_fpDivTest_b_12_cmpReg(REG,255)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist17_fracY_uid13_fpDivTest_b_12_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist17_fracY_uid13_fpDivTest_b_12_cmpReg_q <= redist17_fracY_uid13_fpDivTest_b_12_cmp_q;
        end
    end

    // redist17_fracY_uid13_fpDivTest_b_12_sticky_ena(REG,258)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist17_fracY_uid13_fpDivTest_b_12_sticky_ena_q <= 1'b0;
        end
        else if (redist17_fracY_uid13_fpDivTest_b_12_nor_q == 1'b1)
        begin
            redist17_fracY_uid13_fpDivTest_b_12_sticky_ena_q <= redist17_fracY_uid13_fpDivTest_b_12_cmpReg_q;
        end
    end

    // redist17_fracY_uid13_fpDivTest_b_12_enaAnd(LOGICAL,259)
    assign redist17_fracY_uid13_fpDivTest_b_12_enaAnd_q = redist17_fracY_uid13_fpDivTest_b_12_sticky_ena_q & en;

    // redist17_fracY_uid13_fpDivTest_b_12_rdcnt(COUNTER,250)
    // low=0, high=9, step=1, init=0
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist17_fracY_uid13_fpDivTest_b_12_rdcnt_i <= 4'd0;
            redist17_fracY_uid13_fpDivTest_b_12_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist17_fracY_uid13_fpDivTest_b_12_rdcnt_i == 4'd8)
            begin
                redist17_fracY_uid13_fpDivTest_b_12_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist17_fracY_uid13_fpDivTest_b_12_rdcnt_eq <= 1'b0;
            end
            if (redist17_fracY_uid13_fpDivTest_b_12_rdcnt_eq == 1'b1)
            begin
                redist17_fracY_uid13_fpDivTest_b_12_rdcnt_i <= $unsigned(redist17_fracY_uid13_fpDivTest_b_12_rdcnt_i) + $unsigned(4'd7);
            end
            else
            begin
                redist17_fracY_uid13_fpDivTest_b_12_rdcnt_i <= $unsigned(redist17_fracY_uid13_fpDivTest_b_12_rdcnt_i) + $unsigned(4'd1);
            end
        end
    end
    assign redist17_fracY_uid13_fpDivTest_b_12_rdcnt_q = redist17_fracY_uid13_fpDivTest_b_12_rdcnt_i[3:0];

    // redist17_fracY_uid13_fpDivTest_b_12_rdmux(MUX,251)
    assign redist17_fracY_uid13_fpDivTest_b_12_rdmux_s = en;
    always @(redist17_fracY_uid13_fpDivTest_b_12_rdmux_s or redist17_fracY_uid13_fpDivTest_b_12_wraddr_q or redist17_fracY_uid13_fpDivTest_b_12_rdcnt_q)
    begin
        unique case (redist17_fracY_uid13_fpDivTest_b_12_rdmux_s)
            1'b0 : redist17_fracY_uid13_fpDivTest_b_12_rdmux_q = redist17_fracY_uid13_fpDivTest_b_12_wraddr_q;
            1'b1 : redist17_fracY_uid13_fpDivTest_b_12_rdmux_q = redist17_fracY_uid13_fpDivTest_b_12_rdcnt_q;
            default : redist17_fracY_uid13_fpDivTest_b_12_rdmux_q = 4'b0;
        endcase
    end

    // VCC(CONSTANT,1)
    assign VCC_q = 1'b1;

    // fracY_uid13_fpDivTest(BITSELECT,12)@0
    assign fracY_uid13_fpDivTest_b = b[22:0];

    // redist17_fracY_uid13_fpDivTest_b_12_wraddr(REG,252)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist17_fracY_uid13_fpDivTest_b_12_wraddr_q <= 4'b1001;
        end
        else
        begin
            redist17_fracY_uid13_fpDivTest_b_12_wraddr_q <= redist17_fracY_uid13_fpDivTest_b_12_rdmux_q;
        end
    end

    // redist17_fracY_uid13_fpDivTest_b_12_mem(DUALMEM,249)
    assign redist17_fracY_uid13_fpDivTest_b_12_mem_ia = fracY_uid13_fpDivTest_b;
    assign redist17_fracY_uid13_fpDivTest_b_12_mem_aa = redist17_fracY_uid13_fpDivTest_b_12_wraddr_q;
    assign redist17_fracY_uid13_fpDivTest_b_12_mem_ab = redist17_fracY_uid13_fpDivTest_b_12_rdmux_q;
    assign redist17_fracY_uid13_fpDivTest_b_12_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(23),
        .widthad_a(4),
        .numwords_a(10),
        .width_b(23),
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
    ) redist17_fracY_uid13_fpDivTest_b_12_mem_dmem (
        .clocken1(redist17_fracY_uid13_fpDivTest_b_12_enaAnd_q[0]),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .aclr1(redist17_fracY_uid13_fpDivTest_b_12_mem_reset0),
        .clock1(clk),
        .address_a(redist17_fracY_uid13_fpDivTest_b_12_mem_aa),
        .data_a(redist17_fracY_uid13_fpDivTest_b_12_mem_ia),
        .wren_a(en[0]),
        .address_b(redist17_fracY_uid13_fpDivTest_b_12_mem_ab),
        .q_b(redist17_fracY_uid13_fpDivTest_b_12_mem_iq),
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
    assign redist17_fracY_uid13_fpDivTest_b_12_mem_q = redist17_fracY_uid13_fpDivTest_b_12_mem_iq[22:0];

    // redist17_fracY_uid13_fpDivTest_b_12_outputreg(DELAY,248)
    dspba_delay_ver #( .width(23), .depth(1), .reset_kind("ASYNC") )
    redist17_fracY_uid13_fpDivTest_b_12_outputreg ( .xin(redist17_fracY_uid13_fpDivTest_b_12_mem_q), .xout(redist17_fracY_uid13_fpDivTest_b_12_outputreg_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist18_fracY_uid13_fpDivTest_b_14(DELAY,204)
    dspba_delay_ver #( .width(23), .depth(2), .reset_kind("ASYNC") )
    redist18_fracY_uid13_fpDivTest_b_14 ( .xin(redist17_fracY_uid13_fpDivTest_b_12_outputreg_q), .xout(redist18_fracY_uid13_fpDivTest_b_14_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // paddingY_uid15_fpDivTest(CONSTANT,14)
    assign paddingY_uid15_fpDivTest_q = 23'b00000000000000000000000;

    // fracXIsZero_uid39_fpDivTest(LOGICAL,38)@14
    assign fracXIsZero_uid39_fpDivTest_q = paddingY_uid15_fpDivTest_q == redist18_fracY_uid13_fpDivTest_b_14_q ? 1'b1 : 1'b0;

    // cstAllOWE_uid18_fpDivTest(CONSTANT,17)
    assign cstAllOWE_uid18_fpDivTest_q = 8'b11111111;

    // redist19_expY_uid12_fpDivTest_b_12_notEnable(LOGICAL,268)
    assign redist19_expY_uid12_fpDivTest_b_12_notEnable_q = ~ (en);

    // redist19_expY_uid12_fpDivTest_b_12_nor(LOGICAL,269)
    assign redist19_expY_uid12_fpDivTest_b_12_nor_q = ~ (redist19_expY_uid12_fpDivTest_b_12_notEnable_q | redist19_expY_uid12_fpDivTest_b_12_sticky_ena_q);

    // redist19_expY_uid12_fpDivTest_b_12_mem_last(CONSTANT,265)
    assign redist19_expY_uid12_fpDivTest_b_12_mem_last_q = 5'b01000;

    // redist19_expY_uid12_fpDivTest_b_12_cmp(LOGICAL,266)
    assign redist19_expY_uid12_fpDivTest_b_12_cmp_b = {1'b0, redist19_expY_uid12_fpDivTest_b_12_rdmux_q};
    assign redist19_expY_uid12_fpDivTest_b_12_cmp_q = redist19_expY_uid12_fpDivTest_b_12_mem_last_q == redist19_expY_uid12_fpDivTest_b_12_cmp_b ? 1'b1 : 1'b0;

    // redist19_expY_uid12_fpDivTest_b_12_cmpReg(REG,267)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist19_expY_uid12_fpDivTest_b_12_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist19_expY_uid12_fpDivTest_b_12_cmpReg_q <= redist19_expY_uid12_fpDivTest_b_12_cmp_q;
        end
    end

    // redist19_expY_uid12_fpDivTest_b_12_sticky_ena(REG,270)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist19_expY_uid12_fpDivTest_b_12_sticky_ena_q <= 1'b0;
        end
        else if (redist19_expY_uid12_fpDivTest_b_12_nor_q == 1'b1)
        begin
            redist19_expY_uid12_fpDivTest_b_12_sticky_ena_q <= redist19_expY_uid12_fpDivTest_b_12_cmpReg_q;
        end
    end

    // redist19_expY_uid12_fpDivTest_b_12_enaAnd(LOGICAL,271)
    assign redist19_expY_uid12_fpDivTest_b_12_enaAnd_q = redist19_expY_uid12_fpDivTest_b_12_sticky_ena_q & en;

    // redist19_expY_uid12_fpDivTest_b_12_rdcnt(COUNTER,262)
    // low=0, high=9, step=1, init=0
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist19_expY_uid12_fpDivTest_b_12_rdcnt_i <= 4'd0;
            redist19_expY_uid12_fpDivTest_b_12_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist19_expY_uid12_fpDivTest_b_12_rdcnt_i == 4'd8)
            begin
                redist19_expY_uid12_fpDivTest_b_12_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist19_expY_uid12_fpDivTest_b_12_rdcnt_eq <= 1'b0;
            end
            if (redist19_expY_uid12_fpDivTest_b_12_rdcnt_eq == 1'b1)
            begin
                redist19_expY_uid12_fpDivTest_b_12_rdcnt_i <= $unsigned(redist19_expY_uid12_fpDivTest_b_12_rdcnt_i) + $unsigned(4'd7);
            end
            else
            begin
                redist19_expY_uid12_fpDivTest_b_12_rdcnt_i <= $unsigned(redist19_expY_uid12_fpDivTest_b_12_rdcnt_i) + $unsigned(4'd1);
            end
        end
    end
    assign redist19_expY_uid12_fpDivTest_b_12_rdcnt_q = redist19_expY_uid12_fpDivTest_b_12_rdcnt_i[3:0];

    // redist19_expY_uid12_fpDivTest_b_12_rdmux(MUX,263)
    assign redist19_expY_uid12_fpDivTest_b_12_rdmux_s = en;
    always @(redist19_expY_uid12_fpDivTest_b_12_rdmux_s or redist19_expY_uid12_fpDivTest_b_12_wraddr_q or redist19_expY_uid12_fpDivTest_b_12_rdcnt_q)
    begin
        unique case (redist19_expY_uid12_fpDivTest_b_12_rdmux_s)
            1'b0 : redist19_expY_uid12_fpDivTest_b_12_rdmux_q = redist19_expY_uid12_fpDivTest_b_12_wraddr_q;
            1'b1 : redist19_expY_uid12_fpDivTest_b_12_rdmux_q = redist19_expY_uid12_fpDivTest_b_12_rdcnt_q;
            default : redist19_expY_uid12_fpDivTest_b_12_rdmux_q = 4'b0;
        endcase
    end

    // expY_uid12_fpDivTest(BITSELECT,11)@0
    assign expY_uid12_fpDivTest_b = b[30:23];

    // redist19_expY_uid12_fpDivTest_b_12_wraddr(REG,264)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist19_expY_uid12_fpDivTest_b_12_wraddr_q <= 4'b1001;
        end
        else
        begin
            redist19_expY_uid12_fpDivTest_b_12_wraddr_q <= redist19_expY_uid12_fpDivTest_b_12_rdmux_q;
        end
    end

    // redist19_expY_uid12_fpDivTest_b_12_mem(DUALMEM,261)
    assign redist19_expY_uid12_fpDivTest_b_12_mem_ia = expY_uid12_fpDivTest_b;
    assign redist19_expY_uid12_fpDivTest_b_12_mem_aa = redist19_expY_uid12_fpDivTest_b_12_wraddr_q;
    assign redist19_expY_uid12_fpDivTest_b_12_mem_ab = redist19_expY_uid12_fpDivTest_b_12_rdmux_q;
    assign redist19_expY_uid12_fpDivTest_b_12_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(8),
        .widthad_a(4),
        .numwords_a(10),
        .width_b(8),
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
    ) redist19_expY_uid12_fpDivTest_b_12_mem_dmem (
        .clocken1(redist19_expY_uid12_fpDivTest_b_12_enaAnd_q[0]),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .aclr1(redist19_expY_uid12_fpDivTest_b_12_mem_reset0),
        .clock1(clk),
        .address_a(redist19_expY_uid12_fpDivTest_b_12_mem_aa),
        .data_a(redist19_expY_uid12_fpDivTest_b_12_mem_ia),
        .wren_a(en[0]),
        .address_b(redist19_expY_uid12_fpDivTest_b_12_mem_ab),
        .q_b(redist19_expY_uid12_fpDivTest_b_12_mem_iq),
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
    assign redist19_expY_uid12_fpDivTest_b_12_mem_q = redist19_expY_uid12_fpDivTest_b_12_mem_iq[7:0];

    // redist19_expY_uid12_fpDivTest_b_12_outputreg(DELAY,260)
    dspba_delay_ver #( .width(8), .depth(1), .reset_kind("ASYNC") )
    redist19_expY_uid12_fpDivTest_b_12_outputreg ( .xin(redist19_expY_uid12_fpDivTest_b_12_mem_q), .xout(redist19_expY_uid12_fpDivTest_b_12_outputreg_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist20_expY_uid12_fpDivTest_b_14(DELAY,206)
    dspba_delay_ver #( .width(8), .depth(2), .reset_kind("ASYNC") )
    redist20_expY_uid12_fpDivTest_b_14 ( .xin(redist19_expY_uid12_fpDivTest_b_12_outputreg_q), .xout(redist20_expY_uid12_fpDivTest_b_14_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expXIsMax_uid38_fpDivTest(LOGICAL,37)@14
    assign expXIsMax_uid38_fpDivTest_q = redist20_expY_uid12_fpDivTest_b_14_q == cstAllOWE_uid18_fpDivTest_q ? 1'b1 : 1'b0;

    // excI_y_uid41_fpDivTest(LOGICAL,40)@14
    assign excI_y_uid41_fpDivTest_q = expXIsMax_uid38_fpDivTest_q & fracXIsZero_uid39_fpDivTest_q;

    // redist21_fracX_uid10_fpDivTest_b_10_notEnable(LOGICAL,280)
    assign redist21_fracX_uid10_fpDivTest_b_10_notEnable_q = ~ (en);

    // redist21_fracX_uid10_fpDivTest_b_10_nor(LOGICAL,281)
    assign redist21_fracX_uid10_fpDivTest_b_10_nor_q = ~ (redist21_fracX_uid10_fpDivTest_b_10_notEnable_q | redist21_fracX_uid10_fpDivTest_b_10_sticky_ena_q);

    // redist21_fracX_uid10_fpDivTest_b_10_mem_last(CONSTANT,277)
    assign redist21_fracX_uid10_fpDivTest_b_10_mem_last_q = 4'b0110;

    // redist21_fracX_uid10_fpDivTest_b_10_cmp(LOGICAL,278)
    assign redist21_fracX_uid10_fpDivTest_b_10_cmp_b = {1'b0, redist21_fracX_uid10_fpDivTest_b_10_rdmux_q};
    assign redist21_fracX_uid10_fpDivTest_b_10_cmp_q = redist21_fracX_uid10_fpDivTest_b_10_mem_last_q == redist21_fracX_uid10_fpDivTest_b_10_cmp_b ? 1'b1 : 1'b0;

    // redist21_fracX_uid10_fpDivTest_b_10_cmpReg(REG,279)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist21_fracX_uid10_fpDivTest_b_10_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist21_fracX_uid10_fpDivTest_b_10_cmpReg_q <= redist21_fracX_uid10_fpDivTest_b_10_cmp_q;
        end
    end

    // redist21_fracX_uid10_fpDivTest_b_10_sticky_ena(REG,282)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist21_fracX_uid10_fpDivTest_b_10_sticky_ena_q <= 1'b0;
        end
        else if (redist21_fracX_uid10_fpDivTest_b_10_nor_q == 1'b1)
        begin
            redist21_fracX_uid10_fpDivTest_b_10_sticky_ena_q <= redist21_fracX_uid10_fpDivTest_b_10_cmpReg_q;
        end
    end

    // redist21_fracX_uid10_fpDivTest_b_10_enaAnd(LOGICAL,283)
    assign redist21_fracX_uid10_fpDivTest_b_10_enaAnd_q = redist21_fracX_uid10_fpDivTest_b_10_sticky_ena_q & en;

    // redist21_fracX_uid10_fpDivTest_b_10_rdcnt(COUNTER,274)
    // low=0, high=7, step=1, init=0
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist21_fracX_uid10_fpDivTest_b_10_rdcnt_i <= 3'd0;
        end
        else if (en == 1'b1)
        begin
            redist21_fracX_uid10_fpDivTest_b_10_rdcnt_i <= $unsigned(redist21_fracX_uid10_fpDivTest_b_10_rdcnt_i) + $unsigned(3'd1);
        end
    end
    assign redist21_fracX_uid10_fpDivTest_b_10_rdcnt_q = redist21_fracX_uid10_fpDivTest_b_10_rdcnt_i[2:0];

    // redist21_fracX_uid10_fpDivTest_b_10_rdmux(MUX,275)
    assign redist21_fracX_uid10_fpDivTest_b_10_rdmux_s = en;
    always @(redist21_fracX_uid10_fpDivTest_b_10_rdmux_s or redist21_fracX_uid10_fpDivTest_b_10_wraddr_q or redist21_fracX_uid10_fpDivTest_b_10_rdcnt_q)
    begin
        unique case (redist21_fracX_uid10_fpDivTest_b_10_rdmux_s)
            1'b0 : redist21_fracX_uid10_fpDivTest_b_10_rdmux_q = redist21_fracX_uid10_fpDivTest_b_10_wraddr_q;
            1'b1 : redist21_fracX_uid10_fpDivTest_b_10_rdmux_q = redist21_fracX_uid10_fpDivTest_b_10_rdcnt_q;
            default : redist21_fracX_uid10_fpDivTest_b_10_rdmux_q = 3'b0;
        endcase
    end

    // fracX_uid10_fpDivTest(BITSELECT,9)@0
    assign fracX_uid10_fpDivTest_b = a[22:0];

    // redist21_fracX_uid10_fpDivTest_b_10_wraddr(REG,276)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist21_fracX_uid10_fpDivTest_b_10_wraddr_q <= 3'b111;
        end
        else
        begin
            redist21_fracX_uid10_fpDivTest_b_10_wraddr_q <= redist21_fracX_uid10_fpDivTest_b_10_rdmux_q;
        end
    end

    // redist21_fracX_uid10_fpDivTest_b_10_mem(DUALMEM,273)
    assign redist21_fracX_uid10_fpDivTest_b_10_mem_ia = fracX_uid10_fpDivTest_b;
    assign redist21_fracX_uid10_fpDivTest_b_10_mem_aa = redist21_fracX_uid10_fpDivTest_b_10_wraddr_q;
    assign redist21_fracX_uid10_fpDivTest_b_10_mem_ab = redist21_fracX_uid10_fpDivTest_b_10_rdmux_q;
    assign redist21_fracX_uid10_fpDivTest_b_10_mem_reset0 = areset;
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
    ) redist21_fracX_uid10_fpDivTest_b_10_mem_dmem (
        .clocken1(redist21_fracX_uid10_fpDivTest_b_10_enaAnd_q[0]),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .aclr1(redist21_fracX_uid10_fpDivTest_b_10_mem_reset0),
        .clock1(clk),
        .address_a(redist21_fracX_uid10_fpDivTest_b_10_mem_aa),
        .data_a(redist21_fracX_uid10_fpDivTest_b_10_mem_ia),
        .wren_a(en[0]),
        .address_b(redist21_fracX_uid10_fpDivTest_b_10_mem_ab),
        .q_b(redist21_fracX_uid10_fpDivTest_b_10_mem_iq),
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
    assign redist21_fracX_uid10_fpDivTest_b_10_mem_q = redist21_fracX_uid10_fpDivTest_b_10_mem_iq[22:0];

    // redist21_fracX_uid10_fpDivTest_b_10_outputreg(DELAY,272)
    dspba_delay_ver #( .width(23), .depth(1), .reset_kind("ASYNC") )
    redist21_fracX_uid10_fpDivTest_b_10_outputreg ( .xin(redist21_fracX_uid10_fpDivTest_b_10_mem_q), .xout(redist21_fracX_uid10_fpDivTest_b_10_outputreg_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist22_fracX_uid10_fpDivTest_b_14_inputreg(DELAY,284)
    dspba_delay_ver #( .width(23), .depth(1), .reset_kind("ASYNC") )
    redist22_fracX_uid10_fpDivTest_b_14_inputreg ( .xin(redist21_fracX_uid10_fpDivTest_b_10_outputreg_q), .xout(redist22_fracX_uid10_fpDivTest_b_14_inputreg_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist22_fracX_uid10_fpDivTest_b_14(DELAY,208)
    dspba_delay_ver #( .width(23), .depth(3), .reset_kind("ASYNC") )
    redist22_fracX_uid10_fpDivTest_b_14 ( .xin(redist22_fracX_uid10_fpDivTest_b_14_inputreg_q), .xout(redist22_fracX_uid10_fpDivTest_b_14_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // fracXIsZero_uid25_fpDivTest(LOGICAL,24)@14
    assign fracXIsZero_uid25_fpDivTest_q = paddingY_uid15_fpDivTest_q == redist22_fracX_uid10_fpDivTest_b_14_q ? 1'b1 : 1'b0;

    // redist24_expX_uid9_fpDivTest_b_12_notEnable(LOGICAL,294)
    assign redist24_expX_uid9_fpDivTest_b_12_notEnable_q = ~ (en);

    // redist24_expX_uid9_fpDivTest_b_12_nor(LOGICAL,295)
    assign redist24_expX_uid9_fpDivTest_b_12_nor_q = ~ (redist24_expX_uid9_fpDivTest_b_12_notEnable_q | redist24_expX_uid9_fpDivTest_b_12_sticky_ena_q);

    // redist24_expX_uid9_fpDivTest_b_12_mem_last(CONSTANT,291)
    assign redist24_expX_uid9_fpDivTest_b_12_mem_last_q = 5'b01000;

    // redist24_expX_uid9_fpDivTest_b_12_cmp(LOGICAL,292)
    assign redist24_expX_uid9_fpDivTest_b_12_cmp_b = {1'b0, redist24_expX_uid9_fpDivTest_b_12_rdmux_q};
    assign redist24_expX_uid9_fpDivTest_b_12_cmp_q = redist24_expX_uid9_fpDivTest_b_12_mem_last_q == redist24_expX_uid9_fpDivTest_b_12_cmp_b ? 1'b1 : 1'b0;

    // redist24_expX_uid9_fpDivTest_b_12_cmpReg(REG,293)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist24_expX_uid9_fpDivTest_b_12_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist24_expX_uid9_fpDivTest_b_12_cmpReg_q <= redist24_expX_uid9_fpDivTest_b_12_cmp_q;
        end
    end

    // redist24_expX_uid9_fpDivTest_b_12_sticky_ena(REG,296)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist24_expX_uid9_fpDivTest_b_12_sticky_ena_q <= 1'b0;
        end
        else if (redist24_expX_uid9_fpDivTest_b_12_nor_q == 1'b1)
        begin
            redist24_expX_uid9_fpDivTest_b_12_sticky_ena_q <= redist24_expX_uid9_fpDivTest_b_12_cmpReg_q;
        end
    end

    // redist24_expX_uid9_fpDivTest_b_12_enaAnd(LOGICAL,297)
    assign redist24_expX_uid9_fpDivTest_b_12_enaAnd_q = redist24_expX_uid9_fpDivTest_b_12_sticky_ena_q & en;

    // redist24_expX_uid9_fpDivTest_b_12_rdcnt(COUNTER,288)
    // low=0, high=9, step=1, init=0
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist24_expX_uid9_fpDivTest_b_12_rdcnt_i <= 4'd0;
            redist24_expX_uid9_fpDivTest_b_12_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist24_expX_uid9_fpDivTest_b_12_rdcnt_i == 4'd8)
            begin
                redist24_expX_uid9_fpDivTest_b_12_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist24_expX_uid9_fpDivTest_b_12_rdcnt_eq <= 1'b0;
            end
            if (redist24_expX_uid9_fpDivTest_b_12_rdcnt_eq == 1'b1)
            begin
                redist24_expX_uid9_fpDivTest_b_12_rdcnt_i <= $unsigned(redist24_expX_uid9_fpDivTest_b_12_rdcnt_i) + $unsigned(4'd7);
            end
            else
            begin
                redist24_expX_uid9_fpDivTest_b_12_rdcnt_i <= $unsigned(redist24_expX_uid9_fpDivTest_b_12_rdcnt_i) + $unsigned(4'd1);
            end
        end
    end
    assign redist24_expX_uid9_fpDivTest_b_12_rdcnt_q = redist24_expX_uid9_fpDivTest_b_12_rdcnt_i[3:0];

    // redist24_expX_uid9_fpDivTest_b_12_rdmux(MUX,289)
    assign redist24_expX_uid9_fpDivTest_b_12_rdmux_s = en;
    always @(redist24_expX_uid9_fpDivTest_b_12_rdmux_s or redist24_expX_uid9_fpDivTest_b_12_wraddr_q or redist24_expX_uid9_fpDivTest_b_12_rdcnt_q)
    begin
        unique case (redist24_expX_uid9_fpDivTest_b_12_rdmux_s)
            1'b0 : redist24_expX_uid9_fpDivTest_b_12_rdmux_q = redist24_expX_uid9_fpDivTest_b_12_wraddr_q;
            1'b1 : redist24_expX_uid9_fpDivTest_b_12_rdmux_q = redist24_expX_uid9_fpDivTest_b_12_rdcnt_q;
            default : redist24_expX_uid9_fpDivTest_b_12_rdmux_q = 4'b0;
        endcase
    end

    // expX_uid9_fpDivTest(BITSELECT,8)@0
    assign expX_uid9_fpDivTest_b = a[30:23];

    // redist24_expX_uid9_fpDivTest_b_12_wraddr(REG,290)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist24_expX_uid9_fpDivTest_b_12_wraddr_q <= 4'b1001;
        end
        else
        begin
            redist24_expX_uid9_fpDivTest_b_12_wraddr_q <= redist24_expX_uid9_fpDivTest_b_12_rdmux_q;
        end
    end

    // redist24_expX_uid9_fpDivTest_b_12_mem(DUALMEM,287)
    assign redist24_expX_uid9_fpDivTest_b_12_mem_ia = expX_uid9_fpDivTest_b;
    assign redist24_expX_uid9_fpDivTest_b_12_mem_aa = redist24_expX_uid9_fpDivTest_b_12_wraddr_q;
    assign redist24_expX_uid9_fpDivTest_b_12_mem_ab = redist24_expX_uid9_fpDivTest_b_12_rdmux_q;
    assign redist24_expX_uid9_fpDivTest_b_12_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(8),
        .widthad_a(4),
        .numwords_a(10),
        .width_b(8),
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
    ) redist24_expX_uid9_fpDivTest_b_12_mem_dmem (
        .clocken1(redist24_expX_uid9_fpDivTest_b_12_enaAnd_q[0]),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .aclr1(redist24_expX_uid9_fpDivTest_b_12_mem_reset0),
        .clock1(clk),
        .address_a(redist24_expX_uid9_fpDivTest_b_12_mem_aa),
        .data_a(redist24_expX_uid9_fpDivTest_b_12_mem_ia),
        .wren_a(en[0]),
        .address_b(redist24_expX_uid9_fpDivTest_b_12_mem_ab),
        .q_b(redist24_expX_uid9_fpDivTest_b_12_mem_iq),
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
    assign redist24_expX_uid9_fpDivTest_b_12_mem_q = redist24_expX_uid9_fpDivTest_b_12_mem_iq[7:0];

    // redist24_expX_uid9_fpDivTest_b_12_outputreg(DELAY,286)
    dspba_delay_ver #( .width(8), .depth(1), .reset_kind("ASYNC") )
    redist24_expX_uid9_fpDivTest_b_12_outputreg ( .xin(redist24_expX_uid9_fpDivTest_b_12_mem_q), .xout(redist24_expX_uid9_fpDivTest_b_12_outputreg_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist25_expX_uid9_fpDivTest_b_14(DELAY,211)
    dspba_delay_ver #( .width(8), .depth(2), .reset_kind("ASYNC") )
    redist25_expX_uid9_fpDivTest_b_14 ( .xin(redist24_expX_uid9_fpDivTest_b_12_outputreg_q), .xout(redist25_expX_uid9_fpDivTest_b_14_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expXIsMax_uid24_fpDivTest(LOGICAL,23)@14
    assign expXIsMax_uid24_fpDivTest_q = redist25_expX_uid9_fpDivTest_b_14_q == cstAllOWE_uid18_fpDivTest_q ? 1'b1 : 1'b0;

    // excI_x_uid27_fpDivTest(LOGICAL,26)@14
    assign excI_x_uid27_fpDivTest_q = expXIsMax_uid24_fpDivTest_q & fracXIsZero_uid25_fpDivTest_q;

    // excXIYI_uid130_fpDivTest(LOGICAL,129)@14
    assign excXIYI_uid130_fpDivTest_q = excI_x_uid27_fpDivTest_q & excI_y_uid41_fpDivTest_q;

    // fracXIsNotZero_uid40_fpDivTest(LOGICAL,39)@14
    assign fracXIsNotZero_uid40_fpDivTest_q = ~ (fracXIsZero_uid39_fpDivTest_q);

    // excN_y_uid42_fpDivTest(LOGICAL,41)@14
    assign excN_y_uid42_fpDivTest_q = expXIsMax_uid38_fpDivTest_q & fracXIsNotZero_uid40_fpDivTest_q;

    // fracXIsNotZero_uid26_fpDivTest(LOGICAL,25)@14
    assign fracXIsNotZero_uid26_fpDivTest_q = ~ (fracXIsZero_uid25_fpDivTest_q);

    // excN_x_uid28_fpDivTest(LOGICAL,27)@14
    assign excN_x_uid28_fpDivTest_q = expXIsMax_uid24_fpDivTest_q & fracXIsNotZero_uid26_fpDivTest_q;

    // cstAllZWE_uid20_fpDivTest(CONSTANT,19)
    assign cstAllZWE_uid20_fpDivTest_q = 8'b00000000;

    // excZ_y_uid37_fpDivTest(LOGICAL,36)@14
    assign excZ_y_uid37_fpDivTest_q = redist20_expY_uid12_fpDivTest_b_14_q == cstAllZWE_uid20_fpDivTest_q ? 1'b1 : 1'b0;

    // excZ_x_uid23_fpDivTest(LOGICAL,22)@14
    assign excZ_x_uid23_fpDivTest_q = redist25_expX_uid9_fpDivTest_b_14_q == cstAllZWE_uid20_fpDivTest_q ? 1'b1 : 1'b0;

    // excXZYZ_uid129_fpDivTest(LOGICAL,128)@14
    assign excXZYZ_uid129_fpDivTest_q = excZ_x_uid23_fpDivTest_q & excZ_y_uid37_fpDivTest_q;

    // excRNaN_uid131_fpDivTest(LOGICAL,130)@14 + 1
    assign excRNaN_uid131_fpDivTest_qi = excXZYZ_uid129_fpDivTest_q | excN_x_uid28_fpDivTest_q | excN_y_uid42_fpDivTest_q | excXIYI_uid130_fpDivTest_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    excRNaN_uid131_fpDivTest_delay ( .xin(excRNaN_uid131_fpDivTest_qi), .xout(excRNaN_uid131_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // invExcRNaN_uid142_fpDivTest(LOGICAL,141)@15
    assign invExcRNaN_uid142_fpDivTest_q = ~ (excRNaN_uid131_fpDivTest_q);

    // signY_uid14_fpDivTest(BITSELECT,13)@0
    assign signY_uid14_fpDivTest_b = b[31:31];

    // signX_uid11_fpDivTest(BITSELECT,10)@0
    assign signX_uid11_fpDivTest_b = a[31:31];

    // signR_uid46_fpDivTest(LOGICAL,45)@0 + 1
    assign signR_uid46_fpDivTest_qi = signX_uid11_fpDivTest_b ^ signY_uid14_fpDivTest_b;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    signR_uid46_fpDivTest_delay ( .xin(signR_uid46_fpDivTest_qi), .xout(signR_uid46_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist16_signR_uid46_fpDivTest_q_15(DELAY,202)
    dspba_delay_ver #( .width(1), .depth(14), .reset_kind("ASYNC") )
    redist16_signR_uid46_fpDivTest_q_15 ( .xin(signR_uid46_fpDivTest_q), .xout(redist16_signR_uid46_fpDivTest_q_15_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // sRPostExc_uid143_fpDivTest(LOGICAL,142)@15 + 1
    assign sRPostExc_uid143_fpDivTest_qi = redist16_signR_uid46_fpDivTest_q_15_q & invExcRNaN_uid142_fpDivTest_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    sRPostExc_uid143_fpDivTest_delay ( .xin(sRPostExc_uid143_fpDivTest_qi), .xout(sRPostExc_uid143_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist1_sRPostExc_uid143_fpDivTest_q_5(DELAY,187)
    dspba_delay_ver #( .width(1), .depth(4), .reset_kind("ASYNC") )
    redist1_sRPostExc_uid143_fpDivTest_q_5 ( .xin(sRPostExc_uid143_fpDivTest_q), .xout(redist1_sRPostExc_uid143_fpDivTest_q_5_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist8_fracPostRndF_uid80_fpDivTest_q_5_notEnable(LOGICAL,233)
    assign redist8_fracPostRndF_uid80_fpDivTest_q_5_notEnable_q = ~ (en);

    // redist8_fracPostRndF_uid80_fpDivTest_q_5_nor(LOGICAL,234)
    assign redist8_fracPostRndF_uid80_fpDivTest_q_5_nor_q = ~ (redist8_fracPostRndF_uid80_fpDivTest_q_5_notEnable_q | redist8_fracPostRndF_uid80_fpDivTest_q_5_sticky_ena_q);

    // redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_last(CONSTANT,230)
    assign redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_last_q = 2'b01;

    // redist8_fracPostRndF_uid80_fpDivTest_q_5_cmp(LOGICAL,231)
    assign redist8_fracPostRndF_uid80_fpDivTest_q_5_cmp_q = redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_last_q == redist8_fracPostRndF_uid80_fpDivTest_q_5_rdmux_q ? 1'b1 : 1'b0;

    // redist8_fracPostRndF_uid80_fpDivTest_q_5_cmpReg(REG,232)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist8_fracPostRndF_uid80_fpDivTest_q_5_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist8_fracPostRndF_uid80_fpDivTest_q_5_cmpReg_q <= redist8_fracPostRndF_uid80_fpDivTest_q_5_cmp_q;
        end
    end

    // redist8_fracPostRndF_uid80_fpDivTest_q_5_sticky_ena(REG,235)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist8_fracPostRndF_uid80_fpDivTest_q_5_sticky_ena_q <= 1'b0;
        end
        else if (redist8_fracPostRndF_uid80_fpDivTest_q_5_nor_q == 1'b1)
        begin
            redist8_fracPostRndF_uid80_fpDivTest_q_5_sticky_ena_q <= redist8_fracPostRndF_uid80_fpDivTest_q_5_cmpReg_q;
        end
    end

    // redist8_fracPostRndF_uid80_fpDivTest_q_5_enaAnd(LOGICAL,236)
    assign redist8_fracPostRndF_uid80_fpDivTest_q_5_enaAnd_q = redist8_fracPostRndF_uid80_fpDivTest_q_5_sticky_ena_q & en;

    // redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt(COUNTER,227)
    // low=0, high=2, step=1, init=0
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt_i <= 2'd0;
            redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt_i == 2'd1)
            begin
                redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt_eq <= 1'b0;
            end
            if (redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt_eq == 1'b1)
            begin
                redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt_i <= $unsigned(redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt_i) + $unsigned(2'd2);
            end
            else
            begin
                redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt_i <= $unsigned(redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt_i) + $unsigned(2'd1);
            end
        end
    end
    assign redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt_q = redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt_i[1:0];

    // redist8_fracPostRndF_uid80_fpDivTest_q_5_rdmux(MUX,228)
    assign redist8_fracPostRndF_uid80_fpDivTest_q_5_rdmux_s = en;
    always @(redist8_fracPostRndF_uid80_fpDivTest_q_5_rdmux_s or redist8_fracPostRndF_uid80_fpDivTest_q_5_wraddr_q or redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt_q)
    begin
        unique case (redist8_fracPostRndF_uid80_fpDivTest_q_5_rdmux_s)
            1'b0 : redist8_fracPostRndF_uid80_fpDivTest_q_5_rdmux_q = redist8_fracPostRndF_uid80_fpDivTest_q_5_wraddr_q;
            1'b1 : redist8_fracPostRndF_uid80_fpDivTest_q_5_rdmux_q = redist8_fracPostRndF_uid80_fpDivTest_q_5_rdcnt_q;
            default : redist8_fracPostRndF_uid80_fpDivTest_q_5_rdmux_q = 2'b0;
        endcase
    end

    // GND(CONSTANT,0)
    assign GND_q = 1'b0;

    // fracXExt_uid77_fpDivTest(BITJOIN,76)@14
    assign fracXExt_uid77_fpDivTest_q = {redist22_fracX_uid10_fpDivTest_b_14_q, GND_q};

    // lOAdded_uid57_fpDivTest(BITJOIN,56)@10
    assign lOAdded_uid57_fpDivTest_q = {VCC_q, redist21_fracX_uid10_fpDivTest_b_10_outputreg_q};

    // redist9_lOAdded_uid57_fpDivTest_q_3(DELAY,195)
    dspba_delay_ver #( .width(24), .depth(3), .reset_kind("ASYNC") )
    redist9_lOAdded_uid57_fpDivTest_q_3 ( .xin(lOAdded_uid57_fpDivTest_q), .xout(redist9_lOAdded_uid57_fpDivTest_q_3_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // z4_uid60_fpDivTest(CONSTANT,59)
    assign z4_uid60_fpDivTest_q = 4'b0000;

    // oFracXZ4_uid61_fpDivTest(BITJOIN,60)@13
    assign oFracXZ4_uid61_fpDivTest_q = {redist9_lOAdded_uid57_fpDivTest_q_3_q, z4_uid60_fpDivTest_q};

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
        .outdata_aclr_a("CLEAR0"),
        .clock_enable_input_a("NORMAL"),
        .power_up_uninitialized("FALSE"),
        .init_file("acl_fdiv_memoryC2_uid152_invTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Arria 10")
    ) memoryC2_uid152_invTables_lutmem_dmem (
        .clocken0(en[0]),
        .aclr0(memoryC2_uid152_invTables_lutmem_reset0),
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
    assign memoryC2_uid152_invTables_lutmem_r = memoryC2_uid152_invTables_lutmem_ir[12:0];

    // yPE_uid52_fpDivTest(BITSELECT,51)@0
    assign yPE_uid52_fpDivTest_b = b[13:0];

    // redist12_yPE_uid52_fpDivTest_b_2(DELAY,198)
    dspba_delay_ver #( .width(14), .depth(2), .reset_kind("ASYNC") )
    redist12_yPE_uid52_fpDivTest_b_2 ( .xin(yPE_uid52_fpDivTest_b), .xout(redist12_yPE_uid52_fpDivTest_b_2_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // yT1_uid158_invPolyEval(BITSELECT,157)@2
    assign yT1_uid158_invPolyEval_b = redist12_yPE_uid52_fpDivTest_b_2_q[13:1];

    // prodXY_uid174_pT1_uid159_invPolyEval_cma(CHAINMULTADD,184)@2 + 3
    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_reset = areset;
    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_ena0 = en[0];
    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_ena1 = prodXY_uid174_pT1_uid159_invPolyEval_cma_ena0;
    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_ena2 = prodXY_uid174_pT1_uid159_invPolyEval_cma_ena0;
    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_l[0] = $signed({1'b0, prodXY_uid174_pT1_uid159_invPolyEval_cma_a1[0][12:0]});
    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_p[0] = prodXY_uid174_pT1_uid159_invPolyEval_cma_l[0] * prodXY_uid174_pT1_uid159_invPolyEval_cma_c1[0];
    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_u[0] = prodXY_uid174_pT1_uid159_invPolyEval_cma_p[0][26:0];
    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_w[0] = prodXY_uid174_pT1_uid159_invPolyEval_cma_u[0];
    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_x[0] = prodXY_uid174_pT1_uid159_invPolyEval_cma_w[0];
    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_y[0] = prodXY_uid174_pT1_uid159_invPolyEval_cma_x[0];
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid174_pT1_uid159_invPolyEval_cma_a0 <= '{default: '0};
            prodXY_uid174_pT1_uid159_invPolyEval_cma_c0 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid174_pT1_uid159_invPolyEval_cma_ena0 == 1'b1)
            begin
                prodXY_uid174_pT1_uid159_invPolyEval_cma_a0[0] <= yT1_uid158_invPolyEval_b;
                prodXY_uid174_pT1_uid159_invPolyEval_cma_c0[0] <= memoryC2_uid152_invTables_lutmem_r;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid174_pT1_uid159_invPolyEval_cma_a1 <= '{default: '0};
            prodXY_uid174_pT1_uid159_invPolyEval_cma_c1 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid174_pT1_uid159_invPolyEval_cma_ena2 == 1'b1)
            begin
                prodXY_uid174_pT1_uid159_invPolyEval_cma_a1 <= prodXY_uid174_pT1_uid159_invPolyEval_cma_a0;
                prodXY_uid174_pT1_uid159_invPolyEval_cma_c1 <= prodXY_uid174_pT1_uid159_invPolyEval_cma_c0;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid174_pT1_uid159_invPolyEval_cma_s <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid174_pT1_uid159_invPolyEval_cma_ena1 == 1'b1)
            begin
                prodXY_uid174_pT1_uid159_invPolyEval_cma_s[0] <= prodXY_uid174_pT1_uid159_invPolyEval_cma_y[0];
            end
        end
    end
    dspba_delay_ver #( .width(26), .depth(0), .reset_kind("ASYNC") )
    prodXY_uid174_pT1_uid159_invPolyEval_cma_delay ( .xin(prodXY_uid174_pT1_uid159_invPolyEval_cma_s[0][25:0]), .xout(prodXY_uid174_pT1_uid159_invPolyEval_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign prodXY_uid174_pT1_uid159_invPolyEval_cma_q = prodXY_uid174_pT1_uid159_invPolyEval_cma_qq[25:0];

    // osig_uid175_pT1_uid159_invPolyEval(BITSELECT,174)@5
    assign osig_uid175_pT1_uid159_invPolyEval_b = prodXY_uid174_pT1_uid159_invPolyEval_cma_q[25:12];

    // highBBits_uid161_invPolyEval(BITSELECT,160)@5
    assign highBBits_uid161_invPolyEval_b = osig_uid175_pT1_uid159_invPolyEval_b[13:1];

    // redist14_yAddr_uid51_fpDivTest_b_3(DELAY,200)
    dspba_delay_ver #( .width(9), .depth(3), .reset_kind("ASYNC") )
    redist14_yAddr_uid51_fpDivTest_b_3 ( .xin(yAddr_uid51_fpDivTest_b), .xout(redist14_yAddr_uid51_fpDivTest_b_3_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // memoryC1_uid149_invTables_lutmem(DUALMEM,180)@3 + 2
    // in j@20000000
    assign memoryC1_uid149_invTables_lutmem_aa = redist14_yAddr_uid51_fpDivTest_b_3_q;
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
        .outdata_aclr_a("CLEAR0"),
        .clock_enable_input_a("NORMAL"),
        .power_up_uninitialized("FALSE"),
        .init_file("acl_fdiv_memoryC1_uid149_invTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Arria 10")
    ) memoryC1_uid149_invTables_lutmem_dmem (
        .clocken0(en[0]),
        .aclr0(memoryC1_uid149_invTables_lutmem_reset0),
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
    assign memoryC1_uid149_invTables_lutmem_r = memoryC1_uid149_invTables_lutmem_ir[21:0];

    // s1sumAHighB_uid162_invPolyEval(ADD,161)@5 + 1
    assign s1sumAHighB_uid162_invPolyEval_a = {{1{memoryC1_uid149_invTables_lutmem_r[21]}}, memoryC1_uid149_invTables_lutmem_r};
    assign s1sumAHighB_uid162_invPolyEval_b = {{10{highBBits_uid161_invPolyEval_b[12]}}, highBBits_uid161_invPolyEval_b};
    always @ (posedge clk or posedge areset)
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

    // lowRangeB_uid160_invPolyEval(BITSELECT,159)@5
    assign lowRangeB_uid160_invPolyEval_in = osig_uid175_pT1_uid159_invPolyEval_b[0:0];
    assign lowRangeB_uid160_invPolyEval_b = lowRangeB_uid160_invPolyEval_in[0:0];

    // redist0_lowRangeB_uid160_invPolyEval_b_1(DELAY,186)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist0_lowRangeB_uid160_invPolyEval_b_1 ( .xin(lowRangeB_uid160_invPolyEval_b), .xout(redist0_lowRangeB_uid160_invPolyEval_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // s1_uid163_invPolyEval(BITJOIN,162)@6
    assign s1_uid163_invPolyEval_q = {s1sumAHighB_uid162_invPolyEval_q, redist0_lowRangeB_uid160_invPolyEval_b_1_q};

    // redist13_yPE_uid52_fpDivTest_b_6_notEnable(LOGICAL,244)
    assign redist13_yPE_uid52_fpDivTest_b_6_notEnable_q = ~ (en);

    // redist13_yPE_uid52_fpDivTest_b_6_nor(LOGICAL,245)
    assign redist13_yPE_uid52_fpDivTest_b_6_nor_q = ~ (redist13_yPE_uid52_fpDivTest_b_6_notEnable_q | redist13_yPE_uid52_fpDivTest_b_6_sticky_ena_q);

    // redist13_yPE_uid52_fpDivTest_b_6_mem_last(CONSTANT,241)
    assign redist13_yPE_uid52_fpDivTest_b_6_mem_last_q = 2'b01;

    // redist13_yPE_uid52_fpDivTest_b_6_cmp(LOGICAL,242)
    assign redist13_yPE_uid52_fpDivTest_b_6_cmp_q = redist13_yPE_uid52_fpDivTest_b_6_mem_last_q == redist13_yPE_uid52_fpDivTest_b_6_rdmux_q ? 1'b1 : 1'b0;

    // redist13_yPE_uid52_fpDivTest_b_6_cmpReg(REG,243)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist13_yPE_uid52_fpDivTest_b_6_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist13_yPE_uid52_fpDivTest_b_6_cmpReg_q <= redist13_yPE_uid52_fpDivTest_b_6_cmp_q;
        end
    end

    // redist13_yPE_uid52_fpDivTest_b_6_sticky_ena(REG,246)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist13_yPE_uid52_fpDivTest_b_6_sticky_ena_q <= 1'b0;
        end
        else if (redist13_yPE_uid52_fpDivTest_b_6_nor_q == 1'b1)
        begin
            redist13_yPE_uid52_fpDivTest_b_6_sticky_ena_q <= redist13_yPE_uid52_fpDivTest_b_6_cmpReg_q;
        end
    end

    // redist13_yPE_uid52_fpDivTest_b_6_enaAnd(LOGICAL,247)
    assign redist13_yPE_uid52_fpDivTest_b_6_enaAnd_q = redist13_yPE_uid52_fpDivTest_b_6_sticky_ena_q & en;

    // redist13_yPE_uid52_fpDivTest_b_6_rdcnt(COUNTER,238)
    // low=0, high=2, step=1, init=0
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist13_yPE_uid52_fpDivTest_b_6_rdcnt_i <= 2'd0;
            redist13_yPE_uid52_fpDivTest_b_6_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist13_yPE_uid52_fpDivTest_b_6_rdcnt_i == 2'd1)
            begin
                redist13_yPE_uid52_fpDivTest_b_6_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist13_yPE_uid52_fpDivTest_b_6_rdcnt_eq <= 1'b0;
            end
            if (redist13_yPE_uid52_fpDivTest_b_6_rdcnt_eq == 1'b1)
            begin
                redist13_yPE_uid52_fpDivTest_b_6_rdcnt_i <= $unsigned(redist13_yPE_uid52_fpDivTest_b_6_rdcnt_i) + $unsigned(2'd2);
            end
            else
            begin
                redist13_yPE_uid52_fpDivTest_b_6_rdcnt_i <= $unsigned(redist13_yPE_uid52_fpDivTest_b_6_rdcnt_i) + $unsigned(2'd1);
            end
        end
    end
    assign redist13_yPE_uid52_fpDivTest_b_6_rdcnt_q = redist13_yPE_uid52_fpDivTest_b_6_rdcnt_i[1:0];

    // redist13_yPE_uid52_fpDivTest_b_6_rdmux(MUX,239)
    assign redist13_yPE_uid52_fpDivTest_b_6_rdmux_s = en;
    always @(redist13_yPE_uid52_fpDivTest_b_6_rdmux_s or redist13_yPE_uid52_fpDivTest_b_6_wraddr_q or redist13_yPE_uid52_fpDivTest_b_6_rdcnt_q)
    begin
        unique case (redist13_yPE_uid52_fpDivTest_b_6_rdmux_s)
            1'b0 : redist13_yPE_uid52_fpDivTest_b_6_rdmux_q = redist13_yPE_uid52_fpDivTest_b_6_wraddr_q;
            1'b1 : redist13_yPE_uid52_fpDivTest_b_6_rdmux_q = redist13_yPE_uid52_fpDivTest_b_6_rdcnt_q;
            default : redist13_yPE_uid52_fpDivTest_b_6_rdmux_q = 2'b0;
        endcase
    end

    // redist13_yPE_uid52_fpDivTest_b_6_wraddr(REG,240)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist13_yPE_uid52_fpDivTest_b_6_wraddr_q <= 2'b10;
        end
        else
        begin
            redist13_yPE_uid52_fpDivTest_b_6_wraddr_q <= redist13_yPE_uid52_fpDivTest_b_6_rdmux_q;
        end
    end

    // redist13_yPE_uid52_fpDivTest_b_6_mem(DUALMEM,237)
    assign redist13_yPE_uid52_fpDivTest_b_6_mem_ia = redist12_yPE_uid52_fpDivTest_b_2_q;
    assign redist13_yPE_uid52_fpDivTest_b_6_mem_aa = redist13_yPE_uid52_fpDivTest_b_6_wraddr_q;
    assign redist13_yPE_uid52_fpDivTest_b_6_mem_ab = redist13_yPE_uid52_fpDivTest_b_6_rdmux_q;
    assign redist13_yPE_uid52_fpDivTest_b_6_mem_reset0 = areset;
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
    ) redist13_yPE_uid52_fpDivTest_b_6_mem_dmem (
        .clocken1(redist13_yPE_uid52_fpDivTest_b_6_enaAnd_q[0]),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .aclr1(redist13_yPE_uid52_fpDivTest_b_6_mem_reset0),
        .clock1(clk),
        .address_a(redist13_yPE_uid52_fpDivTest_b_6_mem_aa),
        .data_a(redist13_yPE_uid52_fpDivTest_b_6_mem_ia),
        .wren_a(en[0]),
        .address_b(redist13_yPE_uid52_fpDivTest_b_6_mem_ab),
        .q_b(redist13_yPE_uid52_fpDivTest_b_6_mem_iq),
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
    assign redist13_yPE_uid52_fpDivTest_b_6_mem_q = redist13_yPE_uid52_fpDivTest_b_6_mem_iq[13:0];

    // prodXY_uid177_pT2_uid165_invPolyEval_cma(CHAINMULTADD,185)@6 + 3
    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_reset = areset;
    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_ena0 = en[0];
    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_ena1 = prodXY_uid177_pT2_uid165_invPolyEval_cma_ena0;
    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_ena2 = prodXY_uid177_pT2_uid165_invPolyEval_cma_ena0;
    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_l[0] = $signed({1'b0, prodXY_uid177_pT2_uid165_invPolyEval_cma_a1[0][13:0]});
    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_p[0] = prodXY_uid177_pT2_uid165_invPolyEval_cma_l[0] * prodXY_uid177_pT2_uid165_invPolyEval_cma_c1[0];
    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_u[0] = prodXY_uid177_pT2_uid165_invPolyEval_cma_p[0][38:0];
    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_w[0] = prodXY_uid177_pT2_uid165_invPolyEval_cma_u[0];
    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_x[0] = prodXY_uid177_pT2_uid165_invPolyEval_cma_w[0];
    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_y[0] = prodXY_uid177_pT2_uid165_invPolyEval_cma_x[0];
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid177_pT2_uid165_invPolyEval_cma_a0 <= '{default: '0};
            prodXY_uid177_pT2_uid165_invPolyEval_cma_c0 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid177_pT2_uid165_invPolyEval_cma_ena0 == 1'b1)
            begin
                prodXY_uid177_pT2_uid165_invPolyEval_cma_a0[0] <= redist13_yPE_uid52_fpDivTest_b_6_mem_q;
                prodXY_uid177_pT2_uid165_invPolyEval_cma_c0[0] <= s1_uid163_invPolyEval_q;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid177_pT2_uid165_invPolyEval_cma_a1 <= '{default: '0};
            prodXY_uid177_pT2_uid165_invPolyEval_cma_c1 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid177_pT2_uid165_invPolyEval_cma_ena2 == 1'b1)
            begin
                prodXY_uid177_pT2_uid165_invPolyEval_cma_a1 <= prodXY_uid177_pT2_uid165_invPolyEval_cma_a0;
                prodXY_uid177_pT2_uid165_invPolyEval_cma_c1 <= prodXY_uid177_pT2_uid165_invPolyEval_cma_c0;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid177_pT2_uid165_invPolyEval_cma_s <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid177_pT2_uid165_invPolyEval_cma_ena1 == 1'b1)
            begin
                prodXY_uid177_pT2_uid165_invPolyEval_cma_s[0] <= prodXY_uid177_pT2_uid165_invPolyEval_cma_y[0];
            end
        end
    end
    dspba_delay_ver #( .width(38), .depth(0), .reset_kind("ASYNC") )
    prodXY_uid177_pT2_uid165_invPolyEval_cma_delay ( .xin(prodXY_uid177_pT2_uid165_invPolyEval_cma_s[0][37:0]), .xout(prodXY_uid177_pT2_uid165_invPolyEval_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign prodXY_uid177_pT2_uid165_invPolyEval_cma_q = prodXY_uid177_pT2_uid165_invPolyEval_cma_qq[37:0];

    // osig_uid178_pT2_uid165_invPolyEval(BITSELECT,177)@9
    assign osig_uid178_pT2_uid165_invPolyEval_b = prodXY_uid177_pT2_uid165_invPolyEval_cma_q[37:13];

    // highBBits_uid167_invPolyEval(BITSELECT,166)@9
    assign highBBits_uid167_invPolyEval_b = osig_uid178_pT2_uid165_invPolyEval_b[24:2];

    // redist15_yAddr_uid51_fpDivTest_b_7(DELAY,201)
    dspba_delay_ver #( .width(9), .depth(4), .reset_kind("ASYNC") )
    redist15_yAddr_uid51_fpDivTest_b_7 ( .xin(redist14_yAddr_uid51_fpDivTest_b_3_q), .xout(redist15_yAddr_uid51_fpDivTest_b_7_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // memoryC0_uid146_invTables_lutmem(DUALMEM,179)@7 + 2
    // in j@20000000
    assign memoryC0_uid146_invTables_lutmem_aa = redist15_yAddr_uid51_fpDivTest_b_7_q;
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
        .outdata_aclr_a("CLEAR0"),
        .clock_enable_input_a("NORMAL"),
        .power_up_uninitialized("FALSE"),
        .init_file("acl_fdiv_memoryC0_uid146_invTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Arria 10")
    ) memoryC0_uid146_invTables_lutmem_dmem (
        .clocken0(en[0]),
        .aclr0(memoryC0_uid146_invTables_lutmem_reset0),
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
    assign memoryC0_uid146_invTables_lutmem_r = memoryC0_uid146_invTables_lutmem_ir[31:0];

    // s2sumAHighB_uid168_invPolyEval(ADD,167)@9
    assign s2sumAHighB_uid168_invPolyEval_a = {{1{memoryC0_uid146_invTables_lutmem_r[31]}}, memoryC0_uid146_invTables_lutmem_r};
    assign s2sumAHighB_uid168_invPolyEval_b = {{10{highBBits_uid167_invPolyEval_b[22]}}, highBBits_uid167_invPolyEval_b};
    assign s2sumAHighB_uid168_invPolyEval_o = $signed(s2sumAHighB_uid168_invPolyEval_a) + $signed(s2sumAHighB_uid168_invPolyEval_b);
    assign s2sumAHighB_uid168_invPolyEval_q = s2sumAHighB_uid168_invPolyEval_o[32:0];

    // lowRangeB_uid166_invPolyEval(BITSELECT,165)@9
    assign lowRangeB_uid166_invPolyEval_in = osig_uid178_pT2_uid165_invPolyEval_b[1:0];
    assign lowRangeB_uid166_invPolyEval_b = lowRangeB_uid166_invPolyEval_in[1:0];

    // s2_uid169_invPolyEval(BITJOIN,168)@9
    assign s2_uid169_invPolyEval_q = {s2sumAHighB_uid168_invPolyEval_q, lowRangeB_uid166_invPolyEval_b};

    // invY_uid54_fpDivTest(BITSELECT,53)@9
    assign invY_uid54_fpDivTest_in = s2_uid169_invPolyEval_q[31:0];
    assign invY_uid54_fpDivTest_b = invY_uid54_fpDivTest_in[31:5];

    // redist11_invY_uid54_fpDivTest_b_1(DELAY,197)
    dspba_delay_ver #( .width(27), .depth(1), .reset_kind("ASYNC") )
    redist11_invY_uid54_fpDivTest_b_1 ( .xin(invY_uid54_fpDivTest_b), .xout(redist11_invY_uid54_fpDivTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma(CHAINMULTADD,183)@10 + 3
    assign prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_reset = areset;
    assign prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena0 = en[0];
    assign prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena1 = prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena0;
    assign prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena2 = prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena0;
    assign prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_p[0] = prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_a1[0] * prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_c1[0];
    assign prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_u[0] = prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_p[0][50:0];
    assign prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_w[0] = prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_u[0];
    assign prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_x[0] = prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_w[0];
    assign prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_y[0] = prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_x[0];
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_a0 <= '{default: '0};
            prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_c0 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena0 == 1'b1)
            begin
                prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_a0[0] <= redist11_invY_uid54_fpDivTest_b_1_q;
                prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_c0[0] <= lOAdded_uid57_fpDivTest_q;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_a1 <= '{default: '0};
            prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_c1 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena2 == 1'b1)
            begin
                prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_a1 <= prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_a0;
                prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_c1 <= prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_c0;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_s <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_ena1 == 1'b1)
            begin
                prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_s[0] <= prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_y[0];
            end
        end
    end
    dspba_delay_ver #( .width(51), .depth(0), .reset_kind("ASYNC") )
    prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_delay ( .xin(prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_s[0][50:0]), .xout(prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_q = prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_qq[50:0];

    // osig_uid172_divValPreNorm_uid59_fpDivTest(BITSELECT,171)@13
    assign osig_uid172_divValPreNorm_uid59_fpDivTest_b = prodXY_uid171_divValPreNorm_uid59_fpDivTest_cma_q[50:23];

    // updatedY_uid16_fpDivTest(BITJOIN,15)@12
    assign updatedY_uid16_fpDivTest_q = {GND_q, paddingY_uid15_fpDivTest_q};

    // fracYZero_uid15_fpDivTest(LOGICAL,16)@12 + 1
    assign fracYZero_uid15_fpDivTest_a = {1'b0, redist17_fracY_uid13_fpDivTest_b_12_outputreg_q};
    assign fracYZero_uid15_fpDivTest_qi = fracYZero_uid15_fpDivTest_a == updatedY_uid16_fpDivTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    fracYZero_uid15_fpDivTest_delay ( .xin(fracYZero_uid15_fpDivTest_qi), .xout(fracYZero_uid15_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // divValPreNormYPow2Exc_uid63_fpDivTest(MUX,62)@13
    assign divValPreNormYPow2Exc_uid63_fpDivTest_s = fracYZero_uid15_fpDivTest_q;
    always @(divValPreNormYPow2Exc_uid63_fpDivTest_s or en or osig_uid172_divValPreNorm_uid59_fpDivTest_b or oFracXZ4_uid61_fpDivTest_q)
    begin
        unique case (divValPreNormYPow2Exc_uid63_fpDivTest_s)
            1'b0 : divValPreNormYPow2Exc_uid63_fpDivTest_q = osig_uid172_divValPreNorm_uid59_fpDivTest_b;
            1'b1 : divValPreNormYPow2Exc_uid63_fpDivTest_q = oFracXZ4_uid61_fpDivTest_q;
            default : divValPreNormYPow2Exc_uid63_fpDivTest_q = 28'b0;
        endcase
    end

    // norm_uid64_fpDivTest(BITSELECT,63)@13
    assign norm_uid64_fpDivTest_b = divValPreNormYPow2Exc_uid63_fpDivTest_q[27:27];

    // zeroPaddingInAddition_uid74_fpDivTest(CONSTANT,73)
    assign zeroPaddingInAddition_uid74_fpDivTest_q = 24'b000000000000000000000000;

    // expFracPostRnd_uid75_fpDivTest(BITJOIN,74)@13
    assign expFracPostRnd_uid75_fpDivTest_q = {norm_uid64_fpDivTest_b, zeroPaddingInAddition_uid74_fpDivTest_q, VCC_q};

    // cstBiasM1_uid6_fpDivTest(CONSTANT,5)
    assign cstBiasM1_uid6_fpDivTest_q = 8'b01111110;

    // expXmY_uid47_fpDivTest(SUB,46)@12 + 1
    assign expXmY_uid47_fpDivTest_a = {1'b0, redist24_expX_uid9_fpDivTest_b_12_outputreg_q};
    assign expXmY_uid47_fpDivTest_b = {1'b0, redist19_expY_uid12_fpDivTest_b_12_outputreg_q};
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

    // expR_uid48_fpDivTest(ADD,47)@13
    assign expR_uid48_fpDivTest_a = {{2{expXmY_uid47_fpDivTest_q[8]}}, expXmY_uid47_fpDivTest_q};
    assign expR_uid48_fpDivTest_b = {3'b000, cstBiasM1_uid6_fpDivTest_q};
    assign expR_uid48_fpDivTest_o = $signed(expR_uid48_fpDivTest_a) + $signed(expR_uid48_fpDivTest_b);
    assign expR_uid48_fpDivTest_q = expR_uid48_fpDivTest_o[9:0];

    // divValPreNormHigh_uid65_fpDivTest(BITSELECT,64)@13
    assign divValPreNormHigh_uid65_fpDivTest_in = divValPreNormYPow2Exc_uid63_fpDivTest_q[26:0];
    assign divValPreNormHigh_uid65_fpDivTest_b = divValPreNormHigh_uid65_fpDivTest_in[26:2];

    // divValPreNormLow_uid66_fpDivTest(BITSELECT,65)@13
    assign divValPreNormLow_uid66_fpDivTest_in = divValPreNormYPow2Exc_uid63_fpDivTest_q[25:0];
    assign divValPreNormLow_uid66_fpDivTest_b = divValPreNormLow_uid66_fpDivTest_in[25:1];

    // normFracRnd_uid67_fpDivTest(MUX,66)@13
    assign normFracRnd_uid67_fpDivTest_s = norm_uid64_fpDivTest_b;
    always @(normFracRnd_uid67_fpDivTest_s or en or divValPreNormLow_uid66_fpDivTest_b or divValPreNormHigh_uid65_fpDivTest_b)
    begin
        unique case (normFracRnd_uid67_fpDivTest_s)
            1'b0 : normFracRnd_uid67_fpDivTest_q = divValPreNormLow_uid66_fpDivTest_b;
            1'b1 : normFracRnd_uid67_fpDivTest_q = divValPreNormHigh_uid65_fpDivTest_b;
            default : normFracRnd_uid67_fpDivTest_q = 25'b0;
        endcase
    end

    // expFracRnd_uid68_fpDivTest(BITJOIN,67)@13
    assign expFracRnd_uid68_fpDivTest_q = {expR_uid48_fpDivTest_q, normFracRnd_uid67_fpDivTest_q};

    // expFracPostRnd_uid76_fpDivTest(ADD,75)@13 + 1
    assign expFracPostRnd_uid76_fpDivTest_a = {{2{expFracRnd_uid68_fpDivTest_q[34]}}, expFracRnd_uid68_fpDivTest_q};
    assign expFracPostRnd_uid76_fpDivTest_b = {11'b00000000000, expFracPostRnd_uid75_fpDivTest_q};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            expFracPostRnd_uid76_fpDivTest_o <= 37'b0;
        end
        else if (en == 1'b1)
        begin
            expFracPostRnd_uid76_fpDivTest_o <= $signed(expFracPostRnd_uid76_fpDivTest_a) + $signed(expFracPostRnd_uid76_fpDivTest_b);
        end
    end
    assign expFracPostRnd_uid76_fpDivTest_q = expFracPostRnd_uid76_fpDivTest_o[35:0];

    // fracPostRndF_uid79_fpDivTest(BITSELECT,78)@14
    assign fracPostRndF_uid79_fpDivTest_in = expFracPostRnd_uid76_fpDivTest_q[24:0];
    assign fracPostRndF_uid79_fpDivTest_b = fracPostRndF_uid79_fpDivTest_in[24:1];

    // invYO_uid55_fpDivTest(BITSELECT,54)@9
    assign invYO_uid55_fpDivTest_in = s2_uid169_invPolyEval_q[32:0];
    assign invYO_uid55_fpDivTest_b = invYO_uid55_fpDivTest_in[32:32];

    // redist10_invYO_uid55_fpDivTest_b_5(DELAY,196)
    dspba_delay_ver #( .width(1), .depth(5), .reset_kind("ASYNC") )
    redist10_invYO_uid55_fpDivTest_b_5 ( .xin(invYO_uid55_fpDivTest_b), .xout(redist10_invYO_uid55_fpDivTest_b_5_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // fracPostRndF_uid80_fpDivTest(MUX,79)@14
    assign fracPostRndF_uid80_fpDivTest_s = redist10_invYO_uid55_fpDivTest_b_5_q;
    always @(fracPostRndF_uid80_fpDivTest_s or en or fracPostRndF_uid79_fpDivTest_b or fracXExt_uid77_fpDivTest_q)
    begin
        unique case (fracPostRndF_uid80_fpDivTest_s)
            1'b0 : fracPostRndF_uid80_fpDivTest_q = fracPostRndF_uid79_fpDivTest_b;
            1'b1 : fracPostRndF_uid80_fpDivTest_q = fracXExt_uid77_fpDivTest_q;
            default : fracPostRndF_uid80_fpDivTest_q = 24'b0;
        endcase
    end

    // redist8_fracPostRndF_uid80_fpDivTest_q_5_wraddr(REG,229)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist8_fracPostRndF_uid80_fpDivTest_q_5_wraddr_q <= 2'b10;
        end
        else
        begin
            redist8_fracPostRndF_uid80_fpDivTest_q_5_wraddr_q <= redist8_fracPostRndF_uid80_fpDivTest_q_5_rdmux_q;
        end
    end

    // redist8_fracPostRndF_uid80_fpDivTest_q_5_mem(DUALMEM,226)
    assign redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_ia = fracPostRndF_uid80_fpDivTest_q;
    assign redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_aa = redist8_fracPostRndF_uid80_fpDivTest_q_5_wraddr_q;
    assign redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_ab = redist8_fracPostRndF_uid80_fpDivTest_q_5_rdmux_q;
    assign redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(24),
        .widthad_a(2),
        .numwords_a(3),
        .width_b(24),
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
    ) redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_dmem (
        .clocken1(redist8_fracPostRndF_uid80_fpDivTest_q_5_enaAnd_q[0]),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .aclr1(redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_reset0),
        .clock1(clk),
        .address_a(redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_aa),
        .data_a(redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_ia),
        .wren_a(en[0]),
        .address_b(redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_ab),
        .q_b(redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_iq),
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
    assign redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_q = redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_iq[23:0];

    // redist8_fracPostRndF_uid80_fpDivTest_q_5_outputreg(DELAY,225)
    dspba_delay_ver #( .width(24), .depth(1), .reset_kind("ASYNC") )
    redist8_fracPostRndF_uid80_fpDivTest_q_5_outputreg ( .xin(redist8_fracPostRndF_uid80_fpDivTest_q_5_mem_q), .xout(redist8_fracPostRndF_uid80_fpDivTest_q_5_outputreg_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // betweenFPwF_uid102_fpDivTest(BITSELECT,101)@19
    assign betweenFPwF_uid102_fpDivTest_in = redist8_fracPostRndF_uid80_fpDivTest_q_5_outputreg_q[0:0];
    assign betweenFPwF_uid102_fpDivTest_b = betweenFPwF_uid102_fpDivTest_in[0:0];

    // redist26_expX_uid9_fpDivTest_b_18(DELAY,212)
    dspba_delay_ver #( .width(8), .depth(4), .reset_kind("ASYNC") )
    redist26_expX_uid9_fpDivTest_b_18 ( .xin(redist25_expX_uid9_fpDivTest_b_14_q), .xout(redist26_expX_uid9_fpDivTest_b_18_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist23_fracX_uid10_fpDivTest_b_18_inputreg(DELAY,285)
    dspba_delay_ver #( .width(23), .depth(1), .reset_kind("ASYNC") )
    redist23_fracX_uid10_fpDivTest_b_18_inputreg ( .xin(redist22_fracX_uid10_fpDivTest_b_14_q), .xout(redist23_fracX_uid10_fpDivTest_b_18_inputreg_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist23_fracX_uid10_fpDivTest_b_18(DELAY,209)
    dspba_delay_ver #( .width(23), .depth(3), .reset_kind("ASYNC") )
    redist23_fracX_uid10_fpDivTest_b_18 ( .xin(redist23_fracX_uid10_fpDivTest_b_18_inputreg_q), .xout(redist23_fracX_uid10_fpDivTest_b_18_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // qDivProdLTX_opB_uid100_fpDivTest(BITJOIN,99)@18
    assign qDivProdLTX_opB_uid100_fpDivTest_q = {redist26_expX_uid9_fpDivTest_b_18_q, redist23_fracX_uid10_fpDivTest_b_18_q};

    // lOAdded_uid87_fpDivTest(BITJOIN,86)@14
    assign lOAdded_uid87_fpDivTest_q = {VCC_q, redist18_fracY_uid13_fpDivTest_b_14_q};

    // lOAdded_uid84_fpDivTest(BITJOIN,83)@14
    assign lOAdded_uid84_fpDivTest_q = {VCC_q, fracPostRndF_uid80_fpDivTest_q};

    // qDivProd_uid89_fpDivTest_cma(CHAINMULTADD,182)@14 + 3
    assign qDivProd_uid89_fpDivTest_cma_reset = areset;
    assign qDivProd_uid89_fpDivTest_cma_ena0 = en[0];
    assign qDivProd_uid89_fpDivTest_cma_ena1 = qDivProd_uid89_fpDivTest_cma_ena0;
    assign qDivProd_uid89_fpDivTest_cma_ena2 = qDivProd_uid89_fpDivTest_cma_ena0;
    assign qDivProd_uid89_fpDivTest_cma_p[0] = qDivProd_uid89_fpDivTest_cma_a1[0] * qDivProd_uid89_fpDivTest_cma_c1[0];
    assign qDivProd_uid89_fpDivTest_cma_u[0] = qDivProd_uid89_fpDivTest_cma_p[0][48:0];
    assign qDivProd_uid89_fpDivTest_cma_w[0] = qDivProd_uid89_fpDivTest_cma_u[0];
    assign qDivProd_uid89_fpDivTest_cma_x[0] = qDivProd_uid89_fpDivTest_cma_w[0];
    assign qDivProd_uid89_fpDivTest_cma_y[0] = qDivProd_uid89_fpDivTest_cma_x[0];
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            qDivProd_uid89_fpDivTest_cma_a0 <= '{default: '0};
            qDivProd_uid89_fpDivTest_cma_c0 <= '{default: '0};
        end
        else
        begin
            if (qDivProd_uid89_fpDivTest_cma_ena0 == 1'b1)
            begin
                qDivProd_uid89_fpDivTest_cma_a0[0] <= lOAdded_uid84_fpDivTest_q;
                qDivProd_uid89_fpDivTest_cma_c0[0] <= lOAdded_uid87_fpDivTest_q;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            qDivProd_uid89_fpDivTest_cma_a1 <= '{default: '0};
            qDivProd_uid89_fpDivTest_cma_c1 <= '{default: '0};
        end
        else
        begin
            if (qDivProd_uid89_fpDivTest_cma_ena2 == 1'b1)
            begin
                qDivProd_uid89_fpDivTest_cma_a1 <= qDivProd_uid89_fpDivTest_cma_a0;
                qDivProd_uid89_fpDivTest_cma_c1 <= qDivProd_uid89_fpDivTest_cma_c0;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            qDivProd_uid89_fpDivTest_cma_s <= '{default: '0};
        end
        else
        begin
            if (qDivProd_uid89_fpDivTest_cma_ena1 == 1'b1)
            begin
                qDivProd_uid89_fpDivTest_cma_s[0] <= qDivProd_uid89_fpDivTest_cma_y[0];
            end
        end
    end
    dspba_delay_ver #( .width(49), .depth(0), .reset_kind("ASYNC") )
    qDivProd_uid89_fpDivTest_cma_delay ( .xin(qDivProd_uid89_fpDivTest_cma_s[0][48:0]), .xout(qDivProd_uid89_fpDivTest_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign qDivProd_uid89_fpDivTest_cma_q = qDivProd_uid89_fpDivTest_cma_qq[48:0];

    // qDivProdNorm_uid90_fpDivTest(BITSELECT,89)@17
    assign qDivProdNorm_uid90_fpDivTest_b = qDivProd_uid89_fpDivTest_cma_q[48:48];

    // cstBias_uid7_fpDivTest(CONSTANT,6)
    assign cstBias_uid7_fpDivTest_q = 8'b01111111;

    // qDivProdExp_opBs_uid95_fpDivTest(SUB,94)@17 + 1
    assign qDivProdExp_opBs_uid95_fpDivTest_a = {1'b0, cstBias_uid7_fpDivTest_q};
    assign qDivProdExp_opBs_uid95_fpDivTest_b = {8'b00000000, qDivProdNorm_uid90_fpDivTest_b};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            qDivProdExp_opBs_uid95_fpDivTest_o <= 9'b0;
        end
        else if (en == 1'b1)
        begin
            qDivProdExp_opBs_uid95_fpDivTest_o <= $unsigned(qDivProdExp_opBs_uid95_fpDivTest_a) - $unsigned(qDivProdExp_opBs_uid95_fpDivTest_b);
        end
    end
    assign qDivProdExp_opBs_uid95_fpDivTest_q = qDivProdExp_opBs_uid95_fpDivTest_o[8:0];

    // expPostRndFR_uid81_fpDivTest(BITSELECT,80)@14
    assign expPostRndFR_uid81_fpDivTest_in = expFracPostRnd_uid76_fpDivTest_q[32:0];
    assign expPostRndFR_uid81_fpDivTest_b = expPostRndFR_uid81_fpDivTest_in[32:25];

    // expPostRndF_uid82_fpDivTest(MUX,81)@14
    assign expPostRndF_uid82_fpDivTest_s = redist10_invYO_uid55_fpDivTest_b_5_q;
    always @(expPostRndF_uid82_fpDivTest_s or en or expPostRndFR_uid81_fpDivTest_b or redist25_expX_uid9_fpDivTest_b_14_q)
    begin
        unique case (expPostRndF_uid82_fpDivTest_s)
            1'b0 : expPostRndF_uid82_fpDivTest_q = expPostRndFR_uid81_fpDivTest_b;
            1'b1 : expPostRndF_uid82_fpDivTest_q = redist25_expX_uid9_fpDivTest_b_14_q;
            default : expPostRndF_uid82_fpDivTest_q = 8'b0;
        endcase
    end

    // qDivProdExp_opA_uid94_fpDivTest(ADD,93)@14 + 1
    assign qDivProdExp_opA_uid94_fpDivTest_a = {1'b0, redist20_expY_uid12_fpDivTest_b_14_q};
    assign qDivProdExp_opA_uid94_fpDivTest_b = {1'b0, expPostRndF_uid82_fpDivTest_q};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            qDivProdExp_opA_uid94_fpDivTest_o <= 9'b0;
        end
        else if (en == 1'b1)
        begin
            qDivProdExp_opA_uid94_fpDivTest_o <= $unsigned(qDivProdExp_opA_uid94_fpDivTest_a) + $unsigned(qDivProdExp_opA_uid94_fpDivTest_b);
        end
    end
    assign qDivProdExp_opA_uid94_fpDivTest_q = qDivProdExp_opA_uid94_fpDivTest_o[8:0];

    // redist6_qDivProdExp_opA_uid94_fpDivTest_q_4(DELAY,192)
    dspba_delay_ver #( .width(9), .depth(3), .reset_kind("ASYNC") )
    redist6_qDivProdExp_opA_uid94_fpDivTest_q_4 ( .xin(qDivProdExp_opA_uid94_fpDivTest_q), .xout(redist6_qDivProdExp_opA_uid94_fpDivTest_q_4_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // qDivProdExp_uid96_fpDivTest(SUB,95)@18
    assign qDivProdExp_uid96_fpDivTest_a = {3'b000, redist6_qDivProdExp_opA_uid94_fpDivTest_q_4_q};
    assign qDivProdExp_uid96_fpDivTest_b = {{3{qDivProdExp_opBs_uid95_fpDivTest_q[8]}}, qDivProdExp_opBs_uid95_fpDivTest_q};
    assign qDivProdExp_uid96_fpDivTest_o = $signed(qDivProdExp_uid96_fpDivTest_a) - $signed(qDivProdExp_uid96_fpDivTest_b);
    assign qDivProdExp_uid96_fpDivTest_q = qDivProdExp_uid96_fpDivTest_o[10:0];

    // qDivProdLTX_opA_uid98_fpDivTest(BITSELECT,97)@18
    assign qDivProdLTX_opA_uid98_fpDivTest_in = qDivProdExp_uid96_fpDivTest_q[7:0];
    assign qDivProdLTX_opA_uid98_fpDivTest_b = qDivProdLTX_opA_uid98_fpDivTest_in[7:0];

    // qDivProdFracHigh_uid91_fpDivTest(BITSELECT,90)@17
    assign qDivProdFracHigh_uid91_fpDivTest_in = qDivProd_uid89_fpDivTest_cma_q[47:0];
    assign qDivProdFracHigh_uid91_fpDivTest_b = qDivProdFracHigh_uid91_fpDivTest_in[47:24];

    // qDivProdFracLow_uid92_fpDivTest(BITSELECT,91)@17
    assign qDivProdFracLow_uid92_fpDivTest_in = qDivProd_uid89_fpDivTest_cma_q[46:0];
    assign qDivProdFracLow_uid92_fpDivTest_b = qDivProdFracLow_uid92_fpDivTest_in[46:23];

    // qDivProdFrac_uid93_fpDivTest(MUX,92)@17
    assign qDivProdFrac_uid93_fpDivTest_s = qDivProdNorm_uid90_fpDivTest_b;
    always @(qDivProdFrac_uid93_fpDivTest_s or en or qDivProdFracLow_uid92_fpDivTest_b or qDivProdFracHigh_uid91_fpDivTest_b)
    begin
        unique case (qDivProdFrac_uid93_fpDivTest_s)
            1'b0 : qDivProdFrac_uid93_fpDivTest_q = qDivProdFracLow_uid92_fpDivTest_b;
            1'b1 : qDivProdFrac_uid93_fpDivTest_q = qDivProdFracHigh_uid91_fpDivTest_b;
            default : qDivProdFrac_uid93_fpDivTest_q = 24'b0;
        endcase
    end

    // qDivProdFracWF_uid97_fpDivTest(BITSELECT,96)@17
    assign qDivProdFracWF_uid97_fpDivTest_b = qDivProdFrac_uid93_fpDivTest_q[23:1];

    // redist5_qDivProdFracWF_uid97_fpDivTest_b_1(DELAY,191)
    dspba_delay_ver #( .width(23), .depth(1), .reset_kind("ASYNC") )
    redist5_qDivProdFracWF_uid97_fpDivTest_b_1 ( .xin(qDivProdFracWF_uid97_fpDivTest_b), .xout(redist5_qDivProdFracWF_uid97_fpDivTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // qDivProdLTX_opA_uid99_fpDivTest(BITJOIN,98)@18
    assign qDivProdLTX_opA_uid99_fpDivTest_q = {qDivProdLTX_opA_uid98_fpDivTest_b, redist5_qDivProdFracWF_uid97_fpDivTest_b_1_q};

    // qDividerProdLTX_uid101_fpDivTest(COMPARE,100)@18 + 1
    assign qDividerProdLTX_uid101_fpDivTest_a = {2'b00, qDivProdLTX_opA_uid99_fpDivTest_q};
    assign qDividerProdLTX_uid101_fpDivTest_b = {2'b00, qDivProdLTX_opB_uid100_fpDivTest_q};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            qDividerProdLTX_uid101_fpDivTest_o <= 33'b0;
        end
        else if (en == 1'b1)
        begin
            qDividerProdLTX_uid101_fpDivTest_o <= $unsigned(qDividerProdLTX_uid101_fpDivTest_a) - $unsigned(qDividerProdLTX_uid101_fpDivTest_b);
        end
    end
    assign qDividerProdLTX_uid101_fpDivTest_c[0] = qDividerProdLTX_uid101_fpDivTest_o[32];

    // extraUlp_uid103_fpDivTest(LOGICAL,102)@19
    assign extraUlp_uid103_fpDivTest_q = qDividerProdLTX_uid101_fpDivTest_c & betweenFPwF_uid102_fpDivTest_b;

    // fracPostRndFT_uid104_fpDivTest(BITSELECT,103)@19
    assign fracPostRndFT_uid104_fpDivTest_b = redist8_fracPostRndF_uid80_fpDivTest_q_5_outputreg_q[23:1];

    // fracRPreExcExt_uid105_fpDivTest(ADD,104)@19
    assign fracRPreExcExt_uid105_fpDivTest_a = {1'b0, fracPostRndFT_uid104_fpDivTest_b};
    assign fracRPreExcExt_uid105_fpDivTest_b = {23'b00000000000000000000000, extraUlp_uid103_fpDivTest_q};
    assign fracRPreExcExt_uid105_fpDivTest_o = $unsigned(fracRPreExcExt_uid105_fpDivTest_a) + $unsigned(fracRPreExcExt_uid105_fpDivTest_b);
    assign fracRPreExcExt_uid105_fpDivTest_q = fracRPreExcExt_uid105_fpDivTest_o[23:0];

    // ovfIncRnd_uid109_fpDivTest(BITSELECT,108)@19
    assign ovfIncRnd_uid109_fpDivTest_b = fracRPreExcExt_uid105_fpDivTest_q[23:23];

    // redist3_ovfIncRnd_uid109_fpDivTest_b_1(DELAY,189)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist3_ovfIncRnd_uid109_fpDivTest_b_1 ( .xin(ovfIncRnd_uid109_fpDivTest_b), .xout(redist3_ovfIncRnd_uid109_fpDivTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expFracPostRndInc_uid110_fpDivTest(ADD,109)@20
    assign expFracPostRndInc_uid110_fpDivTest_a = {1'b0, redist7_expPostRndFR_uid81_fpDivTest_b_6_outputreg_q};
    assign expFracPostRndInc_uid110_fpDivTest_b = {8'b00000000, redist3_ovfIncRnd_uid109_fpDivTest_b_1_q};
    assign expFracPostRndInc_uid110_fpDivTest_o = $unsigned(expFracPostRndInc_uid110_fpDivTest_a) + $unsigned(expFracPostRndInc_uid110_fpDivTest_b);
    assign expFracPostRndInc_uid110_fpDivTest_q = expFracPostRndInc_uid110_fpDivTest_o[8:0];

    // expFracPostRndR_uid111_fpDivTest(BITSELECT,110)@20
    assign expFracPostRndR_uid111_fpDivTest_in = expFracPostRndInc_uid110_fpDivTest_q[7:0];
    assign expFracPostRndR_uid111_fpDivTest_b = expFracPostRndR_uid111_fpDivTest_in[7:0];

    // redist7_expPostRndFR_uid81_fpDivTest_b_6_notEnable(LOGICAL,221)
    assign redist7_expPostRndFR_uid81_fpDivTest_b_6_notEnable_q = ~ (en);

    // redist7_expPostRndFR_uid81_fpDivTest_b_6_nor(LOGICAL,222)
    assign redist7_expPostRndFR_uid81_fpDivTest_b_6_nor_q = ~ (redist7_expPostRndFR_uid81_fpDivTest_b_6_notEnable_q | redist7_expPostRndFR_uid81_fpDivTest_b_6_sticky_ena_q);

    // redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_last(CONSTANT,218)
    assign redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_last_q = 3'b010;

    // redist7_expPostRndFR_uid81_fpDivTest_b_6_cmp(LOGICAL,219)
    assign redist7_expPostRndFR_uid81_fpDivTest_b_6_cmp_b = {1'b0, redist7_expPostRndFR_uid81_fpDivTest_b_6_rdmux_q};
    assign redist7_expPostRndFR_uid81_fpDivTest_b_6_cmp_q = redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_last_q == redist7_expPostRndFR_uid81_fpDivTest_b_6_cmp_b ? 1'b1 : 1'b0;

    // redist7_expPostRndFR_uid81_fpDivTest_b_6_cmpReg(REG,220)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist7_expPostRndFR_uid81_fpDivTest_b_6_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist7_expPostRndFR_uid81_fpDivTest_b_6_cmpReg_q <= redist7_expPostRndFR_uid81_fpDivTest_b_6_cmp_q;
        end
    end

    // redist7_expPostRndFR_uid81_fpDivTest_b_6_sticky_ena(REG,223)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist7_expPostRndFR_uid81_fpDivTest_b_6_sticky_ena_q <= 1'b0;
        end
        else if (redist7_expPostRndFR_uid81_fpDivTest_b_6_nor_q == 1'b1)
        begin
            redist7_expPostRndFR_uid81_fpDivTest_b_6_sticky_ena_q <= redist7_expPostRndFR_uid81_fpDivTest_b_6_cmpReg_q;
        end
    end

    // redist7_expPostRndFR_uid81_fpDivTest_b_6_enaAnd(LOGICAL,224)
    assign redist7_expPostRndFR_uid81_fpDivTest_b_6_enaAnd_q = redist7_expPostRndFR_uid81_fpDivTest_b_6_sticky_ena_q & en;

    // redist7_expPostRndFR_uid81_fpDivTest_b_6_rdcnt(COUNTER,215)
    // low=0, high=3, step=1, init=0
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist7_expPostRndFR_uid81_fpDivTest_b_6_rdcnt_i <= 2'd0;
        end
        else if (en == 1'b1)
        begin
            redist7_expPostRndFR_uid81_fpDivTest_b_6_rdcnt_i <= $unsigned(redist7_expPostRndFR_uid81_fpDivTest_b_6_rdcnt_i) + $unsigned(2'd1);
        end
    end
    assign redist7_expPostRndFR_uid81_fpDivTest_b_6_rdcnt_q = redist7_expPostRndFR_uid81_fpDivTest_b_6_rdcnt_i[1:0];

    // redist7_expPostRndFR_uid81_fpDivTest_b_6_rdmux(MUX,216)
    assign redist7_expPostRndFR_uid81_fpDivTest_b_6_rdmux_s = en;
    always @(redist7_expPostRndFR_uid81_fpDivTest_b_6_rdmux_s or redist7_expPostRndFR_uid81_fpDivTest_b_6_wraddr_q or redist7_expPostRndFR_uid81_fpDivTest_b_6_rdcnt_q)
    begin
        unique case (redist7_expPostRndFR_uid81_fpDivTest_b_6_rdmux_s)
            1'b0 : redist7_expPostRndFR_uid81_fpDivTest_b_6_rdmux_q = redist7_expPostRndFR_uid81_fpDivTest_b_6_wraddr_q;
            1'b1 : redist7_expPostRndFR_uid81_fpDivTest_b_6_rdmux_q = redist7_expPostRndFR_uid81_fpDivTest_b_6_rdcnt_q;
            default : redist7_expPostRndFR_uid81_fpDivTest_b_6_rdmux_q = 2'b0;
        endcase
    end

    // redist7_expPostRndFR_uid81_fpDivTest_b_6_wraddr(REG,217)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist7_expPostRndFR_uid81_fpDivTest_b_6_wraddr_q <= 2'b11;
        end
        else
        begin
            redist7_expPostRndFR_uid81_fpDivTest_b_6_wraddr_q <= redist7_expPostRndFR_uid81_fpDivTest_b_6_rdmux_q;
        end
    end

    // redist7_expPostRndFR_uid81_fpDivTest_b_6_mem(DUALMEM,214)
    assign redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_ia = expPostRndFR_uid81_fpDivTest_b;
    assign redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_aa = redist7_expPostRndFR_uid81_fpDivTest_b_6_wraddr_q;
    assign redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_ab = redist7_expPostRndFR_uid81_fpDivTest_b_6_rdmux_q;
    assign redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(8),
        .widthad_a(2),
        .numwords_a(4),
        .width_b(8),
        .widthad_b(2),
        .numwords_b(4),
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
    ) redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_dmem (
        .clocken1(redist7_expPostRndFR_uid81_fpDivTest_b_6_enaAnd_q[0]),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .aclr1(redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_reset0),
        .clock1(clk),
        .address_a(redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_aa),
        .data_a(redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_ia),
        .wren_a(en[0]),
        .address_b(redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_ab),
        .q_b(redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_iq),
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
    assign redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_q = redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_iq[7:0];

    // redist7_expPostRndFR_uid81_fpDivTest_b_6_outputreg(DELAY,213)
    dspba_delay_ver #( .width(8), .depth(1), .reset_kind("ASYNC") )
    redist7_expPostRndFR_uid81_fpDivTest_b_6_outputreg ( .xin(redist7_expPostRndFR_uid81_fpDivTest_b_6_mem_q), .xout(redist7_expPostRndFR_uid81_fpDivTest_b_6_outputreg_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist4_extraUlp_uid103_fpDivTest_q_1(DELAY,190)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist4_extraUlp_uid103_fpDivTest_q_1 ( .xin(extraUlp_uid103_fpDivTest_q), .xout(redist4_extraUlp_uid103_fpDivTest_q_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expRPreExc_uid112_fpDivTest(MUX,111)@20
    assign expRPreExc_uid112_fpDivTest_s = redist4_extraUlp_uid103_fpDivTest_q_1_q;
    always @(expRPreExc_uid112_fpDivTest_s or en or redist7_expPostRndFR_uid81_fpDivTest_b_6_outputreg_q or expFracPostRndR_uid111_fpDivTest_b)
    begin
        unique case (expRPreExc_uid112_fpDivTest_s)
            1'b0 : expRPreExc_uid112_fpDivTest_q = redist7_expPostRndFR_uid81_fpDivTest_b_6_outputreg_q;
            1'b1 : expRPreExc_uid112_fpDivTest_q = expFracPostRndR_uid111_fpDivTest_b;
            default : expRPreExc_uid112_fpDivTest_q = 8'b0;
        endcase
    end

    // invExpXIsMax_uid43_fpDivTest(LOGICAL,42)@14
    assign invExpXIsMax_uid43_fpDivTest_q = ~ (expXIsMax_uid38_fpDivTest_q);

    // InvExpXIsZero_uid44_fpDivTest(LOGICAL,43)@14
    assign InvExpXIsZero_uid44_fpDivTest_q = ~ (excZ_y_uid37_fpDivTest_q);

    // excR_y_uid45_fpDivTest(LOGICAL,44)@14
    assign excR_y_uid45_fpDivTest_q = InvExpXIsZero_uid44_fpDivTest_q & invExpXIsMax_uid43_fpDivTest_q;

    // excXIYR_uid127_fpDivTest(LOGICAL,126)@14
    assign excXIYR_uid127_fpDivTest_q = excI_x_uid27_fpDivTest_q & excR_y_uid45_fpDivTest_q;

    // excXIYZ_uid126_fpDivTest(LOGICAL,125)@14
    assign excXIYZ_uid126_fpDivTest_q = excI_x_uid27_fpDivTest_q & excZ_y_uid37_fpDivTest_q;

    // expRExt_uid114_fpDivTest(BITSELECT,113)@14
    assign expRExt_uid114_fpDivTest_b = expFracPostRnd_uid76_fpDivTest_q[35:25];

    // expOvf_uid118_fpDivTest(COMPARE,117)@14
    assign expOvf_uid118_fpDivTest_a = {{2{expRExt_uid114_fpDivTest_b[10]}}, expRExt_uid114_fpDivTest_b};
    assign expOvf_uid118_fpDivTest_b = {5'b00000, cstAllOWE_uid18_fpDivTest_q};
    assign expOvf_uid118_fpDivTest_o = $signed(expOvf_uid118_fpDivTest_a) - $signed(expOvf_uid118_fpDivTest_b);
    assign expOvf_uid118_fpDivTest_n[0] = ~ (expOvf_uid118_fpDivTest_o[12]);

    // invExpXIsMax_uid29_fpDivTest(LOGICAL,28)@14
    assign invExpXIsMax_uid29_fpDivTest_q = ~ (expXIsMax_uid24_fpDivTest_q);

    // InvExpXIsZero_uid30_fpDivTest(LOGICAL,29)@14
    assign InvExpXIsZero_uid30_fpDivTest_q = ~ (excZ_x_uid23_fpDivTest_q);

    // excR_x_uid31_fpDivTest(LOGICAL,30)@14
    assign excR_x_uid31_fpDivTest_q = InvExpXIsZero_uid30_fpDivTest_q & invExpXIsMax_uid29_fpDivTest_q;

    // excXRYROvf_uid125_fpDivTest(LOGICAL,124)@14
    assign excXRYROvf_uid125_fpDivTest_q = excR_x_uid31_fpDivTest_q & excR_y_uid45_fpDivTest_q & expOvf_uid118_fpDivTest_n;

    // excXRYZ_uid124_fpDivTest(LOGICAL,123)@14
    assign excXRYZ_uid124_fpDivTest_q = excR_x_uid31_fpDivTest_q & excZ_y_uid37_fpDivTest_q;

    // excRInf_uid128_fpDivTest(LOGICAL,127)@14 + 1
    assign excRInf_uid128_fpDivTest_qi = excXRYZ_uid124_fpDivTest_q | excXRYROvf_uid125_fpDivTest_q | excXIYZ_uid126_fpDivTest_q | excXIYR_uid127_fpDivTest_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    excRInf_uid128_fpDivTest_delay ( .xin(excRInf_uid128_fpDivTest_qi), .xout(excRInf_uid128_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // xRegOrZero_uid121_fpDivTest(LOGICAL,120)@14
    assign xRegOrZero_uid121_fpDivTest_q = excR_x_uid31_fpDivTest_q | excZ_x_uid23_fpDivTest_q;

    // regOrZeroOverInf_uid122_fpDivTest(LOGICAL,121)@14 + 1
    assign regOrZeroOverInf_uid122_fpDivTest_qi = xRegOrZero_uid121_fpDivTest_q & excI_y_uid41_fpDivTest_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    regOrZeroOverInf_uid122_fpDivTest_delay ( .xin(regOrZeroOverInf_uid122_fpDivTest_qi), .xout(regOrZeroOverInf_uid122_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expUdf_uid115_fpDivTest(COMPARE,114)@14
    assign expUdf_uid115_fpDivTest_a = {12'b000000000000, GND_q};
    assign expUdf_uid115_fpDivTest_b = {{2{expRExt_uid114_fpDivTest_b[10]}}, expRExt_uid114_fpDivTest_b};
    assign expUdf_uid115_fpDivTest_o = $signed(expUdf_uid115_fpDivTest_a) - $signed(expUdf_uid115_fpDivTest_b);
    assign expUdf_uid115_fpDivTest_n[0] = ~ (expUdf_uid115_fpDivTest_o[12]);

    // regOverRegWithUf_uid120_fpDivTest(LOGICAL,119)@14 + 1
    assign regOverRegWithUf_uid120_fpDivTest_qi = expUdf_uid115_fpDivTest_n & excR_x_uid31_fpDivTest_q & excR_y_uid45_fpDivTest_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    regOverRegWithUf_uid120_fpDivTest_delay ( .xin(regOverRegWithUf_uid120_fpDivTest_qi), .xout(regOverRegWithUf_uid120_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // zeroOverReg_uid119_fpDivTest(LOGICAL,118)@14 + 1
    assign zeroOverReg_uid119_fpDivTest_qi = excZ_x_uid23_fpDivTest_q & excR_y_uid45_fpDivTest_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    zeroOverReg_uid119_fpDivTest_delay ( .xin(zeroOverReg_uid119_fpDivTest_qi), .xout(zeroOverReg_uid119_fpDivTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // excRZero_uid123_fpDivTest(LOGICAL,122)@15
    assign excRZero_uid123_fpDivTest_q = zeroOverReg_uid119_fpDivTest_q | regOverRegWithUf_uid120_fpDivTest_q | regOrZeroOverInf_uid122_fpDivTest_q;

    // concExc_uid132_fpDivTest(BITJOIN,131)@15
    assign concExc_uid132_fpDivTest_q = {excRNaN_uid131_fpDivTest_q, excRInf_uid128_fpDivTest_q, excRZero_uid123_fpDivTest_q};

    // excREnc_uid133_fpDivTest(LOOKUP,132)@15 + 1
    always @ (posedge clk or posedge areset)
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

    // redist2_excREnc_uid133_fpDivTest_q_5(DELAY,188)
    dspba_delay_ver #( .width(2), .depth(4), .reset_kind("ASYNC") )
    redist2_excREnc_uid133_fpDivTest_q_5 ( .xin(excREnc_uid133_fpDivTest_q), .xout(redist2_excREnc_uid133_fpDivTest_q_5_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expRPostExc_uid141_fpDivTest(MUX,140)@20
    assign expRPostExc_uid141_fpDivTest_s = redist2_excREnc_uid133_fpDivTest_q_5_q;
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

    // fracPostRndFPostUlp_uid106_fpDivTest(BITSELECT,105)@19
    assign fracPostRndFPostUlp_uid106_fpDivTest_in = fracRPreExcExt_uid105_fpDivTest_q[22:0];
    assign fracPostRndFPostUlp_uid106_fpDivTest_b = fracPostRndFPostUlp_uid106_fpDivTest_in[22:0];

    // fracRPreExc_uid107_fpDivTest(MUX,106)@19 + 1
    assign fracRPreExc_uid107_fpDivTest_s = extraUlp_uid103_fpDivTest_q;
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            fracRPreExc_uid107_fpDivTest_q <= 23'b0;
        end
        else if (en == 1'b1)
        begin
            unique case (fracRPreExc_uid107_fpDivTest_s)
                1'b0 : fracRPreExc_uid107_fpDivTest_q <= fracPostRndFT_uid104_fpDivTest_b;
                1'b1 : fracRPreExc_uid107_fpDivTest_q <= fracPostRndFPostUlp_uid106_fpDivTest_b;
                default : fracRPreExc_uid107_fpDivTest_q <= 23'b0;
            endcase
        end
    end

    // fracRPostExc_uid137_fpDivTest(MUX,136)@20
    assign fracRPostExc_uid137_fpDivTest_s = redist2_excREnc_uid133_fpDivTest_q_5_q;
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

    // divR_uid144_fpDivTest(BITJOIN,143)@20
    assign divR_uid144_fpDivTest_q = {redist1_sRPostExc_uid143_fpDivTest_q_5_q, expRPostExc_uid141_fpDivTest_q, fracRPostExc_uid137_fpDivTest_q};

    // xOut(GPOUT,4)@20
    assign q = divR_uid144_fpDivTest_q;

endmodule
