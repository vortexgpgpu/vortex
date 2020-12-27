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

// SystemVerilog created from acl_fsqrt
// SystemVerilog created on Sun Dec 27 09:47:21 2020


(* altera_attribute = "-name AUTO_SHIFT_REGISTER_RECOGNITION OFF; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 10037; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 15400; -name MESSAGE_DISABLE 14130; -name MESSAGE_DISABLE 10036; -name MESSAGE_DISABLE 12020; -name MESSAGE_DISABLE 12030; -name MESSAGE_DISABLE 12010; -name MESSAGE_DISABLE 12110; -name MESSAGE_DISABLE 14320; -name MESSAGE_DISABLE 13410; -name MESSAGE_DISABLE 113007; -name MESSAGE_DISABLE 10958" *)
module acl_fsqrt (
    input wire [31:0] a,
    input wire [0:0] en,
    output wire [31:0] q,
    input wire clk,
    input wire areset
    );

    wire [0:0] GND_q;
    wire [0:0] VCC_q;
    wire [7:0] expX_uid6_fpSqrtTest_b;
    wire [0:0] signX_uid7_fpSqrtTest_b;
    wire [7:0] cstAllOWE_uid8_fpSqrtTest_q;
    wire [22:0] cstZeroWF_uid9_fpSqrtTest_q;
    wire [7:0] cstAllZWE_uid10_fpSqrtTest_q;
    wire [22:0] frac_x_uid12_fpSqrtTest_b;
    wire [0:0] excZ_x_uid13_fpSqrtTest_qi;
    reg [0:0] excZ_x_uid13_fpSqrtTest_q;
    wire [0:0] expXIsMax_uid14_fpSqrtTest_qi;
    reg [0:0] expXIsMax_uid14_fpSqrtTest_q;
    wire [0:0] fracXIsZero_uid15_fpSqrtTest_qi;
    reg [0:0] fracXIsZero_uid15_fpSqrtTest_q;
    wire [0:0] fracXIsNotZero_uid16_fpSqrtTest_q;
    wire [0:0] excI_x_uid17_fpSqrtTest_q;
    wire [0:0] excN_x_uid18_fpSqrtTest_q;
    wire [0:0] invExpXIsMax_uid19_fpSqrtTest_q;
    wire [0:0] InvExpXIsZero_uid20_fpSqrtTest_q;
    wire [0:0] excR_x_uid21_fpSqrtTest_q;
    wire [7:0] sBias_uid22_fpSqrtTest_q;
    wire [8:0] expEvenSig_uid24_fpSqrtTest_a;
    wire [8:0] expEvenSig_uid24_fpSqrtTest_b;
    logic [8:0] expEvenSig_uid24_fpSqrtTest_o;
    wire [8:0] expEvenSig_uid24_fpSqrtTest_q;
    wire [7:0] expREven_uid25_fpSqrtTest_b;
    wire [7:0] sBiasM1_uid26_fpSqrtTest_q;
    wire [8:0] expOddSig_uid27_fpSqrtTest_a;
    wire [8:0] expOddSig_uid27_fpSqrtTest_b;
    logic [8:0] expOddSig_uid27_fpSqrtTest_o;
    wire [8:0] expOddSig_uid27_fpSqrtTest_q;
    wire [7:0] expROdd_uid28_fpSqrtTest_b;
    wire [0:0] expX0PS_uid29_fpSqrtTest_in;
    wire [0:0] expX0PS_uid29_fpSqrtTest_b;
    wire [0:0] expOddSelect_uid30_fpSqrtTest_q;
    wire [0:0] expRMux_uid31_fpSqrtTest_s;
    reg [7:0] expRMux_uid31_fpSqrtTest_q;
    wire [23:0] addrFull_uid33_fpSqrtTest_q;
    wire [7:0] yAddr_uid35_fpSqrtTest_b;
    wire [15:0] yForPe_uid36_fpSqrtTest_in;
    wire [15:0] yForPe_uid36_fpSqrtTest_b;
    wire [30:0] expIncPEOnly_uid38_fpSqrtTest_in;
    wire [0:0] expIncPEOnly_uid38_fpSqrtTest_b;
    wire [28:0] fracRPreCR_uid39_fpSqrtTest_in;
    wire [23:0] fracRPreCR_uid39_fpSqrtTest_b;
    wire [24:0] fracPaddingOne_uid41_fpSqrtTest_q;
    wire [23:0] oFracX_uid44_fpSqrtTest_q;
    wire [24:0] oFracXZ_mergedSignalTM_uid47_fpSqrtTest_q;
    wire [24:0] oFracXSignExt_mergedSignalTM_uid52_fpSqrtTest_q;
    wire [0:0] normalizedXForComp_uid54_fpSqrtTest_s;
    reg [24:0] normalizedXForComp_uid54_fpSqrtTest_q;
    wire [24:0] paddingY_uid55_fpSqrtTest_q;
    wire [49:0] updatedY_uid56_fpSqrtTest_q;
    wire [51:0] squaredResultGTEIn_uid55_fpSqrtTest_a;
    wire [51:0] squaredResultGTEIn_uid55_fpSqrtTest_b;
    logic [51:0] squaredResultGTEIn_uid55_fpSqrtTest_o;
    wire [0:0] squaredResultGTEIn_uid55_fpSqrtTest_n;
    wire [0:0] pLTOne_uid58_fpSqrtTest_q;
    wire [24:0] fxpSqrtResPostUpdateE_uid60_fpSqrtTest_a;
    wire [24:0] fxpSqrtResPostUpdateE_uid60_fpSqrtTest_b;
    logic [24:0] fxpSqrtResPostUpdateE_uid60_fpSqrtTest_o;
    wire [24:0] fxpSqrtResPostUpdateE_uid60_fpSqrtTest_q;
    wire [0:0] fracPENotOne_uid62_fpSqrtTest_q;
    wire [0:0] fracPENotOneAndCRRoundsExp_uid63_fpSqrtTest_q;
    wire [0:0] expInc_uid64_fpSqrtTest_qi;
    reg [0:0] expInc_uid64_fpSqrtTest_q;
    wire [8:0] expR_uid66_fpSqrtTest_a;
    wire [8:0] expR_uid66_fpSqrtTest_b;
    logic [8:0] expR_uid66_fpSqrtTest_o;
    wire [8:0] expR_uid66_fpSqrtTest_q;
    wire [0:0] invSignX_uid67_fpSqrtTest_q;
    wire [0:0] inInfAndNotNeg_uid68_fpSqrtTest_q;
    wire [0:0] minReg_uid69_fpSqrtTest_q;
    wire [0:0] minInf_uid70_fpSqrtTest_q;
    wire [0:0] excRNaN_uid71_fpSqrtTest_q;
    wire [2:0] excConc_uid72_fpSqrtTest_q;
    wire [3:0] fracSelIn_uid73_fpSqrtTest_q;
    reg [1:0] fracSel_uid74_fpSqrtTest_q;
    wire [7:0] expRR_uid77_fpSqrtTest_in;
    wire [7:0] expRR_uid77_fpSqrtTest_b;
    wire [1:0] expRPostExc_uid79_fpSqrtTest_s;
    reg [7:0] expRPostExc_uid79_fpSqrtTest_q;
    wire [22:0] fracNaN_uid80_fpSqrtTest_q;
    wire [1:0] fracRPostExc_uid84_fpSqrtTest_s;
    reg [22:0] fracRPostExc_uid84_fpSqrtTest_q;
    wire [0:0] negZero_uid85_fpSqrtTest_qi;
    reg [0:0] negZero_uid85_fpSqrtTest_q;
    wire [31:0] RSqrt_uid86_fpSqrtTest_q;
    wire [11:0] yT1_uid100_invPolyEval_b;
    wire [0:0] lowRangeB_uid102_invPolyEval_in;
    wire [0:0] lowRangeB_uid102_invPolyEval_b;
    wire [11:0] highBBits_uid103_invPolyEval_b;
    wire [21:0] s1sumAHighB_uid104_invPolyEval_a;
    wire [21:0] s1sumAHighB_uid104_invPolyEval_b;
    logic [21:0] s1sumAHighB_uid104_invPolyEval_o;
    wire [21:0] s1sumAHighB_uid104_invPolyEval_q;
    wire [22:0] s1_uid105_invPolyEval_q;
    wire [1:0] lowRangeB_uid108_invPolyEval_in;
    wire [1:0] lowRangeB_uid108_invPolyEval_b;
    wire [21:0] highBBits_uid109_invPolyEval_b;
    wire [29:0] s2sumAHighB_uid110_invPolyEval_a;
    wire [29:0] s2sumAHighB_uid110_invPolyEval_b;
    logic [29:0] s2sumAHighB_uid110_invPolyEval_o;
    wire [29:0] s2sumAHighB_uid110_invPolyEval_q;
    wire [31:0] s2_uid111_invPolyEval_q;
    wire [12:0] osig_uid114_pT1_uid101_invPolyEval_b;
    wire [23:0] osig_uid117_pT2_uid107_invPolyEval_b;
    wire memoryC0_uid88_sqrtTables_lutmem_reset0;
    wire [28:0] memoryC0_uid88_sqrtTables_lutmem_ia;
    wire [7:0] memoryC0_uid88_sqrtTables_lutmem_aa;
    wire [7:0] memoryC0_uid88_sqrtTables_lutmem_ab;
    wire [28:0] memoryC0_uid88_sqrtTables_lutmem_ir;
    wire [28:0] memoryC0_uid88_sqrtTables_lutmem_r;
    wire memoryC1_uid91_sqrtTables_lutmem_reset0;
    wire [20:0] memoryC1_uid91_sqrtTables_lutmem_ia;
    wire [7:0] memoryC1_uid91_sqrtTables_lutmem_aa;
    wire [7:0] memoryC1_uid91_sqrtTables_lutmem_ab;
    wire [20:0] memoryC1_uid91_sqrtTables_lutmem_ir;
    wire [20:0] memoryC1_uid91_sqrtTables_lutmem_r;
    wire memoryC2_uid94_sqrtTables_lutmem_reset0;
    wire [11:0] memoryC2_uid94_sqrtTables_lutmem_ia;
    wire [7:0] memoryC2_uid94_sqrtTables_lutmem_aa;
    wire [7:0] memoryC2_uid94_sqrtTables_lutmem_ab;
    wire [11:0] memoryC2_uid94_sqrtTables_lutmem_ir;
    wire [11:0] memoryC2_uid94_sqrtTables_lutmem_r;
    wire squaredResult_uid42_fpSqrtTest_cma_reset;
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [24:0] squaredResult_uid42_fpSqrtTest_cma_a0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [24:0] squaredResult_uid42_fpSqrtTest_cma_a1 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [24:0] squaredResult_uid42_fpSqrtTest_cma_c0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [24:0] squaredResult_uid42_fpSqrtTest_cma_c1 [0:0];
    wire [49:0] squaredResult_uid42_fpSqrtTest_cma_p [0:0];
    wire [49:0] squaredResult_uid42_fpSqrtTest_cma_u [0:0];
    wire [49:0] squaredResult_uid42_fpSqrtTest_cma_w [0:0];
    wire [49:0] squaredResult_uid42_fpSqrtTest_cma_x [0:0];
    wire [49:0] squaredResult_uid42_fpSqrtTest_cma_y [0:0];
    reg [49:0] squaredResult_uid42_fpSqrtTest_cma_s [0:0];
    wire [49:0] squaredResult_uid42_fpSqrtTest_cma_qq;
    wire [49:0] squaredResult_uid42_fpSqrtTest_cma_q;
    wire squaredResult_uid42_fpSqrtTest_cma_ena0;
    wire squaredResult_uid42_fpSqrtTest_cma_ena1;
    wire squaredResult_uid42_fpSqrtTest_cma_ena2;
    wire prodXY_uid113_pT1_uid101_invPolyEval_cma_reset;
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [11:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_a0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [11:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_a1 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg signed [11:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_c0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg signed [11:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_c1 [0:0];
    wire signed [12:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_l [0:0];
    wire signed [24:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_p [0:0];
    wire signed [24:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_u [0:0];
    wire signed [24:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_w [0:0];
    wire signed [24:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_x [0:0];
    wire signed [24:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_y [0:0];
    reg signed [24:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_s [0:0];
    wire [23:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_qq;
    wire [23:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_q;
    wire prodXY_uid113_pT1_uid101_invPolyEval_cma_ena0;
    wire prodXY_uid113_pT1_uid101_invPolyEval_cma_ena1;
    wire prodXY_uid113_pT1_uid101_invPolyEval_cma_ena2;
    wire prodXY_uid116_pT2_uid107_invPolyEval_cma_reset;
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [15:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_a0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [15:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_a1 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg signed [22:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_c0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg signed [22:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_c1 [0:0];
    wire signed [16:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_l [0:0];
    wire signed [39:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_p [0:0];
    wire signed [39:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_u [0:0];
    wire signed [39:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_w [0:0];
    wire signed [39:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_x [0:0];
    wire signed [39:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_y [0:0];
    reg signed [39:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_s [0:0];
    wire [38:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_qq;
    wire [38:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_q;
    wire prodXY_uid116_pT2_uid107_invPolyEval_cma_ena0;
    wire prodXY_uid116_pT2_uid107_invPolyEval_cma_ena1;
    wire prodXY_uid116_pT2_uid107_invPolyEval_cma_ena2;
    wire [0:0] expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_b;
    wire [22:0] expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_c;
    reg [22:0] redist0_expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_c_1_q;
    reg [0:0] redist1_lowRangeB_uid102_invPolyEval_b_1_q;
    reg [23:0] redist2_fracRPreCR_uid39_fpSqrtTest_b_1_q;
    reg [23:0] redist3_fracRPreCR_uid39_fpSqrtTest_b_5_q;
    reg [0:0] redist4_expIncPEOnly_uid38_fpSqrtTest_b_5_q;
    reg [7:0] redist6_yAddr_uid35_fpSqrtTest_b_3_q;
    reg [7:0] redist7_yAddr_uid35_fpSqrtTest_b_7_q;
    reg [7:0] redist8_expRMux_uid31_fpSqrtTest_q_2_q;
    reg [0:0] redist9_expOddSelect_uid30_fpSqrtTest_q_13_q;
    reg [22:0] redist10_frac_x_uid12_fpSqrtTest_b_2_q;
    reg [0:0] redist12_signX_uid7_fpSqrtTest_b_14_q;
    reg [23:0] redist3_fracRPreCR_uid39_fpSqrtTest_b_5_inputreg_q;
    wire redist5_yForPe_uid36_fpSqrtTest_b_4_mem_reset0;
    wire [15:0] redist5_yForPe_uid36_fpSqrtTest_b_4_mem_ia;
    wire [1:0] redist5_yForPe_uid36_fpSqrtTest_b_4_mem_aa;
    wire [1:0] redist5_yForPe_uid36_fpSqrtTest_b_4_mem_ab;
    wire [15:0] redist5_yForPe_uid36_fpSqrtTest_b_4_mem_iq;
    wire [15:0] redist5_yForPe_uid36_fpSqrtTest_b_4_mem_q;
    wire [1:0] redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt_q;
    (* preserve *) reg [1:0] redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt_i;
    (* preserve *) reg redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt_eq;
    wire [0:0] redist5_yForPe_uid36_fpSqrtTest_b_4_rdmux_s;
    reg [1:0] redist5_yForPe_uid36_fpSqrtTest_b_4_rdmux_q;
    reg [1:0] redist5_yForPe_uid36_fpSqrtTest_b_4_wraddr_q;
    wire [1:0] redist5_yForPe_uid36_fpSqrtTest_b_4_mem_last_q;
    wire [0:0] redist5_yForPe_uid36_fpSqrtTest_b_4_cmp_q;
    reg [0:0] redist5_yForPe_uid36_fpSqrtTest_b_4_cmpReg_q;
    wire [0:0] redist5_yForPe_uid36_fpSqrtTest_b_4_notEnable_q;
    wire [0:0] redist5_yForPe_uid36_fpSqrtTest_b_4_nor_q;
    (* preserve_syn_only *) reg [0:0] redist5_yForPe_uid36_fpSqrtTest_b_4_sticky_ena_q;
    wire [0:0] redist5_yForPe_uid36_fpSqrtTest_b_4_enaAnd_q;
    reg [22:0] redist11_frac_x_uid12_fpSqrtTest_b_13_outputreg_q;
    wire redist11_frac_x_uid12_fpSqrtTest_b_13_mem_reset0;
    wire [22:0] redist11_frac_x_uid12_fpSqrtTest_b_13_mem_ia;
    wire [3:0] redist11_frac_x_uid12_fpSqrtTest_b_13_mem_aa;
    wire [3:0] redist11_frac_x_uid12_fpSqrtTest_b_13_mem_ab;
    wire [22:0] redist11_frac_x_uid12_fpSqrtTest_b_13_mem_iq;
    wire [22:0] redist11_frac_x_uid12_fpSqrtTest_b_13_mem_q;
    wire [3:0] redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt_q;
    (* preserve *) reg [3:0] redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt_i;
    (* preserve *) reg redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt_eq;
    wire [0:0] redist11_frac_x_uid12_fpSqrtTest_b_13_rdmux_s;
    reg [3:0] redist11_frac_x_uid12_fpSqrtTest_b_13_rdmux_q;
    reg [3:0] redist11_frac_x_uid12_fpSqrtTest_b_13_wraddr_q;
    wire [3:0] redist11_frac_x_uid12_fpSqrtTest_b_13_mem_last_q;
    wire [0:0] redist11_frac_x_uid12_fpSqrtTest_b_13_cmp_q;
    reg [0:0] redist11_frac_x_uid12_fpSqrtTest_b_13_cmpReg_q;
    wire [0:0] redist11_frac_x_uid12_fpSqrtTest_b_13_notEnable_q;
    wire [0:0] redist11_frac_x_uid12_fpSqrtTest_b_13_nor_q;
    (* preserve_syn_only *) reg [0:0] redist11_frac_x_uid12_fpSqrtTest_b_13_sticky_ena_q;
    wire [0:0] redist11_frac_x_uid12_fpSqrtTest_b_13_enaAnd_q;
    reg [7:0] redist13_expX_uid6_fpSqrtTest_b_13_outputreg_q;
    wire redist13_expX_uid6_fpSqrtTest_b_13_mem_reset0;
    wire [7:0] redist13_expX_uid6_fpSqrtTest_b_13_mem_ia;
    wire [3:0] redist13_expX_uid6_fpSqrtTest_b_13_mem_aa;
    wire [3:0] redist13_expX_uid6_fpSqrtTest_b_13_mem_ab;
    wire [7:0] redist13_expX_uid6_fpSqrtTest_b_13_mem_iq;
    wire [7:0] redist13_expX_uid6_fpSqrtTest_b_13_mem_q;
    wire [3:0] redist13_expX_uid6_fpSqrtTest_b_13_rdcnt_q;
    (* preserve *) reg [3:0] redist13_expX_uid6_fpSqrtTest_b_13_rdcnt_i;
    (* preserve *) reg redist13_expX_uid6_fpSqrtTest_b_13_rdcnt_eq;
    wire [0:0] redist13_expX_uid6_fpSqrtTest_b_13_rdmux_s;
    reg [3:0] redist13_expX_uid6_fpSqrtTest_b_13_rdmux_q;
    reg [3:0] redist13_expX_uid6_fpSqrtTest_b_13_wraddr_q;
    wire [4:0] redist13_expX_uid6_fpSqrtTest_b_13_mem_last_q;
    wire [4:0] redist13_expX_uid6_fpSqrtTest_b_13_cmp_b;
    wire [0:0] redist13_expX_uid6_fpSqrtTest_b_13_cmp_q;
    reg [0:0] redist13_expX_uid6_fpSqrtTest_b_13_cmpReg_q;
    wire [0:0] redist13_expX_uid6_fpSqrtTest_b_13_notEnable_q;
    wire [0:0] redist13_expX_uid6_fpSqrtTest_b_13_nor_q;
    (* preserve_syn_only *) reg [0:0] redist13_expX_uid6_fpSqrtTest_b_13_sticky_ena_q;
    wire [0:0] redist13_expX_uid6_fpSqrtTest_b_13_enaAnd_q;


    // signX_uid7_fpSqrtTest(BITSELECT,6)@0
    assign signX_uid7_fpSqrtTest_b = a[31:31];

    // redist12_signX_uid7_fpSqrtTest_b_14(DELAY,137)
    dspba_delay_ver #( .width(1), .depth(14), .reset_kind("ASYNC") )
    redist12_signX_uid7_fpSqrtTest_b_14 ( .xin(signX_uid7_fpSqrtTest_b), .xout(redist12_signX_uid7_fpSqrtTest_b_14_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // cstAllZWE_uid10_fpSqrtTest(CONSTANT,9)
    assign cstAllZWE_uid10_fpSqrtTest_q = 8'b00000000;

    // redist13_expX_uid6_fpSqrtTest_b_13_notEnable(LOGICAL,171)
    assign redist13_expX_uid6_fpSqrtTest_b_13_notEnable_q = ~ (en);

    // redist13_expX_uid6_fpSqrtTest_b_13_nor(LOGICAL,172)
    assign redist13_expX_uid6_fpSqrtTest_b_13_nor_q = ~ (redist13_expX_uid6_fpSqrtTest_b_13_notEnable_q | redist13_expX_uid6_fpSqrtTest_b_13_sticky_ena_q);

    // redist13_expX_uid6_fpSqrtTest_b_13_mem_last(CONSTANT,168)
    assign redist13_expX_uid6_fpSqrtTest_b_13_mem_last_q = 5'b01001;

    // redist13_expX_uid6_fpSqrtTest_b_13_cmp(LOGICAL,169)
    assign redist13_expX_uid6_fpSqrtTest_b_13_cmp_b = {1'b0, redist13_expX_uid6_fpSqrtTest_b_13_rdmux_q};
    assign redist13_expX_uid6_fpSqrtTest_b_13_cmp_q = redist13_expX_uid6_fpSqrtTest_b_13_mem_last_q == redist13_expX_uid6_fpSqrtTest_b_13_cmp_b ? 1'b1 : 1'b0;

    // redist13_expX_uid6_fpSqrtTest_b_13_cmpReg(REG,170)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist13_expX_uid6_fpSqrtTest_b_13_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist13_expX_uid6_fpSqrtTest_b_13_cmpReg_q <= redist13_expX_uid6_fpSqrtTest_b_13_cmp_q;
        end
    end

    // redist13_expX_uid6_fpSqrtTest_b_13_sticky_ena(REG,173)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist13_expX_uid6_fpSqrtTest_b_13_sticky_ena_q <= 1'b0;
        end
        else if (redist13_expX_uid6_fpSqrtTest_b_13_nor_q == 1'b1)
        begin
            redist13_expX_uid6_fpSqrtTest_b_13_sticky_ena_q <= redist13_expX_uid6_fpSqrtTest_b_13_cmpReg_q;
        end
    end

    // redist13_expX_uid6_fpSqrtTest_b_13_enaAnd(LOGICAL,174)
    assign redist13_expX_uid6_fpSqrtTest_b_13_enaAnd_q = redist13_expX_uid6_fpSqrtTest_b_13_sticky_ena_q & en;

    // redist13_expX_uid6_fpSqrtTest_b_13_rdcnt(COUNTER,165)
    // low=0, high=10, step=1, init=0
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist13_expX_uid6_fpSqrtTest_b_13_rdcnt_i <= 4'd0;
            redist13_expX_uid6_fpSqrtTest_b_13_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist13_expX_uid6_fpSqrtTest_b_13_rdcnt_i == 4'd9)
            begin
                redist13_expX_uid6_fpSqrtTest_b_13_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist13_expX_uid6_fpSqrtTest_b_13_rdcnt_eq <= 1'b0;
            end
            if (redist13_expX_uid6_fpSqrtTest_b_13_rdcnt_eq == 1'b1)
            begin
                redist13_expX_uid6_fpSqrtTest_b_13_rdcnt_i <= $unsigned(redist13_expX_uid6_fpSqrtTest_b_13_rdcnt_i) + $unsigned(4'd6);
            end
            else
            begin
                redist13_expX_uid6_fpSqrtTest_b_13_rdcnt_i <= $unsigned(redist13_expX_uid6_fpSqrtTest_b_13_rdcnt_i) + $unsigned(4'd1);
            end
        end
    end
    assign redist13_expX_uid6_fpSqrtTest_b_13_rdcnt_q = redist13_expX_uid6_fpSqrtTest_b_13_rdcnt_i[3:0];

    // redist13_expX_uid6_fpSqrtTest_b_13_rdmux(MUX,166)
    assign redist13_expX_uid6_fpSqrtTest_b_13_rdmux_s = en;
    always @(redist13_expX_uid6_fpSqrtTest_b_13_rdmux_s or redist13_expX_uid6_fpSqrtTest_b_13_wraddr_q or redist13_expX_uid6_fpSqrtTest_b_13_rdcnt_q)
    begin
        unique case (redist13_expX_uid6_fpSqrtTest_b_13_rdmux_s)
            1'b0 : redist13_expX_uid6_fpSqrtTest_b_13_rdmux_q = redist13_expX_uid6_fpSqrtTest_b_13_wraddr_q;
            1'b1 : redist13_expX_uid6_fpSqrtTest_b_13_rdmux_q = redist13_expX_uid6_fpSqrtTest_b_13_rdcnt_q;
            default : redist13_expX_uid6_fpSqrtTest_b_13_rdmux_q = 4'b0;
        endcase
    end

    // VCC(CONSTANT,1)
    assign VCC_q = 1'b1;

    // expX_uid6_fpSqrtTest(BITSELECT,5)@0
    assign expX_uid6_fpSqrtTest_b = a[30:23];

    // redist13_expX_uid6_fpSqrtTest_b_13_wraddr(REG,167)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist13_expX_uid6_fpSqrtTest_b_13_wraddr_q <= 4'b1010;
        end
        else
        begin
            redist13_expX_uid6_fpSqrtTest_b_13_wraddr_q <= redist13_expX_uid6_fpSqrtTest_b_13_rdmux_q;
        end
    end

    // redist13_expX_uid6_fpSqrtTest_b_13_mem(DUALMEM,164)
    assign redist13_expX_uid6_fpSqrtTest_b_13_mem_ia = expX_uid6_fpSqrtTest_b;
    assign redist13_expX_uid6_fpSqrtTest_b_13_mem_aa = redist13_expX_uid6_fpSqrtTest_b_13_wraddr_q;
    assign redist13_expX_uid6_fpSqrtTest_b_13_mem_ab = redist13_expX_uid6_fpSqrtTest_b_13_rdmux_q;
    assign redist13_expX_uid6_fpSqrtTest_b_13_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(8),
        .widthad_a(4),
        .numwords_a(11),
        .width_b(8),
        .widthad_b(4),
        .numwords_b(11),
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
    ) redist13_expX_uid6_fpSqrtTest_b_13_mem_dmem (
        .clocken1(redist13_expX_uid6_fpSqrtTest_b_13_enaAnd_q[0]),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .aclr1(redist13_expX_uid6_fpSqrtTest_b_13_mem_reset0),
        .clock1(clk),
        .address_a(redist13_expX_uid6_fpSqrtTest_b_13_mem_aa),
        .data_a(redist13_expX_uid6_fpSqrtTest_b_13_mem_ia),
        .wren_a(en[0]),
        .address_b(redist13_expX_uid6_fpSqrtTest_b_13_mem_ab),
        .q_b(redist13_expX_uid6_fpSqrtTest_b_13_mem_iq),
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
    assign redist13_expX_uid6_fpSqrtTest_b_13_mem_q = redist13_expX_uid6_fpSqrtTest_b_13_mem_iq[7:0];

    // redist13_expX_uid6_fpSqrtTest_b_13_outputreg(DELAY,163)
    dspba_delay_ver #( .width(8), .depth(1), .reset_kind("ASYNC") )
    redist13_expX_uid6_fpSqrtTest_b_13_outputreg ( .xin(redist13_expX_uid6_fpSqrtTest_b_13_mem_q), .xout(redist13_expX_uid6_fpSqrtTest_b_13_outputreg_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // excZ_x_uid13_fpSqrtTest(LOGICAL,12)@13 + 1
    assign excZ_x_uid13_fpSqrtTest_qi = redist13_expX_uid6_fpSqrtTest_b_13_outputreg_q == cstAllZWE_uid10_fpSqrtTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    excZ_x_uid13_fpSqrtTest_delay ( .xin(excZ_x_uid13_fpSqrtTest_qi), .xout(excZ_x_uid13_fpSqrtTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // negZero_uid85_fpSqrtTest(LOGICAL,84)@14 + 1
    assign negZero_uid85_fpSqrtTest_qi = excZ_x_uid13_fpSqrtTest_q & redist12_signX_uid7_fpSqrtTest_b_14_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    negZero_uid85_fpSqrtTest_delay ( .xin(negZero_uid85_fpSqrtTest_qi), .xout(negZero_uid85_fpSqrtTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // cstAllOWE_uid8_fpSqrtTest(CONSTANT,7)
    assign cstAllOWE_uid8_fpSqrtTest_q = 8'b11111111;

    // GND(CONSTANT,0)
    assign GND_q = 1'b0;

    // redist11_frac_x_uid12_fpSqrtTest_b_13_notEnable(LOGICAL,159)
    assign redist11_frac_x_uid12_fpSqrtTest_b_13_notEnable_q = ~ (en);

    // redist11_frac_x_uid12_fpSqrtTest_b_13_nor(LOGICAL,160)
    assign redist11_frac_x_uid12_fpSqrtTest_b_13_nor_q = ~ (redist11_frac_x_uid12_fpSqrtTest_b_13_notEnable_q | redist11_frac_x_uid12_fpSqrtTest_b_13_sticky_ena_q);

    // redist11_frac_x_uid12_fpSqrtTest_b_13_mem_last(CONSTANT,156)
    assign redist11_frac_x_uid12_fpSqrtTest_b_13_mem_last_q = 4'b0111;

    // redist11_frac_x_uid12_fpSqrtTest_b_13_cmp(LOGICAL,157)
    assign redist11_frac_x_uid12_fpSqrtTest_b_13_cmp_q = redist11_frac_x_uid12_fpSqrtTest_b_13_mem_last_q == redist11_frac_x_uid12_fpSqrtTest_b_13_rdmux_q ? 1'b1 : 1'b0;

    // redist11_frac_x_uid12_fpSqrtTest_b_13_cmpReg(REG,158)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist11_frac_x_uid12_fpSqrtTest_b_13_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist11_frac_x_uid12_fpSqrtTest_b_13_cmpReg_q <= redist11_frac_x_uid12_fpSqrtTest_b_13_cmp_q;
        end
    end

    // redist11_frac_x_uid12_fpSqrtTest_b_13_sticky_ena(REG,161)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist11_frac_x_uid12_fpSqrtTest_b_13_sticky_ena_q <= 1'b0;
        end
        else if (redist11_frac_x_uid12_fpSqrtTest_b_13_nor_q == 1'b1)
        begin
            redist11_frac_x_uid12_fpSqrtTest_b_13_sticky_ena_q <= redist11_frac_x_uid12_fpSqrtTest_b_13_cmpReg_q;
        end
    end

    // redist11_frac_x_uid12_fpSqrtTest_b_13_enaAnd(LOGICAL,162)
    assign redist11_frac_x_uid12_fpSqrtTest_b_13_enaAnd_q = redist11_frac_x_uid12_fpSqrtTest_b_13_sticky_ena_q & en;

    // redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt(COUNTER,153)
    // low=0, high=8, step=1, init=0
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt_i <= 4'd0;
            redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt_i == 4'd7)
            begin
                redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt_eq <= 1'b0;
            end
            if (redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt_eq == 1'b1)
            begin
                redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt_i <= $unsigned(redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt_i) + $unsigned(4'd8);
            end
            else
            begin
                redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt_i <= $unsigned(redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt_i) + $unsigned(4'd1);
            end
        end
    end
    assign redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt_q = redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt_i[3:0];

    // redist11_frac_x_uid12_fpSqrtTest_b_13_rdmux(MUX,154)
    assign redist11_frac_x_uid12_fpSqrtTest_b_13_rdmux_s = en;
    always @(redist11_frac_x_uid12_fpSqrtTest_b_13_rdmux_s or redist11_frac_x_uid12_fpSqrtTest_b_13_wraddr_q or redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt_q)
    begin
        unique case (redist11_frac_x_uid12_fpSqrtTest_b_13_rdmux_s)
            1'b0 : redist11_frac_x_uid12_fpSqrtTest_b_13_rdmux_q = redist11_frac_x_uid12_fpSqrtTest_b_13_wraddr_q;
            1'b1 : redist11_frac_x_uid12_fpSqrtTest_b_13_rdmux_q = redist11_frac_x_uid12_fpSqrtTest_b_13_rdcnt_q;
            default : redist11_frac_x_uid12_fpSqrtTest_b_13_rdmux_q = 4'b0;
        endcase
    end

    // frac_x_uid12_fpSqrtTest(BITSELECT,11)@0
    assign frac_x_uid12_fpSqrtTest_b = a[22:0];

    // redist10_frac_x_uid12_fpSqrtTest_b_2(DELAY,135)
    dspba_delay_ver #( .width(23), .depth(2), .reset_kind("ASYNC") )
    redist10_frac_x_uid12_fpSqrtTest_b_2 ( .xin(frac_x_uid12_fpSqrtTest_b), .xout(redist10_frac_x_uid12_fpSqrtTest_b_2_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist11_frac_x_uid12_fpSqrtTest_b_13_wraddr(REG,155)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist11_frac_x_uid12_fpSqrtTest_b_13_wraddr_q <= 4'b1000;
        end
        else
        begin
            redist11_frac_x_uid12_fpSqrtTest_b_13_wraddr_q <= redist11_frac_x_uid12_fpSqrtTest_b_13_rdmux_q;
        end
    end

    // redist11_frac_x_uid12_fpSqrtTest_b_13_mem(DUALMEM,152)
    assign redist11_frac_x_uid12_fpSqrtTest_b_13_mem_ia = redist10_frac_x_uid12_fpSqrtTest_b_2_q;
    assign redist11_frac_x_uid12_fpSqrtTest_b_13_mem_aa = redist11_frac_x_uid12_fpSqrtTest_b_13_wraddr_q;
    assign redist11_frac_x_uid12_fpSqrtTest_b_13_mem_ab = redist11_frac_x_uid12_fpSqrtTest_b_13_rdmux_q;
    assign redist11_frac_x_uid12_fpSqrtTest_b_13_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(23),
        .widthad_a(4),
        .numwords_a(9),
        .width_b(23),
        .widthad_b(4),
        .numwords_b(9),
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
    ) redist11_frac_x_uid12_fpSqrtTest_b_13_mem_dmem (
        .clocken1(redist11_frac_x_uid12_fpSqrtTest_b_13_enaAnd_q[0]),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .aclr1(redist11_frac_x_uid12_fpSqrtTest_b_13_mem_reset0),
        .clock1(clk),
        .address_a(redist11_frac_x_uid12_fpSqrtTest_b_13_mem_aa),
        .data_a(redist11_frac_x_uid12_fpSqrtTest_b_13_mem_ia),
        .wren_a(en[0]),
        .address_b(redist11_frac_x_uid12_fpSqrtTest_b_13_mem_ab),
        .q_b(redist11_frac_x_uid12_fpSqrtTest_b_13_mem_iq),
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
    assign redist11_frac_x_uid12_fpSqrtTest_b_13_mem_q = redist11_frac_x_uid12_fpSqrtTest_b_13_mem_iq[22:0];

    // redist11_frac_x_uid12_fpSqrtTest_b_13_outputreg(DELAY,151)
    dspba_delay_ver #( .width(23), .depth(1), .reset_kind("ASYNC") )
    redist11_frac_x_uid12_fpSqrtTest_b_13_outputreg ( .xin(redist11_frac_x_uid12_fpSqrtTest_b_13_mem_q), .xout(redist11_frac_x_uid12_fpSqrtTest_b_13_outputreg_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // oFracX_uid44_fpSqrtTest(BITJOIN,43)@13
    assign oFracX_uid44_fpSqrtTest_q = {VCC_q, redist11_frac_x_uid12_fpSqrtTest_b_13_outputreg_q};

    // oFracXZ_mergedSignalTM_uid47_fpSqrtTest(BITJOIN,46)@13
    assign oFracXZ_mergedSignalTM_uid47_fpSqrtTest_q = {oFracX_uid44_fpSqrtTest_q, GND_q};

    // oFracXSignExt_mergedSignalTM_uid52_fpSqrtTest(BITJOIN,51)@13
    assign oFracXSignExt_mergedSignalTM_uid52_fpSqrtTest_q = {GND_q, oFracX_uid44_fpSqrtTest_q};

    // expX0PS_uid29_fpSqrtTest(BITSELECT,28)@0
    assign expX0PS_uid29_fpSqrtTest_in = expX_uid6_fpSqrtTest_b[0:0];
    assign expX0PS_uid29_fpSqrtTest_b = expX0PS_uid29_fpSqrtTest_in[0:0];

    // expOddSelect_uid30_fpSqrtTest(LOGICAL,29)@0
    assign expOddSelect_uid30_fpSqrtTest_q = ~ (expX0PS_uid29_fpSqrtTest_b);

    // redist9_expOddSelect_uid30_fpSqrtTest_q_13(DELAY,134)
    dspba_delay_ver #( .width(1), .depth(13), .reset_kind("ASYNC") )
    redist9_expOddSelect_uid30_fpSqrtTest_q_13 ( .xin(expOddSelect_uid30_fpSqrtTest_q), .xout(redist9_expOddSelect_uid30_fpSqrtTest_q_13_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // normalizedXForComp_uid54_fpSqrtTest(MUX,53)@13
    assign normalizedXForComp_uid54_fpSqrtTest_s = redist9_expOddSelect_uid30_fpSqrtTest_q_13_q;
    always @(normalizedXForComp_uid54_fpSqrtTest_s or en or oFracXSignExt_mergedSignalTM_uid52_fpSqrtTest_q or oFracXZ_mergedSignalTM_uid47_fpSqrtTest_q)
    begin
        unique case (normalizedXForComp_uid54_fpSqrtTest_s)
            1'b0 : normalizedXForComp_uid54_fpSqrtTest_q = oFracXSignExt_mergedSignalTM_uid52_fpSqrtTest_q;
            1'b1 : normalizedXForComp_uid54_fpSqrtTest_q = oFracXZ_mergedSignalTM_uid47_fpSqrtTest_q;
            default : normalizedXForComp_uid54_fpSqrtTest_q = 25'b0;
        endcase
    end

    // paddingY_uid55_fpSqrtTest(CONSTANT,54)
    assign paddingY_uid55_fpSqrtTest_q = 25'b0000000000000000000000000;

    // updatedY_uid56_fpSqrtTest(BITJOIN,55)@13
    assign updatedY_uid56_fpSqrtTest_q = {normalizedXForComp_uid54_fpSqrtTest_q, paddingY_uid55_fpSqrtTest_q};

    // addrFull_uid33_fpSqrtTest(BITJOIN,32)@0
    assign addrFull_uid33_fpSqrtTest_q = {expOddSelect_uid30_fpSqrtTest_q, frac_x_uid12_fpSqrtTest_b};

    // yAddr_uid35_fpSqrtTest(BITSELECT,34)@0
    assign yAddr_uid35_fpSqrtTest_b = addrFull_uid33_fpSqrtTest_q[23:16];

    // memoryC2_uid94_sqrtTables_lutmem(DUALMEM,120)@0 + 2
    // in j@20000000
    assign memoryC2_uid94_sqrtTables_lutmem_aa = yAddr_uid35_fpSqrtTest_b;
    assign memoryC2_uid94_sqrtTables_lutmem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("M20K"),
        .operation_mode("ROM"),
        .width_a(12),
        .widthad_a(8),
        .numwords_a(256),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .outdata_reg_a("CLOCK0"),
        .outdata_aclr_a("CLEAR0"),
        .clock_enable_input_a("NORMAL"),
        .power_up_uninitialized("FALSE"),
        .init_file("acl_fsqrt_memoryC2_uid94_sqrtTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Arria 10")
    ) memoryC2_uid94_sqrtTables_lutmem_dmem (
        .clocken0(en[0]),
        .aclr0(memoryC2_uid94_sqrtTables_lutmem_reset0),
        .clock0(clk),
        .address_a(memoryC2_uid94_sqrtTables_lutmem_aa),
        .q_a(memoryC2_uid94_sqrtTables_lutmem_ir),
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
    assign memoryC2_uid94_sqrtTables_lutmem_r = memoryC2_uid94_sqrtTables_lutmem_ir[11:0];

    // yForPe_uid36_fpSqrtTest(BITSELECT,35)@2
    assign yForPe_uid36_fpSqrtTest_in = redist10_frac_x_uid12_fpSqrtTest_b_2_q[15:0];
    assign yForPe_uid36_fpSqrtTest_b = yForPe_uid36_fpSqrtTest_in[15:0];

    // yT1_uid100_invPolyEval(BITSELECT,99)@2
    assign yT1_uid100_invPolyEval_b = yForPe_uid36_fpSqrtTest_b[15:4];

    // prodXY_uid113_pT1_uid101_invPolyEval_cma(CHAINMULTADD,122)@2 + 3
    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_reset = areset;
    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_ena0 = en[0];
    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_ena1 = prodXY_uid113_pT1_uid101_invPolyEval_cma_ena0;
    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_ena2 = prodXY_uid113_pT1_uid101_invPolyEval_cma_ena0;
    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_l[0] = $signed({1'b0, prodXY_uid113_pT1_uid101_invPolyEval_cma_a1[0][11:0]});
    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_p[0] = prodXY_uid113_pT1_uid101_invPolyEval_cma_l[0] * prodXY_uid113_pT1_uid101_invPolyEval_cma_c1[0];
    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_u[0] = prodXY_uid113_pT1_uid101_invPolyEval_cma_p[0][24:0];
    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_w[0] = prodXY_uid113_pT1_uid101_invPolyEval_cma_u[0];
    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_x[0] = prodXY_uid113_pT1_uid101_invPolyEval_cma_w[0];
    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_y[0] = prodXY_uid113_pT1_uid101_invPolyEval_cma_x[0];
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid113_pT1_uid101_invPolyEval_cma_a0 <= '{default: '0};
            prodXY_uid113_pT1_uid101_invPolyEval_cma_c0 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid113_pT1_uid101_invPolyEval_cma_ena0 == 1'b1)
            begin
                prodXY_uid113_pT1_uid101_invPolyEval_cma_a0[0] <= yT1_uid100_invPolyEval_b;
                prodXY_uid113_pT1_uid101_invPolyEval_cma_c0[0] <= memoryC2_uid94_sqrtTables_lutmem_r;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid113_pT1_uid101_invPolyEval_cma_a1 <= '{default: '0};
            prodXY_uid113_pT1_uid101_invPolyEval_cma_c1 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid113_pT1_uid101_invPolyEval_cma_ena2 == 1'b1)
            begin
                prodXY_uid113_pT1_uid101_invPolyEval_cma_a1 <= prodXY_uid113_pT1_uid101_invPolyEval_cma_a0;
                prodXY_uid113_pT1_uid101_invPolyEval_cma_c1 <= prodXY_uid113_pT1_uid101_invPolyEval_cma_c0;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid113_pT1_uid101_invPolyEval_cma_s <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid113_pT1_uid101_invPolyEval_cma_ena1 == 1'b1)
            begin
                prodXY_uid113_pT1_uid101_invPolyEval_cma_s[0] <= prodXY_uid113_pT1_uid101_invPolyEval_cma_y[0];
            end
        end
    end
    dspba_delay_ver #( .width(24), .depth(0), .reset_kind("ASYNC") )
    prodXY_uid113_pT1_uid101_invPolyEval_cma_delay ( .xin(prodXY_uid113_pT1_uid101_invPolyEval_cma_s[0][23:0]), .xout(prodXY_uid113_pT1_uid101_invPolyEval_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_q = prodXY_uid113_pT1_uid101_invPolyEval_cma_qq[23:0];

    // osig_uid114_pT1_uid101_invPolyEval(BITSELECT,113)@5
    assign osig_uid114_pT1_uid101_invPolyEval_b = prodXY_uid113_pT1_uid101_invPolyEval_cma_q[23:11];

    // highBBits_uid103_invPolyEval(BITSELECT,102)@5
    assign highBBits_uid103_invPolyEval_b = osig_uid114_pT1_uid101_invPolyEval_b[12:1];

    // redist6_yAddr_uid35_fpSqrtTest_b_3(DELAY,131)
    dspba_delay_ver #( .width(8), .depth(3), .reset_kind("ASYNC") )
    redist6_yAddr_uid35_fpSqrtTest_b_3 ( .xin(yAddr_uid35_fpSqrtTest_b), .xout(redist6_yAddr_uid35_fpSqrtTest_b_3_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // memoryC1_uid91_sqrtTables_lutmem(DUALMEM,119)@3 + 2
    // in j@20000000
    assign memoryC1_uid91_sqrtTables_lutmem_aa = redist6_yAddr_uid35_fpSqrtTest_b_3_q;
    assign memoryC1_uid91_sqrtTables_lutmem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("M20K"),
        .operation_mode("ROM"),
        .width_a(21),
        .widthad_a(8),
        .numwords_a(256),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .outdata_reg_a("CLOCK0"),
        .outdata_aclr_a("CLEAR0"),
        .clock_enable_input_a("NORMAL"),
        .power_up_uninitialized("FALSE"),
        .init_file("acl_fsqrt_memoryC1_uid91_sqrtTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Arria 10")
    ) memoryC1_uid91_sqrtTables_lutmem_dmem (
        .clocken0(en[0]),
        .aclr0(memoryC1_uid91_sqrtTables_lutmem_reset0),
        .clock0(clk),
        .address_a(memoryC1_uid91_sqrtTables_lutmem_aa),
        .q_a(memoryC1_uid91_sqrtTables_lutmem_ir),
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
    assign memoryC1_uid91_sqrtTables_lutmem_r = memoryC1_uid91_sqrtTables_lutmem_ir[20:0];

    // s1sumAHighB_uid104_invPolyEval(ADD,103)@5 + 1
    assign s1sumAHighB_uid104_invPolyEval_a = {{1{memoryC1_uid91_sqrtTables_lutmem_r[20]}}, memoryC1_uid91_sqrtTables_lutmem_r};
    assign s1sumAHighB_uid104_invPolyEval_b = {{10{highBBits_uid103_invPolyEval_b[11]}}, highBBits_uid103_invPolyEval_b};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            s1sumAHighB_uid104_invPolyEval_o <= 22'b0;
        end
        else if (en == 1'b1)
        begin
            s1sumAHighB_uid104_invPolyEval_o <= $signed(s1sumAHighB_uid104_invPolyEval_a) + $signed(s1sumAHighB_uid104_invPolyEval_b);
        end
    end
    assign s1sumAHighB_uid104_invPolyEval_q = s1sumAHighB_uid104_invPolyEval_o[21:0];

    // lowRangeB_uid102_invPolyEval(BITSELECT,101)@5
    assign lowRangeB_uid102_invPolyEval_in = osig_uid114_pT1_uid101_invPolyEval_b[0:0];
    assign lowRangeB_uid102_invPolyEval_b = lowRangeB_uid102_invPolyEval_in[0:0];

    // redist1_lowRangeB_uid102_invPolyEval_b_1(DELAY,126)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist1_lowRangeB_uid102_invPolyEval_b_1 ( .xin(lowRangeB_uid102_invPolyEval_b), .xout(redist1_lowRangeB_uid102_invPolyEval_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // s1_uid105_invPolyEval(BITJOIN,104)@6
    assign s1_uid105_invPolyEval_q = {s1sumAHighB_uid104_invPolyEval_q, redist1_lowRangeB_uid102_invPolyEval_b_1_q};

    // redist5_yForPe_uid36_fpSqrtTest_b_4_notEnable(LOGICAL,147)
    assign redist5_yForPe_uid36_fpSqrtTest_b_4_notEnable_q = ~ (en);

    // redist5_yForPe_uid36_fpSqrtTest_b_4_nor(LOGICAL,148)
    assign redist5_yForPe_uid36_fpSqrtTest_b_4_nor_q = ~ (redist5_yForPe_uid36_fpSqrtTest_b_4_notEnable_q | redist5_yForPe_uid36_fpSqrtTest_b_4_sticky_ena_q);

    // redist5_yForPe_uid36_fpSqrtTest_b_4_mem_last(CONSTANT,144)
    assign redist5_yForPe_uid36_fpSqrtTest_b_4_mem_last_q = 2'b01;

    // redist5_yForPe_uid36_fpSqrtTest_b_4_cmp(LOGICAL,145)
    assign redist5_yForPe_uid36_fpSqrtTest_b_4_cmp_q = redist5_yForPe_uid36_fpSqrtTest_b_4_mem_last_q == redist5_yForPe_uid36_fpSqrtTest_b_4_rdmux_q ? 1'b1 : 1'b0;

    // redist5_yForPe_uid36_fpSqrtTest_b_4_cmpReg(REG,146)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist5_yForPe_uid36_fpSqrtTest_b_4_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist5_yForPe_uid36_fpSqrtTest_b_4_cmpReg_q <= redist5_yForPe_uid36_fpSqrtTest_b_4_cmp_q;
        end
    end

    // redist5_yForPe_uid36_fpSqrtTest_b_4_sticky_ena(REG,149)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist5_yForPe_uid36_fpSqrtTest_b_4_sticky_ena_q <= 1'b0;
        end
        else if (redist5_yForPe_uid36_fpSqrtTest_b_4_nor_q == 1'b1)
        begin
            redist5_yForPe_uid36_fpSqrtTest_b_4_sticky_ena_q <= redist5_yForPe_uid36_fpSqrtTest_b_4_cmpReg_q;
        end
    end

    // redist5_yForPe_uid36_fpSqrtTest_b_4_enaAnd(LOGICAL,150)
    assign redist5_yForPe_uid36_fpSqrtTest_b_4_enaAnd_q = redist5_yForPe_uid36_fpSqrtTest_b_4_sticky_ena_q & en;

    // redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt(COUNTER,141)
    // low=0, high=2, step=1, init=0
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt_i <= 2'd0;
            redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt_i == 2'd1)
            begin
                redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt_eq <= 1'b0;
            end
            if (redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt_eq == 1'b1)
            begin
                redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt_i <= $unsigned(redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt_i) + $unsigned(2'd2);
            end
            else
            begin
                redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt_i <= $unsigned(redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt_i) + $unsigned(2'd1);
            end
        end
    end
    assign redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt_q = redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt_i[1:0];

    // redist5_yForPe_uid36_fpSqrtTest_b_4_rdmux(MUX,142)
    assign redist5_yForPe_uid36_fpSqrtTest_b_4_rdmux_s = en;
    always @(redist5_yForPe_uid36_fpSqrtTest_b_4_rdmux_s or redist5_yForPe_uid36_fpSqrtTest_b_4_wraddr_q or redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt_q)
    begin
        unique case (redist5_yForPe_uid36_fpSqrtTest_b_4_rdmux_s)
            1'b0 : redist5_yForPe_uid36_fpSqrtTest_b_4_rdmux_q = redist5_yForPe_uid36_fpSqrtTest_b_4_wraddr_q;
            1'b1 : redist5_yForPe_uid36_fpSqrtTest_b_4_rdmux_q = redist5_yForPe_uid36_fpSqrtTest_b_4_rdcnt_q;
            default : redist5_yForPe_uid36_fpSqrtTest_b_4_rdmux_q = 2'b0;
        endcase
    end

    // redist5_yForPe_uid36_fpSqrtTest_b_4_wraddr(REG,143)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist5_yForPe_uid36_fpSqrtTest_b_4_wraddr_q <= 2'b10;
        end
        else
        begin
            redist5_yForPe_uid36_fpSqrtTest_b_4_wraddr_q <= redist5_yForPe_uid36_fpSqrtTest_b_4_rdmux_q;
        end
    end

    // redist5_yForPe_uid36_fpSqrtTest_b_4_mem(DUALMEM,140)
    assign redist5_yForPe_uid36_fpSqrtTest_b_4_mem_ia = yForPe_uid36_fpSqrtTest_b;
    assign redist5_yForPe_uid36_fpSqrtTest_b_4_mem_aa = redist5_yForPe_uid36_fpSqrtTest_b_4_wraddr_q;
    assign redist5_yForPe_uid36_fpSqrtTest_b_4_mem_ab = redist5_yForPe_uid36_fpSqrtTest_b_4_rdmux_q;
    assign redist5_yForPe_uid36_fpSqrtTest_b_4_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(16),
        .widthad_a(2),
        .numwords_a(3),
        .width_b(16),
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
    ) redist5_yForPe_uid36_fpSqrtTest_b_4_mem_dmem (
        .clocken1(redist5_yForPe_uid36_fpSqrtTest_b_4_enaAnd_q[0]),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .aclr1(redist5_yForPe_uid36_fpSqrtTest_b_4_mem_reset0),
        .clock1(clk),
        .address_a(redist5_yForPe_uid36_fpSqrtTest_b_4_mem_aa),
        .data_a(redist5_yForPe_uid36_fpSqrtTest_b_4_mem_ia),
        .wren_a(en[0]),
        .address_b(redist5_yForPe_uid36_fpSqrtTest_b_4_mem_ab),
        .q_b(redist5_yForPe_uid36_fpSqrtTest_b_4_mem_iq),
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
    assign redist5_yForPe_uid36_fpSqrtTest_b_4_mem_q = redist5_yForPe_uid36_fpSqrtTest_b_4_mem_iq[15:0];

    // prodXY_uid116_pT2_uid107_invPolyEval_cma(CHAINMULTADD,123)@6 + 3
    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_reset = areset;
    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_ena0 = en[0];
    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_ena1 = prodXY_uid116_pT2_uid107_invPolyEval_cma_ena0;
    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_ena2 = prodXY_uid116_pT2_uid107_invPolyEval_cma_ena0;
    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_l[0] = $signed({1'b0, prodXY_uid116_pT2_uid107_invPolyEval_cma_a1[0][15:0]});
    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_p[0] = prodXY_uid116_pT2_uid107_invPolyEval_cma_l[0] * prodXY_uid116_pT2_uid107_invPolyEval_cma_c1[0];
    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_u[0] = prodXY_uid116_pT2_uid107_invPolyEval_cma_p[0][39:0];
    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_w[0] = prodXY_uid116_pT2_uid107_invPolyEval_cma_u[0];
    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_x[0] = prodXY_uid116_pT2_uid107_invPolyEval_cma_w[0];
    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_y[0] = prodXY_uid116_pT2_uid107_invPolyEval_cma_x[0];
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid116_pT2_uid107_invPolyEval_cma_a0 <= '{default: '0};
            prodXY_uid116_pT2_uid107_invPolyEval_cma_c0 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid116_pT2_uid107_invPolyEval_cma_ena0 == 1'b1)
            begin
                prodXY_uid116_pT2_uid107_invPolyEval_cma_a0[0] <= redist5_yForPe_uid36_fpSqrtTest_b_4_mem_q;
                prodXY_uid116_pT2_uid107_invPolyEval_cma_c0[0] <= s1_uid105_invPolyEval_q;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid116_pT2_uid107_invPolyEval_cma_a1 <= '{default: '0};
            prodXY_uid116_pT2_uid107_invPolyEval_cma_c1 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid116_pT2_uid107_invPolyEval_cma_ena2 == 1'b1)
            begin
                prodXY_uid116_pT2_uid107_invPolyEval_cma_a1 <= prodXY_uid116_pT2_uid107_invPolyEval_cma_a0;
                prodXY_uid116_pT2_uid107_invPolyEval_cma_c1 <= prodXY_uid116_pT2_uid107_invPolyEval_cma_c0;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid116_pT2_uid107_invPolyEval_cma_s <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid116_pT2_uid107_invPolyEval_cma_ena1 == 1'b1)
            begin
                prodXY_uid116_pT2_uid107_invPolyEval_cma_s[0] <= prodXY_uid116_pT2_uid107_invPolyEval_cma_y[0];
            end
        end
    end
    dspba_delay_ver #( .width(39), .depth(0), .reset_kind("ASYNC") )
    prodXY_uid116_pT2_uid107_invPolyEval_cma_delay ( .xin(prodXY_uid116_pT2_uid107_invPolyEval_cma_s[0][38:0]), .xout(prodXY_uid116_pT2_uid107_invPolyEval_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_q = prodXY_uid116_pT2_uid107_invPolyEval_cma_qq[38:0];

    // osig_uid117_pT2_uid107_invPolyEval(BITSELECT,116)@9
    assign osig_uid117_pT2_uid107_invPolyEval_b = prodXY_uid116_pT2_uid107_invPolyEval_cma_q[38:15];

    // highBBits_uid109_invPolyEval(BITSELECT,108)@9
    assign highBBits_uid109_invPolyEval_b = osig_uid117_pT2_uid107_invPolyEval_b[23:2];

    // redist7_yAddr_uid35_fpSqrtTest_b_7(DELAY,132)
    dspba_delay_ver #( .width(8), .depth(4), .reset_kind("ASYNC") )
    redist7_yAddr_uid35_fpSqrtTest_b_7 ( .xin(redist6_yAddr_uid35_fpSqrtTest_b_3_q), .xout(redist7_yAddr_uid35_fpSqrtTest_b_7_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // memoryC0_uid88_sqrtTables_lutmem(DUALMEM,118)@7 + 2
    // in j@20000000
    assign memoryC0_uid88_sqrtTables_lutmem_aa = redist7_yAddr_uid35_fpSqrtTest_b_7_q;
    assign memoryC0_uid88_sqrtTables_lutmem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("M20K"),
        .operation_mode("ROM"),
        .width_a(29),
        .widthad_a(8),
        .numwords_a(256),
        .lpm_type("altera_syncram"),
        .width_byteena_a(1),
        .outdata_reg_a("CLOCK0"),
        .outdata_aclr_a("CLEAR0"),
        .clock_enable_input_a("NORMAL"),
        .power_up_uninitialized("FALSE"),
        .init_file("acl_fsqrt_memoryC0_uid88_sqrtTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Arria 10")
    ) memoryC0_uid88_sqrtTables_lutmem_dmem (
        .clocken0(en[0]),
        .aclr0(memoryC0_uid88_sqrtTables_lutmem_reset0),
        .clock0(clk),
        .address_a(memoryC0_uid88_sqrtTables_lutmem_aa),
        .q_a(memoryC0_uid88_sqrtTables_lutmem_ir),
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
    assign memoryC0_uid88_sqrtTables_lutmem_r = memoryC0_uid88_sqrtTables_lutmem_ir[28:0];

    // s2sumAHighB_uid110_invPolyEval(ADD,109)@9
    assign s2sumAHighB_uid110_invPolyEval_a = {{1{memoryC0_uid88_sqrtTables_lutmem_r[28]}}, memoryC0_uid88_sqrtTables_lutmem_r};
    assign s2sumAHighB_uid110_invPolyEval_b = {{8{highBBits_uid109_invPolyEval_b[21]}}, highBBits_uid109_invPolyEval_b};
    assign s2sumAHighB_uid110_invPolyEval_o = $signed(s2sumAHighB_uid110_invPolyEval_a) + $signed(s2sumAHighB_uid110_invPolyEval_b);
    assign s2sumAHighB_uid110_invPolyEval_q = s2sumAHighB_uid110_invPolyEval_o[29:0];

    // lowRangeB_uid108_invPolyEval(BITSELECT,107)@9
    assign lowRangeB_uid108_invPolyEval_in = osig_uid117_pT2_uid107_invPolyEval_b[1:0];
    assign lowRangeB_uid108_invPolyEval_b = lowRangeB_uid108_invPolyEval_in[1:0];

    // s2_uid111_invPolyEval(BITJOIN,110)@9
    assign s2_uid111_invPolyEval_q = {s2sumAHighB_uid110_invPolyEval_q, lowRangeB_uid108_invPolyEval_b};

    // fracRPreCR_uid39_fpSqrtTest(BITSELECT,38)@9
    assign fracRPreCR_uid39_fpSqrtTest_in = s2_uid111_invPolyEval_q[28:0];
    assign fracRPreCR_uid39_fpSqrtTest_b = fracRPreCR_uid39_fpSqrtTest_in[28:5];

    // redist2_fracRPreCR_uid39_fpSqrtTest_b_1(DELAY,127)
    dspba_delay_ver #( .width(24), .depth(1), .reset_kind("ASYNC") )
    redist2_fracRPreCR_uid39_fpSqrtTest_b_1 ( .xin(fracRPreCR_uid39_fpSqrtTest_b), .xout(redist2_fracRPreCR_uid39_fpSqrtTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // fracPaddingOne_uid41_fpSqrtTest(BITJOIN,40)@10
    assign fracPaddingOne_uid41_fpSqrtTest_q = {VCC_q, redist2_fracRPreCR_uid39_fpSqrtTest_b_1_q};

    // squaredResult_uid42_fpSqrtTest_cma(CHAINMULTADD,121)@10 + 3
    assign squaredResult_uid42_fpSqrtTest_cma_reset = areset;
    assign squaredResult_uid42_fpSqrtTest_cma_ena0 = en[0];
    assign squaredResult_uid42_fpSqrtTest_cma_ena1 = squaredResult_uid42_fpSqrtTest_cma_ena0;
    assign squaredResult_uid42_fpSqrtTest_cma_ena2 = squaredResult_uid42_fpSqrtTest_cma_ena0;
    assign squaredResult_uid42_fpSqrtTest_cma_p[0] = squaredResult_uid42_fpSqrtTest_cma_a1[0] * squaredResult_uid42_fpSqrtTest_cma_c1[0];
    assign squaredResult_uid42_fpSqrtTest_cma_u[0] = squaredResult_uid42_fpSqrtTest_cma_p[0][49:0];
    assign squaredResult_uid42_fpSqrtTest_cma_w[0] = squaredResult_uid42_fpSqrtTest_cma_u[0];
    assign squaredResult_uid42_fpSqrtTest_cma_x[0] = squaredResult_uid42_fpSqrtTest_cma_w[0];
    assign squaredResult_uid42_fpSqrtTest_cma_y[0] = squaredResult_uid42_fpSqrtTest_cma_x[0];
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            squaredResult_uid42_fpSqrtTest_cma_a0 <= '{default: '0};
            squaredResult_uid42_fpSqrtTest_cma_c0 <= '{default: '0};
        end
        else
        begin
            if (squaredResult_uid42_fpSqrtTest_cma_ena0 == 1'b1)
            begin
                squaredResult_uid42_fpSqrtTest_cma_a0[0] <= fracPaddingOne_uid41_fpSqrtTest_q;
                squaredResult_uid42_fpSqrtTest_cma_c0[0] <= fracPaddingOne_uid41_fpSqrtTest_q;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            squaredResult_uid42_fpSqrtTest_cma_a1 <= '{default: '0};
            squaredResult_uid42_fpSqrtTest_cma_c1 <= '{default: '0};
        end
        else
        begin
            if (squaredResult_uid42_fpSqrtTest_cma_ena2 == 1'b1)
            begin
                squaredResult_uid42_fpSqrtTest_cma_a1 <= squaredResult_uid42_fpSqrtTest_cma_a0;
                squaredResult_uid42_fpSqrtTest_cma_c1 <= squaredResult_uid42_fpSqrtTest_cma_c0;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            squaredResult_uid42_fpSqrtTest_cma_s <= '{default: '0};
        end
        else
        begin
            if (squaredResult_uid42_fpSqrtTest_cma_ena1 == 1'b1)
            begin
                squaredResult_uid42_fpSqrtTest_cma_s[0] <= squaredResult_uid42_fpSqrtTest_cma_y[0];
            end
        end
    end
    dspba_delay_ver #( .width(50), .depth(0), .reset_kind("ASYNC") )
    squaredResult_uid42_fpSqrtTest_cma_delay ( .xin(squaredResult_uid42_fpSqrtTest_cma_s[0][49:0]), .xout(squaredResult_uid42_fpSqrtTest_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign squaredResult_uid42_fpSqrtTest_cma_q = squaredResult_uid42_fpSqrtTest_cma_qq[49:0];

    // squaredResultGTEIn_uid55_fpSqrtTest(COMPARE,56)@13 + 1
    assign squaredResultGTEIn_uid55_fpSqrtTest_a = {2'b00, squaredResult_uid42_fpSqrtTest_cma_q};
    assign squaredResultGTEIn_uid55_fpSqrtTest_b = {2'b00, updatedY_uid56_fpSqrtTest_q};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            squaredResultGTEIn_uid55_fpSqrtTest_o <= 52'b0;
        end
        else if (en == 1'b1)
        begin
            squaredResultGTEIn_uid55_fpSqrtTest_o <= $unsigned(squaredResultGTEIn_uid55_fpSqrtTest_a) - $unsigned(squaredResultGTEIn_uid55_fpSqrtTest_b);
        end
    end
    assign squaredResultGTEIn_uid55_fpSqrtTest_n[0] = ~ (squaredResultGTEIn_uid55_fpSqrtTest_o[51]);

    // pLTOne_uid58_fpSqrtTest(LOGICAL,57)@14
    assign pLTOne_uid58_fpSqrtTest_q = ~ (squaredResultGTEIn_uid55_fpSqrtTest_n);

    // redist3_fracRPreCR_uid39_fpSqrtTest_b_5_inputreg(DELAY,139)
    dspba_delay_ver #( .width(24), .depth(1), .reset_kind("ASYNC") )
    redist3_fracRPreCR_uid39_fpSqrtTest_b_5_inputreg ( .xin(redist2_fracRPreCR_uid39_fpSqrtTest_b_1_q), .xout(redist3_fracRPreCR_uid39_fpSqrtTest_b_5_inputreg_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist3_fracRPreCR_uid39_fpSqrtTest_b_5(DELAY,128)
    dspba_delay_ver #( .width(24), .depth(3), .reset_kind("ASYNC") )
    redist3_fracRPreCR_uid39_fpSqrtTest_b_5 ( .xin(redist3_fracRPreCR_uid39_fpSqrtTest_b_5_inputreg_q), .xout(redist3_fracRPreCR_uid39_fpSqrtTest_b_5_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // fxpSqrtResPostUpdateE_uid60_fpSqrtTest(ADD,59)@14
    assign fxpSqrtResPostUpdateE_uid60_fpSqrtTest_a = {1'b0, redist3_fracRPreCR_uid39_fpSqrtTest_b_5_q};
    assign fxpSqrtResPostUpdateE_uid60_fpSqrtTest_b = {24'b000000000000000000000000, pLTOne_uid58_fpSqrtTest_q};
    assign fxpSqrtResPostUpdateE_uid60_fpSqrtTest_o = $unsigned(fxpSqrtResPostUpdateE_uid60_fpSqrtTest_a) + $unsigned(fxpSqrtResPostUpdateE_uid60_fpSqrtTest_b);
    assign fxpSqrtResPostUpdateE_uid60_fpSqrtTest_q = fxpSqrtResPostUpdateE_uid60_fpSqrtTest_o[24:0];

    // expUpdateCRU_uid61_fpSqrtTest_merged_bit_select(BITSELECT,124)@14
    assign expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_b = fxpSqrtResPostUpdateE_uid60_fpSqrtTest_q[24:24];
    assign expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_c = fxpSqrtResPostUpdateE_uid60_fpSqrtTest_q[23:1];

    // fracPENotOne_uid62_fpSqrtTest(LOGICAL,61)@14
    assign fracPENotOne_uid62_fpSqrtTest_q = ~ (redist4_expIncPEOnly_uid38_fpSqrtTest_b_5_q);

    // fracPENotOneAndCRRoundsExp_uid63_fpSqrtTest(LOGICAL,62)@14
    assign fracPENotOneAndCRRoundsExp_uid63_fpSqrtTest_q = fracPENotOne_uid62_fpSqrtTest_q & expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_b;

    // expIncPEOnly_uid38_fpSqrtTest(BITSELECT,37)@9
    assign expIncPEOnly_uid38_fpSqrtTest_in = s2_uid111_invPolyEval_q[30:0];
    assign expIncPEOnly_uid38_fpSqrtTest_b = expIncPEOnly_uid38_fpSqrtTest_in[30:30];

    // redist4_expIncPEOnly_uid38_fpSqrtTest_b_5(DELAY,129)
    dspba_delay_ver #( .width(1), .depth(5), .reset_kind("ASYNC") )
    redist4_expIncPEOnly_uid38_fpSqrtTest_b_5 ( .xin(expIncPEOnly_uid38_fpSqrtTest_b), .xout(redist4_expIncPEOnly_uid38_fpSqrtTest_b_5_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expInc_uid64_fpSqrtTest(LOGICAL,63)@14 + 1
    assign expInc_uid64_fpSqrtTest_qi = redist4_expIncPEOnly_uid38_fpSqrtTest_b_5_q | fracPENotOneAndCRRoundsExp_uid63_fpSqrtTest_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    expInc_uid64_fpSqrtTest_delay ( .xin(expInc_uid64_fpSqrtTest_qi), .xout(expInc_uid64_fpSqrtTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // sBiasM1_uid26_fpSqrtTest(CONSTANT,25)
    assign sBiasM1_uid26_fpSqrtTest_q = 8'b01111110;

    // expOddSig_uid27_fpSqrtTest(ADD,26)@13
    assign expOddSig_uid27_fpSqrtTest_a = {1'b0, redist13_expX_uid6_fpSqrtTest_b_13_outputreg_q};
    assign expOddSig_uid27_fpSqrtTest_b = {1'b0, sBiasM1_uid26_fpSqrtTest_q};
    assign expOddSig_uid27_fpSqrtTest_o = $unsigned(expOddSig_uid27_fpSqrtTest_a) + $unsigned(expOddSig_uid27_fpSqrtTest_b);
    assign expOddSig_uid27_fpSqrtTest_q = expOddSig_uid27_fpSqrtTest_o[8:0];

    // expROdd_uid28_fpSqrtTest(BITSELECT,27)@13
    assign expROdd_uid28_fpSqrtTest_b = expOddSig_uid27_fpSqrtTest_q[8:1];

    // sBias_uid22_fpSqrtTest(CONSTANT,21)
    assign sBias_uid22_fpSqrtTest_q = 8'b01111111;

    // expEvenSig_uid24_fpSqrtTest(ADD,23)@13
    assign expEvenSig_uid24_fpSqrtTest_a = {1'b0, redist13_expX_uid6_fpSqrtTest_b_13_outputreg_q};
    assign expEvenSig_uid24_fpSqrtTest_b = {1'b0, sBias_uid22_fpSqrtTest_q};
    assign expEvenSig_uid24_fpSqrtTest_o = $unsigned(expEvenSig_uid24_fpSqrtTest_a) + $unsigned(expEvenSig_uid24_fpSqrtTest_b);
    assign expEvenSig_uid24_fpSqrtTest_q = expEvenSig_uid24_fpSqrtTest_o[8:0];

    // expREven_uid25_fpSqrtTest(BITSELECT,24)@13
    assign expREven_uid25_fpSqrtTest_b = expEvenSig_uid24_fpSqrtTest_q[8:1];

    // expRMux_uid31_fpSqrtTest(MUX,30)@13 + 1
    assign expRMux_uid31_fpSqrtTest_s = redist9_expOddSelect_uid30_fpSqrtTest_q_13_q;
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            expRMux_uid31_fpSqrtTest_q <= 8'b0;
        end
        else if (en == 1'b1)
        begin
            unique case (expRMux_uid31_fpSqrtTest_s)
                1'b0 : expRMux_uid31_fpSqrtTest_q <= expREven_uid25_fpSqrtTest_b;
                1'b1 : expRMux_uid31_fpSqrtTest_q <= expROdd_uid28_fpSqrtTest_b;
                default : expRMux_uid31_fpSqrtTest_q <= 8'b0;
            endcase
        end
    end

    // redist8_expRMux_uid31_fpSqrtTest_q_2(DELAY,133)
    dspba_delay_ver #( .width(8), .depth(1), .reset_kind("ASYNC") )
    redist8_expRMux_uid31_fpSqrtTest_q_2 ( .xin(expRMux_uid31_fpSqrtTest_q), .xout(redist8_expRMux_uid31_fpSqrtTest_q_2_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expR_uid66_fpSqrtTest(ADD,65)@15
    assign expR_uid66_fpSqrtTest_a = {1'b0, redist8_expRMux_uid31_fpSqrtTest_q_2_q};
    assign expR_uid66_fpSqrtTest_b = {8'b00000000, expInc_uid64_fpSqrtTest_q};
    assign expR_uid66_fpSqrtTest_o = $unsigned(expR_uid66_fpSqrtTest_a) + $unsigned(expR_uid66_fpSqrtTest_b);
    assign expR_uid66_fpSqrtTest_q = expR_uid66_fpSqrtTest_o[8:0];

    // expRR_uid77_fpSqrtTest(BITSELECT,76)@15
    assign expRR_uid77_fpSqrtTest_in = expR_uid66_fpSqrtTest_q[7:0];
    assign expRR_uid77_fpSqrtTest_b = expRR_uid77_fpSqrtTest_in[7:0];

    // expXIsMax_uid14_fpSqrtTest(LOGICAL,13)@13 + 1
    assign expXIsMax_uid14_fpSqrtTest_qi = redist13_expX_uid6_fpSqrtTest_b_13_outputreg_q == cstAllOWE_uid8_fpSqrtTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    expXIsMax_uid14_fpSqrtTest_delay ( .xin(expXIsMax_uid14_fpSqrtTest_qi), .xout(expXIsMax_uid14_fpSqrtTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // invExpXIsMax_uid19_fpSqrtTest(LOGICAL,18)@14
    assign invExpXIsMax_uid19_fpSqrtTest_q = ~ (expXIsMax_uid14_fpSqrtTest_q);

    // InvExpXIsZero_uid20_fpSqrtTest(LOGICAL,19)@14
    assign InvExpXIsZero_uid20_fpSqrtTest_q = ~ (excZ_x_uid13_fpSqrtTest_q);

    // excR_x_uid21_fpSqrtTest(LOGICAL,20)@14
    assign excR_x_uid21_fpSqrtTest_q = InvExpXIsZero_uid20_fpSqrtTest_q & invExpXIsMax_uid19_fpSqrtTest_q;

    // minReg_uid69_fpSqrtTest(LOGICAL,68)@14
    assign minReg_uid69_fpSqrtTest_q = excR_x_uid21_fpSqrtTest_q & redist12_signX_uid7_fpSqrtTest_b_14_q;

    // cstZeroWF_uid9_fpSqrtTest(CONSTANT,8)
    assign cstZeroWF_uid9_fpSqrtTest_q = 23'b00000000000000000000000;

    // fracXIsZero_uid15_fpSqrtTest(LOGICAL,14)@13 + 1
    assign fracXIsZero_uid15_fpSqrtTest_qi = cstZeroWF_uid9_fpSqrtTest_q == redist11_frac_x_uid12_fpSqrtTest_b_13_outputreg_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    fracXIsZero_uid15_fpSqrtTest_delay ( .xin(fracXIsZero_uid15_fpSqrtTest_qi), .xout(fracXIsZero_uid15_fpSqrtTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // excI_x_uid17_fpSqrtTest(LOGICAL,16)@14
    assign excI_x_uid17_fpSqrtTest_q = expXIsMax_uid14_fpSqrtTest_q & fracXIsZero_uid15_fpSqrtTest_q;

    // minInf_uid70_fpSqrtTest(LOGICAL,69)@14
    assign minInf_uid70_fpSqrtTest_q = excI_x_uid17_fpSqrtTest_q & redist12_signX_uid7_fpSqrtTest_b_14_q;

    // fracXIsNotZero_uid16_fpSqrtTest(LOGICAL,15)@14
    assign fracXIsNotZero_uid16_fpSqrtTest_q = ~ (fracXIsZero_uid15_fpSqrtTest_q);

    // excN_x_uid18_fpSqrtTest(LOGICAL,17)@14
    assign excN_x_uid18_fpSqrtTest_q = expXIsMax_uid14_fpSqrtTest_q & fracXIsNotZero_uid16_fpSqrtTest_q;

    // excRNaN_uid71_fpSqrtTest(LOGICAL,70)@14
    assign excRNaN_uid71_fpSqrtTest_q = excN_x_uid18_fpSqrtTest_q | minInf_uid70_fpSqrtTest_q | minReg_uid69_fpSqrtTest_q;

    // invSignX_uid67_fpSqrtTest(LOGICAL,66)@14
    assign invSignX_uid67_fpSqrtTest_q = ~ (redist12_signX_uid7_fpSqrtTest_b_14_q);

    // inInfAndNotNeg_uid68_fpSqrtTest(LOGICAL,67)@14
    assign inInfAndNotNeg_uid68_fpSqrtTest_q = excI_x_uid17_fpSqrtTest_q & invSignX_uid67_fpSqrtTest_q;

    // excConc_uid72_fpSqrtTest(BITJOIN,71)@14
    assign excConc_uid72_fpSqrtTest_q = {excRNaN_uid71_fpSqrtTest_q, inInfAndNotNeg_uid68_fpSqrtTest_q, excZ_x_uid13_fpSqrtTest_q};

    // fracSelIn_uid73_fpSqrtTest(BITJOIN,72)@14
    assign fracSelIn_uid73_fpSqrtTest_q = {redist12_signX_uid7_fpSqrtTest_b_14_q, excConc_uid72_fpSqrtTest_q};

    // fracSel_uid74_fpSqrtTest(LOOKUP,73)@14 + 1
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            fracSel_uid74_fpSqrtTest_q <= 2'b01;
        end
        else if (en == 1'b1)
        begin
            unique case (fracSelIn_uid73_fpSqrtTest_q)
                4'b0000 : fracSel_uid74_fpSqrtTest_q <= 2'b01;
                4'b0001 : fracSel_uid74_fpSqrtTest_q <= 2'b00;
                4'b0010 : fracSel_uid74_fpSqrtTest_q <= 2'b10;
                4'b0011 : fracSel_uid74_fpSqrtTest_q <= 2'b00;
                4'b0100 : fracSel_uid74_fpSqrtTest_q <= 2'b11;
                4'b0101 : fracSel_uid74_fpSqrtTest_q <= 2'b00;
                4'b0110 : fracSel_uid74_fpSqrtTest_q <= 2'b10;
                4'b0111 : fracSel_uid74_fpSqrtTest_q <= 2'b00;
                4'b1000 : fracSel_uid74_fpSqrtTest_q <= 2'b11;
                4'b1001 : fracSel_uid74_fpSqrtTest_q <= 2'b00;
                4'b1010 : fracSel_uid74_fpSqrtTest_q <= 2'b11;
                4'b1011 : fracSel_uid74_fpSqrtTest_q <= 2'b11;
                4'b1100 : fracSel_uid74_fpSqrtTest_q <= 2'b11;
                4'b1101 : fracSel_uid74_fpSqrtTest_q <= 2'b11;
                4'b1110 : fracSel_uid74_fpSqrtTest_q <= 2'b11;
                4'b1111 : fracSel_uid74_fpSqrtTest_q <= 2'b11;
                default : begin
                              // unreachable
                              fracSel_uid74_fpSqrtTest_q <= 2'bxx;
                          end
            endcase
        end
    end

    // expRPostExc_uid79_fpSqrtTest(MUX,78)@15
    assign expRPostExc_uid79_fpSqrtTest_s = fracSel_uid74_fpSqrtTest_q;
    always @(expRPostExc_uid79_fpSqrtTest_s or en or cstAllZWE_uid10_fpSqrtTest_q or expRR_uid77_fpSqrtTest_b or cstAllOWE_uid8_fpSqrtTest_q)
    begin
        unique case (expRPostExc_uid79_fpSqrtTest_s)
            2'b00 : expRPostExc_uid79_fpSqrtTest_q = cstAllZWE_uid10_fpSqrtTest_q;
            2'b01 : expRPostExc_uid79_fpSqrtTest_q = expRR_uid77_fpSqrtTest_b;
            2'b10 : expRPostExc_uid79_fpSqrtTest_q = cstAllOWE_uid8_fpSqrtTest_q;
            2'b11 : expRPostExc_uid79_fpSqrtTest_q = cstAllOWE_uid8_fpSqrtTest_q;
            default : expRPostExc_uid79_fpSqrtTest_q = 8'b0;
        endcase
    end

    // fracNaN_uid80_fpSqrtTest(CONSTANT,79)
    assign fracNaN_uid80_fpSqrtTest_q = 23'b00000000000000000000001;

    // redist0_expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_c_1(DELAY,125)
    dspba_delay_ver #( .width(23), .depth(1), .reset_kind("ASYNC") )
    redist0_expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_c_1 ( .xin(expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_c), .xout(redist0_expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_c_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // fracRPostExc_uid84_fpSqrtTest(MUX,83)@15
    assign fracRPostExc_uid84_fpSqrtTest_s = fracSel_uid74_fpSqrtTest_q;
    always @(fracRPostExc_uid84_fpSqrtTest_s or en or cstZeroWF_uid9_fpSqrtTest_q or redist0_expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_c_1_q or fracNaN_uid80_fpSqrtTest_q)
    begin
        unique case (fracRPostExc_uid84_fpSqrtTest_s)
            2'b00 : fracRPostExc_uid84_fpSqrtTest_q = cstZeroWF_uid9_fpSqrtTest_q;
            2'b01 : fracRPostExc_uid84_fpSqrtTest_q = redist0_expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_c_1_q;
            2'b10 : fracRPostExc_uid84_fpSqrtTest_q = cstZeroWF_uid9_fpSqrtTest_q;
            2'b11 : fracRPostExc_uid84_fpSqrtTest_q = fracNaN_uid80_fpSqrtTest_q;
            default : fracRPostExc_uid84_fpSqrtTest_q = 23'b0;
        endcase
    end

    // RSqrt_uid86_fpSqrtTest(BITJOIN,85)@15
    assign RSqrt_uid86_fpSqrtTest_q = {negZero_uid85_fpSqrtTest_q, expRPostExc_uid79_fpSqrtTest_q, fracRPostExc_uid84_fpSqrtTest_q};

    // xOut(GPOUT,4)@15
    assign q = RSqrt_uid86_fpSqrtTest_q;

endmodule
