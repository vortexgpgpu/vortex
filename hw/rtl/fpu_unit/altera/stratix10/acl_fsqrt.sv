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

// SystemVerilog created from acl_fsqrt
// SystemVerilog created on Sun Dec 27 09:48:58 2020


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
    wire memoryC0_uid88_sqrtTables_lutmem_enaOr_rst;
    wire memoryC1_uid91_sqrtTables_lutmem_reset0;
    wire [20:0] memoryC1_uid91_sqrtTables_lutmem_ia;
    wire [7:0] memoryC1_uid91_sqrtTables_lutmem_aa;
    wire [7:0] memoryC1_uid91_sqrtTables_lutmem_ab;
    wire [20:0] memoryC1_uid91_sqrtTables_lutmem_ir;
    wire [20:0] memoryC1_uid91_sqrtTables_lutmem_r;
    wire memoryC1_uid91_sqrtTables_lutmem_enaOr_rst;
    wire memoryC2_uid94_sqrtTables_lutmem_reset0;
    wire [11:0] memoryC2_uid94_sqrtTables_lutmem_ia;
    wire [7:0] memoryC2_uid94_sqrtTables_lutmem_aa;
    wire [7:0] memoryC2_uid94_sqrtTables_lutmem_ab;
    wire [11:0] memoryC2_uid94_sqrtTables_lutmem_ir;
    wire [11:0] memoryC2_uid94_sqrtTables_lutmem_r;
    wire memoryC2_uid94_sqrtTables_lutmem_enaOr_rst;
    wire squaredResult_uid42_fpSqrtTest_cma_reset;
    (* preserve_syn_only *) reg [24:0] squaredResult_uid42_fpSqrtTest_cma_ah [0:0];
    (* preserve_syn_only *) reg [24:0] squaredResult_uid42_fpSqrtTest_cma_ch [0:0];
    wire [24:0] squaredResult_uid42_fpSqrtTest_cma_a0;
    wire [24:0] squaredResult_uid42_fpSqrtTest_cma_c0;
    wire [49:0] squaredResult_uid42_fpSqrtTest_cma_s0;
    wire [49:0] squaredResult_uid42_fpSqrtTest_cma_qq;
    reg [49:0] squaredResult_uid42_fpSqrtTest_cma_q;
    wire squaredResult_uid42_fpSqrtTest_cma_ena0;
    wire squaredResult_uid42_fpSqrtTest_cma_ena1;
    wire squaredResult_uid42_fpSqrtTest_cma_ena2;
    wire prodXY_uid113_pT1_uid101_invPolyEval_cma_reset;
    (* preserve_syn_only *) reg [11:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_ah [0:0];
    (* preserve_syn_only *) reg signed [11:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_ch [0:0];
    wire [11:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_a0;
    wire [11:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_c0;
    wire [23:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_s0;
    wire [23:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_qq;
    reg [23:0] prodXY_uid113_pT1_uid101_invPolyEval_cma_q;
    wire prodXY_uid113_pT1_uid101_invPolyEval_cma_ena0;
    wire prodXY_uid113_pT1_uid101_invPolyEval_cma_ena1;
    wire prodXY_uid113_pT1_uid101_invPolyEval_cma_ena2;
    wire prodXY_uid116_pT2_uid107_invPolyEval_cma_reset;
    (* preserve_syn_only *) reg [15:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_ah [0:0];
    (* preserve_syn_only *) reg signed [22:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_ch [0:0];
    wire [15:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_a0;
    wire [22:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_c0;
    wire [38:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_s0;
    wire [38:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_qq;
    reg [38:0] prodXY_uid116_pT2_uid107_invPolyEval_cma_q;
    wire prodXY_uid116_pT2_uid107_invPolyEval_cma_ena0;
    wire prodXY_uid116_pT2_uid107_invPolyEval_cma_ena1;
    wire prodXY_uid116_pT2_uid107_invPolyEval_cma_ena2;
    wire [0:0] expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_b;
    wire [22:0] expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_c;
    reg [22:0] redist0_expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_c_1_q;
    reg [11:0] redist1_memoryC2_uid94_sqrtTables_lutmem_r_1_q;
    reg [0:0] redist2_lowRangeB_uid102_invPolyEval_b_1_q;
    reg [23:0] redist3_fracRPreCR_uid39_fpSqrtTest_b_1_q;
    reg [0:0] redist5_expIncPEOnly_uid38_fpSqrtTest_b_8_q;
    reg [7:0] redist9_expRMux_uid31_fpSqrtTest_q_2_q;
    reg [0:0] redist10_expOddSelect_uid30_fpSqrtTest_q_23_q;
    reg [22:0] redist11_frac_x_uid12_fpSqrtTest_b_3_q;
    reg [22:0] redist11_frac_x_uid12_fpSqrtTest_b_3_delay_0;
    reg [22:0] redist11_frac_x_uid12_fpSqrtTest_b_3_delay_1;
    reg [0:0] redist13_signX_uid7_fpSqrtTest_b_24_q;
    wire redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_reset0;
    wire [23:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_ia;
    wire [2:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_aa;
    wire [2:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_ab;
    wire [23:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_iq;
    wire [23:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_q;
    wire redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_enaOr_rst;
    wire [2:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt_q;
    (* preserve_syn_only *) reg [2:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt_i;
    (* preserve_syn_only *) reg redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt_eq;
    wire [0:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdmux_s;
    reg [2:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdmux_q;
    reg [2:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_wraddr_q;
    wire [3:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_last_q;
    wire [3:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_cmp_b;
    wire [0:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_cmp_q;
    reg [0:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_cmpReg_q;
    wire [0:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_notEnable_q;
    wire [0:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_nor_q;
    (* preserve_syn_only *) reg [0:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_sticky_ena_q;
    wire [0:0] redist4_fracRPreCR_uid39_fpSqrtTest_b_8_enaAnd_q;
    reg [15:0] redist6_yForPe_uid36_fpSqrtTest_b_7_outputreg0_q;
    wire redist6_yForPe_uid36_fpSqrtTest_b_7_mem_reset0;
    wire [15:0] redist6_yForPe_uid36_fpSqrtTest_b_7_mem_ia;
    wire [2:0] redist6_yForPe_uid36_fpSqrtTest_b_7_mem_aa;
    wire [2:0] redist6_yForPe_uid36_fpSqrtTest_b_7_mem_ab;
    wire [15:0] redist6_yForPe_uid36_fpSqrtTest_b_7_mem_iq;
    wire [15:0] redist6_yForPe_uid36_fpSqrtTest_b_7_mem_q;
    wire redist6_yForPe_uid36_fpSqrtTest_b_7_mem_enaOr_rst;
    wire [2:0] redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt_q;
    (* preserve_syn_only *) reg [2:0] redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt_i;
    (* preserve_syn_only *) reg redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt_eq;
    wire [0:0] redist6_yForPe_uid36_fpSqrtTest_b_7_rdmux_s;
    reg [2:0] redist6_yForPe_uid36_fpSqrtTest_b_7_rdmux_q;
    reg [2:0] redist6_yForPe_uid36_fpSqrtTest_b_7_wraddr_q;
    wire [2:0] redist6_yForPe_uid36_fpSqrtTest_b_7_mem_last_q;
    wire [0:0] redist6_yForPe_uid36_fpSqrtTest_b_7_cmp_q;
    reg [0:0] redist6_yForPe_uid36_fpSqrtTest_b_7_cmpReg_q;
    wire [0:0] redist6_yForPe_uid36_fpSqrtTest_b_7_notEnable_q;
    wire [0:0] redist6_yForPe_uid36_fpSqrtTest_b_7_nor_q;
    (* preserve_syn_only *) reg [0:0] redist6_yForPe_uid36_fpSqrtTest_b_7_sticky_ena_q;
    wire [0:0] redist6_yForPe_uid36_fpSqrtTest_b_7_enaAnd_q;
    reg [7:0] redist7_yAddr_uid35_fpSqrtTest_b_7_outputreg0_q;
    wire redist7_yAddr_uid35_fpSqrtTest_b_7_mem_reset0;
    wire [7:0] redist7_yAddr_uid35_fpSqrtTest_b_7_mem_ia;
    wire [2:0] redist7_yAddr_uid35_fpSqrtTest_b_7_mem_aa;
    wire [2:0] redist7_yAddr_uid35_fpSqrtTest_b_7_mem_ab;
    wire [7:0] redist7_yAddr_uid35_fpSqrtTest_b_7_mem_iq;
    wire [7:0] redist7_yAddr_uid35_fpSqrtTest_b_7_mem_q;
    wire redist7_yAddr_uid35_fpSqrtTest_b_7_mem_enaOr_rst;
    wire [2:0] redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt_q;
    (* preserve_syn_only *) reg [2:0] redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt_i;
    (* preserve_syn_only *) reg redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt_eq;
    wire [0:0] redist7_yAddr_uid35_fpSqrtTest_b_7_rdmux_s;
    reg [2:0] redist7_yAddr_uid35_fpSqrtTest_b_7_rdmux_q;
    reg [2:0] redist7_yAddr_uid35_fpSqrtTest_b_7_wraddr_q;
    wire [2:0] redist7_yAddr_uid35_fpSqrtTest_b_7_mem_last_q;
    wire [0:0] redist7_yAddr_uid35_fpSqrtTest_b_7_cmp_q;
    reg [0:0] redist7_yAddr_uid35_fpSqrtTest_b_7_cmpReg_q;
    wire [0:0] redist7_yAddr_uid35_fpSqrtTest_b_7_notEnable_q;
    wire [0:0] redist7_yAddr_uid35_fpSqrtTest_b_7_nor_q;
    (* preserve_syn_only *) reg [0:0] redist7_yAddr_uid35_fpSqrtTest_b_7_sticky_ena_q;
    wire [0:0] redist7_yAddr_uid35_fpSqrtTest_b_7_enaAnd_q;
    reg [7:0] redist8_yAddr_uid35_fpSqrtTest_b_14_outputreg0_q;
    wire redist8_yAddr_uid35_fpSqrtTest_b_14_mem_reset0;
    wire [7:0] redist8_yAddr_uid35_fpSqrtTest_b_14_mem_ia;
    wire [2:0] redist8_yAddr_uid35_fpSqrtTest_b_14_mem_aa;
    wire [2:0] redist8_yAddr_uid35_fpSqrtTest_b_14_mem_ab;
    wire [7:0] redist8_yAddr_uid35_fpSqrtTest_b_14_mem_iq;
    wire [7:0] redist8_yAddr_uid35_fpSqrtTest_b_14_mem_q;
    wire redist8_yAddr_uid35_fpSqrtTest_b_14_mem_enaOr_rst;
    wire [2:0] redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt_q;
    (* preserve_syn_only *) reg [2:0] redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt_i;
    (* preserve_syn_only *) reg redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt_eq;
    wire [0:0] redist8_yAddr_uid35_fpSqrtTest_b_14_rdmux_s;
    reg [2:0] redist8_yAddr_uid35_fpSqrtTest_b_14_rdmux_q;
    reg [2:0] redist8_yAddr_uid35_fpSqrtTest_b_14_wraddr_q;
    wire [2:0] redist8_yAddr_uid35_fpSqrtTest_b_14_mem_last_q;
    wire [0:0] redist8_yAddr_uid35_fpSqrtTest_b_14_cmp_q;
    reg [0:0] redist8_yAddr_uid35_fpSqrtTest_b_14_cmpReg_q;
    wire [0:0] redist8_yAddr_uid35_fpSqrtTest_b_14_notEnable_q;
    wire [0:0] redist8_yAddr_uid35_fpSqrtTest_b_14_nor_q;
    (* preserve_syn_only *) reg [0:0] redist8_yAddr_uid35_fpSqrtTest_b_14_sticky_ena_q;
    wire [0:0] redist8_yAddr_uid35_fpSqrtTest_b_14_enaAnd_q;
    wire redist12_frac_x_uid12_fpSqrtTest_b_23_mem_reset0;
    wire [22:0] redist12_frac_x_uid12_fpSqrtTest_b_23_mem_ia;
    wire [4:0] redist12_frac_x_uid12_fpSqrtTest_b_23_mem_aa;
    wire [4:0] redist12_frac_x_uid12_fpSqrtTest_b_23_mem_ab;
    wire [22:0] redist12_frac_x_uid12_fpSqrtTest_b_23_mem_iq;
    wire [22:0] redist12_frac_x_uid12_fpSqrtTest_b_23_mem_q;
    wire redist12_frac_x_uid12_fpSqrtTest_b_23_mem_enaOr_rst;
    wire [4:0] redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt_q;
    (* preserve_syn_only *) reg [4:0] redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt_i;
    (* preserve_syn_only *) reg redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt_eq;
    wire [0:0] redist12_frac_x_uid12_fpSqrtTest_b_23_rdmux_s;
    reg [4:0] redist12_frac_x_uid12_fpSqrtTest_b_23_rdmux_q;
    reg [4:0] redist12_frac_x_uid12_fpSqrtTest_b_23_wraddr_q;
    wire [5:0] redist12_frac_x_uid12_fpSqrtTest_b_23_mem_last_q;
    wire [5:0] redist12_frac_x_uid12_fpSqrtTest_b_23_cmp_b;
    wire [0:0] redist12_frac_x_uid12_fpSqrtTest_b_23_cmp_q;
    reg [0:0] redist12_frac_x_uid12_fpSqrtTest_b_23_cmpReg_q;
    wire [0:0] redist12_frac_x_uid12_fpSqrtTest_b_23_notEnable_q;
    wire [0:0] redist12_frac_x_uid12_fpSqrtTest_b_23_nor_q;
    (* preserve_syn_only *) reg [0:0] redist12_frac_x_uid12_fpSqrtTest_b_23_sticky_ena_q;
    wire [0:0] redist12_frac_x_uid12_fpSqrtTest_b_23_enaAnd_q;
    wire redist14_expX_uid6_fpSqrtTest_b_23_mem_reset0;
    wire [7:0] redist14_expX_uid6_fpSqrtTest_b_23_mem_ia;
    wire [4:0] redist14_expX_uid6_fpSqrtTest_b_23_mem_aa;
    wire [4:0] redist14_expX_uid6_fpSqrtTest_b_23_mem_ab;
    wire [7:0] redist14_expX_uid6_fpSqrtTest_b_23_mem_iq;
    wire [7:0] redist14_expX_uid6_fpSqrtTest_b_23_mem_q;
    wire redist14_expX_uid6_fpSqrtTest_b_23_mem_enaOr_rst;
    wire [4:0] redist14_expX_uid6_fpSqrtTest_b_23_rdcnt_q;
    (* preserve_syn_only *) reg [4:0] redist14_expX_uid6_fpSqrtTest_b_23_rdcnt_i;
    (* preserve_syn_only *) reg redist14_expX_uid6_fpSqrtTest_b_23_rdcnt_eq;
    wire [0:0] redist14_expX_uid6_fpSqrtTest_b_23_rdmux_s;
    reg [4:0] redist14_expX_uid6_fpSqrtTest_b_23_rdmux_q;
    reg [4:0] redist14_expX_uid6_fpSqrtTest_b_23_wraddr_q;
    wire [5:0] redist14_expX_uid6_fpSqrtTest_b_23_mem_last_q;
    wire [5:0] redist14_expX_uid6_fpSqrtTest_b_23_cmp_b;
    wire [0:0] redist14_expX_uid6_fpSqrtTest_b_23_cmp_q;
    reg [0:0] redist14_expX_uid6_fpSqrtTest_b_23_cmpReg_q;
    wire [0:0] redist14_expX_uid6_fpSqrtTest_b_23_notEnable_q;
    wire [0:0] redist14_expX_uid6_fpSqrtTest_b_23_nor_q;
    (* preserve_syn_only *) reg [0:0] redist14_expX_uid6_fpSqrtTest_b_23_sticky_ena_q;
    wire [0:0] redist14_expX_uid6_fpSqrtTest_b_23_enaAnd_q;


    // signX_uid7_fpSqrtTest(BITSELECT,6)@0
    assign signX_uid7_fpSqrtTest_b = a[31:31];

    // redist13_signX_uid7_fpSqrtTest_b_24(DELAY,138)
    dspba_delay_ver #( .width(1), .depth(24), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    redist13_signX_uid7_fpSqrtTest_b_24 ( .xin(signX_uid7_fpSqrtTest_b), .xout(redist13_signX_uid7_fpSqrtTest_b_24_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // cstAllZWE_uid10_fpSqrtTest(CONSTANT,9)
    assign cstAllZWE_uid10_fpSqrtTest_q = 8'b00000000;

    // redist14_expX_uid6_fpSqrtTest_b_23_notEnable(LOGICAL,205)
    assign redist14_expX_uid6_fpSqrtTest_b_23_notEnable_q = ~ (en);

    // redist14_expX_uid6_fpSqrtTest_b_23_nor(LOGICAL,206)
    assign redist14_expX_uid6_fpSqrtTest_b_23_nor_q = ~ (redist14_expX_uid6_fpSqrtTest_b_23_notEnable_q | redist14_expX_uid6_fpSqrtTest_b_23_sticky_ena_q);

    // redist14_expX_uid6_fpSqrtTest_b_23_mem_last(CONSTANT,202)
    assign redist14_expX_uid6_fpSqrtTest_b_23_mem_last_q = 6'b010100;

    // redist14_expX_uid6_fpSqrtTest_b_23_cmp(LOGICAL,203)
    assign redist14_expX_uid6_fpSqrtTest_b_23_cmp_b = {1'b0, redist14_expX_uid6_fpSqrtTest_b_23_rdmux_q};
    assign redist14_expX_uid6_fpSqrtTest_b_23_cmp_q = redist14_expX_uid6_fpSqrtTest_b_23_mem_last_q == redist14_expX_uid6_fpSqrtTest_b_23_cmp_b ? 1'b1 : 1'b0;

    // redist14_expX_uid6_fpSqrtTest_b_23_cmpReg(REG,204)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist14_expX_uid6_fpSqrtTest_b_23_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist14_expX_uid6_fpSqrtTest_b_23_cmpReg_q <= redist14_expX_uid6_fpSqrtTest_b_23_cmp_q;
        end
    end

    // redist14_expX_uid6_fpSqrtTest_b_23_sticky_ena(REG,207)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist14_expX_uid6_fpSqrtTest_b_23_sticky_ena_q <= 1'b0;
        end
        else if (redist14_expX_uid6_fpSqrtTest_b_23_nor_q == 1'b1)
        begin
            redist14_expX_uid6_fpSqrtTest_b_23_sticky_ena_q <= redist14_expX_uid6_fpSqrtTest_b_23_cmpReg_q;
        end
    end

    // redist14_expX_uid6_fpSqrtTest_b_23_enaAnd(LOGICAL,208)
    assign redist14_expX_uid6_fpSqrtTest_b_23_enaAnd_q = redist14_expX_uid6_fpSqrtTest_b_23_sticky_ena_q & en;

    // redist14_expX_uid6_fpSqrtTest_b_23_rdcnt(COUNTER,199)
    // low=0, high=21, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist14_expX_uid6_fpSqrtTest_b_23_rdcnt_i <= 5'd0;
            redist14_expX_uid6_fpSqrtTest_b_23_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist14_expX_uid6_fpSqrtTest_b_23_rdcnt_i == 5'd20)
            begin
                redist14_expX_uid6_fpSqrtTest_b_23_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist14_expX_uid6_fpSqrtTest_b_23_rdcnt_eq <= 1'b0;
            end
            if (redist14_expX_uid6_fpSqrtTest_b_23_rdcnt_eq == 1'b1)
            begin
                redist14_expX_uid6_fpSqrtTest_b_23_rdcnt_i <= $unsigned(redist14_expX_uid6_fpSqrtTest_b_23_rdcnt_i) + $unsigned(5'd11);
            end
            else
            begin
                redist14_expX_uid6_fpSqrtTest_b_23_rdcnt_i <= $unsigned(redist14_expX_uid6_fpSqrtTest_b_23_rdcnt_i) + $unsigned(5'd1);
            end
        end
    end
    assign redist14_expX_uid6_fpSqrtTest_b_23_rdcnt_q = redist14_expX_uid6_fpSqrtTest_b_23_rdcnt_i[4:0];

    // redist14_expX_uid6_fpSqrtTest_b_23_rdmux(MUX,200)
    assign redist14_expX_uid6_fpSqrtTest_b_23_rdmux_s = en;
    always @(redist14_expX_uid6_fpSqrtTest_b_23_rdmux_s or redist14_expX_uid6_fpSqrtTest_b_23_wraddr_q or redist14_expX_uid6_fpSqrtTest_b_23_rdcnt_q)
    begin
        unique case (redist14_expX_uid6_fpSqrtTest_b_23_rdmux_s)
            1'b0 : redist14_expX_uid6_fpSqrtTest_b_23_rdmux_q = redist14_expX_uid6_fpSqrtTest_b_23_wraddr_q;
            1'b1 : redist14_expX_uid6_fpSqrtTest_b_23_rdmux_q = redist14_expX_uid6_fpSqrtTest_b_23_rdcnt_q;
            default : redist14_expX_uid6_fpSqrtTest_b_23_rdmux_q = 5'b0;
        endcase
    end

    // VCC(CONSTANT,1)
    assign VCC_q = 1'b1;

    // expX_uid6_fpSqrtTest(BITSELECT,5)@0
    assign expX_uid6_fpSqrtTest_b = a[30:23];

    // redist14_expX_uid6_fpSqrtTest_b_23_wraddr(REG,201)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist14_expX_uid6_fpSqrtTest_b_23_wraddr_q <= 5'b10101;
        end
        else
        begin
            redist14_expX_uid6_fpSqrtTest_b_23_wraddr_q <= redist14_expX_uid6_fpSqrtTest_b_23_rdmux_q;
        end
    end

    // redist14_expX_uid6_fpSqrtTest_b_23_mem(DUALMEM,198)
    assign redist14_expX_uid6_fpSqrtTest_b_23_mem_ia = expX_uid6_fpSqrtTest_b;
    assign redist14_expX_uid6_fpSqrtTest_b_23_mem_aa = redist14_expX_uid6_fpSqrtTest_b_23_wraddr_q;
    assign redist14_expX_uid6_fpSqrtTest_b_23_mem_ab = redist14_expX_uid6_fpSqrtTest_b_23_rdmux_q;
    assign redist14_expX_uid6_fpSqrtTest_b_23_mem_reset0 = areset;
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
    ) redist14_expX_uid6_fpSqrtTest_b_23_mem_dmem (
        .clocken1(redist14_expX_uid6_fpSqrtTest_b_23_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist14_expX_uid6_fpSqrtTest_b_23_mem_reset0),
        .clock1(clk),
        .address_a(redist14_expX_uid6_fpSqrtTest_b_23_mem_aa),
        .data_a(redist14_expX_uid6_fpSqrtTest_b_23_mem_ia),
        .wren_a(en[0]),
        .address_b(redist14_expX_uid6_fpSqrtTest_b_23_mem_ab),
        .q_b(redist14_expX_uid6_fpSqrtTest_b_23_mem_iq),
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
    assign redist14_expX_uid6_fpSqrtTest_b_23_mem_q = redist14_expX_uid6_fpSqrtTest_b_23_mem_iq[7:0];
    assign redist14_expX_uid6_fpSqrtTest_b_23_mem_enaOr_rst = redist14_expX_uid6_fpSqrtTest_b_23_enaAnd_q[0] | redist14_expX_uid6_fpSqrtTest_b_23_mem_reset0;

    // excZ_x_uid13_fpSqrtTest(LOGICAL,12)@23 + 1
    assign excZ_x_uid13_fpSqrtTest_qi = redist14_expX_uid6_fpSqrtTest_b_23_mem_q == cstAllZWE_uid10_fpSqrtTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    excZ_x_uid13_fpSqrtTest_delay ( .xin(excZ_x_uid13_fpSqrtTest_qi), .xout(excZ_x_uid13_fpSqrtTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // negZero_uid85_fpSqrtTest(LOGICAL,84)@24 + 1
    assign negZero_uid85_fpSqrtTest_qi = excZ_x_uid13_fpSqrtTest_q & redist13_signX_uid7_fpSqrtTest_b_24_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    negZero_uid85_fpSqrtTest_delay ( .xin(negZero_uid85_fpSqrtTest_qi), .xout(negZero_uid85_fpSqrtTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // cstAllOWE_uid8_fpSqrtTest(CONSTANT,7)
    assign cstAllOWE_uid8_fpSqrtTest_q = 8'b11111111;

    // GND(CONSTANT,0)
    assign GND_q = 1'b0;

    // redist12_frac_x_uid12_fpSqrtTest_b_23_notEnable(LOGICAL,194)
    assign redist12_frac_x_uid12_fpSqrtTest_b_23_notEnable_q = ~ (en);

    // redist12_frac_x_uid12_fpSqrtTest_b_23_nor(LOGICAL,195)
    assign redist12_frac_x_uid12_fpSqrtTest_b_23_nor_q = ~ (redist12_frac_x_uid12_fpSqrtTest_b_23_notEnable_q | redist12_frac_x_uid12_fpSqrtTest_b_23_sticky_ena_q);

    // redist12_frac_x_uid12_fpSqrtTest_b_23_mem_last(CONSTANT,191)
    assign redist12_frac_x_uid12_fpSqrtTest_b_23_mem_last_q = 6'b010001;

    // redist12_frac_x_uid12_fpSqrtTest_b_23_cmp(LOGICAL,192)
    assign redist12_frac_x_uid12_fpSqrtTest_b_23_cmp_b = {1'b0, redist12_frac_x_uid12_fpSqrtTest_b_23_rdmux_q};
    assign redist12_frac_x_uid12_fpSqrtTest_b_23_cmp_q = redist12_frac_x_uid12_fpSqrtTest_b_23_mem_last_q == redist12_frac_x_uid12_fpSqrtTest_b_23_cmp_b ? 1'b1 : 1'b0;

    // redist12_frac_x_uid12_fpSqrtTest_b_23_cmpReg(REG,193)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist12_frac_x_uid12_fpSqrtTest_b_23_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist12_frac_x_uid12_fpSqrtTest_b_23_cmpReg_q <= redist12_frac_x_uid12_fpSqrtTest_b_23_cmp_q;
        end
    end

    // redist12_frac_x_uid12_fpSqrtTest_b_23_sticky_ena(REG,196)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist12_frac_x_uid12_fpSqrtTest_b_23_sticky_ena_q <= 1'b0;
        end
        else if (redist12_frac_x_uid12_fpSqrtTest_b_23_nor_q == 1'b1)
        begin
            redist12_frac_x_uid12_fpSqrtTest_b_23_sticky_ena_q <= redist12_frac_x_uid12_fpSqrtTest_b_23_cmpReg_q;
        end
    end

    // redist12_frac_x_uid12_fpSqrtTest_b_23_enaAnd(LOGICAL,197)
    assign redist12_frac_x_uid12_fpSqrtTest_b_23_enaAnd_q = redist12_frac_x_uid12_fpSqrtTest_b_23_sticky_ena_q & en;

    // redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt(COUNTER,188)
    // low=0, high=18, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt_i <= 5'd0;
            redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt_i == 5'd17)
            begin
                redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt_eq <= 1'b0;
            end
            if (redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt_eq == 1'b1)
            begin
                redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt_i <= $unsigned(redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt_i) + $unsigned(5'd14);
            end
            else
            begin
                redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt_i <= $unsigned(redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt_i) + $unsigned(5'd1);
            end
        end
    end
    assign redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt_q = redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt_i[4:0];

    // redist12_frac_x_uid12_fpSqrtTest_b_23_rdmux(MUX,189)
    assign redist12_frac_x_uid12_fpSqrtTest_b_23_rdmux_s = en;
    always @(redist12_frac_x_uid12_fpSqrtTest_b_23_rdmux_s or redist12_frac_x_uid12_fpSqrtTest_b_23_wraddr_q or redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt_q)
    begin
        unique case (redist12_frac_x_uid12_fpSqrtTest_b_23_rdmux_s)
            1'b0 : redist12_frac_x_uid12_fpSqrtTest_b_23_rdmux_q = redist12_frac_x_uid12_fpSqrtTest_b_23_wraddr_q;
            1'b1 : redist12_frac_x_uid12_fpSqrtTest_b_23_rdmux_q = redist12_frac_x_uid12_fpSqrtTest_b_23_rdcnt_q;
            default : redist12_frac_x_uid12_fpSqrtTest_b_23_rdmux_q = 5'b0;
        endcase
    end

    // frac_x_uid12_fpSqrtTest(BITSELECT,11)@0
    assign frac_x_uid12_fpSqrtTest_b = a[22:0];

    // redist11_frac_x_uid12_fpSqrtTest_b_3(DELAY,136)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist11_frac_x_uid12_fpSqrtTest_b_3_delay_0 <= '0;
            redist11_frac_x_uid12_fpSqrtTest_b_3_delay_1 <= '0;
            redist11_frac_x_uid12_fpSqrtTest_b_3_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist11_frac_x_uid12_fpSqrtTest_b_3_delay_0 <= frac_x_uid12_fpSqrtTest_b;
            redist11_frac_x_uid12_fpSqrtTest_b_3_delay_1 <= redist11_frac_x_uid12_fpSqrtTest_b_3_delay_0;
            redist11_frac_x_uid12_fpSqrtTest_b_3_q <= redist11_frac_x_uid12_fpSqrtTest_b_3_delay_1;
        end
    end

    // redist12_frac_x_uid12_fpSqrtTest_b_23_wraddr(REG,190)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist12_frac_x_uid12_fpSqrtTest_b_23_wraddr_q <= 5'b10010;
        end
        else
        begin
            redist12_frac_x_uid12_fpSqrtTest_b_23_wraddr_q <= redist12_frac_x_uid12_fpSqrtTest_b_23_rdmux_q;
        end
    end

    // redist12_frac_x_uid12_fpSqrtTest_b_23_mem(DUALMEM,187)
    assign redist12_frac_x_uid12_fpSqrtTest_b_23_mem_ia = redist11_frac_x_uid12_fpSqrtTest_b_3_q;
    assign redist12_frac_x_uid12_fpSqrtTest_b_23_mem_aa = redist12_frac_x_uid12_fpSqrtTest_b_23_wraddr_q;
    assign redist12_frac_x_uid12_fpSqrtTest_b_23_mem_ab = redist12_frac_x_uid12_fpSqrtTest_b_23_rdmux_q;
    assign redist12_frac_x_uid12_fpSqrtTest_b_23_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(23),
        .widthad_a(5),
        .numwords_a(19),
        .width_b(23),
        .widthad_b(5),
        .numwords_b(19),
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
    ) redist12_frac_x_uid12_fpSqrtTest_b_23_mem_dmem (
        .clocken1(redist12_frac_x_uid12_fpSqrtTest_b_23_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist12_frac_x_uid12_fpSqrtTest_b_23_mem_reset0),
        .clock1(clk),
        .address_a(redist12_frac_x_uid12_fpSqrtTest_b_23_mem_aa),
        .data_a(redist12_frac_x_uid12_fpSqrtTest_b_23_mem_ia),
        .wren_a(en[0]),
        .address_b(redist12_frac_x_uid12_fpSqrtTest_b_23_mem_ab),
        .q_b(redist12_frac_x_uid12_fpSqrtTest_b_23_mem_iq),
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
    assign redist12_frac_x_uid12_fpSqrtTest_b_23_mem_q = redist12_frac_x_uid12_fpSqrtTest_b_23_mem_iq[22:0];
    assign redist12_frac_x_uid12_fpSqrtTest_b_23_mem_enaOr_rst = redist12_frac_x_uid12_fpSqrtTest_b_23_enaAnd_q[0] | redist12_frac_x_uid12_fpSqrtTest_b_23_mem_reset0;

    // oFracX_uid44_fpSqrtTest(BITJOIN,43)@23
    assign oFracX_uid44_fpSqrtTest_q = {VCC_q, redist12_frac_x_uid12_fpSqrtTest_b_23_mem_q};

    // oFracXZ_mergedSignalTM_uid47_fpSqrtTest(BITJOIN,46)@23
    assign oFracXZ_mergedSignalTM_uid47_fpSqrtTest_q = {oFracX_uid44_fpSqrtTest_q, GND_q};

    // oFracXSignExt_mergedSignalTM_uid52_fpSqrtTest(BITJOIN,51)@23
    assign oFracXSignExt_mergedSignalTM_uid52_fpSqrtTest_q = {GND_q, oFracX_uid44_fpSqrtTest_q};

    // expX0PS_uid29_fpSqrtTest(BITSELECT,28)@0
    assign expX0PS_uid29_fpSqrtTest_in = expX_uid6_fpSqrtTest_b[0:0];
    assign expX0PS_uid29_fpSqrtTest_b = expX0PS_uid29_fpSqrtTest_in[0:0];

    // expOddSelect_uid30_fpSqrtTest(LOGICAL,29)@0
    assign expOddSelect_uid30_fpSqrtTest_q = ~ (expX0PS_uid29_fpSqrtTest_b);

    // redist10_expOddSelect_uid30_fpSqrtTest_q_23(DELAY,135)
    dspba_delay_ver #( .width(1), .depth(23), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    redist10_expOddSelect_uid30_fpSqrtTest_q_23 ( .xin(expOddSelect_uid30_fpSqrtTest_q), .xout(redist10_expOddSelect_uid30_fpSqrtTest_q_23_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // normalizedXForComp_uid54_fpSqrtTest(MUX,53)@23
    assign normalizedXForComp_uid54_fpSqrtTest_s = redist10_expOddSelect_uid30_fpSqrtTest_q_23_q;
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

    // updatedY_uid56_fpSqrtTest(BITJOIN,55)@23
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
        .outdata_sclr_a("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .power_up_uninitialized("FALSE"),
        .init_file("acl_fsqrt_memoryC2_uid94_sqrtTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Stratix 10")
    ) memoryC2_uid94_sqrtTables_lutmem_dmem (
        .clocken0(en[0]),
        .sclr(memoryC2_uid94_sqrtTables_lutmem_reset0),
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
    assign memoryC2_uid94_sqrtTables_lutmem_r = memoryC2_uid94_sqrtTables_lutmem_ir[11:0];
    assign memoryC2_uid94_sqrtTables_lutmem_enaOr_rst = en[0] | memoryC2_uid94_sqrtTables_lutmem_reset0;

    // redist1_memoryC2_uid94_sqrtTables_lutmem_r_1(DELAY,126)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist1_memoryC2_uid94_sqrtTables_lutmem_r_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist1_memoryC2_uid94_sqrtTables_lutmem_r_1_q <= memoryC2_uid94_sqrtTables_lutmem_r;
        end
    end

    // yForPe_uid36_fpSqrtTest(BITSELECT,35)@3
    assign yForPe_uid36_fpSqrtTest_in = redist11_frac_x_uid12_fpSqrtTest_b_3_q[15:0];
    assign yForPe_uid36_fpSqrtTest_b = yForPe_uid36_fpSqrtTest_in[15:0];

    // yT1_uid100_invPolyEval(BITSELECT,99)@3
    assign yT1_uid100_invPolyEval_b = yForPe_uid36_fpSqrtTest_b[15:4];

    // prodXY_uid113_pT1_uid101_invPolyEval_cma(CHAINMULTADD,122)@3 + 5
    // out q@9
    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_reset = areset;
    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_ena0 = en[0] | prodXY_uid113_pT1_uid101_invPolyEval_cma_reset;
    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_ena1 = prodXY_uid113_pT1_uid101_invPolyEval_cma_ena0;
    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_ena2 = prodXY_uid113_pT1_uid101_invPolyEval_cma_ena0;
    always @ (posedge clk)
    begin
        if (0)
        begin
        end
        else
        begin
            if (en == 1'b1)
            begin
                prodXY_uid113_pT1_uid101_invPolyEval_cma_ah[0] <= yT1_uid100_invPolyEval_b;
                prodXY_uid113_pT1_uid101_invPolyEval_cma_ch[0] <= redist1_memoryC2_uid94_sqrtTables_lutmem_r_1_q;
            end
        end
    end

    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_a0 = prodXY_uid113_pT1_uid101_invPolyEval_cma_ah[0];
    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_c0 = prodXY_uid113_pT1_uid101_invPolyEval_cma_ch[0];
    fourteennm_mac #(
        .operation_mode("m18x18_full"),
        .clear_type("sclr"),
        .ay_scan_in_clock("0"),
        .ay_scan_in_width(12),
        .ax_clock("0"),
        .ax_width(12),
        .signed_may("false"),
        .signed_max("true"),
        .input_pipeline_clock("2"),
        .second_pipeline_clock("2"),
        .output_clock("1"),
        .result_a_width(24)
    ) prodXY_uid113_pT1_uid101_invPolyEval_cma_DSP0 (
        .clk({clk,clk,clk}),
        .ena({ prodXY_uid113_pT1_uid101_invPolyEval_cma_ena2, prodXY_uid113_pT1_uid101_invPolyEval_cma_ena1, prodXY_uid113_pT1_uid101_invPolyEval_cma_ena0 }),
        .clr({ prodXY_uid113_pT1_uid101_invPolyEval_cma_reset, prodXY_uid113_pT1_uid101_invPolyEval_cma_reset }),
        .ay(prodXY_uid113_pT1_uid101_invPolyEval_cma_a0),
        .ax(prodXY_uid113_pT1_uid101_invPolyEval_cma_c0),
        .resulta(prodXY_uid113_pT1_uid101_invPolyEval_cma_s0),
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
    dspba_delay_ver #( .width(24), .depth(1), .reset_kind("NONE"), .phase(0), .modulus(1) )
    prodXY_uid113_pT1_uid101_invPolyEval_cma_delay ( .xin(prodXY_uid113_pT1_uid101_invPolyEval_cma_s0), .xout(prodXY_uid113_pT1_uid101_invPolyEval_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign prodXY_uid113_pT1_uid101_invPolyEval_cma_q = prodXY_uid113_pT1_uid101_invPolyEval_cma_qq[23:0];

    // osig_uid114_pT1_uid101_invPolyEval(BITSELECT,113)@9
    assign osig_uid114_pT1_uid101_invPolyEval_b = prodXY_uid113_pT1_uid101_invPolyEval_cma_q[23:11];

    // highBBits_uid103_invPolyEval(BITSELECT,102)@9
    assign highBBits_uid103_invPolyEval_b = osig_uid114_pT1_uid101_invPolyEval_b[12:1];

    // redist7_yAddr_uid35_fpSqrtTest_b_7_notEnable(LOGICAL,171)
    assign redist7_yAddr_uid35_fpSqrtTest_b_7_notEnable_q = ~ (en);

    // redist7_yAddr_uid35_fpSqrtTest_b_7_nor(LOGICAL,172)
    assign redist7_yAddr_uid35_fpSqrtTest_b_7_nor_q = ~ (redist7_yAddr_uid35_fpSqrtTest_b_7_notEnable_q | redist7_yAddr_uid35_fpSqrtTest_b_7_sticky_ena_q);

    // redist7_yAddr_uid35_fpSqrtTest_b_7_mem_last(CONSTANT,168)
    assign redist7_yAddr_uid35_fpSqrtTest_b_7_mem_last_q = 3'b011;

    // redist7_yAddr_uid35_fpSqrtTest_b_7_cmp(LOGICAL,169)
    assign redist7_yAddr_uid35_fpSqrtTest_b_7_cmp_q = redist7_yAddr_uid35_fpSqrtTest_b_7_mem_last_q == redist7_yAddr_uid35_fpSqrtTest_b_7_rdmux_q ? 1'b1 : 1'b0;

    // redist7_yAddr_uid35_fpSqrtTest_b_7_cmpReg(REG,170)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist7_yAddr_uid35_fpSqrtTest_b_7_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist7_yAddr_uid35_fpSqrtTest_b_7_cmpReg_q <= redist7_yAddr_uid35_fpSqrtTest_b_7_cmp_q;
        end
    end

    // redist7_yAddr_uid35_fpSqrtTest_b_7_sticky_ena(REG,173)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist7_yAddr_uid35_fpSqrtTest_b_7_sticky_ena_q <= 1'b0;
        end
        else if (redist7_yAddr_uid35_fpSqrtTest_b_7_nor_q == 1'b1)
        begin
            redist7_yAddr_uid35_fpSqrtTest_b_7_sticky_ena_q <= redist7_yAddr_uid35_fpSqrtTest_b_7_cmpReg_q;
        end
    end

    // redist7_yAddr_uid35_fpSqrtTest_b_7_enaAnd(LOGICAL,174)
    assign redist7_yAddr_uid35_fpSqrtTest_b_7_enaAnd_q = redist7_yAddr_uid35_fpSqrtTest_b_7_sticky_ena_q & en;

    // redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt(COUNTER,165)
    // low=0, high=4, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt_i <= 3'd0;
            redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt_i == 3'd3)
            begin
                redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt_eq <= 1'b0;
            end
            if (redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt_eq == 1'b1)
            begin
                redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt_i <= $unsigned(redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt_i) + $unsigned(3'd4);
            end
            else
            begin
                redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt_i <= $unsigned(redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt_i) + $unsigned(3'd1);
            end
        end
    end
    assign redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt_q = redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt_i[2:0];

    // redist7_yAddr_uid35_fpSqrtTest_b_7_rdmux(MUX,166)
    assign redist7_yAddr_uid35_fpSqrtTest_b_7_rdmux_s = en;
    always @(redist7_yAddr_uid35_fpSqrtTest_b_7_rdmux_s or redist7_yAddr_uid35_fpSqrtTest_b_7_wraddr_q or redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt_q)
    begin
        unique case (redist7_yAddr_uid35_fpSqrtTest_b_7_rdmux_s)
            1'b0 : redist7_yAddr_uid35_fpSqrtTest_b_7_rdmux_q = redist7_yAddr_uid35_fpSqrtTest_b_7_wraddr_q;
            1'b1 : redist7_yAddr_uid35_fpSqrtTest_b_7_rdmux_q = redist7_yAddr_uid35_fpSqrtTest_b_7_rdcnt_q;
            default : redist7_yAddr_uid35_fpSqrtTest_b_7_rdmux_q = 3'b0;
        endcase
    end

    // redist7_yAddr_uid35_fpSqrtTest_b_7_wraddr(REG,167)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist7_yAddr_uid35_fpSqrtTest_b_7_wraddr_q <= 3'b100;
        end
        else
        begin
            redist7_yAddr_uid35_fpSqrtTest_b_7_wraddr_q <= redist7_yAddr_uid35_fpSqrtTest_b_7_rdmux_q;
        end
    end

    // redist7_yAddr_uid35_fpSqrtTest_b_7_mem(DUALMEM,164)
    assign redist7_yAddr_uid35_fpSqrtTest_b_7_mem_ia = yAddr_uid35_fpSqrtTest_b;
    assign redist7_yAddr_uid35_fpSqrtTest_b_7_mem_aa = redist7_yAddr_uid35_fpSqrtTest_b_7_wraddr_q;
    assign redist7_yAddr_uid35_fpSqrtTest_b_7_mem_ab = redist7_yAddr_uid35_fpSqrtTest_b_7_rdmux_q;
    assign redist7_yAddr_uid35_fpSqrtTest_b_7_mem_reset0 = areset;
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
    ) redist7_yAddr_uid35_fpSqrtTest_b_7_mem_dmem (
        .clocken1(redist7_yAddr_uid35_fpSqrtTest_b_7_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist7_yAddr_uid35_fpSqrtTest_b_7_mem_reset0),
        .clock1(clk),
        .address_a(redist7_yAddr_uid35_fpSqrtTest_b_7_mem_aa),
        .data_a(redist7_yAddr_uid35_fpSqrtTest_b_7_mem_ia),
        .wren_a(en[0]),
        .address_b(redist7_yAddr_uid35_fpSqrtTest_b_7_mem_ab),
        .q_b(redist7_yAddr_uid35_fpSqrtTest_b_7_mem_iq),
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
    assign redist7_yAddr_uid35_fpSqrtTest_b_7_mem_q = redist7_yAddr_uid35_fpSqrtTest_b_7_mem_iq[7:0];
    assign redist7_yAddr_uid35_fpSqrtTest_b_7_mem_enaOr_rst = redist7_yAddr_uid35_fpSqrtTest_b_7_enaAnd_q[0] | redist7_yAddr_uid35_fpSqrtTest_b_7_mem_reset0;

    // redist7_yAddr_uid35_fpSqrtTest_b_7_outputreg0(DELAY,163)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist7_yAddr_uid35_fpSqrtTest_b_7_outputreg0_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist7_yAddr_uid35_fpSqrtTest_b_7_outputreg0_q <= redist7_yAddr_uid35_fpSqrtTest_b_7_mem_q;
        end
    end

    // memoryC1_uid91_sqrtTables_lutmem(DUALMEM,119)@7 + 2
    // in j@20000000
    assign memoryC1_uid91_sqrtTables_lutmem_aa = redist7_yAddr_uid35_fpSqrtTest_b_7_outputreg0_q;
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
        .outdata_sclr_a("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .power_up_uninitialized("FALSE"),
        .init_file("acl_fsqrt_memoryC1_uid91_sqrtTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Stratix 10")
    ) memoryC1_uid91_sqrtTables_lutmem_dmem (
        .clocken0(en[0]),
        .sclr(memoryC1_uid91_sqrtTables_lutmem_reset0),
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
    assign memoryC1_uid91_sqrtTables_lutmem_r = memoryC1_uid91_sqrtTables_lutmem_ir[20:0];
    assign memoryC1_uid91_sqrtTables_lutmem_enaOr_rst = en[0] | memoryC1_uid91_sqrtTables_lutmem_reset0;

    // s1sumAHighB_uid104_invPolyEval(ADD,103)@9 + 1
    assign s1sumAHighB_uid104_invPolyEval_a = {{1{memoryC1_uid91_sqrtTables_lutmem_r[20]}}, memoryC1_uid91_sqrtTables_lutmem_r};
    assign s1sumAHighB_uid104_invPolyEval_b = {{10{highBBits_uid103_invPolyEval_b[11]}}, highBBits_uid103_invPolyEval_b};
    always @ (posedge clk)
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

    // lowRangeB_uid102_invPolyEval(BITSELECT,101)@9
    assign lowRangeB_uid102_invPolyEval_in = osig_uid114_pT1_uid101_invPolyEval_b[0:0];
    assign lowRangeB_uid102_invPolyEval_b = lowRangeB_uid102_invPolyEval_in[0:0];

    // redist2_lowRangeB_uid102_invPolyEval_b_1(DELAY,127)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist2_lowRangeB_uid102_invPolyEval_b_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist2_lowRangeB_uid102_invPolyEval_b_1_q <= lowRangeB_uid102_invPolyEval_b;
        end
    end

    // s1_uid105_invPolyEval(BITJOIN,104)@10
    assign s1_uid105_invPolyEval_q = {s1sumAHighB_uid104_invPolyEval_q, redist2_lowRangeB_uid102_invPolyEval_b_1_q};

    // redist6_yForPe_uid36_fpSqrtTest_b_7_notEnable(LOGICAL,159)
    assign redist6_yForPe_uid36_fpSqrtTest_b_7_notEnable_q = ~ (en);

    // redist6_yForPe_uid36_fpSqrtTest_b_7_nor(LOGICAL,160)
    assign redist6_yForPe_uid36_fpSqrtTest_b_7_nor_q = ~ (redist6_yForPe_uid36_fpSqrtTest_b_7_notEnable_q | redist6_yForPe_uid36_fpSqrtTest_b_7_sticky_ena_q);

    // redist6_yForPe_uid36_fpSqrtTest_b_7_mem_last(CONSTANT,156)
    assign redist6_yForPe_uid36_fpSqrtTest_b_7_mem_last_q = 3'b011;

    // redist6_yForPe_uid36_fpSqrtTest_b_7_cmp(LOGICAL,157)
    assign redist6_yForPe_uid36_fpSqrtTest_b_7_cmp_q = redist6_yForPe_uid36_fpSqrtTest_b_7_mem_last_q == redist6_yForPe_uid36_fpSqrtTest_b_7_rdmux_q ? 1'b1 : 1'b0;

    // redist6_yForPe_uid36_fpSqrtTest_b_7_cmpReg(REG,158)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist6_yForPe_uid36_fpSqrtTest_b_7_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist6_yForPe_uid36_fpSqrtTest_b_7_cmpReg_q <= redist6_yForPe_uid36_fpSqrtTest_b_7_cmp_q;
        end
    end

    // redist6_yForPe_uid36_fpSqrtTest_b_7_sticky_ena(REG,161)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist6_yForPe_uid36_fpSqrtTest_b_7_sticky_ena_q <= 1'b0;
        end
        else if (redist6_yForPe_uid36_fpSqrtTest_b_7_nor_q == 1'b1)
        begin
            redist6_yForPe_uid36_fpSqrtTest_b_7_sticky_ena_q <= redist6_yForPe_uid36_fpSqrtTest_b_7_cmpReg_q;
        end
    end

    // redist6_yForPe_uid36_fpSqrtTest_b_7_enaAnd(LOGICAL,162)
    assign redist6_yForPe_uid36_fpSqrtTest_b_7_enaAnd_q = redist6_yForPe_uid36_fpSqrtTest_b_7_sticky_ena_q & en;

    // redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt(COUNTER,153)
    // low=0, high=4, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt_i <= 3'd0;
            redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt_i == 3'd3)
            begin
                redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt_eq <= 1'b0;
            end
            if (redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt_eq == 1'b1)
            begin
                redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt_i <= $unsigned(redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt_i) + $unsigned(3'd4);
            end
            else
            begin
                redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt_i <= $unsigned(redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt_i) + $unsigned(3'd1);
            end
        end
    end
    assign redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt_q = redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt_i[2:0];

    // redist6_yForPe_uid36_fpSqrtTest_b_7_rdmux(MUX,154)
    assign redist6_yForPe_uid36_fpSqrtTest_b_7_rdmux_s = en;
    always @(redist6_yForPe_uid36_fpSqrtTest_b_7_rdmux_s or redist6_yForPe_uid36_fpSqrtTest_b_7_wraddr_q or redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt_q)
    begin
        unique case (redist6_yForPe_uid36_fpSqrtTest_b_7_rdmux_s)
            1'b0 : redist6_yForPe_uid36_fpSqrtTest_b_7_rdmux_q = redist6_yForPe_uid36_fpSqrtTest_b_7_wraddr_q;
            1'b1 : redist6_yForPe_uid36_fpSqrtTest_b_7_rdmux_q = redist6_yForPe_uid36_fpSqrtTest_b_7_rdcnt_q;
            default : redist6_yForPe_uid36_fpSqrtTest_b_7_rdmux_q = 3'b0;
        endcase
    end

    // redist6_yForPe_uid36_fpSqrtTest_b_7_wraddr(REG,155)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist6_yForPe_uid36_fpSqrtTest_b_7_wraddr_q <= 3'b100;
        end
        else
        begin
            redist6_yForPe_uid36_fpSqrtTest_b_7_wraddr_q <= redist6_yForPe_uid36_fpSqrtTest_b_7_rdmux_q;
        end
    end

    // redist6_yForPe_uid36_fpSqrtTest_b_7_mem(DUALMEM,152)
    assign redist6_yForPe_uid36_fpSqrtTest_b_7_mem_ia = yForPe_uid36_fpSqrtTest_b;
    assign redist6_yForPe_uid36_fpSqrtTest_b_7_mem_aa = redist6_yForPe_uid36_fpSqrtTest_b_7_wraddr_q;
    assign redist6_yForPe_uid36_fpSqrtTest_b_7_mem_ab = redist6_yForPe_uid36_fpSqrtTest_b_7_rdmux_q;
    assign redist6_yForPe_uid36_fpSqrtTest_b_7_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(16),
        .widthad_a(3),
        .numwords_a(5),
        .width_b(16),
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
    ) redist6_yForPe_uid36_fpSqrtTest_b_7_mem_dmem (
        .clocken1(redist6_yForPe_uid36_fpSqrtTest_b_7_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist6_yForPe_uid36_fpSqrtTest_b_7_mem_reset0),
        .clock1(clk),
        .address_a(redist6_yForPe_uid36_fpSqrtTest_b_7_mem_aa),
        .data_a(redist6_yForPe_uid36_fpSqrtTest_b_7_mem_ia),
        .wren_a(en[0]),
        .address_b(redist6_yForPe_uid36_fpSqrtTest_b_7_mem_ab),
        .q_b(redist6_yForPe_uid36_fpSqrtTest_b_7_mem_iq),
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
    assign redist6_yForPe_uid36_fpSqrtTest_b_7_mem_q = redist6_yForPe_uid36_fpSqrtTest_b_7_mem_iq[15:0];
    assign redist6_yForPe_uid36_fpSqrtTest_b_7_mem_enaOr_rst = redist6_yForPe_uid36_fpSqrtTest_b_7_enaAnd_q[0] | redist6_yForPe_uid36_fpSqrtTest_b_7_mem_reset0;

    // redist6_yForPe_uid36_fpSqrtTest_b_7_outputreg0(DELAY,151)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist6_yForPe_uid36_fpSqrtTest_b_7_outputreg0_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist6_yForPe_uid36_fpSqrtTest_b_7_outputreg0_q <= redist6_yForPe_uid36_fpSqrtTest_b_7_mem_q;
        end
    end

    // prodXY_uid116_pT2_uid107_invPolyEval_cma(CHAINMULTADD,123)@10 + 5
    // out q@16
    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_reset = areset;
    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_ena0 = en[0] | prodXY_uid116_pT2_uid107_invPolyEval_cma_reset;
    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_ena1 = prodXY_uid116_pT2_uid107_invPolyEval_cma_ena0;
    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_ena2 = prodXY_uid116_pT2_uid107_invPolyEval_cma_ena0;
    always @ (posedge clk)
    begin
        if (0)
        begin
        end
        else
        begin
            if (en == 1'b1)
            begin
                prodXY_uid116_pT2_uid107_invPolyEval_cma_ah[0] <= redist6_yForPe_uid36_fpSqrtTest_b_7_outputreg0_q;
                prodXY_uid116_pT2_uid107_invPolyEval_cma_ch[0] <= s1_uid105_invPolyEval_q;
            end
        end
    end

    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_a0 = prodXY_uid116_pT2_uid107_invPolyEval_cma_ah[0];
    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_c0 = prodXY_uid116_pT2_uid107_invPolyEval_cma_ch[0];
    fourteennm_mac #(
        .operation_mode("m27x27"),
        .clear_type("sclr"),
        .use_chainadder("false"),
        .ay_scan_in_clock("0"),
        .ay_scan_in_width(16),
        .ax_clock("0"),
        .ax_width(23),
        .signed_may("false"),
        .signed_max("true"),
        .input_pipeline_clock("2"),
        .second_pipeline_clock("2"),
        .output_clock("1"),
        .result_a_width(39)
    ) prodXY_uid116_pT2_uid107_invPolyEval_cma_DSP0 (
        .clk({clk,clk,clk}),
        .ena({ prodXY_uid116_pT2_uid107_invPolyEval_cma_ena2, prodXY_uid116_pT2_uid107_invPolyEval_cma_ena1, prodXY_uid116_pT2_uid107_invPolyEval_cma_ena0 }),
        .clr({ prodXY_uid116_pT2_uid107_invPolyEval_cma_reset, prodXY_uid116_pT2_uid107_invPolyEval_cma_reset }),
        .ay(prodXY_uid116_pT2_uid107_invPolyEval_cma_a0),
        .ax(prodXY_uid116_pT2_uid107_invPolyEval_cma_c0),
        .resulta(prodXY_uid116_pT2_uid107_invPolyEval_cma_s0),
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
    dspba_delay_ver #( .width(39), .depth(1), .reset_kind("NONE"), .phase(0), .modulus(1) )
    prodXY_uid116_pT2_uid107_invPolyEval_cma_delay ( .xin(prodXY_uid116_pT2_uid107_invPolyEval_cma_s0), .xout(prodXY_uid116_pT2_uid107_invPolyEval_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign prodXY_uid116_pT2_uid107_invPolyEval_cma_q = prodXY_uid116_pT2_uid107_invPolyEval_cma_qq[38:0];

    // osig_uid117_pT2_uid107_invPolyEval(BITSELECT,116)@16
    assign osig_uid117_pT2_uid107_invPolyEval_b = prodXY_uid116_pT2_uid107_invPolyEval_cma_q[38:15];

    // highBBits_uid109_invPolyEval(BITSELECT,108)@16
    assign highBBits_uid109_invPolyEval_b = osig_uid117_pT2_uid107_invPolyEval_b[23:2];

    // redist8_yAddr_uid35_fpSqrtTest_b_14_notEnable(LOGICAL,183)
    assign redist8_yAddr_uid35_fpSqrtTest_b_14_notEnable_q = ~ (en);

    // redist8_yAddr_uid35_fpSqrtTest_b_14_nor(LOGICAL,184)
    assign redist8_yAddr_uid35_fpSqrtTest_b_14_nor_q = ~ (redist8_yAddr_uid35_fpSqrtTest_b_14_notEnable_q | redist8_yAddr_uid35_fpSqrtTest_b_14_sticky_ena_q);

    // redist8_yAddr_uid35_fpSqrtTest_b_14_mem_last(CONSTANT,180)
    assign redist8_yAddr_uid35_fpSqrtTest_b_14_mem_last_q = 3'b011;

    // redist8_yAddr_uid35_fpSqrtTest_b_14_cmp(LOGICAL,181)
    assign redist8_yAddr_uid35_fpSqrtTest_b_14_cmp_q = redist8_yAddr_uid35_fpSqrtTest_b_14_mem_last_q == redist8_yAddr_uid35_fpSqrtTest_b_14_rdmux_q ? 1'b1 : 1'b0;

    // redist8_yAddr_uid35_fpSqrtTest_b_14_cmpReg(REG,182)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist8_yAddr_uid35_fpSqrtTest_b_14_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist8_yAddr_uid35_fpSqrtTest_b_14_cmpReg_q <= redist8_yAddr_uid35_fpSqrtTest_b_14_cmp_q;
        end
    end

    // redist8_yAddr_uid35_fpSqrtTest_b_14_sticky_ena(REG,185)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist8_yAddr_uid35_fpSqrtTest_b_14_sticky_ena_q <= 1'b0;
        end
        else if (redist8_yAddr_uid35_fpSqrtTest_b_14_nor_q == 1'b1)
        begin
            redist8_yAddr_uid35_fpSqrtTest_b_14_sticky_ena_q <= redist8_yAddr_uid35_fpSqrtTest_b_14_cmpReg_q;
        end
    end

    // redist8_yAddr_uid35_fpSqrtTest_b_14_enaAnd(LOGICAL,186)
    assign redist8_yAddr_uid35_fpSqrtTest_b_14_enaAnd_q = redist8_yAddr_uid35_fpSqrtTest_b_14_sticky_ena_q & en;

    // redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt(COUNTER,177)
    // low=0, high=4, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt_i <= 3'd0;
            redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt_i == 3'd3)
            begin
                redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt_eq <= 1'b0;
            end
            if (redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt_eq == 1'b1)
            begin
                redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt_i <= $unsigned(redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt_i) + $unsigned(3'd4);
            end
            else
            begin
                redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt_i <= $unsigned(redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt_i) + $unsigned(3'd1);
            end
        end
    end
    assign redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt_q = redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt_i[2:0];

    // redist8_yAddr_uid35_fpSqrtTest_b_14_rdmux(MUX,178)
    assign redist8_yAddr_uid35_fpSqrtTest_b_14_rdmux_s = en;
    always @(redist8_yAddr_uid35_fpSqrtTest_b_14_rdmux_s or redist8_yAddr_uid35_fpSqrtTest_b_14_wraddr_q or redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt_q)
    begin
        unique case (redist8_yAddr_uid35_fpSqrtTest_b_14_rdmux_s)
            1'b0 : redist8_yAddr_uid35_fpSqrtTest_b_14_rdmux_q = redist8_yAddr_uid35_fpSqrtTest_b_14_wraddr_q;
            1'b1 : redist8_yAddr_uid35_fpSqrtTest_b_14_rdmux_q = redist8_yAddr_uid35_fpSqrtTest_b_14_rdcnt_q;
            default : redist8_yAddr_uid35_fpSqrtTest_b_14_rdmux_q = 3'b0;
        endcase
    end

    // redist8_yAddr_uid35_fpSqrtTest_b_14_wraddr(REG,179)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist8_yAddr_uid35_fpSqrtTest_b_14_wraddr_q <= 3'b100;
        end
        else
        begin
            redist8_yAddr_uid35_fpSqrtTest_b_14_wraddr_q <= redist8_yAddr_uid35_fpSqrtTest_b_14_rdmux_q;
        end
    end

    // redist8_yAddr_uid35_fpSqrtTest_b_14_mem(DUALMEM,176)
    assign redist8_yAddr_uid35_fpSqrtTest_b_14_mem_ia = redist7_yAddr_uid35_fpSqrtTest_b_7_outputreg0_q;
    assign redist8_yAddr_uid35_fpSqrtTest_b_14_mem_aa = redist8_yAddr_uid35_fpSqrtTest_b_14_wraddr_q;
    assign redist8_yAddr_uid35_fpSqrtTest_b_14_mem_ab = redist8_yAddr_uid35_fpSqrtTest_b_14_rdmux_q;
    assign redist8_yAddr_uid35_fpSqrtTest_b_14_mem_reset0 = areset;
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
    ) redist8_yAddr_uid35_fpSqrtTest_b_14_mem_dmem (
        .clocken1(redist8_yAddr_uid35_fpSqrtTest_b_14_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist8_yAddr_uid35_fpSqrtTest_b_14_mem_reset0),
        .clock1(clk),
        .address_a(redist8_yAddr_uid35_fpSqrtTest_b_14_mem_aa),
        .data_a(redist8_yAddr_uid35_fpSqrtTest_b_14_mem_ia),
        .wren_a(en[0]),
        .address_b(redist8_yAddr_uid35_fpSqrtTest_b_14_mem_ab),
        .q_b(redist8_yAddr_uid35_fpSqrtTest_b_14_mem_iq),
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
    assign redist8_yAddr_uid35_fpSqrtTest_b_14_mem_q = redist8_yAddr_uid35_fpSqrtTest_b_14_mem_iq[7:0];
    assign redist8_yAddr_uid35_fpSqrtTest_b_14_mem_enaOr_rst = redist8_yAddr_uid35_fpSqrtTest_b_14_enaAnd_q[0] | redist8_yAddr_uid35_fpSqrtTest_b_14_mem_reset0;

    // redist8_yAddr_uid35_fpSqrtTest_b_14_outputreg0(DELAY,175)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist8_yAddr_uid35_fpSqrtTest_b_14_outputreg0_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist8_yAddr_uid35_fpSqrtTest_b_14_outputreg0_q <= redist8_yAddr_uid35_fpSqrtTest_b_14_mem_q;
        end
    end

    // memoryC0_uid88_sqrtTables_lutmem(DUALMEM,118)@14 + 2
    // in j@20000000
    assign memoryC0_uid88_sqrtTables_lutmem_aa = redist8_yAddr_uid35_fpSqrtTest_b_14_outputreg0_q;
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
        .outdata_sclr_a("SCLEAR"),
        .clock_enable_input_a("NORMAL"),
        .power_up_uninitialized("FALSE"),
        .init_file("acl_fsqrt_memoryC0_uid88_sqrtTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Stratix 10")
    ) memoryC0_uid88_sqrtTables_lutmem_dmem (
        .clocken0(en[0]),
        .sclr(memoryC0_uid88_sqrtTables_lutmem_reset0),
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
    assign memoryC0_uid88_sqrtTables_lutmem_r = memoryC0_uid88_sqrtTables_lutmem_ir[28:0];
    assign memoryC0_uid88_sqrtTables_lutmem_enaOr_rst = en[0] | memoryC0_uid88_sqrtTables_lutmem_reset0;

    // s2sumAHighB_uid110_invPolyEval(ADD,109)@16
    assign s2sumAHighB_uid110_invPolyEval_a = {{1{memoryC0_uid88_sqrtTables_lutmem_r[28]}}, memoryC0_uid88_sqrtTables_lutmem_r};
    assign s2sumAHighB_uid110_invPolyEval_b = {{8{highBBits_uid109_invPolyEval_b[21]}}, highBBits_uid109_invPolyEval_b};
    assign s2sumAHighB_uid110_invPolyEval_o = $signed(s2sumAHighB_uid110_invPolyEval_a) + $signed(s2sumAHighB_uid110_invPolyEval_b);
    assign s2sumAHighB_uid110_invPolyEval_q = s2sumAHighB_uid110_invPolyEval_o[29:0];

    // lowRangeB_uid108_invPolyEval(BITSELECT,107)@16
    assign lowRangeB_uid108_invPolyEval_in = osig_uid117_pT2_uid107_invPolyEval_b[1:0];
    assign lowRangeB_uid108_invPolyEval_b = lowRangeB_uid108_invPolyEval_in[1:0];

    // s2_uid111_invPolyEval(BITJOIN,110)@16
    assign s2_uid111_invPolyEval_q = {s2sumAHighB_uid110_invPolyEval_q, lowRangeB_uid108_invPolyEval_b};

    // fracRPreCR_uid39_fpSqrtTest(BITSELECT,38)@16
    assign fracRPreCR_uid39_fpSqrtTest_in = s2_uid111_invPolyEval_q[28:0];
    assign fracRPreCR_uid39_fpSqrtTest_b = fracRPreCR_uid39_fpSqrtTest_in[28:5];

    // redist3_fracRPreCR_uid39_fpSqrtTest_b_1(DELAY,128)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist3_fracRPreCR_uid39_fpSqrtTest_b_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist3_fracRPreCR_uid39_fpSqrtTest_b_1_q <= fracRPreCR_uid39_fpSqrtTest_b;
        end
    end

    // fracPaddingOne_uid41_fpSqrtTest(BITJOIN,40)@17
    assign fracPaddingOne_uid41_fpSqrtTest_q = {VCC_q, redist3_fracRPreCR_uid39_fpSqrtTest_b_1_q};

    // squaredResult_uid42_fpSqrtTest_cma(CHAINMULTADD,121)@17 + 5
    // out q@23
    assign squaredResult_uid42_fpSqrtTest_cma_reset = areset;
    assign squaredResult_uid42_fpSqrtTest_cma_ena0 = en[0] | squaredResult_uid42_fpSqrtTest_cma_reset;
    assign squaredResult_uid42_fpSqrtTest_cma_ena1 = squaredResult_uid42_fpSqrtTest_cma_ena0;
    assign squaredResult_uid42_fpSqrtTest_cma_ena2 = squaredResult_uid42_fpSqrtTest_cma_ena0;
    always @ (posedge clk)
    begin
        if (0)
        begin
        end
        else
        begin
            if (en == 1'b1)
            begin
                squaredResult_uid42_fpSqrtTest_cma_ah[0] <= fracPaddingOne_uid41_fpSqrtTest_q;
                squaredResult_uid42_fpSqrtTest_cma_ch[0] <= fracPaddingOne_uid41_fpSqrtTest_q;
            end
        end
    end

    assign squaredResult_uid42_fpSqrtTest_cma_a0 = squaredResult_uid42_fpSqrtTest_cma_ah[0];
    assign squaredResult_uid42_fpSqrtTest_cma_c0 = squaredResult_uid42_fpSqrtTest_cma_ch[0];
    fourteennm_mac #(
        .operation_mode("m27x27"),
        .clear_type("sclr"),
        .use_chainadder("false"),
        .ay_scan_in_clock("0"),
        .ay_scan_in_width(25),
        .ax_clock("0"),
        .ax_width(25),
        .signed_may("false"),
        .signed_max("false"),
        .input_pipeline_clock("2"),
        .second_pipeline_clock("2"),
        .output_clock("1"),
        .result_a_width(50)
    ) squaredResult_uid42_fpSqrtTest_cma_DSP0 (
        .clk({clk,clk,clk}),
        .ena({ squaredResult_uid42_fpSqrtTest_cma_ena2, squaredResult_uid42_fpSqrtTest_cma_ena1, squaredResult_uid42_fpSqrtTest_cma_ena0 }),
        .clr({ squaredResult_uid42_fpSqrtTest_cma_reset, squaredResult_uid42_fpSqrtTest_cma_reset }),
        .ay(squaredResult_uid42_fpSqrtTest_cma_a0),
        .ax(squaredResult_uid42_fpSqrtTest_cma_c0),
        .resulta(squaredResult_uid42_fpSqrtTest_cma_s0),
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
    dspba_delay_ver #( .width(50), .depth(1), .reset_kind("NONE"), .phase(0), .modulus(1) )
    squaredResult_uid42_fpSqrtTest_cma_delay ( .xin(squaredResult_uid42_fpSqrtTest_cma_s0), .xout(squaredResult_uid42_fpSqrtTest_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign squaredResult_uid42_fpSqrtTest_cma_q = squaredResult_uid42_fpSqrtTest_cma_qq[49:0];

    // squaredResultGTEIn_uid55_fpSqrtTest(COMPARE,56)@23 + 1
    assign squaredResultGTEIn_uid55_fpSqrtTest_a = {2'b00, squaredResult_uid42_fpSqrtTest_cma_q};
    assign squaredResultGTEIn_uid55_fpSqrtTest_b = {2'b00, updatedY_uid56_fpSqrtTest_q};
    always @ (posedge clk)
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

    // pLTOne_uid58_fpSqrtTest(LOGICAL,57)@24
    assign pLTOne_uid58_fpSqrtTest_q = ~ (squaredResultGTEIn_uid55_fpSqrtTest_n);

    // redist4_fracRPreCR_uid39_fpSqrtTest_b_8_notEnable(LOGICAL,147)
    assign redist4_fracRPreCR_uid39_fpSqrtTest_b_8_notEnable_q = ~ (en);

    // redist4_fracRPreCR_uid39_fpSqrtTest_b_8_nor(LOGICAL,148)
    assign redist4_fracRPreCR_uid39_fpSqrtTest_b_8_nor_q = ~ (redist4_fracRPreCR_uid39_fpSqrtTest_b_8_notEnable_q | redist4_fracRPreCR_uid39_fpSqrtTest_b_8_sticky_ena_q);

    // redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_last(CONSTANT,144)
    assign redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_last_q = 4'b0100;

    // redist4_fracRPreCR_uid39_fpSqrtTest_b_8_cmp(LOGICAL,145)
    assign redist4_fracRPreCR_uid39_fpSqrtTest_b_8_cmp_b = {1'b0, redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdmux_q};
    assign redist4_fracRPreCR_uid39_fpSqrtTest_b_8_cmp_q = redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_last_q == redist4_fracRPreCR_uid39_fpSqrtTest_b_8_cmp_b ? 1'b1 : 1'b0;

    // redist4_fracRPreCR_uid39_fpSqrtTest_b_8_cmpReg(REG,146)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist4_fracRPreCR_uid39_fpSqrtTest_b_8_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist4_fracRPreCR_uid39_fpSqrtTest_b_8_cmpReg_q <= redist4_fracRPreCR_uid39_fpSqrtTest_b_8_cmp_q;
        end
    end

    // redist4_fracRPreCR_uid39_fpSqrtTest_b_8_sticky_ena(REG,149)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist4_fracRPreCR_uid39_fpSqrtTest_b_8_sticky_ena_q <= 1'b0;
        end
        else if (redist4_fracRPreCR_uid39_fpSqrtTest_b_8_nor_q == 1'b1)
        begin
            redist4_fracRPreCR_uid39_fpSqrtTest_b_8_sticky_ena_q <= redist4_fracRPreCR_uid39_fpSqrtTest_b_8_cmpReg_q;
        end
    end

    // redist4_fracRPreCR_uid39_fpSqrtTest_b_8_enaAnd(LOGICAL,150)
    assign redist4_fracRPreCR_uid39_fpSqrtTest_b_8_enaAnd_q = redist4_fracRPreCR_uid39_fpSqrtTest_b_8_sticky_ena_q & en;

    // redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt(COUNTER,141)
    // low=0, high=5, step=1, init=0
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt_i <= 3'd0;
            redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt_i == 3'd4)
            begin
                redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt_eq <= 1'b0;
            end
            if (redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt_eq == 1'b1)
            begin
                redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt_i <= $unsigned(redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt_i) + $unsigned(3'd3);
            end
            else
            begin
                redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt_i <= $unsigned(redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt_i) + $unsigned(3'd1);
            end
        end
    end
    assign redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt_q = redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt_i[2:0];

    // redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdmux(MUX,142)
    assign redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdmux_s = en;
    always @(redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdmux_s or redist4_fracRPreCR_uid39_fpSqrtTest_b_8_wraddr_q or redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt_q)
    begin
        unique case (redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdmux_s)
            1'b0 : redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdmux_q = redist4_fracRPreCR_uid39_fpSqrtTest_b_8_wraddr_q;
            1'b1 : redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdmux_q = redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdcnt_q;
            default : redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdmux_q = 3'b0;
        endcase
    end

    // redist4_fracRPreCR_uid39_fpSqrtTest_b_8_wraddr(REG,143)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist4_fracRPreCR_uid39_fpSqrtTest_b_8_wraddr_q <= 3'b101;
        end
        else
        begin
            redist4_fracRPreCR_uid39_fpSqrtTest_b_8_wraddr_q <= redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdmux_q;
        end
    end

    // redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem(DUALMEM,140)
    assign redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_ia = redist3_fracRPreCR_uid39_fpSqrtTest_b_1_q;
    assign redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_aa = redist4_fracRPreCR_uid39_fpSqrtTest_b_8_wraddr_q;
    assign redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_ab = redist4_fracRPreCR_uid39_fpSqrtTest_b_8_rdmux_q;
    assign redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(24),
        .widthad_a(3),
        .numwords_a(6),
        .width_b(24),
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
    ) redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_dmem (
        .clocken1(redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_enaOr_rst),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .sclr(redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_reset0),
        .clock1(clk),
        .address_a(redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_aa),
        .data_a(redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_ia),
        .wren_a(en[0]),
        .address_b(redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_ab),
        .q_b(redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_iq),
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
    assign redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_q = redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_iq[23:0];
    assign redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_enaOr_rst = redist4_fracRPreCR_uid39_fpSqrtTest_b_8_enaAnd_q[0] | redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_reset0;

    // fxpSqrtResPostUpdateE_uid60_fpSqrtTest(ADD,59)@24
    assign fxpSqrtResPostUpdateE_uid60_fpSqrtTest_a = {1'b0, redist4_fracRPreCR_uid39_fpSqrtTest_b_8_mem_q};
    assign fxpSqrtResPostUpdateE_uid60_fpSqrtTest_b = {24'b000000000000000000000000, pLTOne_uid58_fpSqrtTest_q};
    assign fxpSqrtResPostUpdateE_uid60_fpSqrtTest_o = $unsigned(fxpSqrtResPostUpdateE_uid60_fpSqrtTest_a) + $unsigned(fxpSqrtResPostUpdateE_uid60_fpSqrtTest_b);
    assign fxpSqrtResPostUpdateE_uid60_fpSqrtTest_q = fxpSqrtResPostUpdateE_uid60_fpSqrtTest_o[24:0];

    // expUpdateCRU_uid61_fpSqrtTest_merged_bit_select(BITSELECT,124)@24
    assign expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_b = fxpSqrtResPostUpdateE_uid60_fpSqrtTest_q[24:24];
    assign expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_c = fxpSqrtResPostUpdateE_uid60_fpSqrtTest_q[23:1];

    // fracPENotOne_uid62_fpSqrtTest(LOGICAL,61)@24
    assign fracPENotOne_uid62_fpSqrtTest_q = ~ (redist5_expIncPEOnly_uid38_fpSqrtTest_b_8_q);

    // fracPENotOneAndCRRoundsExp_uid63_fpSqrtTest(LOGICAL,62)@24
    assign fracPENotOneAndCRRoundsExp_uid63_fpSqrtTest_q = fracPENotOne_uid62_fpSqrtTest_q & expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_b;

    // expIncPEOnly_uid38_fpSqrtTest(BITSELECT,37)@16
    assign expIncPEOnly_uid38_fpSqrtTest_in = s2_uid111_invPolyEval_q[30:0];
    assign expIncPEOnly_uid38_fpSqrtTest_b = expIncPEOnly_uid38_fpSqrtTest_in[30:30];

    // redist5_expIncPEOnly_uid38_fpSqrtTest_b_8(DELAY,130)
    dspba_delay_ver #( .width(1), .depth(8), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    redist5_expIncPEOnly_uid38_fpSqrtTest_b_8 ( .xin(expIncPEOnly_uid38_fpSqrtTest_b), .xout(redist5_expIncPEOnly_uid38_fpSqrtTest_b_8_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expInc_uid64_fpSqrtTest(LOGICAL,63)@24 + 1
    assign expInc_uid64_fpSqrtTest_qi = redist5_expIncPEOnly_uid38_fpSqrtTest_b_8_q | fracPENotOneAndCRRoundsExp_uid63_fpSqrtTest_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    expInc_uid64_fpSqrtTest_delay ( .xin(expInc_uid64_fpSqrtTest_qi), .xout(expInc_uid64_fpSqrtTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // sBiasM1_uid26_fpSqrtTest(CONSTANT,25)
    assign sBiasM1_uid26_fpSqrtTest_q = 8'b01111110;

    // expOddSig_uid27_fpSqrtTest(ADD,26)@23
    assign expOddSig_uid27_fpSqrtTest_a = {1'b0, redist14_expX_uid6_fpSqrtTest_b_23_mem_q};
    assign expOddSig_uid27_fpSqrtTest_b = {1'b0, sBiasM1_uid26_fpSqrtTest_q};
    assign expOddSig_uid27_fpSqrtTest_o = $unsigned(expOddSig_uid27_fpSqrtTest_a) + $unsigned(expOddSig_uid27_fpSqrtTest_b);
    assign expOddSig_uid27_fpSqrtTest_q = expOddSig_uid27_fpSqrtTest_o[8:0];

    // expROdd_uid28_fpSqrtTest(BITSELECT,27)@23
    assign expROdd_uid28_fpSqrtTest_b = expOddSig_uid27_fpSqrtTest_q[8:1];

    // sBias_uid22_fpSqrtTest(CONSTANT,21)
    assign sBias_uid22_fpSqrtTest_q = 8'b01111111;

    // expEvenSig_uid24_fpSqrtTest(ADD,23)@23
    assign expEvenSig_uid24_fpSqrtTest_a = {1'b0, redist14_expX_uid6_fpSqrtTest_b_23_mem_q};
    assign expEvenSig_uid24_fpSqrtTest_b = {1'b0, sBias_uid22_fpSqrtTest_q};
    assign expEvenSig_uid24_fpSqrtTest_o = $unsigned(expEvenSig_uid24_fpSqrtTest_a) + $unsigned(expEvenSig_uid24_fpSqrtTest_b);
    assign expEvenSig_uid24_fpSqrtTest_q = expEvenSig_uid24_fpSqrtTest_o[8:0];

    // expREven_uid25_fpSqrtTest(BITSELECT,24)@23
    assign expREven_uid25_fpSqrtTest_b = expEvenSig_uid24_fpSqrtTest_q[8:1];

    // expRMux_uid31_fpSqrtTest(MUX,30)@23 + 1
    assign expRMux_uid31_fpSqrtTest_s = redist10_expOddSelect_uid30_fpSqrtTest_q_23_q;
    always @ (posedge clk)
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

    // redist9_expRMux_uid31_fpSqrtTest_q_2(DELAY,134)
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist9_expRMux_uid31_fpSqrtTest_q_2_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist9_expRMux_uid31_fpSqrtTest_q_2_q <= expRMux_uid31_fpSqrtTest_q;
        end
    end

    // expR_uid66_fpSqrtTest(ADD,65)@25
    assign expR_uid66_fpSqrtTest_a = {1'b0, redist9_expRMux_uid31_fpSqrtTest_q_2_q};
    assign expR_uid66_fpSqrtTest_b = {8'b00000000, expInc_uid64_fpSqrtTest_q};
    assign expR_uid66_fpSqrtTest_o = $unsigned(expR_uid66_fpSqrtTest_a) + $unsigned(expR_uid66_fpSqrtTest_b);
    assign expR_uid66_fpSqrtTest_q = expR_uid66_fpSqrtTest_o[8:0];

    // expRR_uid77_fpSqrtTest(BITSELECT,76)@25
    assign expRR_uid77_fpSqrtTest_in = expR_uid66_fpSqrtTest_q[7:0];
    assign expRR_uid77_fpSqrtTest_b = expRR_uid77_fpSqrtTest_in[7:0];

    // expXIsMax_uid14_fpSqrtTest(LOGICAL,13)@23 + 1
    assign expXIsMax_uid14_fpSqrtTest_qi = redist14_expX_uid6_fpSqrtTest_b_23_mem_q == cstAllOWE_uid8_fpSqrtTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    expXIsMax_uid14_fpSqrtTest_delay ( .xin(expXIsMax_uid14_fpSqrtTest_qi), .xout(expXIsMax_uid14_fpSqrtTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // invExpXIsMax_uid19_fpSqrtTest(LOGICAL,18)@24
    assign invExpXIsMax_uid19_fpSqrtTest_q = ~ (expXIsMax_uid14_fpSqrtTest_q);

    // InvExpXIsZero_uid20_fpSqrtTest(LOGICAL,19)@24
    assign InvExpXIsZero_uid20_fpSqrtTest_q = ~ (excZ_x_uid13_fpSqrtTest_q);

    // excR_x_uid21_fpSqrtTest(LOGICAL,20)@24
    assign excR_x_uid21_fpSqrtTest_q = InvExpXIsZero_uid20_fpSqrtTest_q & invExpXIsMax_uid19_fpSqrtTest_q;

    // minReg_uid69_fpSqrtTest(LOGICAL,68)@24
    assign minReg_uid69_fpSqrtTest_q = excR_x_uid21_fpSqrtTest_q & redist13_signX_uid7_fpSqrtTest_b_24_q;

    // cstZeroWF_uid9_fpSqrtTest(CONSTANT,8)
    assign cstZeroWF_uid9_fpSqrtTest_q = 23'b00000000000000000000000;

    // fracXIsZero_uid15_fpSqrtTest(LOGICAL,14)@23 + 1
    assign fracXIsZero_uid15_fpSqrtTest_qi = cstZeroWF_uid9_fpSqrtTest_q == redist12_frac_x_uid12_fpSqrtTest_b_23_mem_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("SYNC"), .phase(0), .modulus(1) )
    fracXIsZero_uid15_fpSqrtTest_delay ( .xin(fracXIsZero_uid15_fpSqrtTest_qi), .xout(fracXIsZero_uid15_fpSqrtTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // excI_x_uid17_fpSqrtTest(LOGICAL,16)@24
    assign excI_x_uid17_fpSqrtTest_q = expXIsMax_uid14_fpSqrtTest_q & fracXIsZero_uid15_fpSqrtTest_q;

    // minInf_uid70_fpSqrtTest(LOGICAL,69)@24
    assign minInf_uid70_fpSqrtTest_q = excI_x_uid17_fpSqrtTest_q & redist13_signX_uid7_fpSqrtTest_b_24_q;

    // fracXIsNotZero_uid16_fpSqrtTest(LOGICAL,15)@24
    assign fracXIsNotZero_uid16_fpSqrtTest_q = ~ (fracXIsZero_uid15_fpSqrtTest_q);

    // excN_x_uid18_fpSqrtTest(LOGICAL,17)@24
    assign excN_x_uid18_fpSqrtTest_q = expXIsMax_uid14_fpSqrtTest_q & fracXIsNotZero_uid16_fpSqrtTest_q;

    // excRNaN_uid71_fpSqrtTest(LOGICAL,70)@24
    assign excRNaN_uid71_fpSqrtTest_q = excN_x_uid18_fpSqrtTest_q | minInf_uid70_fpSqrtTest_q | minReg_uid69_fpSqrtTest_q;

    // invSignX_uid67_fpSqrtTest(LOGICAL,66)@24
    assign invSignX_uid67_fpSqrtTest_q = ~ (redist13_signX_uid7_fpSqrtTest_b_24_q);

    // inInfAndNotNeg_uid68_fpSqrtTest(LOGICAL,67)@24
    assign inInfAndNotNeg_uid68_fpSqrtTest_q = excI_x_uid17_fpSqrtTest_q & invSignX_uid67_fpSqrtTest_q;

    // excConc_uid72_fpSqrtTest(BITJOIN,71)@24
    assign excConc_uid72_fpSqrtTest_q = {excRNaN_uid71_fpSqrtTest_q, inInfAndNotNeg_uid68_fpSqrtTest_q, excZ_x_uid13_fpSqrtTest_q};

    // fracSelIn_uid73_fpSqrtTest(BITJOIN,72)@24
    assign fracSelIn_uid73_fpSqrtTest_q = {redist13_signX_uid7_fpSqrtTest_b_24_q, excConc_uid72_fpSqrtTest_q};

    // fracSel_uid74_fpSqrtTest(LOOKUP,73)@24 + 1
    always @ (posedge clk)
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

    // expRPostExc_uid79_fpSqrtTest(MUX,78)@25
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
    always @ (posedge clk)
    begin
        if (areset)
        begin
            redist0_expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_c_1_q <= '0;
        end
        else if (en == 1'b1)
        begin
            redist0_expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_c_1_q <= expUpdateCRU_uid61_fpSqrtTest_merged_bit_select_c;
        end
    end

    // fracRPostExc_uid84_fpSqrtTest(MUX,83)@25
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

    // RSqrt_uid86_fpSqrtTest(BITJOIN,85)@25
    assign RSqrt_uid86_fpSqrtTest_q = {negZero_uid85_fpSqrtTest_q, expRPostExc_uid79_fpSqrtTest_q, fracRPostExc_uid84_fpSqrtTest_q};

    // xOut(GPOUT,4)@25
    assign q = RSqrt_uid86_fpSqrtTest_q;

endmodule
