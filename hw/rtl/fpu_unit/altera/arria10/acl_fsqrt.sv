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
// SystemVerilog created on Mon Jan 18 04:15:46 2021


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
    wire [30:0] expInc_uid38_fpSqrtTest_in;
    wire [0:0] expInc_uid38_fpSqrtTest_b;
    wire [28:0] fracRPostProcessings_uid39_fpSqrtTest_in;
    wire [22:0] fracRPostProcessings_uid39_fpSqrtTest_b;
    wire [8:0] expR_uid40_fpSqrtTest_a;
    wire [8:0] expR_uid40_fpSqrtTest_b;
    logic [8:0] expR_uid40_fpSqrtTest_o;
    wire [8:0] expR_uid40_fpSqrtTest_q;
    wire [0:0] invSignX_uid41_fpSqrtTest_q;
    wire [0:0] inInfAndNotNeg_uid42_fpSqrtTest_q;
    wire [0:0] minReg_uid43_fpSqrtTest_q;
    wire [0:0] minInf_uid44_fpSqrtTest_q;
    wire [0:0] excRNaN_uid45_fpSqrtTest_q;
    wire [2:0] excConc_uid46_fpSqrtTest_q;
    wire [3:0] fracSelIn_uid47_fpSqrtTest_q;
    reg [1:0] fracSel_uid48_fpSqrtTest_q;
    wire [7:0] expRR_uid51_fpSqrtTest_in;
    wire [7:0] expRR_uid51_fpSqrtTest_b;
    wire [1:0] expRPostExc_uid53_fpSqrtTest_s;
    reg [7:0] expRPostExc_uid53_fpSqrtTest_q;
    wire [22:0] fracNaN_uid54_fpSqrtTest_q;
    wire [1:0] fracRPostExc_uid58_fpSqrtTest_s;
    reg [22:0] fracRPostExc_uid58_fpSqrtTest_q;
    wire [0:0] negZero_uid59_fpSqrtTest_qi;
    reg [0:0] negZero_uid59_fpSqrtTest_q;
    wire [31:0] RSqrt_uid60_fpSqrtTest_q;
    wire [11:0] yT1_uid74_invPolyEval_b;
    wire [0:0] lowRangeB_uid76_invPolyEval_in;
    wire [0:0] lowRangeB_uid76_invPolyEval_b;
    wire [11:0] highBBits_uid77_invPolyEval_b;
    wire [21:0] s1sumAHighB_uid78_invPolyEval_a;
    wire [21:0] s1sumAHighB_uid78_invPolyEval_b;
    logic [21:0] s1sumAHighB_uid78_invPolyEval_o;
    wire [21:0] s1sumAHighB_uid78_invPolyEval_q;
    wire [22:0] s1_uid79_invPolyEval_q;
    wire [1:0] lowRangeB_uid82_invPolyEval_in;
    wire [1:0] lowRangeB_uid82_invPolyEval_b;
    wire [21:0] highBBits_uid83_invPolyEval_b;
    wire [29:0] s2sumAHighB_uid84_invPolyEval_a;
    wire [29:0] s2sumAHighB_uid84_invPolyEval_b;
    logic [29:0] s2sumAHighB_uid84_invPolyEval_o;
    wire [29:0] s2sumAHighB_uid84_invPolyEval_q;
    wire [31:0] s2_uid85_invPolyEval_q;
    wire [12:0] osig_uid88_pT1_uid75_invPolyEval_b;
    wire [23:0] osig_uid91_pT2_uid81_invPolyEval_b;
    wire memoryC0_uid62_sqrtTables_lutmem_reset0;
    wire [28:0] memoryC0_uid62_sqrtTables_lutmem_ia;
    wire [7:0] memoryC0_uid62_sqrtTables_lutmem_aa;
    wire [7:0] memoryC0_uid62_sqrtTables_lutmem_ab;
    wire [28:0] memoryC0_uid62_sqrtTables_lutmem_ir;
    wire [28:0] memoryC0_uid62_sqrtTables_lutmem_r;
    wire memoryC1_uid65_sqrtTables_lutmem_reset0;
    wire [20:0] memoryC1_uid65_sqrtTables_lutmem_ia;
    wire [7:0] memoryC1_uid65_sqrtTables_lutmem_aa;
    wire [7:0] memoryC1_uid65_sqrtTables_lutmem_ab;
    wire [20:0] memoryC1_uid65_sqrtTables_lutmem_ir;
    wire [20:0] memoryC1_uid65_sqrtTables_lutmem_r;
    wire memoryC2_uid68_sqrtTables_lutmem_reset0;
    wire [11:0] memoryC2_uid68_sqrtTables_lutmem_ia;
    wire [7:0] memoryC2_uid68_sqrtTables_lutmem_aa;
    wire [7:0] memoryC2_uid68_sqrtTables_lutmem_ab;
    wire [11:0] memoryC2_uid68_sqrtTables_lutmem_ir;
    wire [11:0] memoryC2_uid68_sqrtTables_lutmem_r;
    wire prodXY_uid87_pT1_uid75_invPolyEval_cma_reset;
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [11:0] prodXY_uid87_pT1_uid75_invPolyEval_cma_a0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [11:0] prodXY_uid87_pT1_uid75_invPolyEval_cma_a1 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg signed [11:0] prodXY_uid87_pT1_uid75_invPolyEval_cma_c0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg signed [11:0] prodXY_uid87_pT1_uid75_invPolyEval_cma_c1 [0:0];
    wire signed [12:0] prodXY_uid87_pT1_uid75_invPolyEval_cma_l [0:0];
    wire signed [24:0] prodXY_uid87_pT1_uid75_invPolyEval_cma_p [0:0];
    wire signed [24:0] prodXY_uid87_pT1_uid75_invPolyEval_cma_u [0:0];
    wire signed [24:0] prodXY_uid87_pT1_uid75_invPolyEval_cma_w [0:0];
    wire signed [24:0] prodXY_uid87_pT1_uid75_invPolyEval_cma_x [0:0];
    wire signed [24:0] prodXY_uid87_pT1_uid75_invPolyEval_cma_y [0:0];
    reg signed [24:0] prodXY_uid87_pT1_uid75_invPolyEval_cma_s [0:0];
    wire [23:0] prodXY_uid87_pT1_uid75_invPolyEval_cma_qq;
    wire [23:0] prodXY_uid87_pT1_uid75_invPolyEval_cma_q;
    wire prodXY_uid87_pT1_uid75_invPolyEval_cma_ena0;
    wire prodXY_uid87_pT1_uid75_invPolyEval_cma_ena1;
    wire prodXY_uid87_pT1_uid75_invPolyEval_cma_ena2;
    wire prodXY_uid90_pT2_uid81_invPolyEval_cma_reset;
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [15:0] prodXY_uid90_pT2_uid81_invPolyEval_cma_a0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg [15:0] prodXY_uid90_pT2_uid81_invPolyEval_cma_a1 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg signed [22:0] prodXY_uid90_pT2_uid81_invPolyEval_cma_c0 [0:0];
    (* preserve, altera_attribute =  "-name allow_synch_ctrl_usage off" *) reg signed [22:0] prodXY_uid90_pT2_uid81_invPolyEval_cma_c1 [0:0];
    wire signed [16:0] prodXY_uid90_pT2_uid81_invPolyEval_cma_l [0:0];
    wire signed [39:0] prodXY_uid90_pT2_uid81_invPolyEval_cma_p [0:0];
    wire signed [39:0] prodXY_uid90_pT2_uid81_invPolyEval_cma_u [0:0];
    wire signed [39:0] prodXY_uid90_pT2_uid81_invPolyEval_cma_w [0:0];
    wire signed [39:0] prodXY_uid90_pT2_uid81_invPolyEval_cma_x [0:0];
    wire signed [39:0] prodXY_uid90_pT2_uid81_invPolyEval_cma_y [0:0];
    reg signed [39:0] prodXY_uid90_pT2_uid81_invPolyEval_cma_s [0:0];
    wire [38:0] prodXY_uid90_pT2_uid81_invPolyEval_cma_qq;
    wire [38:0] prodXY_uid90_pT2_uid81_invPolyEval_cma_q;
    wire prodXY_uid90_pT2_uid81_invPolyEval_cma_ena0;
    wire prodXY_uid90_pT2_uid81_invPolyEval_cma_ena1;
    wire prodXY_uid90_pT2_uid81_invPolyEval_cma_ena2;
    reg [0:0] redist0_lowRangeB_uid76_invPolyEval_b_1_q;
    reg [0:0] redist1_negZero_uid59_fpSqrtTest_q_9_q;
    reg [1:0] redist2_fracSel_uid48_fpSqrtTest_q_9_q;
    reg [22:0] redist3_fracRPostProcessings_uid39_fpSqrtTest_b_1_q;
    reg [0:0] redist4_expInc_uid38_fpSqrtTest_b_1_q;
    reg [15:0] redist5_yForPe_uid36_fpSqrtTest_b_2_q;
    reg [7:0] redist7_yAddr_uid35_fpSqrtTest_b_3_q;
    reg [7:0] redist8_yAddr_uid35_fpSqrtTest_b_7_q;
    reg [0:0] redist10_signX_uid7_fpSqrtTest_b_1_q;
    wire redist6_yForPe_uid36_fpSqrtTest_b_6_mem_reset0;
    wire [15:0] redist6_yForPe_uid36_fpSqrtTest_b_6_mem_ia;
    wire [1:0] redist6_yForPe_uid36_fpSqrtTest_b_6_mem_aa;
    wire [1:0] redist6_yForPe_uid36_fpSqrtTest_b_6_mem_ab;
    wire [15:0] redist6_yForPe_uid36_fpSqrtTest_b_6_mem_iq;
    wire [15:0] redist6_yForPe_uid36_fpSqrtTest_b_6_mem_q;
    wire [1:0] redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt_q;
    (* preserve *) reg [1:0] redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt_i;
    (* preserve *) reg redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt_eq;
    wire [0:0] redist6_yForPe_uid36_fpSqrtTest_b_6_rdmux_s;
    reg [1:0] redist6_yForPe_uid36_fpSqrtTest_b_6_rdmux_q;
    reg [1:0] redist6_yForPe_uid36_fpSqrtTest_b_6_wraddr_q;
    wire [1:0] redist6_yForPe_uid36_fpSqrtTest_b_6_mem_last_q;
    wire [0:0] redist6_yForPe_uid36_fpSqrtTest_b_6_cmp_q;
    reg [0:0] redist6_yForPe_uid36_fpSqrtTest_b_6_cmpReg_q;
    wire [0:0] redist6_yForPe_uid36_fpSqrtTest_b_6_notEnable_q;
    wire [0:0] redist6_yForPe_uid36_fpSqrtTest_b_6_nor_q;
    (* preserve_syn_only *) reg [0:0] redist6_yForPe_uid36_fpSqrtTest_b_6_sticky_ena_q;
    wire [0:0] redist6_yForPe_uid36_fpSqrtTest_b_6_enaAnd_q;
    reg [7:0] redist9_expRMux_uid31_fpSqrtTest_q_10_outputreg_q;
    wire redist9_expRMux_uid31_fpSqrtTest_q_10_mem_reset0;
    wire [7:0] redist9_expRMux_uid31_fpSqrtTest_q_10_mem_ia;
    wire [2:0] redist9_expRMux_uid31_fpSqrtTest_q_10_mem_aa;
    wire [2:0] redist9_expRMux_uid31_fpSqrtTest_q_10_mem_ab;
    wire [7:0] redist9_expRMux_uid31_fpSqrtTest_q_10_mem_iq;
    wire [7:0] redist9_expRMux_uid31_fpSqrtTest_q_10_mem_q;
    wire [2:0] redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt_q;
    (* preserve *) reg [2:0] redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt_i;
    (* preserve *) reg redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt_eq;
    wire [0:0] redist9_expRMux_uid31_fpSqrtTest_q_10_rdmux_s;
    reg [2:0] redist9_expRMux_uid31_fpSqrtTest_q_10_rdmux_q;
    reg [2:0] redist9_expRMux_uid31_fpSqrtTest_q_10_wraddr_q;
    wire [3:0] redist9_expRMux_uid31_fpSqrtTest_q_10_mem_last_q;
    wire [3:0] redist9_expRMux_uid31_fpSqrtTest_q_10_cmp_b;
    wire [0:0] redist9_expRMux_uid31_fpSqrtTest_q_10_cmp_q;
    reg [0:0] redist9_expRMux_uid31_fpSqrtTest_q_10_cmpReg_q;
    wire [0:0] redist9_expRMux_uid31_fpSqrtTest_q_10_notEnable_q;
    wire [0:0] redist9_expRMux_uid31_fpSqrtTest_q_10_nor_q;
    (* preserve_syn_only *) reg [0:0] redist9_expRMux_uid31_fpSqrtTest_q_10_sticky_ena_q;
    wire [0:0] redist9_expRMux_uid31_fpSqrtTest_q_10_enaAnd_q;


    // signX_uid7_fpSqrtTest(BITSELECT,6)@0
    assign signX_uid7_fpSqrtTest_b = a[31:31];

    // redist10_signX_uid7_fpSqrtTest_b_1(DELAY,107)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist10_signX_uid7_fpSqrtTest_b_1 ( .xin(signX_uid7_fpSqrtTest_b), .xout(redist10_signX_uid7_fpSqrtTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // cstAllZWE_uid10_fpSqrtTest(CONSTANT,9)
    assign cstAllZWE_uid10_fpSqrtTest_q = 8'b00000000;

    // expX_uid6_fpSqrtTest(BITSELECT,5)@0
    assign expX_uid6_fpSqrtTest_b = a[30:23];

    // excZ_x_uid13_fpSqrtTest(LOGICAL,12)@0 + 1
    assign excZ_x_uid13_fpSqrtTest_qi = expX_uid6_fpSqrtTest_b == cstAllZWE_uid10_fpSqrtTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    excZ_x_uid13_fpSqrtTest_delay ( .xin(excZ_x_uid13_fpSqrtTest_qi), .xout(excZ_x_uid13_fpSqrtTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // negZero_uid59_fpSqrtTest(LOGICAL,58)@1 + 1
    assign negZero_uid59_fpSqrtTest_qi = excZ_x_uid13_fpSqrtTest_q & redist10_signX_uid7_fpSqrtTest_b_1_q;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    negZero_uid59_fpSqrtTest_delay ( .xin(negZero_uid59_fpSqrtTest_qi), .xout(negZero_uid59_fpSqrtTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist1_negZero_uid59_fpSqrtTest_q_9(DELAY,98)
    dspba_delay_ver #( .width(1), .depth(8), .reset_kind("ASYNC") )
    redist1_negZero_uid59_fpSqrtTest_q_9 ( .xin(negZero_uid59_fpSqrtTest_q), .xout(redist1_negZero_uid59_fpSqrtTest_q_9_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // cstAllOWE_uid8_fpSqrtTest(CONSTANT,7)
    assign cstAllOWE_uid8_fpSqrtTest_q = 8'b11111111;

    // expX0PS_uid29_fpSqrtTest(BITSELECT,28)@0
    assign expX0PS_uid29_fpSqrtTest_in = expX_uid6_fpSqrtTest_b[0:0];
    assign expX0PS_uid29_fpSqrtTest_b = expX0PS_uid29_fpSqrtTest_in[0:0];

    // expOddSelect_uid30_fpSqrtTest(LOGICAL,29)@0
    assign expOddSelect_uid30_fpSqrtTest_q = ~ (expX0PS_uid29_fpSqrtTest_b);

    // frac_x_uid12_fpSqrtTest(BITSELECT,11)@0
    assign frac_x_uid12_fpSqrtTest_b = a[22:0];

    // addrFull_uid33_fpSqrtTest(BITJOIN,32)@0
    assign addrFull_uid33_fpSqrtTest_q = {expOddSelect_uid30_fpSqrtTest_q, frac_x_uid12_fpSqrtTest_b};

    // yAddr_uid35_fpSqrtTest(BITSELECT,34)@0
    assign yAddr_uid35_fpSqrtTest_b = addrFull_uid33_fpSqrtTest_q[23:16];

    // memoryC2_uid68_sqrtTables_lutmem(DUALMEM,94)@0 + 2
    // in j@20000000
    assign memoryC2_uid68_sqrtTables_lutmem_aa = yAddr_uid35_fpSqrtTest_b;
    assign memoryC2_uid68_sqrtTables_lutmem_reset0 = areset;
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
        .init_file("acl_fsqrt_memoryC2_uid68_sqrtTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Arria 10")
    ) memoryC2_uid68_sqrtTables_lutmem_dmem (
        .clocken0(en[0]),
        .aclr0(memoryC2_uid68_sqrtTables_lutmem_reset0),
        .clock0(clk),
        .address_a(memoryC2_uid68_sqrtTables_lutmem_aa),
        .q_a(memoryC2_uid68_sqrtTables_lutmem_ir),
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
    assign memoryC2_uid68_sqrtTables_lutmem_r = memoryC2_uid68_sqrtTables_lutmem_ir[11:0];

    // yForPe_uid36_fpSqrtTest(BITSELECT,35)@0
    assign yForPe_uid36_fpSqrtTest_in = frac_x_uid12_fpSqrtTest_b[15:0];
    assign yForPe_uid36_fpSqrtTest_b = yForPe_uid36_fpSqrtTest_in[15:0];

    // redist5_yForPe_uid36_fpSqrtTest_b_2(DELAY,102)
    dspba_delay_ver #( .width(16), .depth(2), .reset_kind("ASYNC") )
    redist5_yForPe_uid36_fpSqrtTest_b_2 ( .xin(yForPe_uid36_fpSqrtTest_b), .xout(redist5_yForPe_uid36_fpSqrtTest_b_2_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // yT1_uid74_invPolyEval(BITSELECT,73)@2
    assign yT1_uid74_invPolyEval_b = redist5_yForPe_uid36_fpSqrtTest_b_2_q[15:4];

    // prodXY_uid87_pT1_uid75_invPolyEval_cma(CHAINMULTADD,95)@2 + 3
    assign prodXY_uid87_pT1_uid75_invPolyEval_cma_reset = areset;
    assign prodXY_uid87_pT1_uid75_invPolyEval_cma_ena0 = en[0];
    assign prodXY_uid87_pT1_uid75_invPolyEval_cma_ena1 = prodXY_uid87_pT1_uid75_invPolyEval_cma_ena0;
    assign prodXY_uid87_pT1_uid75_invPolyEval_cma_ena2 = prodXY_uid87_pT1_uid75_invPolyEval_cma_ena0;
    assign prodXY_uid87_pT1_uid75_invPolyEval_cma_l[0] = $signed({1'b0, prodXY_uid87_pT1_uid75_invPolyEval_cma_a1[0][11:0]});
    assign prodXY_uid87_pT1_uid75_invPolyEval_cma_p[0] = prodXY_uid87_pT1_uid75_invPolyEval_cma_l[0] * prodXY_uid87_pT1_uid75_invPolyEval_cma_c1[0];
    assign prodXY_uid87_pT1_uid75_invPolyEval_cma_u[0] = prodXY_uid87_pT1_uid75_invPolyEval_cma_p[0][24:0];
    assign prodXY_uid87_pT1_uid75_invPolyEval_cma_w[0] = prodXY_uid87_pT1_uid75_invPolyEval_cma_u[0];
    assign prodXY_uid87_pT1_uid75_invPolyEval_cma_x[0] = prodXY_uid87_pT1_uid75_invPolyEval_cma_w[0];
    assign prodXY_uid87_pT1_uid75_invPolyEval_cma_y[0] = prodXY_uid87_pT1_uid75_invPolyEval_cma_x[0];
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid87_pT1_uid75_invPolyEval_cma_a0 <= '{default: '0};
            prodXY_uid87_pT1_uid75_invPolyEval_cma_c0 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid87_pT1_uid75_invPolyEval_cma_ena0 == 1'b1)
            begin
                prodXY_uid87_pT1_uid75_invPolyEval_cma_a0[0] <= yT1_uid74_invPolyEval_b;
                prodXY_uid87_pT1_uid75_invPolyEval_cma_c0[0] <= memoryC2_uid68_sqrtTables_lutmem_r;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid87_pT1_uid75_invPolyEval_cma_a1 <= '{default: '0};
            prodXY_uid87_pT1_uid75_invPolyEval_cma_c1 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid87_pT1_uid75_invPolyEval_cma_ena2 == 1'b1)
            begin
                prodXY_uid87_pT1_uid75_invPolyEval_cma_a1 <= prodXY_uid87_pT1_uid75_invPolyEval_cma_a0;
                prodXY_uid87_pT1_uid75_invPolyEval_cma_c1 <= prodXY_uid87_pT1_uid75_invPolyEval_cma_c0;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid87_pT1_uid75_invPolyEval_cma_s <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid87_pT1_uid75_invPolyEval_cma_ena1 == 1'b1)
            begin
                prodXY_uid87_pT1_uid75_invPolyEval_cma_s[0] <= prodXY_uid87_pT1_uid75_invPolyEval_cma_y[0];
            end
        end
    end
    dspba_delay_ver #( .width(24), .depth(0), .reset_kind("ASYNC") )
    prodXY_uid87_pT1_uid75_invPolyEval_cma_delay ( .xin(prodXY_uid87_pT1_uid75_invPolyEval_cma_s[0][23:0]), .xout(prodXY_uid87_pT1_uid75_invPolyEval_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign prodXY_uid87_pT1_uid75_invPolyEval_cma_q = prodXY_uid87_pT1_uid75_invPolyEval_cma_qq[23:0];

    // osig_uid88_pT1_uid75_invPolyEval(BITSELECT,87)@5
    assign osig_uid88_pT1_uid75_invPolyEval_b = prodXY_uid87_pT1_uid75_invPolyEval_cma_q[23:11];

    // highBBits_uid77_invPolyEval(BITSELECT,76)@5
    assign highBBits_uid77_invPolyEval_b = osig_uid88_pT1_uid75_invPolyEval_b[12:1];

    // redist7_yAddr_uid35_fpSqrtTest_b_3(DELAY,104)
    dspba_delay_ver #( .width(8), .depth(3), .reset_kind("ASYNC") )
    redist7_yAddr_uid35_fpSqrtTest_b_3 ( .xin(yAddr_uid35_fpSqrtTest_b), .xout(redist7_yAddr_uid35_fpSqrtTest_b_3_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // memoryC1_uid65_sqrtTables_lutmem(DUALMEM,93)@3 + 2
    // in j@20000000
    assign memoryC1_uid65_sqrtTables_lutmem_aa = redist7_yAddr_uid35_fpSqrtTest_b_3_q;
    assign memoryC1_uid65_sqrtTables_lutmem_reset0 = areset;
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
        .init_file("acl_fsqrt_memoryC1_uid65_sqrtTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Arria 10")
    ) memoryC1_uid65_sqrtTables_lutmem_dmem (
        .clocken0(en[0]),
        .aclr0(memoryC1_uid65_sqrtTables_lutmem_reset0),
        .clock0(clk),
        .address_a(memoryC1_uid65_sqrtTables_lutmem_aa),
        .q_a(memoryC1_uid65_sqrtTables_lutmem_ir),
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
    assign memoryC1_uid65_sqrtTables_lutmem_r = memoryC1_uid65_sqrtTables_lutmem_ir[20:0];

    // s1sumAHighB_uid78_invPolyEval(ADD,77)@5 + 1
    assign s1sumAHighB_uid78_invPolyEval_a = {{1{memoryC1_uid65_sqrtTables_lutmem_r[20]}}, memoryC1_uid65_sqrtTables_lutmem_r};
    assign s1sumAHighB_uid78_invPolyEval_b = {{10{highBBits_uid77_invPolyEval_b[11]}}, highBBits_uid77_invPolyEval_b};
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            s1sumAHighB_uid78_invPolyEval_o <= 22'b0;
        end
        else if (en == 1'b1)
        begin
            s1sumAHighB_uid78_invPolyEval_o <= $signed(s1sumAHighB_uid78_invPolyEval_a) + $signed(s1sumAHighB_uid78_invPolyEval_b);
        end
    end
    assign s1sumAHighB_uid78_invPolyEval_q = s1sumAHighB_uid78_invPolyEval_o[21:0];

    // lowRangeB_uid76_invPolyEval(BITSELECT,75)@5
    assign lowRangeB_uid76_invPolyEval_in = osig_uid88_pT1_uid75_invPolyEval_b[0:0];
    assign lowRangeB_uid76_invPolyEval_b = lowRangeB_uid76_invPolyEval_in[0:0];

    // redist0_lowRangeB_uid76_invPolyEval_b_1(DELAY,97)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist0_lowRangeB_uid76_invPolyEval_b_1 ( .xin(lowRangeB_uid76_invPolyEval_b), .xout(redist0_lowRangeB_uid76_invPolyEval_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // s1_uid79_invPolyEval(BITJOIN,78)@6
    assign s1_uid79_invPolyEval_q = {s1sumAHighB_uid78_invPolyEval_q, redist0_lowRangeB_uid76_invPolyEval_b_1_q};

    // redist6_yForPe_uid36_fpSqrtTest_b_6_notEnable(LOGICAL,115)
    assign redist6_yForPe_uid36_fpSqrtTest_b_6_notEnable_q = ~ (en);

    // redist6_yForPe_uid36_fpSqrtTest_b_6_nor(LOGICAL,116)
    assign redist6_yForPe_uid36_fpSqrtTest_b_6_nor_q = ~ (redist6_yForPe_uid36_fpSqrtTest_b_6_notEnable_q | redist6_yForPe_uid36_fpSqrtTest_b_6_sticky_ena_q);

    // redist6_yForPe_uid36_fpSqrtTest_b_6_mem_last(CONSTANT,112)
    assign redist6_yForPe_uid36_fpSqrtTest_b_6_mem_last_q = 2'b01;

    // redist6_yForPe_uid36_fpSqrtTest_b_6_cmp(LOGICAL,113)
    assign redist6_yForPe_uid36_fpSqrtTest_b_6_cmp_q = redist6_yForPe_uid36_fpSqrtTest_b_6_mem_last_q == redist6_yForPe_uid36_fpSqrtTest_b_6_rdmux_q ? 1'b1 : 1'b0;

    // redist6_yForPe_uid36_fpSqrtTest_b_6_cmpReg(REG,114)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist6_yForPe_uid36_fpSqrtTest_b_6_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist6_yForPe_uid36_fpSqrtTest_b_6_cmpReg_q <= redist6_yForPe_uid36_fpSqrtTest_b_6_cmp_q;
        end
    end

    // redist6_yForPe_uid36_fpSqrtTest_b_6_sticky_ena(REG,117)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist6_yForPe_uid36_fpSqrtTest_b_6_sticky_ena_q <= 1'b0;
        end
        else if (redist6_yForPe_uid36_fpSqrtTest_b_6_nor_q == 1'b1)
        begin
            redist6_yForPe_uid36_fpSqrtTest_b_6_sticky_ena_q <= redist6_yForPe_uid36_fpSqrtTest_b_6_cmpReg_q;
        end
    end

    // redist6_yForPe_uid36_fpSqrtTest_b_6_enaAnd(LOGICAL,118)
    assign redist6_yForPe_uid36_fpSqrtTest_b_6_enaAnd_q = redist6_yForPe_uid36_fpSqrtTest_b_6_sticky_ena_q & en;

    // redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt(COUNTER,109)
    // low=0, high=2, step=1, init=0
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt_i <= 2'd0;
            redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt_i == 2'd1)
            begin
                redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt_eq <= 1'b0;
            end
            if (redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt_eq == 1'b1)
            begin
                redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt_i <= $unsigned(redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt_i) + $unsigned(2'd2);
            end
            else
            begin
                redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt_i <= $unsigned(redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt_i) + $unsigned(2'd1);
            end
        end
    end
    assign redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt_q = redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt_i[1:0];

    // redist6_yForPe_uid36_fpSqrtTest_b_6_rdmux(MUX,110)
    assign redist6_yForPe_uid36_fpSqrtTest_b_6_rdmux_s = en;
    always @(redist6_yForPe_uid36_fpSqrtTest_b_6_rdmux_s or redist6_yForPe_uid36_fpSqrtTest_b_6_wraddr_q or redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt_q)
    begin
        unique case (redist6_yForPe_uid36_fpSqrtTest_b_6_rdmux_s)
            1'b0 : redist6_yForPe_uid36_fpSqrtTest_b_6_rdmux_q = redist6_yForPe_uid36_fpSqrtTest_b_6_wraddr_q;
            1'b1 : redist6_yForPe_uid36_fpSqrtTest_b_6_rdmux_q = redist6_yForPe_uid36_fpSqrtTest_b_6_rdcnt_q;
            default : redist6_yForPe_uid36_fpSqrtTest_b_6_rdmux_q = 2'b0;
        endcase
    end

    // VCC(CONSTANT,1)
    assign VCC_q = 1'b1;

    // redist6_yForPe_uid36_fpSqrtTest_b_6_wraddr(REG,111)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist6_yForPe_uid36_fpSqrtTest_b_6_wraddr_q <= 2'b10;
        end
        else
        begin
            redist6_yForPe_uid36_fpSqrtTest_b_6_wraddr_q <= redist6_yForPe_uid36_fpSqrtTest_b_6_rdmux_q;
        end
    end

    // redist6_yForPe_uid36_fpSqrtTest_b_6_mem(DUALMEM,108)
    assign redist6_yForPe_uid36_fpSqrtTest_b_6_mem_ia = redist5_yForPe_uid36_fpSqrtTest_b_2_q;
    assign redist6_yForPe_uid36_fpSqrtTest_b_6_mem_aa = redist6_yForPe_uid36_fpSqrtTest_b_6_wraddr_q;
    assign redist6_yForPe_uid36_fpSqrtTest_b_6_mem_ab = redist6_yForPe_uid36_fpSqrtTest_b_6_rdmux_q;
    assign redist6_yForPe_uid36_fpSqrtTest_b_6_mem_reset0 = areset;
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
    ) redist6_yForPe_uid36_fpSqrtTest_b_6_mem_dmem (
        .clocken1(redist6_yForPe_uid36_fpSqrtTest_b_6_enaAnd_q[0]),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .aclr1(redist6_yForPe_uid36_fpSqrtTest_b_6_mem_reset0),
        .clock1(clk),
        .address_a(redist6_yForPe_uid36_fpSqrtTest_b_6_mem_aa),
        .data_a(redist6_yForPe_uid36_fpSqrtTest_b_6_mem_ia),
        .wren_a(en[0]),
        .address_b(redist6_yForPe_uid36_fpSqrtTest_b_6_mem_ab),
        .q_b(redist6_yForPe_uid36_fpSqrtTest_b_6_mem_iq),
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
    assign redist6_yForPe_uid36_fpSqrtTest_b_6_mem_q = redist6_yForPe_uid36_fpSqrtTest_b_6_mem_iq[15:0];

    // GND(CONSTANT,0)
    assign GND_q = 1'b0;

    // prodXY_uid90_pT2_uid81_invPolyEval_cma(CHAINMULTADD,96)@6 + 3
    assign prodXY_uid90_pT2_uid81_invPolyEval_cma_reset = areset;
    assign prodXY_uid90_pT2_uid81_invPolyEval_cma_ena0 = en[0];
    assign prodXY_uid90_pT2_uid81_invPolyEval_cma_ena1 = prodXY_uid90_pT2_uid81_invPolyEval_cma_ena0;
    assign prodXY_uid90_pT2_uid81_invPolyEval_cma_ena2 = prodXY_uid90_pT2_uid81_invPolyEval_cma_ena0;
    assign prodXY_uid90_pT2_uid81_invPolyEval_cma_l[0] = $signed({1'b0, prodXY_uid90_pT2_uid81_invPolyEval_cma_a1[0][15:0]});
    assign prodXY_uid90_pT2_uid81_invPolyEval_cma_p[0] = prodXY_uid90_pT2_uid81_invPolyEval_cma_l[0] * prodXY_uid90_pT2_uid81_invPolyEval_cma_c1[0];
    assign prodXY_uid90_pT2_uid81_invPolyEval_cma_u[0] = prodXY_uid90_pT2_uid81_invPolyEval_cma_p[0][39:0];
    assign prodXY_uid90_pT2_uid81_invPolyEval_cma_w[0] = prodXY_uid90_pT2_uid81_invPolyEval_cma_u[0];
    assign prodXY_uid90_pT2_uid81_invPolyEval_cma_x[0] = prodXY_uid90_pT2_uid81_invPolyEval_cma_w[0];
    assign prodXY_uid90_pT2_uid81_invPolyEval_cma_y[0] = prodXY_uid90_pT2_uid81_invPolyEval_cma_x[0];
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid90_pT2_uid81_invPolyEval_cma_a0 <= '{default: '0};
            prodXY_uid90_pT2_uid81_invPolyEval_cma_c0 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid90_pT2_uid81_invPolyEval_cma_ena0 == 1'b1)
            begin
                prodXY_uid90_pT2_uid81_invPolyEval_cma_a0[0] <= redist6_yForPe_uid36_fpSqrtTest_b_6_mem_q;
                prodXY_uid90_pT2_uid81_invPolyEval_cma_c0[0] <= s1_uid79_invPolyEval_q;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid90_pT2_uid81_invPolyEval_cma_a1 <= '{default: '0};
            prodXY_uid90_pT2_uid81_invPolyEval_cma_c1 <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid90_pT2_uid81_invPolyEval_cma_ena2 == 1'b1)
            begin
                prodXY_uid90_pT2_uid81_invPolyEval_cma_a1 <= prodXY_uid90_pT2_uid81_invPolyEval_cma_a0;
                prodXY_uid90_pT2_uid81_invPolyEval_cma_c1 <= prodXY_uid90_pT2_uid81_invPolyEval_cma_c0;
            end
        end
    end
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            prodXY_uid90_pT2_uid81_invPolyEval_cma_s <= '{default: '0};
        end
        else
        begin
            if (prodXY_uid90_pT2_uid81_invPolyEval_cma_ena1 == 1'b1)
            begin
                prodXY_uid90_pT2_uid81_invPolyEval_cma_s[0] <= prodXY_uid90_pT2_uid81_invPolyEval_cma_y[0];
            end
        end
    end
    dspba_delay_ver #( .width(39), .depth(0), .reset_kind("ASYNC") )
    prodXY_uid90_pT2_uid81_invPolyEval_cma_delay ( .xin(prodXY_uid90_pT2_uid81_invPolyEval_cma_s[0][38:0]), .xout(prodXY_uid90_pT2_uid81_invPolyEval_cma_qq), .ena(en[0]), .clk(clk), .aclr(areset) );
    assign prodXY_uid90_pT2_uid81_invPolyEval_cma_q = prodXY_uid90_pT2_uid81_invPolyEval_cma_qq[38:0];

    // osig_uid91_pT2_uid81_invPolyEval(BITSELECT,90)@9
    assign osig_uid91_pT2_uid81_invPolyEval_b = prodXY_uid90_pT2_uid81_invPolyEval_cma_q[38:15];

    // highBBits_uid83_invPolyEval(BITSELECT,82)@9
    assign highBBits_uid83_invPolyEval_b = osig_uid91_pT2_uid81_invPolyEval_b[23:2];

    // redist8_yAddr_uid35_fpSqrtTest_b_7(DELAY,105)
    dspba_delay_ver #( .width(8), .depth(4), .reset_kind("ASYNC") )
    redist8_yAddr_uid35_fpSqrtTest_b_7 ( .xin(redist7_yAddr_uid35_fpSqrtTest_b_3_q), .xout(redist8_yAddr_uid35_fpSqrtTest_b_7_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // memoryC0_uid62_sqrtTables_lutmem(DUALMEM,92)@7 + 2
    // in j@20000000
    assign memoryC0_uid62_sqrtTables_lutmem_aa = redist8_yAddr_uid35_fpSqrtTest_b_7_q;
    assign memoryC0_uid62_sqrtTables_lutmem_reset0 = areset;
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
        .init_file("acl_fsqrt_memoryC0_uid62_sqrtTables_lutmem.hex"),
        .init_file_layout("PORT_A"),
        .intended_device_family("Arria 10")
    ) memoryC0_uid62_sqrtTables_lutmem_dmem (
        .clocken0(en[0]),
        .aclr0(memoryC0_uid62_sqrtTables_lutmem_reset0),
        .clock0(clk),
        .address_a(memoryC0_uid62_sqrtTables_lutmem_aa),
        .q_a(memoryC0_uid62_sqrtTables_lutmem_ir),
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
    assign memoryC0_uid62_sqrtTables_lutmem_r = memoryC0_uid62_sqrtTables_lutmem_ir[28:0];

    // s2sumAHighB_uid84_invPolyEval(ADD,83)@9
    assign s2sumAHighB_uid84_invPolyEval_a = {{1{memoryC0_uid62_sqrtTables_lutmem_r[28]}}, memoryC0_uid62_sqrtTables_lutmem_r};
    assign s2sumAHighB_uid84_invPolyEval_b = {{8{highBBits_uid83_invPolyEval_b[21]}}, highBBits_uid83_invPolyEval_b};
    assign s2sumAHighB_uid84_invPolyEval_o = $signed(s2sumAHighB_uid84_invPolyEval_a) + $signed(s2sumAHighB_uid84_invPolyEval_b);
    assign s2sumAHighB_uid84_invPolyEval_q = s2sumAHighB_uid84_invPolyEval_o[29:0];

    // lowRangeB_uid82_invPolyEval(BITSELECT,81)@9
    assign lowRangeB_uid82_invPolyEval_in = osig_uid91_pT2_uid81_invPolyEval_b[1:0];
    assign lowRangeB_uid82_invPolyEval_b = lowRangeB_uid82_invPolyEval_in[1:0];

    // s2_uid85_invPolyEval(BITJOIN,84)@9
    assign s2_uid85_invPolyEval_q = {s2sumAHighB_uid84_invPolyEval_q, lowRangeB_uid82_invPolyEval_b};

    // expInc_uid38_fpSqrtTest(BITSELECT,37)@9
    assign expInc_uid38_fpSqrtTest_in = s2_uid85_invPolyEval_q[30:0];
    assign expInc_uid38_fpSqrtTest_b = expInc_uid38_fpSqrtTest_in[30:30];

    // redist4_expInc_uid38_fpSqrtTest_b_1(DELAY,101)
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    redist4_expInc_uid38_fpSqrtTest_b_1 ( .xin(expInc_uid38_fpSqrtTest_b), .xout(redist4_expInc_uid38_fpSqrtTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // redist9_expRMux_uid31_fpSqrtTest_q_10_notEnable(LOGICAL,127)
    assign redist9_expRMux_uid31_fpSqrtTest_q_10_notEnable_q = ~ (en);

    // redist9_expRMux_uid31_fpSqrtTest_q_10_nor(LOGICAL,128)
    assign redist9_expRMux_uid31_fpSqrtTest_q_10_nor_q = ~ (redist9_expRMux_uid31_fpSqrtTest_q_10_notEnable_q | redist9_expRMux_uid31_fpSqrtTest_q_10_sticky_ena_q);

    // redist9_expRMux_uid31_fpSqrtTest_q_10_mem_last(CONSTANT,124)
    assign redist9_expRMux_uid31_fpSqrtTest_q_10_mem_last_q = 4'b0101;

    // redist9_expRMux_uid31_fpSqrtTest_q_10_cmp(LOGICAL,125)
    assign redist9_expRMux_uid31_fpSqrtTest_q_10_cmp_b = {1'b0, redist9_expRMux_uid31_fpSqrtTest_q_10_rdmux_q};
    assign redist9_expRMux_uid31_fpSqrtTest_q_10_cmp_q = redist9_expRMux_uid31_fpSqrtTest_q_10_mem_last_q == redist9_expRMux_uid31_fpSqrtTest_q_10_cmp_b ? 1'b1 : 1'b0;

    // redist9_expRMux_uid31_fpSqrtTest_q_10_cmpReg(REG,126)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist9_expRMux_uid31_fpSqrtTest_q_10_cmpReg_q <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            redist9_expRMux_uid31_fpSqrtTest_q_10_cmpReg_q <= redist9_expRMux_uid31_fpSqrtTest_q_10_cmp_q;
        end
    end

    // redist9_expRMux_uid31_fpSqrtTest_q_10_sticky_ena(REG,129)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist9_expRMux_uid31_fpSqrtTest_q_10_sticky_ena_q <= 1'b0;
        end
        else if (redist9_expRMux_uid31_fpSqrtTest_q_10_nor_q == 1'b1)
        begin
            redist9_expRMux_uid31_fpSqrtTest_q_10_sticky_ena_q <= redist9_expRMux_uid31_fpSqrtTest_q_10_cmpReg_q;
        end
    end

    // redist9_expRMux_uid31_fpSqrtTest_q_10_enaAnd(LOGICAL,130)
    assign redist9_expRMux_uid31_fpSqrtTest_q_10_enaAnd_q = redist9_expRMux_uid31_fpSqrtTest_q_10_sticky_ena_q & en;

    // redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt(COUNTER,121)
    // low=0, high=6, step=1, init=0
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt_i <= 3'd0;
            redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt_eq <= 1'b0;
        end
        else if (en == 1'b1)
        begin
            if (redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt_i == 3'd5)
            begin
                redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt_eq <= 1'b1;
            end
            else
            begin
                redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt_eq <= 1'b0;
            end
            if (redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt_eq == 1'b1)
            begin
                redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt_i <= $unsigned(redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt_i) + $unsigned(3'd2);
            end
            else
            begin
                redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt_i <= $unsigned(redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt_i) + $unsigned(3'd1);
            end
        end
    end
    assign redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt_q = redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt_i[2:0];

    // redist9_expRMux_uid31_fpSqrtTest_q_10_rdmux(MUX,122)
    assign redist9_expRMux_uid31_fpSqrtTest_q_10_rdmux_s = en;
    always @(redist9_expRMux_uid31_fpSqrtTest_q_10_rdmux_s or redist9_expRMux_uid31_fpSqrtTest_q_10_wraddr_q or redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt_q)
    begin
        unique case (redist9_expRMux_uid31_fpSqrtTest_q_10_rdmux_s)
            1'b0 : redist9_expRMux_uid31_fpSqrtTest_q_10_rdmux_q = redist9_expRMux_uid31_fpSqrtTest_q_10_wraddr_q;
            1'b1 : redist9_expRMux_uid31_fpSqrtTest_q_10_rdmux_q = redist9_expRMux_uid31_fpSqrtTest_q_10_rdcnt_q;
            default : redist9_expRMux_uid31_fpSqrtTest_q_10_rdmux_q = 3'b0;
        endcase
    end

    // sBiasM1_uid26_fpSqrtTest(CONSTANT,25)
    assign sBiasM1_uid26_fpSqrtTest_q = 8'b01111110;

    // expOddSig_uid27_fpSqrtTest(ADD,26)@0
    assign expOddSig_uid27_fpSqrtTest_a = {1'b0, expX_uid6_fpSqrtTest_b};
    assign expOddSig_uid27_fpSqrtTest_b = {1'b0, sBiasM1_uid26_fpSqrtTest_q};
    assign expOddSig_uid27_fpSqrtTest_o = $unsigned(expOddSig_uid27_fpSqrtTest_a) + $unsigned(expOddSig_uid27_fpSqrtTest_b);
    assign expOddSig_uid27_fpSqrtTest_q = expOddSig_uid27_fpSqrtTest_o[8:0];

    // expROdd_uid28_fpSqrtTest(BITSELECT,27)@0
    assign expROdd_uid28_fpSqrtTest_b = expOddSig_uid27_fpSqrtTest_q[8:1];

    // sBias_uid22_fpSqrtTest(CONSTANT,21)
    assign sBias_uid22_fpSqrtTest_q = 8'b01111111;

    // expEvenSig_uid24_fpSqrtTest(ADD,23)@0
    assign expEvenSig_uid24_fpSqrtTest_a = {1'b0, expX_uid6_fpSqrtTest_b};
    assign expEvenSig_uid24_fpSqrtTest_b = {1'b0, sBias_uid22_fpSqrtTest_q};
    assign expEvenSig_uid24_fpSqrtTest_o = $unsigned(expEvenSig_uid24_fpSqrtTest_a) + $unsigned(expEvenSig_uid24_fpSqrtTest_b);
    assign expEvenSig_uid24_fpSqrtTest_q = expEvenSig_uid24_fpSqrtTest_o[8:0];

    // expREven_uid25_fpSqrtTest(BITSELECT,24)@0
    assign expREven_uid25_fpSqrtTest_b = expEvenSig_uid24_fpSqrtTest_q[8:1];

    // expRMux_uid31_fpSqrtTest(MUX,30)@0 + 1
    assign expRMux_uid31_fpSqrtTest_s = expOddSelect_uid30_fpSqrtTest_q;
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

    // redist9_expRMux_uid31_fpSqrtTest_q_10_wraddr(REG,123)
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            redist9_expRMux_uid31_fpSqrtTest_q_10_wraddr_q <= 3'b110;
        end
        else
        begin
            redist9_expRMux_uid31_fpSqrtTest_q_10_wraddr_q <= redist9_expRMux_uid31_fpSqrtTest_q_10_rdmux_q;
        end
    end

    // redist9_expRMux_uid31_fpSqrtTest_q_10_mem(DUALMEM,120)
    assign redist9_expRMux_uid31_fpSqrtTest_q_10_mem_ia = expRMux_uid31_fpSqrtTest_q;
    assign redist9_expRMux_uid31_fpSqrtTest_q_10_mem_aa = redist9_expRMux_uid31_fpSqrtTest_q_10_wraddr_q;
    assign redist9_expRMux_uid31_fpSqrtTest_q_10_mem_ab = redist9_expRMux_uid31_fpSqrtTest_q_10_rdmux_q;
    assign redist9_expRMux_uid31_fpSqrtTest_q_10_mem_reset0 = areset;
    altera_syncram #(
        .ram_block_type("MLAB"),
        .operation_mode("DUAL_PORT"),
        .width_a(8),
        .widthad_a(3),
        .numwords_a(7),
        .width_b(8),
        .widthad_b(3),
        .numwords_b(7),
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
    ) redist9_expRMux_uid31_fpSqrtTest_q_10_mem_dmem (
        .clocken1(redist9_expRMux_uid31_fpSqrtTest_q_10_enaAnd_q[0]),
        .clocken0(VCC_q[0]),
        .clock0(clk),
        .aclr1(redist9_expRMux_uid31_fpSqrtTest_q_10_mem_reset0),
        .clock1(clk),
        .address_a(redist9_expRMux_uid31_fpSqrtTest_q_10_mem_aa),
        .data_a(redist9_expRMux_uid31_fpSqrtTest_q_10_mem_ia),
        .wren_a(en[0]),
        .address_b(redist9_expRMux_uid31_fpSqrtTest_q_10_mem_ab),
        .q_b(redist9_expRMux_uid31_fpSqrtTest_q_10_mem_iq),
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
    assign redist9_expRMux_uid31_fpSqrtTest_q_10_mem_q = redist9_expRMux_uid31_fpSqrtTest_q_10_mem_iq[7:0];

    // redist9_expRMux_uid31_fpSqrtTest_q_10_outputreg(DELAY,119)
    dspba_delay_ver #( .width(8), .depth(1), .reset_kind("ASYNC") )
    redist9_expRMux_uid31_fpSqrtTest_q_10_outputreg ( .xin(redist9_expRMux_uid31_fpSqrtTest_q_10_mem_q), .xout(redist9_expRMux_uid31_fpSqrtTest_q_10_outputreg_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expR_uid40_fpSqrtTest(ADD,39)@10
    assign expR_uid40_fpSqrtTest_a = {1'b0, redist9_expRMux_uid31_fpSqrtTest_q_10_outputreg_q};
    assign expR_uid40_fpSqrtTest_b = {8'b00000000, redist4_expInc_uid38_fpSqrtTest_b_1_q};
    assign expR_uid40_fpSqrtTest_o = $unsigned(expR_uid40_fpSqrtTest_a) + $unsigned(expR_uid40_fpSqrtTest_b);
    assign expR_uid40_fpSqrtTest_q = expR_uid40_fpSqrtTest_o[8:0];

    // expRR_uid51_fpSqrtTest(BITSELECT,50)@10
    assign expRR_uid51_fpSqrtTest_in = expR_uid40_fpSqrtTest_q[7:0];
    assign expRR_uid51_fpSqrtTest_b = expRR_uid51_fpSqrtTest_in[7:0];

    // expXIsMax_uid14_fpSqrtTest(LOGICAL,13)@0 + 1
    assign expXIsMax_uid14_fpSqrtTest_qi = expX_uid6_fpSqrtTest_b == cstAllOWE_uid8_fpSqrtTest_q ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    expXIsMax_uid14_fpSqrtTest_delay ( .xin(expXIsMax_uid14_fpSqrtTest_qi), .xout(expXIsMax_uid14_fpSqrtTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // invExpXIsMax_uid19_fpSqrtTest(LOGICAL,18)@1
    assign invExpXIsMax_uid19_fpSqrtTest_q = ~ (expXIsMax_uid14_fpSqrtTest_q);

    // InvExpXIsZero_uid20_fpSqrtTest(LOGICAL,19)@1
    assign InvExpXIsZero_uid20_fpSqrtTest_q = ~ (excZ_x_uid13_fpSqrtTest_q);

    // excR_x_uid21_fpSqrtTest(LOGICAL,20)@1
    assign excR_x_uid21_fpSqrtTest_q = InvExpXIsZero_uid20_fpSqrtTest_q & invExpXIsMax_uid19_fpSqrtTest_q;

    // minReg_uid43_fpSqrtTest(LOGICAL,42)@1
    assign minReg_uid43_fpSqrtTest_q = excR_x_uid21_fpSqrtTest_q & redist10_signX_uid7_fpSqrtTest_b_1_q;

    // cstZeroWF_uid9_fpSqrtTest(CONSTANT,8)
    assign cstZeroWF_uid9_fpSqrtTest_q = 23'b00000000000000000000000;

    // fracXIsZero_uid15_fpSqrtTest(LOGICAL,14)@0 + 1
    assign fracXIsZero_uid15_fpSqrtTest_qi = cstZeroWF_uid9_fpSqrtTest_q == frac_x_uid12_fpSqrtTest_b ? 1'b1 : 1'b0;
    dspba_delay_ver #( .width(1), .depth(1), .reset_kind("ASYNC") )
    fracXIsZero_uid15_fpSqrtTest_delay ( .xin(fracXIsZero_uid15_fpSqrtTest_qi), .xout(fracXIsZero_uid15_fpSqrtTest_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // excI_x_uid17_fpSqrtTest(LOGICAL,16)@1
    assign excI_x_uid17_fpSqrtTest_q = expXIsMax_uid14_fpSqrtTest_q & fracXIsZero_uid15_fpSqrtTest_q;

    // minInf_uid44_fpSqrtTest(LOGICAL,43)@1
    assign minInf_uid44_fpSqrtTest_q = excI_x_uid17_fpSqrtTest_q & redist10_signX_uid7_fpSqrtTest_b_1_q;

    // fracXIsNotZero_uid16_fpSqrtTest(LOGICAL,15)@1
    assign fracXIsNotZero_uid16_fpSqrtTest_q = ~ (fracXIsZero_uid15_fpSqrtTest_q);

    // excN_x_uid18_fpSqrtTest(LOGICAL,17)@1
    assign excN_x_uid18_fpSqrtTest_q = expXIsMax_uid14_fpSqrtTest_q & fracXIsNotZero_uid16_fpSqrtTest_q;

    // excRNaN_uid45_fpSqrtTest(LOGICAL,44)@1
    assign excRNaN_uid45_fpSqrtTest_q = excN_x_uid18_fpSqrtTest_q | minInf_uid44_fpSqrtTest_q | minReg_uid43_fpSqrtTest_q;

    // invSignX_uid41_fpSqrtTest(LOGICAL,40)@1
    assign invSignX_uid41_fpSqrtTest_q = ~ (redist10_signX_uid7_fpSqrtTest_b_1_q);

    // inInfAndNotNeg_uid42_fpSqrtTest(LOGICAL,41)@1
    assign inInfAndNotNeg_uid42_fpSqrtTest_q = excI_x_uid17_fpSqrtTest_q & invSignX_uid41_fpSqrtTest_q;

    // excConc_uid46_fpSqrtTest(BITJOIN,45)@1
    assign excConc_uid46_fpSqrtTest_q = {excRNaN_uid45_fpSqrtTest_q, inInfAndNotNeg_uid42_fpSqrtTest_q, excZ_x_uid13_fpSqrtTest_q};

    // fracSelIn_uid47_fpSqrtTest(BITJOIN,46)@1
    assign fracSelIn_uid47_fpSqrtTest_q = {redist10_signX_uid7_fpSqrtTest_b_1_q, excConc_uid46_fpSqrtTest_q};

    // fracSel_uid48_fpSqrtTest(LOOKUP,47)@1 + 1
    always @ (posedge clk or posedge areset)
    begin
        if (areset)
        begin
            fracSel_uid48_fpSqrtTest_q <= 2'b01;
        end
        else if (en == 1'b1)
        begin
            unique case (fracSelIn_uid47_fpSqrtTest_q)
                4'b0000 : fracSel_uid48_fpSqrtTest_q <= 2'b01;
                4'b0001 : fracSel_uid48_fpSqrtTest_q <= 2'b00;
                4'b0010 : fracSel_uid48_fpSqrtTest_q <= 2'b10;
                4'b0011 : fracSel_uid48_fpSqrtTest_q <= 2'b00;
                4'b0100 : fracSel_uid48_fpSqrtTest_q <= 2'b11;
                4'b0101 : fracSel_uid48_fpSqrtTest_q <= 2'b00;
                4'b0110 : fracSel_uid48_fpSqrtTest_q <= 2'b10;
                4'b0111 : fracSel_uid48_fpSqrtTest_q <= 2'b00;
                4'b1000 : fracSel_uid48_fpSqrtTest_q <= 2'b11;
                4'b1001 : fracSel_uid48_fpSqrtTest_q <= 2'b00;
                4'b1010 : fracSel_uid48_fpSqrtTest_q <= 2'b11;
                4'b1011 : fracSel_uid48_fpSqrtTest_q <= 2'b11;
                4'b1100 : fracSel_uid48_fpSqrtTest_q <= 2'b11;
                4'b1101 : fracSel_uid48_fpSqrtTest_q <= 2'b11;
                4'b1110 : fracSel_uid48_fpSqrtTest_q <= 2'b11;
                4'b1111 : fracSel_uid48_fpSqrtTest_q <= 2'b11;
                default : begin
                              // unreachable
                              fracSel_uid48_fpSqrtTest_q <= 2'bxx;
                          end
            endcase
        end
    end

    // redist2_fracSel_uid48_fpSqrtTest_q_9(DELAY,99)
    dspba_delay_ver #( .width(2), .depth(8), .reset_kind("ASYNC") )
    redist2_fracSel_uid48_fpSqrtTest_q_9 ( .xin(fracSel_uid48_fpSqrtTest_q), .xout(redist2_fracSel_uid48_fpSqrtTest_q_9_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // expRPostExc_uid53_fpSqrtTest(MUX,52)@10
    assign expRPostExc_uid53_fpSqrtTest_s = redist2_fracSel_uid48_fpSqrtTest_q_9_q;
    always @(expRPostExc_uid53_fpSqrtTest_s or en or cstAllZWE_uid10_fpSqrtTest_q or expRR_uid51_fpSqrtTest_b or cstAllOWE_uid8_fpSqrtTest_q)
    begin
        unique case (expRPostExc_uid53_fpSqrtTest_s)
            2'b00 : expRPostExc_uid53_fpSqrtTest_q = cstAllZWE_uid10_fpSqrtTest_q;
            2'b01 : expRPostExc_uid53_fpSqrtTest_q = expRR_uid51_fpSqrtTest_b;
            2'b10 : expRPostExc_uid53_fpSqrtTest_q = cstAllOWE_uid8_fpSqrtTest_q;
            2'b11 : expRPostExc_uid53_fpSqrtTest_q = cstAllOWE_uid8_fpSqrtTest_q;
            default : expRPostExc_uid53_fpSqrtTest_q = 8'b0;
        endcase
    end

    // fracNaN_uid54_fpSqrtTest(CONSTANT,53)
    assign fracNaN_uid54_fpSqrtTest_q = 23'b00000000000000000000001;

    // fracRPostProcessings_uid39_fpSqrtTest(BITSELECT,38)@9
    assign fracRPostProcessings_uid39_fpSqrtTest_in = s2_uid85_invPolyEval_q[28:0];
    assign fracRPostProcessings_uid39_fpSqrtTest_b = fracRPostProcessings_uid39_fpSqrtTest_in[28:6];

    // redist3_fracRPostProcessings_uid39_fpSqrtTest_b_1(DELAY,100)
    dspba_delay_ver #( .width(23), .depth(1), .reset_kind("ASYNC") )
    redist3_fracRPostProcessings_uid39_fpSqrtTest_b_1 ( .xin(fracRPostProcessings_uid39_fpSqrtTest_b), .xout(redist3_fracRPostProcessings_uid39_fpSqrtTest_b_1_q), .ena(en[0]), .clk(clk), .aclr(areset) );

    // fracRPostExc_uid58_fpSqrtTest(MUX,57)@10
    assign fracRPostExc_uid58_fpSqrtTest_s = redist2_fracSel_uid48_fpSqrtTest_q_9_q;
    always @(fracRPostExc_uid58_fpSqrtTest_s or en or cstZeroWF_uid9_fpSqrtTest_q or redist3_fracRPostProcessings_uid39_fpSqrtTest_b_1_q or fracNaN_uid54_fpSqrtTest_q)
    begin
        unique case (fracRPostExc_uid58_fpSqrtTest_s)
            2'b00 : fracRPostExc_uid58_fpSqrtTest_q = cstZeroWF_uid9_fpSqrtTest_q;
            2'b01 : fracRPostExc_uid58_fpSqrtTest_q = redist3_fracRPostProcessings_uid39_fpSqrtTest_b_1_q;
            2'b10 : fracRPostExc_uid58_fpSqrtTest_q = cstZeroWF_uid9_fpSqrtTest_q;
            2'b11 : fracRPostExc_uid58_fpSqrtTest_q = fracNaN_uid54_fpSqrtTest_q;
            default : fracRPostExc_uid58_fpSqrtTest_q = 23'b0;
        endcase
    end

    // RSqrt_uid60_fpSqrtTest(BITJOIN,59)@10
    assign RSqrt_uid60_fpSqrtTest_q = {redist1_negZero_uid59_fpSqrtTest_q_9_q, expRPostExc_uid53_fpSqrtTest_q, fracRPostExc_uid58_fpSqrtTest_q};

    // xOut(GPOUT,4)@10
    assign q = RSqrt_uid60_fpSqrtTest_q;

endmodule
