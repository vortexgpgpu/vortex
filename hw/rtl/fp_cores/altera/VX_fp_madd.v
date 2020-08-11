`include "VX_define.vh"

module VX_fp_madd (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [`ISTAG_BITS-1:0] tag_in,

    input wire [`NUM_THREADS-1:0][31:0]  dataa,
    input wire [`NUM_THREADS-1:0][31:0]  datab,
    input wire [`NUM_THREADS-1:0][31:0]  datac,
    output wire [`NUM_THREADS-1:0][31:0] result, 

    input wire  negate,

    output wire [`ISTAG_BITS-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);    
    wire enable0, enable1;
    assign ready_in = enable0 && enable1;

    wire [`NUM_THREADS-1:0][31:0] result_st0, result_st1;
    wire [`ISTAG_BITS-1:0] out_tag_st0, out_tag_st1;
    wire in_valid_st0, out_valid_st0, out_valid_st1;

    genvar i;

    for (i = 0; i < `NUM_THREADS; i++) begin
        twentynm_fp_mac mac_fp_wys0 (
            // inputs
            .accumulate(),
            .chainin_overflow(),
            .chainin_invalid(),
            .chainin_underflow(),
            .chainin_inexact(),
            .ax(datac[i]),
            .ay(datab[i]),
            .az(dataa[i]),
            .clk({2'b00,clk}),
            .ena({2'b11,enable0}),
            .aclr(2'b00),
            .chainin(),
            // outputs
            .overflow(),
            .invalid(),
            .underflow(),
            .inexact(),
            .chainout_overflow(),
            .chainout_invalid(),
            .chainout_underflow(),
            .chainout_inexact(),
            .resulta(result_st0[i]),
            .chainout()
        );
        defparam mac_fp_wys0.operation_mode = "sp_mult_add"; 
        defparam mac_fp_wys0.use_chainin = "false"; 
        defparam mac_fp_wys0.adder_subtract = "false"; 
        defparam mac_fp_wys0.ax_clock = "0"; 
        defparam mac_fp_wys0.ay_clock = "0"; 
        defparam mac_fp_wys0.az_clock = "0"; 
        defparam mac_fp_wys0.output_clock = "0"; 
        defparam mac_fp_wys0.accumulate_clock = "none"; 
        defparam mac_fp_wys0.ax_chainin_pl_clock = "0"; 
        defparam mac_fp_wys0.accum_pipeline_clock = "none"; 
        defparam mac_fp_wys0.mult_pipeline_clock = "0"; 
        defparam mac_fp_wys0.adder_input_clock = "0"; 
        defparam mac_fp_wys0.accum_adder_clock = "none"; 

        twentynm_fp_mac mac_fp_wys1 (
            // inputs
            .accumulate(),
            .chainin_overflow(),
            .chainin_invalid(),
            .chainin_underflow(),
            .chainin_inexact(),
            .ax(32'h0),
            .ay(result_st0[i]),
            .az(),
            .clk({2'b00,clk}),
            .ena({2'b11,enable1}),
            .aclr(2'b00),
            .chainin(),
            // outputs
            .overflow(),
            .invalid(),
            .underflow(),
            .inexact(),
            .chainout_overflow(),
            .chainout_invalid(),
            .chainout_underflow(),
            .chainout_inexact(),
            .resulta(result_st1[i]),
            .chainout()
        );
        defparam mac_fp_wys1.operation_mode = "sp_add"; 
        defparam mac_fp_wys1.use_chainin = "false"; 
        defparam mac_fp_wys1.adder_subtract = "true"; 
        defparam mac_fp_wys1.ax_clock = "0"; 
        defparam mac_fp_wys1.ay_clock = "0"; 
        defparam mac_fp_wys1.az_clock = "none"; 
        defparam mac_fp_wys1.output_clock = "0"; 
        defparam mac_fp_wys1.accumulate_clock = "none"; 
        defparam mac_fp_wys1.ax_chainin_pl_clock = "none"; 
        defparam mac_fp_wys1.accum_pipeline_clock = "none"; 
        defparam mac_fp_wys1.mult_pipeline_clock = "none"; 
        defparam mac_fp_wys1.adder_input_clock = "0"; 
        defparam mac_fp_wys1.accum_adder_clock = "none";
    end

    VX_shift_register #(
        .DATAW(`ISTAG_BITS + 1 + 1),
        .DEPTH(1)
    ) shift_reg0 (
        .clk(clk),
        .reset(reset),
        .enable(enable0),
        .in ({tag_in,      (valid_in && ~negate), (valid_in && negate)}),
        .out({out_tag_st0, out_valid_st0,         in_valid_st0})
    );

    VX_shift_register #(
        .DATAW(`ISTAG_BITS + 1),
        .DEPTH(1)
    ) shift_reg1 (
        .clk(clk),
        .reset(reset),
        .enable(enable1),
        .in({in_tag_st0, in_valid_st0}),
        .out({out_tag_st1, out_valid_st1})
    );

    wire out_stall = ~ready_out && valid_out;
    assign enable0 = ~out_stall;
    assign enable1 = ~out_stall && ~(out_valid_st0 && out_valid_st1); // stall the negate stage if dual outputs

    assign result    = out_valid_st0 ? result_st0  : result_st1;
    assign tag_out   = out_valid_st0 ? out_tag_st0 : out_tag_st1;
    assign valid_out = out_valid_st0 || out_valid_st1; 

endmodule
