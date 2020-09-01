`include "VX_define.vh"

`ifndef SYNTHESIS
`include "float_dpi.vh"
`endif

module VX_fp_nmadd #( 
    parameter TAGW = 1,
    parameter LANES = 1
) (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,

    input wire  do_sub,

    input wire [LANES-1:0][31:0]  dataa,
    input wire [LANES-1:0][31:0]  datab,
    input wire [LANES-1:0][31:0]  datac,
    output wire [LANES-1:0][31:0] result, 

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
); 

    wire stall = ~ready_out && valid_out;

    reg do_sub_r;

    for (genvar i = 0; i < LANES; i++) begin

        wire [31:0] result_madd;
        wire [31:0] result_msub; 

        wire [31:0] result_st0 = do_sub_r ? result_msub : result_madd;

    `ifdef QUARTUS
        twentynm_fp_mac mac_fp_madd (
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
            .ena({2'b11,~stall}),
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
            .resulta(result_madd),
            .chainout()
        );
        defparam mac_fp_madd.operation_mode = "sp_mult_add"; 
        defparam mac_fp_madd.use_chainin = "false"; 
        defparam mac_fp_madd.adder_subtract = "false"; 
        defparam mac_fp_madd.ax_clock = "0"; 
        defparam mac_fp_madd.ay_clock = "0"; 
        defparam mac_fp_madd.az_clock = "0"; 
        defparam mac_fp_madd.output_clock = "0"; 
        defparam mac_fp_madd.accumulate_clock = "none"; 
        defparam mac_fp_madd.ax_chainin_pl_clock = "0"; 
        defparam mac_fp_madd.accum_pipeline_clock = "none"; 
        defparam mac_fp_madd.mult_pipeline_clock = "0"; 
        defparam mac_fp_madd.adder_input_clock = "0"; 
        defparam mac_fp_madd.accum_adder_clock = "none"; 

        twentynm_fp_mac mac_fp_msub (
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
            .resulta(result_msub),
            .chainout()
        );
        defparam mac_fp_msub.operation_mode = "sp_mult_add"; 
        defparam mac_fp_msub.use_chainin = "false"; 
        defparam mac_fp_msub.adder_subtract = "true"; 
        defparam mac_fp_msub.ax_clock = "0"; 
        defparam mac_fp_msub.ay_clock = "0"; 
        defparam mac_fp_msub.az_clock = "0"; 
        defparam mac_fp_msub.output_clock = "0"; 
        defparam mac_fp_msub.accumulate_clock = "none"; 
        defparam mac_fp_msub.ax_chainin_pl_clock = "0"; 
        defparam mac_fp_msub.accum_pipeline_clock = "none"; 
        defparam mac_fp_msub.mult_pipeline_clock = "0"; 
        defparam mac_fp_msub.adder_input_clock = "0"; 
        defparam mac_fp_msub.accum_adder_clock = "none";

        twentynm_fp_mac mac_fp_neg (
            // inputs
            .accumulate(),
            .chainin_overflow(),
            .chainin_invalid(),
            .chainin_underflow(),
            .chainin_inexact(),
            .ax(32'h0),
            .ay(result_st0),
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
            .resulta(result[i]),
            .chainout()
        );
        defparam mac_fp_neg.operation_mode = "sp_add"; 
        defparam mac_fp_neg.use_chainin = "false"; 
        defparam mac_fp_neg.adder_subtract = "true"; 
        defparam mac_fp_neg.ax_clock = "0"; 
        defparam mac_fp_neg.ay_clock = "0"; 
        defparam mac_fp_neg.az_clock = "none"; 
        defparam mac_fp_neg.output_clock = "0"; 
        defparam mac_fp_neg.accumulate_clock = "none"; 
        defparam mac_fp_neg.ax_chainin_pl_clock = "none"; 
        defparam mac_fp_neg.accum_pipeline_clock = "none"; 
        defparam mac_fp_neg.mult_pipeline_clock = "none"; 
        defparam mac_fp_neg.adder_input_clock = "0"; 
        defparam mac_fp_neg.accum_adder_clock = "none";
    `else
        reg valid_in_st0;
        always @(posedge clk) begin
           valid_in_st0 <= reset ? 0 : valid_in; 
           dpi_fmadd(5*LANES+i, ~stall, valid_in, dataa[i], datab[i], datac[i], result_madd);
           dpi_fmsub(6*LANES+i, ~stall, valid_in, dataa[i], datab[i], datac[i], result_msub);
           dpi_fsub(7*LANES+i, ~stall, valid_in_st0, 32'b0, result_st0, result[i]);
        end
    `endif
    end    

    always @(posedge clk) begin
        if (~stall) begin
            do_sub_r <= do_sub;
        end
    end

    VX_shift_register #(
        .DATAW(TAGW + 1),
        .DEPTH(`LATENCY_FNMADD)
    ) shift_reg1 (
        .clk(clk),
        .reset(reset),
        .enable(~stall),
        .in({tag_in,   valid_in}),
        .out({tag_out, valid_out})
    );

    assign ready_in  = ~stall;

endmodule