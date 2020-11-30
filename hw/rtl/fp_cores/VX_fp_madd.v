`include "VX_define.vh"

`ifndef SYNTHESIS
`include "float_dpi.vh"
`endif

module VX_fp_madd #( 
    parameter TAGW = 1,
    parameter LANES = 1
) (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,

    input wire  do_sub,  
    input wire  do_neg, 
    
    input wire [LANES-1:0][31:0]  dataa,
    input wire [LANES-1:0][31:0]  datab,
    input wire [LANES-1:0][31:0]  datac,
    output wire [LANES-1:0][31:0] result, 

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);    
    
    wire stall = ~ready_out && valid_out;
    wire enable = ~stall;

    reg do_sub_r, do_neg_r;

    for (genvar i = 0; i < LANES; i++) begin
        
        wire [31:0] result_madd;
        wire [31:0] result_msub;

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
            .clk({2'b00, clk}),
            .ena({2'b00, enable}),
            .aclr({reset, reset}),
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
            .clk({2'b00, clk}),
            .ena({2'b00, enable}),
            .aclr({reset, reset}),
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
    `else
        integer fmadd_h, fmsub_h;
        initial begin
            fmadd_h = dpi_register();
            fmsub_h = dpi_register();
        end
        always @(posedge clk) begin
           dpi_fmadd(fmadd_h, enable, dataa[i], datab[i], datac[i], result_madd);
           dpi_fmsub(fmsub_h, enable, dataa[i], datab[i], datac[i], result_msub);
        end
    `endif

        wire [31:0] result_unqual = do_sub_r ? result_msub : result_madd;
        assign result[i][31]   = result_unqual[31] ^ do_neg_r;
        assign result[i][30:0] = result_unqual[30:0];
    end
    
    VX_shift_register #(
        .DATAW(1 + TAGW + 1 + 1),
        .DEPTH(`LATENCY_FMADD)
    ) shift_reg (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .in({valid_in,   tag_in,  do_sub,   do_neg}),
        .out({valid_out, tag_out, do_sub_r, do_neg_r})
    );

    assign ready_in = enable;

endmodule
