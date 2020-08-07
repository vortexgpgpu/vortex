`include "VX_define.vh"

module VX_fp_mul (
    input wire clk,
    input wire reset,   

    output wire in_ready,
    input wire  in_valid,

    input wire [`ISTAG_BITS-1:0] in_tag,

    input wire [`NUM_THREADS-1:0][31:0]  dataa,
    input wire [`NUM_THREADS-1:0][31:0]  datab,
    output wire [`NUM_THREADS-1:0][31:0] result, 

    output wire [`ISTAG_BITS-1:0] out_tag,

    input wire  out_ready,
    output wire out_valid
);    
    wire stall = ~out_ready && out_valid;
    wire enable = ~stall;
    assign in_ready = enable;

    genvar i;

    for (i = 0; i < `NUM_THREADS; i++) begin
        twentynm_fp_mac mac_fp_wys (
            // inputs
            .accumulate(),
            .chainin_overflow(),
            .chainin_invalid(),
            .chainin_underflow(),
            .chainin_inexact(),
            .ax(),
            .ay(datab[i]),
            .az(dataa[i]),
            .clk({2'b00,clk}),
            .ena({2'b11,enable}),
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
        defparam mac_fp_wys.operation_mode = "sp_mult"; 
        defparam mac_fp_wys.use_chainin = "false"; 
        defparam mac_fp_wys.adder_subtract = "false"; 
        defparam mac_fp_wys.ax_clock = "none"; 
        defparam mac_fp_wys.ay_clock = "0"; 
        defparam mac_fp_wys.az_clock = "0"; 
        defparam mac_fp_wys.output_clock = "0"; 
        defparam mac_fp_wys.accumulate_clock = "none"; 
        defparam mac_fp_wys.ax_chainin_pl_clock = "none"; 
        defparam mac_fp_wys.accum_pipeline_clock = "none"; 
        defparam mac_fp_wys.mult_pipeline_clock = "0"; 
        defparam mac_fp_wys.adder_input_clock = "none"; 
        defparam mac_fp_wys.accum_adder_clock = "none"; 
    end

    VX_shift_register #(
        .DATAW(`ISTAG_BITS + 1),
        .DEPTH(1)
    ) shift_reg (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .in({in_tag, in_valid}),
        .out({out_tag, out_valid})
    );

endmodule