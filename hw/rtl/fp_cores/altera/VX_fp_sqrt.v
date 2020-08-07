`include "VX_define.vh"

module VX_fp_sqrt (
    input wire clk,
    input wire reset,   

    output wire in_ready,
    input wire  in_valid,

    input wire [`ISTAG_BITS-1:0] in_tag,

    input wire [`NUM_THREADS-1:0][31:0]  dataa,
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
        acl_fp_sqrt fsqrt (
            .clk    (clk),
            .areset (1'b0),
            .en     (enable),
            .a      (dataa[i]),
            .q      (result[i])
        );
    end

    VX_shift_register #(
        .DATAW(`ISTAG_BITS + 1),
        .DEPTH(`LATENCY_FSQRT)
    ) shift_reg (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .in({in_tag, in_valid}),
        .out({out_tag, out_valid})
    );

endmodule
