`include "VX_define.vh"

module VX_fp_sqrt (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [`ISTAG_BITS-1:0] tag_in,

    input wire [`NUM_THREADS-1:0][31:0]  dataa,
    output wire [`NUM_THREADS-1:0][31:0] result, 

    output wire [`ISTAG_BITS-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);    
    wire stall  = ~ready_out && valid_out;
    wire enable = ~stall;
    assign ready_in = enable;

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
        .in ({tag_in,  valid_in}),
        .out({tag_out, valid_out})
    );

endmodule
