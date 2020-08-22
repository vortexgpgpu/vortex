`include "VX_define.vh"

module VX_fp_div #( 
    parameter TAGW = 1,
    parameter LANES = 1
) (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,

    input wire [LANES-1:0][31:0]  dataa,
    input wire [LANES-1:0][31:0]  datab,
    output wire [LANES-1:0][31:0] result, 

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);    
    wire stall  = ~ready_out && valid_out;
    wire enable = ~stall;
    assign ready_in = enable;

    for (genvar i = 0; i < LANES; i++) begin
        acl_fp_div fdiv (
            .clk    (clk),
            .areset (1'b0),
            .en     (enable),
            .a      (dataa[i]),
            .b      (datab[i]),
            .q      (result[i])
        );
    end

    VX_shift_register #(
        .DATAW(TAGW + 1),
        .DEPTH(`LATENCY_FDIV)
    ) shift_reg (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .in ({tag_in,  valid_in}),
        .out({tag_out, valid_out})
    );

endmodule
