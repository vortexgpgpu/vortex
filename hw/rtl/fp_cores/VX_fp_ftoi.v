`include "VX_define.vh"

`ifndef SYNTHESIS
`include "float_dpi.vh"
`endif

module VX_fp_ftoi #( 
    parameter TAGW = 1,
    parameter LANES = 1
) (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,

    input wire  is_signed,

    input wire [LANES-1:0][31:0]  dataa,
    output wire [LANES-1:0][31:0] result, 

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);    
    wire stall = ~ready_out && valid_out;
    wire enable = ~stall;

    reg is_signed_r;
    
    for (genvar i = 0; i < LANES; i++) begin

        wire [31:0] result_s;
        wire [31:0] result_u;

    `ifdef QUARTUS       
        acl_ftoi ftoi (
            .clk    (clk),
            .areset (reset),
            .en     (enable),
            .a      (dataa[i]),
            .q      (result_s)
        );

        acl_ftou ftou (
            .clk    (clk),
            .areset (reset),
            .en     (enable),
            .a      (dataa[i]),
            .q      (result_u)
        );        
    `else
        integer ftoi_h, ftou_h;
        initial begin
            ftoi_h = dpi_register();
            ftou_h = dpi_register();
        end
        always @(posedge clk) begin
           dpi_ftoi(ftoi_h, enable, dataa[i], result_s);
           dpi_ftou(ftou_h, enable, dataa[i], result_u);
        end
    `endif

        assign result[i] = is_signed_r ? result_s : result_u;
    end

    VX_shift_register #(
        .DATAW(1 + TAGW + 1),
        .DEPTH(`LATENCY_FTOI)
    ) shift_reg (
        .clk(clk),
        .reset(reset),
        .enable(enable),
        .in ({valid_in,  tag_in,  is_signed}),
        .out({valid_out, tag_out, is_signed_r})
    );

    assign ready_in = enable;

endmodule
