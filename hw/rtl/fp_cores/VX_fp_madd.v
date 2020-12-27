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
        acl_fmadd fmadd (
            .clk    (clk),
            .areset (reset),
            .en     (enable),
            .a      (dataa[i]),
            .b      (datab[i]),
            .c      (datac[i]),
            .q      (result_madd)
        );

         acl_fmsub fmsub (
            .clk    (clk),
            .areset (reset),
            .en     (enable),
            .a      (dataa[i]),
            .b      (datab[i]),
            .c      (datac[i]),
            .q      (result_msub)
        );
    `else
        integer fmadd_h, fmsub_h;
        initial begin
            fmadd_h = dpi_register();
            fmsub_h = dpi_register();
        end
        always @(posedge clk) begin
           dpi_fmadd (fmadd_h, enable, dataa[i], datab[i], datac[i], result_madd);
           dpi_fmsub (fmsub_h, enable, dataa[i], datab[i], datac[i], result_msub);
        end
    `endif

        wire [31:0] result_unqual = do_sub_r ? result_msub : result_madd;
        assign result[i][31]   = result_unqual[31] ^ do_neg_r;
        assign result[i][30:0] = result_unqual[30:0];
    end
    
    VX_shift_register #(
        .DATAW  (1 + TAGW + 1 + 1),
        .DEPTH  (`LATENCY_FMADD),
        .RESETW (1)
    ) shift_reg (
        .clk(clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({valid_in,  tag_in,  do_sub,   do_neg}),
        .data_out ({valid_out, tag_out, do_sub_r, do_neg_r})
    );

    assign ready_in = enable;

endmodule
