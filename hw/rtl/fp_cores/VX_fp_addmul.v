`include "VX_define.vh"

`ifndef SYNTHESIS
`include "float_dpi.vh"
`endif

module VX_fp_addmul #( 
    parameter TAGW = 1,
    parameter LANES = 1
) (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,

    input wire  do_sub,
    input wire  do_mul,    

    input wire [LANES-1:0][31:0]  dataa,
    input wire [LANES-1:0][31:0]  datab,
    output wire [LANES-1:0][31:0] result, 

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);    
    
    wire stall = ~ready_out && valid_out;
    wire enable = ~stall;

    reg do_sub_r, do_mul_r;

    for (genvar i = 0; i < LANES; i++) begin
        
        wire [31:0] result_add;
        wire [31:0] result_sub;
        wire [31:0] result_mul;

    `ifdef QUARTUS
        acl_fadd fadd (
            .clk    (clk),
            .areset (reset),
            .en     (enable),
            .a      (dataa[i]),
            .b      (datab[i]),
            .q      (result_add)
        );

        acl_fsub fsub (
            .clk    (clk),
            .areset (reset),
            .en     (enable),
            .a      (dataa[i]),
            .b      (datab[i]),
            .q      (result_sub)
        );

        acl_fmul fmul (
            .clk    (clk),
            .areset (reset),
            .en     (enable),
            .a      (dataa[i]),
            .b      (datab[i]),
            .q      (result_mul)
        );
    `else
        integer fadd_h, fsub_h, fmul_h;
        initial begin
            fadd_h = dpi_register();
            fsub_h = dpi_register();
            fmul_h = dpi_register();
        end
        always @(posedge clk) begin
           dpi_fadd (fadd_h, enable, dataa[i], datab[i], result_add);
           dpi_fsub (fsub_h, enable, dataa[i], datab[i], result_sub);
           dpi_fmul (fmul_h, enable, dataa[i], datab[i], result_mul);
        end
    `endif

        assign result[i] = do_mul_r ? result_mul : (do_sub_r ? result_sub : result_add);
    end
    
    VX_shift_register #(
        .DATAW  (1 + TAGW + 1 + 1),
        .DEPTH  (`LATENCY_FADDMUL),
        .RESETW (1)
    ) shift_reg (
        .clk(clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({valid_in,  tag_in,  do_sub,   do_mul}),
        .data_out ({valid_out, tag_out, do_sub_r, do_mul_r})
    );

    assign ready_in = enable;

endmodule
