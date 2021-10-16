`include "VX_fpu_define.vh"

module VX_fp_div #( 
    parameter TAGW = 1,
    parameter LANES = 1
) (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,

    input wire [`INST_FRM_BITS-1:0] frm,
    
    input wire [LANES-1:0][31:0]  dataa,
    input wire [LANES-1:0][31:0]  datab,
    output wire [LANES-1:0][31:0] result,  

    output wire has_fflags,
    output fflags_t [LANES-1:0] fflags,

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);    
    wire stall = ~ready_out && valid_out;
    wire enable = ~stall;

    for (genvar i = 0; i < LANES; i++) begin        
    `ifdef VERILATOR
        reg [31:0] r;
        fflags_t f;

        always @(*) begin        
            dpi_fdiv (enable && valid_in, dataa[i], datab[i], frm, r, f);
        end
        `UNUSED_VAR (f)

        VX_shift_register #(
            .DATAW  (32),
            .DEPTH  (`LATENCY_FDIV),
            .RESETW (1)
        ) shift_req_dpi (
            .clk      (clk),
            .reset    (reset),
            .enable   (enable),
            .data_in  (r),
            .data_out (result[i])
        );
    `else
        `RESET_RELAY (fdiv_reset);

        acl_fdiv fdiv (
            .clk    (clk),
            .areset (fdiv_reset),
            .en     (enable),
            .a      (dataa[i]),
            .b      (datab[i]),
            .q      (result[i])
        );
    `endif
    end

    VX_shift_register #(
        .DATAW  (1 + TAGW),
        .DEPTH  (`LATENCY_FDIV),
        .RESETW (1)
    ) shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({valid_in,  tag_in}),
        .data_out ({valid_out, tag_out})
    );

    assign ready_in = enable;

    `UNUSED_VAR (frm)
    assign has_fflags = 0;
    assign fflags = 0;

endmodule
