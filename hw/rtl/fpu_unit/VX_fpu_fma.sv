`include "VX_fpu_define.vh"

module VX_fpu_fma #(
    parameter NUM_LANES = 1, 
    parameter TAGW = 1
) (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,
    
    input wire [`INST_FRM_BITS-1:0] frm,

    input wire  do_madd,
    input wire  do_sub,
    input wire  do_neg,

    input wire [NUM_LANES-1:0][31:0]  dataa,
    input wire [NUM_LANES-1:0][31:0]  datab,
    input wire [NUM_LANES-1:0][31:0]  datac,
    output wire [NUM_LANES-1:0][31:0] result,  

    output wire has_fflags,
    output fflags_t [NUM_LANES-1:0] fflags,

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);

`ifdef QUARTUS
    
    VX_acl_fma #(
        .NUM_LANES  (NUM_LANES),
        .TAGW       (TAGW)
    ) fp_fma (
        .clk        (clk), 
        .reset      (reset),   
        .valid_in   (valid_in),
        .ready_in   (ready_in),
        .tag_in     (tag_in),
        .frm        (frm),
        .do_madd    (do_madd),
        .do_sub     (do_sub),
        .do_neg     (do_neg),
        .dataa      (dataa), 
        .datab      (datab),
        .datac      (datac),
        .has_fflags (has_fflags),
        .fflags     (fflags),   
        .result     (result),
        .tag_out    (tag_out),
        .valid_out  (valid_out),
        .ready_out  (ready_out)
    );

`elsif VIVADO

    VX_xil_fma #(
        .NUM_LANES  (NUM_LANES),
        .TAGW       (TAGW)
    ) fp_fma (
        .clk        (clk), 
        .reset      (reset),   
        .valid_in   (valid_in),
        .ready_in   (ready_in),
        .tag_in     (tag_in),
        .frm        (frm),
        .do_madd    (do_madd),
        .do_sub     (do_sub),
        .do_neg     (do_neg),
        .dataa      (dataa), 
        .datab      (datab),
        .datac      (datac),
        .has_fflags (has_fflags),
        .fflags     (fflags),   
        .result     (result),
        .tag_out    (tag_out),
        .valid_out  (valid_out),
        .ready_out  (ready_out)
    );

`else

    wire stall = ~ready_out && valid_out;
    wire enable = ~stall;

    for (genvar i = 0; i < NUM_LANES; ++i) begin       
        reg [31:0] a, b, c;
        reg [31:0] r;
        fflags_t f;

        always @(*) begin
            if (do_madd) begin
                // MADD/MSUB/NMADD/NMSUB
                a = do_neg ? {~dataa[i][31], dataa[i][30:0]} : dataa[i];                    
                b = datab[i];
                c = (do_neg ^ do_sub) ? {~datac[i][31], datac[i][30:0]} : datac[i];
            end else begin
                if (do_neg) begin
                    // MUL
                    a = dataa[i];
                    b = datab[i];
                    c = 0;
                end else begin
                    // ADD/SUB
                    a = 32'h3f800000; // 1.0f
                    b = dataa[i];
                    c = do_sub ? {~datab[i][31], datab[i][30:0]} : datab[i];
                end
            end    
        end

        always @(*) begin        
            dpi_fmadd (enable && valid_in, a, b, c, frm, r, f);
        end
        `UNUSED_VAR (f)

        VX_shift_register #(
            .DATAW  (32),
            .DEPTH  (`LATENCY_FMA)
        ) shift_req_dpi (
            .clk      (clk),
            `UNUSED_PIN (reset),
            .enable   (enable),
            .data_in  (r),
            .data_out (result[i])
        );
    end
    
    VX_shift_register #(
        .DATAW  (1 + TAGW),
        .DEPTH  (`LATENCY_FMA),
        .RESETW (1)
    ) shift_reg (
        .clk(clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  ({valid_in,  tag_in}),
        .data_out ({valid_out, tag_out})
    );

    assign ready_in = enable;

    `UNUSED_VAR (frm)
    assign has_fflags = 0;
    assign fflags = 0;

`endif

endmodule
