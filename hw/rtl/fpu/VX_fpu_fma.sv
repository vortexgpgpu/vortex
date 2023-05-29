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
    output wire [NUM_LANES-1:0][`FP_FLAGS_BITS-1:0] fflags,

    output wire [TAGW-1:0] tag_out,

    input wire  ready_out,
    output wire valid_out
);
    `UNUSED_VAR (frm)

    wire stall = ~ready_out && valid_out;
    wire enable = ~stall;

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

    reg [NUM_LANES-1:0][31:0] a, b, c;

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        always @(*) begin
            if (do_madd) begin
                // MADD / MSUB / NMADD / NMSUB
                a[i] = do_neg ? {~dataa[i][31], dataa[i][30:0]} : dataa[i];                    
                b[i] = datab[i];
                c[i] = (do_neg ^ do_sub) ? {~datac[i][31], datac[i][30:0]} : datac[i];
            end else begin
                if (do_neg) begin
                    // MUL
                    a[i] = dataa[i];
                    b[i] = datab[i];
                    c[i] = '0;
                end else begin
                    // ADD / SUB
                    a[i] = 32'h3f800000; // 1.0f
                    b[i] = dataa[i];
                    c[i] = do_sub ? {~datab[i][31], datab[i][30:0]} : datab[i];
                end
            end    
        end
    end

`ifdef QUARTUS
    
    for (genvar i = 0; i < NUM_LANES; ++i) begin
        acl_fmadd fmadd (
            .clk    (clk),
            .areset (1'b0),
            .en     (enable),
            .a      (a[i]),
            .b      (b[i]),
            .c      (c[i]),
            .q      (result[i])
        );
    end

`elsif VIVADO

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        wire [2:0] tuser;
        
        xil_fma fma (
            .aclk                (clk),
            .aclken              (enable),
            .s_axis_a_tvalid     (1'b1),
            .s_axis_a_tdata      (a[i]),
            .s_axis_b_tvalid     (1'b1),
            .s_axis_b_tdata      (b[i]),
            .s_axis_c_tvalid     (1'b1),
            .s_axis_c_tdata      (c[i]),
            `UNUSED_PIN (m_axis_result_tvalid),
            .m_axis_result_tdata (result[i]),
            .m_axis_result_tuser (tuser)
        );
                        // NV,     DZ,   OF,       UF,       NX
        assign fflags[i] = {tuser[2], 1'b0, tuser[1], tuser[0], 1'b0};
    end

`else

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        reg [`XLEN-1:0] r;
        fflags_t f;
        `UNUSED_VAR (f)

        always @(*) begin        
            dpi_fmadd (enable && valid_in, a[i], b[i], c[i], frm, r, f);
        end        

        VX_shift_register #(
            .DATAW  (`XLEN),
            .DEPTH  (`LATENCY_FMA)
        ) shift_req_dpi (
            .clk      (clk),
            `UNUSED_PIN (reset),
            .enable   (enable),
            .data_in  (r),
            .data_out (result[i])
        );
    end

    assign has_fflags = 1'b0;
    assign fflags = '0;

`endif

endmodule
