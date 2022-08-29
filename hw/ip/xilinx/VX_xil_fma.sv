`include "VX_fpu_define.vh"

module VX_xil_fma #(
    parameter NUM_LANES = 1, 
    parameter TAGW = 1,
    parameter LATENCY = 16 
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
    wire stall = ~ready_out && valid_out;
    wire enable = ~stall;

    for (genvar i = 0; i < NUM_LANES; ++i) begin       
        reg [31:0] a, b, c;
        wire [2:0] tuser;

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

        xil_fma fma (
            .aclk                    (clk),
            .aclken                  (enable),
            .s_axis_a_tvalid         (1'b1),
            .s_axis_a_tdata          (a),
            .s_axis_b_tvalid         (1'b1),
            .s_axis_b_tdata          (b),
            .s_axis_c_tvalid         (1'b1),
            .s_axis_c_tdata          (c),
            `UNUSED_PIN (m_axis_result_tvalid),
            .m_axis_result_tdata     (result[i]),
            .m_axis_result_tuser     (tuser)
        );

        assign fflags[i].NX = 1'b0;
        assign fflags[i].UF = tuser[0];
        assign fflags[i].OF = tuser[1];
        assign fflags[i].DZ = 1'b0;
        assign fflags[i].NV = tuser[2];
    end
    
    VX_shift_register #(
        .DATAW  (TAGW),
        .DEPTH  (LATENCY),
        .RESETW (1)
    ) shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (enable),
        .data_in  (tag_in),
        .data_out (tag_out)
    );

    assign ready_in = enable;

    `UNUSED_VAR (frm)
    assign has_fflags = 1;    

endmodule
