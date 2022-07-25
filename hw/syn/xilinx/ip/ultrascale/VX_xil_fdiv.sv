`include "VX_fpu_define.vh"

module VX_xil_fdiv #(
    parameter NUM_LANES = 1,
    parameter TAGW = 1,
    parameter LATENCY = 28 
) (
    input wire clk,
    input wire reset,   

    output wire ready_in,
    input wire  valid_in,

    input wire [TAGW-1:0] tag_in,

    input wire [`INST_FRM_BITS-1:0] frm,
    
    input wire [NUM_LANES-1:0][31:0]  dataa,
    input wire [NUM_LANES-1:0][31:0]  datab,
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
        wire [3:0] tuser;
        wire tvalid_in;

        `RESET_RELAY (fdiv_reset, reset);

        assign tvalid_in = enable && valid_in;

        xil_fdiv fdiv (
            .aclk                 (clk),
            .aresetn              (~fdiv_reset),
            .s_axis_a_tvalid      (tvalid_in),
            .s_axis_a_tdata       (a),
            .s_axis_b_tvalid      (tvalid_in),
            .s_axis_b_tdata       (b),
            .m_axis_result_tvalid (valid_out),
            .m_axis_result_tdata  (result[i]),
            .m_axis_result_tuser  (tuser[i])
        );

        assign fflags[i].NX = 1'b0;
        assign fflags[i].UF = tuser[0];
        assign fflags[i].OF = tuser[1];
        assign fflags[i].DZ = tuser[3];
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
