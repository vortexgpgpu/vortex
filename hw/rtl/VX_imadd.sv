`include "VX_define.vh"

module VX_imadd #(    
    parameter NUM_LANES  = 1,
    parameter DATA_WIDTH = 1,
    parameter MAX_SHIFT  = 0,
    parameter TAG_WIDTH  = 1,
    parameter SIGNED     = 0,
    parameter SHIFT_WIDTH = `LOG2UP(MAX_SHIFT+1)
) (
    input wire clk,
    input wire reset,
    
    // Inputs
    input wire                          valid_in,
    input wire [SHIFT_WIDTH-1:0]        shift_in,
    input wire [NUM_LANES-1:0][DATA_WIDTH-1:0] data_in1, 
    input wire [NUM_LANES-1:0][DATA_WIDTH-1:0] data_in2,
    input wire [NUM_LANES-1:0][DATA_WIDTH-1:0] data_in3,    
    input wire [TAG_WIDTH-1:0]          tag_in,
    output wire                         ready_in,

    // Outputs
    output wire                         valid_out,
    output wire [NUM_LANES-1:0][DATA_WIDTH-1:0] data_out,
    output wire [TAG_WIDTH-1:0]         tag_out,
    input wire                          ready_out
);

    localparam PROD_WIDTH = DATA_WIDTH + MAX_SHIFT;

    wire                                 valid_in_s;
    wire [SHIFT_WIDTH-1:0]               shift_in_s;
    wire [TAG_WIDTH-1:0]                 tag_in_s;
    wire [NUM_LANES-1:0][DATA_WIDTH-1:0] data_in3_s;    
    wire [NUM_LANES-1:0][PROD_WIDTH-1:0] mul_result;
    wire [NUM_LANES-1:0][DATA_WIDTH-1:0] result;

    wire stall_in, stall_out;    

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        VX_multiplier #(
            .WIDTHA  (DATA_WIDTH),
            .WIDTHB  (DATA_WIDTH),
            .WIDTHP  (PROD_WIDTH),
            .SIGNED  (SIGNED),
            .LATENCY (`LATENCY_IMUL)
        ) multiplier (
            .clk    (clk),
            .enable (~stall_in),
            .dataa  (data_in1[i]),
            .datab  (data_in2[i]),
            .result (mul_result[i])
        );
    end

    assign stall_in = valid_in_s && stall_out;
    
    VX_shift_register #(
        .DATAW  (1 + SHIFT_WIDTH + NUM_LANES * DATA_WIDTH + TAG_WIDTH),
        .DEPTH  (`LATENCY_IMUL),
        .RESETW (1)
    ) shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_in),
        .data_in  ({valid_in,   shift_in,   data_in3,   tag_in}),
        .data_out ({valid_in_s, shift_in_s, data_in3_s, tag_in_s})
    );

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        wire [DATA_WIDTH-1:0] mul_shift;
        if (SIGNED) begin
            assign mul_shift = DATA_WIDTH'($signed(mul_result[i]) >> $signed(shift_in_s));
        end else begin
            assign mul_shift = DATA_WIDTH'(mul_result[i] >> shift_in_s);
        end
        assign result[i] = mul_shift + data_in3_s[i];
    end

    ///////////////////////////////////////////////////////////////////////////

    assign stall_out = valid_out && ~ready_out;

    VX_pipe_register #(
        .DATAW  (1 + NUM_LANES * DATA_WIDTH + TAG_WIDTH),
        .DEPTH  (2),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (~stall_out),
        .data_in  ({valid_in_s, result,   tag_in_s}),
        .data_out ({valid_out,  data_out, tag_out})
    );

    // can accept new request?
    assign ready_in = ~stall_in;
    
endmodule
