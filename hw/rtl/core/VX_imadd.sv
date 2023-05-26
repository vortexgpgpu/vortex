`include "VX_define.vh"

module VX_imadd #(    
    parameter NUM_LANES  = 1,
    parameter DATA_WIDTH = 1,
    parameter MAX_SHIFT  = 0,
    parameter TAG_WIDTH  = 1,
    parameter SIGNED     = 0,
    parameter SHIFT_WIDTH = `LOG2UP(MAX_SHIFT+1)
) (
    input wire                          clk,
    input wire                          reset,
    
    // Inputs
    input wire                          valid_in,
    input wire [SHIFT_WIDTH-1:0]        shift_in,
    input wire [NUM_LANES-1:0][DATA_WIDTH-1:0] data1_in, 
    input wire [NUM_LANES-1:0][DATA_WIDTH-1:0] data2_in,
    input wire [NUM_LANES-1:0][DATA_WIDTH-1:0] data3_in,    
    input wire [TAG_WIDTH-1:0]          tag_in,
    output wire                         ready_in,

    // Outputs
    output wire                         valid_out,
    output wire [NUM_LANES-1:0][DATA_WIDTH-1:0] data_out,
    output wire [TAG_WIDTH-1:0]         tag_out,
    input wire                          ready_out
);

    localparam PROD_WIDTH = DATA_WIDTH + MAX_SHIFT;

    wire                                 valid_m, valid_s;
    wire                                 ready_m, ready_s;
    wire [SHIFT_WIDTH-1:0]               shift_m;
    wire [TAG_WIDTH-1:0]                 tag_m, tag_s;
    wire [NUM_LANES-1:0][DATA_WIDTH-1:0] data3_m;      
    wire [NUM_LANES-1:0][DATA_WIDTH-1:0] result_m, result_s;
    wire [NUM_LANES-1:0][PROD_WIDTH-1:0] mul_result;

    // multiplication    

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        VX_multiplier #(
            .A_WIDTH (DATA_WIDTH),
            .B_WIDTH (DATA_WIDTH),
            .R_WIDTH (PROD_WIDTH),
            .SIGNED  (SIGNED),
            .LATENCY (`LATENCY_IMUL)
        ) multiplier (
            .clk    (clk),
            .enable (ready_in),
            .dataa  (data1_in[i]),
            .datab  (data2_in[i]),
            .result (mul_result[i])
        );
    end

    // can accept new request?
    assign ready_in = ~valid_m || ready_m;
    
    VX_shift_register #(
        .DATAW  (1 + SHIFT_WIDTH + NUM_LANES * DATA_WIDTH + TAG_WIDTH),
        .DEPTH  (`LATENCY_IMUL),
        .RESETW (1)
    ) shift_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (ready_in),
        .data_in  ({valid_in, shift_in, data3_in, tag_in}),
        .data_out ({valid_m,  shift_m,  data3_m,  tag_m})
    );

    // shift and add

    for (genvar i = 0; i < NUM_LANES; ++i) begin
        wire [DATA_WIDTH-1:0] shift_result;
        if (SIGNED != 0) begin
            assign shift_result = DATA_WIDTH'($signed(mul_result[i]) >>> $signed(shift_m));
        end else begin
            assign shift_result = DATA_WIDTH'(mul_result[i] >> shift_m);
        end
        assign result_m[i] = shift_result + data3_m[i];
    end

    assign ready_m = ~valid_s || ready_s;

    VX_pipe_register #(
        .DATAW  (1 + NUM_LANES * DATA_WIDTH + TAG_WIDTH),
        .RESETW (1)
    ) pipe_reg (
        .clk      (clk),
        .reset    (reset),
        .enable   (ready_m),
        .data_in  ({valid_m, result_m, tag_m}),
        .data_out ({valid_s, result_s, tag_s})
    );

    // output result

    VX_skid_buffer #(
        .DATAW (NUM_LANES * DATA_WIDTH + TAG_WIDTH)
    ) rsp_sbuf (
        .clk       (clk),
        .reset     (reset),
        .valid_in  (valid_s),
        .ready_in  (ready_s),
        .data_in   ({result_s, tag_s}),
        .data_out  ({data_out, tag_out}),
        .valid_out (valid_out),
        .ready_out (ready_out)
    );    
    
endmodule
