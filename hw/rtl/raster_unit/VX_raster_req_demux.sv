`include "VX_raster_define.vh"

module VX_raster_req_demux #(
    parameter NUM_INPUTS     = 1,
    parameter NUM_OUTPUTS    = 1,
    parameter NUM_LANES      = 1,
    parameter BUFFERED       = 0,
    parameter string ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,

    // input request   
    VX_raster_req_if.slave  req_in_if [NUM_INPUTS],

    // output requests
    VX_raster_req_if.master req_out_if [NUM_OUTPUTS]
);
    localparam REQ_DATAW = NUM_LANES * (1 + $bits(raster_stamp_t)) + 1;

    wire [NUM_INPUTS-1:0]                 req_valid_in;
    wire [NUM_INPUTS-1:0][REQ_DATAW-1:0]  req_data_in;
    wire [NUM_INPUTS-1:0]                 req_ready_in;

    wire [NUM_OUTPUTS-1:0]                req_valid_out;
    wire [NUM_OUTPUTS-1:0][REQ_DATAW-1:0] req_data_out;
    wire [NUM_OUTPUTS-1:0]                req_ready_out;

    VX_stream_arb #(
        .NUM_INPUTS (NUM_INPUTS),
        .NUM_OUTPUTS(NUM_OUTPUTS),
        .DATAW      (REQ_DATAW),
        .BUFFERED   (BUFFERED),
        .ARBITER    (ARBITER)
    ) req_arb (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (req_valid_in),
        .data_in    (req_data_in),
        .ready_in   (req_ready_in),
        .valid_out  (req_valid_out),
        .data_out   (req_data_out),
        .ready_out  (req_ready_out)
    );

    for (genvar i = 0; i < NUM_INPUTS; ++i) begin
        assign req_valid_in[i] = req_in_if[i].valid;
        assign req_data_in[i] = {req_in_if[i].tmask, req_in_if[i].stamps, req_in_if[i].empty};
        assign req_in_if[i].ready = req_ready_in[i];
    end
    
    for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
        assign req_out_if[i].valid = req_valid_out[i];
        assign {req_out_if[i].tmask, req_out_if[i].stamps, req_out_if[i].empty} = req_data_out[i];
        assign req_ready_out[i] = req_out_if[i].ready;
    end

endmodule
