`include "VX_raster_define.vh"

module VX_raster_req_demux #(
    parameter NUM_REQS       = 1,
    parameter NUM_LANES      = 1,
    parameter BUFFERED       = 0,
    parameter string ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,

    // input request   
    VX_raster_req_if.slave  req_in_if,

    // output requests
    VX_raster_req_if.master req_out_if [NUM_REQS]
);
    localparam REQ_DATAW = NUM_LANES * (1 + $bits(raster_stamp_t)) + 1;

    wire [NUM_REQS-1:0]                req_valid_out;
    wire [NUM_REQS-1:0][REQ_DATAW-1:0] req_data_out;
    wire [NUM_REQS-1:0]                req_ready_out;

    VX_stream_demux #(
        .NUM_OUTPUTS (NUM_REQS),
        .DATAW       (REQ_DATAW),
        .BUFFERED    (BUFFERED),
        .ARBITER     (ARBITER)
    ) req_demux (
        .clk        (clk),
        .reset      (reset),
        `UNUSED_PIN (sel_in),
        .valid_in   (req_in_if.valid),
        .data_in    ({req_in_if.tmask, req_in_if.stamps, req_in_if.empty}),
        .ready_in   (req_in_if.ready),
        .valid_out  (req_valid_out),
        .data_out   (req_data_out),
        .ready_out  (req_ready_out)
    );
    
    for (genvar i = 0; i < NUM_REQS; i++) begin
        assign req_out_if[i].valid = req_valid_out[i];
        assign {req_out_if[i].tmask, req_out_if[i].stamps, req_out_if[i].empty} = req_data_out[i];
        assign req_ready_out[i] = req_out_if[i].ready;
    end

endmodule
