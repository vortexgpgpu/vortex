`include "VX_raster_define.vh"

module VX_raster_req_mux #(
    parameter NUM_REQS       = 1,
    parameter NUM_LANES      = 1,
    parameter BUFFERED       = 0,
    parameter string ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,

    // input requests    
    VX_raster_req_if.slave  req_in_if [NUM_REQS],

    // output request
    VX_raster_req_if.master req_out_if
);

    localparam REQ_DATAW = NUM_LANES * (1 + $bits(raster_stamp_t)) + 1;

    wire [NUM_REQS-1:0]                req_valid_in;
    wire [NUM_REQS-1:0][REQ_DATAW-1:0] req_data_in;
    wire [NUM_REQS-1:0]                req_ready_in;

    for (genvar i = 0; i < NUM_REQS; i++) begin
        assign req_valid_in[i] = req_in_if[i].valid;
        assign req_data_in[i] = {req_in_if[i].tmask, req_in_if[i].stamps, req_in_if[i].empty};
        assign req_in_if[i].ready = req_ready_in[i];
    end        

    VX_stream_mux #(            
        .NUM_INPUTS (NUM_REQS),
        .DATAW      (REQ_DATAW),
        .BUFFERED   (BUFFERED),
        .ARBITER    (ARBITER)
    ) req_mux (
        .clk       (clk),
        .reset     (reset),
        `UNUSED_PIN (sel_in),
        .valid_in  (req_valid_in),
        .data_in   (req_data_in),
        .ready_in  (req_ready_in),
        .valid_out (req_out_if.valid),
        .data_out  ({req_out_if.tmask, req_out_if.stamps, req_out_if.empty}),
        .ready_out (req_out_if.ready)
    );

endmodule
