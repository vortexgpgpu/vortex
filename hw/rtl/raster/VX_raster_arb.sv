`include "VX_raster_define.vh"

module VX_raster_arb #(
    parameter NUM_INPUTS     = 1,
    parameter NUM_OUTPUTS    = 1,
    parameter NUM_LANES      = 1,
    parameter BUFFERED       = 0,
    parameter `STRING ARBITER = "R"
) (
    input wire              clk,
    input wire              reset,

    // input request   
    VX_raster_bus_if.slave  bus_in_if [NUM_INPUTS],

    // output requests
    VX_raster_bus_if.master bus_out_if [NUM_OUTPUTS]
);
    localparam REQ_DATAW = NUM_LANES * $bits(raster_stamp_t) + 1;

    wire [NUM_INPUTS-1:0]                 req_valid_in;
    wire [NUM_INPUTS-1:0][REQ_DATAW-1:0]  req_data_in;
    wire [NUM_INPUTS-1:0]                 req_ready_in;

    wire [NUM_OUTPUTS-1:0]                req_valid_out;
    wire [NUM_OUTPUTS-1:0][REQ_DATAW-1:0] req_data_out;
    wire [NUM_OUTPUTS-1:0]                req_ready_out;

    wire [NUM_INPUTS-1:0] done_mask;
    for (genvar i = 0; i < NUM_INPUTS; ++i) begin
        assign done_mask[i] = bus_in_if[i].req_done;
    end
    wire done_all = (& done_mask);

    for (genvar i = 0; i < NUM_INPUTS; ++i) begin
        assign req_valid_in[i] = bus_in_if[i].req_valid;
        assign req_data_in[i]  = {bus_in_if[i].req_stamps, done_all};
        assign bus_in_if[i].req_ready = req_ready_in[i];
    end

    VX_stream_arb #(
        .NUM_INPUTS (NUM_INPUTS),
        .NUM_OUTPUTS(NUM_OUTPUTS),
        .DATAW      (REQ_DATAW),
        .ARBITER    (ARBITER),
        .BUFFERED   (BUFFERED),
        .MAX_FANOUT (4)
    ) req_arb (
        .clk        (clk),
        .reset      (reset),
        .valid_in   (req_valid_in),        
        .ready_in   (req_ready_in),
        .data_in    (req_data_in),
        .data_out   (req_data_out),
        .valid_out  (req_valid_out),
        .ready_out  (req_ready_out)
    );
    
    for (genvar i = 0; i < NUM_OUTPUTS; ++i) begin
        assign bus_out_if[i].req_valid = req_valid_out[i];
        assign {bus_out_if[i].req_stamps, bus_out_if[i].req_done} = req_data_out[i];
        assign req_ready_out[i] = bus_out_if[i].req_ready;
    end

endmodule
