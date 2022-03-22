`include "VX_raster_define.vh"

module VX_raster_req_arb #(
    parameter NUM_REQS = 1
) (
    input wire clk,
    input wire reset,

    // input requests    
    VX_raster_req_if.slave     req_in_if[NUM_REQS],

    // output request
    VX_raster_req_if.master    req_out_if
);
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    // TODO
    for (genvar i = 0; i < NUM_REQS; ++i) begin
        wire valid = req_in_if[i].valid;
        `UNUSED_VAR (valid)
        assign req_in_if[i].stamp = '0;
        assign req_in_if[i].empty = '0;
        assign req_in_if[i].ready = 0;
    end

    // TODO
    assign req_out_if.valid = 0;
    `UNUSED_VAR (req_out_if.stamp)
    `UNUSED_VAR (req_out_if.empty)
    `UNUSED_VAR (req_out_if.ready)

endmodule
