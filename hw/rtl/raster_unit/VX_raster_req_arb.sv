`include "VX_raster_define.vh"

module VX_raster_req_arb #(
    parameter NUM_REQS = 1
) (
    input wire clk,
    input wire reset,

    // input request   
    VX_raster_req_if.slave    req_in_if,

    // output requests
    VX_raster_req_if.master   req_out_if[NUM_REQS]
);
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

    // TODO
    `UNUSED_VAR (req_in_if.valid)
    `UNUSED_VAR (req_in_if.tmask)
    `UNUSED_VAR (req_in_if.stamps)
    `UNUSED_VAR (req_in_if.empty)
    assign req_in_if.ready = 0;

    // TODO
    for (genvar i = 0; i < NUM_REQS; ++i) begin        
        assign req_out_if[i].valid  = 0;
        assign req_out_if[i].tmask  = '0;
        assign req_out_if[i].stamps = '0;
        assign req_out_if[i].empty  = 0;
        wire valid = req_out_if[i].ready;
        `UNUSED_VAR (ready)
    end

endmodule
