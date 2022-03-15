`include "VX_raster_define.vh"

// Top unit for the raster unit
// Instantiates the following modules:
//      1. DCR connections
//      2. Requests switch
//      3. Raster slices
//      4. Response switch

module VX_raster_unit #(  
    parameter CLUSTER_ID  = 0,
    parameter NUM_SLICES  = 1,  // number of raster slices
    parameter NUM_OUTPUTS = 4   // number of output queues    
    // TODO
) (
    input wire clk,
    input wire reset,

    // PERF
`ifdef PERF_ENABLE
    VX_raster_perf_if.master raster_perf_if,
`endif

    // Memory interface
    VX_dcache_req_if.master cache_req_if,
    VX_dcache_rsp_if.slave  cache_rsp_if,

    // Inputs
    VX_raster_dcr_if.master raster_dcr_if,
    VX_raster_req_if.slave  raster_req_if,

    // Outputs
    VX_raster_rsp_if.master raster_rsp_if
);

// TODO: remove
`IGNORE_WARNINGS_BEGIN

    raster_dcrs_t raster_dcrs;

    // Raster unit dcr block
    VX_raster_dcr #(
    ) raster_dcr (
        .clk        (clk),
        .reset      (reset),

        // inputs
        .dcr_wr_valid (0),
        .raster_dcr_if(raster_dcr_if)

        // output
    );

    // TODO: Add requests switch here

    VX_raster_req_switch #(
    ) raster_req_switch (
        .clk    (clk),
        .reset  (reset)
    );

    // TODO: Add raster slices in generate block here
    /*for (genvar i = 0; i < NUM_SLICES; ++i) begin
        VX_raster_slice #(
        ) raster_slice (
            .clk    (clk),
            .reset  (reset)
        );
    end*/

    // TODO: Add response switch here
    VX_raster_rsp_switch #(
    ) raster_rsp_switch (
        .clk    (clk),
        .reset  (reset)
    );

    // TODO: remove
    assign raster_dcrs = raster_dcr_if.data;
    `UNUSED_VAR (raster_dcrs)

    // TODO: remove
    `UNUSED_VAR (raster_req_if.valid)
    `UNUSED_VAR (raster_req_if.uuid)
    `UNUSED_VAR (raster_req_if.cid)
    `UNUSED_VAR (raster_req_if.wid)
    `UNUSED_VAR (raster_req_if.tmask)
    `UNUSED_VAR (raster_req_if.PC)
    `UNUSED_VAR (raster_req_if.rd)
    `UNUSED_VAR (raster_req_if.wb)
    `UNUSED_VAR (raster_req_if.tmask)
    assign raster_req_if.ready = 0;

    // TODO: remove
    assign raster_rsp_if.valid = 0;
    assign raster_rsp_if.uuid  = 0;
    assign raster_rsp_if.cid   = 0;
    assign raster_rsp_if.wid   = 0;
    assign raster_rsp_if.tmask = 0;
    assign raster_rsp_if.PC    = 0;
    assign raster_rsp_if.rd    = 0;
    assign raster_rsp_if.wb    = 0;
    assign raster_rsp_if.stamp = '0;
    assign raster_rsp_if.last  = 0;
    `UNUSED_VAR (raster_rsp_if.ready)

    // TODO: remove
    assign raster_perf_if.mem_reads = 0;
    assign raster_perf_if.mem_latency = 0;

    // TODO: remove
    assign cache_req_if.valid = 0;
    assign cache_req_if.rw = 0;
    assign cache_req_if.byteen = 0;
    assign cache_req_if.addr = 0;
    assign cache_req_if.data = 0;     
    assign cache_req_if.tag = 0;
    `UNUSED_VAR (cache_req_if.ready)

    // TODO: remove
    `UNUSED_VAR (cache_rsp_if.valid)
    `UNUSED_VAR (cache_rsp_if.tmask)
    `UNUSED_VAR (cache_rsp_if.data)        
    `UNUSED_VAR (cache_rsp_if.tag)
    assign cache_rsp_if.ready = 0;

// TODO: remove
`IGNORE_WARNINGS_END

endmodule