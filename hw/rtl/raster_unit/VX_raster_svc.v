`include "VX_raster_define.vh"

module VX_raster_svc #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    // Inputs    
    VX_raster_svc_if.slave raster_svc_req_if,    
    VX_raster_req_if.slave raster_req_if,
        
    // Outputs
    VX_commit_if.master     raster_svc_rsp_if,
    VX_gpu_csr_if.slave     raster_csr_if    
);
    // CSRs access

    VX_raster_req_if #() csr_write_if ();

    assign csr_write_if.valid  = raster_svc_req_if.valid & raster_req_if.valid & raster_svc_rsp_if.ready;
    assign csr_write_if.tmask  = raster_req_if.tmask;
    assign csr_write_if.stamps = raster_req_if.stamps;
    assign csr_write_if.empty  = raster_req_if.empty;

    assign raster_req_if.ready = raster_svc_req_if.valid & csr_write_if.ready & raster_svc_rsp_if.ready;

    VX_raster_csr #(
        .CORE_ID    (CORE_ID)
    ) raster_csr (
        .clk        (clk),
        .reset      (reset),

        // inputs        
        .raster_svc_req_if (raster_svc_req_if),
        .csr_write_if (csr_write_if),
        // outputs
        .raster_csr_if (raster_csr_if)
    );

    assign raster_svc_req_if.ready = raster_req_if.valid & csr_write_if.ready & raster_svc_rsp_if.ready;

    assign raster_svc_rsp_if.valid = raster_svc_req_if.valid & raster_req_if.valid & csr_write_if.ready;
    assign raster_svc_rsp_if.uuid  = raster_svc_req_if.uuid;
    assign raster_svc_rsp_if.wid   = raster_svc_req_if.wid;
    assign raster_svc_rsp_if.tmask = raster_svc_req_if.tmask;
    assign raster_svc_rsp_if.PC    = raster_svc_req_if.PC;

    for (genvar i = 0; i < `NUM_THREADS; ++i) begin
        assign raster_svc_rsp_if.data[i] = {31'(csr_write_if.stamps[i].pid), csr_write_if.empty};
    end

    assign raster_svc_rsp_if.rd    = raster_svc_req_if.rd;
    assign raster_svc_rsp_if.wb    = raster_svc_req_if.wb;

endmodule