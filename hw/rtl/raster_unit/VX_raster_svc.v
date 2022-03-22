`include "VX_raster_define.vh"

module VX_raster_svc #(
    parameter CORE_ID = 0
) (
    input wire clk,
    input wire reset,

    // Inputs    
    VX_raster_svc_if.slave  raster_svc_req_if,    
    VX_raster_req_if.master raster_req_if,
        
    // Outputs
    VX_commit_if.master     raster_svc_rsp_if,
    VX_gpu_csr_if.slave     raster_csr_if    
);
    // CSRs access

    VX_raster_csr #(
        .CORE_ID    (CORE_ID)
    ) raster_csr (
        .clk        (clk),
        .reset      (reset),

        // inputs        
        .raster_svc_req_if (raster_svc_req_if),
        .raster_req_if (raster_req_if),

        // outputs
        .raster_csr_if (raster_csr_if)
    );

    assign raster_req_if.valid = raster_svc_req_if.valid & raster_svc_rsp_if.ready;
    assign raster_req_if.tmask = raster_svc_req_if.tmask;
    assign raster_svc_req_if.ready = raster_req_if.ready & raster_svc_rsp_if.ready;

    assign raster_svc_rsp_if.valid = raster_svc_req_if.valid & raster_req_if.ready;
    assign raster_svc_rsp_if.uuid  = raster_svc_req_if.uuid;
    assign raster_svc_rsp_if.wid   = raster_svc_req_if.wid;
    assign raster_svc_rsp_if.tmask = raster_svc_req_if.tmask;
    assign raster_svc_rsp_if.PC    = raster_svc_req_if.PC;
    assign raster_svc_rsp_if.rd    =  raster_svc_req_if.rd;
    assign raster_svc_rsp_if.wb    =  raster_svc_req_if.wb;

endmodule