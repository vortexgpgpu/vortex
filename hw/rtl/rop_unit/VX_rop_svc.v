`include "VX_rop_define.vh"

module VX_rop_svc #(
    parameter CORE_ID = 0
) (
    input wire clk,    
    input wire reset,

    // Inputs    
    VX_rop_svc_if.slave     rop_svc_req_if,    
    VX_gpu_csr_if.slave     rop_csr_if,  

    // Outputs    
    VX_commit_if.master     rop_svc_rsp_if,
    VX_rop_req_if.master    rop_req_if
);
    // CSRs access

    rop_csrs_t rop_csrs;

    VX_rop_csr #(
        .CORE_ID    (CORE_ID)
    ) rop_csr (
        .clk        (clk),
        .reset      (reset),

        // inputs
        .rop_csr_if (rop_csr_if),

        // outputs
        .rop_csrs   (rop_csrs)
    );

    `UNUSED_VAR (rop_csrs)

    assign rop_req_if.valid = 0;
    assign rop_req_if.tmask = 0;
    
    `UNUSED_VAR (rop_req_if.ready);

    assign rop_req_if.valid = rop_svc_req_if.valid & rop_svc_rsp_if.ready;
    assign rop_req_if.tmask = rop_svc_req_if.tmask;
    assign rop_req_if.pos_x = rop_svc_req_if.pos_x;
    assign rop_req_if.pos_y = rop_svc_req_if.pos_y;
    assign rop_req_if.color = rop_svc_req_if.color;
    assign rop_req_if.depth = rop_svc_req_if.depth;
    assign rop_req_if.backface = rop_svc_req_if.backface;
    assign rop_svc_req_if.ready = rop_req_if.ready & rop_svc_rsp_if.ready;

    assign rop_svc_rsp_if.valid = rop_svc_req_if.valid & rop_req_if.ready;
    assign rop_svc_rsp_if.uuid  = rop_svc_req_if.uuid;
    assign rop_svc_rsp_if.wid   = rop_svc_req_if.wid;
    assign rop_svc_rsp_if.tmask = rop_svc_req_if.tmask;
    assign rop_svc_rsp_if.PC    = rop_svc_req_if.PC;
    assign rop_svc_rsp_if.rd    = '0;
    assign rop_svc_rsp_if.wb    = 0;

endmodule