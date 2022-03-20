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
    assign rop_req_if.pos_x = 0;
    assign rop_req_if.pos_y = 0;
    assign rop_req_if.color = 0;
    assign rop_req_if.depth = 0;
    assign rop_req_if.backface = 0;
    `UNUSED_VAR (rop_req_if.ready);

    `UNUSED_VAR (rop_svc_req_if.valid)
    `UNUSED_VAR (rop_svc_req_if.uuid)
    `UNUSED_VAR (rop_svc_req_if.wid)
    `UNUSED_VAR (rop_svc_req_if.tmask)
    `UNUSED_VAR (rop_svc_req_if.PC)
    `UNUSED_VAR (rop_svc_req_if.pos_x)
    `UNUSED_VAR (rop_svc_req_if.pos_y)
    `UNUSED_VAR (rop_svc_req_if.backface)
    `UNUSED_VAR (rop_svc_req_if.color)
    `UNUSED_VAR (rop_svc_req_if.depth)
    assign rop_svc_req_if.ready = 0;

    assign rop_svc_rsp_if.valid = 0;

endmodule