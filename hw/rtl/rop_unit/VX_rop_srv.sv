`include "VX_rop_define.vh"

module VX_rop_srv #(
    parameter CORE_ID = 0
) (
    input wire clk,    
    input wire reset,

    // Inputs    
    VX_rop_req_if.slave     rop_req_if,
    VX_gpu_csr_if.slave     rop_csr_if,

    // Outputs
    VX_commit_if.master     rop_rsp_if
);

    // TODO
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

endmodule