`include "VX_rop_define.vh"

module VX_rop_srv #(
    parameter CORE_ID = 0
) (
    input wire clk,    
    input wire reset,

    // Inputs    
    VX_rop_srv_if.slave     rop_srv_if,
    VX_gpu_csr_if.slave     rop_csr_if,

    // Outputs
    VX_commit_if.master     rop_rsp_if
);
    `UNUSED_VAR (rop_srv_if.valid)
    `UNUSED_VAR (rop_srv_if.uuid)
    `UNUSED_VAR (rop_srv_if.wid)
    `UNUSED_VAR (rop_srv_if.tmask)
    `UNUSED_VAR (rop_srv_if.PC)
    `UNUSED_VAR (rop_srv_if.pos_x)
    `UNUSED_VAR (rop_srv_if.pos_y)
    `UNUSED_VAR (rop_srv_if.color)
    `UNUSED_VAR (rop_srv_if.depth)
    assign rop_srv_if.ready = 0;

    // TODO
    `UNUSED_VAR (clk)
    `UNUSED_VAR (reset)

endmodule